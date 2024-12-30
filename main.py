from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
from scripts.usefull import signal_of_opening, write_csv
from scripts.Neural_network.NN_main import NN_main
from scripts.classical_mathods.classical_method import classical_main

DIRECTORY = '<<Project_directory>>'

# CHOOSE METHOD OF DETECTION
Neural_Network = True
Classical_method = True if not Neural_Network else False

settings ='<<video_file_name>>'
def connect_to_stream(signal):
    """
    Checking the connection and initialization the stream.
    :param signal: String localization of video stream.
    :return: Video stream.
    """
    # Video stream initialization
    cap = cv2.VideoCapture(signal)

    # Ensuring of connection
    if not cap.isOpened():
        print("Sorry. Can't recover the video stream.")
        return None
    return cap

def extract_movement_sector(frame):
    """
        Extracts a specific polygonal region of interest (ROI) from a given image frame.
        The function uses a predefined polygon to create a mask, isolates the desired
        area, and crops it to a bounding rectangle.

        Parameters:
            frame (numpy.ndarray): The input image from which the region of interest will be extracted.

        Returns:
            numpy.ndarray: The cropped image containing the region of interest.
        """
    # Get the dimensions of the input frame
    height, width, _ = frame.shape

    # Create an empty mask (black - ignores pixels)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the polygon (e.g., a triangle with angled sides)
    points = np.array([
        [0, 0.57 * height],
        [0.6 * width, 0.29 * height],
        [0.68 * width, 0.29 * height],
        [0.35 * width, height]
    ], dtype=np.float32).astype(np.int32)

    # Draw a white polygon on the mask
    cv2.fillPoly(mask, [points], 255)

    # Extract the area of interest (apply the mask)
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Crop the image to the bounding box around the polygon
    x, y, w, h = cv2.boundingRect(points)
    frame = frame[y:y + h, x:x + w]

    return frame

def detect_movement(gray_frame1, frame, Movement_detected= False):
    """
    Detection of movement function.
    :param gray_frame1: numpy.ndarray
        Grayscale image of cropped first frame from video.
    :param frame: numpy.ndarray
        Image of cropped frame from video.
    :param Movement_detected: BOOL
        State of movement detection.
    :return:
        Actualized movement detection.
    """
    gray_frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.GaussianBlur(gray_frame2, (21, 21), 0)

    # Subtract images
    frame_delta = cv2.absdiff(gray_frame1, gray_frame2)
    thresh = cv2.threshold(frame_delta, 85, 255, cv2.THRESH_BINARY)[1]

    # Dilatation to extract difference
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check of detected differences
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Threshold of minimal difference
            continue
        Movement_detected = True

    return Movement_detected

def main():
    # Load stream 1-rtsp from your ip camera, 2-your video
    method = ''
    frame_num = -1
    results = {}

    #Signal source: 1 - rtsp from camera, 2 - video
    stream = 'rtsp://<<login>>:<<password>>@<<camera_IP>>:554/Streaming/Channels/101'  # 1
    # stream = os.path.join(DIRECTORY, f'/videos/camera_{settings}.mp4')  # 2
    cap = connect_to_stream(stream)

    # writing video to the file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(DIRECTORY, f'/videos/camera.mp4'), fourcc, 20.0,
                          (width, height))
    if Neural_Network:
        out_nn = cv2.VideoWriter(os.path.join(DIRECTORY, f'/videos/YOLO{settings}.mp4'), fourcc, 20.0,
                                (width, height))
    if Classical_method:
        out_classical = cv2.VideoWriter(os.path.join(DIRECTORY, f'/videos/Classical{settings}.mp4'), fourcc,
                                        20.0, (width, height))
    # Scaling factor (for example, reduce to 50% of original size)
    scaling_factor = 0.5
    retry_count = 0
    processing_delay = 0  # time delay for processing (depending on fps)
    last_delay = 0
    if Neural_Network:
        # YOLO model for various objects recognition
        coco_model = YOLO('yolo11n.pt')

        # YOLO model for license plate localization
        license_plate_detector = YOLO(
            os.path.join(DIRECTORY, '/runs/detect_license_plate/train/weights/best.pt'))
        '''
        list of recognized object from dataset coco:
            0: person
            1: bicycle
            2: car
            3: motorcycle
            4: airplane
            5: bus
            6: train
            7: truck
        '''
        vehicles = [2, 3, 5, 7]

    ##############################################################################
    # Extract sector of movement
    ret, frame1 = cap.read()
    frame1 = extract_movement_sector(frame1)
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame1 = cv2.GaussianBlur(gray_frame1, (21, 21), 0)
    ###############################################################################
    try:
        while True:
            frame_num += 1
            if cap is None or not cap.isOpened():
                # Try to recover the signal
                print("Connection Lost. Recovering connection...")
                cap = connect_to_stream(stream)
                retry_count += 1
                if retry_count > 20:  # Setting retries counter
                    print("Too many tries of reconect.")
                    break
                time.sleep(0.5)  # Time before new recovering
                continue
            # reading signal actuality and frame
            ret, frame = cap.read()
            out.write(frame)
            if not ret:
                print("No signal.")
                cap.release()
                cap = None  # None for recovering the connection
                continue

            if processing_delay > 0:
                # Decrease the delay (time countdown)
                last_delay = processing_delay
                processing_delay -= 1
                print(f"Skipping processing for {processing_delay / 2} more seconds.")
                if Neural_Network:
                    out_nn.write(frame)
                if Classical_method:
                    out_classical.write(frame)
                continue  # Skip processing this frame
            if last_delay == 1:
                last_delay = 0
                signal_of_opening(False)

            recognized = False  # Flag of recognition state
            if frame is not None:

                ##############################################
                # Movement check section
                cropped_frame = extract_movement_sector(frame)
                Movement_detected = detect_movement(gray_frame1, cropped_frame)
                #####################################################
                if not Movement_detected:
                    if Neural_Network:
                        out_nn.write(frame)
                    if Classical_method:
                        out_classical.write(frame)
                    continue

                if Neural_Network:
                    method = 'Neural_Network'
                    recognized, processed_frame, results, list_for_printing = NN_main(DIRECTORY, frame, recognized, results,
                                                                                    frame_num, coco_model,
                                                                                    vehicles, license_plate_detector)
                    cv2.putText(processed_frame, method, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    if recognized:
                        cv2.rectangle(frame, (list_for_printing[1][0], list_for_printing[1][1]), (list_for_printing[1][2], list_for_printing[1][3]), (255, 0, 0), 2)
                        cv2.rectangle(frame, (list_for_printing[0][0], list_for_printing[0][1]), (list_for_printing[0][2], list_for_printing[0][3]), (0, 255, 0), 2)
                        cv2.putText(frame, list_for_printing[2], (list_for_printing[0][0], list_for_printing[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2)
                    out_nn.write(processed_frame)
                elif Classical_method:
                    method = 'Classical_method'
                    recognized, processed_frame, results, list_for_printing = classical_main(DIRECTORY, frame, cropped_frame,
                                                                                            recognized,results, frame_num, gray_frame1)
                    cv2.putText(processed_frame, method, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    if recognized:
                        cv2.rectangle(frame, list_for_printing[0], list_for_printing[1], (0, 255, 0), 2)
                        cv2.putText(frame, list_for_printing[3], list_for_printing[2], cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2)
                    out_classical.write(processed_frame)
                else:
                    break

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # Show the resized frame with detections
                resized_processed_frame = cv2.resize(processed_frame, None, fx=scaling_factor, fy=scaling_factor)
                cv2.imshow('License Plate Detection', resized_processed_frame)
                # cv2.moveWindow("License Plate Detection", 1000, 800)
                if cv2.waitKey(1) == ord('w'):
                    cv2.waitKey(0)

                # Sending signal
                try:
                    if recognized:
                        signal_of_opening(True)
                        # In 2 frames per second downloading
                        processing_delay = 240  # 60 seconds
                except Exception as e:
                    print(f"Error  occurred: {e}. Execute further iteration.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Saving operation...")
    finally:
        # Release video capture, written video and close windows
        print("check")
        cap.release()
        out.release()
        if Neural_Network:
            out_nn.release()
        if Classical_method:
            out_classical.release()
        cv2.destroyAllWindows()
        write_csv(results, f'./results/{method}{settings}.csv')

if __name__ == '__main__':
    main()