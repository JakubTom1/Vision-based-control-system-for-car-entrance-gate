from ultralytics import YOLO
import cv2
import os
from scripts.Neural_network.util import get_car, read_license_plate
import time





def NN_main(DIRECTORY, frame, recognized, results, frame_num, coco_model, vehicles, license_plate_detector):
    """
    Execution of neural network algorithm for license plate recognition.
    :param DIRECTORY: str
        The directory path of the project, used for saving and loading resources.
    :param frame: numpy.ndarray
        The current video frame or image to process.
    :param recognized: BOOL
        Current state of recognition (input always false).
    :param results: dict
        A dictionary to append detection results, including bounding boxes, class IDs, and confidences.
    :param frame_num: int
        The index of the current frame being processed.
    :param coco_model: Model
        A pre-trained object detection model (YOLO) for vehicle detection.
    :param vehicles: list
        A list of vehicle classes that the object detection model recognizes (e.g., car, truck).
    :param license_plate_detector: Model
        A pre-trained model or algorithm for detecting and recognizing license plates.
    :return: BOOL
        Current state of recognition.
    :return: numpy.ndarray
        The current video frame with marked recognition bounding box.
    :return: dict
        A dictionary to append detection results, including bounding boxes, class IDs, and confidences.
    """
    list_for_printing =[]
    height, width, _ = frame.shape
    frame = frame[0:height, 0:int(0.8*width)]
    start_time = time.time()
    results[frame_num] = {}
    # Detect vehicles
    detections = coco_model(frame)[0]  # usage of coco_model database to recognize objects
    detections_list = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection  # reading coordinates, precision and class of detected object
        if int(class_id) in vehicles:
            detections_list.append([x1, y1, x2, y2, score])  # adding recognized vehicle to detected list
    if len(detections_list) == 0:
        return recognized, frame, results, list_for_printing
    # Track vehicles
    # track_ids = motion_tracker.update(np.asarray(detections_list))
    track_ids = detections_list
    x1, y1, x2, y2 = 0, 0, 0, 0
    x1_car, y1_car, x2_car, y2_car =0, 0, 0, 0
    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate  # reading cordinates, precision and class of license plate
        # Assign license plate to car
        x1_car, y1_car, x2_car, y2_car, car_id = get_car(license_plate, track_ids)
        if car_id != -1:
            # Crop and process license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]  # extraction of license plate from the image

            # Read license plate text
            license_plate_text, license_plate_text_score, straight_license_plate = read_license_plate(license_plate_crop, DIRECTORY)
            if license_plate_text is not None and license_plate_text_score > 0.2:
                # Draw bounding box and text for license plate
                print("Rozpoznano: ", license_plate_text, "\nWynik: ", license_plate_text_score)
                recognized = True
                break
    end_time = time.time()  # Zapisz czas zakończenia

    execution_time = end_time - start_time  # Oblicz czas wykonania
    print(f"Czas wykonania: {execution_time:.6f} sekund")
    if recognized:
        results[frame_num] = {'text': license_plate_text,
                              'text_score': license_plate_text_score,
                              'time': execution_time}
        list_for_printing = [[int(x1), int(y1), int(x2), int(y2)], [int(x1_car), int(y1_car), int(x2_car), int(y2_car)], license_plate_text]
    return recognized, frame, results, list_for_printing