import string
from ultralytics import YOLO
import cv2
import numpy as np
import os
from scripts.usefull import my_license_plates

#
license_plate_code = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}

def license_complies_format(license_plates):
    """
    Check if the license plate text complies with the required format.

    Args:
        license_plates (list): List of formatted, possible license plates.

    Returns:
        tuple: (bool, str) True if the license plate complies with the format, text of confirmed license plate.
    """
    myPlates_list, myPlates_lists = my_license_plates()
    for text in license_plates:
        if text in myPlates_list:
            return True, text

    return False, None


def sort_points(pts):
    # Sortowanie według sumy i różnicy współrzędnych
    pts = sorted(pts, key=lambda x: (x[1], x[0]))  # Najpierw według y, potem x
    top_points = sorted(pts[:2], key=lambda x: x[0])
    bottom_points = sorted(pts[2:], key=lambda x: x[0])
    return np.array([top_points[0], top_points[1], bottom_points[0], bottom_points[1]], dtype="float32")
def straighten_license_plate(license_plate_crop, license_plate_reader):
    """
    Preprocessing of the license plate for correcting possible errors.
    :param license_plate_crop: (PIL.Image.Image) Cropped image containing the license plate.
    :param license_plate_reader: (ultralytics.models.yolo.model.YOLO) Model of the license plate numbers.
    :return: (PIL.Image.Image) Straighten image of letters.
    """
    numbers_params = []
    license_plate_numbers = license_plate_reader(license_plate_crop)[0]
    license_plate_crop_copy = license_plate_crop.copy()
    for license_plate_number in license_plate_numbers.boxes.data.tolist():  # decoding each license plate number
        x1, y1, x2, y2, score, class_id = license_plate_number
        if score > 0.7:     # threshold of each character correction
            number = license_plate_code[class_id]
            numbers_params.append([number, x1, y1, x2, y2, score])
            cv2.rectangle(license_plate_crop_copy, (int(x1), int(y1)), (int(x2),int(y2)), (0, 255, 0), 2)
    if numbers_params == []:    # in case of no license plate recognition
        return license_plate_crop
    numbers_params.sort(key=lambda item: item[1])    # sorting from the left to right
    #cv2.imshow('cropped_license_plate_segmented', license_plate_crop_copy)
    # Corners of my characters detection preprocessing
    correction = 1
    pts1 = np.float32([[numbers_params[0][1]-correction, numbers_params[0][2]-correction],
                       [numbers_params[-1][3]+correction, numbers_params[-1][2]-correction],
                       [numbers_params[0][1]-correction, numbers_params[0][4]+correction],
                       [numbers_params[-1][3]+correction, numbers_params[-1][4]+correction]])

    # Final rectangle sizes
    width = int(max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3])))
    height = int(max(np.linalg.norm(pts1[0] - pts1[2]), np.linalg.norm(pts1[1] - pts1[3])))


    pts2 = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])   # Final rectangle corners

    # Perspective transformation
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(license_plate_crop, M, (width, height))
    warped= cv2.resize(warped, None, fx=4, fy=4)

    cv2.imshow("warped", warped)
    return warped

def read_license_plate(license_plate_crop, DIRECTORY):
    """
    Read the license plate text from the given cropped image.

    :param license_plate_crop: (PIL.Image.Image) Cropped image containing the license plate.
    :param DIRECTORY: (str) Directory of the project
    :return: Tuple containing the formatted license plate text,its confidence score and straighten image of license plate..
    """
    numbers_params = []

    license_plate_reader = YOLO(os.path.join(DIRECTORY, 'Neural_Network_NPR/runs/detect/train2/weights/best.pt'))
    straight_license_plate = straighten_license_plate(license_plate_crop, license_plate_reader)
    license_plate_numbers = license_plate_reader(straight_license_plate)[0]
    straight_license_plate_copy = straight_license_plate.copy()
    for license_plate_number in license_plate_numbers.boxes.data.tolist():  # decoding each license plate number
        x1, y1, x2, y2, score, class_id = license_plate_number
        if score > 0.5:     # threshold of each character correction
            number = license_plate_code[class_id]
            numbers_params.append([number, x1, y1, x2, y2, score])
            cv2.rectangle(straight_license_plate_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    numbers_params.sort(key=lambda item: item[1])   # sorting from the left to right
    text = ''.join([param[0] for param in numbers_params])
    #cv2.imshow("straight_license_plate_copy",straight_license_plate_copy)
    flag, confirmed_license = license_complies_format([text])     # check if conditions are satisfied
    if flag:
        mean_score = sum([param[5] for param in numbers_params]) / len(numbers_params)
        return confirmed_license, mean_score, straight_license_plate

    return None, None, None

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    :param license_plate: (tuple) Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
    :param vehicle_track_ids: (list) List of vehicle track IDs and their corresponding coordinates.
    :return: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate
    found_car_idx = -1
    for j in range(len(vehicle_track_ids)):   # iterating all detected vehicles in a frame
        x1_car, y1_car, x2_car, y2_car, car_id = vehicle_track_ids[j]   # reading coordinates of particular vehicle
        if x1 > x1_car and y1 > y1_car and x2 < x2_car and y2 < y2_car:     # checking if license plate rectangle is located in vehicles rectangle
            found_car_idx = j
            break
    if found_car_idx >= 0:
        return vehicle_track_ids[found_car_idx]
    else:
        return -1, -1, -1, -1, -1
