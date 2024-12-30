import cv2
import numpy as np
import os
import time
from scripts.classical_mathods.license_plate_localization import license_plate_localization

from scripts.classical_mathods.license_plate_preprocessing import preprocess
from scripts.classical_mathods.characters_segmentation import characters_segmentation
from scripts.classical_mathods.characters_recognition import characters_recognition

def classical_main(DIRECTORY, frame, cropped_frame, recognized, results, frame_num, gray_frame1):
    """
    Execution of classical technics algorithm for license plate recognition.
    :param DIRECTORY: str
        The directory path of the project, used for saving and loading resources.
    :param frame: numpy.ndarray
        The current video frame or image to process.
    :param cropped_frame: numpy.ndarray

    :param recognized: BOOL
        Current state of recognition (input always false).
    :param results: dict
        A dictionary to append detection results, including bounding boxes, class IDs, and confidences.
    :param frame_num: int
        The index of the current frame being processed.
    :return: BOOL
        Current state of recognition.
    :return: numpy.ndarray
        The current video frame with recognized license plate numbers.
    :return: dict
        A dictionary to append detection results, including bounding boxes, class IDs, and confidences.
    """
    results[frame_num] = {}
    start_time = time.time()
    cropped_license_plates = license_plate_localization(cropped_frame)  # all possible images of cropped license plates
    for idx, (license_plate, (x, y, w, h)) in enumerate(cropped_license_plates):
        license_plate = cv2.resize(license_plate, None, fx=4, fy=4)
        preprocessed_license_plate = preprocess(license_plate)     # returning preprocessed, cropped license plate
        if preprocessed_license_plate is None:
            continue
        characters_images_list = characters_segmentation(preprocessed_license_plate) # List of tuples containing image, Horizontal position and Vertical position
        if characters_images_list == [] or characters_images_list is None:
            continue
        recognized, my_license_plate, my_license_plate_score = characters_recognition(DIRECTORY, characters_images_list)
        if recognized:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 255, 0)
            thickness = 2
            height, width, _ = frame.shape
            corner_1 = (x, int(0.29 * height) + y)
            corner_2 = (x + w, int(0.29 * height) + y + h)
            license_number_print = (x, int(0.29 * height)+y-10)
            print("Rozpoznano: ", my_license_plate, "\nWynik: ", my_license_plate_score)
            break
    end_time = time.time()  # Zapisz czas zakończenia

    execution_time = end_time - start_time  # Oblicz czas wykonania
    print(f"Czas wykonania: {execution_time:.6f} sekund")
    list_for_printing = []
    # Saving to csv dictionary
    if recognized:
        results[frame_num] = {'text': my_license_plate,
                              'text_score': my_license_plate_score,
                              'time': execution_time}
        list_for_printing = [corner_1,corner_2,license_number_print, my_license_plate]
    return recognized, frame, results, list_for_printing

