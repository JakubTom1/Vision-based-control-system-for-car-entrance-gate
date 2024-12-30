import cv2
import numpy as np
import os
import json
from scripts.usefull import my_license_plates

def license_complies_format(license_plate):
    """
    Check if the license plate text complies with the required format.

    Args:
        license_plates (list): List of formatted, possible license plates.

    Returns:
        tuple: (bool, str) True if the license plate complies with the format, text of confirmed license plate.
    """
    myPlates_list, myPlates_lists = my_license_plates()

    if license_plate in myPlates_list:
        return True, license_plate

    return False, None

# Function to load templates
def load_templates(template_folder):
    templates = {}
    for filename in os.listdir(template_folder):
        if filename.endswith('.png'):
            template = cv2.imread(os.path.join(template_folder, filename), cv2.IMREAD_GRAYSCALE)
            template = cv2.resize(template, (28, 28))
            #template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            templates[filename[:-4]] = template  # Store without .png extension
    return templates


# Function to perform template matching
def match_characters(input_images, templates):
    score = 0
    license_plate = ''
    confidency = []
    for image, x, y in input_images:
        best_match = ''
        max_val = -1
        if image is None or image.size == 0:
            print(f"Skipping empty or invalid image at ({x}, {y})")
            continue
        try:
            image = cv2.resize(image, (28, 28))
        except Exception as e:
            print(f"Error resizing image at ({x}, {y}): {e}")
            continue

        for char, template in templates.items():
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val_temp, _, _ = cv2.minMaxLoc(result)
            if max_val_temp > max_val:
                max_val = max_val_temp
                best_match = char
        if max_val > 0.2:  # Threshold for matching
            license_plate = license_plate + best_match
        else:
            license_plate = license_plate + '_'
        confidency.append(max_val)
    if len(confidency)>0:
        score = sum(confidency)/len(confidency)
    return license_plate, score


# Main function
def characters_recognition(DIRECTORY, characters_list):
    template_folder = os.path.join(DIRECTORY, "/scripts/classical_mathods/template")
    # Load character templates
    templates = load_templates(template_folder)
    # Match characters
    license_plate, score = match_characters(characters_list, templates)
    print(license_plate,score)
    # Is my license
    recognized, my_license_plate = license_complies_format(license_plate)

    return recognized, my_license_plate, score

