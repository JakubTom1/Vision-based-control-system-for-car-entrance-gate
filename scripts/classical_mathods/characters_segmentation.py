import cv2
import numpy as np
import math
import imutils
import os
import random
def characters_segmentation(license_plate):
    """
    Segmentation of license plate image to extract each character picture, its coordinates and shape.
    :param license_plate: Image of extracted, preprocessed license plate
    :return: List of tuples containing image, Horizontal position and Vertical position.
    """

    license_plate_copy = license_plate.copy()
    gray_license = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)  # Grayscale conversion
    height, width = gray_license.shape  # Shape reading

    # Binarization of the image
    ret, threshold = cv2.threshold(gray_license, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Every contour extraction
    contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Reorganization of contours
    contours = imutils.grab_contours(contours)

    # Constructing image for future extracted contours visualization
    text_mask = np.zeros((height, width), dtype=np.uint8)

    # To achieve better modularity we'll store the character bounding box inside
    # an array.
    bounding_contours = []
    characters_list = []
    idx = 1

    # Iterate through all the possible contour in the contours array
    for i, contour in enumerate(contours):

        # Reading parameters of rectangle containing the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Out of bonds check
        if x < 1 or x > width-1:
            continue
        # Check if our contour is not too small or too big for char
        if w < 0.2*width and  0.6*height <= h < 0.9*height:
            cv2.drawContours(text_mask, [contour], 0, 255, -1)
            # Adding character contour to the list
            bounding_contours.append(contour)

            # Cutting character section from binarized image
            if h / w > 2:
                character = threshold[y:y + h, x - int(w / 2):x + (2 * w)]
            else:
                character = threshold[y:y + h, x:x + w]
            characters_list.append((character, x, y))

            # Visualization
            cv2.rectangle(license_plate_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #idx = random.randint(1,1000)
            #cv2.imshow(f'char{idx}', character)
            #cv2.imwrite(f'char{idx}.png', character)
            idx += 1
            continue
    # Sort the remaining characters from left to right
    characters_list = sorted(characters_list, key=lambda x: x[1])
    if len(characters_list) <7 or len(characters_list)>8:
        return None
    #cv2.imshow('text_mask', text_mask)
    cv2.imshow('license with rectangles', license_plate_copy)

    return characters_list

