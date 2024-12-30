import cv2
import numpy as np
import math
import imutils

def perspective_transformation(frame,points):

    # Selecting the corners
    top_left = points[0]
    bottom_left = points[2]
    top_right = points[1]
    bottom_right = points[3]

    correction = 0  # Correcting parameter, to ensure all license plate is present
    pts1 = np.float32([[top_left[0] - correction, top_left[1] - correction],
                       [top_right[0] + correction, top_right[1] - correction],
                       [bottom_left[0] - correction, bottom_left[1] + correction],
                       [bottom_right[0] + correction, bottom_right[1] + correction]])

    # Final rectangle sizes
    width = int(max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3])))
    height = int(max(np.linalg.norm(pts1[0] - pts1[2]), np.linalg.norm(pts1[1] - pts1[3])))

    pts2 = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])  # Final rectangle corners

    # Perspective transformation
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped


def lines_top_ends(license_plate, edged_frame, lines):
    cdstP = cv2.cvtColor(edged_frame, cv2.COLOR_GRAY2BGR)
    lines_with_lengths = []
    # Each line length calculation and writing as touple of parameters: ((x1, y1, x2, y2), length)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(cdstP, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #cv2.line(license_plate, (x1, y1), (x2, y2), (0, 0, 255), 2)
        lines_with_lengths.append([(x1, y1, x2, y2), np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)])

    # Posortuj linie malejąco według długości
    lines_with_lengths.sort(key=lambda x: x[1], reverse=True)
    top_left = None
    bottom_left = None
    top_right = None
    bottom_right = None

    height, width = edged_frame.shape
    borders = [
        (0, 0, width, 0),  # Top border
        (width, 0, width, height),  # Right border
        (width, height, 0, height),  # Bottom border
        (0, height, 0, 0)  # Left border
    ]
    intersection_points = []
    for line in lines_with_lengths:
        line_coords = line[0]
        for border in borders:
            x1, y1, x2, y2 = line_coords
            x3, y3, x4, y4 = border

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None  # Lines are parallel

            intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

            if intersect_x is not None and intersect_y is not None:
                # Check if intersection is within the image bounds
                if 0 <= intersect_x <= width and 0 <= intersect_y <= height:
                    intersection_points.append((intersect_x, intersect_y))

    for x, y in intersection_points:
        # find top-left corner
        if top_left is None or (x + y < top_left[0] + top_left[1]) and x < (0.1 * width):
            top_left = (abs(x), abs(y))

        # find bottom-left corner
        if bottom_left is None or (x - y < bottom_left[0] - bottom_left[1]) and y > (0.5*height) and x < (0.1*width):
            bottom_left = (abs(x), abs(y))

        # find top-right corner
        if top_right is None or (x - y > top_right[0] - top_right[1]) and x > (0.99 * width):
            top_right = (abs(x), abs(y))

        # find bottom-right corner
        if bottom_right is None or (x + y > bottom_right[0] + bottom_right[1]) and y > (0.5*height) and x > (0.99 * width):
            bottom_right = (abs(x), abs(y))

    corners = [top_left, top_right, bottom_left, bottom_right]
    if any(corner is None for corner in corners):
        return None
    print(corners)
    cv2.imshow("lines", cdstP)
    #cv2.imshow("lines_original", license_plate)
    return corners
def cropping_with_hugh(license_plate, edged_frame):
    cdstP = cv2.cvtColor(edged_frame, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(edged_frame, 1, math.pi/180, 100)

    crop_regions = []
    cropped_images =[]
    if lines is not None:
        # list of corners
        points = lines_top_ends(license_plate, edged_frame, lines)

        if points is not None:
            warped = perspective_transformation(license_plate, points)

            cv2.imshow("warped", warped)
            return warped
    return None

def contours_fc(license_plate):
    license_plate_copy = license_plate.copy()
    gray_license = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)  # Grayscale conversion
    height, width = gray_license.shape  # Shape reading

    # Binarization of the image
    ret, threshold = cv2.threshold(gray_license, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Every contour extraction
    contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Reorganization of contours
    contours = imutils.grab_contours(contours)

    if not contours:
        print("No contours found!")
        return license_plate

        # Find the largest slanted rectangle
    largest_contour = max(contours, key=cv2.contourArea)

    # Check if the largest contour is valid
    if largest_contour is None or len(largest_contour) < 3:
        print("No valid contour found!")
        return license_plate

    # Get the minimum area rectangle for the largest contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    # Check if the box points are valid
    if box is None or len(box) == 0:
        print("No valid box points!")
        return license_plate

    # Draw the contour for visualization
    cv2.drawContours(license_plate_copy, [box], 0, (0, 0, 255), 2)

    # Calculate the angle to straighten the rectangle
    angle = rect[-1]
    if angle < -45:
        angle += 90  # Correct for OpenCV's angle range

    # Rotate the entire image to align the rectangle
    center = rect[0]  # Center of the rectangle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(license_plate, rotation_matrix, (width, height))

    # Crop the aligned region
    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    ret, rotated_threshold = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the region
    cropped = rotated[y:y + h, x:x + w]

    # Show the results
    #cv2.imshow("Original with Contour", license_plate_copy)
    cv2.imshow("Straightened License Plate", cropped)
    return cropped

def preprocess(license_plate):
    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.5)
    #threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edged_frame = cv2.Canny(blur, 30, 255)
    cropped_license = cropping_with_hugh(license_plate, edged_frame)
    #cropped_license1 = contours_fc(license_plate)
    return cropped_license