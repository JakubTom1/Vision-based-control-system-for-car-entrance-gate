import cv2
import os
import numpy as np
import time
import imutils

def morphological_operations(gray_license):
    start_time = time.time()
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    blackhat = cv2.morphologyEx(gray_license, cv2.MORPH_BLACKHAT, rectKern)

    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    light = cv2.morphologyEx(gray_license, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gradX = np.absolute(gradX)
    gradX = 255 * ((gradX - np.min(gradX)) / (np.max(gradX) - np.min(gradX)))
    gradX = gradX.astype("uint8")
    gradX = cv2.GaussianBlur(gradX, (9, 9), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)

    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.medianBlur(thresh, 5)
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    end_time = time.time()  # Zapisz czas zakończenia

    execution_time = end_time - start_time  # Oblicz czas wykonania
    print(f"Czas wykonania rekonstrukcji: {execution_time:.6f} sekund")
    '''cv2.imshow('blackhat', blackhat)
    cv2.imshow('light', light)
    cv2.imshow('thresh', thresh)
    cv2.imshow('gradX', gradX)'''

    return thresh
def license_plate_localization(frame):
    frame_copy = frame.copy()
    gray_license = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_license.shape
    thresh = morphological_operations(gray_license)


    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    cropped_images = []
    for i, contour in enumerate(contours):
        # Reading parameters of rectangle containing the contour
        x, y, w, h = cv2.boundingRect(contour)
        if w/h > 2 and 0.06*width < w < 0.5*width:
            #cv2.imshow(f"cropped{i}", frame[y:y + h, x:x + w])
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if y-0.2*h>0 and y+h+0.5*h<height and x-0.05*w>0 and x+1.05*w<width:
                cropped_images.append([frame[y-int(0.2*h):y+h+int(0.5*h), x-int(0.05*w):x+w], (x, y, w, h)])
    #cv2.imshow('frame_copy', frame_copy)
    return cropped_images
