import cv2
import numpy as np
from time import time

min_width = 80  # Minimum rectangle width
min_height = 80  # Minimum rectangle height
offset = 6  # Allowed pixel error for the counting line position
line_pos = 550  # Y-coordinate of the counting line
fps = 60  # Video FPS

detections = [] #List to store the center coordinates of detected objects
cars_count = 0
total_frames = 0
total_cars = 0
start_time = time()

def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('vehicle.mp4')
#subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
subtractor = cv2.createBackgroundSubtractorMOG2()


while True:
    ret, frame1 = cap.read()
    if not ret:
        break
    total_frames += 1
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    img_sub = subtractor.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, line_pos), (1200, line_pos), (255, 127, 0), 3)
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= min_width) and (h >= min_height)
        if not validate_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = get_center(x, y, w, h)
        detections.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detections:
            if y < (line_pos + offset) and y > (line_pos - offset):
                cars_count += 1
                detections.remove((x, y))
                #total_cars += 1
                print("Car detected: " + str(cars_count))
                

    cv2.putText(frame1, "CAR COUNT: " + str(cars_count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Original Video", frame1)
    cv2.imshow("Detect", dilated)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
