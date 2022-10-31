import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_blurred = cv2.GaussianBlur(frame_gray, (7, 7), 3)

    thresh = cv2.adaptiveThreshold(
        frame_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sudoku_contour = None

    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            sudoku_contour = approx
            break

    if sudoku_contour is None:
        cv2.putText(
            frame, "Not found", (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3
        )
    else:
        cv2.drawContours(frame, [sudoku_contour], -1, (0, 255, 0), 10)

        cv2.imshow("webcam", frame)
        key = cv2.waitKey(1)
        if key == ord("s"):
            warped = four_point_transform(frame, approx.reshape(4, 2))
            cv2.imwrite(
                "G:\Program\PyLearn\Assignment_28\output\sudoku.jpg",
                warped,
            )
        if key == 27:
            break
