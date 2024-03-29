import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        break

    width, height = frame.shape

    detectArea = frame[ (width // 5) * 2 : (width // 5) * 3, (height // 5) * 2 : (height // 5) * 3]
    kernel = np.ones((25, 25)) / 625
    frame = cv2.filter2D(frame, -1, kernel)

    alpha = 3  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    enhanced_target = cv2.convertScaleAbs(detectArea, alpha=alpha, beta=beta)

    frame[(width // 5) * 2 : (width // 5) * 3, (height // 5) * 2 : (height // 5) * 3] = detectArea

    target = np.average(frame[(width // 5) * 2 : (width // 5) * 3, (height // 5) * 2 : (height // 5) * 3])
    if target <= 85:
        color = "Black"
    elif target > 85 and target <= 150:
        color = "Gray"
    elif target > 150:
        color = "White"

    cv2.putText(frame, color, (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    cv2.imshow("output", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
