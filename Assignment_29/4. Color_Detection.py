import cv2 as cv
import numpy as np

# سفید - خاکستری - سیاه - آبی - قرمز - سبز - سرخابی - زرد - فیروزه ای

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    width, height, _ = frame.shape

    b, g, r = cv.split(frame)

    target = frame[
        (width // 8) * 3 : (width // 8) * 5, (height // 8) * 3 : (height // 8) * 5
    ]

    b_target, g_target, r_target = cv.split(target)

    alpha = 2  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    r_target = cv.convertScaleAbs(r_target, alpha=alpha, beta=beta)
    g_target = cv.convertScaleAbs(g_target, alpha=alpha, beta=beta)
    b_target = cv.convertScaleAbs(b_target, alpha=alpha, beta=beta)

    r_avg = round(np.average(r_target))
    g_avg = round(np.average(g_target))
    b_avg = round(np.average(b_target))

    target = cv.merge((b_target, g_target, r_target))

    kernel = np.ones((45, 45), np.float32) / 2025
    b = cv.filter2D(b, -1, kernel, borderType=cv.BORDER_CONSTANT)
    g = cv.filter2D(g, -1, kernel, borderType=cv.BORDER_CONSTANT)
    r = cv.filter2D(r, -1, kernel, borderType=cv.BORDER_CONSTANT)

    frame = cv.merge((b, g, r))

    frame[
        (width // 8) * 3 : (width // 8) * 5, (height // 8) * 3 : (height // 8) * 5
    ] = target
    cv.rectangle(
        frame,
        (height // 8 * 3, width // 8 * 3),
        ((height // 8 * 5), (width // 8 * 5)),
        (0, 255, 0),
        4,
    )

    # detect color and put text ... r_avg , b_avg ...
    if r_avg > 220 and g_avg > 220 and b_avg > 220:
        cv.putText(frame, "White", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    elif r_avg < 60 and g_avg < 60 and b_avg < 60:
        cv.putText(frame, "Black", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    elif 170 > r_avg > 70 and 170 > g_avg > 70 and 170 > b_avg > 70:
        cv.putText(frame, "Gray", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

    elif r_avg > 170 and g_avg < 40 and b_avg < 40:
        cv.putText(frame, "Red", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    elif r_avg < 40 and g_avg > 170 and b_avg < 40:
        cv.putText(frame, "Green", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    elif r_avg < 40 and g_avg < 40 and b_avg > 170:
        cv.putText(frame, "Blue", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

    elif r_avg > 170 and g_avg > 170 and b_avg < 40:
        cv.putText(frame, "Yellow", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    elif r_avg < 40 and g_avg > 170 and b_avg > 170:
        cv.putText(frame, "Cyan", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    elif r_avg > 170 and g_avg < 40 and b_avg > 170:
        cv.putText(frame, "Magenta", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

    cv.imshow("cam", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
