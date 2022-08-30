import cv2 as cv
import numpy as np
import dlib


def rmv_back(img, position):
    img2gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    background = cv.bitwise_and(position, position, mask=mask_inv)
    mask_img = cv.bitwise_and(img, img, mask=mask)
    emoji_without_back = cv.add(mask_img, background)

    return emoji_without_back


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faceDetector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detector = cv.CascadeClassifier("haarcascade_eye.xml")
mouth_detector = cv.CascadeClassifier("haarcascade_smile.xml")

emoji = cv.imread("face with peeking eye.png")
eye = cv.imread("eye.png")
mouth = cv.imread("mouth2.png")

cap = cv.VideoCapture(0)
key = 0
chooseEffect = 0

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    faces = faceDetector.detectMultiScale(frame, 1.2, minNeighbors=5)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    for (x, y, w, h) in faces:
        facePosition = frame[y : y + h, x : x + w]

        if chooseEffect == 1:  # emoji on face
            img = cv.resize(emoji, (w, h))
            emoji_without_back = rmv_back(img, facePosition)

            frame[y : y + h, x : x + h] = emoji_without_back

        elif chooseEffect == 2:  # eye and mouth
            eyes = eye_detector.detectMultiScale(facePosition, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                eyePosition = frame[y + ey : y + ey + eh, x + ex : x + ex + ew]

                img = cv.resize(eye, (ew, eh))
                eye_without_back = rmv_back(img, eyePosition)
                frame[y + ey : y + ey + eh, x + ex : x + ex + ew] = eye_without_back

            lips = mouth_detector.detectMultiScale(facePosition, 1.9, minNeighbors=20)
            for (mx, my, mw, mh) in lips:
                mouthPosition = frame[y + my : y + my + mh, x + mx : x + mx + mw]

                img = cv.resize(mouth, (mw, mh))
                mouth_without_back = rmv_back(img, mouthPosition)
                frame[y + my : y + my + mh, x + mx : x + mx + mw] = mouth_without_back

        elif chooseEffect == 3:  # checkered face
            sq = cv.resize(frame[y : y + h, x : x + w], (10, 10))
            checkered_face = cv.resize(sq, (w, h), interpolation=cv.INTER_NEAREST)
            frame[y : y + h, x : x + h] = checkered_face

        elif chooseEffect == 4:  # face landmarks

            faces = detector(gray)
            for faces in faces:
                x1 = faces.left()
                y1 = faces.top()
                x2 = faces.right()
                y2 = faces.bottom()
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                landmarks = predictor(gray, faces)
                print(landmarks)

                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv.circle(frame, (x, y), 4, (21, 231, 254), -1)

        elif chooseEffect == 5:  # Blur
            blur = cv.GaussianBlur(facePosition, (91, 91), 0)
            frame[y : y + h, x : x + w] = blur

    # manage user choice
    if key == 27:  # ese
        break
    elif key == 49:
        chooseEffect = 1
    elif key == 50:
        chooseEffect = 2
    elif key == 51:
        chooseEffect = 3
    elif key == 52:
        chooseEffect = 4
    elif key == 53:
        chooseEffect = 5

    cv.imshow("output", frame)
    key = cv.waitKey(1)

cap.release()
