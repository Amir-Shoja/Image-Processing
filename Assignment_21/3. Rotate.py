import cv2

img = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_21\p3.jpg")
img = cv2.rotate(img, cv2.ROTATE_180)

cv2.imwrite("rotate.jpg", img)
