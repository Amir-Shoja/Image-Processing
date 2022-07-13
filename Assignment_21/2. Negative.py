import cv2

img1 = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_21\p1.jpg")
img2 = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_21\p2.jpg")

img_not1 = cv2.bitwise_not(img1)
img_not2 = cv2.bitwise_not(img2)

cv2.imwrite("Invert2.jpg", img_not1)
cv2.imwrite("Invert1.jpg", img_not2)
