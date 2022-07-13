import cv2

img = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_21\lion.png", 0)

for i in range(100):
    for j in range(100 - i, 100 - i - 10, -1):
        img[i - 10 : i, j] = 0
for i in range(10):
    for j in range(90, 100):
        img[i, 90 - i : 100 - i] = 0

cv2.imwrite("Corner.jpg", img)
cv2.imshow("", img)
cv2.waitKey()
