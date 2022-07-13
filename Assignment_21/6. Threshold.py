import cv2
import numpy as np

img = np.zeros((255, 255), dtype=np.uint8)
img[:] = 255
height, width = img.shape

for i in range(height):
    for j in range(width):
        img[i, :] = 255 - i

cv2.imwrite("G:\Program\Python\PyLearn\IP\Assignment_21\Threshold.jpg", img)
cv2.imshow("", img)
cv2.waitKey()
