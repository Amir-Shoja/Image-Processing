import cv2
import numpy as np

img = cv2.imread("Input/flower_input.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result = np.zeros(img.shape)
mask = np.ones((13, 13)) / 169

rows, cols = img.shape
for i in range(6, rows - 6):
    for j in range(6, cols - 6):
        if img[i, j] < 180:
            small_img = img[i - 6 : i + 7, j - 6 : j + 7]
            result[i, j] = np.sum(small_img * mask)
        else:
            result[i, j] = img[i, j]

cv2.imwrite("Output/flower_blur.jpg", result)
