import cv2
import numpy as np

img = cv2.imread("Input/building.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result1 = np.zeros(img.shape)
result2 = np.zeros(img.shape)

mask1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
mask2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

rows, cols = img.shape
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        small_img = img[i - 1 : i + 2, j - 1 : j + 2]
        result1[i, j] = np.sum(small_img * mask1)
        result2[i, j] = np.sum(small_img * mask2)

cv2.imwrite("Output/building_filter1.jpg", result1)
cv2.imwrite("Output/building_filter2.jpg", result2)
