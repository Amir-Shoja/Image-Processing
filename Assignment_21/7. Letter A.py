import numpy as np
import cv2

img = np.zeros((400, 300), dtype=np.uint8)
img[::] = 255

img[110:135, 45:145] = 0
img[235:260, 45:145] = 0

img[135:235, 20:45] = 0
img[135:235, 145:170] = 0

img[260:360, 20:45] = 0
img[260:360, 145:170] = 0

cv2.imwrite("letterA.jpg", img)
cv2.imshow("", img)
cv2.waitKey()
