import cv2
import numpy as np

img = np.zeros((800, 800), dtype=np.uint8)
height, width = img.shape

for i in range(0, height, 100):
    for j in range(0, width, 100):
        if (i // 100) % 2 != 0 and (j // 100) % 2 == 0:
                img[i : i + 100, j : j + 100] = 255
        elif (i // 100) % 2 == 0 and (j // 100) % 2 != 0:
                img[i : i + 100, j : j + 100] = 255

cv2.imwrite('G:\Program\Python\PyLearn\IP\Assignment_21\chessBoard.jpg' , img)
cv2.imshow("chess board", img)
cv2.waitKey()
