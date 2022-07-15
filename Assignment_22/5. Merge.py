import cv2
import numpy as np

amir = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_22\pics\Amir.jpg", 0)
jim = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_22\pics\Jim.jpg", 0)

height, width = amir.shape
completeImg = np.zeros((height, width * 4), dtype="uint8")

completeImg[0:height, 0:width] = amir
completeImg[0:height, width : width * 2] = amir // 2 + jim // 4
completeImg[0:height, width * 2 : width * 3] = amir // 4 + jim // 2
completeImg[0:height, width * 3 : width * 4] = jim

cv2.imwrite("G:\Program\Python\PyLearn\IP\Assignment_22\merge.jpg", completeImg)
