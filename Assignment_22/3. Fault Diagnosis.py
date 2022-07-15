import cv2
import numpy as np

originImg = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_22\pics\hw2\Board-origin.bmp", 0)
testImg = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_22\pics\hw2\Board-test.bmp", 0)

cv2.imwrite("FaultDiagnosis.jpg", cv2.subtract(originImg , cv2.flip(testImg,1))*2)