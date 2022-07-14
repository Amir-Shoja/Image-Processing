from unittest import result
import cv2
import numpy as np

img1 = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_22\pics\hw2\pa.tif", 1)
img2 = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_22\pics\hw2\pb.tif", 1)

result = img2 - img1

cv2.imwrite("result_a&b.jpg", result)
