import cv2
import numpy as np

image = []
nameFile = ["one", "Two", "Three", "Four"]

for i in range(4):
    finalImg = np.zeros((1000, 1000), dtype="uint8")
    for j in range(1, 5):
        img = cv2.imread(f"G:\Program\Python\PyLearn\IP\Assignment_22\pics\hw2\BlackHole\{nameFile[i]}\{j}.jpg",0)
        finalImg += img // 6
    image.append(finalImg)

completeImg = np.zeros((2000, 2000), dtype="uint8")

completeImg[0:1000, 0:1000] = image[0]
completeImg[0:1000, 1000:2000] = image[1]
completeImg[1000:2000, 0:1000] = image[2]
completeImg[1000:2000, 1000:2000] = image[3]

cv2.imwrite("G:\Program\Python\PyLearn\IP\Assignment_22\completeImg.jpg", completeImg)
