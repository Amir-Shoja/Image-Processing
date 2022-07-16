import cv2
import random

img = cv2.imread("G:\Program\Python\PyLearn\IP\Assignment_22\pics\hw2\chess pieces.jpg")

height, width, z = img.shape
noisePixels = random.randint(1000, 2000)

for i in range(noisePixels):
    y = random.randint(0, height - 1)
    x = random.randint(0, width - 1)
    choices = random.choice([True, False])
    if choices == True:
        img[y][x] = 255
    else:
        img[y][x] = 0

cv2.imwrite("G:\Program\Python\PyLearn\IP\Assignment_22\chessNoise.jpg", img)
