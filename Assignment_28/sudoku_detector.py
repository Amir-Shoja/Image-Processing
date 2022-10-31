import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform


parser = argparse.ArgumentParser(description="Amir Sudoku Detector v1.0")
parser.add_argument("--input", type=str, help="path of your input image")
parser.add_argument("--output", type=str, help="path of your output image")
parser.add_argument("--kernel_size", type=int, help="blur kernel size", default=7)
args = parser.parse_args()

img = cv.imread(args.input)
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

imgBlurred = cv.GaussianBlur(imgGray, (args.kernel_size, args.kernel_size), 3)

thresh = cv.adaptiveThreshold(
    imgBlurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2
)

contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = contours[0]

contours = sorted(contours, key=cv.contourArea, reverse=True)

sudoku_contour = None

for contour in contours:
    epsilon = 0.1 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        sudoku_contour = approx
        break

if sudoku_contour is None:
    print("can't find")

else:
    result = cv.drawContours(img, [sudoku_contour], -1, (0, 255, 0), 15)

    warped = four_point_transform(img, approx.reshape(4, 2))
    warped = cv.resize(warped, (500, 500))
    plt.imshow(warped)
    cv.imwrite(args.output, warped)
