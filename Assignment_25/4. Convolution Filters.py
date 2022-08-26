import cv2
import numpy as np

img = cv2.imread("Input\jupiter.jpg", 0)
rows, cols = img.shape
completeImg = np.zeros((rows, cols * 4), dtype="uint8")


def filters(size):
    result = np.zeros(img.shape)
    mask = np.ones((size, size)) / (size**2)

    for i in range(size // 2, rows - size // 2):
        for j in range(size // 2, cols - size // 2):
            small_img = img[i - size // 2 : i + size // 2 + 1, j - size // 2 : j + size // 2 + 1]
            result[i, j] = np.sum(small_img * mask)
    return result


filterSize = [3, 5, 7, 15]
i = 1

for item in filterSize:
    picture = filters(item)
    # cv2.imwrite(f"Output/jupiter{i}.jpg", picture)
    completeImg[0:rows, cols * (i - 1) : cols * i] = picture
    i += 1

cv2.imwrite("Output/jupiter.jpg", completeImg)
