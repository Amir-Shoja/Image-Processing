import cv2

result = 0
for i in range(15):
    img = cv2.imread(f"G:\Program\Python\PyLearn\IP\Assignment_22\pics\hw2\highway\h{i}.jpg")
    result += img//15

cv2.imwrite("highWay.jpg", result)