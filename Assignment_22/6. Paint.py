import cv2 as cv

img = cv.imread("G:\Program\Python\PyLearn\IP\Assignment_22\pics\Jim.jpg" ,0)

negative = 255 - img
blur = cv.GaussianBlur(negative, (21,21), 0)

sketch = img / (255 - blur)
sketch = sketch * 255

cv.imwrite('G:\Program\Python\PyLearn\IP\Assignment_22\Paint of jim.jpg' , sketch)