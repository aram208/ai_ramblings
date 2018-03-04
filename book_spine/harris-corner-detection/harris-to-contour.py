# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

img_o = cv.imread('../originals/3.jpg', 0)

corners1 = cv.cornerHarris(src = img_o, blockSize = 4, ksize = 11, k = 0, borderType = cv.BORDER_CONSTANT)
cv.imwrite("3_4-11-0.jpg", corners1)


# apply contours and boxes
img_g = cv.imread('3_4-11-0.jpg', 0)
ret,thresh = cv.threshold(img_g, 80, 255, cv.THRESH_TOZERO)
#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
#th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # works better
im2,contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.imwrite("harris_contours.jpg", im2)

green_contoured_image = cv.imread('../originals/3.jpg', cv.IMREAD_COLOR)
#cv.drawContours(green_contoured_image, contours, -1, (0, 255, 0), 5)
#cv.imwrite("3_contour_j.jpg", green_contoured_image)
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > 4000 :
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(green_contoured_image,(x,y),(x+w,y+h),(0,255,0),5)

cv.imwrite("h2k_4000.jpg", green_contoured_image)