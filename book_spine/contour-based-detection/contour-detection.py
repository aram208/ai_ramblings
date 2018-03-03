# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('../originals/3.jpg', cv2.IMREAD_GRAYSCALE)

# BEGIN_FIXED------------------------------------------------------------------
ret,thresh = cv2.threshold(img, 80, 255, cv2.THRESH_TOZERO)
#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
#th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # works better

im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


cv2.imwrite("3_contour_a.jpg", im2)
# END_FIXED--------------------------------------------------------------------
image1 = cv2.imread('originals/3.jpg', cv2.IMREAD_COLOR)
cv2.drawContours(image1, contours, -1, (0, 255, 0), 5)
cv2.imwrite("3_contour_b.jpg", image1)

i = 0
largeContours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 3000 :
        print("idx: " + str(i) + " area: " + str(area))
        largeContours.append(cnt)
        i += 1
img_colored = cv2.imread('../originals/3.jpg', cv2.IMREAD_COLOR )
cv2.drawContours(img_colored, largeContours, -1, (0, 255, 0), 5)
cv2.imwrite("3_contour_i.jpg", img_colored)

# bounding rect
green_contoured_image = cv2.imread('../originals/3.jpg', cv2.IMREAD_COLOR)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 3000 :
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(green_contoured_image,(x,y),(x+w,y+h),(0,255,0),5)
cv2.imwrite("3_contour_j.jpg", green_contoured_image)

# min area rect (for tilted books)
orig_image = cv2.imread('../originals/3.jpg', cv2.IMREAD_COLOR)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 3000 :
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(orig_image,[box],0,(0,255,0),5)
cv2.imwrite("3_contour_k.jpg", orig_image)
# -----------------------------------------------------------------------------
