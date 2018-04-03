# -*- coding: utf-8 -*-

import numpy as np
import cv2
import imutils
import math

image = cv2.imread("images/7.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
res = imutils.resize(gray, width = 1000)

bilat = cv2.bilateralFilter(res, 9, 41, 41)
cv2.imwrite("v2_bilat.jpg", bilat)
can = cv2.Canny(bilat, 30, 150)
cv2.imwrite("v2_canny.jpg", can)


# Hough lines =================================================================

image = cv2.imread("v2_canny.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

lines = cv2.HoughLines(image = gray, 
                       rho = 1, 
                       theta = np.pi / 180, 
                       threshold = 255)

original = cv2.imread("images/7.jpg")
original = imutils.resize(original, width=1000)
cdst = original.copy()
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        
        # for vertical lines
        if theta > np.pi/180*170 or theta < np.pi/180*10 :
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
        # for horizontal lines	
        #if( theta>CV_PI/180*80 && theta<CV_PI/180*100)

        
cv2.imwrite("v2_hough_lines.jpg", cdst)

