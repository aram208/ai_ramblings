# -*- coding: utf-8 -*-

import numpy as np
import cv2
import imutils
import math

# Gray - Scharr - Thresh =================================================== :)
image = cv2.imread("images/7.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
res = imutils.resize(gray, width = 1000)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

# compute the Scharr gradient of the blackhat image and scale the result into the range [0, 255]
gradX = cv2.Sobel(res, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
threshA = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite("v1_thresh.jpg", thresh)
cv2.imwrite("v1_threshA.jpg", threshA)

# Contour detection on edged image ================================== epic fail
image = cv2.imread("v1_thresh.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img, cnts, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow("contours", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
rects = []

for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.4 * perimeter, True)
    
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if not (ar >= 0.95 and ar <= 1.05):
            rects.append(c)

dst = image.copy()
cv2.drawContours(dst, rects, -1, (0, 255, 0), 2)
cv2.imwrite("v1_contours.jpg", dst)

# Hough lines =================================================================
image = cv2.imread("v1_contours.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

lines = cv2.HoughLines(image = gray, 
                       rho = 1, 
                       theta = np.pi / 180, 
                       threshold = 325)
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

        
cv2.imwrite("v1_hough_lines.jpg", cdst)

