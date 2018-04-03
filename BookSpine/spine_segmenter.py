#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:42:09 2018

@author: aram
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import imutils

# Canny on original (grayscale) ============================================ :)
image = cv2.imread("images/7.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
res = imutils.resize(gray, width = 1000)
#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#res = imutils.resize(thresh, width = 1000)
#cv2.imshow("1. thresh", res)
#res = cv2.GaussianBlur(res, (5, 5), 0)
#cv2.imshow("2. gauss", res)
can = cv2.Canny(res, 30, 150)
cv2.imshow("3. cannied on original", can)
cv2.imwrite("non-bilat-canned.jpg", can)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Canny on bilat =========================================================== :)
image = cv2.imread("images/7.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
res = imutils.resize(gray, width = 1000)

bilat = cv2.bilateralFilter(res, 9, 41, 41)

can = cv2.Canny(bilat, 30, 150)
cv2.imshow("3. cannied on bilat", can)
cv2.imwrite("bilat-canned.jpg", can)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Gray - Gauss - Blackhat - Scharr - Canny ================================= :(
image = cv2.imread("images/7.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
res = imutils.resize(gray, width = 1000)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

gaus = cv2.GaussianBlur(res, (5, 5), 0)
cv2.imshow("gaus", gaus)
cv2.imwrite("gaus.jpg", gaus)

blackhat = cv2.morphologyEx(gaus, cv2.MORPH_BLACKHAT, rectKernel)
cv2.imshow("blackhat", blackhat)
cv2.imwrite("blackhat.jpg", blackhat)

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

cv2.imshow("gradX", gradX)
cv2.imwrite("gradX.jpg", gradX)

can = cv2.Canny(gradX, 30, 150)
cv2.imshow("canny", can)
cv2.imwrite("canny.jpg", can)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Gray - Scharr - Canny ==================================================== :)
image = cv2.imread("images/7.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
res = imutils.resize(gray, width = 1000)

# compute the Scharr gradient of the blackhat image and scale the result into the range [0, 255]
gradX = cv2.Sobel(res, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
cv2.imshow("gradX", gradX)
cv2.imwrite("gradX.jpg", gradX)

can = cv2.Canny(gradX, 30, 150)
cv2.imshow("canny", can)
cv2.imwrite("canny.jpg", can)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Gray - Scharr - Thresh =================================================== :)
image = cv2.imread("images/7.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
res = imutils.resize(gray, width = 1000)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

# compute the Scharr gradient of the blackhat image and scale the result into the range [0, 255]
gradX = cv2.Sobel(res, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
cv2.imwrite("gradX.jpg", gradX)

# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
threshA = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite("thresh.jpg", thresh)
cv2.imwrite("threshA.jpg", threshA)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Hough lines =================================================================
import math
image = cv2.imread("contours.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

lines = cv2.HoughLines(image = gray, 
                       rho = 1, 
                       theta = np.pi / 180, 
                       threshold = 300)
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

        
cv2.imwrite("hough_lines.jpg", cdst)

# Hough lines (Probabilistic) =================================================
image = cv2.imread("thresh.jpg")
cdstP = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
linesP = cv2.HoughLinesP(image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 
                         rho = 1, 
                         theta = np.pi/180, 
                         threshold = 100, 
                         minLineLength = 10) # maxLineGap

a, b, c = linesP.shape
cdstP = image.copy()

for i in range(a):
    x0, y0, x1, y1 = linesP[i][0]
    cv2.line(cdstP, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    #x1, y1, x2, y2 = line[0]
    #cv2.line(cdstP, (x1,y1), (x2,y2), (0, 255, 0), 2)

cv2.imwrite('hough_linesP.jpg',cdstP)

# Contour detection on edged image ================================== epic fail
image = cv2.imread("thresh.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img, cnts, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
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
cv2.imwrite("contours.jpg", dst)

#imagePaths = glob.glob("images/*.jpg") =======================================
imagePaths = ['images/3.jpg', 'images/13.jpg', 'images/14.jpg', 'images/7.jpg']

for path in imagePaths:
    print(path)

'''
for filename in os.listdir(img_folder):
    print(os.path.join(img_folder,filename))
'''
images = []
for filename in imagePaths:
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #res = imutils.resize(thresh, width = 760)
    images.append(thresh)


for i in range(len(images)):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(str(i) + ".jpg")
    plt.xticks([]),plt.yticks([])
plt.show()



