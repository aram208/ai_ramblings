# -*- coding: utf-8 -*-

import numpy as np
import cv2
import imutils
import argparse
import glob
import re

def do_segment_v1(filename, outDir):

    fprefix = re.search(r'images\/(.*?).jpg', filename).group(1) + "_"
    outDir += fprefix
    
    # Gray - Scharr - Thresh =================================================== :)
    image = cv2.imread(filename)
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
    cv2.imwrite(outDir + "v1_thresh.jpg", thresh)
    cv2.imwrite(outDir + "v1_threshA.jpg", threshA)
    
    # Contour detection on edged image ================================== epic fail
    image = cv2.imread(outDir + "v1_thresh.jpg")
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
    cv2.imwrite(outDir + "v1_contours.jpg", dst)
    
    # Hough lines =================================================================
    image = cv2.imread(outDir + "v1_contours.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    final, _ = imutils.hough_lines(gray, backgroundImage = filename, threshold = 325)
    cv2.imwrite(outDir + "hough_lines.jpg", final)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inpDir", required = True, help = "Path to the input images folder")
ap.add_argument("-o", "--outDir", required = True, help = "Path to the output images folder")
args = vars(ap.parse_args())

outDir = args["outDir"] + "/v1/"
inpDir = args["inpDir"]

imagePaths = glob.glob(inpDir + "/*.jpg")
for filename in imagePaths:
    do_segment_v1(filename, outDir)