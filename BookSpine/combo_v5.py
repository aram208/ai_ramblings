# -*- coding: utf-8 -*-
# same as v4 only with bucketing the roh values to find the median
import cv2
import imutils
import numpy as np
import argparse
import glob
import re

def do_segment_v5(filename, outDir):
     
    fprefix = re.search(r'images\/(.*?).jpg', filename).group(1) + "_"
    outDir += fprefix

    print('fprefix: ' + fprefix)
    print('outdir:' + outDir)

    # Let's try masking!!
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = imutils.resize(gray, width = 1000)
    
    mask = np.zeros(res.shape[:2], dtype = 'uint8')
    (cX, cY) = (res.shape[1] // 2, res.shape[0] // 2)
    hdelta = res.shape[0] // 3 // 2 # half of the third of the height
    cv2.rectangle(mask, (0, cY - hdelta), (999, cY + hdelta), 255, -1)
    
    masked = cv2.bitwise_and(res, res, mask = mask)
    #cv2.imwrite(outDir + "masked.jpg", masked)
    
    bilat = cv2.bilateralFilter(masked, 9, 41, 41)
    #cv2.imwrite(outDir + "bilat.jpg", bilat)
    
    kernelH = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    horizontal_img = cv2.erode(bilat, kernelH, iterations=1)
    horizontal_img = cv2.dilate(horizontal_img, kernelH, iterations=1)
    cv2.imwrite(outDir + "H_processed.jpg", horizontal_img)
    
    gradX = cv2.Sobel(horizontal_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    cv2.imwrite(outDir + "x-sobel.jpg", gradX)
    
    gradY = cv2.Sobel(gradX, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradY = np.absolute(gradY)
    (minVal, maxVal) = (np.min(gradY), np.max(gradY))
    gradY = (255 * ((gradY - minVal) / (maxVal - minVal))).astype("uint8")
    cv2.imwrite(outDir + "y-sobel.jpg", gradY)
    
    # Hough lines =================================================================
    image = cv2.imread(outDir + "y-sobel.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bimage = np.ones(gray.shape[:2], dtype='uint8')
    bimage = cv2.cvtColor(bimage, cv2.COLOR_GRAY2BGR)
    
    final, _ = imutils.hough_lines(gray, backgroundImage = filename, threshold = 200)
    cv2.imwrite(outDir + "hough_lines.jpg", final)
    
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inpDir", required = True, help = "Path to the input images folder")
ap.add_argument("-o", "--outDir", required = True, help = "Path to the output images folder")
args = vars(ap.parse_args())

outDir = args["outDir"] + "/v5/"
inpDir = args["inpDir"]

imagePaths = glob.glob(inpDir + "/*.jpg")
for filename in imagePaths:
    do_segment_v5(filename, outDir)