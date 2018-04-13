# -*- coding: utf-8 -*-
import cv2
import imutils
import argparse
import glob
import re

def do_segment_v2(filename, outDir):

    fprefix = re.search(r'images\/(.*?).jpg', filename).group(1) + "_"
    outDir += fprefix
    
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = imutils.resize(gray, width = 1000)
    
    bilat = cv2.bilateralFilter(res, 9, 41, 41)
    cv2.imwrite(outDir + "v2_bilat.jpg", bilat)
    can = cv2.Canny(bilat, 30, 150)
    cv2.imwrite(outDir + "v2_canny.jpg", can)
    
    
    # Hough lines =================================================================
    
    image = cv2.imread(outDir + "v2_canny.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    final, _ = imutils.hough_lines(gray, backgroundImage = filename, threshold = 255)
    cv2.imwrite(outDir + "hough_lines.jpg", final)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inpDir", required = True, help = "Path to the input images folder")
ap.add_argument("-o", "--outDir", required = True, help = "Path to the output images folder")
args = vars(ap.parse_args())

outDir = args["outDir"] + "/v2/"
inpDir = args["inpDir"]

imagePaths = glob.glob(inpDir + "/*.jpg")
for filename in imagePaths:
    do_segment_v2(filename, outDir)