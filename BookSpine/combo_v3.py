# -*- coding: utf-8 -*-

import cv2
import imutils
import numpy as np
import re
import argparse
import glob

def do_segment_v3(filename, outDir):
    
    fprefix = re.search(r'images\/(.*?).jpg', filename).group(1) + "_"
    outDir += fprefix
    
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = imutils.resize(gray, width = 1000)
    
    # Let's try masking!!
    mask = np.zeros(res.shape[:2], dtype = 'uint8')
    (cX, cY) = (res.shape[1] // 2, res.shape[0] // 2)
    hdelta = res.shape[0] // 3 // 2 # half of the third of the height
    cv2.rectangle(mask, (0, cY - hdelta), (999, cY + hdelta), 255, -1)
    
    masked = cv2.bitwise_and(res, res, mask = mask)
    #cv2.imshow("Masked", masked)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    bilat = cv2.bilateralFilter(masked, 9, 41, 41)
    cv2.imwrite(outDir + "bilat.jpg", bilat)
    can = cv2.Canny(bilat, 30, 150)
    cv2.imwrite(outDir + "canny.jpg", can)
    
    # Hough lines =================================================================
    image = cv2.imread(outDir + "canny.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    final, _ = imutils.hough_lines(gray, backgroundImage = filename, threshold = 150)
    cv2.imwrite(outDir + "hough_lines.jpg", final)
    
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inpDir", required = True, help = "Path to the input images folder")
ap.add_argument("-o", "--outDir", required = True, help = "Path to the output images folder")
args = vars(ap.parse_args())

outDir = args["outDir"] + "/v3/"
inpDir = args["inpDir"]

imagePaths = glob.glob(inpDir + "/*.jpg")
for filename in imagePaths:
    do_segment_v3(filename, outDir)