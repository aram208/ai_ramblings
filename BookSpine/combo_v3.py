# -*- coding: utf-8 -*-

import cv2
import imutils
import numpy as np

# Let's try masking!!
filename = "images/7.jpg"
outDir = "v3/"
image = cv2.imread(filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
res = imutils.resize(gray, width = 1000)

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
imutils.hough_lines(gray, backgroundImage = filename, outDir = outDir, threshold = 150)