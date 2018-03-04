# -*- coding: utf-8 -*-

from shape_detector import ShapeDetector
import argparse
import imutils
import cv2

filename = "../harris-corner-detection/3_4-11-0.jpg"
#filename = "../originals/3.jpg"
image = cv2.imread(filename)

# smaller size images can be approximated better
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert to grayscale (as always)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#cv2.imwrite("blurred.jpg", blurred)

#ret,thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

#cv2.imwrite("new_test.jpg", thresh)

cnts = cv2.findContours(blurred.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]


image1 = cv2.imread('../originals/3_w300.jpg', cv2.IMREAD_COLOR)
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 5)
cv2.imwrite("3_contour_new.jpg", image1)

print("contours found: " + str(len(cnts)))

sd = ShapeDetector()

for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
    M = cv2.moments(c)

    if M["m00"] > 0:
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)
 
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
     
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)






