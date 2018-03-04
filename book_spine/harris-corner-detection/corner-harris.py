# -*- coding: utf-8 -*-

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread('originals/3.jpg', 0)

"""
Harris corner detector.

Parameters
    src	Input single-channel 8-bit or floating-point image.
    dst	Image to store the Harris detector responses. It has the type CV_32FC1 and the same size as src .
    blockSize	Neighborhood size (see the details on cornerEigenValsAndVecs ).
    ksize	Aperture parameter for the Sobel operator.
    k	Harris detector free parameter. See the formula below.
    borderType	Pixel extrapolation method. See BorderTypes. 

"""

corners = cv.cornerHarris(src = img, blockSize = 4, ksize = 11, k = 0, borderType = cv.BORDER_CONSTANT)
#cv.rectangle(dst1, (500, 500), (1100, 200), (255, 255, 255), 5)
cv.imwrite("3_corners.jpg", corners)


img = cv.imread('originals/3.jpg',0)
edges = cv.Canny(img, 50, 50)
cv.imwrite("3_edges.jpg", edges)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


"""
image	8-bit, single-channel binary source image. The image may be modified by the function.
lines	Output vector of lines. Each line is represented by a two-element vector (ρ,θ) . ρ is the distance from the coordinate origin (0,0) (top-left corner of the image). θ is the line rotation angle in radians ( 0∼vertical line,π/2∼horizontal line ).
rho	Distance resolution of the accumulator in pixels.
theta	Angle resolution of the accumulator in radians.
threshold	Accumulator threshold parameter. Only those lines are returned that get enough votes
srn	For the multi-scale Hough transform, it is a divisor for the distance resolution rho . The coarse accumulator distance resolution is rho and the accurate accumulator resolution is rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these parameters should be positive.
stn	For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
min_theta	For standard and multi-scale Hough transform, minimum angle to check for lines. Must fall between 0 and max_theta.
max_theta	For standard and multi-scale Hough transform, maximum angle to check for lines. Must fall between min_theta and CV_PI. 
"""
img_mine = cv.imread('4.jpg',0)
lines = cv.HoughLinesP(image = img_mine, rho = 1, theta = 3.14/180, threshold = 100)
i = 0
for line in lines:
    rho = line[0][0]
    theta = line[0][1]
    print("rho: " + str(rho) + " theta: " + str(theta))
    a = np.cos(theta) 
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    cv.line(img_mine, (int(x0 + 1000*(-b)), int(y0 + 100*a)), (int(x0 - 1000*(-b)), int(y0 - 100*a)), (0,255,0), 2)

cv.imwrite("4_lines.jpg", img_mine)
    
    #if i < 10:
    #    print("rho: " + str(rho) + " theta: " + str(theta) + " a: " + str(a) + " b: " + str(b))
    #i += 1

