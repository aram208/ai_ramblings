# -*- coding: utf-8 -*-

"""
docs: https://docs.opencv.org/3.4.1/da/d5c/tutorial_canny_detector.html

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('3_4-11-0.jpg.jpg',0)
edges = cv.Canny(img, 50, 200)

cv.imwrite("harris-canny.jpg", edges)

"""
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
"""