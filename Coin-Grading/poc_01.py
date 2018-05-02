#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 09:57:09 2018

@author: aram
"""
'''
import cv2
from matplotlib import pyplot as plt

file69_o = "images/ASE_69/69_MS_1_o.JPG"
file70_o = "images/ASE_70/70_MS_1_o.JPG"

file69_r = "images/ASE_69/69_MS_1_r.JPG"
file70_r = "images/ASE_70/70_MS_1_r.JPG"

image69_o = cv2.imread(file69_o)
image70_o = cv2.imread(file70_o)

image69_r = cv2.imread(file69_r)
image70_r = cv2.imread(file70_r)

#image69 = cv2.cvtColor(image69, cv2.COLOR_BGR2GRAY)
#image70 = cv2.cvtColor(image70, cv2.COLOR_BGR2GRAY)
#res69 = imutils.resize(image69, width = 1000)

hist69_o = cv2.calcHist([image69_o], channels = [0], mask = None, histSize = [256], ranges=[0, 255])
hist70_o = cv2.calcHist([image70_o], channels = [0], mask = None, histSize = [256], ranges=[0, 255])

hist69_r = cv2.calcHist([image69_r], channels = [0], mask = None, histSize = [256], ranges=[0, 255])
hist70_r = cv2.calcHist([image70_r], channels = [0], mask = None, histSize = [256], ranges=[0, 255])

f = plt.figure(figsize=(25, 8))

f.add_subplot(2,2,1)
plt.title("ASE 2018 Obverse Grade: 69")
plt.ylim([0, 25000])
plt.plot(hist69_o)

f.add_subplot(2,2,2)
plt.plot(hist70_o)
plt.ylim([0, 25000])
plt.title("ASE 2018 Obverse Grade: 70")

f.add_subplot(2,2,3)
plt.title("ASE 2018 Reverse Grade: 69")
plt.ylim([0, 25000])
plt.plot(hist69_r)

f.add_subplot(2,2,4)
plt.plot(hist70_r)
plt.ylim([0, 25000])
plt.title("ASE 2018 Reverse Grade: 70")

plt.show()

histograms = {}
for i in range(1, 6):
    
    file69_o = "images/ASE_69/69_MS_" + str(i) + "_o.JPG"
    file69_r = "images/ASE_69/69_MS_" + str(i) + "_r.JPG"
    file70_o = "images/ASE_70/70_MS_" + str(i) + "_o.JPG"
    file70_r = "images/ASE_70/70_MS_" + str(i) + "_r.JPG"
    
    image69_o = cv2.imread(file69_o)
    image69_r = cv2.imread(file69_r)
    image70_o = cv2.imread(file70_o)
    image70_r = cv2.imread(file70_r)
    
    hist69_o = cv2.calcHist([image69_o], channels = [0], mask = None, histSize = [256], ranges=[0, 255])
    hist69_r = cv2.calcHist([image69_r], channels = [0], mask = None, histSize = [256], ranges=[0, 255])
    hist70_o = cv2.calcHist([image70_o], channels = [0], mask = None, histSize = [256], ranges=[0, 255])    
    hist70_r = cv2.calcHist([image70_r], channels = [0], mask = None, histSize = [256], ranges=[0, 255])
    
    histograms[file69_o] = hist69_o
    histograms[file69_r] = hist69_r
    histograms[file70_o] = hist70_o
    histograms[file70_r] = hist70_r

f = plt.figure(figsize=(25, 8))
yLim = 25000
j = 1
for i in range(1, 6):
    
    fileNames = []
    file69_o = "images/ASE_69/69_MS_" + str(i) + "_o.JPG"
    file69_r = "images/ASE_69/69_MS_" + str(i) + "_r.JPG"
    file70_o = "images/ASE_70/70_MS_" + str(i) + "_o.JPG"
    file70_r = "images/ASE_70/70_MS_" + str(i) + "_r.JPG"
        
    f.add_subplot(5,2,j)
    plt.title(file69_o)
    plt.ylim([0, yLim])
    plt.plot(histograms[file69_o])
    j += 1
    
    f.add_subplot(5,2,j)
    plt.title(file70_o)
    plt.ylim([0, yLim])
    plt.plot(histograms[file70_o])
    j += 1
    
    f.add_subplot(5,2,j)
    plt.title(file69_r)
    plt.ylim([0, yLim])
    plt.plot(histograms[file69_r])
    j += 1
    
    f.add_subplot(5,2,j)
    plt.title(file70_r)
    plt.ylim([0, yLim])
    plt.plot(histograms[file70_r])
    j += 1
    
plt.show()    


import numpy as np
import matplotlib.pyplot as plt

tabularData_g70 = {}
tabularData_g69 = {}
x = np.array([1, 2, 3])
fig = plt.figure(figsize=(8,5))

for i in range(1, 6):
    
    file69_o = "images/ASE_69/69_MS_" + str(i) + "_o.JPG"
    file69_r = "images/ASE_69/69_MS_" + str(i) + "_r.JPG"
    file70_o = "images/ASE_70/70_MS_" + str(i) + "_o.JPG"
    file70_r = "images/ASE_70/70_MS_" + str(i) + "_r.JPG"
    
    tabularData_g69[file69_o] = [np.max(histograms[file69_o][0:50]),np.max(histograms[file69_o][51:200]),np.max(histograms[file69_o][201:255])]
    tabularData_g69[file69_r] = [np.max(histograms[file69_r][0:50]),np.max(histograms[file69_r][51:200]),np.max(histograms[file69_r][201:255])]
    tabularData_g70[file70_o] = [np.max(histograms[file70_o][0:50]),np.max(histograms[file70_o][51:200]),np.max(histograms[file70_o][201:255])]
    tabularData_g70[file70_r] = [np.max(histograms[file70_r][0:50]),np.max(histograms[file70_r][51:200]),np.max(histograms[file70_r][201:255])]
    
    fig.add_subplot(121)
    plt.plot(x, tabularData_g69[file69_o], 'g')
    plt.plot(x, tabularData_g70[file70_o], 'b')
    
    fig.add_subplot(122)
    plt.plot(x, tabularData_g69[file69_r], 'g')
    plt.plot(x, tabularData_g70[file70_r], 'b')

plt.show()

# =============================================================================

import imutils

file69_o = "images/ASE_69/69_MS_1_o.JPG"
file70_o = "images/ASE_70/70_MS_1_o.JPG"

file69_r = "images/ASE_69/69_MS_1_r.JPG"
file70_r = "images/ASE_70/70_MS_1_r.JPG"

image69_o = cv2.imread(file69_o)
image70_o = cv2.imread(file70_o)
image69_r = cv2.imread(file69_r)
image70_r = cv2.imread(file70_r)

image69_o = imutils.resize(image69_o, width = 800)
image69_r = imutils.resize(image69_o, width = 800)
image70_o = imutils.resize(image70_o, width = 800)
image70_r = imutils.resize(image70_r, width = 800)

image69_o = cv2.cvtColor(image69_o, cv2.COLOR_BGR2GRAY)
image70_o = cv2.cvtColor(image70_o, cv2.COLOR_BGR2GRAY)
image69_r = cv2.cvtColor(image69_r, cv2.COLOR_BGR2GRAY)
image70_r = cv2.cvtColor(image70_r, cv2.COLOR_BGR2GRAY)


cv2.imshow('imagethreshed', image69_o)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, image69_o_thresh = cv2.threshold(image69_o, 25, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('original', image69_o)
cv2.imshow('imagethreshed', image69_o_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# =============================================================================
file69_o = "images/ASE_69/69_MS_1_o.JPG"
file70_o = "images/ASE_70/70_MS_1_o.JPG"
file69_r = "images/ASE_69/69_MS_1_r.JPG"
file70_r = "images/ASE_70/70_MS_1_r.JPG"

image69_o = cv2.imread(file69_o)
image70_o = cv2.imread(file70_o)
image69_r = cv2.imread(file69_r)
image70_r = cv2.imread(file70_r)

image69_o = imutils.resize(image69_o, width = 800)
image69_r = imutils.resize(image69_o, width = 800)
image70_o = imutils.resize(image70_o, width = 800)
image70_r = imutils.resize(image70_r, width = 800)

image69_r_hsv = cv2.cvtColor(image69_r, cv2.COLOR_BGR2HSV)
cv2.imshow('imagethreshed', image69_r_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================
import numpy as np
import imutils

file69_o = "images/ASE_69/69_MS_1_o.JPG"
file70_o = "images/ASE_70/70_MS_1_o.JPG"
file69_r = "images/ASE_69/69_MS_1_r.JPG"
file70_r = "images/ASE_70/70_MS_1_r.JPG"

image69_o = cv2.imread(file69_o)
image70_o = cv2.imread(file70_o)
image69_r = cv2.imread(file69_r)
image70_r = cv2.imread(file70_r)

image69_o = imutils.resize(image69_o, width = 600)
image69_r = imutils.resize(image69_o, width = 600)
image70_o = imutils.resize(image70_o, width = 600)
image70_r = imutils.resize(image70_r, width = 600)

(B_69_o, G_69_o, R_69_o) = cv2.split(image69_o)
(B_70_o, G_70_o, R_70_o) = cv2.split(image70_o)
(B_69_r, G_69_r, R_69_r) = cv2.split(image69_r)
(B_70_r, G_70_r, R_70_r) = cv2.split(image70_r)

cv2.imshow('B_70_o', B_70_o)
cv2.imshow('G_70_o', G_70_o)
cv2.imshow('R_70_o', R_70_o)

cv2.waitKey(0)
cv2.destroyAllWindows()

zeros = np.zeros(image69_o.shape[:2], dtype = "uint8")
cv2.imshow('B_70_o', cv2.merge([B_70_o, zeros, zeros]))
cv2.imshow('G_70_o', cv2.merge([zeros, G_70_o, zeros]))
cv2.imshow('R_70_o', cv2.merge([zeros, zeros, R_70_o]))

cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt

file69_o = "images/ASE_69/69_MS_1_r.JPG"
image69_o = cv2.imread(file69_o)
image69_o = imutils.resize(image69_o, width = 600)

boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image69_o, lower, upper)
	output = cv2.bitwise_and(image69_o, image69_o, mask = mask)
 
	# show the images
	cv2.imshow("images", np.hstack([image69_o, output]))
	cv2.waitKey(0)

cv2.imshow('image69_o', image69_o)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ================================= HSV =======================================
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt

file69_o = "images/ASE_70/70_MS_1_o.JPG"
image69_o = cv2.imread(file69_o)
image69_o = imutils.resize(image69_o)

hsv = cv2.cvtColor(image69_o,cv2.COLOR_BGR2HSV)

hist = cv2.calcHist([hsv], channels = [0], mask = None, histSize = [180, 256], ranges = [0, 180, 0, 256])

cv2.imshow('image69_o', hist)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# =============================================================================
import cv2
from matplotlib import pyplot as plt

hist_o_h = {} 
hist_o_s = {} 
hist_o_v = {}

def calcHSVHistograms(image, mask = None, h_hist_size = 50, s_hist_size = 50, v_hist_size = 128):
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    h = cv2.calcHist([hsv_image[:,:,0]], channels = [0], mask = mask, histSize = [h_hist_size], ranges=[1, 255])
    s = cv2.calcHist([hsv_image[:,:,1]], channels = [0], mask = mask, histSize = [s_hist_size], ranges=[1, 255])
    v = cv2.calcHist([hsv_image[:,:,2]], channels = [0], mask = mask, histSize = [v_hist_size], ranges=[1, 255])
    
    return h, s, v

for i in range(1, 6):
    
    file69_o = "images/ASE_69/69_MS_" + str(i) + "_o.JPG"
    file69_r = "images/ASE_69/69_MS_" + str(i) + "_r.JPG"
    file70_o = "images/ASE_70/70_MS_" + str(i) + "_o.JPG"
    file70_r = "images/ASE_70/70_MS_" + str(i) + "_r.JPG"
    
    img_69_o = cv2.imread(file69_o)
    img_69_r = cv2.imread(file69_r)
    img_70_o = cv2.imread(file70_o)
    img_70_r = cv2.imread(file70_r)
    
    hist_o_h[file69_o], hist_o_s[file69_o], hist_o_v[file69_o] = calcHSVHistograms(img_69_o)
    hist_o_h[file69_r], hist_o_s[file69_r], hist_o_v[file69_r] = calcHSVHistograms(img_69_r)
    hist_o_h[file70_o], hist_o_s[file70_o], hist_o_v[file70_o] = calcHSVHistograms(img_70_o)
    hist_o_h[file70_r], hist_o_s[file70_r], hist_o_v[file70_r] = calcHSVHistograms(img_70_r)

    
fig = plt.figure(figsize=(25,6))
for i in range(1, 6):
 
    file69_o = "images/ASE_69/69_MS_" + str(i) + "_o.JPG"
    file69_r = "images/ASE_69/69_MS_" + str(i) + "_r.JPG"
    file70_o = "images/ASE_70/70_MS_" + str(i) + "_o.JPG"
    file70_r = "images/ASE_70/70_MS_" + str(i) + "_r.JPG"
    
    fig.add_subplot(131)
    plt.title('Hue (Obverse)')
    plt.plot(hist_o_h[file69_o], 'g')
    plt.plot(hist_o_h[file70_o], 'b')
    
    fig.add_subplot(132)
    plt.text(str(i))
    plt.title('Saturation (Obverse)')
    plt.plot(hist_o_s[file69_o], 'g')
    plt.plot(hist_o_s[file70_o], 'b')
    
    fig.add_subplot(133)
    plt.title('Value (Obverse)')
    plt.plot(hist_o_v[file69_o], 'g')
    plt.plot(hist_o_v[file70_o], 'b')

plt.show()





'''    
t70_columns = ('obv_dark','obv_medium','obv_light', 'rev_dark','rev_medium','rev_light')
t70_rows = ['70_MS_%d' % x for x in range(1, 6)] 

import matplotlib.pyplot as plt
from matplotlib import six
import pandas as pd
import numpy as np

df = pd.DataFrame()
df['x'] = np.arange(0,11)
df['y'] = df['x']*2

fig = plt.figure(figsize=(8,5))

ax1 = fig.add_subplot(121)
ax1.scatter(x=df['x'],y=df['y'])

ax2 = fig.add_subplot(122)
font_size=14
bbox=[0, 0, 1, 1]
ax2.axis('off')
mpl_table = ax2.table(cellText = df.values, rowLabels = df.index, bbox=bbox, colLabels=df.columns)
mpl_table.auto_set_font_size(False)
mpl_table.set_fontsize(font_size)
'''
    