# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

file69_o = "images/ASE_69/69_MS_1_o.JPG"
file69_r = "images/ASE_69/69_MS_1_r.JPG"

image69_r = cv2.imread(file69_r)
image69_o = cv2.imread(file69_o)

# ============

(cX, cY) = (image69_r.shape[1] // 2, image69_r.shape[0] // 2)
    
mask_outer = np.zeros(image69_r.shape[:2], dtype="uint8")
cv2.circle(mask_outer, (cX, cY), 1000, 255, -1) 

mask_inner = np.ones(image69_r.shape[:2], dtype="uint8")
cv2.circle(mask_inner, (cX, cY), 950, 255, -1) 

ring_mask = cv2.bitwise_xor(mask_outer, mask_inner)
masked = cv2.bitwise_and(image69_r, image69_r, mask = ring_mask)

cv2.imshow("masked", masked)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ============

masked_image = imutils.apply_ring_mask(image69_r, radius_inner = 940, radius_outer = 1000)

cv2.imshow("masked", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

hist69_o = cv2.calcHist([image69_o], channels = [0], mask = ring_mask, histSize = [255], ranges=[0, 255])
hist69_r = cv2.calcHist([image69_r], channels = [0], mask = ring_mask, histSize = [255], ranges=[0, 255])

fig = plt.figure(figsize = (20, 20))

fig.add_subplot(121)
plt.plot(hist69_o)
plt.ylim([0, 5000])
plt.title("Obverse 69")

fig.add_subplot(122)
plt.plot(hist69_r)
plt.ylim([0, 5000])
plt.title("Reverse 69")

plt.show()

# =============================================================================
file69_1_r_rusty_edge = "images/ASE_69/69_MS_1_r_rust.jpg"
image = cv2.imread(file69_1_r_rusty_edge)

image_b = image[:,:,0]
image_g = image[:,:,1]
image_r = image[:,:,2]

cv2.imshow("b", image_b)
cv2.imshow("g", image_g)
cv2.imshow("r", image_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================

import cv2
from matplotlib import pyplot as plt
import imutils

rusty = ['images/ASE_69/69_MS_1_r.JPG']
clear_69 = ['images/ASE_69/69_MS_2_r.JPG', 'images/ASE_69/69_MS_3_r.JPG', 'images/ASE_69/69_MS_4_r.JPG', 'images/ASE_69/69_MS_5_r.JPG']
clear_70 = ['images/ASE_70/70_MS_1_r.JPG', 'images/ASE_70/70_MS_2_r.JPG', 'images/ASE_70/70_MS_3_r.JPG', 'images/ASE_70/70_MS_4_r.JPG', 'images/ASE_70/70_MS_5_r.JPG']

rusty_hist = {}
clear_hist_69 = {}
clear_hist_70 = {}

for filename in clear_69:
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_masked = imutils.apply_ring_mask(gray, radius_inner = 950, radius_outer = 1000)
    
    hist = cv2.calcHist([gray_masked], channels=[0], mask=None, histSize = [255], ranges=[1, 250])
    clear_hist_69[filename] = hist

for filename in clear_70:
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_masked = imutils.apply_ring_mask(gray, radius_inner = 950, radius_outer = 1000)
    
    hist = cv2.calcHist([gray_masked], channels=[0], mask=None, histSize = [255], ranges=[1, 250])
    clear_hist_70[filename] = hist
    
for filename in rusty:
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_masked = imutils.apply_ring_mask(gray, radius_inner = 950, radius_outer = 1000)
    
    hist = cv2.calcHist([gray_masked], channels=[0], mask=None, histSize = [255], ranges=[1, 250])
    rusty_hist[filename] = hist
    
for key in rusty_hist:
    plt.plot(rusty_hist[key], 'r')

for key in clear_hist_69:
    plt.plot(clear_hist_69[key], 'g')

for key in clear_hist_70:
    plt.plot(clear_hist_70[key], 'b')
    
plt.show()


