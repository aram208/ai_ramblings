# -*- coding: utf-8 -*-

"""
im = cv2.imread('c:/data/ph.jpg')
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
idx =0 
for cnt in contours:
    idx += 1
    x,y,w,h = cv2.boundingRect(cnt)
    roi=im[y:y+h,x:x+w]
    cv2.imwrite(str(idx) + '.jpg', roi)
    #cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
cv2.imshow('img',im)
cv2.waitKey(0)    
"""

import cv2

"""
im1 = cv2.imread('originals_multiple/cropped/resized/1.jpg')
im2 = cv2.imread('originals_multiple/cropped/resized/1.jpg')
im3 = cv2.imread('originals_multiple/cropped/resized/1.jpg')
im4 = cv2.imread('originals_multiple/cropped/resized/1.jpg')
im5 = cv2.imread('originals_multiple/cropped/resized/1.jpg')
imgray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(imgray, 220, 255, cv2.THRESH_BINARY)

#ret,thresh1 = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
#ret,thresh2 = cv2.threshold(imgray,200,255,cv2.THRESH_BINARY_INV)
#ret,thresh3 = cv2.threshold(imgray,200,255,cv2.THRESH_TRUNC)
#ret,thresh4 = cv2.threshold(imgray,200,255,cv2.THRESH_TOZERO)
#ret,thresh5 = cv2.threshold(imgray,200,255,cv2.THRESH_TOZERO_INV)


im2, contours1, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im1, contours1, -1, (0,255,0), 3)
cv2.imwrite("test_marked_m_1.jpg", im1)

im2, contours2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im2, contours2, -1, (0,255,0), 3)
cv2.imwrite("test_marked_m_2.jpg", im2)

im2, contours3, hierarchy = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im3, contours3, -1, (0,255,0), 3)
cv2.imwrite("test_marked_m_3.jpg", im3)

im2, contours4, hierarchy = cv2.findContours(thresh4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im4, contours4, -1, (0,255,0), 3)
cv2.imwrite("test_marked_m_4.jpg", im4)

im2, contours5, hierarchy = cv2.findContours(thresh5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im5, contours5, -1, (0,255,0), 3)
cv2.imwrite("test_marked_m_5.jpg", im5)

"""


def detect(gray, frame):
    ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    idx =0 
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        #roi = frame[y:y+h, x:x+w]
        #cv2.imwrite(str(idx) + '.jpg', roi)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,0),2)
    
    return frame

# get the last frame coming from the webcam
video_capture = cv2.VideoCapture(0) # 0 for internal webcam, 1 for external

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray = gray, frame = frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

