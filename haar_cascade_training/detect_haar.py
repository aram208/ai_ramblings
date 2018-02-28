#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 23:09:54 2018

@author: aram
"""
import cv2

spine_cascade = cv2.CascadeClassifier('data/cascade.xml')

def detect(gray, frame):
    # detectMultiScale returns coordinates of the upper left corner and 
    # the width/height of the rectangle that has a face in it.
    # Parameters: https://docs.opencv.org/3.1.0/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    # 1.3 scaleFactor
    # 5 - amount of neighbors to retain
    spines = spine_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in spines:
        # draw the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
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