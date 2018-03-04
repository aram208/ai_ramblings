# -*- coding: utf-8 -*-

import cv2

# TODO rename to spine or rectangular detector
class ShapeDetector:
    
    def __init__(self):
        pass
    
    # detect using Ramer-Douglas-Peucker algorithm
    def detect(self, c):
        shape = "unidentified"
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.4 * perimeter, True)
        
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
 
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
 
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
 
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
 
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
 
        # return the name of the shape
        return shape