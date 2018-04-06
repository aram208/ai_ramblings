# Import the necessary packages
import numpy as np
import cv2
import math

def translate(image, x, y):
	# Define the translation matrix and perform the translation
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Return the translated image
	return shifted

def rotate(image, angle, center = None, scale = 1.0):
	# Grab the dimensions of the image
	(h, w) = image.shape[:2]

	# If the center is None, initialize it as the center of
	# the image
	if center is None:
		center = (w // 2, h // 2)

	# Perform the rotation
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Return the rotated image
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

def hough_lines(image, backgroundImage, mask_height = 1000, threshold = 135):
    
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    lines = cv2.HoughLines(image = image, 
                           rho = 1, 
                           theta = np.pi / 180, 
                           threshold = threshold)
    
    original = cv2.imread(backgroundImage)
    original = resize(original, width=1000)
    cdst = original.copy()
    if lines is not None:
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            
            # for vertical lines
            if theta > np.pi/180*170 or theta < np.pi/180*10 :
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + mask_height * (-b)), int(y0 + mask_height * (a)))
                pt2 = (int(x0 - mask_height * (-b)), int(y0 - mask_height * (a)))
                cv2.line(cdst, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
            # for horizontal lines	
            #if( theta>CV_PI/180*80 && theta<CV_PI/180*100)
    
            
    #cv2.imwrite(outDir + "hough_lines.jpg", cdst)
    return cdst

'''    
def hough_lines_new(image, backgroundImage, outDir, threshold = 135):
    
    lines = cv2.HoughLines(image = image, 
                           rho = 1, 
                           theta = np.pi / 180, 
                           threshold = threshold)
    
    buckets = []
    
    original = resize(backgroundImage, width=1000)
    cdst = original.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            
            # for vertical lines
            if theta > np.pi/180*170 or theta < np.pi/180*10 :
                
                # we know this line is relatively vertical, but let's see if it
                # crosses any of the lines in the buckets. If yes, add to that bucket
                for bucket in buckets:
                
                # ==============
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(cdst, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                # ===============
    
            
    cv2.imwrite(outDir + "hough_lines.jpg", cdst)
'''   
    

'''
def lines_cross():
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
'''    
    
    
    
    
    