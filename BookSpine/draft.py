# -*- coding: utf-8 -*-

import cv2
import numpy as np
import imutils
from collections import defaultdict

'''
class Bucket():

    def __init__(self):
        super(self).__init__()
        self.lines = []
'''

def edgedetect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    # Some values seem to go above 255. However RGB channels has to be within 0-255
    sobel[sobel > 255] = 255;

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    print(A)
    # make sure we are not dealing with a singular matrix
    if not (A[0][0] * A[1][1] - A[0][1]*A[1][0] == 0):
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        # TODO: make sure to return only those intersection points which fall 
        # within the image boundaries around the center
        return [[x0, y0]]
    else:
        return [[0, 0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))
    return intersections

def segmented_intersections_v2(lines):
    
    intersections = []
    for i in range(0, len(lines)):
        line1 = lines[i]
        for j in range(i + 1, len(lines)):
            line2 = lines[j]
            intersections.append(intersection(line1[0], line2[0]))
    
    return intersections
            
        

'''
cv::Point2f start, end;
double length = cv::norm(end - start);
# Finds the intersection of two lines, or returns false.
# The lines are defined by (o1, p1) and (o2, p2).
def intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r)
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}
'''


filename = "images/8.jpg"
image = cv2.imread(filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
res = imutils.resize(gray, width = 1000)

mask = np.zeros(res.shape[:2], dtype = 'uint8')
(cX, cY) = (res.shape[1] // 2, res.shape[0] // 2)
hdelta = res.shape[0] // 3 // 2 # half of the third of the height
cv2.rectangle(mask, (0, cY - hdelta), (999, cY + hdelta), 255, -1)

masked = cv2.bitwise_and(res, res, mask = mask)

bilat = cv2.bilateralFilter(masked, 9, 41, 41)

kernelH = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
horizontal_img = cv2.erode(bilat, kernelH, iterations=1)
horizontal_img = cv2.dilate(horizontal_img, kernelH, iterations=1)

gradX = cv2.Sobel(horizontal_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
cv2.imwrite("x-sobel.jpg", gradX)

# Hough lines =================================================================
image = cv2.imread("x-sobel.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# original threshold 200
final, lines = imutils.hough_lines(gray, backgroundImage = filename, threshold = 200)

intersections = segmented_intersections_v2(lines)

cv2.imshow("masked", final)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
For Savitzky-Golay smoothing, first need to install scipy and scipy.signal.
The code is as follows (output image also shown):
from scipy.signal import savgol_filter
...
# Use Savitzky-Golay filter to smoothen contour.
window_size = int(round(min(img.shape[0], img.shape[1]) * 0.05)) # Consider each window to be 5% of image dimensions
x = savgol_filter(contour[:,0,0], window_size * 2 + 1, 3)
y = savgol_filter(contour[:,0,1], window_size * 2 + 1, 3)

approx = np.empty((x.size, 1, 2))
approx[:,0,0] = x
approx[:,0,1] = y
approx = approx.astype(int)
contour = approx
'''
