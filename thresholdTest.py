import numpy as np
import cv2
import sys

impath = sys.argv[1]

image = cv2.imread(impath)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edge = cv2.Canny(image,380, 405)

binary, dst = cv2.threshold(edge,130,255,0)


#fast = cv2.FastFeatureDetector()
#kp = fast.detect(edge, None)
# kp, des = orb.compute(edge, kp)

#img2 = cv2.drawKeypoints(image,kp,color=(0,255,0), flags=0)


#contours, h = cv2.FindContours(binary,1,2)
#cv2.watershed(image,m)

contours, h = cv2.findContours(dst,1,2)


cv2.drawContours(image, contours,-1, (0,255,0))

#cv2.imshow("binary", image)

cv2.waitKey(0)
