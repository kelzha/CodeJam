import numpy as np
import cv2
import sys

impath = sys.argv[1]

image = np.load(impath)

binary, dst = cv2.threshold(image,80,255,0)

#contours, h = cv2.FindContours(binary,1,2)

cv2.imshow("binary", dst)
cv2.waitKey(0)
