import thresholdTest as tTest
import sys
import crop
import cv2
import face_rec as frec

filepath = sys.argv[1]
rect = tTest.getRectangle(filepath)
# print rect

contents = open('avgHW.csv','r').read()

H,W = contents.split(',')
H = int(H)
W = int(W)
centerY = (rect[3] + rect[1])/2
centerX = (rect[2] + rect[0])/2
# print centerX
# print centerY

crop.cropp(filepath,centerX,centerY,W,H)

filepath += "_edited"

arr = frec.makeArray(filepath)

frec.eigenFacial(arr)
