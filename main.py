import thresholdTest as tTest
import sys
import crop
import cv2
from face_rec.recognize import EigenfaceRecon
from face_rec.utils import make_array

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

filepath = "edited_" + filepath

arr = make_array(filepath)

recon = EigenfaceRecon()
print recon.recognize(arr)
