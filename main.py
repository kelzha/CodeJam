import thresholdTest as tTest
import sys
import os
import crop
import cv2
from face_rec.recognize import EigenfaceRecon,FishRecon
from face_rec.utils import make_array,judge
from PIL import Image

filepath = sys.argv[1]
im = Image.open(filepath)
filtername,exts = os.path.splitext(filepath)
filepath = filtername + '.png'
im.save(filepath)

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

recon = FishRecon()
best_idx,max_match = recon.recognize(arr)
match = judge(max_match,int(best_idx))

# print max_match

if match > -1:
	sys.stdout.write("%i\n" % match)
else:
	sys.stdout.write("Face not in Database.")
