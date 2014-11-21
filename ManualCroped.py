import numpy as np
import cv2
import sys
from PIL import Image
import glob
import crop
import aspectRatio
import os

files = glob.glob("./Manual Crop/*")
for imageFile in files:
	im = Image.open(imageFile)
	rect = (0, 0, im.size[0], im.size[1])
	crop.cropp(imageFile,rect)