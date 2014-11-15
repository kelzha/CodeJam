from PIL import Image
import cv2
import glob
import os
import numpy

files = glob.glob("*.png")

for imageFile in files:
    filepath,filename = os.path.split(imageFile)
    name,exts = os.path.splitext(filename)

    cvImage = cv2.imread(imageFile,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    numpy.save(name,cvImage)
