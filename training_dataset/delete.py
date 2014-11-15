from PIL import Image
import glob
import os

files = glob.glob("*.jpg")

for imageFile in files:
	filepath,filename = os.path.split(imageFile)
	filterame,exts = os.path.splitext(filename)
	os.remove(imageFile)

