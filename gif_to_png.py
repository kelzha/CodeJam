from PIL import Image
import glob
import os

files = glob.glob("./training_dataset/*.gif")
for imageFile in files:
	filepath,filename = os.path.split(imageFile)
	filtername,exts = os.path.splitext(filename)
	im = Image.open(imageFile)
	im.save('./training_dataset/' + filtername + '.png','PNG')