from PIL import Image
import numpy as np

def make_array(filename):
	"""Makes and returns row vector from an filename to a CROPPED png image
	"""
	im = Image.open(filename).convert("L")
	arr = np.asarray(im).flatten()
	return arr

def get_IDs(trained_filenames):
	split_ID = lambda x:int(x.split('\\')[1].split('_')[0])
	IDs = [split_ID(name) for name in trained_filenames]
	return IDs

def judge(score, ID):
	if score < -20:
		return -1
	else:
		return ID
	