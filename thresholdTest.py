import numpy as np
import cv2
import sys
from PIL import Image
import glob
import crop
import aspectRatio
import os
'''THIS WILL NOT WORK IF THE PERSON'S FACE IS WITHIN 20 PIXELS OF THE SIDE'''

def count(i, delta, image):
	count = 0
	for j in i:
		for k in j:
			if k[0] > delta and k[0] < image.shape[1] - delta:
				count += 1
	return count

'''Calculates the coordinates of a rectangle surrounding contours, returns a tuple'''
def calcminmax(image, cont, lengthx, lengthy, minx, miny, maxx, maxy):
	delta = 21
	for i in cont:
		if count(i, delta, image) > lengthx:
			for j in i:
				for k in j:
					if k[0] > 10 and k[1] > delta and k[0] < image.shape[1] - delta and k[1] < image.shape[0]-10:
						if k[0] < minx:
							minx = k[0]
						if k[0] > maxx:
							maxx = k[0]
		if len(i) > lengthy:
			for j in i:
				for k in j:
					if k[0] > 10 and k[1] > 10 and k[0] < image.shape[1] - 10 and k[1] < image.shape[0]-10:
						if k[1] < miny:
							miny = k[1]
						if k[1] > maxy:
							maxy = k[1]
	result = (minx, miny, maxx, maxy)
	return result

'''Contrasts the image to make it so we can see shadows using underflow, returns an image'''
def contrastImage(image, minx, miny, maxx, maxy):
	for pixels in image:
		for pixel in pixels:
			if pixel[0] < minx or pixel[0] > maxx or pixel[0] < miny or pixel[0] > maxy:
				pixel[0] = pixel[0] - 20
				pixel[1] = pixel[1] - 20
				pixel[2] = pixel[2] - 20
	return image

'''Gets the final bounding rectangle of the face'''
def getRectangle(impath):
	filepath,filename = os.path.split(impath)
	filtername,exts = os.path.splitext(filename)

	'''Reads the first image and makes a preliminary bounding box'''
	image = cv2.imread(impath)
	edge = cv2.Canny(image, 300, 300)
	binary, dst = cv2.threshold(edge,130,255,0)
	contours, h = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	rect1 = calcminmax(image, contours, 60, 60, 1000000, 1000000, -1, -1)


	#cv2.rectangle(image, (rect1[0], rect1[1]), (rect1[2], rect1[3]), (0,255,0))	
	
	'''Reads the image again and contrasts it for shadows'''
	image = contrastImage(image, rect1[0], rect1[1], rect1[2], rect1[3])
	edge = cv2.Canny(image, 250, 250)
	binary, dst = cv2.threshold(edge,130,255,0)
	contours, h = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	rect = calcminmax(image, contours, 60, 20, rect1[0], rect1[1], rect1[2], rect1[3])
	print filtername
	
	'''image = cv2.imread(impath)
	cv2.drawContours(image, contours,-1, (0,255,0))
	cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0))
	cv2.imshow(filtername, image)
	cv2.waitKey(0)'''
	return rect

# files = glob.glob("./training_dataset/" + "*.png")

# ARList = [0] * len(files)
# CenterX = [0] * len(files)
# CenterY = [0] * len(files)
# Width = [0] * len(files)
# Height = [0] * len(files)
# counter = 0

# for imageFile in files:
# 	rect = getRectangle(imageFile)
# 	width = rect[2] - rect[0]
# 	height = rect[3] - rect[1]
# 	Width[counter] = width
# 	print Width[counter]
# 	Height[counter] = height
# 	print Height[counter]
# 	CenterX[counter] = rect[0] + width/2
# 	CenterY[counter] = rect[1] + height/2
# 	ARList[counter] = float(width)/float(height)
# 	print ARList[counter]
# 	counter += 1

# AR = aspectRatio.getAverage(ARList)
# width = aspectRatio.getAverage(Width)
# height = aspectRatio.getAverage(Height)

# counter = 0
# for imageFile in files:
# 	crop.cropp(imageFile, CenterX[counter], CenterY[counter], width, height)
# 	counter = counter + 1
