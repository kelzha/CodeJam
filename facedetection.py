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
						if k[0] < minx and k[1] > 90:
							minx = k[0]
						if k[0] > maxx and k[1] > 90:
							maxx = k[0]
		if len(i) > lengthy:
			for j in i:
				for k in j:
					if k[0] > 10 and k[1] > 10 and k[0] < image.shape[1] - 10 and k[1] < image.shape[0]-10:
						if k[1] < miny and k[1] > 100:
							miny = k[1]
						if k[1] > maxy and k[1] > 100:
							maxy = k[1]
	result = (minx, miny, maxx, maxy)
	return result

'''Contrasts the image to make it so we can see shadows using underflow, returns an image'''
def contrastImage(image, minx, miny, maxx, maxy):
	counterY = 0
	for pixels in image:
		counterX = 0
		for pixel in pixels:
			if counterX < minx or counterX > maxx or counterY < miny or counterY > maxy:
				pixel = pixel - 20
				pixel = pixel - 20
				pixel = pixel - 20
				counterX += 1
		counterY += 1
	return image

def findOuterSet(cont, miny, lengthy):
	conts = set()
	for i in cont:
		if len(i) > lengthy:
			for j in i:
				for k in j:
					if k[1] < miny + 30 and k[1] > miny - 30:
						conts.add(k[0])
	#print conts
	return conts

def mean(s):
	sum = 0
	for i in s:
		sum += i
	mean = float(sum)/float(len(s))
	return mean

def outerMostSet(s, imageFile):
	minx = imageFile.shape[1]
	maxx = 0
	for i in s:
		if i < minx:
			minx = i
		if i > maxx:
			maxx = i
	return (minx, maxx)

def cropPicture(centerX, centerY, H, W):
	rect = (centerX - W/2, centerY - H/2, centerX + W/2, centerY + H/2)
	return rect

'''Gets the final bounding rectangle of the face'''
def getRectangle(impath):
	filepath,filename = os.path.split(impath)
	filtername,exts = os.path.splitext(filename)

	'''Reads the first image and makes a preliminary bounding box'''
	image = cv2.imread(impath, 0)
	#equ = cv2.equalizeHist(image)
	'''cv2.imshow(filtername, image)
	cv2.waitKey(0)
	cv2.imshow(filtername, equ)
	cv2.waitKey(0)'''
	edge = cv2.Canny(image, 150, 150)
	binary, dst = cv2.threshold(edge,130,255,0)
	contours, h = cv2.findContours(dst,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	rect = calcminmax(image, contours, 45, 5, image.shape[1], image.shape[0], 0, 0)
	
	print image.shape[0]
	delta = float(85)/float(480) * float(image.shape[0])
	centerY = float(rect[1] + delta)
	m = outerMostSet(findOuterSet(contours, centerY, 15), image)
	Width = float(m[1] - m[0])
	centerX = float(m[0]) + Width/2
	Height = float(Width * 1.418)
	rect1 = cropPicture(int(centerX),int(centerY), int(Height), int(Width))
	#cv2.rectangle(image, (rect1[0], rect1[1]), (rect1[2], rect1[3]), (0,255,0))	
	
	'''Reads the image again and contrasts it for shadows'''
	#image = contrastImage(image, rect1[0], rect1[1], rect1[2], rect1[3])
	#edge = cv2.Canny(image, 250, 250)
	#binary, dst = cv2.threshold(edge,130,255,0)
	#contours, h = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#rect = calcminmax(image, contours, 60, 20, rect1[0], rect1[1], rect1[2], rect1[3])
	
	image = cv2.imread(impath)
	cv2.drawContours(image, contours,-1, (0,255,0))
	cv2.rectangle(image, (rect1[0], rect[1]), (rect1[2], rect1[3]), (0, 0, 255))
	cv2.rectangle(image, (int(centerX)-10, int(centerY)-10), (int(centerX)+10, int(centerY)+10), (255,0,0))
	cv2.rectangle(image, (rect1[0], rect1[1]), (rect1[2], rect1[3]), (0,255,0))
	cv2.imshow(filtername, image)
	cv2.waitKey(0)
	return rect1

'''
files = glob.glob("./participantdataset/*/*")
--- 	
ARList = [0] * len(files)
CenterX = [0] * len(files)
CenterY = [0] * len(files)
Width = [0] * len(files)
Height = [0] * len(files)
counter = 0
rectList = [(0, 0, 0, 0)] * len(files)
for imageFile in files:
	rect = getRectangle(imageFile)
	print rect
	crop.cropp(imageFile, rect)
 	width = rect[2] - rect[0]
 	height = rect[3] - rect[1]
 	Width[counter] = width
 	print Width[counter]
 	Height[counter] = height
 	print Height[counter]
	CenterX[counter] = rect[0] + width/2
 	CenterY[counter] = rect[1] + height/2
 	ARList[counter] = float(width)/float(height)
 	print ARList[counter]
 	counter += 1'''

'''
AR = aspectRatio.getAverage(ARList)
width = aspectRatio.getAverage(Width)
height = aspectRatio.getAverage(Height)

counter = 0
for imageFile in files:
 	crop.cropp(imageFile, )
 	counter = counter + 1'''
