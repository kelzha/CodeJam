import face_rec.tools as eigenTools
import os
from PIL import Image
from glob import glob
import numpy as np

file_list = glob('./training_dataset_cropped/*.png')
file_list2 = glob('./fullDataSetB/*.png')
# open image
im = Image.open(file_list[0]).convert("L")

# get original dimensions
H,W = np.shape(im)

# print 'shape=',(H,W)

im_number = len(file_list)
im_number2 = len(file_list2)

# fill array with rows as image
# and columns as pixels
arr = np.zeros([im_number,H*W])
idArray = np.zeros([im_number,1])
testArray = np.zeros([im_number2,H*W])
testId = np.zeros([im_number2,1])
fileArray = []
fileTestArray = []

testCounter =  0

for i in range(im_number):
	filepath,filename = os.path.split(file_list[i])
	filtername,exts = os.path.splitext(filename)
	fileArray.append(filename)
	

	im = Image.open(file_list[i]).convert("L")
	imArray = np.asarray(im)

	
	idArray[i] = int(filtername.split('_')[0])
	arr[i,:] = np.reshape(np.asarray(im),[1,H*W])

for i in range(im_number2):	
	filepath, filename = os.path.split(file_list2[i])
	filtername,exts = os.path.splitext(filename)
	fileTestArray.append(filename)

	m = Image.open(file_list2[i]).convert("L")
	imArray = np.asarray(im)

	testArray[i] = np.reshape(np.asarray(im),[1,H*W])
	testId[i] = int(filtername.split('_')[0])
	# print int(filtername.split('_')[0])

print len(arr)
facial = eigenTools.EigenFacial(arr,file_list)
scores = facial.train()

# new_score = facial.get_score(arr[5],facial.mean_image)
# print facial.recognize() + ' ' + 
# assert(np.max(new_score - scores[0]) < 10**-10)
# assert(facial.recognize(arr[0]) == 0)
for i in range(len(testArray)):
	# new_score = facial.get_score(testFace,facial.mean_image)
	if idArray[facial.recognize(testArray[i])] != testId[i]:
		print fileTestArray[i]
