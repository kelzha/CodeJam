import face_rec.tools as eigenTools
import os
from PIL import Image
from glob import glob
import numpy as np




file_list = glob('./training_dataset_cropped/*.png')

# open image
im = Image.open(file_list[0]).convert("L")
# get original dimensions
H,W = np.shape(im)
# print 'shape=',(H,W)

im_number = len(file_list)
# fill array with rows as image
# and columns as pixels
arr = np.zeros([im_number,H*W])
idArray = np.zeros([im_number,1])
testArray = np.zeros([im_number -99,H*W])
testId = np.zeros([im_number -99,1])

testCounter =  0
print file_list[0]

for i in range(im_number):
	filepath,filename = os.path.split(file_list[i])
	filtername,exts = os.path.splitext(filename)
	

	im = Image.open(file_list[i]).convert("L")
	imArray = np.asarray(im)

	
	idArray[i] = int(filtername.split('_')[0])
	arr[i,:] = np.reshape(np.asarray(im),[1,H*W])

	testArray[testCounter] = np.reshape(np.asarray(im),[1,H*W])
	testId[testCounter] =int(filtername.split('_')[0])
	# print int(filtername.split('_')[0])
	testCounter+=1

print len(arr)
facial = eigenTools.EigenFacial(arr[:50],file_list[:50])
scores = facial.train()

# new_score = facial.get_score(arr[5],facial.mean_image)
# print facial.recognize() + ' ' + 
# assert(np.max(new_score - scores[0]) < 10**-10)
# assert(facial.recognize(arr[0]) == 0)
for i in range(len(testArray)):
	# new_score = facial.get_score(testFace,facial.mean_image)
	if idArray[facial.recognize(testArray[i])] !=testId[i]:
		print file_list[facial.recognize(testArray[i])] 
