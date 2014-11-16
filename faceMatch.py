import face_rec.tools2 as eigenTools
import os
from PIL import Image
from glob import glob
import numpy as np




file_list = glob('./training_dataset_cropped2/*.png')
file_list_test = glob('./fullDataSetB_cropped2/*.png')
# open image
im = Image.open(file_list[0]).convert("L")
# get original dimensions
H,W = np.shape(im)
halfW = W/2
otherHalfW = H - halfW
# print 'shape=',(H,W)

im_number = len(file_list)
im_number_test = len(file_list_test)
# fill array with rows as image
# and columns as pixels



arrL = np.zeros([im_number,H*halfW])
arrR = np.zeros([im_number,H*otherHalfW])
idArray = np.zeros([im_number,1])
testArrayL = np.zeros([im_number_test,H*halfW])
testArrayR = np.zeros([im_number_test,H*otherHalfW])
testId = np.zeros([im_number_test,1])





for i in range(len(arrL)):
	filepath,filename = os.path.split(file_list[i])
	filtername,exts = os.path.splitext(filename)
	

	im = Image.open(file_list[i]).convert("L")
	imArray = np.asarray(im)
	imArrayLeft = imArray[:,halfW+1:]
	imArrayRight = imArray[:,:otherHalfW]
	
	idArray[i] = int(filtername.split('_')[0])

	arrL[i,:] = np.reshape(imArrayLeft,[1,H*halfW])
	arrR[i,:] = np.reshape(imArrayRight,[1,H*otherHalfW])

for i in range(len(testArrayL)):

	filepath,filename = os.path.split(file_list_test[i])
	filtername,exts = os.path.splitext(filename)
	im = Image.open(file_list_test[i]).convert("L")
	imArray = np.asarray(im)
	imArrayLeft = imArray[:,halfW+1:]
	imArrayRight = imArray[:,:otherHalfW]

	testArrayL[i] = np.reshape(imArrayLeft,[1,H*halfW])
	testArrayR[i] = np.reshape(imArrayRight,[1,H*otherHalfW])
	testId[i] =int(filtername.split('_')[0])
	# print int(filtername.split('_')[0])



facial = eigenTools.EigenTwoFacial(arrL,arrR,file_list[:len(arrL)])
scores = facial.train()

# new_score = facial.get_score(arr[5],facial.mean_image)
# print facial.recognize() + ' ' + 
# assert(np.max(new_score - scores[0]) < 10**-10)
# assert(facial.recognize(arr[0]) == 0)
numTests = 0
incorrect = 0
for i in range(len(testArrayL)):
	# new_score = facial.get_score(testFace,facial.mean_image)
	
	if idArray[facial.recognize(testArrayL[i],testArrayR[i])] !=testId[i]:
		print file_list[facial.recognize(testArrayL[i],testArrayR[i])]
		incorrect +=1 
	# print [idArray[facial.recognize(testArray[i])],testId[i] ]
	# print idArray[facial.recognize(testArray[i])]
	# print facial.recognize(testArray[i])
	numTests  += 1
print float(incorrect)/float(numTests-len(arrL))
print i