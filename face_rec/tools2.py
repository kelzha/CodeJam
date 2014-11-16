#TODO:
#Adjust number of eigenfaces
#

import numpy as np
import itertools
import math
from scipy import stats

def make_eigenfaces(cropped_figs,number_of_eigenfaces = None):
    """Inputs : 
	o a Nx(H*W) array of pictures, where N is the number of faces and H*W are the
		height and width of a picture. The dimensions must be the same for all
		figures. 
	o Optional: the number of eigenfaces to return. 

	Outputs (eigenfaces,mean_image): 
	o Eigenfaces : self-explanatory
	o Mean image : self-explanatory

	Example usage:
	eigenfaces,mean_image = eigenfaces(images)
	"""
    from sklearn.decomposition.pca import PCA
	#Make sure the array is oriented properly. The number of columns should 
	#be greater than the number of rows.
    try:
        num_rows,num_cols = cropped_figs.shape
    except ValueError:
        raise AssertionError("""I need a 2-D array of image vectors. To use me
you should flatten() all your images and vstack() them in order to obtain a
[5,900] if you have 5 pictures that are 30x30""")

    if num_rows > num_cols:
        cropped_figs = cropped_figs.T

	#Find the mean image
    mean_image = np.mean(cropped_figs, axis=0)

	#Normalize all faces w.r.t to the mean image. This is also called "centering"
    centered_figs = cropped_figs - mean_image

	#PCA the centered figures. Note that the eigenvectors are already ordered
	#by eigenvalues.
    skpca = PCA()
    skpca.fit(centered_figs)

    return skpca.components_[:number_of_eigenfaces],mean_image,centered_figs

def project_face(the_face,eigenfaces,mean_face=None):
	"""Projects the face in the face space. If no mean_face is supplied, it is 
	assumed that the face to project is already centered. If not, center first
	"""
	if mean_face is None:
		return np.dot(the_face,eigenfaces.T)
	
	return np.dot(the_face - mean_face,eigenfaces.T)

class EigenTwoFacial(object):
	"""Class that can be trained with respect to an input database (figures 
		and labels) in order to identify a novel target face

	Example usage:
		facial = EigenFacial(faces,labels)
		facial.train()
		facial.predict(novel_face,name_of_method)
	"""
	def __init__(self,leftFaces,rightFaces,labels):
		assert(leftFaces.shape[0] == len(labels) or leftFaces.shape[1] == len(labels))

		self.leftFaces = leftFaces; self.labels = labels
		self.rightFaces = rightFaces;
		self.num_of_faces = leftFaces.shape[0]

	def get_score(self,facesL,facesR,mean_faceL = None,mean_faceR =None):
		"""Returns a MxN array of scores where M is the number of faces and N
		is the number of eigenfaces
		"""

		return project_face(facesL,self.eigenfacesL,mean_faceL), project_face(facesR,self.eigenfacesR,mean_faceR)


	def train(self,k = None):
		"""k is the number of eigenvectors to make"""

		#Make Eigenfaces
		self.eigenfacesL,self.mean_imageL,self.centered_facesL = \
			make_eigenfaces(self.leftFaces)
		self.eigenfacesR,self.mean_imageR,self.centered_facesR = \
			make_eigenfaces(self.rightFaces)

		self.num_of_eigenfaces = self.eigenfacesL.shape[0]

		#Assign scores to labels
		self.scoresL,self.scoresR  = self.get_score(self.centered_facesL,self.centered_facesR)
			
		

	def compute_distance(self,score):

		return np.linalg.norm(self.scores - score,axis = 1)

	def recognize(self,novel_faceL,novel_faceR,name_of_method='euler'):
		
		novel_scoreL,novel_scoreR = self.get_score(novel_faceL,novel_faceR,self.mean_imageL,self.mean_imageR)
		

		diffL = np.linalg.norm(self.scoresL - novel_scoreL,axis = 1)
		diffR = np.linalg.norm(self.scoresR - novel_scoreR,axis = 1)
	
		# diffL = (diffL- np.min(diffL))/(np.max(diffL)- np.min(diffL))
		# diffR = (diffR- np.min(diffR))/(np.max(diffR)- np.min(diffR))
		diffL = self.sigmoid(diffL)
		diffR = self.sigmoid(diffR)
		# best_fit = (diffL+diffR).argmin()
		# diffL = np.tanh(diffL)
		# diffR = np.tanh(diffR)
		best_fit = int(stats.mode((diffL+diffR).argsort()[:5])[0][0])
		print best_fit
		return best_fit
	def sigmoid(self,x):
		std = np.std(x)
		mu = np.mean(x)
		gauss = [((xin - (mu-2*std)))/(2*std) for xin in x]
		y = [(float(1)/ float(1+math.exp(-xin))) for xin in gauss]
		
		return np.asarray(y)
		
if __name__ == '__main__':
	import os
	from PIL import Image
	data_dir = "C:/Users/Selimb/Documents/yaleB/CroppedYale/CroppedYale"
	os.chdir(data_dir)

	
	def create_filenames(data_dir, view_list):
	    # loads the pictures into a list
	    # data_dir: the CroppedYale folder
	    # view_list: the views you wish to grab
	    dir_list = os.listdir(data_dir)
	    file_list = []
	    for dir in dir_list:
	        for view in view_list:
	            filename = "%s/%s_%s.png" % (dir, dir, view)
	            file_list.append(filename)
	    return(file_list)

	view_list = ['P00A+000E+00', 'P00A+005E+10' , 'P00A+005E-10' , 'P00A+010E+00']

	file_list = create_filenames(data_dir, view_list)
	# open image
	im = Image.open(file_list[0]).convert("L")
	# get original dimensions
	H,W = np.shape(im)
	# print 'shape=',(H,W)

	im_number = len(file_list)
	# fill array with rows as image
	# and columns as pixels
	arr = np.zeros([im_number,H*W])

	for i in range(im_number):
	    im = Image.open(file_list[i]).convert("L")
	    arr[i,:] = np.reshape(np.asarray(im),[1,H*W])

	facial = EigenFacial(arr[:20],file_list[:20])
	scores = facial.train()

	new_score = facial.get_score(arr[0],facial.mean_image)
	assert(np.max(new_score - scores[0]) < 10**-10)
	assert(facial.recognize(arr[0]) == 0)











