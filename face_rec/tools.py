#TODO:
#Adjust number of eigenfaces
#

import numpy as np
import itertools

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

class EigenFacial(object):
	"""Class that can be trained with respect to an input database (figures 
		and labels) in order to identify a novel target face

	Example usage:
		facial = EigenFacial(faces,labels)
		facial.train()
		facial.predict(novel_face,name_of_method)
	"""
	def __init__(self,faces,labels):
		assert(faces.shape[0] == len(labels) or faces.shape[1] == len(labels))

		self.faces = faces; self.labels = labels
		self.num_of_faces = faces.shape[0]

	def get_score(self,faces,mean_face = None):
		"""Returns a MxN array of scores where M is the number of faces and N
		is the number of eigenfaces
		"""

		return project_face(faces,self.eigenfaces,mean_face)

	def train(self,k = None):
		"""k is the number of eigenvectors to make"""

		#Make Eigenfaces
		self.eigenfaces,self.mean_image,self.centered_faces = \
			make_eigenfaces(self.faces)
		self.num_of_eigenfaces = self.eigenfaces.shape[0]

		#Assign scores to labels
		self.scores = self.get_score(self.centered_faces)		
		return self.scores

	def compute_distance(self,score):

		return np.linalg.norm(self.scores - score,axis = 1)

	def recognize(self,novel_face,name_of_method='euler'):
		
		novel_score = self.get_score(novel_face,self.mean_image)

		diff = np.linalg.norm(self.scores - novel_score,axis = 1)
		best_fit = diff.argmin()

		return best_fit
		
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











