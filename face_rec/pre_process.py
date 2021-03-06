"""This module is merely used to pre-process the trained dataset."""

import numpy as np
import itertools
from sklearn.lda import LDA
from sklearn.decomposition.pca import PCA
import algorithms

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

class FishTrainer(object):
	"""Class that can be trained with respect to an input database
	This means calculating mean, scalings, coefficients, intercept and 
	arranging classes.

	Example usage:
		fish = FishTrainer()
		fishRecon = recognize.FishRecon(fish.data())
		"""
	def __init__(self,train,labels):
		self.lda = LDA()
		self.lda.fit(train,labels)

	def train(self):
		pass

	def data(self):
		return (self.lda.xbar_,
				self.lda.scalings_,
		        self.lda.coef_.T,
				self.lda.intercept_,
				self.lda.classes_)

	def save(self,folder_path):
		"""What needs to be saved to directly open a trained FishRecon()
		is 
		o mean
		o scalings
		o coefficients
		o intercept
		o classes

		These are all ndarrays
		"""
		np.save(folder_path+'mean',self.lda.xbar_)
		np.save(folder_path+'scalings',self.lda.scalings_)
		np.save(folder_path+'coef',self.lda.coef_.T)
		np.save(folder_path+'intercept',self.lda.intercept_)
		np.save(folder_path+'classes',self.lda.classes_)
		# return d

class EigenTrainer(object):

    def __init__(self,faces,labels):
		assert(faces.shape[0] == len(labels) or faces.shape[1] == len(labels))

		self.faces = faces; self.labels = labels
		self.num_of_faces = faces.shape[0]

    def get_score(self,faces):
		centered_faces = faces - self.mean_image
		return np.dot(centered_faces,self.eigenfaces)		

    def train(self,k = 0):
        [self.eigenvalues,self.eigenfaces,self.mean_image] = algorithms.pca(self.faces,self.labels, k)

        self.scores = self.get_score(self.faces)
        # self.get_score(self.faces)
    def save(self,folder):
		np.save(folder+'eigenfaces',self.eigenfaces)
		np.save(folder+'eigenvalues',self.eigenvalues)
		np.save(folder+'mean_image',self.mean_image)
		np.save(folder+'labels',self.labels)
		np.save(folder+'scores',self.scores)

       	



class EigenFacialTrainer(object):
	"""Class that can be trained with respect to an input database (figures 
		and labels). This means calculating eigenfaces, scores and mean_face.
	
	Example usage:
		facial = EigenFacialTrainer(faces,labels)
		facial.train()
		facial.save()
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

	def train(self,k = None, do_return = False):
		"""k is the number of eigenvectors to make"""

		#Make Eigenfaces
		self.eigenfaces,self.mean_image,self.centered_faces = \
			make_eigenfaces(self.faces)
		self.num_of_eigenfaces = self.eigenfaces.shape[0]

		#Assign scores to labels
		self.scores = self.get_score(self.centered_faces)
		if do_return:	
			return self.scores

	def save(self,folder_path):
		"""What needs to be saved to directly open a trained EigenFacial
		is 
		o eigenfaces
		o mean_image
		o labels
		o scores

		These are all ndarrays
		"""
		np.save(folder_path+'eigenfaces',self.eigenfaces)
		np.save(folder_path+'mean_image',self.mean_image)
		np.save(folder_path+'labels',self.labels)
		np.save(folder_path+'scores',self.scores)

		return ['eigenfaces','mean_image','labels','scores']

	# def recognize(self,novel_face,name_of_method='euler',return_more = False):
		
	# 	novel_score = self.get_score(novel_face,self.mean_image)

	# 	diff = np.linalg.norm(self.scores - novel_score,axis = 1)

	# 	sorted_idx = diff.argsort()
	# 	# print sorted_idx
	# 	if return_more:
	# 		return sorted_idx[:return_more],diff[sorted_idx[:return_more]]

	# 	return int(sorted_idx[0])

def pre_process(filenames):
	"""Takes care of the whole pre-processing given only the filenames of the
	cropped images. 

	1. Makes an array out of all the filenames in the dataset (this is the 
		input)
	2. Processes the filenames to contain only the person_ID's with some basic
		string splitting.
	3. Creates eigenfaces and all that jazz in the EigenFacialTrainer. 
	4. Saves the relevant arrays as .npy files. These files will be loaded when
		analyzing a new picture such that the trainer does not need to be run 
		everytime the script is called.
	"""
	import utils
	from PIL import Image

	#Open the first image in the list of files and get its shape. 
	#For the code to work, all the other images need to have the same width
	#and height. No checking is done as this would increase computation time
	#for basically no reason. In any case, an error will be shown.
	im = Image.open(filenames[0]).convert("L")
	H,W = np.shape(im)

	#arr is a M dimensional array of N dimensional vectors. The vectors are the
	#flattened images. M corresponds to the number of people (pictures)
	#While we're looping over filenames, we also process them
	arr = np.zeros([len(filenames), H * W])
	for index,filename in enumerate(filenames):
	    # im = Image.open(filename).convert("L")
	    # arr[i,:] = np.reshape(np.asarray(im),[1,H*W])
	    arr[index,:] = utils.make_array(filename)

	IDs = utils.get_IDs(filenames)

	#Train EigenFacial
	facial = EigenFacialTrainer(arr,IDs)
	facial.train()
	arr_names = facial.save('bin/eigenfaces/')

	#Train FishFacial
	fish = FishTrainer(arr,IDs)
	fish.save('bin/fishfaces/')

if __name__ == '__main__':
	"""If running python script, give folder name as argument
	"""
	import glob
	from utils import make_array,get_IDs
	from PIL import Image

	trained_names = glob.glob('doug_last_minute/*.png')
	im = Image.open(trained_names[0]).convert("L")
	H,W = np.shape(im)

	arr = np.zeros([len(trained_names), H * W])
	for index,filename in enumerate(trained_names):

	    # im = Image.open(filename).convert("L")
	    # arr[i,:] = np.reshape(np.asarray(im),[1,H*W])
		try:
			arr[index,:] = make_array(filename)
		except ValueError:
			# pass
			print filename
	
	IDs = []
	for a_name in trained_names:
		un_split = a_name.split('\\')[1].split('_')
		try:
			ID = int(un_split[0])
			# print ID
			IDs.append(un_split[0])
		except ValueError:
			ID = un_split[0]+'_'+un_split[1].split('.')[0]
			IDs.append(ID)
	# split_ID = lambda x:x.split('\\')[1].split('_')[0]
	# IDs = [split_ID(name) for name in trained_names]
	# print IDs

	# trainer = EigenFacialTrainer(arr,IDs)
	# trainer.train()
	# trainer.save('bin/phase2_eigenfacial_minsample/')

	trainer = FishTrainer(arr,IDs)
	trainer.save('bin/phase2_fish_doug_lastminute/')











