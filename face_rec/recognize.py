import numpy as np
from pre_process import FishTrainer

def project_face(centered_face,eigenfaces):
	"""Projects the novel centered face in the eigenface space.
	This essentially returns its SCORE. 
	"""
	return np.dot(centered_face,eigenfaces.T)

# class BaseRecon(object):
# 	"""Base model to predict ID of faces after training"""
# 	def __init__(self,array_files):
# 		pass

# 	def load(self,files):
# 		for a_file in files
# 		load(files)

class EigenfaceRecon(object):
	"""Used to predict ID of faces"""
	def __init__(self):
		folder = 'bin/eigenfaces/'
		self.eigenfaces = np.load(folder+'eigenfaces.npy')
		self.labels = np.load(folder+'labels.npy')
		self.mean_image = np.load(folder+'mean_image.npy')
		self.scores = np.load(folder+'scores.npy')

	def recognize(self,novel_vector):
		"""For now, this uses euclidian difference.
		Explanations:
		-------------
		Let "k" be the number of eigenfaces, "N" the number of trained images,
		"P" the number of pixels (H*W).

		The novel_vector is then a [1,P] row vector. The dot product between
		the novel vector and the tranposed eigenfaces [P,k] is calculated. 
		Note that the novel vector is first "centered", i.e. 
		centered_vector = novel_vector - mean_image

		The above operation returns a column vector [k,1] of scores. These scores 
		were previously calculated for the trained faces.

		The eigenface-wise difference between the novel score and the trained 
		scores is calculated. This results in a [k,N] matrix of differences. 
		For instance, the first row, first column, is the difference between
		the novel_vector and the first person vector for the first eigenface 
		score. This array is "diff" in the code below.

		The diff matrix is then normalized along its rows to obtain scores.

		Procedure:
		----------
		1. Calculates novel score for novel_vector. 
		2. Calculates difference (eigenface-wise) between novel score and set 
			of trained scores. 
		3. Use a norm method to normalize the person-wise error. The lowest
			error is the best match.
		"""
		novel_score = project_face(novel_vector-self.mean_image,
								self.eigenfaces)
		diff = self.scores - novel_score
		normalized = np.linalg.norm(diff,axis = 1)

		best_fit = normalized.argmin()

		return int(self.labels[best_fit])

class FishRecon(object):
	"""Uses Linear Discriminant Analysis on set of data, similarly to PCA.
	"""
	def __init__(self):
		self.mean = np.load('bin/fishfaces/mean.npy')
		self.scalings = np.load('bin/fishfaces/scalings.npy')
		self.coeffs = np.load('bin/fishfaces/coef.npy')
		self.intercept = np.load('bin/fishfaces/intercept.npy')
		self.classes = np.load('bin/fishfaces/classes.npy')

		# fish=FishTrainer(train,labels)
		# self.mean,self.scalings,self.coeffs,self.intercept,self.classes = fish.data()

	def recognize(self,novel_vector,only_index = True):
	    X = np.dot(novel_vector - self.mean,self.scalings)
	    res = np.dot(X, self.coeffs)+self.intercept
	    best = res.argmax()
	    if only_index:
	    	return int(self.classes[best])
	    else:
		    return int(self.classes[best]),float(res.max())
                                


		
		

                                



