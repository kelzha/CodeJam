import numpy as np
import utils

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

		match = utils.judge(score,self.labels[best_fit])

		if match > -1:
			return "ID : %i" % match
		else:
			return "Face not in database"


