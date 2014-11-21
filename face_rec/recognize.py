import numpy as np
from pre_process import FishTrainer
from utils import make_array
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def project_face(centered_face,eigenfaces):
	"""Projects the novel centered face in the eigenface space.
	This essentially returns its SCORE. 
	"""
	return np.dot(centered_face,eigenfaces)

# class BaseRecon(object):
# 	"""Base model to predict ID of faces after training"""
# 	def __init__(self,array_files):
# 		pass

# 	def load(self,files):
# 		for a_file in files
# 		load(files)
def euclidian_diff(trained_scores,novel_score):
	difference = trained_scores - novel_score
	return np.linalg.norm(difference,axis = 1)

def mahcos_diff(trained_scores,novel_score,eigenvalues):
	# print "hi"
	sigma = np.sqrt(eigenvalues)
	m = trained_scores/sigma
	n = novel_score/sigma

	return -1.0 * m.dot(n) / (np.linalg.norm(m)*np.linalg.norm(n))

def mah_euclid_diff(trained_scores,novel_score,eigenvalues):
	sigma = np.sqrt(eigenvalues)
	m = trained_scores/sigma
	n = novel_score/sigma

	print sigma
	print m
	print n
	return np.linalg.norm(m-n,axis=0)

def soft_diff(trained_scores,novel_score,eigenvalues):
	lam = eigenvalues**0.2
	diff = np.empty()

class EigenRecon(object):

	def __init__(self,folder):
		self.eigenfaces = np.load(folder+'eigenfaces.npy')
		self.labels = np.load(folder+'labels.npy')
		self.mean_image = np.load(folder+'mean_image.npy')
		self.scores = np.load(folder+'scores.npy')
		self.eigenvalues = np.load(folder+'eigenvalues.npy')

	def recognize(self,novel_vector):
		novel_score = project_face(novel_vector-self.mean_image,
								self.eigenfaces)

		#Euclidian difference.
		diff = euclidian_diff(self.scores,novel_score)
		# diff = mahcos_diff(self.scores,novel_score,self.eigenvalues)
		# diff = mah_euclid_diff(self.scores,novel_score,self.eigenvalues)

		# best_fit = diff.argmin()
		# return (self.labels[best_fit], diff.min())
		plt.imshow(self.mean_image.reshape(226,160),cmap=cm.Greys_r)
		plt.show()
		for eigenface in self.eigenfaces[:10]:
			plt.imshow(self.mean_image.reshape(226,160),cmap=cm.Greys_r)
			plt.show()
		best_fits = diff.argsort()[:10]
		return (np.array(self.labels)[best_fits],diff[best_fits])


############################################################################
#LEGACY
class EigenfaceRecon(object):
	"""Used to predict ID of faces"""
	def __init__(self,folder):
		# folder = 'bin/eigenfaces/'
		print folder+'eigenfaces.npy'
		self.eigenfaces = np.load(folder+'eigenfaces.npy')
		self.labels = np.load(folder+'labels.npy')
		self.mean_image = np.load(folder+'mean_image.npy')
		self.scores = np.load(folder+'scores.npy')
		# for eigenface in self.eigenfaces[:5]:
			# plt.imshow(self.eigenfaces.reshape(226,160),cmap=cm.Greys_r)
			# plt.show()

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
								self.eigenfaces.T)

		# #Euclidian difference.
		diff = euclidian_diff(self.scores,novel_score)

		# best_fit = diff.argmin()
		# # print "I AM IN THE WRONG DAMN FN"
		# return (int(self.labels[best_fit]), diff.min())
		# plt.imshow(self.mean_image.reshape(226,160),cmap=cm.Greys_r)
		# plt.show()
		# for eigenface in self.eigenfaces[:10]:
		# 	plt.imshow(eigenface.reshape(226,160),cmap=cm.Greys_r)
		# 	plt.show()
		half = int(len(diff)/2)
		best_fits = diff.argsort()[:half]
		# print self.labels[best_fits]
		d = {}
		for a_diff,a_label in zip(diff[best_fits],self.labels[best_fits]):
			try:
				d[a_label][0] += a_diff
				d[a_label][1] += 1
			except KeyError:
				d[a_label] = [a_diff,1]

		# print d
		import operator
		for keys in d:
			d[keys] = d[keys][0]/d[keys][1]
			# print keys
			# print d[keys]

		sorted_d = sorted(d.items(), key=operator.itemgetter(1))
		for x in sorted_d:
			print x
		# print sorted_d

		return (np.array(self.labels)[best_fits],diff[best_fits])



class FishRecon(object):
	def __init__(self,folder):
		self.mean = np.load(folder+'mean.npy')
		self.scalings = np.load(folder+'scalings.npy')
		self.coeffs = np.load(folder+'coef.npy')
		self.intercept = np.load(folder+'intercept.npy')
		self.classes = np.load(folder+'classes.npy')

# fish=FishTrainer(train,labels)
# self.mean,self.scalings,self.coeffs,self.intercept,self.classes = fish.data()

	def recognize(self,novel_vector):
		X = np.dot(novel_vector - self.mean,self.scalings)
		res = np.dot(X, self.coeffs)+self.intercept
		best = res.argmax()

		return self.classes[best],res.max()
		

# if only_index:
# 	return int(self.classes[best])
# else:
   #  return int(self.classes[best]),float(res.max())

def main():
	recon = EigenRecon('bin/phase2_dataset_1/')

	a_picture = make_array('doug_dune/IMG_20141121_042505.bmp')

	predicted,fit = recon.recognize(a_picture)

	print predicted
	print fit






if __name__ == '__main__':
	main()
                        



		
		

                                



