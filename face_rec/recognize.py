def project_face(centered_face,eigenfaces):
	"""Projects the novel centered face in the eigenface space.
	This essentially returns its SCORE. 
	"""
	if mean_face is None:
		return np.dot(the_face,eigenfaces.T)
	
	return np.dot(the_face - mean_face,eigenfaces.T)

class BaseRecon(object):
	"""Base model to predict ID of faces after training"""
	def __init__(self,array_files):
		

	def load(self,files):
		for a_file in files
		load(files)


class EigenfaceRecon(object):
	"""Used to predict ID of faces"""
	def __init__(self):