import numpy as np

def getAR(H,W):
	"Take numpy array of Height and Width values for all DB entries and compute optimal AR"

	listAR = H/W

	return np.mean(listAR)


