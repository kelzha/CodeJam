import numpy as np

def getAR(H,W):
	"Take numpy array of Height and Width values for all DB entries and compute optimal AR"

	listAR = W/H

	return np.mean(listAR)

def newWidth(AR,H):
	newW = AR * H

	return newW


