import numpy as np

def getAverage(ratios):
	"Take numpy array of AR values for all DB entries and compute optimal AR"
	return np.mean(ratios)

def newWidth(AR,H):
	newW = AR * H

	return newW


