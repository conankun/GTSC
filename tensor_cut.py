import numpy as np
import scipy
from scipy.sparse import csr_matrix
def tran_matrix(P, x):
	n = np.max(P)
	m = P.shape[0]

	RT = np.zeros((int(n), int(n)))
	m = P.shape[0] - 1 # tensor dimension
	NZ = P[0].shape[0] # total number of non-zero items in the tensor
	for i in range(NZ):
		ind1 = int(P[0][i])
		ind2 = int(P[1][i])
		val = P[m][i]
		mul = 1
		for j in range(2, m):
			mul = mul * x[int(P[j][i]) - 1]
		itemVal = val * mul
		RT[ind2, ind1] += itemVal
	return csr_matrix(RT)

def sweep_cut(Px, x, permEv):
	n = len(permEv)
	per = np.argsort(permEv)
	tranS1 = np.zeros(n + 1)
	tranS2 = np.zeros(n + 1)
	volS1 = np.zeros(n + 1)
	PS1 = np.zeros(n + 1)
	PS2 = np.zeros(n + 1)
	cut = np.zeros(n - 1)

	tPx = Px.transpose()
	v = np.ones(n)
	v = csr_matrix(Px*v)
	v = v.T
	print('Calculate cuts')
	for i in range(n-1):
		ind = permEv[i]
		tempRow = tPx[:, ind]
		tempCol = Px[:, ind]
		tempRowInd = tempRow.indices
		tempColInd = tempCol.indices

		tranS1[i+1] = tranS1[i]
		tranS2[i+1] = tranS2[i]

		for j in tempRowInd:
			if per[j] > i:
				tranS1[i + 1] += x[ind] * tempRow[j] / v[ind]
			if per[j] < i:
				tranS2[i + 1] = tranS2[i + 1] - x[ind] * tempRow[j] / v[ind]

		for j in tempColInd:
			if per[j] > i:
				tranS2[i + 1] += x[j] * tempCol[j] / v[j]
			if per[j] < i:
				tranS1[i + 1] = tranS1[i + 1] - x[j] * tempCol[j] / v[j]

		PS1[i + 1] = tranS1[i + 1]
		volS1[i + 1] = volS1[i] + x[ind]
		prob1 = PS1[i + 1]/volS1[i + 1]

		PS2[i + 1] = tranS2[i + 1]
		prob2 = PS2[i + 1] / (1 - volS1[i + 1])

	cutPoint = np.argmin(cut)
	return (cutPoint, cut, cut[cutPoint], np.array([PS1[cutPoint+1], PS2[cutPoint+1], volS1[cutPoint+1]]))