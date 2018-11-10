################################################################################
# This file contains some utility functions for general tensor spectral        #
# clustering                                                                   #
# Author: Junghyun Kim                                                         #
# Email: conankunioi@gmail.com                                                 #
################################################################################

import numpy as np

def read_tensor(filepath):
	f = open(filepath, 'r')
	lines = f.readlines()
	dim = 0
	P = None

	line_number = 0
	for line in lines:
		elem = list(map(float, line.split()))
		current_dim = len(elem)
		if dim != 0 and dim != current_dim:
			print('Invalid Tensor: Dimension in each line is different.')
			exit()
		dim = current_dim
		if line_number == 0:
			P = np.zeros((len(lines), dim))
		for i in range(current_dim):
			P[line_number, i] = elem[i]
		line_number += 1
	f.close()
	return P.T

def norm_tensor(P):
	m, n = P.shape
	# since last column is value
	m -= 1
	print (n, m)
	tab = {}
	for k in range(n):
		tempArray = []
		for i in range(1, m):
			tempArray.append(P[i,k])
		tempKey = tuple(tempArray)
		print(tempKey)
		if tempKey in tab:
			tab[tempKey] += P[m][k]
		else:
			tab[tempKey] = P[m][k]
	for k in range(n):
		tempArray = []
		for i in range(1, m):
			tempArray.append(P[i,k])
		tempKey = tuple(tempArray)
		P[m][k] = P[m][k] / tab[tempKey]
	print (P)

if __name__ == '__main__':
	P = read_tensor('data/test.tns')
	norm_P = norm_tensor(P)
	print (norm_P)