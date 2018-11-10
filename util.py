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

if __name__ == '__main__':
	P = read_tensor('data/test.tns')
	print (P)