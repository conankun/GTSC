################################################################################
# This file contains some utility functions for general tensor spectral        #
# clustering                                                                   #
# Author: Junghyun Kim                                                         #
# Email: conankunioi@gmail.com                                                 #
################################################################################
import sys
import heapq
import numpy as np
import shift_fixpoint as sf
import tensor_cut as tc
from numpy import linalg as LA
GAMMA = 0.2

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
	return P
class algPara:
	# ALPHA - FLOAT64
	# MIN, MAX_NUM - INT64
	# PHI - FLOAT64
	self algPara(ALPHA, MIN_NUM, MAX_NUM, PHI):
		self.ALPHA = ALPHA
		self.MIN_NUM = MIN_NUM
		self.MAX_NUM = MAX_NUM
		self.PHI = PHI
class cutTree:
	"""
	n (int): the size of the clustering.
	cutValue (float): the cutValue from sweep_cut
	Ptran (float): the probability from the cluster to the other cluster
	invPtran (float): the probability from the other cluster to this cluster
	Pvol (float): volume of the nodes in the cluster (the probability of the cluster)
	subInd (Array): the nodes indices in this group
	tenInd (Array): tensor indices (from parent) in this group
	data (Array): tensor data
	left (cutTree): the left child
	right (cutTree): the right child
	"""
	def __init__(self, n, cutValue, Ptran, invPtran, Pvol, subInd=None, tenInd=None, data=None, left=None, right=None):
		self.n = n
		self.cutValue = cutValue
		self.Ptran = Ptran
		self.invPtran = invPtran
		self.Pvol = Pvol
		self.subInd = subInd
		self.tenInd = tenInd
		self.data = data
		self.left = left
		self.right = right
	def isless(anotherTree):
		return (self.n > anotherTree.n)

def generate_treeNodes(parentNode, permEv, cutPoint, cutValue, para):
	leftNode = cutTree(cutPoint, cutValue, para[0], para[1], para[2], left=None, right=None)
	rightNode = cutTree(parentNode.n - cutPoint, cutValue, para[1], para[0], 1-para[2], left=None, right=None)
	
	leftNode_subInd = np.array([parentNode.subInd[permEv[i]] for i in range(cutPoint)])
	leftNode_tenInd = np.array([permEv[i] for i in range(cutPoint)])
	leftNode.subInd = leftNode_subInd
	leftNode.tenInd = leftNode_tenInd

	rightNode_subInd = [parentNode.subInd[permEv[i]] for i in range(cutPoint, len(permEv))]
	rightNode_tenInd = [permEv[i] for i in range(cutPoint, len(permEv))]
	rightNode.subInd = rightNode_subInd
	rightNode.tenInd = leftNode_tenInd

	leftNode.data = cut_tensor(parentNode, leftNode)
	rightNode.data = cut_tensor(parentNode, rightNode)

	parentNode.subInd = np.array([], dtype='int64')
	parentNode.tenInd = np.array([], dtype='int64')
	parentNode.data = np.array([])

	if (cutPoint >= parentNode.n - cutPoint) or (para[0] == 0 and para[1] == 0 and para[2] == 0):
		parentNode.left = leftNode
		parentNode.right = rightNode
		return (leftNode, rightNode)
	else:
		parentNode.left = rightNode
		parentNode.right = leftNode
		return (rightNode, leftNode)

def mymult(output, b, Adat, xT):
	e = np.ones(b.shape[0])
	output = Adat.dot(b) + ((xT.dot(b))[0, :]).dot(e-Adat.dot(e))
	return output
def mymult2(output, b, Adat):
	n = b.shape[0]
	e = np.ones(n)
	tempOut = Adat.dot(b)
	output = tempOut + (np.sum(b) - np.sum(tempOut)) / n * e
	return output

def compute_egiv(P, al, ga):
	n = P.max()

	v = np.ones(n)
	print("Computing the super-spacey random surfer vector")
	x = sf.shift_fix(P, v, al, ga, n)
	xT = x.T
	print("Generating Transition Matrix: P[x]")
	RT = tc.tran_matrix(P, x)
	# A = MyMatrixFcn{Float64}(n, n, (output, b) -> mymult(output, b, RT, xT))
	print("Solving the eigenvector problem for P[x]")
	eigenvalue, eigenvector = LA.eigsh(A, 2, which='LM')
	return (eigenvector, RT, x)

def cut_tensor(parentNode, treeNode):
	n = parentNode.n
	P = parentNode.data
	nz = P.shape[1]
	m = P.shape[0] - 1

	tempInd = treeNode.tenInd

	tempDic = [tempInd[i] >= i for i in range(len(tempInd))]
	validInd = np.zeros(n, dtype=bool)
	for item in tempInd:
		validInd[item]=true

	flag = np.ones(nz, dtype=bool)
	for i in range(nz):
		for j in range(m):
			if not validInd[P[j][i]]:
				flag[i] = false
	# need double check
	newP = np.array([P[i][flag[i]] for i in range(m + 1)])
	for i in range(len(newP[0])):
		for j in range(m):
			newP[j][i] = tempDic[newP[j][i]]
	return newP
def refine(P, treeNode):
	allIndex = np.zeros(treeNode.n)
	for i in range(P.shape[1]):
		allIndex[P[0][i]] = 1
	permIndex = allIndex.argsort()
	if (allIndex[permIndex[0]] == 1) or (len(P.shape[1]) == 0):
		return P
	print('Process Empty Indices in sub-tensor')
	cutPoint = 0
	while allIndex[permIndex[cutPoint]] == 0:
		cutPoint += 1
	(t1, t2) = generate_treeNodes(treeNode, permIndex, cutPoint-1, 0, [0, 0, 0])
	return t2.data

def tensor_speclustering(P, algParameters):
	n = np.max(P)
	m = P.shape[0]
	# same as cutTree() in original code
	rootNode = cutTree(0, 0, 0, 0, 0, 
		np.array([], dtype='int64'), np.array([], dtype='int64'), 
		np.array([], dtype='int64'), None, None)
	rootNode.n = n
	rootNode.subInd = [ii for ii in range(rootNode.n)]
	rootNode.tenInd = [ii for ii in range(rootNode.n)]
	P = norm_tensor(P)
	rootNode.data = P

	temp1 = cutTree(0, 0, 0, 0, 0, 
		np.array([], dtype='int64'), np.array([], dtype='int64'), 
		np.array([], dtype='int64'), None, None)
	temp2 = cutTree(0, 0, 0, 0, 0, 
		np.array([], dtype='int64'), np.array([], dtype='int64'), 
		np.array([], dtype='int64'), None, None)
	h = heapq.heapify([temp1, temp2])
	dummyP = P

	for i in range(sys.maxsize):
		print('--------------- Calculating #' + str(i) + ' cut ---------------')
		if i != 0:
			hp = heapq.heappop(h)
			if hp.n <= algParameters.MIN_NUM:
				print('Completed Recursive Two-way Cut')
				heapq.heappush(h, hp)
				return (rootNode, h)
			dummyP = refine(hp.data, hp)
			if (len(dummyP.shape[1]) == 0) or np.max(dummyP[0]) <= algParameters.MIN_NUM:
				print("Tensor size smaller than MIN_NUM")
				continue
		print ('Tensor size ' + str(np.max(dummyP[0])) + ' with ' + str(len(dummyP[0])) + ' non-zeros.')
		(ev, RT, x) = compute_egiv(dummyP, algParameters.ALPHA, GAMMA)
		permEv = np.argsort(np.real(ev[:, 1]))
		print ('Generating the sweep_cut')
		(cutPoint, cutArray, cutValue, para) = sweep_cut(RT, x, permEv)
		if (cutValue > algParameters.PHI) and np.max(dummyP[0]) < algParameters.MAX_NUM:
			print('Did not cut the tensor as biased conductance (' + str(cutValue) + ') > PHI')
			continue
		if i == 0:
			(t1, t2) = generate_treeNodes(rootNode, permEv, cutPoint, cutValue, para)
			print ('-- Split ' + str(rootNode.n) + ' into ' + str(t1.n) + ' and ' + str(t2.n) + ' --')
		else:
			if hp.n > len(cutArray) + 1:
				hp = hp.right
			(t1, t2) = generate_treeNodes(hp, permEv, cutPoint, cutValue, para)
			print ('-- Split ' + str(hp.n) + ' into ' + str(t1.n) + ' and ' + str(t2.n) + ' --')
			assert(hp.n == len(cutArray) + 1)
		heapq.heappush(h, t1)
		heapq.heappush(h, t2)
	return (rootNode, h)

def print_words(wordDic, treeNode, k, sem):
	tempInd = treeNode.subInd
	print('------------------ Semantic value is ' + str(sem) + ' ------------------')
	print('Cut paramters are: Ptran = ' + str(treeNode.Ptran) + ', invPtran = '+ + str(treeNode.invPtran) + ', Pvol = ' + str(treeNode.Pvol))
	print('Number of total words are ' + str(treeNode.n))
	for i in range(min([k, len(tempInd)])):
		print(wordDic[tempInd[i]])
	print('------------------')

def asso_matrix(P, rootNode):
	m = P.shape[0] - 1
	indVec = np.zeros(rootNode.n, dtype='int64')
	print('Traverse Tree')
	traCount = trav_tree(rootNode, indVec, 0)
	print('Generating association Matrix')
	assert(traCount - 1 == np.max(indVec))
	mat = np.zeros((traCount - 1, traCount - 1))
	for i in range(P.shape[0]):
		col1 = indVec[P[0, i]]
		col2 = indVec[P[1, i]]
		if col1 == col2:
			continue
		val = P[m + 1][i]
		mat[col1, col1] += val
		mat[col2, col1] += val
	d = np.diagonal(mat)
	mat = (mat - d).dot(np.linalg.pinv(d))
	return mat

def trav_tree(treeNode, indVec, startInd):
	if (treeNode.left == None) and (treeNode.right == None):
		tempInd = treeNod.subInd
		for i in tempInd:
			indVec[i] = startInd
		return startInd + 1
	else:
		newStart = trav_tree(treeNode.left, indVec, startInd)
		return trav_tree(treeNode.right, indVec, newStart)

def print_tree_word(treeNode, semVec, startInd, wordDic, semTol_low=0, semTol_up=1, numTol_low=0, numTol_up=100):
	if (treeNode.left == None) and (treeNode.right == None):
		if (semVec[startInd] > semTol_low) and (semVec[startInd] < semTol_up) and (treeNode.n < numTol_upm) and (treeNode.n > numTol_low):
			print_words(wordDic, treeNode, 100, semVec[startInd])
		return startInd + 1
	else:
		newStart = print_tree_word(treeNode.left, semVec, startInd, wordDic, semTol_low, semTol_up, numTol_low, numTol_up)
		return print_tree_word(treeNode.right, semVec, newStart, wordDic, semTol_low, semTol_up, numTol_low, numTol_up)
if __name__ == '__main__':
	P = read_tensor('data/test.tns')
	norm_P = norm_tensor(P)
	print (norm_P)