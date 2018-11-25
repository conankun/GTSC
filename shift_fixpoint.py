import numpy as np
def compute_kron(P, index, x):
	m = P.shape[0] - 1
	result = 1
	for i in range(m - 1):
		result = result * x[int(P[i + 1, index]) - 1]
	return result

def sparse_kron(P, x):
	n = x.shape[0]
	m = P.shape[0] -1
	Px = np.zeros(n)
	for ind in range(P.shape[1]):
		Px[int(P[0, ind])] += P[m, ind] * compute_kron(P, ind, x)
	return Px

def shift_fix(P, v, alpha, gamma, n):
	maxiter = 10000
	tol = 1e-6
	e = np.ones(n) / n
	# need to know x_old's size
	x_old = np.random.randint(n, size=(v.shape[0]))
	for i in range(maxiter):
		Px = sparse_kron(P, x_old)
		x_new = (alpha/(1.0 + gamma)) * Px + ((1-alpha)/(1.0 + gamma))*v + (gamma/(1.0 + gamma))*x_old
		x_new = x_new / np.sum(x_new)
		res = np.sum(np.abs(x_new - x_old))
		if res <= tol:
			return x_new
		x_old = x_new
	raise Exception('Cannot find the fix point within' + str(maxiter) + ' iterations')
