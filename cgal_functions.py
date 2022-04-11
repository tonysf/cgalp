import numpy as np
import pylab as pl
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.stats as spst
import scipy.special as spspec
import time
import sys

##############################
# Shows support of matrix
##############################

def mat_sup(in_mat, pm):
	out_mat = np.zeros(in_mat.shape)
	index = np.where(np.abs(in_mat) >=pm)
	out_mat[index] = np.ones(in_mat.shape)[index]
	return out_mat

##############################
# Hard thresh holding operator
##############################

def mat_ht(in_mat, pm):
	out_mat = np.zeros(in_mat.shape)
	index = np.where(np.abs(in_mat) >= pm)
	out_mat[index] = in_mat[index]
	return out_mat

##############################
# Soft thresh holding operator
##############################

def mat_st(in_mat, pm):
	out_mat = np.zeros(in_mat.shape)
	index = np.where(np.abs(in_mat) >= pm)
	out_mat[index] = in_mat[index] - np.sign(in_mat[index]) * pm
	return out_mat

##############################
# mask_mat computes the mask
# index (index) and the char
# matrix (mask)
##############################

def mask_mat(in_mat, density):
	N = in_mat.size
	index_vec = np.random.choice(N, int(density*N), replace=False)
	index_vec = np.sort(index_vec)
	index = np.unravel_index(index_vec, in_mat.shape)
	mask = np.zeros(in_mat.shape)
	mask[index] = np.ones(in_mat.shape)[index]
	#for i in range(index[0].size):
	#	mask[index[0][i], index[1][i]] = 1
	return mask, index, index_vec

###############################
# mask_op applies the mask to
# the input matrix and returns
# the vector of observations
###############################

def mask_op(in_mat, rav_index):
	return np.ravel(in_mat)[rav_index]
	
###############################
# mask_trans applies the adjoint
# to an observation vector and
# returns a matrix
###############################

def mask_trans(masked, mask, index):
	out_mat = np.zeros(mask.shape)
	out_mat[index] = masked
	return out_mat

#project on the diagonal subspace with weights omega
def PV(prod, omega):
	out_mat = omega[0] * prod[0] + omega[1] * prod[1]
	return np.array([out_mat, out_mat])
	
#project on the orthogonal complement of the diagonal subspace with weights omega
def PVo(prod, omega):
	return prod - PV(prod, omega)

#############################################################################################

#####################
# GFB functions
#####################


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho + 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def l1ballproj(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


#These are my attempts to rewrite l1ballproj from Jalal's matlab code

# def l1ballproj(x, radi):
	# if radi < 0:
		# print('Radius must not be negative')
		# print('Inputted radius:')
		# print(radi)
		# sys.exit()
	# if radi == 0:
		# return 0 * x
	# n = x.size	
	# s0 = np.flip(np.sort(np.insert(np.abs(x), 0, 0)), 0)
	# s = np.cumsum(s0)
	# s = np.flip(s, 0)
	# s = s - np.multiply(s0, np.insert(np.flip(np.arange(n)+1, 0), 0, 0))
	# indices = np.where(s > radi)[0]
	# if indices.size == 0:
		# return x
	# position = np.argmax(indices)
	# try:
		# M = indices[position[-1]]
		# M = M - 1
	# except:
		# print('indices')
		# print(np.where(s > radi)[0].shape)
		# print('size')
		# print(len(np.where(s > radi)))
		# M = indices[position]
		# M = M - 1
	# t = (((s[M+1] - radi)/(s[M+1] - s[M])) * (s0[M] - s0[M+1])) + s0[M+1]
	# y = np.zeros(x.shape)
	# I = np.where(np.abs(x)>t)
	# y[I] = x[I] * np.max(1 - t/np.abs(x[I]), 0)
	# return y
	
# def l1ballproj(x, radi):
	# if radi < 0:
		# print('Radius must not be negative')
		# print('Inputted radius:')
		# print(radi)
		# sys.exit()
	# if radi == 0:
		# return 0 * x
	# n = x.size	
	# s0 = np.flip(np.sort(np.insert(np.abs(x), 0, 0)), 0)
	# s = np.cumsum(s0)
	# s = np.flip(s, 0)
	# s = s - np.multiply(s0, np.insert(np.flip(np.arange(n)+1, 0), 0, 0))
	# indices = np.where(s > radi)[0]
	# if indices.size == 0:
		# return x
	# position = np.argmax(indices)
	# try:
		# M = indices[position[-1]]
		# M = M - 1
	# except:
		# print('indices')
		# print(np.where(s > radi)[0].shape)
		# print('size')
		# print(len(np.where(s > radi)))
		# M = indices[position]
		# M = M - 1
	# t = (((s[M+1] - radi)/(s[M+1] - s[M])) * (s0[M] - s0[M+1])) + s0[M+1]
	# y = mat_st(x, t)
	# return y
	
#####################################
# Project on the nuclear ball
#####################################

def nucballproj(in_mat, pm):
	U, S, Vh = np.linalg.svd(in_mat)
	S = l1ballproj(S, pm)
	return (U.dot(np.diag(S))).dot(Vh)

#####################################
# GFB algorithm
#####################################

def gfb(prod_X, Yp_masked, mask, mask_ind, rav_mask_ind, delta, itera):
	global time
	X = prod_X.sum(axis = 0)/3.
	prod_U = np.zeros(prod_X.shape)
	zeta = 1
	#############
	for i in range(itera):
		zeta = 1
		aux = (2 * X) - prod_X[0]
		inner_arg = mask_op(aux, rav_mask_ind)
		prod_U[0] = aux + mask_trans(Yp_masked - inner_arg + mat_st(inner_arg - Yp_masked, 1), mask, mask_ind)
		prod_X[0] = prod_X[0] + zeta * (prod_U[0] - X)
		prod_U[1] = nucballproj(2 * X - prod_X[1], delta[0])
		prod_X[1] = prod_X[1] + zeta * (prod_U[1] - X)
		prod_U[2] = l1ballproj((2 * X - prod_X[2]).flatten() , delta[1]).reshape(prod_X[2].shape)
		prod_X[2] = prod_X[2] + zeta * (prod_U[2] - X)
		X = prod_X.sum(axis = 0)/3.
		#print('gfb iteration = ' + str(i))
	return X, prod_X
	
def gfb_time(prod_X, Yp_masked, mask, mask_ind, rav_mask_ind, delta, itera):
	global time
	X = prod_X.sum(axis = 0)/3.
	prod_U = np.zeros(prod_X.shape)
	zeta = 1
	#############
	start = time.clock()
	for i in range(itera):
		zeta = 1
		aux = (2 * X) - prod_X[0]
		inner_arg = mask_op(aux, rav_mask_ind)
		prod_U[0] = aux + mask_trans(Yp_masked - inner_arg + mat_st(inner_arg - Yp_masked, 1), mask, mask_ind)
		prod_X[0] = prod_X[0] + zeta * (prod_U[0] - X)
		prod_U[1] = nucballproj(2 * X - prod_X[1], delta[0])
		prod_X[1] = prod_X[1] + zeta * (prod_U[1] - X)
		prod_U[2] = l1ballproj((2 * X - prod_X[2]).flatten() , delta[1]).reshape(prod_X[2].shape)
		prod_X[2] = prod_X[2] + zeta * (prod_U[2] - X)
		X = prod_X.sum(axis = 0)/3.
	stop = time.clock()
	gfb_time = stop - start
	return gfb_time

#breg takes the product vector prod_U, the optimum value sol_obj, Yp_masked and vstar
def breg(prod_U, sol_obj, Yp_masked, rav_mask_ind, vstar):
	ip = np.zeros(prod_U.shape[0])
	for i in range(prod_U.shape[0]):
		ip[i] = np.multiply(vstar[i], prod_U[i]).sum()
	return np.linalg.norm(mask_op(prod_U[0], rav_mask_ind) - Yp_masked, ord=1) - sol_obj - (ip.sum())
	
def breg_ip(prod_U, sol_obj, Yp_masked, rav_mask_ind, vstar):
	ip = np.zeros(prod_U.shape[0])
	for i in range(prod_U.shape[0]):
		ip[i] = np.multiply(vstar[i], prod_U[i]).sum()
	return -(ip.sum())
	
def vstarf(X, prod_X):
	return X - prod_X
	
def gfb_breg(prod_X, sol_X, sol_obj, vstar, Yp_masked, mask, mask_ind, rav_mask_ind, delta, itera):
	breg_vals = np.zeros(itera)
	X = prod_X.sum(axis = 0)/3.
	prod_U = np.zeros(prod_X.shape)
	zeta = 1
	#############
	for i in range(itera):
		zeta = 1
		aux = (2 * X) - prod_X[0]
		inner_arg = mask_op(aux, rav_mask_ind)
		prod_U[0] = aux + mask_trans(Yp_masked - inner_arg + mat_st(inner_arg - Yp_masked, 1), mask, mask_ind)
		prod_X[0] = prod_X[0] + zeta * (prod_U[0] - X)
		prod_U[1] = nucballproj(2 * X - prod_X[1], delta[0])
		prod_X[1] = prod_X[1] + zeta * (prod_U[1] - X)
		prod_U[2] = l1ballproj((2 * X - prod_X[2]).flatten() , delta[1]).reshape(prod_X[2].shape)
		prod_X[2] = prod_X[2] + zeta * (prod_U[2] - X)
		X = prod_X.sum(axis = 0)/3.
		#print('gfb breg iteration = ' + str(i))
		breg_vals[i] = breg(prod_U, sol_obj, Yp_masked, rav_mask_ind, vstar)
		if i > 1:
			breg_vals[i] = min(breg_vals[i-1], breg(prod_U, sol_obj, Yp_masked, rav_mask_ind, vstar))
		else:
			breg_vals[i] = breg(prod_U, sol_obj, Yp_masked, rav_mask_ind, vstar)
	return X, breg_vals
	
def gfb_erg(prod_X, sol_X, sol_obj, vstar, Yp_masked, mask, mask_ind, rav_mask_ind, delta, itera):
	breg_vals = np.zeros(itera)
	X = prod_X.sum(axis = 0)/3.
	prod_U = np.zeros(prod_X.shape)
	zeta = 1
	#############
	for i in range(itera):
		if i == 0:
			erg_U = prod_U
		else:
			erg_U = (prod_U + (i * erg_U))/(i + 1)
		zeta = 1
		aux = (2 * X) - prod_X[0]
		inner_arg = mask_op(aux, rav_mask_ind)
		prod_U[0] = aux + mask_trans(Yp_masked - inner_arg + mat_st(inner_arg - Yp_masked, 1), mask, mask_ind)
		prod_X[0] = prod_X[0] + zeta * (prod_U[0] - X)
		prod_U[1] = nucballproj(2 * X - prod_X[1], delta[0])
		prod_X[1] = prod_X[1] + zeta * (prod_U[1] - X)
		prod_U[2] = l1ballproj((2 * X - prod_X[2]).flatten() , delta[1]).reshape(prod_X[2].shape)
		prod_X[2] = prod_X[2] + zeta * (prod_U[2] - X)
		X = prod_X.sum(axis = 0)/3.
		#print('gfb breg iteration = ' + str(i))
		breg_vals[i] = breg(erg_U, sol_obj, Yp_masked, rav_mask_ind, vstar)
		if i > 1:
			breg_vals[i] = min(breg_vals[i-1], breg(erg_U, sol_obj, Yp_masked, rav_mask_ind, vstar))
		else:
			breg_vals[i] = breg(erg_U, sol_obj, Yp_masked, rav_mask_ind, vstar)
	return X, breg_vals

#############################################################################################

######################
# CGAL functions
######################


#the gradient of g^beta where g(Ax) = |Ax-y|_1 and A is the masking operator
def g_grad(X, Yp_masked, mask, mask_ind, rav_mask_ind, beta, nucomp):
	masked_X = mask_op(X, rav_mask_ind)
	return (1. / beta) * mask_trans(masked_X - (Yp_masked + mat_st(masked_X - Yp_masked, nucomp * beta)), mask, mask_ind)

#gradient in product space
def obj_grad(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, beta, Rho, nu):
	grad = np.zeros(prod_X.shape)
	PVoMu = PVo(prod_Mu, omega)
	PVoX = PVo(prod_X, omega)
	for i in range(grad.shape[0]):
		grad[i] = (nu[i] * g_grad(prod_X[i], Yp_masked, mask, mask_ind, rav_mask_ind, beta, nu[i])) + (omega[i] * (PVoMu[i] + (Rho * PVoX[i])))
	return grad
	
def primal_var_update(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, beta, stepsize, Rho, nu, delta):
	step_direc = np.zeros(prod_X.shape)
	grad = obj_grad(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, beta, Rho, nu)
	#compute step for nuc norm
	U, S, Th = spl.svds(grad[0], 1)
	step_direc[0] = -delta[0] * U.dot(Th)
	#compute step for l1 vec norm
	ind = np.unravel_index(np.argmax(np.absolute(grad[1]), axis=None), grad[1].shape)
	sign = np.sign(grad[1][ind])
	step_direc[1][ind] = -delta[1] * sign
	return ((1 - stepsize) * prod_X) + (stepsize * step_direc)

def dual_var_update(prod_X, prod_Mu, omega, stepsize):
	return prod_Mu + stepsize * PVo(prod_X, omega)

def cgal(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, nu, delta, itera):
	Rho = 15
	for i in range(itera):
		stepsize = 1./(1+i)
		beta = 1./np.sqrt(1+i)
		prod_X = primal_var_update(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, beta, stepsize, Rho, nu, delta)
		prod_Mu = dual_var_update(prod_X, prod_Mu, omega, stepsize)
		#
		#print('CGAL iteration = ' + str(i))
	return prod_X, prod_Mu

def cgal_time(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, nu, delta, itera):
	global time
	Rho = 15
	start = time.clock()
	for i in range(itera):
		stepsize = 1./(1+i)
		beta = 1./np.sqrt(1+i)
		prod_X = primal_var_update(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, beta, stepsize, Rho, nu, delta)
		prod_Mu = dual_var_update(prod_X, prod_Mu, omega, stepsize)
	stop = time.clock()
	cgal_time = stop - start
	return cgal_time

def lagrangian(prod_X, prod_Mu, Yp_masked, rav_mask_ind, omega, nu):
	lagr_out = 0.
	PVoX = PVo(prod_X, omega)
	for i in range(prod_X.shape[0]):
		lagr_out += nu[i] * np.linalg.norm(mask_op(prod_X[i], rav_mask_ind) - Yp_masked, ord=1) + omega[i] * (np.multiply(PVoX[i], prod_Mu[i]).sum())
	return lagr_out

def cgal_lagr(prod_X, prod_Mu, sol_X, sol_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, nu, delta, itera):
	Rho = 15
	lagr_vals = np.zeros(itera)
	for i in range(itera):
		stepsize = 1./(1+i)
		beta = 1./np.sqrt(1+i)
		prod_X = primal_var_update(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, beta, stepsize, Rho, nu, delta)
		prod_Mu = dual_var_update(prod_X, prod_Mu, omega, stepsize)
		#
		lagr_vals[i] = lagrangian(prod_X, sol_Mu, Yp_masked, rav_mask_ind, omega, nu)
		#print('CGAL lagr iteration = ' + str(i))
	return prod_X, prod_Mu, lagr_vals

def cgal_erg(prod_X, prod_Mu, sol_X, sol_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, nu, delta, itera):
	Rho = 15
	lagr_vals = np.zeros(itera)
	Gamma = 0
	for i in range(itera):
		stepsize = 1./(1+i)
		if i == 0:
			erg_X = prod_X
		else:
			erg_X = ((stepsize * prod_X) + (Gamma * erg_X))/(Gamma + stepsize)
		Gamma += stepsize
		beta = 1./np.sqrt(1+i)
		prod_X = primal_var_update(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, beta, stepsize, Rho, nu, delta)
		prod_Mu = dual_var_update(prod_X, prod_Mu, omega, stepsize)
		#
		lagr_vals[i] = lagrangian(erg_X, sol_Mu, Yp_masked, rav_mask_ind, omega, nu)
		#print('CGAL lagr iteration = ' + str(i))
	lagr_vals = lagr_vals - lagr_vals[itera - 1]
	return prod_X, prod_Mu, lagr_vals
	
def cgal_erg_2(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, nu, delta, itera):
	Rho = 15
	Gamma = 0
	for i in range(itera):
		stepsize = 1./(1+i)
		if i == 0:
			erg_X = prod_X
		else:
			erg_X = ((stepsize * prod_X) + (Gamma * erg_X))/(Gamma + stepsize)
		Gamma += stepsize
		beta = 1./np.sqrt(1+i)
		prod_X = primal_var_update(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, beta, stepsize, Rho, nu, delta)
		prod_Mu = dual_var_update(prod_X, prod_Mu, omega, stepsize)
		#
		#print('CGAL lagr iteration = ' + str(i))
	return erg_X, prod_Mu

def cgal_log(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, nu, delta, itera, a, b):
	Rho = (2 ** (2 - b)) + 1
	dd = (b + 1.)/2
	for i in range(itera):
		stepsize = (np.log(i + 2) ** a)/((i + 1) ** (1 - b))
		beta = 1./((1 + i) ** (1 - dd))
		prod_X = primal_var_update(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, beta, stepsize, Rho, nu, delta)
		prod_Mu = dual_var_update(prod_X, prod_Mu, omega, stepsize)
		#
		print('CGAL iteration = ' + str(i))
	return prod_X, prod_Mu

def cgal_log_erg(prod_X, prod_Mu, sol_X, sol_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, nu, delta, itera, a, b):
	Rho = (2 ** (2 - b)) + 1
	dd = (b + 1.)/2
	lagr_vals = np.zeros(itera)
	Gamma = 0
	for i in range(itera):
		stepsize = (np.log(i + 2) ** a)/((i + 1) ** (1 - b))
		if i == 0:
			erg_X = prod_X
		else:
			erg_X = ((stepsize * prod_X) + (Gamma * erg_X))/(Gamma + stepsize)
		Gamma += stepsize
		beta = 1./((1 + i) ** (1 - dd))
		prod_X = primal_var_update(prod_X, prod_Mu, Yp_masked, mask, mask_ind, rav_mask_ind, omega, beta, stepsize, Rho, nu, delta)
		prod_Mu = dual_var_update(prod_X, prod_Mu, omega, stepsize)
		#
		lagr_vals[i] = lagrangian(erg_X, sol_Mu, Yp_masked, rav_mask_ind, omega, nu)
		print('CGAL lagr iteration = ' + str(i))
	lagr_vals = lagr_vals - lagr_vals[itera - 1]
	return prod_X, prod_Mu, lagr_vals
