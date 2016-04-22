#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Generate numbers distributed as Standard Normal from Uniform distribution

import numpy as np
from numpy import random
from numpy import linalg
from scipy.special import erfinv
import matplotlib.pyplot as plt
import math

# Create vector to generate covariance matrix elements
X = np.arange(-5., 5.1, 0.1)
X[50] = 0.0  # There is a problem in the central value; not exactly 0.

# Create and populate covariance matrix
sigma = np.zeros((np.shape(X)[0], np.shape(X)[0]))
for i in np.arange(0, np.shape(X)[0]):
	for j in np.arange(0, np.shape(X)[0]):
		sigma[i, j] = np.exp(-0.5*(X[i] - X[j])**2)

# Check if it is a positive definite matrix
det_sigma = linalg.det(sigma)
# print(det_sigma)  # It is not

# To 'repair' the matrix, eigenvalues are needed
eigenvalues = linalg.eigvalsh(sigma)
print(eigenvalues.min())  # -3.04355621e-15

# Fix matrix with Sigma + epsilon*1
# with epsilon = 2.843557e-15
# A value for just get positive each eigenvalue

epsilon = 2.843557e-15
new_sigma = sigma + epsilon * np.identity(np.shape(X)[0])

# Test new matrix
new_eigenvalues = linalg.eigvalsh(new_sigma)
print(new_eigenvalues.min())  # Now it is ok

# Obtain Cholesky decomposition for the new sigma matrix
chol_sigma = linalg.cholesky(new_sigma) 

# Create a vector of multivariate normal (0, 1) samples
stand_normal = random.multivariate_normal(np.zeros(np.shape(X)[0]), np.identity(np.shape(X)[0]))
#print(stand_normal)

# Histogram plot for multivariate normal (0, 1) samples
num_bins = 50  # Number of bins. In this case, arbitrary number
n, bins, patches = plt.hist(stand_normal, num_bins, histtype='step')
plt.xlabel('Value')
plt.ylabel('Number of points')
# plt.savefig('T2_p2_4_standard.pdf')
plt.show()

# Obtain desired distribution: multivariate normal (0, Sigma)
final_normal = np.dot(chol_sigma, stand_normal)
# print(np.shape(final_normal))

# Histogram plot for multivariate normal (0, Sigma)
num_bins = 50  # Number of bins. In this case, arbitrary number
n, bins, patches = plt.hist(final_normal, num_bins, histtype='step')
plt.xlabel('Value')
plt.ylabel('Number of points')
# plt.savefig('T2_p2_4_exp.pdf')
plt.show()


