#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Generate numbers distributed as Standard Normal from Uniform distribution

import numpy as np
from numpy import random
from scipy.special import erfinv
import matplotlib.pyplot as plt
import math

N = 1000000  # Number of generated points

# Generate uniform distribution
X = random.uniform(0.0, 1.0, N)

# Histogram plot for X
num_bins = 50  # Number of bins. In this case, arbitrary number
n, bins, patches = plt.hist(X, num_bins, histtype='step')
plt.xlabel('Value')
plt.ylabel('Number of points')
# plt.savefig('T2_p2_2_unif.pdf')
plt.show()

# Standard distribution CDF is F(x) = 0.5*[1 + erf((x - mu) / (sigma * sqrt(2)))]

# Evaluate uniform values in inverse CDF to obtain Standard Distribution
Y = math.sqrt(2) * erfinv(2 * X -1)  # Inverse CDF

#print Y

# Histogram plot
num_bins = 50  # Number of bins. In this case, arbitrary number
n, bins, patches = plt.hist(Y, num_bins, histtype='step')
plt.xlabel('Value')
plt.ylabel('Number of points')
# plt.savefig('T2_p2_2_standard.pdf')
plt.show()
