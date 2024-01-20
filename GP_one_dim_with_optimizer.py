# import math
# import os
# import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pandas as pd
# import torch
# from scipy import optimize
# from scipy.spatial import distance_matrix
# from d2l import torch as d2l


# RBF function
def rbf_kernel(x1, x2, length_scale, a):
    """RBF kernel computation."""
    dist_sq = np.subtract.outer(x1, x2)**2
    return a * np.exp(-dist_sq / (2 * length_scale**2))

# log-likelihood estimation
def neg_log_likelihood(params, train_x, train_y, noise):
    """Compute the negative log likelihood for Gaussian process."""
    ls, a = params
    K = rbf_kernel(train_x, train_x, ls, a) + noise**2 * np.eye(len(train_x))
    try:
        L = np.linalg.cholesky(K)
        log_det = 2 * np.sum(np.log(np.diagonal(L)))
        inv_y = np.linalg.solve(L.T, np.linalg.solve(L, train_y))
        return 0.5 * np.dot(train_y, inv_y) + 0.5 * log_det + 0.5 * len(train_x) * np.log(2 * np.pi)
    except np.linalg.LinAlgError:
        # In case the kernel matrix is not positive definite
        return np.inf

# define data
def data_maker1(x, sig):
    return np.sin(x) + 0.5 * np.sin(4 * x) + np.random.randn(x.shape[0]) * sig

sig = 0
train_x, test_x = np.linspace(0, 5, 10), np.linspace(0, 5, 500)
train_y, test_y = data_maker1(train_x, sig=sig), data_maker1(test_x, sig=0.)

print(train_x.shape)

# Initial hyperparameters
initial_ls = 0.2
initial_a = 1

# Optimization
result = minimize(neg_log_likelihood, [initial_ls, initial_a], args=(train_x, train_y, 0.0), 
                  method='L-BFGS-B', bounds=[(1e-5, None), (1e-5, None)])


# Optimized hyperparameters
optimized_ls, optimized_a = 2, 1#result.x

# specify a GP prior over the latent noise-free function

# mean = np.zeros(test_x.shape[0])
cov = rbf_kernel(test_x, test_x, optimized_ls, optimized_a)

#prior_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=5)
#plt.plot(test_x, prior_samples.T, color='black', alpha=0.5)

#  let’s make predictions with these hypers (ls a)
K_x_xstar = rbf_kernel(train_x, test_x, optimized_ls, optimized_a)
K_x_x = rbf_kernel(train_x, train_x, optimized_ls, optimized_a)
K_xstar_xstar = cov
# K_xstar_xstar = rbf_kernel(test_x, test_x, ls, a)

post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x)) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x)) @ K_x_xstar

'''post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x + \
                sig ** 2 * np.eye(train_x.shape[0]))) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x + \
                sig ** 2 * np.eye(train_x.shape[0]))) @ K_x_xstar'''

lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

plt.scatter(train_x, train_y)
plt.plot(test_x, test_y, linewidth=2.)
plt.plot(test_x, post_mean, linewidth=2.)
plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'New Data', 'Predictions', '95% Interval'])
plt.show()



# poster curves
post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
plt.scatter(train_x, train_y)
plt.plot(test_x, test_y, linewidth=2.)
plt.plot(test_x, post_mean, linewidth=2.)
plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
plt.show()

plt.scatter(train_x, train_y)
plt.plot(test_x, test_y, linewidth=2.)
plt.legend(['Observed Data', 'True Function'])
plt.show()
