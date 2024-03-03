import time
import numpy as np
from numpy.linalg import pinv
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# Modified from source: Dr.E Dr.Cui's private code 

# Currenly, this GPR assumes input is normalized, and output is not normalized
class GprModel:

    def __init__(self, ):
        # Fcm hyperparameters
        self.target_scaler = None
        self.optimized_ls = 1, 
        self.optimized_a = 1, 
        self.optimized_c = 1,
        # A small number to ensure non-sparse matrix is invertible
        self.alpha = 1e-10
        self.sub_train_x = None
        self.sub_train_y = None
      
        # Those variables are here to make sklearn know this model is fitted
        _estimator_type = "regressor"
        dummy_ = "dummy"

    # Kernel function, RBF + constant
    # a is 2*sigma_f^2 as a single hyperparameter
    def rbf_kernel(self, x1, x2, length_scale, a, c=1):
        """RBF kernel computation."""
        dist_sq = cdist(x1, x2)**2
        return a * np.exp(-dist_sq / (2 * length_scale**2)) + c

    # log-likelihood estimation
    def neg_log_likelihood(self, params, train_x, train_y, noise):
        """Compute the negative log likelihood for Gaussian process."""
        ls, a, c = params
        K = self.rbf_kernel(train_x, train_x, ls, a, c=c) + noise**2 * np.eye(len(train_x))
        try:
            L = np.linalg.cholesky(K)
            log_det = 2 * np.sum(np.log(np.diagonal(L)))
            inv_y = np.linalg.solve(L.T, np.linalg.solve(L, train_y))
            return 0.5 * np.dot(train_y.reshape(-1, train_y.shape[0])[0], inv_y.reshape(-1, train_y.shape[0])[0]) + 0.5 * log_det + 0.5 * len(train_x) * np.log(2 * np.pi)
        except np.linalg.LinAlgError:
            # In case the kernel matrix is not positive definite
            return np.inf
        
    def subsample_data(self, data, sample_size = 10000):
        indices = np.random.choice(data.shape[0], sample_size, replace=False)
    
        return indices

    def preprocess_data(self, train_y):
        self.target_scaler = StandardScaler()  
        scaled_train_y = self.target_scaler.fit_transform(train_y)
        return scaled_train_y
    
    # Compute the partition matrix on the training data
    def fit(self, x, y):
        #indices = self.subsample_data(x, 2000)
        self.sub_train_x = x#x[indices]
        self.sub_train_y = y#y[indices]

        self.sub_train_y = self.preprocess_data(self.sub_train_y)

        # Initial hyperparameters
        initial_ls = 1
        initial_a = 1
        initial_c = 1

        # Optimization
        result = minimize(self.neg_log_likelihood, 
                        [initial_ls, initial_a, initial_c], 
                        args=(self.sub_train_x, self.sub_train_y, 0.0), 
                        method='L-BFGS-B', 
                        bounds=[(1e-5, None), (1e-5, None), (1e-5, None)]
                        )
        # Optimized hyperparameters
        print(result.x)
        self.optimized_ls, self.optimized_a, self.optimized_c = result.x

    def predict(self, gp_x_test , y_test, normalized=False):
        if not normalized:
            gp_y_test = self.target_scaler.transform(y_test)
        else:
            gp_y_test = y_test
        # Compute the RMSE for prediction
        gp_rmse = []
        gp_means = []
        for row_index in range(0, gp_x_test.shape[0], 1000):
            # specify a GP prior over the latent noise-free function
            cov = self.rbf_kernel(gp_x_test[row_index:row_index+1000, :], gp_x_test[row_index:row_index+1000, :], self.optimized_ls, self.optimized_a, self.optimized_c)

            #  let’s make predictions with these hypers (ls a)
            K_x_xstar = self.rbf_kernel(self.sub_train_x, gp_x_test[row_index:row_index+1000, :], self.optimized_ls, self.optimized_a, self.optimized_c)
                    
            K_x_x = self.rbf_kernel(self.sub_train_x, self.sub_train_x, self.optimized_ls, self.optimized_a, self.optimized_c) + self.alpha*np.random.rand(self.sub_train_x.shape[0], self.sub_train_x.shape[0])
            K_xstar_xstar = cov

            post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x)) @ self.sub_train_y
            #post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x)) @ K_x_xstar

            # Calculate the upper and lower bound for 95% confidence interval, revert the normalization
            #lw_bd =  target_scaler.inverse_transform(post_mean - (np.sqrt(np.diag(post_cov))*2).reshape(-1, 1))
            #up_bd =  target_scaler.inverse_transform(post_mean + (np.sqrt(np.diag(post_cov))*2).reshape(-1, 1))
            
            if not normalized:
                rmse = np.sqrt(np.mean((self.target_scaler.inverse_transform(post_mean) - self.target_scaler.inverse_transform(gp_y_test[row_index:row_index+1000, :]))**2))
                gp_rmse.append(rmse)
                gp_means.append(self.target_scaler.inverse_transform(post_mean).flatten())
            else:
                rmse = np.sqrt(np.mean((post_mean - gp_y_test[row_index:row_index+1000, :])**2))
                gp_rmse.append(rmse)
                gp_means.append(post_mean.flatten())
        
        print("Root Mean Squared Error:", np.mean(gp_rmse))

        # prediction set
        return  gp_means, np.mean(gp_rmse)