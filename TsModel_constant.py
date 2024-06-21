import time
import numpy as np
from numpy.linalg import pinv
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist

# T-S fuzzy model for regression
# Modified from source: Dr.E's private code
class TsModel_constant:

    def __init__(self, number_of_rules=30, fuzzification_coefficient=2, early_end_threshold=1e-5, max_iteration=200, constant_rule=True):
        # Fcm hyperparameters
        self.FCM_Nc = number_of_rules
        self.FCM_m = fuzzification_coefficient
        self.tol = early_end_threshold
        self.max_iter = max_iteration
        # Weight that is trained and used
        self.W = None
        self.cen = None
        self.qs = []
        # Container variables
        self.N = None
        self.z = None
        self.H = None
        self.U = None
        self.constant_rule = constant_rule
        # Those variables are here to make sklearn know this model is fitted
        _estimator_type = "regressor"
        dummy_ = "dummy"

    def __predict__(self, h):
        return np.dot(h, self.W).flatten()

    def __compute_cluster__(self, x):
        self.N = x.shape[0]
        self.U = np.random.rand(self.N, self.FCM_Nc)
        self.U = self.U / np.sum(self.U, axis=1, keepdims=True)

        # Fuzzy C-Means clustering
        for _ in range(self.max_iter):
            u_old = self.U.copy()
            mf = self.U ** self.FCM_m
            self.cen = np.dot(mf.T, x) / np.sum(mf, axis=0, keepdims=True).T
            dists = cdist(x, self.cen)
            tmp = np.power(dists, -2 / (self.FCM_m - 1))
            self.U = tmp / np.sum(tmp, axis=1, keepdims=True)
            q = np.sum((self.U ** self.FCM_m) * (dists ** 2))
            self.qs.append(q)

            # Check stoping
            if np.linalg.norm(self.U - u_old) < self.tol:
                break

    def __compute_partition_matrix__(self, x):
        # 生成模糊规则矩阵 H
        if self.constant_rule:
            z = np.hstack((np.ones((self.N, 1)), np.zeros((self.N, x.shape[1])))) 
        else:
            z = np.hstack((np.ones((self.N, 1)), x))
        self.H = np.zeros((self.N, self.FCM_Nc * (x.shape[1] + 1)))
        for j in range(self.FCM_Nc):
            uj = np.tile(self.U[:, j], (x.shape[1] + 1, 1)).T
            self.H[:, (j * (x.shape[1] + 1)):((j + 1) * (x.shape[1] + 1))] = uj * z

    # Compute the partition matrix on the training data
    def fit(self, x, y, alertanative_loss_function = None):
        start_time = time.time()
        self.__compute_cluster__(x)
        # 模型训练 TODO find if this can be optimized
        self.__compute_partition_matrix__(x)

        print(self.H.shape)
        print(self.H)
        print(self.U.shape)
        print(self.U)

        # if alertanative_loss_function is not None:
        #     # TODO remove this out for better flexibility
        #     self.W = alertanative_loss_function(x, y, self.H)
        # else:
        #     # 计算权重 W = (F*F.T)^-1*F.T*y = a
        self.W = pinv(np.dot(self.U.T, self.U)).dot(self.U.T).dot(y)
        time_used = time.time() - start_time
        # 在训练集上进行预测
        y_hat_train = self.__predict__(self.U)
        mse_train = mean_squared_error(y, y_hat_train)
        rmse_train = sqrt(mean_squared_error(y, y_hat_train))
        print("FCM training RMSE:", rmse_train)

        return mse_train, rmse_train, time_used

    def predict(self, x):
        if self.W is None:
            print('Model not trained')
        # prediction
        n_prediction = x.shape[0]
        z_prediction = np.hstack((np.ones((n_prediction, 1)), x))
        h_prediction = np.zeros((n_prediction, self.FCM_Nc * (x.shape[1] + 1)))
        dists_prediction = cdist(x, self.cen)
        tmp_prediction = np.power(dists_prediction, -2 / (self.FCM_m - 1))
        u_prediction = tmp_prediction / np.sum(tmp_prediction, axis=1, keepdims=True)

        if self.constant_rule:
            # Only use a_0i
            z_prediction = np.hstack((np.ones((n_prediction, 1)), np.zeros((n_prediction, x.shape[1])))) 
        # 生成模糊规则矩阵 H_test
        for j in range(self.FCM_Nc):
            uj_prediction = np.tile(u_prediction[:, j], (x.shape[1] + 1, 1)).T
            h_prediction[:, (j * (x.shape[1] + 1)):((j + 1) * (x.shape[1] + 1))] = uj_prediction * z_prediction

        # prediction set
        return self.__predict__(u_prediction)
    
    def evaluate(self, x, y):
        y_hat = self.predict(x)
        mse = mean_squared_error(y, y_hat)
        rmse = sqrt(mean_squared_error(y, y_hat))
        print("TS-model RMSE:", rmse)
        return mse, rmse
