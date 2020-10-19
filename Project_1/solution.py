import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, ExpSineSquared, RBF
from sklearn.kernel_approximation import Nystroem,RBFSampler
import time

THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.01


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array
        return: float
    """
    cost = (true - predicted) ** 2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(np.logical_and(predicted < true, predicted > THRESHOLD), mask)
    mask_w3 = np.logical_and(predicted < THRESHOLD, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2
    cost[mask_w3] = cost[mask_w3] * W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(predicted < true, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2

    # reward for correctly identified safe regions
    reward = W4 * np.logical_and(predicted <= THRESHOLD, true <= THRESHOLD)

    return np.mean(cost) - np.mean(reward)


"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:
    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)
It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model:
    def __init__(self):
        """
            TODO: enter your code here
        """
        self.kernel = RBF()
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9)
        pass

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        print(self.gp)
        y, sigma = self.gp.predict(test_x, return_std=True)
        
        ## dummy code below
        #y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """

        """ Task 1 : Model selection"""
        idx = np.arange(train_x.shape[0])
        np.random.shuffle(idx)
        n = 2000
        active = idx[:n]
        test_idx = idx[1000:]
        train_x_reduced = train_x[active,:]
        train_y_reduced = train_y[active]
        # feature_map_nystroem = Nystroem(gamma=.2, random_state=1, n_components=300)
        # data_transformed = feature_map_nystroem.fit_transform(train_x,train_y)
        # rbf_feature = RBFSampler(gamma=.2, random_state=1, n_components = 100)
        # feature_matrix = rbf_feature.fit_transform(train_x,train_y)
        # gram_matrix_approx = feature_matrix @ feature_matrix.T
        # print(rbf_feature.get_params())
        # length_scales = [1, 2, 3, 4]
        kernels = [RBF(1.27),Matern(length_scale=4,nu=1.5),Matern(length_scale=2.5,nu=2.5)]
        
        # kernel = RBF(1)
        # gram_matrix = kernel((train_x[:1000,:]))
        # import matplotlib.pyplot as plt
        # plt.matshow(gram_matrix)
        # plt.matshow(gram_matrix_approx)
        # plt.show()
        
        # Optimization method:
        P_likelihood = []
        Hyper_params = [None] * len(kernels)
        costs = [None] * len(kernels)
        fitted_kernels = [None] * len(kernels)
        for i, kernel in enumerate(kernels):
            print(kernel)
            # kernel = RBF(1)
            # print(kernel(train_x))
            self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1, n_restarts_optimizer=50)
            start_time = time.time()
            self.gp.fit(train_x_reduced,train_y_reduced)
            print("--- %s seconds ---" % (time.time() - start_time))
            # print(gp._get_param_names())
            # print(gp.get_params())
            print("Kernel type:", kernel)
            params = self.gp.get_params(deep=True)
            print(params)
            Hyper_params[i] = self.gp.kernel_.theta
            print(f"Hyperparameters {i}:", Hyper_params[i])
            P_likelihood.append(self.gp.log_marginal_likelihood())
            fitted_kernels[i] = self.gp.kernel_
            print("kernel:", str(self.gp.kernel_) , P_likelihood[i])
            prediction = self.predict(train_x[test_idx,:])
            costs[i] = cost_function(train_y[test_idx], prediction)

            
        print(costs)
        best_kernel = fitted_kernels[np.argmin(costs)]
        print(best_kernel)
        self.gp = GaussianProcessRegressor(kernel=best_kernel, alpha=1, n_restarts_optimizer=50)
        pass


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=",")
    train_y = np.loadtxt(train_y_name, delimiter=",")

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=",")

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)
    eval_x = np.loadtxt(train_x_name, delimiter=",")[1000:1100,:]
    eval_y = np.loadtxt(train_y_name, delimiter=",")[1000:1100]
    prediction2 = M.predict(eval_x)
    cost = cost_function(eval_y, prediction2)
    print('cost ',cost)

if __name__ == "__main__":
    main()