import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, ExpSineSquared

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
        pass

    def predict(self, test_x):
        """
            TODO: enter your code here
        """

        ## dummy code below
        y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """

        """ Task 1 : Model selection"""

        length_scales = [1, 2, 3, 4]
        kernels = [Matern(), RationalQuadratic()]
        # kernels = [ExpSineSquared()]

        # Grid search:
        # for kernel in kernels:
        #     gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        #     gp.X_train_ = train_x
        #     gp.y_train_ = train_y
        #     gp.kernel_ = kernel
        #     for length_scale in length_scales:
        #         log_like_GS = gp.log_marginal_likelihood([length_scale])
        #         print(log_like_GS)

        # Optimization method:
        P_likelihood = []
        Hyper_params = [None] * len(kernels)
        for i, kernel in enumerate(kernels):
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-30)
            gp.fit(train_x, train_y)
            # print(gp._get_param_names())
            # print(gp.get_params())
            print("Kernel type:", kernel)
            Hyper_params[i] = gp.kernel_.theta
            print(f"Hyperparameters {i}:", Hyper_params[i])
            P_likelihood.append(gp.log_marginal_likelihood())
            print(f"kernel {i}:", P_likelihood[i])
        pass


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=",")[:1000, :]
    train_y = np.loadtxt(train_y_name, delimiter=",")[:1000]

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=",")

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    # print(prediction)


if __name__ == "__main__":
    main()
