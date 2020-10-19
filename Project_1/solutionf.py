import numpy as np
import gpytorch
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import torch

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

class Model_template(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel):
        super().__init__(train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale

    @output_scale.setter
    def output_scale(self, value):
        """Set output scale."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])
        self.covar_module.outputscale = value

    @property
    def length_scale(self):
        """Get length scale."""
        ls = self.covar_module.base_kernel.kernels[0].lengthscale
        if ls is None:
            ls = torch.tensor(0.0)
        return ls

    @length_scale.setter
    def length_scale(self, value):
        """Set length scale."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])

        try:
            self.covar_module.lengthscale = value
        except RuntimeError:
            pass

        try:
            self.covar_module.base_kernel.lengthscale = value
        except RuntimeError:
            pass

        try:
            for kernel in self.covar_module.base_kernel.kernels:
                kernel.lengthscale = value
        except RuntimeError:
            pass

class Model():
    def __init__(self):

        pass

    def get_kernel(self, kernel, composition="addition"):
        base_kernel = []
        if "RBF" in kernel:
            base_kernel.append(gpytorch.kernels.RBFKernel())
        if "linear" in kernel:
            base_kernel.append(gpytorch.kernels.LinearKernel())
        if "quadratic" in kernel:
            base_kernel.append(gpytorch.kernels.PolynomialKernel(power=2))
        if "Matern-1/2" in kernel:
            base_kernel.append(gpytorch.kernels.MaternKernel(nu=1 / 2))
        if "Matern-3/2" in kernel:
            base_kernel.append(gpytorch.kernels.MaternKernel(nu=3 / 2))
        if "Matern-5/2" in kernel:
            base_kernel.append(gpytorch.kernels.MaternKernel(nu=5 / 2))
        if "Cosine" in kernel:
            base_kernel.append(gpytorch.kernels.CosineKernel())

        if composition == "addition":
            base_kernel = gpytorch.kernels.AdditiveKernel(*base_kernel)
        elif composition == "product":
            base_kernel = gpytorch.kernels.ProductKernel(*base_kernel)
        else:
            raise NotImplementedError
        kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        return kernel

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        self.fitted_model.eval()
        with torch.no_grad():
            out = self.fitted_model(test_x)
            lower, upper = out.confidence_region()
        y = out.mean
        # y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        return y

    def fit_model(self, train_x, train_y):
        X = train_x
        y = train_y
        kernels = [ "Matern-3/2", "Matern-5/2"]
        best_kernel = {}
        for kernel in kernels:
            print("Training kernel: ", kernel)
            model_temp = Model_template(X, y, self.get_kernel(kernel)).double()
            model_temp.train()
            optimizer = torch.optim.Adam([{'params': model_temp.parameters()}], lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model_temp.likelihood, model_temp)
            training_iter = 1

            losses = []
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model_temp(X)
                # Calc loss and backprop gradients
                loss = -mll(output, y) #approximation of the log marginal likelihood --> loss function for parameter optimzation: minimize it
                loss.backward() #gradient of loss function with respect to its parameters (Jacobian)

                losses.append(loss.item())
                optimizer.step()

                if not (i%25):
                      print("Iter - %d -- Loss %f"%(i, losses[-1]))

            plt.plot(losses, Linewidth = '2', label=f"{kernel}")
            best_kernel[kernel] = (loss.item(), model_temp)
        plt.legend(loc="best")
        plt.title("Hyperparameter Optimization: Marginal Likelihood Evolution")
        plt.xlabel("Num Iteration", Fontsize = 14)
        plt.ylabel("MLL Loss", Fontsize = 14)

        plt.show()
        best_model = min(best_kernel.values(), key=lambda x: x[0])[1]
        self.fitted_model = best_model


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    import pandas as pd
    train_x = torch.tensor(pd.read_csv(train_x_name, delimiter=",").values)
    eval_x = torch.tensor(pd.read_csv(train_x_name, delimiter=",").values)[1000:1100,:]
    train_y = torch.tensor(pd.read_csv(train_y_name, delimiter=",").values)
    eval_y = torch.tensor(pd.read_csv(train_y_name, delimiter=",").values)[1000:1100]
    train_y = torch.reshape(train_y, (train_y.shape[0],))

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = torch.tensor(pd.read_csv(test_x_name, delimiter=",").values)

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(eval_x)

    print(prediction,eval_y.numpy())
    cost = cost_function(eval_y.numpy(), prediction.numpy())
    # print('cost ',cost)

if __name__ == "__main__":
    main()