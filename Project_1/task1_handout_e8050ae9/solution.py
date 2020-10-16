import numpy as np
import gpytorch
from gpytorch.lazy import LazyTensor as LT
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

class Model():
    def __init__(self):
        # self.train_x = None
        # self.train_y = None
        # self.test_x = None
        # self.fitted_model = None
        pass

    # @property
    # def train_x(self):
    #     return self.__train_x
    #
    # @train_x.setter
    # def train_x(self, value):
    #     if value is not None:
    #         if not torch.is_tensor(value):
    #             self.__train_x = torch.tensor(value)
    #         else:
    #             self.__train_x = value
    #
    # @property
    # def train_y(self):
    #     return self.__train_y
    #
    # @train_y.setter
    # def train_y(self, value):
    #     if value is not None:
    #         if not torch.is_tensor(value):
    #             self.__train_y = torch.tensor(value)
    #         else:
    #             self.__train_y = value
    #         if not len(self.__train_y.shape) == 1:
    #             self.__train_y = torch.reshape(self.__train_y, (self.__train_y.shape[0],))
    #
    # @property
    # def test_x(self):
    #     return self.__test_x
    #
    # @test_x.setter
    # def test_x(self, value):
    #     if value is not None:
    #         if not torch.is_tensor(value):
    #             self.__test_x = torch.tensor(value)
    #         else:
    #             self.__test_x = value
    #
    # @property
    # def fitted_model(self):
    #     return self.__fitted_model
    #
    # @fitted_model.setter
    # def fitted_model(self, value):
    #     if value is not None:
    #         if isinstance(value, Model_template):
    #             self.__fitted_model = value

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
        if test_x is not None:
            if torch.is_tensor(test_x):
                self.test_x = test_x

        self.fitted_model.eval()
        with torch.no_grad():
            test_train_covar = gpytorch.lazy.delazify(self.fitted_model.covar_module(self.test_x, self.train_x))
            train_test_covar = test_train_covar.transpose(-1, -2)
            test_test_covar = gpytorch.lazy.delazify(self.fitted_model.covar_module(self.test_x, self.test_x))
            train_train_covar = gpytorch.lazy.delazify(self.fitted_model.covar_module(self.train_x, self.train_x))
            # choosing n random columns for approximation
            idx = np.arange(self.train_x.shape[0])
            np.random.shuffle(idx)
            n = 750
            active = idx[:n]
            passive = idx[n:]

            noise_inv = 1/self.fitted_model.likelihood.noise
            identity = torch.eye(self.train_x.shape[0])

            Kmm = train_train_covar[...,active][active,...]
            Kmn = train_train_covar[active,:]
            Knm = train_train_covar[:,active]
            prior_covar_inv = noise_inv*identity - (noise_inv**2)*Knm@torch.pinverse(Kmm+noise_inv*Kmn@Knm)@Kmn

            mean_correction_rhs = prior_covar_inv@self.train_y
            posterior_mean = test_train_covar @ mean_correction_rhs
            covar_correction_rhs = prior_covar_inv@train_test_covar
            posterior_covar = gpytorch.lazy.lazify(test_test_covar + test_train_covar @ covar_correction_rhs.mul(-1))

            out_approx = gpytorch.distributions.MultivariateNormal(posterior_mean, posterior_covar)
            # lower_approx, upper_approx = out_approx.confidence_region()
            # out_base = self.fitted_model(self.test_x)
            # lower_base, upper_base = out_base.confidence_region()
        # y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        return out_approx.mean

    def fit_model(self, train_x, train_y):
        if self.train_x is None:
            if not torch.is_tensor(train_x):
                self.train_x = torch.tensor(train_x)
            else:
                self.train_x = train_x
        elif not torch.is_tensor(self.train_x):
            self.train_x = torch.tensor(train_x)

        if self.train_y is None:
            if not torch.is_tensor(train_y):
                self.train_y = torch.tensor(train_y)
            else:
                self.train_y = train_y
        elif not torch.is_tensor(self.train_y):
            self.train_y = torch.tensor(train_y)

        if not len(self.train_y.shape) == 1:
            self.train_y = torch.reshape(self.train_y, (self.train_y.shape[0],))

        kernels = ["RBF", "quadratic", "Matern-1/2", "Matern-3/2", "Matern-5/2"]
        best_kernel = {}
        for kernel in kernels:
            print("Training kernel: ", kernel)
            model_temp = Model_template(self.train_x, self.train_y, self.get_kernel(kernel)).double()
            model_temp.train()
            optimizer = torch.optim.Adam([{'params': model_temp.parameters()}], lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model_temp.likelihood, model_temp)
            training_iter = 100

            losses = []
            for i in range(training_iter):
                optimizer.zero_grad() # Zero gradients from previous iteration
                output = model_temp(self.train_x) # Output from model
                loss = -mll(output, self.train_y) #approximation of the log marginal likelihood --> loss function for parameter optimzation: minimize it
                loss.backward() #gradient of loss function with respect to its parameters (Jacobian)

                losses.append(loss.item())
                optimizer.step()

                if not (i%25):
                      print("Iter - %d -- Loss %f"%(i, losses[-1]))

            # plt.plot(losses, Linewidth = '2', label=f"{kernel}")
            best_kernel[kernel] = (loss.item(), model_temp)
        # plt.legend(loc="best")
        # plt.title("Hyperparameter Optimization: Marginal Likelihood Evolution")
        # plt.xlabel("Num Iteration", Fontsize = 14)
        # plt.ylabel("MLL Loss", Fontsize = 14)
        #
        # plt.show()
        best_model = min(best_kernel.values(), key=lambda x: x[0])[1]
        self.fitted_model = best_model


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    import pandas as pd
    train_x = np.loadtxt(train_x_name, delimiter=",")
    train_y = np.loadtxt(train_y_name, delimiter=",")

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=",")

    M = Model()
    FIT = False
    if FIT:
        M.fit_model(train_x, train_y)
        for param_name, param in M.fitted_model.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.item()}')
    else:
        print("Set fitted model: ")
        X = torch.tensor(train_x)
        y = torch.tensor(train_y)
        y = torch.reshape(y, (y.shape[0],))
        fitted_model = Model_template(X, y, M.get_kernel("Matern-1/2")).double()
        M.fitted_model = fitted_model
        actual_likelihood_noise = torch.tensor(data = [0.0017], dtype=torch.float64)
        M.fitted_model.likelihood.noise_covar.noise = actual_likelihood_noise
        actual_outputscale = torch.tensor(data = [0.1415], dtype=torch.float64)
        M.fitted_model.covar_module.outputscale = actual_outputscale
        actual_lengthscale = torch.tensor(data = [[2.6226]], dtype = torch.float64)
        M.fitted_model.covar_module.base_kernel.kernels[0].lengthscale = actual_lengthscale

    M.train_x = torch.tensor(train_x)
    train_tensor_y = torch.tensor(train_y)
    M.train_y = torch.reshape(train_tensor_y, (train_tensor_y.shape[0],))
    M.test_x = test_x
    prediction = M.predict(test_x)
    # M.plot_model()

    # print(prediction)


if __name__ == "__main__":
    main()