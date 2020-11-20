import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Sum, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import matplotlib.pyplot as plt


domain = np.array([[0, 5]])


""" Solution """


class BO_algo:
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        ## Accuracy
        self.length_scale_f = 0.5
        self.nu_f = 2.5
        self.variance_f = 0.5
        self.noise_f = 0.15  # Accuracy

        m52_f = ConstantKernel(self.variance_f) * Matern(
            length_scale=self.length_scale_f, nu=self.nu_f
        ) + WhiteKernel(noise_level=self.noise_f ** 2)
        # self.gpr_f = GaussianProcessRegressor(kernel=m52_f, alpha=self.noise_f ** 2)
        self.gpr_f = GaussianProcessRegressor(kernel=m52_f)

        ## Speed
        self.length_scale_v = 0.5
        self.nu_v = 2.5
        self.variance_v = np.sqrt(2)

        self.noise_v = 0.0001  # Speed
        self.mean_mapping_v = 1.5

        self.min_v = 1.2

        m52_v = ConstantKernel(self.variance_v) * Matern(
            length_scale=self.length_scale_v, nu=self.nu_v
        ) + WhiteKernel(noise_level=self.noise_v ** 2)
        # Ck = ConstantKernel(self.mean_mapping_v)
        # summed_k = Sum(m52, Ck)
        # self.gpr_v = GaussianProcessRegressor(kernel=m52_v, alpha=self.noise_v ** 2)
        self.gpr_v = GaussianProcessRegressor(kernel=m52_v)

        ## X- and f- and v- sample
        np.random.seed(0)
        self.X_sample = np.random.rand(1) * 5
        self.f_sample = f(self.X_sample[0])
        self.f_sample = self.f_sample.reshape(-1, 1)
        self.v_sample = v(self.X_sample[0]) - self.mean_mapping_v
        for i in range(len(self.X_sample) - 1):
            self.f_sample = np.append(self.f_sample, f(self.X_sample[i + 1]))
            self.v_sample = np.append(self.v_sample, v(self.X_sample[i + 1]))

        # Fit the GPs
        self.gpr_f.fit(self.X_sample.reshape(-1, 1), self.f_sample.reshape(-1, 1))
        self.gpr_v.fit(self.X_sample.reshape(-1, 1), self.v_sample.reshape(-1, 1))

        self.v_treshold = 1.2

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        # xx = np.linspace(0, 5, 1000)
        # yy_f, sigma_f = self.gpr_f.predict(xx.reshape(-1, 1), return_std=True)
        # yy_v, sigma_v = self.gpr_v.predict(xx.reshape(-1, 1), return_std=True)

        # yyf = f(xx[0])
        # for i in range(len(xx) - 1):
        #     yyf = np.append(yyf, f(xx[i + 1]))
        # plt.plot(self.X_sample, self.f_sample, "r.")
        # plt.plot(self.X_sample[-1], self.f_sample[-1], ".", color="orange")
        # plt.fill_between(
        #     xx.reshape(len(xx)),
        #     yy_f.reshape(len(xx)) - sigma_f,
        #     yy_f.reshape(len(xx)) + sigma_f,
        #     color="gray",
        #     alpha=0.2,
        # )

        # plt.plot(xx, yy_f)
        # plt.plot(xx, yy_v)
        # plt.fill_between(
        #     xx.reshape(len(xx)),
        #     yy_v.reshape(len(xx)) - sigma_v,
        #     yy_v.reshape(len(xx)) + sigma_v,
        #     color="gray",
        #     alpha=0.2,
        # )
        # plt.plot(xx, yyf)
        # plt.show()
        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain, approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        mu_f, sigma_f = self.gpr_f.predict(np.asarray(x).reshape(-1, 1), return_std=True)
        mu_sample_f = self.gpr_f.predict(self.X_sample.reshape(-1, 1))
        mu_v, sigma_v = self.gpr_v.predict(np.asarray(x).reshape(-1, 1), return_std=True)
        mu_v = mu_v + self.mean_mapping_v
        mu_sample_v = self.gpr_v.predict(self.X_sample.reshape(-1, 1)) + self.mean_mapping_v
        xi = 0.01

        sigma_f = sigma_f.reshape(-1, 1)
        sigma_v = sigma_v.reshape(-1, 1)

        # mask = mu_sample_v > self.v_treshold
        # if (mask == False).all():
        #     mask = mu_sample_v < self.v_treshold
        # mu_sample_opt = np.max(mu_sample_f[mask])

        mu_sample_opt = np.max(mu_sample_f)

        # mu_sample_opt_v = np.max(mu_sample_v[mask])
        # v_penalty = (mu_sample_v[mask])[np.argmax(mu_sample_f[mask])]
        # mask = (mu_sample_v > self.v_treshold)[0]
        # if (mask == False).all():
        #     mu_sample_opt = np.max(mu_sample_f)
        #     v_penalty = mu_sample_v[np.argmax(mu_sample_f)]
        # else:
        #     mu_sample_opt = mu_sample_f[mask].max()
        #     v_penalty = mu_sample_v[mu_sample_f[mask].argmax()]

        # xi = (np.max(self.f_sample) - mu_f) / sigma_f
        with np.errstate(divide="warn"):
            imp_f = mu_f - mu_sample_opt - xi
            Z_f = imp_f / (sigma_f)
            ei_f = imp_f * norm.cdf(Z_f) + sigma_f * norm.pdf(Z_f)

            imp_v = mu_v - self.v_treshold  # - v_penalty - xi
            Z_v = imp_v / (sigma_v)
            Pr_v = norm.cdf(Z_v)  # + sigma_v * norm.pdf(Z_v)
        # print(mu_v, sigma_v, self.v_treshold)
        # g = np.log(mu_v) - np.log(self.v_treshold)
        # print(ei_f, Pr_v)
        ei = ei_f * Pr_v
        # print(ei,x,-(self.v_treshold-0.8*mu_v)*np.abs(ei))
        # if mu_v < self.v_treshold:
        #     ei -= (self.v_treshold - 0.8 * mu_v) ** 2 * np.abs(ei) * 10
        # else:
        #     ei += self.v_treshold*2*ei*10
        ei[sigma_f == 0.0] = 0.0
        ei[sigma_v == 0.0] = 0.0
        # print(ei,mu_v,x,v_penalty)
        return ei[0]

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        self.X_sample = np.append(self.X_sample, x)
        self.f_sample = np.append(self.f_sample, f)
        self.v_sample = np.append(self.v_sample, v - self.mean_mapping_v)

        # Fit the GPs
        self.gpr_f.fit(self.X_sample.reshape(-1, 1), self.f_sample.reshape(-1, 1))
        self.gpr_v.fit(self.X_sample.reshape(-1, 1), self.v_sample.reshape(-1, 1))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        # xx = np.linspace(0,5,1000)
        # yy = self.gpr_f.predict(xx.reshape(-1, 1))

        # yyf = f(xx[0])
        # for i in range(len(xx)-1):
        #     yyf = np.append(yyf, f(xx[i+1]))
        # plt.plot(self.X_sample,self.f_sample,'r.')
        # plt.plot(self.X_sample[-1],self.f_sample[-1],'g.')
        # plt.plot(xx,yy)
        # plt.plot(xx,yyf)
        # plt.show()

        mask = self.v_sample > 0
        if (mask == False).all():
            mask = self.X_sample > 0

        X_sample = self.X_sample[mask]
        f_sample = self.f_sample[mask]
        X_solution = X_sample[np.where(f_sample == f_sample.max())]
        # print(X_solution)

        # X_solution = self.X_sample[np.where(self.f_sample == self.f_sample.max())]
        return X_solution


""" Toy problem to check code works as expected """


def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return -np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return x / 3


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), (
            f"The function next recommendation must return a numpy array of "
            f"shape (1, {domain.shape[0]})"
        )

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), (
        f"The function get solution must return a numpy array of shape (" f"1, {domain.shape[0]})"
    )
    assert check_in_domain(solution), (
        f"The function get solution must return a point within the "
        f"domain, {solution} returned instead"
    )

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = 0 - f(solution)

    print(
        f"Optimal value: 0\nProposed solution {solution}\nSolution value "
        f"{f(solution)}\nRegret{regret}"
    )


if __name__ == "__main__":
    main()
