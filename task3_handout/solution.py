import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm


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

        m52 = ConstantKernel(self.variance_f) * Matern(
            length_scale=self.length_scale_f, nu=self.nu_f
        )
        self.gpr_f = GaussianProcessRegressor(kernel=m52, alpha=self.noise_f ** 2)

        ## Speed
        self.length_scale_v = 0.5
        self.nu_v = 2.5
        self.variance_v = np.sqrt(2)

        self.noise_v = 0.0001  # Speed
        self.mean_mapping_v = 1.5

        self.min_v = 1.2

        m52 = ConstantKernel(self.variance_v) * Matern(
            length_scale=self.length_scale_v, nu=self.nu_v
        ) + ConstantKernel(self.mean_mapping_v)
        self.gpr_v = GaussianProcessRegressor(kernel=m52, alpha=self.noise_v ** 2)

        ## X- and f- and v- sample
        np.random.seed(0)
        self.X_sample = np.random.rand(1) * 5
        self.f_sample = f(self.X_sample)
        self.v_sample = v(self.X_sample)

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
        mu_sample_v = self.gpr_v.predict(self.X_sample.reshape(-1, 1))

        sigma_f = sigma_f.reshape(-1, 1)
        sigma_v = sigma_v.reshape(-1, 1)

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [...]

        mask = (mu_sample_v > self.v_treshold)[0]
        if (mask == False).all():
            mu_sample_opt = np.max(mu_sample_f)
            v_penalty = mu_sample_v[np.argmax(mu_sample_f)]
        else:
            mu_sample_opt = mu_sample_f[mask].max()
            v_penalty = mu_sample_v[mu_sample_f[mask].argmax()]

        # xi = (np.max(self.f_sample) - mu_f) / sigma_f
        with np.errstate(divide="warn"):
            imp_f = mu_sample_opt - mu_f
            Z_f = imp_f / sigma_f
            ei_f = imp_f * norm.cdf(Z_f) + sigma_f * norm.pdf(Z_f)

            imp_v = v_penalty - mu_v
            Z_v = imp_v / sigma_v
            ei_v = imp_v * norm.cdf(Z_v) + sigma_v * norm.pdf(Z_v)

            ei = ei_f + ei_v * 0.9

            # if v_penalty < self.v_treshold:
            #     ei += v_penalty
            ei[sigma_f == 0.0] = 0.0
            ei[sigma_v == 0.0] = 0.0

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
        self.v_sample = np.append(self.v_sample, v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        mask = self.v_sample > self.v_treshold
        if (mask == False).all():
            mask = self.X_sample > 0

        X_sample = self.X_sample[mask]
        f_sample = self.f_sample[mask]

        return X_sample[np.where(f_sample == f_sample.max())]


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
