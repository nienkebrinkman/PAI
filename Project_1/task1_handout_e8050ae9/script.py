from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

X, y = load_iris(return_X_y=True)
kernel = 1.0 * RBF(1.0)


# gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X, y)
# gpc.score(X, y)

# gpc.predict_proba(X[:2, :])

