"""Auditable exact GP, analytic marginal-likelihood gradient, conformal diagnostics."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize


class ExactGP:
    def __init__(self, noise=.03, optimize=True):
        if noise <= 0:
            raise ValueError('noise must be positive')
        self.noise = noise
        self.optimize = optimize

    @staticmethod
    def kernel(x, z, log_params):
        ell, amp = np.exp(log_params)
        d2 = np.sum((x[:, None, :] - z[None, :, :])**2, axis=2)
        return amp**2 * np.exp(-.5*d2/ell**2)

    def nll_gradient(self, params):
        k = self.kernel(self.x, self.x, params)
        cov = k + np.eye(len(k)) * (self.noise**2 + 1e-9)
        factor = cho_factor(cov, lower=True)
        alpha = cho_solve(factor, self.y)
        nll = .5*self.y @ alpha + np.log(np.diag(factor[0])).sum() + len(k)*.5*np.log(2*np.pi)
        inv = cho_solve(factor, np.eye(len(k)))
        w = inv - np.outer(alpha, alpha)
        d2 = np.sum((self.x[:, None, :] - self.x[None, :, :])**2, axis=2)
        dk = (k*d2/np.exp(2*params[0]), 2*k)
        grad = np.array([.5*np.sum(w*deriv) for deriv in dk])
        return float(nll), grad

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) < 3:
            raise ValueError('Need >=3 matching observations')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Missing/nonfinite observations are not zero')
        self.center, self.scale = x.mean(0), np.maximum(x.std(0), .05)
        self.x = (x-self.center)/self.scale
        self.ymean = float(y.mean())
        self.y = y-self.ymean
        initial = np.array([np.log(2.), np.log(max(y.std(), .05))])
        if self.optimize:
            result = minimize(self.nll_gradient, initial, jac=True, method='L-BFGS-B',
                              bounds=[(-2, 3), (-5, 2)], options={'maxiter': 70})
            self.params = result.x
            self.optimizer = {'success': bool(result.success), 'message': str(result.message),
                              'nll': float(result.fun)}
        else:
            self.params = initial
            self.optimizer = {'success': True, 'message': 'fixed hyperparameters'}
        cov = self.kernel(self.x, self.x, self.params) + np.eye(len(x))*(self.noise**2+1e-9)
        self.factor = cho_factor(cov, lower=True)
        self.alpha = cho_solve(self.factor, self.y)
        return self

    def predict(self, x, observation=True):
        x = (np.asarray(x)-self.center)/self.scale
        cross = self.kernel(self.x, x, self.params)
        mean = self.ymean + cross.T @ self.alpha
        solved = solve_triangular(self.factor[0], cross, lower=True)
        var = np.exp(2*self.params[1]) - np.sum(solved**2, axis=0)
        var = np.maximum(var, 0) + (self.noise**2 if observation else 0)
        return mean, np.sqrt(var)


class Ridge:
    def fit(self, x, y, penalty=1.):
        self.center, self.scale = x.mean(0), np.maximum(x.std(0), .05)
        a = np.column_stack((np.ones(len(x)), (x-self.center)/self.scale))
        reg = np.eye(a.shape[1])*penalty
        reg[0, 0] = 0
        self.coef = np.linalg.solve(a.T@a+reg, a.T@y)
        return self

    def predict(self, x):
        return np.column_stack((np.ones(len(x)), (x-self.center)/self.scale)) @ self.coef


def conformal_radius(y, mean, std, alpha=.1):
    """Split-conformal order statistic; infinity when finite n cannot support alpha."""
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    y, mean, std = map(np.asarray, (y, mean, std))
    if y.shape != mean.shape or std.shape != y.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError('Calibration arrays must be nonempty matching vectors')
    if not all(np.isfinite(v).all() for v in (y, mean, std)) or np.any(std <= 0):
        raise ValueError('Invalid calibration observations or scales')
    k = int(np.ceil((len(y)+1)*(1-alpha)))
    if k > len(y):
        return float('inf')
    return float(np.sort(np.abs(y-mean)/std)[k-1])


def metrics(y, mean, std=None, radius=None):
    result = {'rmse': float(np.sqrt(np.mean((y-mean)**2))), 'mae': float(np.mean(np.abs(y-mean)))}
    if std is not None:
        result['gaussian_nll'] = float(np.mean(.5*np.log(2*np.pi*std**2)+.5*((y-mean)/std)**2))
    if radius is not None:
        result.update(coverage=float(np.mean(np.abs(y-mean) <= radius*std)),
                      mean_width=float(np.mean(2*radius*std)))
    return result
