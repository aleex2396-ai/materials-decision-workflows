"""Invented public benchmark; constants have no relationship to lab measurements."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MaterialBatch:
    x: np.ndarray
    quality: np.ndarray
    gain: np.ndarray
    defect: np.ndarray
    cost: np.ndarray
    groups: np.ndarray


def descriptors(x):
    """Simplex fractions + process + physically motivated pair interactions."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError('Expected finite (n,4) composition/process array')
    if (x < 0).any() or (x > 1).any() or not np.allclose(x[:, :3].sum(1), 1):
        raise ValueError('Composition must lie on simplex; process in [0,1]')
    c0, c1, c2, p = x.T
    return np.column_stack((x, c0*c1, c1*c2, c0*p, p*p))


def properties(x):
    c0, c1, c2, p = np.asarray(x).T
    quality = 1.1 - 2.2*(c0-.3)**2 - 1.5*(c1-.45)**2 + .14*np.sin(7*p) + .18*c0*p
    gain = .16 + .20*c0 + .10*c1 + .07*p + .035*np.sin(5*c1*p)
    defect = .07 + .35*(p-.45)**2 + .22*c2 + .06*np.sin(7*c0)**2
    cost = .25*c0 + .8*c1 + .4*c2 + .25*p*p
    return quality, gain, defect, cost


def sample(seed, n, shift=False, noise=True):
    rng = np.random.default_rng(seed)
    concentration = [1.5, 1.5, 1.5] if not shift else [.4, 4., .6]
    c = rng.dirichlet(concentration, n)
    p = rng.uniform(0, 1, n) if not shift else rng.beta(7, 1, n)
    x = np.column_stack((c, p))
    q, b, d, cost = properties(x)
    if noise:
        q = q + rng.normal(0, .025, n)
        b = b + rng.normal(0, .006, n)
        d = d + rng.normal(0, .01, n)
    return MaterialBatch(x, q, b, d, cost, np.arange(n))


def pareto_mask(values):
    """All columns minimised; equality does not dominate."""
    a = np.asarray(values)
    return np.array([not np.any(np.all(a <= row, axis=1) & np.any(a < row, axis=1)) for row in a])


def material_process(composition_process, horizon=6):
    """Invented known surface/core thermal-deviation model; no fitted lab constants."""
    from .dr_control import LiftedLQ
    c0,c1,c2,p = composition_process
    A = np.array([[.87-.07*c2,.07+.03*c1],[.09+.02*c0,.78+.08*p]])
    B = np.array([[.12+.22*c0+.05*p],[.04+.06*c1]])
    problem = LiftedLQ.time_invariant(A,B,horizon,np.diag([1.,2.+2*c2]),np.array([[.4]]))
    time = np.arange(horizon+1)
    temporal = (.35+.35*p)**np.abs(time[:,None]-time[None,:])
    spatial = np.array([[1.,.45],[.45,.75]])*(2/1.75)
    return problem, np.kron(temporal,spatial)


def gaussian_trajectories(seed, moment, n):
    """Independent augmented trajectories, preserving within-path correlations."""
    return np.random.default_rng(seed).multivariate_normal(np.zeros(len(moment)),moment,n)


def standard_normal_trajectories(seed, n, dimension):
    return np.random.default_rng(seed).normal(size=(n,dimension))


def same_moment_laws(seed, moment, n, degrees=6):
    """Gaussian and elliptical Student laws with exactly equal population moments."""
    if degrees <= 4:
        raise ValueError('Use finite fourth moments for Monte Carlo error diagnostics')
    root = np.linalg.cholesky(moment)
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(n,len(moment)))@root.T
    student = rng.normal(size=(n,len(moment)))@root.T
    student *= np.sqrt((degrees-2)/rng.chisquare(degrees,n))[:,None]
    return {'gaussian':gaussian,'student':student}


def uniform_gain_episodes(seed, low, high, noise_bound, n):
    rng = np.random.default_rng(seed)
    return rng.uniform(low,high,n),rng.uniform(-noise_bound,noise_bound,(n,2))
