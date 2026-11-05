import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from crfml.models import ExactGP, conformal_radius
from crfml.simulator import sample, descriptors, pareto_mask
from crfml.study import selection_counterexample


def test_gp_analytic_gradient_matches_finite_difference():
    data = sample(13, 18)
    gp = ExactGP(optimize=False).fit(descriptors(data.x), data.quality)
    theta = np.array([.3, -.7])
    numeric = approx_derivative(lambda v: gp.nll_gradient(v)[0], theta).reshape(-1)
    np.testing.assert_allclose(gp.nll_gradient(theta)[1], numeric, rtol=2e-5, atol=2e-6)


def test_posterior_matches_direct_gaussian_conditioning():
    data = sample(14, 15)
    gp = ExactGP(optimize=False).fit(descriptors(data.x), data.quality)
    x = descriptors(sample(15, 5).x)
    cross = gp.kernel(gp.x, (x-gp.center)/gp.scale, gp.params)
    k = gp.kernel(gp.x, gp.x, gp.params)+np.eye(15)*(gp.noise**2+1e-9)
    mean, std = gp.predict(x, observation=False)
    np.testing.assert_allclose(mean, gp.ymean+cross.T@np.linalg.solve(k,gp.y))
    np.testing.assert_allclose(std**2, np.exp(2*gp.params[1])-np.diag(cross.T@np.linalg.solve(k,cross)), atol=1e-10)


def test_conformal_order_statistic_and_small_sample_abstention():
    assert conformal_radius(np.arange(9.), np.zeros(9), np.ones(9), .1) == 8
    assert np.isinf(conformal_radius(np.arange(8.), np.zeros(8), np.ones(8), .1))
    with pytest.raises(ValueError):
        conformal_radius(np.ones(4), np.ones(4), np.zeros(4))


def test_simplex_and_pareto_contracts():
    with pytest.raises(ValueError):
        descriptors(np.ones((5,4)))
    assert pareto_mask([[1,3],[2,2],[3,4],[1,3]]).tolist() == [True,True,False,True]


def test_postselection_counterexample_matches_exact_probability():
    result = selection_counterexample(repetitions=30000)
    assert abs(result['selected_coverage']-result['selected_exact_coverage']) < .012
    assert result['fixed_candidate_coverage'] > .88
    assert result['selected_coverage'] < .15
    assert result['simultaneous_selected_coverage'] > .9
