# Decision-focused covariance estimation through a minimax-regret controller

This component learns two covariance regularization parameters by differentiating the downstream causal controller. Its mathematical baseline is the Wasserstein minimax-regret SDP of Al Taha, Yan and Bitar, [*A Distributionally Robust Approach to Regret Optimal Control using the Wasserstein Distance*](https://arxiv.org/abs/2304.06783). Reproducing that controller, differentiating its regular interior optimum, and conducting a controlled synthetic estimation experiment are separate claims. This project does not claim a new control theorem or a state-of-the-art learning algorithm.

`src/crfml/decision_learning.py` contains the estimator, derivatives and numerical safeguards. `tests/test_decision_learning.py` independently checks calculus and controller re-solves. The companion `DECISION_FOCUSED_THEORY.md` discusses transfer guarantees and a covariance-error ordering counterexample. Public benchmark data come only from the independent generator in `simulator.py`; the algebraic unit-test fixtures are independent of benchmark seeds.

## Model and estimand

The finite-horizon linear dynamics and quadratic cost weights are known. The disturbance vector contains the random initial state and subsequent additive process disturbances. A boolean causal mask defines the entries of the linear disturbance-feedback matrix K that the controller can use. No dynamics identification occurs in this component.

Let D be positive definite, Kstar be the noncausal optimal controller, and define

\[
 E=K-K_\star,\qquad C(K)=E^\top DE.
\]

The nominal law is the **nondegenerate zero-mean Gaussian** N(0,M), where M is positive definite. The covariance estimator is not declared to be the raw empirical distribution. For a fixed positive Wasserstein radius r and a strictly interior dual scalar gamma,

\[
 S=(\gamma I-C)^{-1},\qquad
 f(K,\gamma,M)=\gamma(r^2-\operatorname{tr}M)+\gamma^2\operatorname{tr}(MS),
 \qquad \gamma>\lambda_{\max}(C).
\]

The controller minimizes f over the causal entries of K and gamma. The exact robust interpretation is conditional on the nominal law, radius, known model, quadratic cost and causal policy class. A chosen radius alone is not a statistical coverage theorem. Neither simulation nor covariance fitting certifies physical chemistry, constraints absent from the LQ model, or correctness of learned dynamics.

## Covariance family and matched density baseline

For training samples xi_i, compute the uncentered second moment

\[
 \widehat M_{\rm train}=n_{\rm train}^{-1}\sum_i\xi_i\xi_i^\top,
 \qquad s=\operatorname{tr}(\widehat M_{\rm train})/d.
\]

This uses the declared zero-mean model; it does not estimate and subtract a mean. The estimator is

\[
 M_\theta=(1-\alpha)\widehat M_{\rm train}
 +\alpha\operatorname{diag}(\widehat M_{\rm train})+\beta sI,
 \qquad \theta=(\alpha,\log\beta).
\]

Default bounds are alpha in [0,1] and beta in [0.001,1], with initial alpha=0.25, beta=0.05. The positive ridge remains explicit even when the training sample count is smaller than d. Its scale s depends only on training data. All-zero training data are rejected. The parameter derivatives are

\[
 \partial_\alpha M=\operatorname{diag}(\widehat M_{\rm train})-\widehat M_{\rm train},
 \qquad \partial_{\log\beta}M=\beta sI.
\]

Decision fitting minimizes the empirical validation regret

\[
 J(\theta)=\operatorname{tr}\{C(K_\theta)\widehat M_{\rm val}\}.
\]

The matched likelihood baseline uses **the same training and validation draws, covariance family, parameter bounds, initial point and outer iteration cap**. It minimizes Gaussian validation negative log likelihood, dropping only the common constant:

\[
 L(\theta)=\tfrac12[\log\det M_\theta+
 \operatorname{tr}(M_\theta^{-1}\widehat M_{\rm val})].
\]

Its analytic derivative is the Frobenius inner product of
\(\tfrac12[M^{-1}-M^{-1}\widehat M_{\rm val}M^{-1}]\) with each covariance derivative. Equal outer iteration caps do not mean equal computation: only the decision objective requires repeated SDPs. The experiment records the actual evaluations and controller-solve counts.

## Covariance scale and the supplementary fixed-trace ablation

The primary estimator changes both covariance shape and scale. In particular,
trace(M_theta)=(1+beta)trace(M_train). For any fixed controller and c>0, the reduced dual gives the exact identity

\[
 W(K;cM,r)=cW(K;M,r/\sqrt{c}).
\]

Thus beta can change the controller's effective robustness radius, as well as covariance shape. A gain from the primary estimator cannot be attributed solely to learning correlations. The primary parameterization and endpoints remain unchanged.

The supplementary setting `trace_normalize=True` instead uses

\[
 \overline M_\theta=t\frac{M_\theta}{q},\qquad
 t=\operatorname{tr}(\widehat M_{\rm train}),\quad q=\operatorname{tr}(M_\theta).
\]

This keeps the training second-moment trace fixed while learning covariance shape. Training scale is still estimated and can vary across training draws; it cannot vary with theta within this fit. The exact derivative is

\[
 \partial_j\overline M_\theta=
 \frac{t}{q}\left[\partial_j M_\theta-
 M_\theta\frac{\operatorname{tr}(\partial_jM_\theta)}{q}\right].
\]

The trace of each derivative is zero. Both the decision and likelihood fitting functions accept this flag and apply the same mapping. Covariance and NLL derivatives are independently checked by centered differences; normalized decision hypergradients are checked against five independent controller requests. An additional test verifies the covariance-scale/radius identity and equality of the corresponding optimal controllers. The default is `False`, preserving the primary estimator exactly.

## Covariance sensitivity and what it does not prove

For fixed K and gamma,

\[
 \frac{\partial f}{\partial M}=\gamma^2S-\gamma I.
\]

At a regular optimum the envelope theorem gives this same derivative for the optimized robust value. If C has eigenvalue lambda, the corresponding sensitivity is

\[
 \frac{\gamma\lambda}{\gamma-\lambda}.
\]

Consequently covariance errors aligned with high-regret eigenvectors can matter more than equally large errors in directions with small lambda; sensitivity increases as the spectral gap shrinks. This is a sensitivity of the **optimized robust value when its nominal law changes**. It is not automatically the sensitivity of actual regret when a fitted controller is transferred to a fixed test law. The latter involves how K changes, as computed below. The tests separately check the fixed-controller derivative and the envelope derivative after independent controller re-solves.

## Exact differential calculations

Let z contain free causal entries of K followed by gamma. The gradient is

\[
 \nabla_K f=2\gamma^2DESMS,
\]

\[
 \partial_\gamma f=r^2-\operatorname{tr}M
 +2\gamma\operatorname{tr}(MS)-\gamma^2\operatorname{tr}(MS^2).
\]

Only allowed entries of the K gradient are retained. For a direction (dK,dgamma),

\[
 dC=dK^\top DE+E^\top DdK,
 \qquad dS=S(dC-d\gamma I)S.
\]

The code assembles an **analytic Hessian** from the resulting directional derivatives:

\[
 d(\nabla_K f)=4\gamma d\gamma DESMS+
 2\gamma^2[D(dK)SMS+DE(dS)MS+DESM(dS)],
\]

\[
 d(\partial_\gamma f)=2d\gamma\operatorname{tr}(MS)
 +2\gamma\operatorname{tr}(M(dS))
 -2\gamma d\gamma\operatorname{tr}(MS^2)
 -\gamma^2\operatorname{tr}\{M((dS)S+S(dS))\}.
\]

Centered finite differences independently test this Hessian; finite differences are not used to construct it. Mixed z/M derivatives follow by replacing M with dM in the matrix-gradient formula and replacing the gamma derivative with

\[
 -\operatorname{tr}(dM)+2\gamma\operatorname{tr}((dM)S)
 -\gamma^2\operatorname{tr}((dM)S^2).
\]

At a stationary optimum with a nonsingular Hessian H, the implicit function theorem gives

\[
 \frac{dz^*}{d\theta}=-H^{-1}\partial_\theta\nabla_z f.
\]

Since validation moment data remain fixed,

\[
 \nabla_K J=2DE\widehat M_{\rm val},\qquad
 \partial_\gamma J=0,\qquad
 \nabla_\theta J=(\nabla_z J)^\top\frac{dz^*}{d\theta}.
\]

This is differentiation of the smooth interior reduced optimum, not differentiation of arbitrary SDP solver iterates, and not a derivative through the random sample generator. The outer problem need not be convex; the method reports local optimization status.

## Numerical acceptance criteria

Each learning controller begins with a fresh `dr_control.synthesize` SDP solve. `polish_controller` then applies Newton refinement to the same reduced objective, with spectral-feasibility backtracking. It records the initial and polished objectives, refinement steps and stationarity. The independent SDP and polished objective must agree to relative tolerance 1e-4, normalized by max(1,abs(polished value)); the solver already checks its own feasibility and independently recomputes its objective.

The numerical adapter makes at most two attempts: Clarabel first, then SCS only if a solver or numerical acceptance check fails. Both solve the identical problem at the same acceptance tolerance. It records solver names, elapsed time, accepted or rejected status, raw failures and parsed diagnostic fields. An accepted first attempt never triggers the fallback. If both fail, their logs remain attached to the exception. This recovery policy was introduced after the initial supplementary run recorded a scaled-complementarity failure; independent SCS passed the original thresholds on that same problem. The original failure was retained locally. Recovery changes neither the estimator endpoint nor its acceptance thresholds, and a rejected point is never treated as a certificate.

The default refinement target is infinity-norm gradient at most 3e-10. `implicit_hypergradient` independently rejects a point unless:

- The nominal second moment is finite, symmetric and strictly positive definite.
- The radius is positive and gamma-lambda_max(C) is strictly positive.
- The gradient infinity norm is at most 2e-7.
- The spectral gap divided by max(1,abs(gamma)) is at least 1e-8.
- The analytic Hessian is symmetric to relative tolerance 1e-8, positive definite and has condition number at most 1e10.
- The relative implicit linear-equation residual is at most 1e-9 and the derivative is finite.

A zero-regret problem with no finite interior dual optimum, singular Hessian, failed factorization, corrupted solve result, or poor stationarity causes an explicit failure. No ridge is silently added to H and no unverified gradient is substituted. These numerical checks support a numerical claim; they are not an interval-arithmetic certificate.

Before decision optimization, the default verification requests the base controller and four perturbed controllers independently. Each request takes one conic attempt unless the bounded recovery policy is needed. Centered differences at step 2e-4 in each theta coordinate must agree with the implicit gradient within 2e-6 absolute plus 0.003 relative tolerance. This check uses validation regret only. Subsequent evaluations continue to enforce the stationarity and conditioning checks. The unit tests use stricter tolerances on an independent small fixture.

## Budgets, selection and held-out evaluation

The fitting APIs accept only training and validation arrays. The caller must establish that these draws are disjoint. Both fitting methods record sample counts, actual objective evaluations, optimization status, every evaluated parameter vector and its validation objective. The decision method additionally records every gradient, spectral gap and Hessian condition number, along with the verification report. Controller requests and actual conic attempts are separate budgets. A callback returning `(point, diagnostics)` exposes the latter; with a packed-point-only callback the attempt count is explicitly unknown. The default L-BFGS-B cap is 12 outer iterations; a cap reached before convergence is retained as a nonconvergence report.

Every line-search evaluation remains in the trajectory, including evaluations that the optimizer later rejects. A failed numerical check raises `DecisionFitError`; its `partial_fit` field preserves the failure stage, attempted parameter vector, completed trajectory, verification report and attempted controller-solve count. The experiment can record the failure instead of replacing the endpoint or quietly switching to an easier objective. The returned `test_samples_accessed: 0` describes the fitting function's API and computation; the experiment runner remains responsible for withholding test outcomes until selection has completed.

The final evaluation compares covariance errors, predictive density scores and transferred policy regret separately. A method can improve one and worsen another. The theorem does not guarantee that decision tuning will outperform likelihood tuning on finite validation samples. Synthetic learning results do not validate laboratory process dynamics or physical materials performance.

## API example

```python
geometry = RegretGeometry(problem.D, problem.Kstar, problem.causal_mask)
solve_diagnostics = []

def fresh_solver(M):
    z, diagnostics = solve_learning_controller(problem, M, radius)
    solve_diagnostics.append(diagnostics)
    return z, diagnostics

fit = fit_decision_covariance(
    training_samples, validation_samples, geometry, radius, fresh_solver
)
K, gamma = geometry.unpack(fit["controller"])
density_fit = fit_likelihood_covariance(training_samples, validation_samples)
```

The saved experiment should retain `solve_diagnostics` alongside the fit trajectories. Test samples are consumed only by the separate, frozen evaluation stage.
