# Mathematical contract of the implementation

The results below describe `src/crfml/models.py`, `discovery.py`, `control.py` and `study.py`. They separate probabilistic prediction, experiment selection and a deterministic control certificate. The methods and elementary lemmas are established ideas; this document claims no new theorem or state-of-the-art algorithm.

## 1. Exact GP regression and its likelihood gradient

For training-only standardised descriptors z_i and centred responses y_i, the implementation uses

\[
k_\theta(z,z')=s^2\exp[-\|z-z'\|^2/(2\ell^2)],\quad
\theta=(\log\ell,\log s),\quad
C=K_\theta+(\sigma^2+10^{-9})I.
\]

The response mean is estimated from training data and then held fixed. Noise sigma is supplied, not estimated. One shared length scale is used after per-feature standardisation; this is not an ARD kernel. The tiny diagonal addition is numerical jitter, separate from physical observation noise. Let C=LL^T and v=C^{-1}y, computed through triangular solves. The negative log marginal likelihood is

\[
\mathcal L(\theta)=\tfrac12 y^TC^{-1}y+\sum_i\log L_{ii}+\tfrac n2\log(2\pi).
\]

Using \(dC^{-1}=-C^{-1}(dC)C^{-1}\) and \(d\log\det C=\operatorname{tr}(C^{-1}dC)\) gives the exact derivative implemented in `nll_gradient`:

\[
\partial_{\theta_j}\mathcal L
=\tfrac12\operatorname{tr}[(C^{-1}-vv^T)\partial_{\theta_j}C],
\quad
\partial_{\log\ell}K_{ij}=K_{ij}\|z_i-z_j\|^2/\ell^2,
\quad \partial_{\log s}K=2K.
\]

The sign is for **negative** log likelihood. Its analytic gradient is checked against numerical differentiation in `tests/test_models.py`. L-BFGS-B finds a bounded local hyperparameter solution; the likelihood is generally nonconvex. The code records optimiser status, which must not be replaced with a claim of global fitting optimality. These are the standard GP equations; see [Rasmussen and Williams, chapters 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf) and [5](https://gaussianprocess.org/gpml/chapters/RW5.pdf).

For a new descriptor z, conditioning the joint Gaussian yields

\[
\mu(z)=\bar y+k_z^Tv,\qquad
s_f^2(z)=k(z,z)-\|L^{-1}k_z\|^2,\qquad
s_y^2(z)=s_f^2(z)+\sigma^2.
\]

`observation=False` returns latent uncertainty; the default includes observation noise. Small negative variance from floating-point cancellation is clipped to zero. Direct Gaussian conditioning independently checks the implementation. The dense factorisation costs O(n^3) time and O(n^2) memory. Fitted hyperparameters and the fitted mean are treated as known at prediction time, so their uncertainty is not integrated out. The posterior need not be calibrated if kernel, noise or data assumptions are wrong.

## 2. Finite-sample calibration and post-selection

Fit the predictor and positive scale on independent training data. On n calibration observations, compute scores R_i=|Y_i-mu(X_i)|/s_y(X_i). For alpha in (0,1), choose k=ceil((n+1)(1-alpha)). Set q=R_(k) when k<=n, and infinity otherwise. The interval is mu(x) +/- q s_y(x).

If calibration and the next test score are exchangeable conditional on the fitted model, the next score has uniform rank among n+1 scores in the continuous case. Therefore its probability of exceeding the kth order statistic is at most alpha. Ties preserve the lower coverage bound. The extra n+1 and the possible infinite interval are necessary; a convenient interpolated percentile is not this finite-sample rule. This is a **marginal** guarantee over calibration and the new point, not a per-input or per-realised-calibration guarantee. See the [quantile lemma and split-conformal construction](https://arxiv.org/abs/1904.06019).

The study's counterexample uses independent Gaussian estimation errors E_j~N(0,1) for M candidates with equal true value zero. Each interval E_j +/- z has coverage 1-alpha when z=Phi^{-1}(1-alpha/2). Optimisation chooses j*=argmax E_j. Consequently

\[
\Pr\{|E_{j^*}|\le z\}
=\Pr\{-z\le\max_j E_j\le z\}
=\Phi(z)^M-\Phi(-z)^M.
\]

At M=50 and alpha=0.1 this is about 0.0769, despite 0.9 fixed-candidate coverage. The executable counterexample compares this exact value with Monte Carlo estimates. It uses a known Gaussian error law, not calibrated materials data and not the full discovery algorithm.

Using z_M=Phi^{-1}(1-alpha/(2M)) gives each candidate error probability alpha/M. The union bound then establishes simultaneous coverage at least 1-alpha, and hence coverage after any selector. Independence is used for the exact maximum distribution, but is unnecessary for this union bound. Ordinary marginal conformal calibration on heterogeneous inputs does not supply the required per-candidate premise.

A related distinction is confidence in the probability mass of a calibrated set. For IID continuous scores with CDF F,

\[
\Pr\{F(R_{(k)})<p\}
=\sum_{j=k}^{n}{n\choose j}p^j(1-p)^{n-j}.
\]

This follows by counting scores below F^{-1}(p). For the maximum score k=n, at least 59 independent calibration units are needed for p=0.95 with failure confidence delta=0.05, since 0.95^n<=0.05. This is stronger and different from marginal conformal prediction. It parallels the order-statistic analysis in [Alexeenko and Bitar](https://arxiv.org/abs/2004.03069).

## 3. Experiment selection is a heuristic, not the safety theorem

For a Gaussian predicted latent quality Q~N(mu,sigma^2), improvement over incumbent q_best has expectation

\[
\operatorname{EI}=(\mu-q_{best})\Phi(z)+\sigma\phi(z),
\qquad z=(\mu-q_{best})/\sigma.
\]

The implemented constrained-EI score multiplies this by Phi((d_max-mu_d)/sigma_d), the surrogate probability of meeting a defect threshold. This product treats objective and constraint posteriors as independent. Quality EI uses latent variance; the defect feasibility term uses the default observation variance. The incumbent is selected from noisy observations. Thus this is a plug-in acquisition heuristic; it does not integrate over uncertainty in the best latent feasible value. Hyperparameters are refitted at every acquisition round.

A nonzero feasibility probability does not prohibit an infeasible experiment. The initial candidates are random, not certified safe seeds. Accordingly the discovery loop records sampled defect violations, and does not inherit [SafeOpt's reachable-safe-set theorem](https://proceedings.mlr.press/v37/sui15.html). The relevant [BoTorch implementation](https://github.com/meta-pytorch/botorch/blob/b0dd01be61d517c387e234903a6e2cc38c057ea2/botorch/acquisition/analytic.py) is a reference for this distinction and for more stable log-space EI variants.

## 4. Robust MPC is a convex QP under the stated assumptions

The public control model is

\[
x_{t+1}=a x_t+b u_t+w_t,\quad
0\le a<1,\quad b\in[b_-,b_+],\quad
0\le u_t\le u_{max},\quad |w_t|\le\bar w.
\]

a and state measurements are exact; b is constant within a rollout. The safety constraint is **only the upper limit** x<=xmax. No lower bound, chemical validity, or real equipment safety is proved. The synthetic model and constants are invented independently of private measurements.

For horizon H define c_i=a^{i+1}x_t and the lower-triangular matrix S_ij=a^{i-j} for j<=i, otherwise zero, with zero-based indices. Unrolling the recurrence gives the componentwise upper trajectory

\[
x_{t+1:t+H}\le c+b_+SU+\bar w S\mathbf1.
\]

All coefficients are nonnegative, so the bound is attained by b=b_+ and every disturbance equal to +wbar for the chosen nonnegative plan. Enforcing it is therefore an exact robust counterpart for these upper trajectory constraints and the open-loop plan. Changing signs, uncertain a, or state-dependent bounds requires a new derivation.

The cost uses a nominal gain b_hat. Let G=b_hat S, D be the first-difference matrix, p=(u_previous,0,...,0)^T, and r the constant reference vector. Then

\[
J(U)=\|c+GU-r\|_2^2+\lambda\|U\|_2^2+\rho\|DU-p\|_2^2
=U^TQU+2f^TU+\text{constant},
\]

\[
Q=G^TG+\lambda I+\rho D^TD,\quad
f=G^T(c-r)-\rho D^Tp.
\]

With nonnegative effort lambda and slew rho, Q is positive semidefinite; positive lambda makes it positive definite. Configuration validation enforces positive effort and nonnegative slew. The Hessian is 2Q. The feasible set contains linear inequalities and box bounds, so this is a convex QP. SLSQP solves it numerically; the code checks constraint residuals independently and falls back to zero on solver failure. Feasibility checks alone do not establish a reported numerical solution's optimality to arbitrary precision.

## 5. Gain-set contraction and the probing tradeoff

Given a positive executed input u_t and exact observed transition, set z_t=x_(t+1)-a x_t. Bounded noise implies

\[
b\in[(z_t-\bar w)/u_t,(z_t+\bar w)/u_t].
\]

Intersecting with the old gain interval preserves the true b by induction. The updated width satisfies

\[
W_{t+1}\le\min\{W_t,2\bar w/u_t\}.
\]

Larger positive inputs tighten this **upper bound** on width, but need not cause strict contraction in every realised transition. At u=0 there is no gain information; only consistency with |z_t|<=wbar can be checked. Empty intersection detects inconsistency, whereas a nonempty set can still be wrong when assumptions fail.

From an upper-safe state, the largest one-step safe nonnegative input is

\[
u_{probe}=\min\{u_{max},(x_{max}-a x_t-\bar w)/b_+\},
\]

provided the numerator is nonnegative. It is the maximal feasible action because the worst-case next state increases linearly with u. It minimises the one-observation width bound 2wbar/u over positive feasible actions. The `safe_probe` policy applies this rule for its initial probing rounds, then uses adaptive MPC. It is not a solution to a stochastic Bellman equation or a globally optimal dual-control policy: probing can increase transient tracking cost or overshoot the reference within the allowed safe region.

## 6. Backup invariance and recursive feasibility

Assume a*xmax+wbar<=xmax. For any x<=xmax, u=0 gives

\[
x^+=a x+w\le a x_{max}+\bar w\le x_{max}.
\]

Thus the upper-safe half-line is invariant under the zero-input backup, and the all-zero horizon plan is feasible from every upper-safe state. A feasible robust first action preserves upper safety; recomputing the plan is then feasible by the same backup argument. A gain-set update that retains truth preserves this reasoning. This establishes conditional recursive feasibility without a conventional terminal equality constraint for this particular one-sided model.

It does not establish asymptotic stability or convergence to the reference. After disturbance/gain bounds fail, emergency zero input may help but is no longer certified by this proof. The code distinguishes a controller's conditional certificate from whether the simulator's hidden truth satisfies assumptions. Historical observation of zero violations is empirical evidence, not a universal theorem.

## 7. Scope of the research claim

The mathematically justified connection is that a learnt predictor can improve a nominal objective while a separate robust envelope governs admissibility; informative safe actions can contract that envelope. Cost, gain uncertainty, constraint violations and prediction accuracy measure different aspects of the resulting behaviour. The experiment must show their tradeoff, including cases in which probing or the GP does not help.

The scalar controller described in this document is bounded-disturbance adaptive MPC. Its policy named `oracle` knows the true gain but still solves receding-horizon robust MPC; it is not a noncausal clairvoyant lower bound on total realised cost. Report its gap as a reference comparison, not guaranteed nonnegative regret.

The repository's separate v2 controller implements [Wasserstein minimax-regret control](https://arxiv.org/abs/2304.06783) in `dr_control.py`, with a true noncausal comparator and the source paper's known-dynamics assumptions. See [DECISION_LEARNING.md](DECISION_LEARNING.md) and [DECISION_FOCUSED_THEORY.md](DECISION_FOCUSED_THEORY.md). Its distributional guarantee has no hard state/input constraints and does not replace the bounded-MPC safety proof above. The later [moment-ambiguity formulation](https://arxiv.org/abs/2512.10906) remains related work rather than an implemented method.
