# Wasserstein distributionally robust regret control

Primary source: Feras Al Taha, Shuhao Yan and Eilyan Bitar, *A Distributionally Robust Approach to Regret Optimal Control using the Wasserstein Distance*, arXiv:2304.06783v2, 16 August 2023; CDC 2023, DOI 10.1109/CDC49753.2023.10384311. The full [primary manuscript](https://arxiv.org/pdf/2304.06783) and its [HTML mathematical rendering](https://arxiv.org/html/2304.06783v2) were inspected, including Theorems 2–4 and their proofs. The derivation below states the parameterization and numerical counterpart used by `src/crfml/dr_control.py`.

## Lifted multi-state dynamics and information pattern

For t=0,...,T−1, let x[t+1]=A[t]x[t]+B[t]u[t]+w[t], with n-dimensional states, m-dimensional inputs, known time-varying matrices, and perfect state observation. Stack z=(x[0],w[0],...,w[T−1]) in R^d, x=(x[0],...,x[T]) in R^d and u=(u[0],...,u[T−1]) in R^p, where d=n(T+1) and p=mT. The lifted dynamics are

\[
x=Fu+Gz,\qquad F\in\mathbb R^{d\times p},\quad G\in\mathbb R^{d\times d}.
\]

The top state block has F[0,:]=0 and G[0,0]=I. Each next block is obtained by multiplying the previous block row by A[t], then inserting B[t] in input block t and I in disturbance block t+1. This recursion supports arbitrary time-varying matrices and is tested against explicit simulation.

A policy is u=Kz. Its block K[t,j] can be nonzero only for j≤t: u[t] can use x[0] and w[0],...,w[t−1], but never w[t] or future disturbances. The last disturbance block is therefore unavailable to every input. Disturbances can be reconstructed recursively from fully observed states and the known model. Thus this parameterization implements a causal feedback policy rather than an open-loop plan that sees the realized future.

For symmetric Q⪰0 and R≻0, define J(u,z)=xᵀQx+uᵀRu. The implementation accepts full lifted weights, including cross-time quadratic terms.

## Noncausal oracle and exact regret

Let D=R+FᵀQF≻0. Setting the derivative of J with respect to an unrestricted input trajectory to zero gives

\[
K_\star=-D^{-1}F^TQG,\qquad
C_\star=G^TQG-G^TQFD^{-1}F^TQG.
\]

Completing the square yields

\[
J(Kz,z)=z^TC_\star z+z^T(K-K_\star)^TD(K-K_\star)z.
\]

Consequently C_reg(K)=(K−K*)ᵀD(K−K*)⪰0 is the pointwise regret matrix and C_cost(K)=C_reg(K)+C* is the total-cost matrix. The oracle is generally noncausal; it is not an ordinary controller given the true plant parameter. The implementation verifies the identity against direct state/input costs for realized disturbances.

## Nominal law and Wasserstein ambiguity

The ambiguity set is all finite-second-moment probability laws P on R^d satisfying W₂(P,P₀)≤r, using Euclidean transport cost on the **whole disturbance trajectory**, including x[0]. It does not impose independent disturbances over time.

The paper's Assumption 1 requires P₀ to be absolutely continuous. A raw empirical law is discrete and does not meet that assumption. Our data-based constructor instead uses an explicitly smoothed empirical law

\[
P_0=\frac1N\sum_{i=1}^N\mathcal N(z_i,\tau^2 I),\qquad
M_0=\mathbb E_{P_0}[zz^T]=\frac1N\sum_i z_i z_i^T+\tau^2 I,
\quad \tau>0.
\]

This smoothing is part of the declared nominal model, not a hidden numerical ridge or an automatic statistical certificate. Alternatively, a nonsingular Gaussian nominal law may be supplied through its second moment. M₀ is the uncentered second moment, not just covariance: subtracting the empirical mean changes the problem when the mean is nonzero. Positive definiteness is checked. The algorithms depend on P₀ through M₀, but the declared law still matters for interpreting a Wasserstein radius and sampling a worst-case witness.

The radius is an explicit robustness parameter. Radius sensitivity does not by itself establish that the unknown population lies inside the ball. No tail/concentration premise or out-of-sample confidence level is silently inferred from the samples.

## Inner quadratic adversary: scalar dual and constructive witness

For fixed nonzero C⪰0, Theorem 2 gives

\[
\sup_{P:W_2(P,P_0)\le r}\mathbb E_P[z^TCz]
=\min_{\gamma>\lambda_{\max}(C)}
\gamma(r^2-\operatorname{tr}M_0)
+\gamma^2\operatorname{tr}\{M_0(\gamma I-C)^{-1}\}.
\]

To see the algebra, the transport dual maximizes zᵀCz−γ‖z−v‖² for each nominal sample v. Its finite maximizer is z=γ(γI−C)⁻¹v. Substituting and averaging supplies the expression above. The strict inequality is essential; a semidefinite boundary cannot simply be inverted.

Theorem 3 characterizes the unique optimal scalar γ by

\[
\operatorname{tr}\left[(\gamma(\gamma I-C)^{-1}-I)^2M_0\right]=r^2.
\]

For `C=V diag(λ)Vᵀ`, put `a[j]=(VᵀM₀V)[j,j]`. The scalar equation is `Σ_j a[j](λ[j]/(γ−λ[j]))²=r²`, monotone decreasing on `γ>max λ`. A separately implemented bracketed root solve evaluates any controller without consulting the SDP variables. Its numerically stable dual objective is `γr²+Σ_j a[j]γλ[j]/(γ−λ[j])`.

Set T*=γ(γI−C)⁻¹. P*=T*#P₀ is an admissible worst-case distribution; its second moment is M*=T*M₀T*ᵀ. The implementation reports the transport budget tr((T*−I)M₀(T*−I)ᵀ), primal value tr(CM*), scalar-dual value and their gap. A Gaussian Wasserstein-distance calculation supplies a further independent check. For r=0, the value is tr(CM₀) and T*=I. For C=0, the value is zero and no boundary-saturating adversary is required.

## Exact convex synthesis SDP

Write H=D^(1/2)(K−K*) in R^(p×d), using the symmetric positive definite square root. Let B=0 for regret and B=C* for total cost. Then C(K)=HᵀH+B. This common representation permits a controlled comparison of cost-robustness and regret-robustness.

The decision variables are causal K∈R^(p×d), γ≥0 and symmetric X⪰0∈R^(d×d). Minimize

\[
\gamma(r^2-\operatorname{tr}M_0)+\operatorname{tr}X
\]

subject to

\[
\begin{bmatrix}
\gamma I-B&H^T\\H&I_p
\end{bmatrix}\succ0,
\qquad
\begin{bmatrix}
X&\gamma M_0^{1/2}&0\\
\gamma M_0^{1/2}&\gamma I-B&H^T\\
0&H&I_p
\end{bmatrix}\succeq0.
\]

The large matrix has dimension 2d+p. Its entries are affine in the variables. Taking the bottom Schur complement yields γI−B−HᵀH=γI−C(K), and then the second Schur complement yields X⪰γ²M₀^(1/2)(γI−C(K))⁻¹M₀^(1/2). Minimizing tr X recovers the scalar dual. This is a congruence-transformed version of Theorem 4's SDP, whose bottom block is D⁻¹. The total-cost counterpart follows by retaining C* in the middle block.

The exact mathematical problem has a strict first inequality. The numerical solver enforces a declared small positive margin on that block and reports minimum eigenvalues, causality residual, γ−λmax(C), solver status and the difference between the SDP objective and the independently recomputed worst-case value. Solver termination alone is not acceptance. The resulting finite-precision problem is a margin-perturbed SDP, and residual thresholds are exposed rather than described as an exact numerical proof.

The code rejects failed numerical KKT checks on the returned conic duals. Scalar nonnegativity constraints have explicit dual multipliers, so stationarity and complementarity also cover an endpoint at zero. For each stationarity block, the infinity norm of the summed Lagrangian-gradient terms is divided by one plus the sum of their individual infinity norms. The maximum scaled stationarity residual and each negative dual eigenvalue divided by max(1, dual spectral norm) must be at most 10 times the declared tolerance. Summed absolute cone complementarity, including scalar cones, is divided by 1+|solver objective| and must be at most 100 times the declared tolerance; this aggregate budget matches the existing objective-recomputation scale. These thresholds are fixed functions of the declared tolerance, independent of experiment outcomes. Acceptance is reported through `kkt_validation_passed` and `numerical_validation_passed`; failed checks raise.

These finite-precision checks are numerical evidence for the margin-perturbed convex solve, not a rigorous upper bound on global synthesis suboptimality. In particular, a stationarity residual is not the global objective-gap parameter in an optimization error theorem without a separate conversion justified by conditioning and feasibility. The worst-case distribution's transport-duality gap and recomputed objective concern evaluation of a fixed K, and do not certify that K is optimal. A bilevel derivative requires a further reliable stationarity/curvature check of the reduced objective.

## Baselines that answer distinct questions

| Method | Optimization objective | Interpretation |
|---|---|---|
| `nominal` | min_K tr(C_reg(K) M₀) | Also the nominal expected-cost optimizer: C* is independent of K. These are one optimizer, not two artificially distinct baselines. |
| `energy_cost` | min_K λmax(C_cost(K)) | Finite-horizon squared induced gain from disturbance energy to weighted state/input energy; the H-infinity cost baseline. |
| `energy_regret` | min_K λmax(C_reg(K)) | Worst-case regret over the unit Euclidean disturbance ball. |
| `dr_cost` | min_K sup_(W₂≤r) E[J(Kz,z)] | Distributionally robust expected cost. |
| `dr_regret` | min_K sup_(W₂≤r) E[J(Kz,z)−J(K*z,z)] | Distributionally robust regret relative to the noncausal oracle. |

The energy baselines use the LMI [[ηI−B,Hᵀ],[H,I]]⪰0 and minimize η. The nominal problem is a convex weighted least-squares problem in the free causal entries of K. All methods use the same lifted system and causal mask. Every controller can be cross-evaluated for nominal cost, nominal regret, worst-case Wasserstein cost, worst-case Wasserstein regret and energy-normalized regret. Cross-evaluation must use the adversary for the corresponding loss; a controller's worst-case cost distribution need not maximize its regret.

## Validation and scope

Required numerical checks are: lifted-versus-recursive dynamics; noncausal first-order optimality; pointwise completion-of-square identity; forbidden future-feedback entries equal zero; nominal cost/regret optimizer equivalence; exact scalar dual versus constructive worst-case transport; monotonic fixed-controller value with radius; PSD/causality residuals; the DR-regret solution evaluated with an independent scalar dual should not exceed feasible baseline regret beyond the declared solver tolerance. A one-step example also permits direct scalar optimization over its sole causal gain, independent of the SDP modeling layer.

This implementation is finite-horizon unconstrained LQ control with known dynamics. The ambiguity is in disturbance laws. The Wasserstein ball includes unbounded-support distributions, so this result supplies a worst-case **expected regret** guarantee, not hard state/input constraints, recursive MPC feasibility, unknown-dynamics robustness or a generic nonlinear guarantee. Those would require different formulations. The scalar bounded-disturbance MPC in `control.py` remains a separate method with different assumptions and a different objective.
