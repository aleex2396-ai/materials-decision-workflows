# Learning second moments for causal Wasserstein regret control

This note develops the mathematical diagnostics for the decision-focused experiment. The inner controller is the established formulation of [Al Taha, Yan and Bitar (2023)](https://arxiv.org/abs/2304.06783). The learning method uses standard implicit differentiation; [the review](../research/DECISION_FOCUSED_REVIEW.md) identifies the closest prior work. The propositions below are reference-problem derivations, not a claim of previously unknown theory.

## 1. Reduced control problem and two different losses

Use the paper's lifted dynamics \(x=Fu+Gw\), cost \(x^TQx+u^TRu\), and causal disturbance-feedback controller \(u=Kw\). Define

\[
D=R+F^TQF\succ0,\quad K_{\rm nc}=-D^{-1}F^TQG,\quad
\Delta=K-K_{\rm nc},\quad C_K=\Delta^TD\Delta\succeq0.
\]

The trajectory-wise regret is \(w^TC_Kw\). Let \(M=E[ww^T]\succ0\) be the nominal **second moment**, equal to covariance only for zero-mean disturbances. A full-dimensional Gaussian nominal law is a convenient way to meet the source theorem's absolute-continuity assumption. The radius \(r>0\) refers to the type-2 Wasserstein metric on the full lifted disturbance trajectory.

For fixed \(K\), the worst-case expected regret is

\[
W(K;M)=\inf_{\gamma>\lambda_{\max}(C_K)} f(K,\gamma;M),
\]

\[
f(K,\gamma;M)=\gamma r^2+\operatorname{tr}(M A(K,\gamma)),\qquad
A=\gamma^2(\gamma I-C_K)^{-1}-\gamma I.
\]

Write \(V(M)=\inf_{K\in\mathcal K}W(K;M)\). Causality is enforced by representing only the permitted entries of \(K\). In the augmented disturbance ordering \((x_0,w_0,\ldots,w_{T-1})\), block row \(t\) permits column blocks \(0,\ldots,t\). The resulting semidefinite program and this reduced representation describe the same inner problem under the source assumptions. The reduced function is smooth only inside its strict spectral domain.

The implemented learner minimizes a different outer loss:

\[
L_{\rm val}(\theta)=\operatorname{tr}\big(C_{K(M_\theta)}M_{\rm val}\big).
\]

This is expected regret under validation disturbances, after synthesizing a robust controller using the estimated nominal moment \(M_\theta\). It is neither the optimized robust value \(V(M_\theta)\) nor the robust transfer value \(W(K(M_\theta);M_{\rm val})\). These quantities answer different questions and have different gradients.

The shrinkage family in the implementation is

\[
M_\theta=(1-\alpha)S+\alpha\operatorname{diag}(S)
+\beta\frac{\operatorname{tr}S}{d}I,\quad\theta=(\alpha,\log\beta).
\]

Here \(S\) uses only the estimation split. Validation data tune \(\theta\); test data evaluate the selected estimator/controller once. Positive \(\beta\) and \(\operatorname{tr}S>0\) give a positive definite matrix even when \(S\) is singular. This family can shrink correlation and inflate all eigenvalues; it cannot represent an arbitrary covariance correction. Both decision-focused and likelihood-focused learners must use this same family and data budget for a meaningful loss comparison. See [DECISION_LEARNING.md](DECISION_LEARNING.md) for executable API details.

## 2. Envelope sensitivity is not a policy hypergradient

Let \(S_\gamma=(\gamma I-C_K)^{-1}\). Holding \((K,\gamma)\) fixed,

\[
\nabla_M f=A=\gamma^2S_\gamma-\gamma I
=\gamma C_K(\gamma I-C_K)^{-1}\succeq0.
\]

If a unique regular optimal pair exists, the envelope theorem gives \(\nabla_M V(M)=A_*\). Without differentiability, the same matrix is a **supergradient** of the concave function \(V\): because \(V\) is an infimum of affine functions of \(M\),

\[
V(M+H)\le V(M)+\operatorname{tr}(A_*H).
\]

This sensitivity quantifies how the *reoptimized robust value* changes when its nominal second moment changes. It does not differentiate the controller transferred to a fixed validation environment. Optimizing a predictor using only this envelope can also encourage making its own nominal uncertainty look artificially small. It is not the declared decision-focused validation objective.

The sensitivity is anisotropic. If \(C_K=\operatorname{diag}(c,0)\), \(c>0\), then

\[
W(K;M)=c(\sqrt{M_{11}}+r)^2.
\]

It is independent of \(M_{22}\). Equal-Frobenius perturbations in the two diagonal directions therefore have different risk effects. This fixed-controller observation motivates a task-aware loss, but by itself says nothing about which fitted controller transfers better. Section 5 provides the stronger policy comparison.

## 3. Implicit differentiation of the reduced optimum

Let \(z\) collect the free causal entries of \(K\) and \(\gamma\), and let \(z^*(\theta)\) satisfy the interior stationarity equations

\[
g(z,\theta)=\nabla_z f(z;M_\theta)=0.
\]

If \(f\) is twice continuously differentiable in a neighborhood, the optimum is interior, and \(H=\nabla^2_{zz}f\) is nonsingular, the classical implicit function theorem gives

\[
\frac{dz^*}{d\theta}=-H^{-1}\frac{\partial g}{\partial\theta},\qquad
\nabla_\theta L_{\rm val}=\left(\nabla_zL_{\rm val}\right)^T\frac{dz^*}{d\theta}.
\]

The final identity assumes the validation moment is fixed independently of \(\theta\), as in this experiment. There is no envelope shortcut because \(z^*\) minimizes the training robust objective, not the validation loss.

The differential identities needed to compute these quantities are

\[
dC=(dK)^TD\Delta+\Delta^TD(dK),\qquad
dS_\gamma=S_\gamma(dC-d\gamma I)S_\gamma.
\]

For unrestricted matrix notation before projection to the causal entries,

\[
\nabla_Kf=2\gamma^2D\Delta S_\gamma M S_\gamma,
\]

\[
\partial_\gamma f=r^2-\operatorname{tr}M
+2\gamma\operatorname{tr}(MS_\gamma)
-\gamma^2\operatorname{tr}(MS_\gamma^2).
\]

For a moment perturbation \(E\), the mixed derivatives are

\[
d_M(\nabla_K f)[E]=2\gamma^2D\Delta S_\gamma E S_\gamma,
\]

\[
d_M(\partial_\gamma f)[E]
=-\operatorname{tr}E+2\gamma\operatorname{tr}(ES_\gamma)
-\gamma^2\operatorname{tr}(ES_\gamma^2).
\]

The outer derivative is \(\nabla_K L_{\rm val}=2D\Delta M_{\rm val}\) and \(\partial_\gamma L_{\rm val}=0\). The shrinkage derivatives are \(\partial_\alpha M=\operatorname{diag}(S)-S\) and \(\partial_{\log\beta}M=\beta(\operatorname{tr}S/d)I\). Differentiating the displayed gradients with the resolvent identity gives analytic Hessian-vector products.

A numerical SDP solution supplies the starting point; stationarity polishing improves derivative accuracy in the reduced representation. This is not a rigorous global-optimality certificate. A small stationarity residual alone is insufficient: inspect the spectral margin, Hessian conditioning, the linear-system residual, and agreement with fresh perturbed SDP solutions. Near singularity, an unstable derivative should be reported as a failed regularity/accuracy check. Adding a ridge to the differentiation system changes the derivative unless its approximation is explicitly analyzed. Bounded outer training iterations do not establish convergence to a global bilevel optimum.

## 4. A finite robust policy-transfer bound

**Proposition.** Fix the dynamics, cost, causal policy class and radius \(r>0\). Let \(M,\widehat M\succ0\). Assume the true problem has an attained strictly feasible optimal pair \((K_*,\gamma_*)\). Let \((\widehat K,\widehat\gamma)\) be strictly feasible and within \(\eta\ge0\) of the global optimum for \(\widehat M\), in objective value. Define \(A_*\) and \(\widehat A\) using the displayed resolvent. Then

\[
0\le W(\widehat K;M)-V(M)
\le\eta+\operatorname{tr}\big((M-\widehat M)(\widehat A-A_*)\big).
\]

Consequently,

\[
W(\widehat K;M)-V(M)
\le\eta+\|M-\widehat M\|_*\|\widehat A-A_*\|_{\rm op},
\]

where \(\|\cdot\|_*\) is the nuclear norm. One may replace the last factor by \(\|\widehat A\|_{\rm op}+\|A_*\|_{\rm op}\). For either pair \(j\), with spectral margin \(\tau_j=\gamma_j-\|C_j\|_{\rm op}>0\),

\[
\|A_j\|_{\rm op}=\frac{\gamma_j\|C_j\|_{\rm op}}{\tau_j}.
\]

The alternative Frobenius inequality is \(\eta+\|M-\widehat M\|_F\|\widehat A-A_*\|_F\).

**Proof.** True robust evaluation reoptimizes the scalar dual, so \(W(\widehat K;M)\le f(\widehat K,\widehat\gamma;M)\). Approximate optimality on the estimated moment gives \(f(\widehat K,\widehat\gamma;\widehat M)\le f(K_*,\gamma_*;\widehat M)+\eta\). Subtract \(f(K_*,\gamma_*;M)=V(M)\), using the exact affine dependence of \(f\) on \(M\), to obtain the trace bound. Nuclear/operator duality gives the norm bound. Diagonalizing \(C_j\) shows that the eigenvalues of \(A_j\) are \(\gamma_j c_i/(\gamma_j-c_i)\); this expression increases with \(c_i\ge0\), giving the spectral-margin formula. The lower bound follows from optimality of \(V(M)\). \(\square\)

This is a deterministic transfer statement, not a sampling/generalization theorem. It requires the same fixed radius and dynamics in both problems and a genuine objective-gap tolerance \(\eta\); a solver's stationarity residual is not automatically that tolerance. The bound concerns robust transfer, whereas the implemented validation objective is ordinary expected regret. No ordering of ordinary expected regret follows from this proposition. The bound also explains why a small moment error can be dangerous near a small dual spectral margin, but it need not be tight in every direction.

## 5. An exact covariance-error/policy-performance ordering reversal

Consider one step with scalar dynamics \(x_1=x_0+u_0+w_0\), terminal state weight one and effort weight one. The initial-state cost may be any fixed nonnegative weight because it cancels from regret. Here

\[
D=2,\quad K_{\rm nc}=(-1/2,-1/2),\quad K=(k,0).
\]

Set \(z=k+1/2\) and \(v=(z,1/2)^T\). Then \(C_K=2vv^T\). For any positive definite second moment \(M\) and radius \(r\ge0\),

\[
W(K;M)=2\left(\sqrt{v^TMv}+r\|v\|_2\right)^2.\tag{1}
\]

To see this, the sole positive eigenvalue of \(C_K\) is \(c=2\|v\|^2\), with direction \(e=v/\|v\|\). The reduced scalar dual depends only on \(s=e^TMe\). Its value is \(c(\sqrt{s}+r)^2\), yielding (1). At \(r=0\) this follows directly by expectation, with no finite-\(\gamma\) assertion.

Let the true moment be \(M_0=I_2\), and compare

\[
M_A=\begin{pmatrix}1&0\\0&1+\delta\end{pmatrix},\qquad
M_B=\begin{pmatrix}1&\epsilon\\\epsilon&1\end{pmatrix},
\]

where \(0<\epsilon<1\) and \(\delta>\sqrt2\epsilon\). Their covariance errors satisfy

\[
\|M_B-M_0\|_F=\sqrt2\epsilon<\delta=\|M_A-M_0\|_F.
\]

For \(M_A\), both norms in (1) are uniquely minimized at \(z_A=0\). Thus its fitted controller is exactly optimal for the true moment. For \(M_B\),

\[
W_B(z)=2\left(\sqrt{z^2+\epsilon z+1/4}
+r\sqrt{z^2+1/4}\right)^2,
\]

and

\[
W_B'(0)=2(1+r)\epsilon>0.
\]

Since \(M_B\succ0\), the first norm is strictly convex on the affine line \(v=(z,1/2)\); the sum remains positive, strictly convex and coercive. Squaring preserves strict convexity here. There is therefore a unique minimizer \(z_B<0\), which is not the true optimum. Under the true moment,

\[
W(K;I_2)=2(1+r)^2(z^2+1/4),
\]

so the exact robust transfer excesses are

\[
W(K_A;I_2)-V(I_2)=0,\qquad
W(K_B;I_2)-V(I_2)=2(1+r)^2z_B^2>0.
\]

This proves the policy ordering reversal for every \(r\ge0\), rather than merely demonstrating a fixed-controller value sensitivity. The ordinary true-distribution expected-regret excess is likewise \(2z_B^2>0\). For example, \(\delta=0.4\), \(\epsilon=0.1\) meet all inequalities. A direct scalar minimization, the general spectral evaluator and the SDP synthesis can independently verify the same instance. This is an existence counterexample: it disproves a universal implication from smaller covariance Frobenius error to better decisions; it does not show that a decision-focused estimator wins on every distribution.

## 6. What would constitute useful experimental evidence

The mathematical question is whether the matched decision-focused shrinkage loss improves transferred decisions at finite data budgets, and when it loses. Paired repeated simulations should compare identical estimation/validation/test samples, radius, covariance family and optimization budgets. Report both covariance error and ordinary expected regret, robust evaluation where declared, numerical gradient failures, train/validation/test gaps, spectral margins and training cost. Include nominal-radius or no-learning controls and a distribution-shift condition selected before final test results.

The counterexample and transfer proposition explain mechanisms. The independent SDP, scalar-dual and transport calculations check computation. Neither a favorable counterexample nor a successful gradient check establishes empirical generalization, and the experiment should preserve negative outcomes.
