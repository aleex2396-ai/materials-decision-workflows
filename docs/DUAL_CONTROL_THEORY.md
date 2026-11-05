# A controlled two-step dual-control reference

This module answers a narrow question: when does using an informative action improve the cost of a later safe action? It is a finite-dimensional reference with an exact posterior and exact terminal decision, and a **numerical first-stage Bellman search**. It is not a new general dual-control algorithm. The related literature and retrieval provenance are in [the review](../research/DUAL_CONTROL_REVIEW.md).

## 1. Model and assumptions

The scalar system is

\[
x_{t+1}=a x_t+b u_t+w_t,\qquad 0\le u_t\le u_{\max},\quad x_t\le x_{\max}.
\]

The gain is fixed during both stages, initially \(b\sim\operatorname{Unif}[\ell,h]\), with \(0<\ell\le h\). States and \(a\) are known exactly. Process disturbances are independent \(\operatorname{Unif}[-\bar w,\bar w]\), independent of the initial gain. Controls depend causally on observed states. There is only an upper state constraint. These assumptions differ from Gaussian GP regression and from adversarial bounded noise.

The configuration requires \(0\le a<1\), \(a x_{\max}+\bar w\le x_{\max}\), effort weight \(\lambda>0\), and slew weight \(\rho\ge0\). Zero input therefore preserves the upper-safe set. The largest input safe for every gain in the current support and every bounded disturbance is

\[
U(x,[\ell,h])=\min\left\{u_{\max},\frac{x_{\max}-a x-\bar w}{h}\right\}.
\]

For an upper-safe state this quantity is nonnegative. The implementation clips round-off to the input interval. This is a pathwise safety argument using support bounds; it does not rely on the Bayesian probabilities being well calibrated. An incorrect gain/noise support or state measurement error invalidates the stated guarantee.

## 2. An exact finite-dimensional Bayesian update

Write the observed innovation as \(z=x^+-a x\). For \(u>0\) and \(\bar w>0\),

\[
p(z\mid b,u)=\frac{1}{2\bar w}\mathbf 1\{|z-bu|\le\bar w\}.
\]

The likelihood is constant wherever positive. Multiplication by a uniform prior gives another uniform distribution, on

\[
\ell^+=\max\{\ell,(z-\bar w)/u\},\qquad
h^+=\min\{h,(z+\bar w)/u\}.
\]

Induction shows that the full history is summarized by this interval. This uniformity is a consequence of the particular likelihood and prior, not a generic property of set-membership identification. If \(u=0\), the observation is independent of \(b\) and the belief is unchanged. If \(\bar w=0\) and \(u>0\), the gain becomes known exactly. A transition with an empty support intersection raises an error instead of silently widening or repairing the model.

## 3. Closed-form terminal Bayes action

Let \(v\) be the previous input, \(r\) the next-state target, and define the stage cost

\[
c=(x^+-r)^2+\lambda u^2+\rho(u-v)^2.
\]

For a uniform belief,

\[
m=E[b]=(\ell+h)/2,\quad s_2=E[b^2]=(\ell^2+\ell h+h^2)/3,
\quad \sigma_b^2=(h-\ell)^2/12.
\]

Thus the expected cost is the strictly convex quadratic

\[
q(u)=(a x+m u-r)^2+\sigma_b^2u^2+\bar w^2/3
+\lambda u^2+\rho(u-v)^2=A u^2+2B u+C,
\]

where \(A=s_2+\lambda+\rho>0\), \(B=(a x-r)m-\rho v\), and \(C=(a x-r)^2+\bar w^2/3+\rho v^2\). Its constrained optimum is

\[
u_1^*(x,[\ell,h],v;r)=\operatorname{clip}_{[0,U(x,[\ell,h])]}(-B/A).
\]

`bayes_one_step` returns this action and its value. This is a Bayes action: simply replacing \(b\) by its mean would discard \(\sigma_b^2 u^2\).

## 4. Two-stage Bellman value and a matched information ablation

Let \(P\) denote the current uniform belief and \(P^+(z,u)\) its update. With distinct targets \(r_0,r_1\), the informed objective at a fixed first action is

\[
J_{\rm adapt}(u)=q_0(u;x,P,v,r_0)
+E_z\!\left[V_1(a x+z,P^+(z,u),u;r_1)\right].
\]

The numerical reference minimizes this objective over a uniform grid in \([0,U(x,P)]\). It considers the effect of the first action on the next physical state **and** on the next belief. These are the two effects in dual control.

The frozen-belief ablation has the same two stages, state observations, objective weights, integration law and first-action grid. At the second stage it chooses

\[
u_F(z)=u_1^*(a x+z,P,u;r_1),
\]

using the original belief and its original support cap. Its *actual* conditional cost is evaluated under \(P^+(z,u)\), not under the stale prior. Consequently,

\[
J_F(u)=q_0(u)+E_z[q_1(u_F(z);a x+z,P^+(z,u),u,r_1)].
\]

Evaluating the frozen action under the prior again would mix two probability laws and would not measure the actual cost of that policy. Both policies observe the next state, so this comparison isolates permission to update the uncertain gain, including the resulting relaxation of the support-based safe cap. It does not separately isolate posterior-mean adaptation from constraint relaxation. Both effects belong to the declared ablation.

### Information-value certificate

For each observation, write the posterior quadratic as \(q_z(u)=A_z u^2+2B_z u+C_z\), and its safe minimizer as \(u_A\). Since \(h^+\le h\), the posterior feasible interval contains the frozen feasible interval; \(u_F\) remains feasible. The first-order optimality condition gives

\[
q_z(u_F)-q_z(u_A)
=A_z(u_F-u_A)^2+q_z'(u_A)(u_F-u_A)
\ge A_z(u_F-u_A)^2\ge0.
\]

If \(u_A\) is interior, the first inequality is equality. If the actions differ on an event of positive probability, expected improvement for that same first action is strictly positive. Therefore

\[
\inf_u J_{\rm adapt}(u)\le\inf_u J_F(u).
\]

If a frozen minimizer exists and has strictly positive information benefit, the optimized inequality is strict. On the common finite grid, nonnegative quadrature weights preserve the weak inequality up to floating-point error. This certificate proves a comparison within the assumed two-stage model; it does not prove superiority over all adaptive controllers or in misspecified dynamics.

### A strict-benefit example and no-benefit cases

Take \(a=0\), \(P=\operatorname{Unif}[0.2,0.6]\), \(\bar w=0.025\), \(\lambda=0.04\), \(\rho=0\), \(u_{\max}=0.8\), \(x_{\max}=1\), and second target \(r_1=0.2\). At the fixed safe first action \(u=0.5\), the frozen terminal action is

\[
u_F=\frac{0.2(0.4)}{(0.2^2+0.2\cdot0.6+0.6^2)/3+0.04}=0.375.
\]

For innovations \(z\in(0.075,0.085)\), the posterior is uniform on \([0.2,(z+0.025)/0.5]\), with upper endpoint below \(0.22\). The adapted terminal action is greater than \(0.49\) and at most \(0.5\); all terminal caps exceed these actions. The event has probability \(\int_0^{0.01}100s\,ds=0.005\). Since \(A_z\ge0.08\), the expected information benefit is greater than

\[
0.005\times0.08\times(0.49-0.375)^2=5.29\times10^{-6}>0.
\]

This proves a strictly positive benefit for the declared first action without selecting a successful random seed. It does not assert that this first action minimizes either two-stage objective. Conversely, a known gain or first action \(u=0\) gives identical informed/frozen terminal beliefs and zero information benefit. A nonzero action can also have zero decision benefit when both terminal actions saturate at the same bound; information alone need not change the decision.

## 5. Numerical integration and its limits

For \(u>0\), \(\bar w>0\), \(h>\ell\), the innovation density is

\[
p(z)=\frac{h^+(z)-\ell^+(z)}{2\bar w(h-\ell)}
\]

on \([\ell u-\bar w,h u+\bar w]\). It is piecewise linear, with possible breakpoints \(\ell u\pm\bar w,h u\pm\bar w\). `observation_quadrature` splits at all distinct breakpoints and uses Gauss-Legendre nodes on each segment. The posterior continuation is not generally a polynomial, and changes in its active constraints can introduce additional kinks. Accordingly the code reports quadrature order and action-grid spacing, and studies must report sensitivity to both. Weights are not silently renormalized. Degenerate gain, zero action and zero-noise cases have explicit branches.

The tests check normalization, innovation mean/variance, and both posterior moment tower identities. They compare the terminal optimum with independent numerical integration and scalar minimization, and compare frozen-policy cost with direct joint gain/noise Monte Carlo. These checks complement the derivation; passing a few refinement cases does not certify every configuration or a uniform error rate.

The zero-noise limit deserves care: an arbitrarily small positive action can identify the gain exactly while zero action cannot. The value can be discontinuous at zero, and a continuous-action infimum need not be attained. Grid refinement alone must not be advertised as a general proof of exact optimality.

## 6. Why adversarial bounded noise and iid uniform noise differ

Consider any causal policy with \(0\le u_t\le u_{\max}\), identical initial state, and two gains \(b_1<b_2\) in the prior. Let \(\delta=b_2-b_1\), with \(\delta u_{\max}\le2\bar w\). Construct disturbances after the common action is chosen:

\[
w_t^{(1)}=\delta u_t/2,\qquad w_t^{(2)}=-\delta u_t/2.
\]

Both are admissible and

\[
b_1u_t+w_t^{(1)}=b_2u_t+w_t^{(2)}.
\]

Induction gives identical observed histories and hence identical subsequent actions. For a randomized policy, condition on the same random seed. Thus no estimator that is uniformly valid over all admissible disturbance sequences can exclude either gain on that history, at any finite time. Its feasible-set diameter is at least \(\delta\). This is an elementary indistinguishability construction with a causal adversary, not an iid-noise lower bound and not a claim of a new identification theorem.

Under the stochastic model the conclusion changes. Apply a constant \(u>0\), use the tight true noise bound, and let \(w_1,\ldots,w_N\) be iid uniform. Before intersection with the prior, the feasible interval is

\[
\left[b+\frac{\max_i w_i-\bar w}{u},\quad
b+\frac{\min_i w_i+\bar w}{u}\right].
\]

Its expected width is

\[
\frac{2\bar w-E[\max_i w_i-\min_i w_i]}{u}
=\frac{4\bar w}{u(N+1)}.
\]

Intersection with the prior can only reduce width. The extrema converge almost surely to the support endpoints, so the width converges to zero. This relies on a tight bound and probability mass arbitrarily close to both endpoints. A conservative assumed bound \(\bar w_{\rm assumed}>\bar w_{\rm true}\) leaves limiting untruncated width \(2(\bar w_{\rm assumed}-\bar w_{\rm true})/u\). Unknown time-varying gain, measurement noise and unmodeled dynamics also change the argument.

These two results are compatible: stochastic extreme observations eventually reveal support boundaries almost surely; a bounded adversary may systematically mask a pair of gains forever. The study reports them as different assumptions and different questions.

## 7. Scope of the evidence

`src/crfml/dual.py` is an inspectable controlled reference, not the project's claim to a novel algorithm. Its main value is exposing a correct matched ablation, an analytic information-benefit certificate, and the distinction between robust containment and stochastic identification. The prior here is invented and independent of confidential observations. No fitted GP is being interpreted as a uniform posterior, and no simulator result establishes physical chemistry or deployment safety.
