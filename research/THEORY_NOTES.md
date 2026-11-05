# Coverage, selection and bounded uncertainty

These notes distinguish elementary derivations from cited results. They are not a novelty claim. The current implementation uses a GP for predictions and a separately justified bounded-uncertainty controller. A GP interval is not the controller's safety certificate.

## 1. Selection can destroy marginal coverage

Let X take values A and B with probabilities 0.9 and 0.1, and let the true response be Y(A)=Y(B)=0. Define a fitted predictor with intervals C(A)=[-0.1,0.1] and C(B)=[0.9,1.1], and predicted objective means 0 and 1. Then

\[
\Pr\{Y(X)\in C(X)\}=0.9,
\qquad x^*=\arg\max_x\widehat\mu(x)=B,
\qquad \Pr\{Y(x^*)\in C(x^*)\}=0.
\]

The optimiser systematically selects the region on which the predictor fails. This construction disproves an implication; it is not an empirical claim about the GP implemented here. It also does not show that split conformal always fails: a particular calibration procedure may return wider intervals. It shows why reporting random-test coverage cannot alone establish selected-design coverage.

For a fixed candidate set D of size M, suppose an actual theorem establishes \(\Pr(E_x)\le\alpha/M\) **for every candidate x**, where E_x is its prediction-error event under the relevant experiment. Then \(\Pr(\bigcup_{x\in D}E_x)\le\alpha\). On the simultaneous good event, any data-dependent selector in D is covered. The union bound needs no independence between candidates. Ordinary marginal conformal coverage over a heterogeneous input distribution does not supply its premise. The bound also needs adjustment if the candidate set is chosen using the same outcomes in a way not covered by the theorem.

## 2. A predictive rank guarantee and a tolerance guarantee differ

Fix a model using an independent training sample. For n exchangeable calibration scores \(R_i=|Y_i-\widehat\mu(X_i)|/\widehat s(X_i)\), with positive scale, use

\[
k=\lceil(n+1)(1-\alpha)\rceil,\qquad
q=R_{(k)}\quad\text{if }k\le n;\quad q=+\infty\text{ otherwise}.
\]

Exchangeability makes the next score's rank uniform in the no-ties case; ties make this upper interval conservative. Thus the interval \(\widehat\mu(x)\pm q\widehat s(x)\) has marginal coverage at least \(1-\alpha\). This rank argument is the basis of the [conformal quantile lemma and split construction](https://arxiv.org/abs/1904.06019). It averages over calibration and the new point; it does not promise a particular realised calibration set attains conditional coverage \(1-\alpha\). For non-IID chronological lab data, even marginal exchangeability is unverified.

For comparison, assume scores are IID with a continuous CDF F and choose a fixed order k. Let p be a desired probability mass and delta a confidence-failure level. The probability that the calibrated set has too little mass is

\[
\Pr_{R_{1:n}}\{F(R_{(k)})<p\}
=\Pr\{\operatorname{Binomial}(n,p)\ge k\}
=\sum_{j=k}^{n}{n\choose j}p^j(1-p)^{n-j}.
\]

To see this, set \(t=F^{-1}(p)\). The kth score lies below t exactly when at least k of the n independent scores lie below t. Choose k so that this binomial tail is at most delta. If no k in 1,...,n works, a finite empirical order statistic cannot certify the requested p,delta pair. The order-statistic/binomial argument is also used in [Alexeenko and Bitar, section III](https://arxiv.org/abs/2004.03069); it requires calibration independent of the selected set shape/model.

For k=n, p=0.95 and delta=0.05, the condition is \(0.95^n\le0.05\), requiring n>=59. By contrast, 95% split-conformal prediction first admits a finite empirical order statistic at n=19. These statements answer different probability questions. The sample counts count independent units, not correlated repeated measurements from one specimen.

## 3. The implemented control certificate is deterministic and conditional

The executable model in `src/crfml/control.py` is

\[
x_{t+1}=a x_t+b u_t+w_t,\quad 0\le a<1,\quad
0\le u_t\le u_{\max},\quad b\in[\underline b_t,\overline b_t],\quad |w_t|\le\bar w.
\]

It assumes exact state measurements and a fixed unknown b. The constraint is the **upper state limit only**, \(x_t\le x_{\max}\); no lower state limit or chemical validity is proved. Because all input and propagation coefficients are nonnegative, a horizon-h upper bound is

\[
x_{t+h}\le a^h x_t+
\overline b_t\sum_{j=0}^{h-1}a^{h-1-j}u_{t+j}
+\bar w\sum_{j=0}^{h-1}a^j.
\]

The bound follows by unrolling the recurrence and maximising every nonnegative disturbance/input-gain contribution. Enforcing it below xmax yields linear inequalities in the input sequence. With quadratic tracking, effort and slew cost, the planning problem is a convex QP. Learning changes the nominal cost model and can shrink the gain interval, but safety comes from the interval containing the true gain and the disturbance bound, not from the GP posterior.

For u_t>0, the observed transition implies

\[
b\in\left[\frac{x_{t+1}-a x_t-\bar w}{u_t},
\frac{x_{t+1}-a x_t+\bar w}{u_t}\right].
\]

Intersecting this interval with the previous gain set retains the true b by induction when assumptions hold. A zero input carries no gain information; it only permits checking whether the residual lies within the disturbance bound. An empty intersection detects inconsistency. A nonempty intersection does **not** prove model correctness: some violations remain compatible with a different gain.

## 4. Backup invariance and what it does not prove

If \(a x_{\max}+\bar w\le x_{\max}\), then u=0 maps every x<=xmax back into that upper-safe half-line for every admissible disturbance. Consequently zero input is a feasible backup plan at every horizon from an upper-safe state. Executing a feasible first robust action preserves the condition needed to plan again. The same implication survives gain-set contraction, provided the true fixed b remains inside it. Numerical feasibility must be checked after the solver; solver success alone is insufficient.

This proves conditional recursive feasibility of the upper constraint for this scalar model and backup, not convergence to the tracking reference, asymptotic stability of a general plant, or safety after an out-of-envelope disturbance. An emergency zero input after inconsistency is a useful response but loses its certificate if the disturbance/model envelope is false. These limits should appear alongside simulation counts.

## 5. Statistical calibration is optional evidence, not an implicit hard bound

If a residual quantile estimates \(\bar w\), it represents a probability set rather than a bound that holds for every future time. For a horizon H, obtaining per-time conditional failure probabilities at most alpha/H would allow a union bound of alpha over the trajectory; marginal IID calibration alone does not establish that premise for adaptively controlled residuals. Alternatively calibrate a trajectory-level maximum score using independent trajectories drawn under the relevant policy, and state the resulting distributional scope. The current synthetic generator may supply a hard bound by construction, which must be identified as a simulator assumption.

Weighted covariate-shift calibration requires extra information and an unchanged response law; bounded-set robustness does not automatically provide a Wasserstein ambiguity guarantee. [Yan, Parise and Bitar](https://arxiv.org/abs/2205.03748) explicitly model information about distribution distances. A simulated shift stress test is diagnostic evidence, not proof that this information is available in the lab.
