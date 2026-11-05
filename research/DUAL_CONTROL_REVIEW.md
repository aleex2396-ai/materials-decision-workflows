# Evidence for the two-step dual-control reference

Research date: 2026-11-05. This targeted extension asks when a safe informative action reduces the total cost of two subsequent decisions. It supports the derivation in [DUAL_CONTROL_THEORY.md](../docs/DUAL_CONTROL_THEORY.md), and is not an exhaustive novelty search. All external queries used public methods terms and public identifiers only.

## Primary-source evidence

| Source | What was inspected | Consequence for this project |
|---|---|---|
| Klenske and Hennig (2016), *Dual Control for Approximate Bayesian Reinforcement Learning*. [JMLR article](https://jmlr.org/papers/v17/15-162.html), [full manuscript](https://jmlr.org/papers/volume17/15-162/15-162.pdf) | Article and full PDF, abstract, introduction and Bayesian-control formulation | Actions affect both future states and information; tractability motivates approximations. A belief-state Bellman calculation is established methodology. Our scalar finite-horizon example does not replace their more general approximations. |
| Lee and Lee (2009), *An approximate dynamic programming based approach to dual adaptive control*. [Publisher DOI](https://doi.org/10.1016/j.jprocont.2008.11.009), [OpenAlex W2091755682](https://openalex.org/W2091755682) | Publisher preview and OpenAlex metadata | Their uncertain-parameter hyperstate and sampled ADP provide a precedent for retaining information dynamics. We use an analytically closed interval posterior and a tiny numerical reference instead of claiming tractable general-purpose belief-space control. |
| Arcari, Hewing and Zeilinger (2020), *An Approximate Dynamic Programming Approach for Dual Stochastic Model Predictive Control*. [Primary manuscript](https://arxiv.org/abs/1911.03728), [full PDF](https://arxiv.org/pdf/1911.03728), DOI [10.1016/j.ifacol.2020.12.2280](https://doi.org/10.1016/j.ifacol.2020.12.2280), [W2987231576](https://openalex.org/W2987231576) | Full manuscript | The future learning effect distinguishes dual from certainty-equivalent planning. Finite quadrature and an action grid are numerical approximations, even if the terminal action and posterior are analytic. |
| Fernandez-Canti, Blesa, Puig and Tornil-Sin, *Set-membership identification and fault detection using a Bayesian framework*. [Author-institution record](https://apollo.upc.es/publications/show/1548), DOI [10.1080/00207721.2014.948946](https://doi.org/10.1080/00207721.2014.948946), [W2087732402](https://openalex.org/W2087732402) | Institution publication record/abstract and metadata; not the full proof | Uniform bounded measurement noise and a flat prior connect Bayesian feasible support with set-membership methods. We derive the scalar uniform-posterior closure directly; the source does not establish that arbitrary priors or likelihoods stay uniform. Online metadata is 2014; the journal issue is 2016. |
| Akcay, *The size of membership-set in a probabilistic framework*. DOI [10.1016/j.automatica.2003.10.002](https://doi.org/10.1016/j.automatica.2003.10.002), [W1993241146](https://openalex.org/W1993241146) | Publisher abstract/preview and metadata; not the full theorem proof | Membership-set size under iid bounded noise and excitation has an established literature. We give only an elementary scalar constant-input calculation and do not infer its rates for adaptive multivariable systems. OpenAlex indexes the online date in 2003; the journal issue is February 2004. |
| Fogel and Huang (1982), *On the value of information in system identification—Bounded noise case*. DOI [10.1016/0005-1098(82)90110-8](https://doi.org/10.1016/0005-1098(82)90110-8), [W2061938142](https://openalex.org/W2061938142) | Publisher abstract and metadata | Set intersections, redundant observations and identifiability are established bounded-noise questions. The adversarial two-gain construction in our notes is an elementary illustration, not a claimed new lower bound. |
| Lu, Cannon and Koksal-Rivet, *Robust adaptive MPC: performance and parameter estimation*. [Primary manuscript](https://arxiv.org/abs/1911.00865), [full PDF](https://arxiv.org/pdf/1911.00865), DOI [10.1002/rnc.5175](https://doi.org/10.1002/rnc.5175), [W3081272029](https://openalex.org/W3081272029) | Full manuscript, estimation assumptions and convergence discussion | Persistent excitation and disturbance-bound assumptions matter for parameter-set convergence. Robust containment alone does not imply shrinking uncertainty. Our exact-state scalar model omits their more general robust adaptive MPC machinery. Online 2020; issue 2021. |

## What the reference tests

The implementation uses a fixed positive gain, known scalar state dynamics, exact state observations, an initially uniform gain belief, and iid uniform process disturbances. The likelihood is constant on the set-membership intersection, so the interval endpoints are sufficient statistics and the posterior remains uniform. The terminal Bayes action is a clipped quadratic minimizer. The first stage integrates the exact innovation density and searches a finite safe action grid.

The informative/frozen comparison keeps the horizon and actual state observations equal. The frozen controller chooses its terminal action using the original belief, but its realized conditional cost is integrated under the updated posterior. This avoids a subtle but material error: evaluating two policies under different probability laws. The analytic conditional strong-convexity gap certifies nonnegative information value for each first action and provides a strict-benefit example; it does not establish that a heuristic probing schedule is optimal.

An adversarial-noise construction shows that two gains separated by delta can be masked forever when delta times the maximum input is at most twice the noise bound. This statement quantifies over causal policies and uses matched disturbances inductively. A separate iid-uniform calculation gives expected untruncated interval width `4*wbar/(u*(N+1))` under constant positive excitation. The claims concern different disturbance assumptions and cannot be substituted for one another.

## Numerical and experimental boundaries

- Exact posterior and terminal optimum do not make the first-stage continuous optimization exact. Report grid spacing, quadrature order and refinement effects.
- The support-based cap gives only the declared upper-state constraint under a correct fixed-gain/noise model. It is not a learned probabilistic safety certificate.
- Learning may have no decision value when the gain is known, the first action is zero, or terminal actions remain identically saturated.
- Zero process noise creates a discontinuity between zero and arbitrarily small nonzero excitation; existence of a continuous-action optimizer needs care.
- The uniform belief is an independent synthetic premise. It is not a posterior extracted from confidential lab data, nor a GP posterior approximation whose validity has been proved.
- This is a controlled reference and an ablation, not a novel general method. The broader research contribution must be stated around tested, reproducible findings rather than the presence of established components.

## OpenAlex retrieval provenance

Successful identifier lookups were performed at approximately 2026-11-05 11:50:56–11:50:58 UTC. The endpoint form was `https://api.openalex.org/works/https://doi.org/{DOI}`; no private query terms or API keys were sent. Records resolved as follows:

| Query identifier | Returned work ID | Use |
|---|---|---|
| `10.1016/j.jprocont.2008.11.009` | `W2091755682` | Dual adaptive ADP precedent |
| `10.48550/arXiv.1911.03728` | `W2987231576` | Resolved to journal DOI `10.1016/j.ifacol.2020.12.2280` |
| `10.1080/00207721.2014.948946` | `W2087732402` | Bayesian/set-membership connection |
| `10.1016/j.automatica.2003.10.002` | `W1993241146` | Stochastic membership-set size |
| `10.1016/0005-1098(82)90110-8` | `W2061938142` | Bounded-noise identification |
| `10.1002/rnc.5175` | `W3081272029` | Robust adaptive MPC and estimation |

OpenAlex supplies discovery and bibliographic provenance; scientific statements above are restricted to the primary material actually inspected. The JMLR paper was read directly from the publisher and has no claimed OpenAlex lookup in this extension. Generic public web queries additionally covered `dual adaptive dynamic programming`, `set membership bounded noise consistency`, and `Feldbaum dual control`. GitHub software research for the broader project is documented separately in [GITHUB_REVIEW.md](GITHUB_REVIEW.md); this compact reference copies no third-party implementation.
