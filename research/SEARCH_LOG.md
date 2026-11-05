# Public methods search provenance

Date: 2026-11-05. This selected log documents the methodological sources retained in the public repository. It is not a systematic review or a complete history of exploratory searches.

1. Primary-source web queries included `site.proceedings.mlr.press safe exploration optimization gaussian processes Sui 2015` and `site.proceedings.mlr.press conformal prediction distribution shift weighted exchangeability`.
2. Exact OpenAlex DOI/arXiv lookups supplied paper-level records for safe bandits, uncertainty-set estimation, nonstationary chance constraints, Wasserstein DRO, regret control, conformal inference, materials Bayesian optimisation and related software. See `retrieve_openalex.ps1`, the selected `openalex/*.json` snapshots and `bibliography.json`.
3. The public bundle contains the selected paper records used by the methods reviews. Broad exploratory retrievals and unrelated records are excluded; raw search counts are not treated as counts of verified studies.
4. Some anonymous OpenAlex keyword/title requests were rate-limited, including a SafeOpt title lookup. No successful result is claimed for those requests. Exact-identifier lookups supplied the successful paper records, and SafeOpt was verified through its primary publication.
5. Additional primary-source searches covered nonstationary chance constraints, Wasserstein regret-optimal control, nonparametric uncertainty sets, Wasserstein DRO and materials Bayesian optimisation. Source links and inspection depth appear in [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md).
6. GitHub repository queries included `BoTorch`, `safe-control-gym`, `Wasserstein regret optimal control`, `SafeOpt` and `materials bax`. Metadata, README files, licences and relevant source functions were inspected for BoTorch, safe-control-gym and SafeOpt; the BAX README and tree were inspected. Machine-readable findings are in `github_evidence.json`.
7. The materials-BAX README led to the public NIST Fe-Co-Ni benchmark directory. The dataset was located but was not downloaded or used. Dataset-specific provenance and reuse terms require verification before adding it.
8. The decision-focused extension has its own [review and retrieval account](DECISION_FOCUSED_REVIEW.md), including exact-identifier OpenAlex lookups and inspected software commits.

Access limitations: IEEE pages returned robot-check/errors, so author-hosted or arXiv papers were used. The materials BAX publisher page was partly available via search, while a later direct fetch encountered a publisher redirect. Its abstract and author repository establish relevance; full methodological verification of BAX is not claimed.
