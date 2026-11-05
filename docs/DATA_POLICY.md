# Public data and provenance policy

This repository distributes a reusable workflow with independently generated synthetic examples. Its public datasets are not anonymized or pseudonymized private measurements. Removing names, changing column labels or scaling measured values would not satisfy this project's publication boundary.

## Included material

| Material | Public provenance |
|---|---|
| Simulator and test fixtures | Invented constants, artificial records and declared random seeds |
| Learned public models and result files | Fitted only on those independently generated examples |
| Plots and reports | Derived only from the public synthetic runs |
| Literature metadata | Selected public OpenAlex records with retrieval provenance |
| Software references | Public repository evidence and properly attributed primary sources |
| Local-data adapter | Generic contracts and validation code; caller supplies the actual schema |

The synthetic generator does not read private inputs and was not calibrated to confidential observations. Scientific constants are illustrative and are not asserted to reproduce a real material or process.

## Excluded material

Private measurements, database dumps, workbook exports, images, experiment identifiers, file paths, source hashes, access logs, actual extraction schemas and credentials are excluded. So are private-data summaries, plots, fitted weights, embeddings, calibration parameters and synthetic surrogates fitted to confidential observations. A derived artifact can still reveal its source even when the original table is absent.

Local evaluation partitions, their sizes and access histories are not public benchmark evidence. Public claims use the reproducible synthetic studies. The optional adapter writes caller-owned results to a local directory; it is not invoked by the public study commands and does not upload anything.

## Publication controls

1. Start from a fresh allowlisted export, not a copy or clone of a mixed-data working directory. Source Git history is never copied.
2. Require current source/configuration hashes, complete declared result groups and matching artifact hashes. These checks detect stale or inconsistent outputs; they do not prove that an arbitrary relabelled dataset is synthetic.
3. Inspect file contents and figures, including comments, test fixtures, metadata and provenance fields. Automated path/credential checks supplement this review.
4. Inspect the staged/tracked paths before creating a publication commit. The tracked-file guard rejects paths outside the publication profile, including tracked files that an ignore rule would otherwise hide.
5. Keep issue reports, pull requests and CI logs free of private examples. Supply an invented minimal reproduction instead.

The export manifest lists every included file with its byte count and SHA-256 digest. Only those reviewed artifacts and the manifest belong in the initial public commit. No branch, tag, archive, release attachment or Git LFS object from a private working repository is carried over.

Ignore rules help avoid accidental additions, but they do not erase previously committed content. GitHub documents why published sensitive history can remain in clones, forks or cached references in its [removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository). The release process therefore starts with clean history. A repository-local GitHub no-reply author address can keep a personal email out of commit metadata; see [GitHub's commit email guidance](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address).

This is a publication policy and an inspected provenance boundary. It is not a mathematical anonymization guarantee for arbitrary data contributed later. New real public datasets require their own provenance, permitted-use review and explicit publication decision.
