# Public release checklist

Run this checklist from the repository root before creating the public GitHub repository.

## Code and reproducibility

- [ ] Create a clean environment and run `pip install -e '.[dev]'`.
- [ ] Run `pytest` successfully.
- [ ] Run one small CPU smoke evaluation for H0, H1, and F1 with non-sensitive fixtures.
- [ ] Run `python scripts/compare_runs.py` on aligned rank files and inspect the output JSON.
- [ ] Confirm `configs/f1_frozen.json` exactly matches the paper configuration.
- [ ] Confirm no module loads a target index before ranking.

## Content and privacy

- [ ] Search for private usernames, absolute paths, API keys, tokens, and host names.
- [ ] Keep raw data, checkpoints, vectors, dialogue logs, belief artifacts, and generated traces out of Git.
- [ ] Confirm all source comments and public documentation are English.
- [ ] Check all third-party dependencies, datasets, and checkpoints have compatible distribution terms.

## Paper evidence

- [ ] Archive original `ranks.npz`, `report.json`, and paired comparison JSON files outside the public repository.
- [ ] Regenerate and archive Mistral paired-comparison statistics before claiming its confidence intervals or p-values.
- [ ] Ensure every reported aggregate number is traceable to an archived run report.

## Publication

- [ ] Add an author-approved `LICENSE` file and remove `LICENSE_PENDING.md`.
- [ ] Add authors, paper title, DOI, and citation metadata after they are final.
- [ ] Initialize a new Git repository inside this directory, review `git status`, and make the first commit.
