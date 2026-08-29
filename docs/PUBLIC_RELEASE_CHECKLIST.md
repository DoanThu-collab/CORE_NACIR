# Public release checklist

Run this checklist from the repository root before a paper-facing release.

## Code and reproducibility

- [ ] Create a clean environment and run `pip install -e '.[dev]'`.
- [ ] Install OpenAI CLIP when reproducing the CLIP ViT-L/14 condition.
- [ ] Run `pytest` successfully.
- [ ] Run one smoke evaluation for H0, Current, and Persistent with non-sensitive fixtures.
- [ ] Confirm `scripts/evaluate.py` writes strict provenance fields into `ranks.npz`.
- [ ] Run `scripts/compare_runs_strict.py` (or the guarded canonical comparator) on aligned archives.
- [ ] Confirm a cross-space comparison is rejected before paired statistics.
- [ ] Confirm `configs/nacir_minus_frozen.json` matches the frozen paper configuration.
- [ ] Confirm no module uses a target index before scoring/ranking.

## Content and privacy

- [ ] Search for private usernames, absolute paths, API keys, tokens, host names, and environment-specific workspace paths.
- [ ] Keep raw data, checkpoints, vectors, dialogue logs, belief artifacts, and generated traces out of Git.
- [ ] Confirm public documentation uses the canonical names H0 / Current / Persistent NACIR.
- [ ] Check all third-party dependencies, datasets, and checkpoints for compatible distribution terms.

## Paper evidence

- [ ] Preserve the six canonical ChatIR rank archives outside the public repository.
- [ ] Preserve paper-facing cross-host, clean-persistence, text-control, and weighting outputs outside the public repository.
- [ ] Ensure every paired claim is traceable to archives with verified pairing provenance.
- [ ] Ensure every reported aggregate number is traceable to an archived run or analysis report.

## Publication

- [ ] Add an author-approved `LICENSE` file and remove `LICENSE_PENDING.md`.
- [ ] Add authors, paper title, venue/DOI, and citation metadata after they are final.
- [ ] Run a final repository-wide search for legacy method names and machine-specific paths.
- [ ] Review `git status`, tests, and the release commit/tag before publication.
