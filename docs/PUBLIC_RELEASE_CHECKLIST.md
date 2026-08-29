# Public release checklist

Run this checklist from the repository root before a paper-facing release.

## Code and reproducibility

- [ ] Create a clean environment and run `pip install -e '.[dev]'`.
- [ ] For CLIP reproduction, run `pip install -e '.[clip]'` and keep the pinned OpenAI/CLIP revision.
- [ ] Run `pytest` successfully.
- [ ] Run one smoke evaluation for H0, Current, and Persistent with non-sensitive fixtures.
- [ ] Confirm `scripts/evaluate.py` writes strict provenance fields into `ranks.npz`.
- [ ] Run `PYTHONPATH=src python scripts/audit_rank_archives.py runs_final outputs` on every local paper-facing archive collection.
- [ ] Run `scripts/compare_runs_strict.py` (or the guarded canonical comparator) on aligned archives.
- [ ] Confirm a cross-space comparison is rejected before paired statistics.
- [ ] Confirm Current/Persistent evaluation metadata records the declared adapter/model revision.
- [ ] Confirm Holm correction is structurally defined on feedback-conditioned turns `1..T-1` rather than selected from observed results.
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

- [x] Release the repository code under the MIT License in `LICENSE`.
- [ ] Add authors, paper title, venue/DOI, and citation metadata after they are final.
- [ ] Run a final repository-wide search for legacy method names and machine-specific paths.
- [ ] Review `git status`, tests, and the release commit/tag before publication.
