Reliable Recovery Around a Frozen Robot Policy
============================================
Authors: Jinjin Zhao and James Thompson.
Author links: https://www.linkedin.com/in/jinjzhao and https://www.linkedin.com/in/astrojams1
Draft date: October 28, 2026.

Build the paper:
  pdflatex main.tex
  pdflatex main.tex
or:
  tectonic main.tex

Run the artifact tests (standard library only, no model/network required):
  python3 -m unittest discover -s tests -v

Rebuild the evidence audit from the original project checkout:
  python3 papers/reliable_recovery/audit_results.py
The audit command requires local outputs/libero_inference/paper_oracle_v1.
It validates the manuscript's counts and publishes an offline snapshot of
selected saved evidence and current scripts. It does not run robot episodes.

Verify the current local snapshot:
  python3 papers/reliable_recovery/bundle.py outputs/libero_inference/paper_oracle_v1/paper_evidence_snapshots

The manuscript distinguishes:
- Original local baseline: 83/100 rollouts.
- Initial recovery sweep: 8/14 selected missed-grasp episodes.
- Cross-version union: 12/14; not one controller's success rate.
- Mixed-version gallery: 8/12 success records.
- Snapshot software: 8 tests including 4 hard process-exit checkpoints.

The preserved snapshot is integrity-verified, not behavior-certified.
It contains current source plus historical results, not complete historical
controller versions, model weights, simulator assets, or dependency locks.
A fixed-state constructor means different seed labels do not establish
different LIBERO initial scene configurations. The manuscript states this.

Submission preparation:
- Review author spelling, manuscript claims, and desired public artifact release.
- Select the appropriate arXiv category and license in the submission interface.
- No arXiv submission or public release has been made by this task.
- The arxiv_source.tar.gz archive contains only main.tex, with embedded references.
- The companion_source.zip contains the manuscript and audit/test implementation.

This is a short exploratory technical report, not a benchmark leaderboard claim.
