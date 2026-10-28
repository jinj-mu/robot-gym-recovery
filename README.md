# Robot Gym Recovery

Research code and evidence for **Reliable Recovery Around a Frozen Robot Policy: A Case Study in Missed Grasps and Configuration Preservation**, by [Jinjin Zhao](https://www.linkedin.com/in/jinjzhao) and [James Thompson](https://www.linkedin.com/in/astrojams1).

[Read the paper (PDF)](papers/reliable_recovery/paper.pdf) · [LaTeX source](papers/reliable_recovery/main.tex) · [Eight successful recoveries (video)](demo/eight_successes.mp4)

This project explores missed-grasp detection and retries around a frozen π₀ robot policy, together with offline experiment snapshots that preserve the previously selected configuration when an update is interrupted.

## Results and scope

- Local baseline: **83/100** rollout successes.
- Missed-grasp detection: **14/14** selected failures; **1/21** false alarms on success-labeled replays.
- Initial complete recovery sweep: **8/14**.
- Best results across multiple controller versions: **12/14**. This is retrospective coverage, not one controller's verified success rate.
- The mixed-version gallery contains **8/12** successes; four previously recoverable cases failed in later renders.
- Snapshot software: **8 tests**, including forced process exits at four publication checkpoints.

This is an exploratory case study. Scene initialization, unequal action budgets, adaptive tuning, and missing historical controller snapshots limit the conclusions. The current controller does not reproduce all historical successes. The paper has **not been submitted to or accepted by arXiv**.

## Reproduce the artifact checks offline

Python 3.11+ is recommended; these checks use only the standard library:

```bash
python3 -m unittest discover -s papers/reliable_recovery/tests -v
python3 papers/reliable_recovery/audit_results.py
python3 papers/reliable_recovery/bundle.py outputs/libero_inference/paper_oracle_v1/paper_evidence_snapshots
```

The audit reconstructs the reported counts from the included JSON records, verifies assertions, and creates a checksummed local snapshot. It does not simulate the robot. The snapshot tests verify artifact integrity and process-interruption behavior, not deterministic robot execution or power-loss durability.

## Repository contents

- `scripts/lerobot_libero_deployable_critic.py`: camera/proprioception missed-grasp monitor.
- `scripts/lerobot_libero_deployable_harness.py`: retry and policy handback wrapper; includes experimental placement logic with a fixed basket prior.
- `scripts/lerobot_libero_stress_miss_pick.py`: replay detection and recovery evaluation.
- `scripts/lerobot_libero_grasp_retry.py`: privileged geometry controller, used by labeling/scoring and separate oracle experiments.
- `papers/reliable_recovery/`: manuscript, evidence audit, snapshot implementation, and tests.
- `outputs/libero_inference/paper_oracle_v1/`: selected experiment summaries and baseline trajectories. Paths were normalized relative to the repository; private local paths and unused video references do not supply additional files.
- `robot_gym/`: earlier MuJoCo toy environments and training utilities, distinct from the paper's LIBERO study.
- `demo/`: a compilation of eight observed successful recoveries. It is a qualitative demonstration, not a controlled evaluation.

## Robot simulation prerequisites

Robot runs require LeRobot, LIBERO, MuJoCo, PyTorch, compatible dependencies, simulator assets, and access to the pretrained model. They are not included in this repository. The root `pyproject.toml` describes the earlier toy environments, not the full LIBERO runtime.

The original local setup used Apple MPS and `lerobot/pi0_libero_finetuned`. For reference, the local checkouts at release time were:

- LeRobot: `128d3324e3202ce1fca1340fb8d7941edecce9d3`
- LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`

These revisions document the current workspace; they do not certify the exact dependencies of every historical run. `scripts/lerobot_libero_mac_setup.sh` is an experimental setup helper that installs dependencies and writes LIBERO configuration. Review it before running. It is not a locked environment installer.

With those dependencies and assets configured, run from the repository root, use a new output directory, and preserve the original evidence:

```bash
export PYTHONPATH="$PWD/third_party/LIBERO:${PYTHONPATH:-}"
export LIBERO_DIR="$PWD/third_party/LIBERO"
export MUJOCO_GL=cgl
third_party/lerobot/.venv/bin/python scripts/lerobot_libero_stress_miss_pick.py \
  --mode detect --output-dir outputs/reproduction_detect
```

Full robot simulation has not been rerun for this public release. CI checks the offline artifact tests, recorded-result audit, and Python syntax only.

## Build the paper

```bash
cd papers/reliable_recovery
pdflatex main.tex
pdflatex main.tex
```

Alternatively, use `tectonic main.tex`. References are embedded in the source. The authors' names link to their LinkedIn profiles.
