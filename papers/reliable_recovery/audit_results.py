"""Rebuild manuscript counts and a checksummed offline snapshot from local evidence."""
import hashlib
import json
from pathlib import Path
from bundle import publish

ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
BASE = ROOT / "outputs/libero_inference/paper_oracle_v1"


def main():
    files = [BASE / "baseline_summary.json", BASE / "failure_atlas_gt.json",
             BASE / "miss_pick_stress/detect_results.json", BASE / "recovered_sxs_12/gallery.json"]
    files += sorted(BASE.glob("*/recover_results.json"))
    data = {str(p.relative_to(BASE)): json.loads(p.read_text()) for p in files}
    assert data["baseline_summary.json"]["n"] == 100
    assert data["baseline_summary.json"]["successes"] == 83
    labels = data["failure_atlas_gt.json"]["episodes"]
    seeds = sorted(e["seed"] for e in labels if e["mode_gt"] == "leave_empty")
    run_rows, union = [], set()
    for name, d in data.items():
        if name.endswith("recover_results.json"):
            rows = d["episodes"]
            wins = sorted(e["seed"] for e in rows if e.get("success"))
            union.update(wins)
            run_rows.append({"run": name.split("/")[0], "n": len(rows), "successes": len(wins),
                             "episodes": [{"seed": e["seed"], "success": e.get("success", False)} for e in rows]})
    initial = next(r for r in run_rows if r["run"] == "miss_pick_stress")
    assert initial["successes"] == 8 and initial["n"] == 14
    assert len(union & set(seeds)) == 12
    d = data["miss_pick_stress/detect_results.json"]["summary"]
    assert d["leave_empty"]["TP"] == 14 and d["success_negatives"]["FP"] == 1
    assert d["success_negatives"]["n"] == 21
    gallery = data["recovered_sxs_12/gallery.json"]
    assert sum(e["success"] for e in gallery["episodes"]) == 8
    assert len(gallery["episodes"]) == 12
    audit = {"baseline": {k: data["baseline_summary.json"][k] for k in ("n", "successes", "fails")},
             "detection": d, "miss_pick_seeds": seeds, "runs": run_rows,
             "cross_version_successes": sorted(union & set(seeds)),
             "gallery": [{k: e[k] for k in ("seed", "object", "success")} for e in gallery["episodes"]],
             "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (PAPER / "audit_results.json").write_text(json.dumps(audit, indent=2) + "\n")
    script_names = ["lerobot_libero_collect.py", "lerobot_libero_live.py", "lerobot_libero_gt_label_fails.py",
                    "lerobot_libero_deployable_critic.py", "lerobot_libero_deployable_harness.py",
                    "lerobot_libero_stress_miss_pick.py", "lerobot_libero_hybrid_recover.py"]
    selected = [str(p.relative_to(ROOT)) for p in files]
    selected += ["scripts/" + n for n in script_names]
    selected += ["papers/reliable_recovery/" + n for n in ("main.tex", "README.txt", "test_results.txt", "audit_results.json", "bundle.py", "audit_results.py", "tests/test_bundle.py")]
    dest = publish(BASE / "paper_evidence_snapshots", ROOT, selected,
                   {"status": "integrity-verified evidence snapshot; controller behavior not certified",
                    "policy": "lerobot/pi0_libero_finetuned", "weights_included": False,
                    "historical_source_versions_available": False})
    print("Verified: baseline 83/100; initial recovery 8/14; cross-version union 12/14; gallery 8/12.")
    print("Offline snapshot:", dest)


if __name__ == "__main__":
    main()
