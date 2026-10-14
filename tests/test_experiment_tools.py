"""
The sweep runner and the aggregator (experiments/sweep.py, experiments/aggregate.py).
"""
import json
import math
import os
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import aggregate  # noqa: E402
import sweep  # noqa: E402


def write_spec(tmp_path, **spec):
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    return str(path)


def test_sweep_override_precedence_and_layout(tmp_path):
    spec = sweep.load_sweep(write_spec(tmp_path, name="s", config="configs/smoke_bypass.yaml", common=["run.rounds=5"],
                                       seeds=[1, 2], variants={"a": [], "b": ["run.rounds=7"]}))
    runs = sweep.build_runs(spec, str(tmp_path / "out"), ["run.rounds=9"])
    assert [(r["variant"], r["seed"]) for r in runs] == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]
    b1 = runs[2]["args"]
    # config < common < variant < command line
    assert b1.index("run.rounds=5") < b1.index("run.rounds=7") < b1.index("run.rounds=9")
    assert "seed=1" in b1 and runs[2]["dir"].endswith(os.path.join("s", "b", "seed1"))
    assert spec["reference"] == "a"


def test_sweep_rejects_unknown_reference(tmp_path):
    with pytest.raises(SystemExit):
        sweep.load_sweep(write_spec(tmp_path, name="s", config="configs/smoke_bypass.yaml", reference="x",
                                    variants={"a": []}))


def fake_run(path, accs, step=100.0, N=10):
    os.makedirs(path)
    rounds = []
    for i, acc in enumerate(accs, 1):
        rounds.append({"round": i, "close_time": step * i, "duration": step, "clients": [i % N, (i + 1) % N],
                       "accepted": 2 if i % 2 else 1, "accuracy": acc, "bytes_total": 2 ** 30 * i,
                       "compute_total": 3600.0 * i, "selection": {"components": {"comp": 0.5, "strag": 0.8}}})
    with open(os.path.join(path, "rounds.jsonl"), "w") as handle:
        handle.write("\n".join(json.dumps(r) for r in rounds))
    with open(os.path.join(path, "summary.jsonl"), "w") as handle:
        handle.write(json.dumps({"time": step * len(accs), "traffic_totals": {"model": {"bytes": 90},
                                                                              "probe": {"bytes": 10}}}))
    with open(os.path.join(path, "config.resolved.yaml"), "w") as handle:
        yaml.safe_dump({"network": {"N": N}}, handle)


def test_run_metrics_on_a_synthetic_run(tmp_path):
    fake_run(str(tmp_path / "r"), [0.1, 0.3, 0.5, 0.6])
    m, series = aggregate.run_metrics(str(tmp_path / "r"), target=0.45)
    assert m["tta_h"] == pytest.approx(300 / 3600) and m["cta_gib"] == pytest.approx(3) and m["rta_h"] == 3
    assert m["final_acc"] == pytest.approx((0.3 + 0.5 + 0.6) / 3) and m["best_acc"] == 0.6
    assert m["straggler_pct"] == pytest.approx(100 * (1 - 6 / 8))
    assert m["overhead_pct"] == pytest.approx(10.0)
    assert m["phi_adh"] == 0.8  # legacy "strag" component name
    assert math.isinf(aggregate.run_metrics(str(tmp_path / "r"), target=0.9)[0]["tta_h"])


def test_mean_ci():
    mean, ci, n = aggregate.mean_ci([1.0, 2.0, 3.0])
    assert (mean, n) == (2.0, 3) and ci == pytest.approx(4.303 * 1.0 / math.sqrt(3))
    assert math.isinf(aggregate.mean_ci([1.0, math.inf])[0])


def test_aggregate_outputs(tmp_path):
    root = tmp_path / "sweep"
    for v, accs in (("ref", [0.2, 0.4, 0.5]), ("better", [0.3, 0.5, 0.6])):
        for seed in (1, 2):
            fake_run(str(root / v / ("seed%d" % seed)), [a + 0.01 * seed for a in accs])
    os.makedirs(root / "ref" / "seed3")  # incomplete run: skipped
    (root / "sweep.yaml").write_text(yaml.safe_dump({"variants": {"ref": [], "better": []}, "reference": "ref"}))
    result = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "aggregate.py"), str(root)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = root / "aggregate"
    for name in ("per_run.csv", "summary.csv", "table.md", "heatmap.png", "accuracy_vs_time.png",
                 "accuracy_vs_bytes.png", "accuracy_vs_compute.png"):
        assert (out / name).exists(), name
    assert "ref (ref)" in (out / "table.md").read_text() and "Skipping 1 incomplete" in result.stdout


@pytest.mark.slow
def test_tiny_sweep_end_to_end(tmp_path):
    """Two variants (COMPASS and the Plexus port) x one seed, training bypassed; then aggregate."""
    spec = write_spec(tmp_path, name="tiny", config="configs/smoke_bypass.yaml", common=["run.rounds=2"], seeds=[1],
                      variants={"compass": [], "plexus": ["protocol.selector=plexus"]})
    out = tmp_path / "runs"
    result = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "sweep.py"), spec, "--jobs", "2",
                             "--out", str(out)], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    for variant in ("compass", "plexus"):
        run_dir = out / "tiny" / variant / "seed1"
        rounds = [json.loads(line) for line in open(run_dir / "rounds.jsonl")]
        assert rounds and all("compute_total" in r and "bytes_total" in r for r in rounds)
        assert rounds[-1]["compute_total"] > 0 and rounds[-1]["bytes_total"] > 0
        summary = json.loads(open(run_dir / "summary.jsonl").read().splitlines()[-1])
        assert summary["compute_total"] > 0
    rerun = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "sweep.py"), spec, "--out", str(out)],
                           capture_output=True, text=True, timeout=300)
    assert "2 complete, 0 to run" in rerun.stdout
    agg = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "aggregate.py"), str(out / "tiny")],
                         capture_output=True, text=True, timeout=300)
    assert agg.returncode == 0, agg.stderr
