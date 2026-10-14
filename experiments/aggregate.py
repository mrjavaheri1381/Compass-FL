"""
Aggregate a sweep (experiments/sweep.py) across seeds: per-run metrics, mean and 95% confidence interval per variant,
time / communication / resources to a target accuracy (TTA, CTA, RTA, as in the Plexus paper), a comparison table, a
heatmap relative to the reference variant, and accuracy curves against time, bytes and training compute.

    python experiments/aggregate.py runs/sweeps/weights
    python experiments/aggregate.py runs/sweeps/weights --target 0.55          # absolute target accuracy
    python experiments/aggregate.py runs/sweeps/weights --target-frac 0.9      # default: 0.9 x lowest variant mean

Outputs in <sweep dir>/aggregate/: per_run.csv, summary.csv, table.md, heatmap.png, accuracy_vs_{time,bytes,compute}.png.
Runs without summary.jsonl (incomplete) are skipped and listed.

The target accuracy (when not given) is target_frac times the lowest mean final accuracy among the variants, so that
every variant is expected to reach it; a run that never reaches it gets TTA = CTA = RTA = inf and is counted.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

# Two-sided 95% Student t quantiles by degrees of freedom (1..30); larger df use 1.96
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
       2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]

# (key, label, lower is better) -- the order of the table and the heatmap
METRICS = [
    ("final_acc", "final accuracy", False),
    ("best_acc", "best accuracy", False),
    ("tta_h", "TTA [h]", True),
    ("cta_gib", "CTA [GiB]", True),
    ("rta_h", "RTA [device-h]", True),
    ("round_mean_s", "mean round [s]", True),
    ("round_p90_s", "p90 round [s]", True),
    ("rounds", "rounds completed", False),
    ("straggler_pct", "stragglers [%]", True),
    ("updates_per_round", "updates / round", False),
    ("jain", "Jain index", False),
    ("coverage", "coverage", False),
    ("overhead_pct", "selection overhead [%]", True),
    ("phi_comp", "Phi_comp", False),
    ("phi_comm", "Phi_comm", False),
    ("phi_coh", "Phi_coh", False),
    ("phi_repr", "Phi_repr", False),
    ("phi_fair", "Phi_fair", False),
    ("phi_adh", "Phi_adh", False),
]
GIB = 2.0 ** 30


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run(run_dir):
    rounds = sorted(read_jsonl(os.path.join(run_dir, "rounds.jsonl")), key=lambda r: r["round"])
    summary = (read_jsonl(os.path.join(run_dir, "summary.jsonl")) or [{}])[-1]
    with open(os.path.join(run_dir, "config.resolved.yaml")) as handle:
        config = yaml.safe_load(handle)
    return rounds, summary, config


def cumulative(rounds, key, fallback):
    """Cumulative series of ``key`` (logged since the compute/bytes accounting) or of the per-round fallback."""
    if rounds and all(key in r for r in rounds):
        return [float(r[key]) for r in rounds]
    total, out = 0.0, []
    for r in rounds:
        total += fallback(r)
        out.append(total)
    return out


def curve(rounds, x_values):
    points = [(x, r["accuracy"]) for x, r in zip(x_values, rounds) if r.get("accuracy") is not None]
    return points


def run_metrics(run_dir, target):
    rounds, summary, config = load_run(run_dir)
    N = config["network"]["N"]
    times = [float(r["close_time"]) for r in rounds]
    bytes_total = cumulative(rounds, "bytes_total",
                             lambda r: sum(c.get("bytes", 0) for c in r.get("traffic", {}).values()))
    compute = cumulative(rounds, "compute_total", lambda r: 0.0)
    accs = [p[1] for p in curve(rounds, times)]
    m = {"rounds": len(rounds), "sim_time_h": float(summary.get("time", times[-1] if times else 0)) / 3600}
    m["final_acc"] = float(np.mean(accs[-3:])) if accs else math.nan
    m["best_acc"] = float(max(accs)) if accs else math.nan
    m["tta_h"] = m["cta_gib"] = m["rta_h"] = math.inf
    if target is not None:
        for i, r in enumerate(rounds):
            if r.get("accuracy") is not None and r["accuracy"] >= target:
                m["tta_h"], m["cta_gib"], m["rta_h"] = times[i] / 3600, bytes_total[i] / GIB, compute[i] / 3600
                break
    durations = [float(r["duration"]) for r in rounds if r.get("duration") is not None]
    m["round_mean_s"] = float(np.mean(durations)) if durations else math.nan
    m["round_p90_s"] = float(np.percentile(durations, 90)) if durations else math.nan
    selected = aggregated = 0
    counts = np.zeros(N)
    for r in rounds:
        clients = r.get("clients") or []
        for c in clients:
            if 0 <= int(c) < N:
                counts[int(c)] += 1
        selected += len(clients)
        if "accepted" in r:
            aggregated += r["accepted"]
        elif "contributors" in r:
            aggregated += len(r["contributors"])
    m["straggler_pct"] = 100.0 * (1 - aggregated / selected) if selected else math.nan
    m["updates_per_round"] = aggregated / len(rounds) if rounds else math.nan
    m["jain"] = float(counts.sum() ** 2 / (N * (counts ** 2).sum())) if counts.sum() > 0 else math.nan
    m["coverage"] = float((counts > 0).mean())
    traffic = summary.get("traffic_totals", {})
    total = sum(v.get("bytes", 0) for v in traffic.values())
    model = traffic.get("model", {}).get("bytes", 0) + traffic.get("handover", {}).get("bytes", 0)
    m["overhead_pct"] = 100.0 * (total - model) / total if total and "model" in traffic else math.nan
    for name in ("comp", "comm", "coh", "repr", "fair", "adh"):
        values = []
        for r in rounds:
            comps = r.get("selection", {}).get("components") or {}
            value = comps.get(name, comps.get("strag") if name == "adh" else None)
            if value is not None:
                values.append(value)
        m["phi_" + name] = float(np.mean(values)) if values else math.nan
    series = {"time_h": [t / 3600 for t in times], "bytes_gib": [b / GIB for b in bytes_total],
              "compute_h": [c / 3600 for c in compute], "acc": [r.get("accuracy") for r in rounds]}
    return m, series


def mean_ci(values):
    values = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if len(values) == 0:
        return math.nan, math.nan, 0
    if np.isinf(values).any():
        return math.inf, math.nan, len(values)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, math.nan, 1
    t = T95[len(values) - 2] if len(values) - 1 <= len(T95) else 1.96
    return mean, float(t * values.std(ddof=1) / math.sqrt(len(values))), len(values)


def fmt(mean, ci):
    if math.isinf(mean):
        return "not reached"
    if math.isnan(mean):
        return "-"
    digits = 3 if abs(mean) < 10 else 1
    return ("%.*f" % (digits, mean)) + ("" if math.isnan(ci) else " ± %.*f" % (digits, ci))


def step_curve(xs, accs, grid):
    """Accuracy known at each grid point: the last evaluation at or before it (nan before the first)."""
    pts = [(x, a) for x, a in zip(xs, accs) if a is not None]
    out, j, last = [], 0, math.nan
    for g in grid:
        while j < len(pts) and pts[j][0] <= g:
            last = pts[j][1]
            j += 1
        out.append(last)
    return np.array(out)


def plot_curves(series_by_variant, out_dir, reference):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, xlabel, name, log in (("time_h", "simulated time [h]", "accuracy_vs_time.png", False),
                                   ("bytes_gib", "communication volume [GiB]", "accuracy_vs_bytes.png", True),
                                   ("compute_h", "training resource usage [device-h]", "accuracy_vs_compute.png",
                                    True)):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for variant, runs in series_by_variant.items():
            ends = [s[key][-1] for s in runs if s[key]]
            if not ends or max(ends) <= 0:
                continue
            lo = min(s[key][0] for s in runs if s[key])
            grid = np.geomspace(max(lo, 1e-6), min(ends), 200) if log else np.linspace(0, min(ends), 200)
            ys = np.array([step_curve(s[key], s["acc"], grid) for s in runs])
            mean = np.nanmean(ys, axis=0)
            ax.plot(grid, mean, label=variant, linewidth=2 if variant == reference else 1.2)
            if len(runs) > 1:
                ci = 1.96 * np.nanstd(ys, axis=0, ddof=1) / math.sqrt(len(runs))
                ax.fill_between(grid, mean - ci, mean + ci, alpha=0.15)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test accuracy")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name), dpi=160)
        plt.close(fig)


def plot_heatmap(summary, variants, reference, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [(k, label, lower) for k, label, lower in METRICS
            if math.isfinite(summary[reference].get(k, (math.nan,))[0]) and summary[reference][k][0] != 0]
    others = [v for v in variants if v != reference]
    if not keys or not others:
        return
    rel = np.full((len(others), len(keys)), np.nan)
    for i, v in enumerate(others):
        for j, (k, _, lower) in enumerate(keys):
            value = summary[v][k][0]
            if math.isfinite(value):
                change = 100.0 * (value - summary[reference][k][0]) / abs(summary[reference][k][0])
                rel[i, j] = -change if lower else change   # positive = better
    fig, ax = plt.subplots(figsize=(1.0 + 0.65 * len(keys), 1.2 + 0.45 * len(others)))
    bound = np.nanmax(np.abs(rel)) if np.isfinite(rel).any() else 1.0
    image = ax.imshow(rel, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([label for _, label, _ in keys], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels(others, fontsize=8)
    for i in range(len(others)):
        for j in range(len(keys)):
            if np.isfinite(rel[i, j]):
                ax.text(j, i, "%+.0f" % rel[i, j], ha="center", va="center", fontsize=6)
    ax.set_title("change vs %s [%%] (green = better)" % reference, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmap.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir")
    parser.add_argument("--target", type=float, default=None, help="absolute target accuracy for TTA/CTA/RTA")
    parser.add_argument("--target-frac", type=float, default=0.9)
    parser.add_argument("--reference", default=None, help="default: the sweep's reference variant")
    args = parser.parse_args()

    spec_path = os.path.join(args.sweep_dir, "sweep.yaml")
    spec = yaml.safe_load(open(spec_path)) if os.path.exists(spec_path) else {}
    variants = [v for v in (spec.get("variants") or sorted(os.listdir(args.sweep_dir)))
                if os.path.isdir(os.path.join(args.sweep_dir, v)) and v != "aggregate"]
    reference = args.reference or spec.get("reference") or variants[0]
    runs, skipped = defaultdict(list), []
    for v in variants:
        for name in sorted(os.listdir(os.path.join(args.sweep_dir, v))):
            path = os.path.join(args.sweep_dir, v, name)
            if os.path.isdir(path) and name.startswith("seed"):
                (runs[v] if os.path.exists(os.path.join(path, "summary.jsonl")) else skipped).append(path)
    if skipped:
        print("Skipping %d incomplete runs: %s" % (len(skipped), ", ".join(skipped)))
    variants = [v for v in variants if runs[v]]
    if not variants:
        sys.exit("No complete runs in %s" % args.sweep_dir)
    if reference not in variants:
        reference = variants[0]

    target = args.target
    if target is None:
        finals = [np.nanmean([run_metrics(p, None)[0]["final_acc"] for p in runs[v]]) for v in variants]
        finals = [f for f in finals if not math.isnan(f)]
        target = args.target_frac * min(finals) if finals else None
    print("Target accuracy for TTA/CTA/RTA: %s" % ("none (no accuracy logged)" if target is None else "%.4f" % target))

    out_dir = os.path.join(args.sweep_dir, "aggregate")
    os.makedirs(out_dir, exist_ok=True)
    per_run, series, summary = [], defaultdict(list), {}
    for v in variants:
        metrics = []
        for path in runs[v]:
            m, s = run_metrics(path, target)
            metrics.append(m)
            series[v].append(s)
            per_run.append({"variant": v, "run": os.path.basename(path), **m})
        summary[v] = {k: mean_ci([m[k] for m in metrics]) for k, _, _ in METRICS}
        summary[v]["not_reached"] = sum(math.isinf(m["tta_h"]) for m in metrics)
        summary[v]["seeds"] = len(metrics)

    with open(os.path.join(out_dir, "per_run.csv"), "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)
    with open(os.path.join(out_dir, "summary.csv"), "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "metric", "mean", "ci95", "n", "ratio_vs_reference"])
        for v in variants:
            for k, _, _ in METRICS:
                mean, ci, n = summary[v][k]
                ref = summary[reference][k][0]
                ratio = mean / ref if ref and math.isfinite(ref) and math.isfinite(mean) else math.nan
                writer.writerow([v, k, mean, ci, n, ratio])
    shown = [(k, label) for k, label, _ in METRICS
             if any(math.isfinite(summary[v][k][0]) or math.isinf(summary[v][k][0]) for v in variants)]
    with open(os.path.join(out_dir, "table.md"), "w") as handle:
        handle.write("Target accuracy: %s. Mean ± 95%% CI over seeds.\n\n" % (
            "none" if target is None else "%.4f" % target))
        handle.write("| variant | seeds | " + " | ".join(label for _, label in shown) + " |\n")
        handle.write("|---" * (len(shown) + 2) + "|\n")
        for v in variants:
            cells = [fmt(*summary[v][k][:2]) for k, _ in shown]
            mark = " (ref)" if v == reference else ""
            reached = "" if not summary[v]["not_reached"] else " (%d not reached)" % summary[v]["not_reached"]
            handle.write("| %s%s | %d%s | %s |\n" % (v, mark, summary[v]["seeds"], reached, " | ".join(cells)))
    plot_heatmap(summary, variants, reference, out_dir)
    if any(a is not None for runs_ in series.values() for s in runs_ for a in s["acc"]):
        plot_curves(series, out_dir, reference)
    print(open(os.path.join(out_dir, "table.md")).read())
    print("Outputs in %s" % out_dir)


if __name__ == "__main__":
    main()
