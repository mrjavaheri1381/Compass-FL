"""
Standard plots from one or more run directories (plotting only; it never runs a simulation).

    python scripts/plot_compass.py runs/compass/seed42 runs/plexus/seed42 --out plots/

Produces, per set of runs:
- accuracy_vs_time.png        test accuracy against simulated time;
- accuracy_vs_round.png       test accuracy against the round index;
- round_duration.png          round duration per round;
- participation.png           participation counts per node (sorted) and the Jain fairness index over time;
- phi_components.png          Phi components and U of the selected group per round (COMPASS-family selectors);
- stragglers.png              on-time / late / missing clients per round, and the censored fraction;
- overhead.png                selection messages and bytes per round (context, probing, proposals);
- estimate_error.png          relative error of the f and B estimates (simulation only).
"""
import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run_dir: str) -> List[Dict]:
    path = os.path.join(run_dir, "rounds.jsonl")
    with open(path) as handle:
        return sorted((json.loads(line) for line in handle if line.strip()), key=lambda r: r["round"])


def label(run_dir: str) -> str:
    parts = os.path.normpath(run_dir).split(os.sep)
    return "/".join(parts[-2:])


def jain(counts) -> float:
    values = list(counts)
    if not values or sum(values) == 0:
        return 0.0
    return sum(values) ** 2 / (len(values) * sum(v * v for v in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = {label(run): load(run) for run in args.runs}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, name), dpi=150)
        plt.close(fig)

    # Accuracy
    for x_key, name in (("close_time", "accuracy_vs_time.png"), ("round", "accuracy_vs_round.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for run, rows in data.items():
            points = [(r[x_key], r["accuracy"]) for r in rows if r.get("accuracy") is not None]
            if points:
                ax.plot(*zip(*points), label=run)
        ax.set_xlabel("simulated time (s)" if x_key == "close_time" else "round")
        ax.set_ylabel("test accuracy")
        ax.legend(fontsize=7)
        save(fig, name)

    # Round duration
    fig, ax = plt.subplots(figsize=(6, 4))
    for run, rows in data.items():
        durations = [(r["round"], r["duration"]) for r in rows if r.get("duration") is not None]
        if durations:
            ax.plot(*zip(*durations), label=run)
    ax.set_xlabel("round")
    ax.set_ylabel("round duration (s)")
    ax.legend(fontsize=7)
    save(fig, "round_duration.png")

    # Participation fairness
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for run, rows in data.items():
        counts: Counter = Counter()
        index = []
        for r in rows:
            counts.update(r.get("clients") or r.get("contributors") or [])
            index.append((r["round"], jain(counts.values())))
        left.plot(sorted(counts.values(), reverse=True), label=run)
        if index:
            right.plot(*zip(*index), label=run)
    left.set_xlabel("node (sorted)")
    left.set_ylabel("rounds participated")
    right.set_xlabel("round")
    right.set_ylabel("Jain index (over participating nodes)")
    left.legend(fontsize=7)
    save(fig, "participation.png")

    # Phi components
    fig, ax = plt.subplots(figsize=(7, 4))
    for run, rows in data.items():
        for key in ("comp", "comm", "coh", "repr", "fair", "strag"):
            points = [(r["round"], r["selection"]["components"][key]) for r in rows
                      if r.get("selection", {}).get("components", {}).get(key) is not None]
            if points:
                ax.plot(*zip(*points), label="%s %s" % (run, key), linewidth=1)
        points = [(r["round"], r["selection"]["utility"]) for r in rows
                  if isinstance(r.get("selection", {}).get("utility"), (int, float))]
        if points:
            ax.plot(*zip(*points), "k--", label="%s U" % run)
    ax.set_xlabel("round")
    ax.set_ylabel("value")
    ax.legend(fontsize=6, ncol=2)
    save(fig, "phi_components.png")

    # Stragglers
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for run, rows in data.items():
        rounds = [r["round"] for r in rows if "on_time" in r]
        if rounds:
            left.plot(rounds, [r["on_time"] for r in rows if "on_time" in r], label="%s on time" % run)
            left.plot(rounds, [r["late"] + r["missing"] for r in rows if "on_time" in r], "--",
                      label="%s late+missing" % run)
            right.plot(rounds, [r.get("censored_fraction", 0) for r in rows if "on_time" in r], label=run)
    left.set_xlabel("round")
    left.set_ylabel("clients")
    right.set_xlabel("round")
    right.set_ylabel("censored bandwidth samples")
    left.legend(fontsize=7)
    save(fig, "stragglers.png")

    # Overhead
    fig, ax = plt.subplots(figsize=(6, 4))
    for run, rows in data.items():
        points = [(r["round"], sum(r["traffic"][c]["bytes"] for c in ("ctx", "probe", "propose")))
                  for r in rows if "traffic" in r]
        if points:
            ax.plot(*zip(*points), label=run)
    ax.set_xlabel("round")
    ax.set_ylabel("selection bytes (context + probes + proposals)")
    ax.legend(fontsize=7)
    save(fig, "overhead.png")

    # Estimate error
    fig, ax = plt.subplots(figsize=(6, 4))
    for run, rows in data.items():
        for key in ("f_rel_err_measured", "B_rel_err_measured"):
            points = [(r["round"], r["estimate_error"][key]) for r in rows
                      if r.get("estimate_error", {}).get(key) is not None]
            if points:
                ax.plot(*zip(*points), label="%s %s" % (run, key))
    ax.set_xlabel("round")
    ax.set_ylabel("mean relative error")
    ax.legend(fontsize=7)
    save(fig, "estimate_error.png")
    print("Plots written to %s" % args.out)


if __name__ == "__main__":
    main()
