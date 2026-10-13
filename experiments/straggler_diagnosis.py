"""
Why do clients become stragglers? Attributes every straggler of a COMPASS-FL run to a cause, using the simulator's
ground truth (which the protocol itself never sees).

    python run.py --config configs/default.yaml data.bypass_training=true run.rounds=60 --run-dir runs/diag
    python experiments/straggler_diagnosis.py runs/diag

Causes, in this order (the first that applies):
  dropout      the client was silent this round (no acknowledgement)
  slowdown     a sudden-slowdown event hit the client this round
  first        first participation: the deadline came from the class prior only (m = 0)
  contention   the upload overlapped other uploads (k > 1) or was censored as congested
  fluctuation  none of the above (per-round fluctuation larger than the z-sigma margin)
For every client it also reports actual / predicted duration from the acknowledgement, and for first-time clients the
ratio between the true and the prior capacity of their class.
"""
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "pyipv8")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from dlsim.compass.config import config_from_dict  # noqa: E402
from dlsim.compass.env.devices import DeviceModel  # noqa: E402


def main(run_dir: str) -> None:
    with open(os.path.join(run_dir, "config.resolved.yaml")) as handle:
        config = config_from_dict(yaml.safe_load(handle))
    with open(os.path.join(run_dir, "meta.json")) as handle:
        meta = json.load(handle)
    rounds = [json.loads(line) for line in open(os.path.join(run_dir, "rounds.jsonl"))]
    devices = DeviceModel(config.env.devices, config.network.N, config.seed)
    priors = {int(k): v for k, v in meta["environment"]["devices"]["priors"].items()}

    seen = set()
    causes = Counter()
    by_class = defaultdict(lambda: [0, 0])
    first_stats = {"first": [0, 0], "repeat": [0, 0]}
    ratios = {"first": [], "repeat": []}
    prior_ratio = defaultdict(list)
    total = 0
    for record in sorted(rounds, key=lambda r: r["round"]):
        t = record["round"]
        for key, detail in record["client_detail"].items():
            node = int(key)
            kappa = devices.devices[node].kappa
            first = node not in seen
            seen.add(node)
            total += 1
            straggler = not detail.get("aggregated")
            group = "first" if first else "repeat"
            first_stats[group][0] += 1
            first_stats[group][1] += straggler
            by_class[kappa][0] += 1
            by_class[kappa][1] += straggler
            if detail.get("t_ack") is not None and detail.get("t_last") is not None and detail.get("raw_deadline"):
                predicted = detail["raw_deadline"] - detail["t_ack"]
                ratios[group].append((detail["t_last"] - detail["t_ack"]) / predicted)
            if first:
                prior_ratio[kappa].append(devices.devices[node].f / priors[kappa]["f"])
            if not straggler:
                continue
            if devices.dropout(node, t) or detail.get("t_ack") is None:
                cause = "dropout"
            elif devices.slowdown(node, t) > 1:
                cause = "slowdown"
            elif first:
                cause = "first"
            elif detail.get("k", 1) > 1 or detail.get("censored"):
                cause = "contention"
            else:
                cause = "fluctuation"
            causes[cause] += 1

    stragglers = sum(causes.values())
    print("Run: %s  (%d rounds, %d client slots, %d stragglers = %.1f%%)" % (
        run_dir, len(rounds), total, stragglers, 100 * stragglers / max(1, total)))
    print("\nCause of each straggler:")
    for cause in ("dropout", "slowdown", "first", "contention", "fluctuation"):
        print("  %-12s %4d  (%5.1f%% of stragglers)" % (cause, causes[cause], 100 * causes[cause] / max(1, stragglers)))
    print("\nStraggler rate by participation:")
    for group, (count, bad) in first_stats.items():
        print("  %-7s %4d clients, %4d stragglers (%.1f%%)" % (group, count, bad, 100 * bad / max(1, count)))
    print("\nStraggler rate by device class (0 = slowest tier):")
    for kappa in sorted(by_class):
        count, bad = by_class[kappa]
        print("  class %d: %4d clients, %4d stragglers (%.1f%%)" % (kappa, count, bad, 100 * bad / max(1, count)))
    print("\nActual / predicted duration from the acknowledgement (> 1 = slower than the deadline's mu + z sigma):")
    for group, values in ratios.items():
        if values:
            q = np.percentile(values, [10, 50, 90, 99])
            print("  %-7s p10 %.2f  median %.2f  p90 %.2f  p99 %.2f  (n=%d)" % (group, *q, len(values)))
    durations = np.array([r["duration"] for r in rounds])
    accepted = np.array([r["accepted"] for r in rounds])
    print("\nRound duration: mean %.1fs, median %.1fs, p90 %.1fs, max %.1fs | aggregated updates per round: mean %.2f"
          % (durations.mean(), np.median(durations), np.percentile(durations, 90), durations.max(), accepted.mean()))
    summary = {"rounds": len(rounds), "slots": total, "straggler_pct": 100 * stragglers / max(1, total),
               "first_straggler_pct": 100 * first_stats["first"][1] / max(1, first_stats["first"][0]),
               "repeat_straggler_pct": 100 * first_stats["repeat"][1] / max(1, first_stats["repeat"][0]),
               "causes": dict(causes), "duration_mean": float(durations.mean()),
               "duration_p90": float(np.percentile(durations, 90)), "duration_max": float(durations.max()),
               "accepted_mean": float(accepted.mean()), "sim_time_total": float(rounds[-1]["close_time"])}
    with open(os.path.join(run_dir, "diagnosis.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print("\nTrue capacity / class prior, first-time clients (spread inside a class):")
    for kappa in sorted(prior_ratio):
        q = np.percentile(prior_ratio[kappa], [10, 50, 90])
        print("  class %d: p10 %.2f  median %.2f  p90 %.2f" % (kappa, *q))


if __name__ == "__main__":
    main(sys.argv[1])
