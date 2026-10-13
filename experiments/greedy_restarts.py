"""
Sensitivity of greedy_swap to the number of seed pairs R_s, against the certified optima of
experiments/selection_optimality.py (same instances: same seed, population, pools and state evolution).

    python experiments/greedy_restarts.py --rounds 200 --rs 3 10 20

The instance options must match the run that produced the optima (defaults do); the state evolution is driven by
greedy_swap with its class defaults, exactly as in that run.
"""
import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "pyipv8"), os.path.join(ROOT, "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import selection_optimality as so  # noqa: E402
from exact_vs_scip import _so_args  # noqa: E402
from dlsim.compass.solvers import Budget  # noqa: E402
from dlsim.compass.solvers.greedy_swap import GreedySwapSolver  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--rs", type=int, nargs="+", default=[3, 10, 20])
    parser.add_argument("--i-max", type=int, default=50)
    parser.add_argument("--optima", default=os.path.join(ROOT, "experiments", "results", "selection_optimality",
                                                         "results.csv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "greedy_restarts"))
    args, rest = parser.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    optimum = {}
    with open(args.optima) as handle:
        for row in csv.DictReader(handle):
            if row["solver"] == "exact":
                if row["certified"] != "True":
                    sys.exit("Round %s has no certified optimum" % row["round"])
                optimum[int(row["round"])] = float(row["utility"])

    so_args = _so_args(args.rounds, rest)
    pop = so.build_population(so_args)
    params = so.make_params(so_args, pop)
    instances = so.generate_instances(so_args, pop, params)

    rows = []
    for inst in instances:
        problem = so.to_problem(inst, so_args.n, params)
        for rs in args.rs:
            start = time.perf_counter()
            result = GreedySwapSolver(R_s=rs, I_max=args.i_max).solve(problem, Budget(), np.random.default_rng(0))
            elapsed = time.perf_counter() - start
            gap = max(0.0, 1.0 - result.utility / optimum[inst["round"]])
            rows.append({"round": inst["round"], "R_s": rs, "utility": result.utility,
                         "optimum": optimum[inst["round"]], "gap_pct": 100 * gap, "time": elapsed,
                         "evaluations": result.evaluations})
        if inst["round"] % 50 == 0:
            print("%d/%d rounds" % (inst["round"], len(instances)), flush=True)

    with open(os.path.join(args.out, "results.csv"), "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    print("\nR_s | mean gap% | p95 gap% | max gap% | optimal rounds | mean time s | max time s | mean evals")
    for rs in args.rs:
        sel = [r for r in rows if r["R_s"] == rs]
        g = np.array([r["gap_pct"] for r in sel])
        t = np.array([r["time"] for r in sel])
        entry = {"R_s": rs, "rounds": len(sel), "mean_gap_pct": g.mean(), "p95_gap_pct": float(np.percentile(g, 95)),
                 "max_gap_pct": g.max(), "optimal_rounds": int((g < 1e-6).sum()), "mean_time_s": t.mean(),
                 "max_time_s": t.max(), "mean_evaluations": float(np.mean([r["evaluations"] for r in sel]))}
        summary.append(entry)
        print("%3d | %9.4f | %8.4f | %8.4f | %8d/%d | %11.4f | %10.4f | %10d" % (
            rs, entry["mean_gap_pct"], entry["p95_gap_pct"], entry["max_gap_pct"], entry["optimal_rounds"],
            len(sel), entry["mean_time_s"], entry["max_time_s"], entry["mean_evaluations"]))
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.boxplot([[r["gap_pct"] for r in rows if r["R_s"] == rs] for rs in args.rs],
                 labels=["R_s=%d" % rs for rs in args.rs])
    left.set_ylabel("gap to the certified optimum (%)")
    left.set_title("greedy_swap: optimality gap (%d rounds)" % len(instances))
    right.bar(["R_s=%d" % rs for rs in args.rs], [s["mean_time_s"] for s in summary], color="tab:blue", alpha=0.7)
    right.set_ylabel("mean time per solve (s)")
    twin = right.twinx()
    twin.plot(["R_s=%d" % rs for rs in args.rs], [100 * s["optimal_rounds"] / len(instances) for s in summary],
              "o-", color="tab:red")
    twin.set_ylabel("rounds solved to optimality (%)", color="tab:red")
    right.set_title("time and optimal rounds")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "greedy_restarts.png"), dpi=160)
    print("\nOutputs in %s" % args.out)


if __name__ == "__main__":
    main()
