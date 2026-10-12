"""
Selection-only experiment: how close does each solver get to the true optimum of the COMPASS-FL selection problem?

No training and no network simulation. A population of N clients is built from the same environment as the full
simulator: FedScale device profiles (capacity f), WonderNetwork round-trip times (L), and Dirichlet label
distributions (similarities S and row sums r over the whole population). Every round:

  1. the hash pool of round t gives an aggregator a and M candidate clients;
  2. every solver selects n of the M candidates maximizing U(C, a) with the paper's utility;
  3. ``exact`` (branch-and-bound, warm-started by greedy_swap) gives the optimum, certified, or an upper bound if
     it hits its time limit, so every reported gap is a guaranteed bound;
  4. the state evolves as in the protocol: the clients chosen by the driver solver (greedy_swap, the paper's
     default) get tau_i <- t and a reliability update from a simulated lateness.

All local-search solvers evaluate moves incrementally (``IncrementalState``: a swap or an addition only subtracts
what leaves and adds what enters, O(1) per move). Rounds are independent once generated, so they are solved in
parallel processes, one round per process, each solver single-threaded.

    python experiments/selection_optimality.py --N 1000 --M 50 --n 10 --rounds 200 --workers 4

Outputs in --out: results.csv (one row per round and solver), summary.csv, fitness.png, time.png, meta.json.
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "pyipv8")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from dlsim.compass.config import load_config  # noqa: E402
from dlsim.compass.core import estimators as est  # noqa: E402
from dlsim.compass.core.hashing import rng_seed  # noqa: E402
from dlsim.compass.core.utility import SelectionProblem, UtilityParams, evaluate  # noqa: E402
from dlsim.compass.env.devices import DeviceModel  # noqa: E402
from dlsim.compass.env.latency import LatencyModel  # noqa: E402
from dlsim.compass.solvers import Budget, create_solver  # noqa: E402
from dlsim.compass.solvers.exact import ExactSolver  # noqa: E402

HEURISTICS = ["greedy_swap", "tabu", "simulated_annealing", "genetic", "lns", "random"]


# ---------------------------------------------------------------------- population and instances

def build_population(args):
    config = load_config(overrides=["network.N=%d" % args.N])
    devices = DeviceModel(config.env.devices, args.N, args.seed)
    kappas = [d.kappa for d in devices.devices]
    latency = LatencyModel(config.env.latency, kappas, args.seed)
    # Per-client label skew: p_i ~ Dir(alpha) over the classes and D_i ~ samples / N labels drawn from p_i. (The
    # simulator splits each class over the clients instead; with 1000 clients that leaves many clients almost empty,
    # and this experiment only needs realistic label vectors.)
    rng = np.random.default_rng(rng_seed(args.seed, "labels"))
    sizes = np.maximum(args.min_samples, rng.poisson(args.samples / args.N, args.N))
    counts = np.array([rng.multinomial(size, rng.dirichlet([args.alpha] * args.classes)) for size in sizes],
                      dtype=float)
    parts = [np.arange(int(size)) for size in sizes]
    Y = counts / np.linalg.norm(counts, axis=1, keepdims=True)
    S = Y @ Y.T
    r = S.sum(axis=1)
    f = np.array([d.f for d in devices.devices])
    return {"f": f, "kappa": np.array(kappas), "rtt": latency.rtt, "S": S, "r": r, "net_sum": float(r.sum()),
            "D": np.array([len(p) for p in parts]), "device_stats": devices.stats, "latency_stats": latency.stats}


def make_params(args, pop):
    rtt = pop["rtt"][~np.eye(args.N, dtype=bool)]
    L_max = args.L_max if args.L_max > 0 else float(rtt.max()) * 1.01  # every candidate reachable (50 candidates)
    f_ref = float(np.percentile(pop["f"], 75))
    weights = dict(zip(("comp", "comm", "data", "fair", "strag"), args.weights))
    return UtilityParams.create(args.form, weights, args.w_c, args.w_r, f_ref, L_max, args.dtau_cap, args.eps)


def generate_instances(args, pop, params):
    """
    The round-by-round instances. The state (tau, Q) evolves with the choices of the driver solver.
    """
    from dlsim.compass.core import hashing
    ids = [i.to_bytes(4, "big") for i in range(args.N)]
    tau = np.zeros(args.N)
    Q = np.ones(args.N)
    driver = create_solver(args.driver)
    late_prob = np.array(args.late_prob)
    instances = []
    for t in range(1, args.rounds + 1):
        pool = [int.from_bytes(x, "big") for x in hashing.pool(ids, t, args.M + 1)]
        aggregator, candidates = pool[0], sorted(pool[1:])
        idx = np.array(candidates)
        inst = {"round": t, "aggregator": aggregator, "ids": candidates, "f": pop["f"][idx],
                "L": pop["rtt"][aggregator, idx], "stale": t - tau[idx], "Q": Q[idx].copy(), "r": pop["r"][idx],
                "S": pop["S"][np.ix_(idx, idx)], "net_sum": pop["net_sum"]}
        instances.append(inst)
        problem = to_problem(inst, args.n, params)
        chosen = driver.solve(problem, Budget(), np.random.default_rng(rng_seed(args.seed, "driver", t)))
        rng = np.random.default_rng(rng_seed(args.seed, "lateness", t))
        for i in chosen.members:
            node = candidates[i]
            tau[node] = t
            u = rng.random()
            kappa = pop["kappa"][node]
            if u < args.dropout:
                lateness = math.inf
            elif u < args.dropout + late_prob[kappa]:
                lateness = float(rng.exponential(args.mean_lateness))
            else:
                lateness = 0.0
            Q[node] = est.update_q(Q[node], lateness, args.alpha_q, args.gamma)
    return instances


def to_problem(inst, n, params):
    return SelectionProblem(ids=list(inst["ids"]), n=n, f=inst["f"], L=inst["L"], stale=inst["stale"], Q=inst["Q"],
                            r=inst["r"], S=inst["S"], net_sum=inst["net_sum"], params=params,
                            aggregator=inst["aggregator"])


# ---------------------------------------------------------------------- solving (worker)

def solve_round(job):
    inst, n, params_dict, solvers, exact_time, seed = job
    params = UtilityParams.from_dict(params_dict)
    problem = to_problem(inst, n, params)
    rows = []
    for name in solvers:
        rng = np.random.default_rng(rng_seed(seed, "solver", name, inst["round"]))
        if name == "exact":
            solver = ExactSolver(max_combinations=10 ** 15, warm_start="greedy_swap")
            budget = Budget(max_time=exact_time)
        else:
            solver = create_solver(name)
            budget = Budget()
        start = time.perf_counter()
        result = solver.solve(problem, budget, rng)
        elapsed = time.perf_counter() - start
        check = evaluate(problem, result.members).utility
        assert abs(check - result.utility) < 1e-12, (name, check, result.utility)
        rows.append({"round": inst["round"], "solver": name, "utility": result.utility, "time": elapsed,
                     "evaluations": result.evaluations, "certified": result.certificate == "optimal",
                     "bound": result.bound if result.bound is not None else float("nan"),
                     "members": "-".join(str(problem.ids[i]) for i in result.members)})
    return rows


# ---------------------------------------------------------------------- reporting

def summarize(rows, solvers):
    by_round = {}
    for row in rows:
        by_round.setdefault(row["round"], {})[row["solver"]] = row
    for rnd, entries in by_round.items():
        exact = entries.get("exact")
        best_known = max(e["utility"] for e in entries.values())
        reference = exact["utility"] if exact and exact["certified"] else best_known
        upper = exact["utility"] if exact and exact["certified"] else (exact["bound"] if exact else best_known)
        for e in entries.values():
            e["reference"] = reference
            e["gap"] = 1.0 - e["utility"] / reference          # vs optimum (or best known if not certified)
            e["gap_bound"] = 1.0 - e["utility"] / upper          # certified upper bound on the gap
    summary = []
    for name in solvers:
        sel = [r for r in rows if r["solver"] == name]
        gaps = np.array([r["gap"] for r in sel])
        times = np.array([r["time"] for r in sel])
        summary.append({"solver": name, "rounds": len(sel), "mean_gap_pct": 100 * gaps.mean(),
                        "median_gap_pct": 100 * np.median(gaps), "max_gap_pct": 100 * gaps.max(),
                        "optimal_pct": 100 * float(np.mean(gaps < 1e-9)),
                        "max_certified_gap_bound_pct": 100 * max(r["gap_bound"] for r in sel),
                        "mean_time_s": times.mean(), "median_time_s": float(np.median(times)),
                        "max_time_s": times.max(), "mean_evaluations": float(np.mean([r["evaluations"] for r in sel]))})
    return summary


def write_csv(path, rows, columns):
    with open(path, "w") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(("%.10g" % row[c]) if isinstance(row[c], float) else str(row[c])
                                  for c in columns) + "\n")


def plot(rows, solvers, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(solvers, plt.cm.tab10.colors))

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    for name in solvers:
        sel = sorted((r for r in rows if r["solver"] == name), key=lambda r: r["round"])
        left.plot([r["round"] for r in sel], [r["utility"] for r in sel], label=name, color=colors[name],
                  linewidth=1.8 if name == "exact" else 1.0, alpha=0.95 if name in ("exact", "greedy_swap") else 0.7)
    left.set_xlabel("round")
    left.set_ylabel("utility U(C, a)")
    left.set_title("Fitness of the selected group per round")
    left.legend(fontsize=8, ncol=2)
    heur = [s for s in solvers if s != "exact"]
    right.boxplot([[100 * r["gap"] for r in rows if r["solver"] == s] for s in heur], labels=heur, vert=True,
                  showfliers=True)
    right.set_yscale("symlog", linthresh=0.1)
    right.set_ylabel("optimality gap (%) vs exact")
    right.set_title("Gap to the optimum (all rounds)")
    right.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fitness.png"), dpi=160)
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    for name in solvers:
        sel = sorted((r for r in rows if r["solver"] == name), key=lambda r: r["round"])
        left.plot([r["round"] for r in sel], [r["time"] for r in sel], label=name, color=colors[name], linewidth=1.0)
    left.set_yscale("log")
    left.set_xlabel("round")
    left.set_ylabel("computation time per solve (s, log)")
    left.set_title("Computation time per round")
    left.legend(fontsize=8, ncol=2)
    means = [np.mean([r["time"] for r in rows if r["solver"] == s]) for s in solvers]
    right.bar(range(len(solvers)), means, color=[colors[s] for s in solvers])
    right.set_yscale("log")
    right.set_xticks(range(len(solvers)))
    right.set_xticklabels(solvers, rotation=40, ha="right")
    right.set_ylabel("mean time (s, log)")
    right.set_title("Mean computation time")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "time.png"), dpi=160)
    plt.close(fig)


def snapshot(rows, args) -> None:
    """
    Partial results and plots while the experiment runs (rounds completed so far, unordered).
    """
    rows = sorted(rows, key=lambda r: (r["round"], args.solvers.index(r["solver"])))
    complete = [r for r in rows if sum(1 for x in rows if x["round"] == r["round"]) == len(args.solvers)]
    if not complete:
        return
    summary = summarize(complete, args.solvers)
    write_csv(os.path.join(args.out, "summary.partial.csv"), summary, list(summary[0]))
    plot(complete, args.solvers, args.out)


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--M", type=int, default=50, help="candidate clients per round")
    parser.add_argument("--n", type=int, default=10, help="clients selected per round")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--solvers", nargs="+", default=HEURISTICS + ["exact"])
    parser.add_argument("--driver", default="greedy_swap", help="solver whose choices drive tau and Q")
    parser.add_argument("--exact-time", type=float, default=600.0, help="time limit of exact per round (s)")
    parser.add_argument("--form", default="multiplicative")
    parser.add_argument("--weights", type=float, nargs=5, default=[0.2] * 5, metavar=("COMP", "COMM", "DATA",
                                                                                     "FAIR", "STRAG"))
    parser.add_argument("--w-c", type=float, default=0.5)
    parser.add_argument("--w-r", type=float, default=0.5)
    parser.add_argument("--dtau-cap", type=float, default=20.0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--L-max", type=float, default=0.0, help="0 = every candidate reachable")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet label skew")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--alpha-q", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--late-prob", type=float, nargs=3, default=[0.3, 0.15, 0.05])
    parser.add_argument("--mean-lateness", type=float, default=15.0)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="write partial results and plots")
    parser.add_argument("--resume", action="store_true", help="continue from results.partial.jsonl")
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "selection_optimality"))
    args = parser.parse_args()
    if abs(sum(args.weights) - 1) > 1e-9 or min(args.weights) <= 0:
        sys.exit("weights must be > 0 and sum to 1")
    os.makedirs(args.out, exist_ok=True)

    started = time.time()
    pop = build_population(args)
    params = make_params(args, pop)
    instances = generate_instances(args, pop, params)
    print("Population of %d clients and %d instances ready in %.1f s" % (args.N, len(instances), time.time() - started))

    partial = os.path.join(args.out, "results.partial.jsonl")
    rows = []
    if args.resume and os.path.exists(partial):
        with open(partial) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        print("Resuming: %d rounds already solved" % len({r["round"] for r in rows}))
    elif os.path.exists(partial):
        os.remove(partial)
    done_rounds = {r["round"] for r in rows}
    jobs = [(inst, args.n, params.to_dict(), args.solvers, args.exact_time, args.seed) for inst in instances
            if inst["round"] not in done_rounds]
    with get_context("spawn").Pool(args.workers) as pool, open(partial, "a") as checkpoint:
        for done, result in enumerate(pool.imap_unordered(solve_round, jobs), len(done_rounds) + 1):
            rows.extend(result)
            for row in result:
                checkpoint.write(json.dumps(row) + "\n")
            checkpoint.flush()
            if done % args.checkpoint_every == 0 or done == len(instances):
                print("%d/%d rounds solved (%.0f s)" % (done, len(instances), time.time() - started), flush=True)
                snapshot(rows, args)
    rows.sort(key=lambda r: (r["round"], args.solvers.index(r["solver"])))

    summary = summarize(rows, args.solvers)
    write_csv(os.path.join(args.out, "results.csv"), rows,
              ["round", "solver", "utility", "reference", "gap", "gap_bound", "time", "evaluations", "certified",
               "bound", "members"])
    write_csv(os.path.join(args.out, "summary.csv"), summary, list(summary[0]))
    plot(rows, args.solvers, args.out)
    with open(os.path.join(args.out, "meta.json"), "w") as handle:
        json.dump({"args": vars(args), "utility": params.to_dict(), "wall_time_s": time.time() - started,
                   "devices": {k: v for k, v in pop["device_stats"].items() if k != "fedscale_sampled_ids"},
                   "latency": pop["latency_stats"], "D_median": float(np.median(pop["D"]))}, handle, indent=2,
                  default=str)
    print("\n%-20s %10s %10s %10s %10s %12s %12s" % ("solver", "mean gap%", "med gap%", "max gap%", "optimal%",
                                                    "mean time s", "max time s"))
    for s in summary:
        print("%-20s %10.4f %10.4f %10.4f %10.1f %12.5f %12.5f" % (
            s["solver"], s["mean_gap_pct"], s["median_gap_pct"], s["max_gap_pct"], s["optimal_pct"],
            s["mean_time_s"], s["max_time_s"]))
    print("\nOutputs in %s" % args.out)


if __name__ == "__main__":
    main()
