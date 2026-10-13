"""
Exact solvers compared: our branch-and-bound (``exact``) versus a generic MINLP formulation solved by SCIP.

Both receive the same instances as experiments/selection_optimality.py (same population, same rounds). The MINLP is
an independent formulation of the same problem, so agreement also cross-validates the optimum.

MINLP (multiplicative utility, maximized in log space):
    x_i in {0,1}, sum_i x_i = n
    y_ij in [0,1], y_ij <= x_i, y_ij <= x_j, y_ij >= x_i + x_j - 1        (y_ij = x_i x_j exactly)
    z <= f_i + M (1 - x_i), z <= f_ref                                       (z = min f, clipped at f_ref)
    w >= L_i x_i                                                             (w = max L)
    P = sum_{i<j} S_ij y_ij,  R = sum r_i x_i,  F = sum fair_i x_i,  G = sum Q_i x_i
    max  w_comp log(z / f_ref) + w_comm log(1 - w / L_max)
         + w_data [ w_c log(P / C(n,2)) + w_r (log R - 0.5 log((n + 2P) net)) ]
         + w_fair log(F / n) + w_strag log(G / n)
The -log(n + 2P) part is not concave, so SCIP solves the problem by spatial branch-and-bound (global optimality).
Lower clipping at epsilon is not modeled; the chosen set is re-evaluated with the paper's utility afterwards.

    python experiments/exact_vs_scip.py --rounds 10 --time-limit 1800
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "pyipv8"), os.path.join(ROOT, "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from dlsim.compass.core.utility import evaluate  # noqa: E402
from dlsim.compass.solvers import Budget  # noqa: E402
from dlsim.compass.solvers.exact import ExactSolver  # noqa: E402
import selection_optimality as so  # noqa: E402


def solve_scip(problem, time_limit: float, verbose: bool = False, threads: int = 1):
    from pyscipopt import Model, log, quicksum

    p = problem.params
    w = p.weight_map
    k, n = problem.size, problem.n
    m = Model("compass")
    if not verbose:
        m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("parallel/maxnthreads", threads)
    m.setParam("parallel/minnthreads", threads)
    x = [m.addVar(vtype="B", name="x%d" % i) for i in range(k)]
    m.addCons(quicksum(x) == n)
    y = {}
    for i in range(k):
        for j in range(i + 1, k):
            y[i, j] = v = m.addVar(lb=0, ub=1, name="y%d_%d" % (i, j))
            m.addCons(v <= x[i])
            m.addCons(v <= x[j])
            m.addCons(v >= x[i] + x[j] - 1)
    f = problem.f
    big_m = float(f.max() - f.min())
    z = m.addVar(lb=float(f.min()), ub=min(float(f.max()), p.f_ref), name="z")
    for i in range(k):
        m.addCons(z <= float(f[i]) + big_m * (1 - x[i]))
    L = problem.L
    wl = m.addVar(lb=float(L.min()), ub=float(L.max()), name="w")
    for i in range(k):
        m.addCons(wl >= float(L[i]) * x[i])
    pairs = n * (n - 1) / 2
    P = m.addVar(lb=0, ub=pairs, name="P")
    m.addCons(P == quicksum(float(problem.S[i, j]) * v for (i, j), v in y.items()))
    R = m.addVar(lb=1e-9, ub=float(np.sort(problem.r)[-n:].sum()), name="R")
    m.addCons(R == quicksum(float(problem.r[i]) * x[i] for i in range(k)))
    F = m.addVar(lb=1e-9, ub=n, name="F")
    m.addCons(F == quicksum(float(problem.fair[i]) * x[i] for i in range(k)))
    G = m.addVar(lb=1e-9, ub=n, name="G")
    m.addCons(G == quicksum(float(problem.Q[i]) * x[i] for i in range(k)))
    t = m.addVar(lb=None, ub=None, name="t")
    objective = (w["comp"] * log(z / p.f_ref) + w["comm"] * log(1 - wl / p.L_max)
                 + w["data"] * (p.w_c * log(P / pairs + 1e-12) + p.w_r * (log(R) - 0.5 * log(n + 2 * P)
                                                                         - 0.5 * math.log(problem.net_sum)))
                 + w["fair"] * log(F / n) + w["strag"] * log(G / n))
    m.addCons(t <= objective)
    m.setObjective(t, "maximize")
    start = time.perf_counter()
    if threads > 1:
        m.solveConcurrent()  # Concurrent racing of differently configured SCIP solvers, one per thread
    else:
        m.optimize()
    elapsed = time.perf_counter() - start
    status = m.getStatus()
    members = None
    if m.getNSols() > 0:
        sol = m.getBestSol()
        members = sorted(i for i in range(k) if m.getSolVal(sol, x[i]) > 0.5)
    bound = math.exp(m.getDualbound()) if status in ("optimal", "timelimit") else float("nan")
    return members, elapsed, status, bound, m.getNNodes()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--warm-start", default="greedy_swap", help="warm start of our B&B (none to disable)")
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "exact_vs_scip"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--threads", type=int, nargs="+", default=None,
                        help="only compare SCIP with these thread counts (no B&B), e.g. --threads 1 8")
    known, rest = parser.parse_known_args()
    so_args = _so_args(known.rounds, rest)  # population/instance options of selection_optimality.py
    os.makedirs(known.out, exist_ok=True)

    pop = so.build_population(so_args)
    params = so.make_params(so_args, pop)
    instances = so.generate_instances(so_args, pop, params)
    if known.threads:
        _compare_threads(instances, so_args, params, known)
        return
    rows = []
    for inst in instances:
        problem = so.to_problem(inst, so_args.n, params)
        warm = None if known.warm_start == "none" else known.warm_start
        start = time.perf_counter()
        bb = ExactSolver(max_combinations=10 ** 15, warm_start=warm).solve(problem, Budget(max_time=known.time_limit),
                                                                           np.random.default_rng(0))
        bb_time = time.perf_counter() - start
        members, scip_time, status, bound, nodes = solve_scip(problem, known.time_limit, known.verbose)
        scip_u = evaluate(problem, members).utility if members is not None else float("nan")
        row = {"round": inst["round"], "bb_utility": bb.utility, "bb_time": bb_time,
               "bb_certified": bb.certificate == "optimal", "bb_nodes": bb.evaluations,
               "scip_utility": scip_u, "scip_time": scip_time, "scip_status": status, "scip_bound": bound,
               "scip_nodes": nodes, "same_set": members == bb.members,
               "rel_diff": (bb.utility - scip_u) / bb.utility if members is not None else float("nan")}
        rows.append(row)
        print("round %3d | B&B U=%.6f %7.1fs %s | SCIP U=%.6f %7.1fs %-9s | same set: %s" % (
            row["round"], row["bb_utility"], bb_time, "opt" if row["bb_certified"] else "lim", scip_u, scip_time,
            status, row["same_set"]), flush=True)
    with open(os.path.join(known.out, "results.json"), "w") as handle:
        json.dump(rows, handle, indent=1)
    bb_t = np.array([r["bb_time"] for r in rows])
    sc_t = np.array([r["scip_time"] for r in rows])
    print("\nB&B : mean %.1fs, median %.1fs, max %.1fs, certified %d/%d" % (
        bb_t.mean(), np.median(bb_t), bb_t.max(), sum(r["bb_certified"] for r in rows), len(rows)))
    print("SCIP: mean %.1fs, median %.1fs, max %.1fs, optimal %d/%d" % (
        sc_t.mean(), np.median(sc_t), sc_t.max(), sum(r["scip_status"] == "optimal" for r in rows), len(rows)))
    print("Same optimal set in %d/%d rounds; B&B faster in %d/%d rounds (speed-up median x%.1f)" % (
        sum(r["same_set"] for r in rows), len(rows), sum(b < s for b, s in zip(bb_t, sc_t)), len(rows),
        float(np.median(sc_t / bb_t))))
    _plot(rows, known.out)


def _compare_threads(instances, so_args, params, known):
    from dlsim.compass.solvers import create_solver
    rows = []
    log_path = os.path.join(known.out, "threads_M%d_n%d.log" % (so_args.M, so_args.n))
    log = open(log_path, "a")

    def say(text):
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    say("== SCIP threads %s | N=%d M=%d n=%d rounds=%d time limit %.0fs | %s" % (
        known.threads, so_args.N, so_args.M, so_args.n, len(instances), known.time_limit, time.strftime("%H:%M:%S")))
    for inst in instances:
        problem = so.to_problem(inst, so_args.n, params)
        entry = {"round": inst["round"]}
        start = time.perf_counter()
        greedy = create_solver("greedy_swap").solve(problem, Budget(), np.random.default_rng(0))
        entry["greedy"] = {"time": time.perf_counter() - start, "utility": greedy.utility, "members": greedy.members}
        say("round %3d | start %s | candidates=%d | greedy_swap U=%.6f in %.3fs" % (
            inst["round"], time.strftime("%H:%M:%S"), problem.size, greedy.utility, entry["greedy"]["time"]))
        for threads in known.threads:
            members, elapsed, status, bound, nodes = solve_scip(problem, known.time_limit, known.verbose, threads)
            entry[threads] = {"time": elapsed, "status": status, "nodes": nodes,
                              "utility": evaluate(problem, members).utility if members is not None else None,
                              "members": members}
            u = entry[threads]["utility"]
            gap = 1 - greedy.utility / u if u else float("nan")
            say("round %3d | SCIP threads=%d: %8.1fs %-9s U=%.6f dual bound=%.6f nodes=%d | greedy gap %.4f%%%s" % (
                inst["round"], threads, elapsed, status, u if u else float("nan"), bound, nodes, 100 * gap,
                " (same set)" if members == greedy.members else ""))
        rows.append(entry)
        with open(os.path.join(known.out, "threads_M%d_n%d.json" % (so_args.M, so_args.n)), "w") as handle:
            json.dump(rows, handle, indent=1, default=str)
    base = known.threads[0]
    for threads in known.threads[1:]:
        speedups = [r[base]["time"] / r[threads]["time"] for r in rows]
        same = sum(r[base]["members"] == r[threads]["members"] for r in rows)
        say("threads=%d vs %d: speed-up %s, same set %d/%d" % (
            threads, base, ", ".join("x%.2f" % v for v in speedups), same, len(rows)))
    times = [r[base]["time"] for r in rows]
    say("SCIP threads=%d: mean %.1fs, max %.1fs, optimal %d/%d" % (
        base, np.mean(times), max(times), sum(r[base]["status"] == "optimal" for r in rows), len(rows)))
    log.close()


def _so_args(rounds, rest):
    saved = sys.argv
    sys.argv = ["selection_optimality.py", "--rounds", str(rounds)] + rest
    try:
        parser_args = _parse_so()
    finally:
        sys.argv = saved
    return parser_args


def _parse_so():
    # The population and instance options of selection_optimality.py (same defaults)
    import argparse as ap
    parser = ap.ArgumentParser()
    for name, kwargs in [("--N", dict(type=int, default=1000)), ("--M", dict(type=int, default=50)),
                         ("--n", dict(type=int, default=10)), ("--rounds", dict(type=int, default=10)),
                         ("--seed", dict(type=int, default=42)), ("--driver", dict(default="greedy_swap")),
                         ("--form", dict(default="multiplicative")),
                         ("--weights", dict(type=float, nargs=5, default=[0.2] * 5)),
                         ("--w-c", dict(type=float, default=0.5)), ("--w-r", dict(type=float, default=0.5)),
                         ("--dtau-cap", dict(type=float, default=20.0)), ("--eps", dict(type=float, default=1e-3)),
                         ("--L-max", dict(type=float, default=0.0)), ("--alpha", dict(type=float, default=0.3)),
                         ("--classes", dict(type=int, default=10)), ("--samples", dict(type=int, default=50000)),
                         ("--min-samples", dict(type=int, default=10)), ("--alpha-q", dict(type=float, default=0.3)),
                         ("--gamma", dict(type=float, default=0.1)), ("--dropout", dict(type=float, default=0.03)),
                         ("--late-prob", dict(type=float, nargs=3, default=[0.3, 0.15, 0.05])),
                         ("--mean-lateness", dict(type=float, default=15.0))]:
        parser.add_argument(name, **kwargs)
    return parser.parse_args()


def _plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rounds = [r["round"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    ax.bar([x - width / 2 for x in range(len(rows))], [r["bb_time"] for r in rows], width, label="our B&B (exact)")
    ax.bar([x + width / 2 for x in range(len(rows))], [r["scip_time"] for r in rows], width, label="SCIP (MINLP)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(rounds)
    ax.set_xlabel("round (instance)")
    ax.set_ylabel("time to proven optimum (s, log)")
    ax.set_title("Exact solvers: 50 candidates, 10 selected")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "exact_vs_scip.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
