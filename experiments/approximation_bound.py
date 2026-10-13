"""
Numerical check of the approximation theorem for the full utility U (docs/approximation_proof.tex), on the instances
of experiments/selection_optimality.py with their certified optima.

    python experiments/approximation_bound.py --rounds 200

For every round it computes, from the optimal set C*:
  Gamma        the guaranteed ratio of the theorem (depends only on n, the weights and C* itself);
  seed ratio   U(C_c) / U* for the seed used in the proof (thresholds of C*, center = member of C* nearest to its
               centroid), which the theorem says is >= Gamma;
  center ratio U / U* of CENTER-2, the practical variant that enumerates only the capacity and latency thresholds
               (all centers), and of CENTER-2 followed by 1-swap local search.
"""
import argparse
import csv
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "pyipv8"), os.path.join(ROOT, "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import selection_optimality as so  # noqa: E402
from exact_vs_scip import _so_args  # noqa: E402
from dlsim.compass.core.utility import IncrementalState, evaluate  # noqa: E402
from dlsim.compass.solvers.base import Counter, Budget  # noqa: E402
from dlsim.compass.solvers.greedy_swap import local_search  # noqa: E402


def gamma(problem, members):
    """The guarantee of the theorem for the optimal set ``members`` (raw, unclipped statistics)."""
    p = problem.params
    w = dict(p.weights)
    n = len(members)
    idx = np.array(members)
    sub = problem.S[np.ix_(idx, idx)]
    P = (sub.sum() - np.trace(sub)) / 2.0
    coh = 2.0 * P / (n * (n - 1))
    g_coh = max(2.0 - 1.0 / coh, 2.0 / n) if coh > 0 else 1.0
    q = (1.0 + (n - 1) * coh) / n                      # ||mean of C*||^2
    rho = (1.0 + math.sqrt(2.0)) * math.sqrt(max(0.0, 1.0 - q))
    beta = math.asin(rho / math.sqrt(q)) if rho < math.sqrt(q) else math.pi / 2
    cos_star = problem.r[idx].sum() / math.sqrt((n + 2 * P) * problem.net_sum)
    theta = math.acos(min(1.0, cos_star))
    g_repr = math.cos(min(math.pi / 2, theta + beta)) / cos_star
    # Clipping floor (Lemma 1): a clipped component is >= eps, and the optimum's is <= 1
    g_coh = max(g_coh, p.eps / min(1.0, max(p.eps, coh)))
    g_repr = max(g_repr, p.eps / min(1.0, max(p.eps, cos_star)))
    fair = problem.fair[idx]
    Q = problem.Q[idx]
    k_fair = fair.min() / fair.mean()
    k_adh = Q.min() / Q.mean()
    total = (g_coh ** (w["data"] * p.w_c) * g_repr ** (w["data"] * p.w_r) * k_fair ** w["fair"] * k_adh ** w["adh"])
    return total, {"coh": coh, "g_coh": g_coh, "repr": cos_star, "g_repr": g_repr, "k_fair": k_fair, "k_adh": k_adh}


def center_set(problem, eligible, c):
    """c and the n - 1 eligible nodes most similar to c (ties by index)."""
    others = [j for j in eligible if j != c]
    order = sorted(others, key=lambda j: (-problem.S[c, j], j))
    return [c] + order[:problem.n - 1]


def proof_seed(problem, members):
    """The seed used in the proof: thresholds of C*, center = member of C* nearest to its centroid."""
    idx = np.array(members)
    phi, lam = problem.f[idx].min(), problem.L[idx].max()
    alpha, beta = problem.fair[idx].min(), problem.Q[idx].min()
    eligible = [j for j in range(problem.size) if problem.f[j] >= phi and problem.L[j] <= lam
                and problem.fair[j] >= alpha and problem.Q[j] >= beta]
    # nearest to the centroid <=> largest sum of similarities to C*
    c = max(members, key=lambda i: (problem.S[i, idx].sum(), -i))
    return center_set(problem, eligible, c)


def center2(problem):
    """CENTER-2: all (capacity, latency) thresholds x all centers; returns the best set by U."""
    best, best_u = None, -1.0
    order = {c: sorted((j for j in range(problem.size) if j != c), key=lambda j: (-problem.S[c, j], j))
             for c in range(problem.size)}
    for phi in sorted(set(problem.f)):
        for lam in sorted(set(problem.L)):
            ok = (problem.f >= phi) & (problem.L <= lam)
            if ok.sum() < problem.n:
                continue
            for c in np.flatnonzero(ok):
                chosen = [c] + [j for j in order[c] if ok[j]][:problem.n - 1]
                u = evaluate(problem, chosen).utility
                if u > best_u:
                    best, best_u = chosen, u
    return best, best_u


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--skip-center2", action="store_true")
    parser.add_argument("--optima", default=os.path.join(ROOT, "experiments", "results", "selection_optimality",
                                                         "results.csv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "approximation_bound"))
    args, rest = parser.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    optimum = {}
    with open(args.optima) as handle:
        for row in csv.DictReader(handle):
            if row["solver"] == "exact" and row["certified"] == "True":
                optimum[int(row["round"])] = (float(row["utility"]), [int(x) for x in row["members"].split("-")])

    so_args = _so_args(args.rounds, rest)
    pop = so.build_population(so_args)
    params = so.make_params(so_args, pop)
    rows = []
    for inst in so.generate_instances(so_args, pop, params):
        t = inst["round"]
        if t not in optimum:
            continue
        problem = so.to_problem(inst, so_args.n, params)
        u_star, ids = optimum[t]
        members = [inst["ids"].index(i) for i in ids]
        assert abs(evaluate(problem, members).utility - u_star) < 1e-9
        g, parts = gamma(problem, members)
        seed = proof_seed(problem, members)
        row = {"round": t, "U_star": u_star, "Gamma": g, "seed_ratio": evaluate(problem, seed).utility / u_star,
               **parts}
        if not args.skip_center2:
            chosen, u = center2(problem)
            state = IncrementalState(problem, chosen)
            local_search(problem, state, Counter(Budget()), 50)
            row["center2_ratio"] = u / u_star
            row["center2_swap_ratio"] = state.utility() / u_star
        rows.append(row)
        if row["seed_ratio"] < g - 1e-12:
            print("VIOLATION in round %d: seed ratio %.6f < Gamma %.6f" % (t, row["seed_ratio"], g), flush=True)
        if t % 20 == 0:
            print("%d rounds" % t, flush=True)

    with open(os.path.join(args.out, "results.csv"), "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"rounds": len(rows), "violations": int(sum(r["seed_ratio"] < r["Gamma"] - 1e-12 for r in rows))}
    for key in [k for k in rows[0] if k != "round"]:
        values = np.array([r[key] for r in rows])
        summary[key] = {"min": float(values.min()), "median": float(np.median(values)), "mean": float(values.mean()),
                        "max": float(values.max())}
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
