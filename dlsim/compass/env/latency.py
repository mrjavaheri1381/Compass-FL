"""
Latency model (task G2; decisions Q4, RQ6).

Ground truth RTT between nodes i and j:

    L_ij = W[city_i][city_j] + access(kappa_i) + access(kappa_j)

where W is the WonderNetwork city-to-city ping matrix shipped with Plexus (``data/latencies.txt``, ms). The matrix is
cleaned without clamping: for each pair, if both directions are valid (> 0) their mean is used, if only one is valid it
is used, and if neither is the pair is invalid; cities that still have invalid pairs are excluded from node assignment
(fewest exclusions, greedily). Every simulated message takes a one-way delay of L_ij / 2 times a log-normal jitter
(``plexus_compat`` uses the full L_ij, as the original Plexus simulation did); a probe's RTT is the sum of two such
delays.
"""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from dlsim.compass.core.hashing import rng_seed


def clean_matrix(raw: Sequence[Sequence[float]]):
    """
    Returns (symmetric matrix in ms with NaN for invalid pairs, list of excluded cities, statistics).
    """
    n = len(raw)
    W = np.full((n, n), np.nan)
    stats = {"cities": n, "entries_negative": 0, "entries_zero_offdiag": 0, "pairs_both_valid": 0,
             "pairs_one_valid": 0, "pairs_invalid": 0}
    for i in range(n):
        W[i, i] = max(0.0, float(raw[i][i])) if math.isfinite(raw[i][i]) else 0.0
        for j in range(n):
            if i != j:
                if raw[i][j] < 0:
                    stats["entries_negative"] += 1
                elif raw[i][j] == 0:
                    stats["entries_zero_offdiag"] += 1
    for i in range(n):
        for j in range(i + 1, n):
            a, b = float(raw[i][j]), float(raw[j][i])
            va, vb = math.isfinite(a) and a > 0, math.isfinite(b) and b > 0
            if va and vb:
                value = (a + b) / 2
                stats["pairs_both_valid"] += 1
            elif va or vb:
                value = a if va else b
                stats["pairs_one_valid"] += 1
            else:
                value = np.nan
                stats["pairs_invalid"] += 1
            W[i, j] = W[j, i] = value
    excluded: List[int] = []
    alive = list(range(n))
    while True:
        counts = {c: sum(1 for d in alive if d != c and math.isnan(W[c, d])) for c in alive}
        worst = max(alive, key=lambda c: (counts[c], -c)) if alive else None
        if worst is None or counts[worst] == 0:
            break
        alive.remove(worst)
        excluded.append(worst)
    stats["cities_excluded"] = excluded
    stats["cities_valid"] = len(alive)
    return W, alive, stats


class LatencyModel:

    def __init__(self, config, kappas: Sequence[int], seed: int) -> None:
        self.config = config
        self.N = len(kappas)
        self.kappas = list(kappas)
        self.stats: Dict = {}
        self.jitter_rng = np.random.default_rng(rng_seed(seed, "jitter"))
        access = np.array([config.access_delay_ms[k] for k in kappas]) / 1000.0
        if config.model == "wondernetwork":
            with open(config.matrix_path) as handle:
                raw = [[float(x) for x in line.strip().split(",")] for line in handle if line.strip()]
            W, valid, stats = clean_matrix(raw)
            self.stats.update(stats)
            self.cities = self._assign(valid, W, np.random.default_rng(rng_seed(seed, "cities")))
            base = W[np.ix_(self.cities, self.cities)] / 1000.0
        else:
            rng = np.random.default_rng(rng_seed(seed, "uniform_rtt"))
            low, high = config.uniform_rtt
            upper = np.triu(rng.uniform(low, high, (self.N, self.N)), 1)
            base = upper + upper.T
            self.cities = list(range(self.N))
        self.rtt = base + access[:, None] + access[None, :]
        np.fill_diagonal(self.rtt, 0.0)
        offdiag = self.rtt[~np.eye(self.N, dtype=bool)]
        self.stats.update({"rtt_median_s": float(np.median(offdiag)), "rtt_p90_s": float(np.percentile(offdiag, 90)),
                           "city_assignment": config.city_assignment,
                           "distinct_cities_used": len(set(self.cities))})

    def _assign(self, valid: List[int], W: np.ndarray, rng: np.random.Generator) -> List[int]:
        cfg = self.config
        if cfg.city_assignment == "round_robin":
            return [valid[i % len(valid)] for i in range(self.N)]
        centers = [valid[int(i)] for i in rng.choice(len(valid), size=min(cfg.regions, len(valid)), replace=False)]
        regions = []
        for center in centers:
            nearest = sorted(valid, key=lambda c: (W[center, c] if c != center else -1.0, c))
            regions.append(nearest[:cfg.cities_per_region])
        weights = np.asarray(cfg.region_weights[:len(regions)], dtype=float)
        weights = weights / weights.sum()
        out = []
        for _ in range(self.N):
            region = regions[int(rng.choice(len(regions), p=weights))]
            out.append(region[int(rng.integers(len(region)))])
        self.stats["regions"] = regions
        return out

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.rtt[~np.eye(self.N, dtype=bool)], q))

    def one_way(self, i: int, j: int) -> float:
        """
        Sampled one-way delay of one message from i to j (seconds).
        """
        if i == j:
            return 0.0
        base = self.rtt[i, j] if self.config.plexus_compat else self.rtt[i, j] / 2.0
        cv = self.config.jitter_cv
        if cv <= 0:
            return float(base)
        sigma = math.sqrt(math.log(1 + cv * cv))
        return float(base * self.jitter_rng.lognormal(-0.5 * sigma * sigma, sigma))

    def mean_one_way(self, i: int, j: int) -> float:
        return float(self.rtt[i, j] if self.config.plexus_compat else self.rtt[i, j] / 2.0)
