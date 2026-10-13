"""
The COMPASS-FL utility U(C, a) (paper Sec. 3.4).

For a candidate aggregator a and a client set C drawn from its neighborhood:

    Phi_comp  = min(1, min_{i in C} f_i / f_ref)
    Phi_comm  = 1 - max_{i in C} L_{i,a} / L_max                     (L is the RTT)
    Phi_coh   = sum_{i<j in C} S_ij / C(|C|, 2)                      (1 for |C| = 1)
    Phi_repr  = sum_{i in C} r_i / sqrt((|C| + 2 P) * net_sum)        (P = sum_{i<j in C} S_ij)
    Phi_data  = Phi_coh^w_c * Phi_repr^w_r
    Phi_fair  = mean_{i in C} min(dtau_i, dtau_cap) / dtau_cap
    Phi_adh   = mean_{i in C} Q_i    (deadline adherence)

Every component is clipped to [eps, 1]. U is the weighted product (multiplicative, default) or the weighted sum
(additive) of (comp, comm, data, fair, adh). Partial sets use |C| in place of n.

All values are computed from a handful of set aggregates (size, min f, max L, P, R, fairness sum, Q sum), so the full
evaluation and the O(1) incremental evaluation used by the solvers share exactly the same formula.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

COMPONENTS = ("comp", "comm", "data", "fair", "adh")


@dataclass(frozen=True)
class UtilityParams:
    form: str
    weights: Tuple[Tuple[str, float], ...]  # ((name, w), ...) in COMPONENTS order; hashable and immutable
    w_c: float
    w_r: float
    f_ref: float
    L_max: float
    dtau_cap: float
    eps: float

    @classmethod
    def create(cls, form: str = "multiplicative", weights: Optional[Dict[str, float]] = None, w_c: float = 0.5,
               w_r: float = 0.5, f_ref: float = 1.0, L_max: float = 1.0, dtau_cap: float = 20.0,
               eps: float = 1e-3) -> "UtilityParams":
        weights = weights or {name: 0.2 for name in COMPONENTS}
        return cls(form, tuple((name, float(weights[name])) for name in COMPONENTS), float(w_c), float(w_r),
                   float(f_ref), float(L_max), float(dtau_cap), float(eps))

    @classmethod
    def from_config(cls, config, f_ref: float, L_max: float) -> "UtilityParams":
        u = config.utility
        w = u.weights
        return cls.create(u.form, {"comp": w.comp, "comm": w.comm, "data": w.data, "fair": w.fair,
                                   "adh": w.adh}, u.w_c, u.w_r, f_ref, L_max, u.dtau_cap, u.eps)

    @property
    def weight_map(self) -> Dict[str, float]:
        return dict(self.weights)

    def to_dict(self) -> Dict:
        return {"form": self.form, "weights": self.weight_map, "w_c": self.w_c, "w_r": self.w_r,
                "f_ref": self.f_ref, "L_max": self.L_max, "dtau_cap": self.dtau_cap, "eps": self.eps}

    @classmethod
    def from_dict(cls, data: Dict) -> "UtilityParams":
        return cls.create(**data)


@dataclass
class SelectionProblem:
    """
    One local selection problem: choose n members of the neighborhood for candidate aggregator ``aggregator``.

    Arrays are indexed by neighborhood position and the neighborhood is sorted by node id, so "ties broken by node id"
    means "ties broken by the smaller index".
    """
    ids: List                 # Node ids, ascending
    n: int                    # Target group size (n')
    f: np.ndarray             # Capacity estimates f_i (view snapshot)
    L: np.ndarray             # RTT L_{i,a}
    stale: np.ndarray         # dtau_i
    Q: np.ndarray             # Deadline-adherence score Q_i
    r: np.ndarray             # Row sums r_i
    S: np.ndarray             # Pairwise similarities (diag = 1)
    net_sum: float            # Estimate of sum_{j,j' in V} S_jj'
    params: UtilityParams
    aggregator: object = None
    fair: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        k = len(self.ids)
        for name in ("f", "L", "stale", "Q", "r"):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (k,):
                raise ValueError("%s must have shape (%d,)" % (name, k))
            setattr(self, name, array)
        self.S = np.asarray(self.S, dtype=float)
        if self.S.shape != (k, k):
            raise ValueError("S must have shape (%d, %d)" % (k, k))
        if list(self.ids) != sorted(self.ids):
            raise ValueError("The neighborhood must be sorted by node id")
        cap = self.params.dtau_cap
        self.fair = np.minimum(self.stale, cap) / cap

    @property
    def size(self) -> int:
        return len(self.ids)

    def subset(self, indices: Sequence[int], n: Optional[int] = None) -> "SelectionProblem":
        """
        The problem restricted to some members (used by LNS rebuilds and by tests).
        """
        idx = list(indices)
        return SelectionProblem([self.ids[i] for i in idx], self.n if n is None else n, self.f[idx], self.L[idx],
                                self.stale[idx], self.Q[idx], self.r[idx], self.S[np.ix_(idx, idx)], self.net_sum,
                                self.params, self.aggregator)


@dataclass
class Evaluation:
    utility: float
    components: Dict[str, float]  # comp, comm, coh, repr, data, fair, adh (clipped)


def _clip(value: float, eps: float) -> float:
    if math.isnan(value):
        return eps
    return min(1.0, max(eps, value))


def components_from_aggregates(params: UtilityParams, size: int, min_f: float, max_L: float, P: float, R: float,
                               fair_sum: float, q_sum: float, net_sum: float) -> Dict[str, float]:
    eps = params.eps
    if size <= 0:
        return {"comp": eps, "comm": eps, "coh": eps, "repr": eps, "data": eps, "fair": eps, "adh": eps}
    comp = _clip(min_f / params.f_ref, eps) if params.f_ref > 0 else 1.0
    comm = _clip(1.0 - max_L / params.L_max, eps) if params.L_max > 0 else 1.0
    coh = _clip(P / (size * (size - 1) / 2.0), eps) if size >= 2 else 1.0
    gram = size + 2.0 * P
    denominator = math.sqrt(gram * net_sum) if gram > 0 and net_sum > 0 else 0.0
    rep = _clip(R / denominator, eps) if denominator > 0 else eps
    data = 1.0
    if params.w_c > 0:
        data *= coh ** params.w_c
    if params.w_r > 0:
        data *= rep ** params.w_r
    data = _clip(data, eps)
    fair = _clip(fair_sum / size, eps)
    adh = _clip(q_sum / size, eps)
    return {"comp": comp, "comm": comm, "coh": coh, "repr": rep, "data": data, "fair": fair, "adh": adh}


def utility_from_components(params: UtilityParams, components: Dict[str, float]) -> float:
    if params.form == "additive":
        return sum(w * components[name] for name, w in params.weights)
    log_u = 0.0
    for name, w in params.weights:
        log_u += w * math.log(components[name])
    return math.exp(log_u)


def evaluate(problem: SelectionProblem, members: Sequence[int]) -> Evaluation:
    """
    The single, pure utility function: used by candidates, the verifier and every solver.
    """
    idx = list(members)
    if len(set(idx)) != len(idx):
        raise ValueError("Duplicate members in %s" % idx)
    size = len(idx)
    if size == 0:
        comps = components_from_aggregates(problem.params, 0, 0, 0, 0, 0, 0, 0, problem.net_sum)
        return Evaluation(utility_from_components(problem.params, comps), comps)
    sub = problem.S[np.ix_(idx, idx)]
    P = float((sub.sum() - np.trace(sub)) / 2.0)
    comps = components_from_aggregates(problem.params, size, float(problem.f[idx].min()), float(problem.L[idx].max()),
                                       P, float(problem.r[idx].sum()), float(problem.fair[idx].sum()),
                                       float(problem.Q[idx].sum()), problem.net_sum)
    return Evaluation(utility_from_components(problem.params, comps), comps)


class IncrementalState:
    """
    Incremental bookkeeping of a client set (paper Sec. 4.5): sigma_v = sum_{j in C} S_vj for every neighbor v, the sums
    P and R, the fairness and adherence sums, and the two smallest capacities and two largest latencies in C. Adding
    a node or evaluating a swap costs O(1); applying a swap costs O(k).

    Note: the paper's P' = P - sigma_u + sigma_v - S_uv holds when sigma_u excludes the diagonal term S_uu of the member
    u (sigma_u = sum_{j in C, j != u} S_uj). ``sigma`` here includes it for members, so S_uu is subtracted explicitly.
    """

    def __init__(self, problem: SelectionProblem, members: Sequence[int] = ()) -> None:
        self.problem = problem
        self.members: List[int] = []
        self.in_set = np.zeros(problem.size, dtype=bool)
        self.sigma = np.zeros(problem.size)
        self.P = 0.0
        self.R = 0.0
        self.fair_sum = 0.0
        self.q_sum = 0.0
        for v in members:
            self.add(v)

    # -- extremes (two smallest f, two largest L), recomputed in O(n) only when the set changes
    def _extremes(self) -> None:
        f_sorted = sorted(self.members, key=lambda i: (self.problem.f[i], i))
        l_sorted = sorted(self.members, key=lambda i: (-self.problem.L[i], i))
        self.min_f = [self.problem.f[i] for i in f_sorted[:2]]
        self.min_f_idx = f_sorted[:1]
        self.max_L = [self.problem.L[i] for i in l_sorted[:2]]
        self.max_L_idx = l_sorted[:1]

    @property
    def size(self) -> int:
        return len(self.members)

    def add(self, v: int) -> None:
        if self.in_set[v]:
            raise ValueError("%d already in the set" % v)
        p = self.problem
        self.P += self.sigma[v]
        self.R += p.r[v]
        self.fair_sum += p.fair[v]
        self.q_sum += p.Q[v]
        self.sigma += p.S[:, v]
        self.in_set[v] = True
        self.members.append(v)
        self._extremes()

    def remove(self, v: int) -> None:
        """
        Inverse of ``add`` (used when a depth-first search backtracks).
        """
        if not self.in_set[v]:
            raise ValueError("%d not in the set" % v)
        p = self.problem
        self.sigma -= p.S[:, v]
        self.P -= self.sigma[v]  # sigma_v now excludes v itself: sum_{j in C \ {v}} S_vj
        self.R -= p.r[v]
        self.fair_sum -= p.fair[v]
        self.q_sum -= p.Q[v]
        self.in_set[v] = False
        self.members.remove(v)
        self._extremes()

    def swap(self, u: int, v: int) -> None:
        p = self.problem
        self.P = self.P - (self.sigma[u] - p.S[u, u]) + self.sigma[v] - p.S[u, v]
        self.R += p.r[v] - p.r[u]
        self.fair_sum += p.fair[v] - p.fair[u]
        self.q_sum += p.Q[v] - p.Q[u]
        self.sigma += p.S[:, v] - p.S[:, u]
        self.in_set[u] = False
        self.in_set[v] = True
        self.members[self.members.index(u)] = v
        self._extremes()

    def _evaluate(self, size, min_f, max_L, P, R, fair_sum, q_sum) -> float:
        p = self.problem
        comps = components_from_aggregates(p.params, size, min_f, max_L, P, R, fair_sum, q_sum, p.net_sum)
        return utility_from_components(p.params, comps)

    def utility(self) -> float:
        if not self.members:
            return self._evaluate(0, 0, 0, 0, 0, 0, 0)
        return self._evaluate(self.size, self.min_f[0], self.max_L[0], self.P, self.R, self.fair_sum, self.q_sum)

    def utility_if_added(self, v: int) -> float:
        p = self.problem
        min_f = min(self.min_f[0], p.f[v]) if self.members else p.f[v]
        max_L = max(self.max_L[0], p.L[v]) if self.members else p.L[v]
        return self._evaluate(self.size + 1, min_f, max_L, self.P + self.sigma[v], self.R + p.r[v],
                              self.fair_sum + p.fair[v], self.q_sum + p.Q[v])

    def utility_if_swapped(self, u: int, v: int) -> float:
        p = self.problem
        # Removing u exposes at most the second-smallest capacity / second-largest latency
        rest_min_f = (self.min_f[1] if len(self.min_f) > 1 else math.inf) if u == self.min_f_idx[0] \
            else self.min_f[0]
        rest_max_L = (self.max_L[1] if len(self.max_L) > 1 else -math.inf) if u == self.max_L_idx[0] \
            else self.max_L[0]
        P = self.P - (self.sigma[u] - p.S[u, u]) + self.sigma[v] - p.S[u, v]
        return self._evaluate(self.size, min(rest_min_f, p.f[v]), max(rest_max_L, p.L[v]), P,
                              self.R - p.r[u] + p.r[v], self.fair_sum - p.fair[u] + p.fair[v],
                              self.q_sum - p.Q[u] + p.Q[v])
