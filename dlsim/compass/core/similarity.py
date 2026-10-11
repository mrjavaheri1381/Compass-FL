"""
Similarity rows, row sums and pool-level assembly (paper Sec. 3.2, 4.3, 4.8).

Each node holds its row {S_ij : j in V \\ {i}} over the current membership (Q6) and r_i = 1 + sum_j S_ij (S_ii = 1).
In context collection every pool member reports (r_i, its row restricted to the pool). The aggregator uses
S_ij := (S_ij + S_ji) / 2 when both endpoints report (Q9; this also halves independent DP noise), flags pairs whose
reports differ by more than the tolerance, and estimates sum_{j,j'} S_jj' = sum_j r_j by (N / M_resp) sum_{pool} r_j.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np


def normalize(counts: Sequence[float]) -> np.ndarray:
    vector = np.asarray(counts, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def cosine(y_i: Sequence[float], y_j: Sequence[float]) -> float:
    return float(np.dot(normalize(y_i), normalize(y_j)))


class SimilarityRow:
    """
    The private similarity row of one node.
    """

    def __init__(self, owner: Hashable) -> None:
        self.owner = owner
        self.row: Dict[Hashable, float] = {}

    def set(self, peer: Hashable, value: float) -> None:
        if peer != self.owner:
            self.row[peer] = float(value)

    def remove(self, peer: Hashable) -> None:
        self.row.pop(peer, None)

    def knows(self, peer: Hashable) -> bool:
        return peer == self.owner or peer in self.row

    @property
    def r(self) -> float:
        return 1.0 + sum(self.row.values())

    def report(self, pool: Sequence[Hashable]) -> Tuple[float, Dict[Hashable, float]]:
        """
        (r_i, {S_ij : j in pool \\ {i}}) -- the only thing a pool member sends in context collection.
        """
        return self.r, {j: self.row[j] for j in pool if j != self.owner and j in self.row}


@dataclass
class PoolSimilarity:
    members: List[Hashable]                 # Pool members kept (ascending id)
    S: np.ndarray                           # |members| x |members|, diag 1
    r: np.ndarray
    flags: List[Tuple[Hashable, Hashable, float]] = field(default_factory=list)
    one_sided: int = 0                      # Pairs reported by only one endpoint
    dropped: List[Hashable] = field(default_factory=list)  # Members removed because a pair was unknown to both

    def index(self, node: Hashable) -> int:
        return self.members.index(node)


def assemble(reports: Dict[Hashable, Tuple[float, Dict[Hashable, float]]], tolerance: float) -> PoolSimilarity:
    """
    Build the pool similarity matrix from the members' reports. A member that shares a pair unknown to both
    endpoints (e.g. a join still in progress) is dropped -- no default similarity is ever substituted (Q9).
    """
    members = sorted(reports)
    dropped: List[Hashable] = []
    while True:
        unknown_count = {node: 0 for node in members}
        for x, i in enumerate(members):
            for j in members[x + 1:]:
                if j not in reports[i][1] and i not in reports[j][1]:
                    unknown_count[i] += 1
                    unknown_count[j] += 1
        worst = max(members, key=lambda node: (unknown_count[node], node)) if members else None
        if worst is None or unknown_count[worst] == 0:
            break
        members.remove(worst)
        dropped.append(worst)

    size = len(members)
    S = np.eye(size)
    flags: List[Tuple[Hashable, Hashable, float]] = []
    one_sided = 0
    for x, i in enumerate(members):
        for y in range(x + 1, size):
            j = members[y]
            a = reports[i][1].get(j)
            b = reports[j][1].get(i)
            if a is not None and b is not None:
                value = (a + b) / 2.0
                if abs(a - b) > tolerance:
                    flags.append((i, j, abs(a - b)))
            else:
                value = a if a is not None else b
                one_sided += 1
            S[x, y] = S[y, x] = value
    r = np.array([reports[i][0] for i in members], dtype=float)
    return PoolSimilarity(members, S, r, flags, one_sided, dropped)


def consistency_tolerance(ckks_tol: float, k: float, sigma_dp: float) -> float:
    return ckks_tol + k * sigma_dp * math.sqrt(2.0)


def estimate_network_sum(r_pool: Sequence[float], N: int) -> float:
    """
    sum_{j,j' in V} S_jj' = sum_j r_j  ~  (N / M_resp) sum_{j in pool} r_j.
    """
    values = list(r_pool)
    if not values:
        return 0.0
    return N / len(values) * float(sum(values))
