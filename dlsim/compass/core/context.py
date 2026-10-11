"""
The selection context that the current aggregator assembles for round t+1 and sends to the eligible candidates, and
the verification of proposals (paper Sec. 4.3, 4.6).

The context is built once, from one immutable view snapshot (``snapshot_id``). Candidates and the verifier build their
``SelectionProblem`` from the same context, so a correct proposal is always reproduced exactly by the verifier.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from dlsim.compass.core.utility import SelectionProblem, UtilityParams, evaluate


@dataclass
class SelectionContext:
    round: int                                   # The round being selected (t+1)
    k: int                                       # Resample index of the pool
    snapshot_id: Tuple
    pool: List[Hashable]                         # Responding pool members, ascending id
    S: np.ndarray                                # Pool similarity matrix
    r: np.ndarray                                # Row sums
    f: np.ndarray                                # From the snapshot
    B: np.ndarray
    Q: np.ndarray
    stale: np.ndarray                            # dtau_i^(t+1) = (t+1) - tau_i
    net_sum: float
    params: UtilityParams
    n: int                                       # Group size n'
    k_nb: int
    eligible: List[Hashable]
    excluded_clients: List[Hashable] = field(default_factory=list)  # e.g. A(t) when forbidden as a client

    def index(self, node: Hashable) -> int:
        return self.pool.index(node)

    def problem(self, aggregator: Hashable, members: Sequence[Hashable], rtt: Dict[Hashable, float]
                ) -> SelectionProblem:
        """
        The selection problem of ``aggregator`` over ``members`` (sorted by id internally).
        """
        ordered = sorted(members)
        idx = [self.index(node) for node in ordered]
        return SelectionProblem(ids=ordered, n=self.n, f=self.f[idx], L=np.array([rtt[node] for node in ordered]),
                                stale=self.stale[idx], Q=self.Q[idx], r=self.r[idx], S=self.S[np.ix_(idx, idx)],
                                net_sum=self.net_sum, params=self.params, aggregator=aggregator)

    def size_bytes(self) -> int:
        """
        Approximate wire size: O(M^2) similarities plus O(M) per-node scalars.
        """
        m = len(self.pool)
        return 8 * (m * (m - 1) // 2 + 6 * m) + 32 * m + 256


@dataclass
class Proposal:
    aggregator: Hashable
    clients: List[Hashable]
    utility: float
    latencies: Dict[Hashable, float]             # L_{i,a} of the proposed clients (RTT)
    components: Dict[str, float] = field(default_factory=dict)
    snapshot_id: Tuple = ()
    neighborhood_size: int = 0
    evaluations: int = 0
    solve_time: float = 0.0
    solver: str = ""
    certificate: Optional[str] = None


@dataclass
class Verdict:
    ok: bool
    reason: str
    utility: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)


def verify(context: SelectionContext, proposal: Proposal, tolerance: float) -> Verdict:
    """
    Recompute U from the context plus the proposal's latencies; reject on any inconsistency.
    """
    if proposal.snapshot_id != context.snapshot_id:
        return Verdict(False, "snapshot_mismatch")
    if proposal.aggregator not in context.eligible:
        return Verdict(False, "aggregator_not_eligible")
    clients = list(proposal.clients)
    if len(clients) != context.n or len(set(clients)) != len(clients):
        return Verdict(False, "wrong_group_size")
    pool = set(context.pool)
    for node in clients:
        if node == proposal.aggregator or node not in pool or node in context.excluded_clients:
            return Verdict(False, "client_not_allowed")
        latency = proposal.latencies.get(node)
        if latency is None or not math.isfinite(latency) or latency < 0 or latency >= context.params.L_max:
            return Verdict(False, "invalid_latency")
    problem = context.problem(proposal.aggregator, clients, proposal.latencies)
    evaluation = evaluate(problem, range(problem.size))
    if not math.isfinite(proposal.utility) or abs(evaluation.utility - proposal.utility) > tolerance:
        return Verdict(False, "utility_mismatch", evaluation.utility, evaluation.components)
    return Verdict(True, "ok", evaluation.utility, evaluation.components)


def rank(proposals: Sequence[Proposal]) -> List[Proposal]:
    """
    Highest verified utility first; ties broken by aggregator id.
    """
    return sorted(proposals, key=lambda p: (-p.utility, p.aggregator))
