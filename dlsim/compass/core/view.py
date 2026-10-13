"""
The population view (paper Sec. 3.3 and 4.7): one entry per member node, carried with the model from aggregator to
aggregator and written only by aggregators.

Entry: tau, Q, f, B, sigma, m, device class kappa, declared dataset size D, plus bookkeeping that is also written only
by aggregators: the missed-response counter (Q6), the round of the last applied state record (late-record ordering,
P2-7), the number of similarity-consistency flags (Q9) and the last observed training loss (oort_like baseline).

State updates of a round arrive as ``StateRecord``s. A record is applied in the order Q -> sigma -> f, B -> m (paper
Sec. 4.7). Censored uploads skip the B and sigma updates and never count as late (RQ1, RQ5). A record whose round is
not newer than the last record applied to that node is stale: it is discarded and reported, and EMA updates are never
reordered.
"""
import copy
import math
from dataclasses import dataclass, field, replace
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

from dlsim.compass.core import estimators as est


@dataclass
class NodeState:
    kappa: int
    D: float
    tau: int = 0
    Q: float = 1.0
    f: float = 1.0
    B: float = 1.0
    sigma: float = 0.0
    m: int = 0
    miss_count: int = 0
    last_record_round: int = -1
    flags: int = 0
    loss: Optional[float] = None
    loss_round: int = -1
    joined_round: int = 0


@dataclass(frozen=True)
class ClassPrior:
    f: float
    B: float
    sigma: float


@dataclass
class StateRecord:
    """
    What an aggregator observed about one client in one round.
    lateness: l_i = max(0, t_last - d_i'), or inf if nothing arrived within the grace period.
    """
    node: Hashable
    round: int
    lateness: float
    measurement: Optional[est.Measurement] = None
    loss: Optional[float] = None


@dataclass(frozen=True)
class ViewSnapshot:
    """
    An immutable copy of the view taken when the selection context is assembled (Sec. 4.7, "Timing of state
    updates"). Candidates and the verifier use only this.
    """
    snapshot_id: Tuple
    round: int
    entries: Mapping[Hashable, NodeState]
    num_members: int

    def get(self, node: Hashable) -> NodeState:
        return self.entries[node]


class PopulationView:

    def __init__(self, priors: Dict[int, ClassPrior], Q0: float = 1.0) -> None:
        self.priors = dict(priors)
        self.Q0 = Q0
        self.entries: Dict[Hashable, NodeState] = {}
        self.snapshot_counter = 0
        self.version = 0

    # ------------------------------------------------------------------ membership

    def add_node(self, node: Hashable, kappa: int, D: float, round: int = 0) -> NodeState:
        """
        A newly joining node: tau = 0, Q = Q0, m = 0, and the class priors for f, B and sigma.
        """
        prior = self.priors[kappa]
        state = NodeState(kappa=kappa, D=float(D), tau=0, Q=self.Q0, f=prior.f, B=prior.B, sigma=prior.sigma, m=0,
                          joined_round=round)
        self.entries[node] = state
        self.version += 1
        return state

    def remove_node(self, node: Hashable) -> bool:
        self.version += 1
        return self.entries.pop(node, None) is not None

    def members(self) -> List[Hashable]:
        return sorted(self.entries)

    def __contains__(self, node: Hashable) -> bool:
        return node in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, node: Hashable) -> NodeState:
        return self.entries[node]

    # ------------------------------------------------------------------ round updates

    def mark_participated(self, nodes: Iterable[Hashable], round: int) -> None:
        for node in nodes:
            if node in self.entries:
                self.entries[node].tau = max(self.entries[node].tau, round)
        self.version += 1

    def staleness(self, node: Hashable, round: int) -> int:
        return round - self.entries[node].tau

    def record_miss(self, node: Hashable) -> int:
        state = self.entries.get(node)
        if state is None:
            return 0
        state.miss_count += 1
        self.version += 1
        return state.miss_count

    def reset_miss(self, node: Hashable) -> None:
        state = self.entries.get(node)
        if state is not None and state.miss_count:
            state.miss_count = 0
            self.version += 1

    def add_flag(self, node: Hashable) -> None:
        if node in self.entries:
            self.entries[node].flags += 1
            self.version += 1

    def predict(self, node: Hashable, E: int, c: float, Z: float, upload_rate: Optional[float] = None) -> float:
        state = self.entries[node]
        return est.predict_completion(E, state.D, c, Z, state.f, state.B, upload_rate)

    def apply_record(self, record: StateRecord, E: int, c: float, Z: float, alpha: float, gamma: float,
                     beta: float) -> str:
        """
        Apply one state record. Returns ``applied``, ``stale`` or ``unknown`` (node no longer a member).
        """
        state = self.entries.get(record.node)
        if state is None:
            return "unknown"
        if state.last_record_round >= record.round:
            return "stale"

        measurement = record.measurement
        lateness = record.lateness
        if measurement is not None and measurement.censored and math.isfinite(lateness):
            lateness = 0.0  # Congested uploads never count as late (RQ1)

        # 1. Deadline adherence (Q)
        state.Q = est.update_q(state.Q, lateness, alpha, gamma)

        if measurement is not None and math.isfinite(lateness) and measurement.T_cmp > 0:
            step = est.step_size(beta, state.m)
            # 2. Prediction-error variance, with mu from the estimates before this measurement
            if not measurement.censored:
                mu = est.predict_completion(E, state.D, c, Z, state.f, state.B)
                if math.isfinite(mu):
                    state.sigma = math.sqrt(est.ema(state.sigma ** 2, (measurement.delta_hat - mu) ** 2, step))
            # 3. Capacity and bandwidth
            f_hat, B_hat = est.measure(E, state.D, c, Z, measurement)
            if f_hat is not None:
                state.f = est.ema(state.f, f_hat, step)
            if B_hat is not None and not measurement.censored:
                state.B = est.ema(state.B, B_hat, step)
            # 4. Measurement count
            state.m += 1

        if record.loss is not None:
            state.loss = record.loss
            state.loss_round = record.round
        state.last_record_round = record.round
        self.version += 1
        return "applied"

    # ------------------------------------------------------------------ snapshots and transport

    def snapshot(self, round: int, nodes: Optional[Iterable[Hashable]] = None, tag: Tuple = ()) -> ViewSnapshot:
        self.snapshot_counter += 1
        selected = self.entries if nodes is None else {n: self.entries[n] for n in nodes if n in self.entries}
        frozen = {node: replace(state) for node, state in selected.items()}
        # Entries are copies, so later changes to the live view never reach a snapshot. (A plain dict keeps the
        # snapshot picklable, which is needed to ship it to the candidates.)
        return ViewSnapshot((round,) + tuple(tag) + (self.snapshot_counter,), round, frozen, len(self.entries))

    def copy(self) -> "PopulationView":
        return copy.deepcopy(self)
