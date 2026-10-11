"""
Round closing (paper Sec. 4.7 "Predictive round closing"; decisions RQ1-RQ5).

A ``RoundClosingPolicy`` decides, for one round at the aggregator, which clients are stragglers and when the round
closes. It is pure: the aggregator feeds it events (acknowledgements, arrivals, the clock) and asks for the next time
it must be woken up.

For every client the predictive per-client deadline is always computed (RQ3), whatever the policy:

    d_i  = t_ack_i + mu_i + z sigma_i,      mu_i = E D_i c / f_i + Z / min(B_i, B_down(a) / k_i)
    d_i' = min(d_i, dispatch + delta_dl)    (RQ2)

k_i is the number of acknowledged clients of the round whose predicted upload windows
[t_ack_j + E D_j c / f_j, + Z / B_j] overlap client i's (RQ1); every d_i is recomputed when an acknowledgement
arrives. With ``contention_aware = False`` the upload term is Z / B_i (the paper's formula).

Policies:
- predictive:     a client is a straggler at d_i' (or at dispatch + delta_dl without an acknowledgement); the round
                  closes when every client delivered or is a straggler, and never later than dispatch + delta_dl.
- wait_all:       wait for every client, capped at dispatch + delta_dl.
- fixed_deadline: one deadline dispatch + delta_dl for everyone.
- over_selection: n' = ceil((1 + o) n) clients were selected; close when the first n delivered, or at the cap.

Lateness l_i = max(0, t_last - d_i') for any update arriving by d_i' + g (it carries a measurement), l_i = inf
otherwise or without an acknowledgement (RQ4). Stragglers are never aggregated.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

from dlsim.compass.core import estimators as est


@dataclass
class ClientInfo:
    """
    What the aggregator knows about a client when it dispatches the round (from the view it holds).
    """
    node: Hashable
    D: float
    f: float
    B: float
    sigma: float


@dataclass
class ClientTrack:
    info: ClientInfo
    t_ack: Optional[float] = None
    deadline: Optional[float] = None       # d_i' (effective)
    raw_deadline: Optional[float] = None   # d_i
    k: int = 1
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    utilization: float = 0.0
    status: str = "pending"                # pending, delivered, straggler
    aggregated: bool = False
    finalized: bool = False                # A state record has been produced

    @property
    def arrived(self) -> bool:
        return self.t_last is not None


@dataclass
class ClosingParams:
    E: int
    c: float
    Z: float
    z: float
    delta_dl: float
    grace: float
    rho: float
    contention_aware: bool
    B_down_aggregator: float
    n_target: int                          # n (over_selection closes after n deliveries)


class RoundClosingPolicy:
    name = "base"

    def __init__(self, clients: Sequence[ClientInfo], dispatch: float, params: ClosingParams) -> None:
        self.params = params
        self.dispatch = dispatch
        self.cap = dispatch + params.delta_dl
        self.tracks: Dict[Hashable, ClientTrack] = {c.node: ClientTrack(c) for c in clients}
        self.closed = False
        self.close_time: Optional[float] = None

    # ------------------------------------------------------------------ deadlines

    def _windows(self) -> Dict[Hashable, Tuple[float, float]]:
        p = self.params
        out = {}
        for node, track in self.tracks.items():
            if track.t_ack is None:
                continue
            info = track.info
            start = track.t_ack + (p.E * info.D * p.c / info.f if info.f > 0 else math.inf)
            out[node] = (start, start + (p.Z / info.B if info.B > 0 else math.inf))
        return out

    def _recompute_deadlines(self) -> None:
        p = self.params
        windows = self._windows()
        for node, track in self.tracks.items():
            if track.t_ack is None:
                continue
            info = track.info
            k = 1
            if p.contention_aware:
                start, end = windows[node]
                k = sum(1 for other, (s, e) in windows.items() if other == node or (s < end and start < e))
                rate = est.contention_rate(info.B, p.B_down_aggregator, k)
            else:
                rate = info.B
            mu = est.predict_completion(p.E, info.D, p.c, p.Z, info.f, info.B, upload_rate=rate)
            track.k = k
            track.raw_deadline = track.t_ack + mu + p.z * info.sigma
            track.deadline = min(track.raw_deadline, self.cap)

    # ------------------------------------------------------------------ events

    def on_ack(self, node: Hashable, now: float) -> None:
        track = self.tracks.get(node)
        if track is None or track.t_ack is not None:
            return
        track.t_ack = now
        self._recompute_deadlines()

    def on_arrival(self, node: Hashable, t_first: float, t_last: float, utilization: float) -> str:
        """
        Returns ``accepted`` (will be aggregated), ``late`` (recorded only) or ``ignored``.
        """
        track = self.tracks.get(node)
        if track is None or track.arrived:
            return "ignored"
        track.t_first, track.t_last, track.utilization = t_first, t_last, utilization
        if not self.closed and track.status == "pending" and self._accepts(track, t_last):
            track.status = "delivered"
            track.aggregated = True
            self._maybe_close(t_last)
            return "accepted"
        if track.status == "pending":
            track.status = "straggler"
        return "late"

    def _accepts(self, track: ClientTrack, now: float) -> bool:
        return True

    def advance(self, now: float) -> None:
        """
        Process every deadline up to ``now``.
        """
        if self.closed:
            return
        for track in self.tracks.values():
            if track.status != "pending":
                continue
            if self._is_straggler(track, now):
                track.status = "straggler"
        if now >= self.cap:
            for track in self.tracks.values():
                if track.status == "pending":
                    track.status = "straggler"
        self._maybe_close(now)

    def _is_straggler(self, track: ClientTrack, now: float) -> bool:
        return False

    def _maybe_close(self, now: float) -> None:
        if self.closed:
            return
        if all(t.status != "pending" for t in self.tracks.values()) or now >= self.cap or self._extra_close():
            self.closed = True
            self.close_time = now
            for track in self.tracks.values():
                if track.status == "pending":
                    track.status = "straggler"

    def _extra_close(self) -> bool:
        return False

    def next_event(self) -> Optional[float]:
        """
        The next time the policy must be advanced (None when nothing is pending).
        """
        if self.closed:
            return None
        times = [self.cap]
        times += self._pending_deadlines()
        return min(times)

    def _pending_deadlines(self) -> List[float]:
        return []

    # ------------------------------------------------------------------ records

    def accepted(self) -> List[Hashable]:
        return [node for node, t in self.tracks.items() if t.aggregated]

    def grace_end(self) -> float:
        """
        Until when late arrivals are still recorded: max over stragglers with an acknowledgement of d_i' + g.
        """
        ends = [t.deadline + self.params.grace for t in self.tracks.values()
                if t.status == "straggler" and t.deadline is not None and not t.arrived]
        return max(ends) if ends else (self.close_time if self.close_time is not None else self.dispatch)

    def lateness(self, track: ClientTrack) -> float:
        if track.t_ack is None or track.deadline is None or not track.arrived:
            return math.inf
        if track.t_last > track.deadline + self.params.grace:
            return math.inf
        return max(0.0, track.t_last - track.deadline)

    def measurement(self, track: ClientTrack) -> Optional[est.Measurement]:
        if not math.isfinite(self.lateness(track)) or track.t_first is None:
            return None
        return est.Measurement(track.t_ack, track.t_first, track.t_last,
                               censored=track.utilization >= self.params.rho)

    def resolvable(self, track: ClientTrack, now: float) -> bool:
        """
        Whether the client's state record is final at ``now``: it arrived, or it can no longer arrive in grace.
        """
        if track.arrived:
            return True
        if track.t_ack is None:
            return self.closed
        return track.status == "straggler" and now >= track.deadline + self.params.grace


class PredictivePolicy(RoundClosingPolicy):
    name = "predictive"

    def _accepts(self, track: ClientTrack, now: float) -> bool:
        return track.deadline is not None and now <= track.deadline

    def _is_straggler(self, track: ClientTrack, now: float) -> bool:
        if track.t_ack is None:
            return now >= self.cap
        return now >= track.deadline

    def _pending_deadlines(self) -> List[float]:
        return [t.deadline for t in self.tracks.values() if t.status == "pending" and t.deadline is not None]


class WaitAllPolicy(RoundClosingPolicy):
    name = "wait_all"


class FixedDeadlinePolicy(RoundClosingPolicy):
    name = "fixed_deadline"

    def _accepts(self, track: ClientTrack, now: float) -> bool:
        return now <= self.cap


class OverSelectionPolicy(RoundClosingPolicy):
    name = "over_selection"

    def _extra_close(self) -> bool:
        return sum(1 for t in self.tracks.values() if t.aggregated) >= self.params.n_target


POLICIES = {cls.name: cls for cls in (PredictivePolicy, WaitAllPolicy, FixedDeadlinePolicy, OverSelectionPolicy)}


def make_policy(name: str, clients: Sequence[ClientInfo], dispatch: float, params: ClosingParams
                ) -> RoundClosingPolicy:
    return POLICIES[name](clients, dispatch, params)
