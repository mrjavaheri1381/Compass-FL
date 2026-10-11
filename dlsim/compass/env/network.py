"""
Flow-level network with per-node upload and download capacities and max-min fair sharing (decision Q3).

A message from i to j first travels the one-way propagation delay, then becomes a flow limited by i's upload and j's
download capacity. Whenever a flow starts or ends, all rates are recomputed by progressive filling (max-min fairness),
so concurrent uploads to one aggregator share its ingress link and slow each other down.

For every node the network integrates the incoming rate and the download capacity over time. The time-averaged
ingress utilization of the receiver during a flow, (integral of incoming rate) / (integral of capacity) over
[t_first, t_last], is returned with the flow and used for censoring bandwidth samples (RQ5).

Messages smaller than ``small_message_bytes`` (probes, acknowledgements, selection messages) skip bandwidth sharing:
they take the propagation delay plus size / min(up, down). They are counted in the byte statistics.
"""
import asyncio
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

EPS_BYTES = 1e-6


@dataclass
class Timing:
    ok: bool
    t_send: float
    t_first: float = 0.0
    t_last: float = 0.0
    ingress_utilization: float = 0.0
    size: int = 0


@dataclass
class Flow:
    flow_id: int
    src: int
    dst: int
    size: float
    remaining: float
    t_first: float
    future: asyncio.Future
    t_send: float = 0.0
    rate: float = 0.0
    in_integral_start: float = 0.0
    cap_integral_start: float = 0.0


class FairNetwork:

    def __init__(self, num_nodes: int, small_message_bytes: int = 4096) -> None:
        self.up = [1.0] * num_nodes
        self.down = [1.0] * num_nodes
        self.small = small_message_bytes
        self.flows: Dict[int, Flow] = {}
        self.next_id = 0
        self.last_update = 0.0
        self.in_rate = [0.0] * num_nodes
        self.in_integral = [0.0] * num_nodes
        self.cap_integral = [0.0] * num_nodes
        self.timer: Optional[asyncio.TimerHandle] = None
        self.offline: set = set()
        self.bytes_sent = [0] * num_nodes
        self.bytes_received = [0] * num_nodes
        self._loop = None  # Bound to the event loop of the first transfer

    @property
    def loop(self):
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
            self.last_update = self._loop.time()
        return self._loop

    def now(self) -> float:
        return self.loop.time()

    # ------------------------------------------------------------------ capacities and liveness

    def set_capacity(self, node: int, up: float, down: float) -> None:
        if self.up[node] == up and self.down[node] == down:
            return
        if self._loop is None or not self.flows:  # Nothing in flight: no need to reallocate
            if self._loop is not None:
                self._advance()
            self.up[node], self.down[node] = float(up), float(down)
            return
        self._advance()
        self.up[node], self.down[node] = float(up), float(down)
        self._reallocate()

    def set_offline(self, node: int, offline: bool) -> None:
        if offline:
            self.offline.add(node)
            self._advance()
            for flow in [f for f in self.flows.values() if f.src == node or f.dst == node]:
                self._finish(flow, ok=False)
            self._reallocate()
        else:
            self.offline.discard(node)

    # ------------------------------------------------------------------ transfers

    async def transfer(self, src: int, dst: int, size: int, delay: float) -> Timing:
        """
        Deliver ``size`` bytes from src to dst after a propagation delay; returns the timing seen by dst.
        """
        t_send = self.now()
        if src in self.offline or dst in self.offline:
            return Timing(False, t_send)
        if delay > 0:
            await asyncio.sleep(delay)
        if src in self.offline or dst in self.offline:
            return Timing(False, t_send)
        t_first = self.now()
        if size < self.small:
            rate = min(self.up[src], self.down[dst])
            duration = size / rate if rate > 0 else 0.0
            if duration > 0:
                await asyncio.sleep(duration)
            if src in self.offline or dst in self.offline:
                return Timing(False, t_send)
            self.bytes_sent[src] += size
            self.bytes_received[dst] += size
            return Timing(True, t_send, t_first, self.now(), 0.0, size)

        self._advance()
        future = self.loop.create_future()
        flow = Flow(self.next_id, src, dst, float(size), float(size), t_first, future, t_send=t_send,
                    in_integral_start=self.in_integral[dst], cap_integral_start=self.cap_integral[dst])
        self.next_id += 1
        self.flows[flow.flow_id] = flow
        self._reallocate()
        return await future

    # ------------------------------------------------------------------ internals

    def _advance(self) -> None:
        now = self.now()
        dt = now - self.last_update
        if dt > 0:
            for flow in self.flows.values():
                flow.remaining -= flow.rate * dt
            for node in range(len(self.up)):
                self.in_integral[node] += self.in_rate[node] * dt
                self.cap_integral[node] += self.down[node] * dt
        self.last_update = now

    def _reallocate(self) -> None:
        """
        Max-min fair rates by progressive filling over the upload and download links.
        """
        flows = sorted(self.flows.values(), key=lambda f: f.flow_id)
        residual_up = {f.src: self.up[f.src] for f in flows}
        residual_down = {f.dst: self.down[f.dst] for f in flows}
        unfrozen = {f.flow_id: f for f in flows}
        while unfrozen:
            share = math.inf
            for link, cap in residual_up.items():
                count = sum(1 for f in unfrozen.values() if f.src == link)
                if count:
                    share = min(share, cap / count)
            for link, cap in residual_down.items():
                count = sum(1 for f in unfrozen.values() if f.dst == link)
                if count:
                    share = min(share, cap / count)
            share = max(share, 0.0)
            # Freeze the flows on every bottleneck link at the fair share
            frozen = []
            for f in unfrozen.values():
                up_count = sum(1 for g in unfrozen.values() if g.src == f.src)
                down_count = sum(1 for g in unfrozen.values() if g.dst == f.dst)
                if residual_up[f.src] / up_count <= share + 1e-12 or residual_down[f.dst] / down_count <= share + 1e-12:
                    frozen.append(f)
            for f in frozen:
                f.rate = share
                residual_up[f.src] -= share
                residual_down[f.dst] -= share
                del unfrozen[f.flow_id]
        self.in_rate = [0.0] * len(self.up)
        for f in flows:
            self.in_rate[f.dst] += f.rate
        self._schedule()

    def _schedule(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        soonest = None
        for flow in self.flows.values():
            if flow.rate > 0:
                eta = max(0.0, flow.remaining) / flow.rate
                if soonest is None or eta < soonest[0] or (eta == soonest[0] and flow.flow_id < soonest[1]):
                    soonest = (eta, flow.flow_id)
        if soonest is not None:
            self.timer = self.loop.call_at(self.now() + soonest[0], self._on_timer)

    def _on_timer(self) -> None:
        self.timer = None
        self._advance()
        done = [f for f in self.flows.values() if f.remaining <= EPS_BYTES * max(1.0, f.size) or
                (f.rate > 0 and f.remaining / f.rate < 1e-12)]
        for flow in sorted(done, key=lambda f: f.flow_id):
            self._finish(flow, ok=True)
        self._reallocate()

    def _finish(self, flow: Flow, ok: bool) -> None:
        self.flows.pop(flow.flow_id, None)
        now = self.now()
        utilization = 0.0
        cap = self.cap_integral[flow.dst] - flow.cap_integral_start
        if cap > 0:
            utilization = (self.in_integral[flow.dst] - flow.in_integral_start) / cap
        transferred = int(flow.size - max(0.0, flow.remaining))
        self.bytes_sent[flow.src] += transferred
        self.bytes_received[flow.dst] += transferred
        if not flow.future.done():
            flow.future.set_result(Timing(ok, flow.t_send, flow.t_first, now, utilization, int(flow.size)))
