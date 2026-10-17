"""
The per-node protocol of COMPASS-FL (paper Sec. 4), driven by the simulated network of ``dlsim.compass.env.world``.

Round t at aggregator A(t) (``run_round``):
  dispatch theta^(t-1) to C(t) -> acknowledgements and updates feed the RoundClosingPolicy -> the round closes ->
  FedAvg weighted by D_i over non-stragglers -> tau_i <- t for all of C(t) -> state records (Q, sigma, f, B) of
  resolved clients are applied -> ``round_done`` to A(t-1) -> selection for t+1 -> handover.
Selection for t+1 (``select_compass``):
  pool M^(t+1,k) by hash -> context collection (r_i and similarity rows only) -> similarity assembly with averaging and
  consistency flags -> immutable view snapshot -> eligibility from the snapshot -> context to eligible candidates ->
  each candidate probes the pool (P probes, median RTT), builds N_a (k_nb nearest with L < L_max, ties by id; declines
  if |N_a| < n'), solves with the configured solver and proposes (a, C_a, U_a, {L_ia}) -> verification against the
  snapshot -> ranked list. No proposal -> next hash index k; still none -> A(t) serves round t+1 itself.
Handover (``handover_chain``):
  theta^(t) and the view go to the best proposal; A(t) keeps theta, the view and the ranking until A(t+1) acknowledges
  and reports its round closed; on a timeout it falls back to the next-ranked proposal.
Late arrivals (``apply_or_forward``):
  updates that arrive after their deadline but within the grace period produce state records; they are applied to the
  view after the snapshot, and forwarded to the successor if they arrive after the handover. Stale records (the node
  was updated in a later round already) are discarded and logged.
"""
import asyncio
import math
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from dlsim.compass.aggregation import fedavg
from dlsim.compass.closing import ClientInfo, ClosingParams, RoundClosingPolicy, make_policy
from dlsim.compass.core import hashing
from dlsim.compass.core.context import Proposal, SelectionContext, rank, verify
from dlsim.compass.core.hashing import rng_seed
from dlsim.compass.core.neighborhood import neighborhood, neighborhood_or_decline
from dlsim.compass.core.similarity import SimilarityRow, assemble, consistency_tolerance, estimate_network_sum
from dlsim.compass.core.utility import evaluate
from dlsim.compass.core.view import PopulationView, StateRecord
from dlsim.compass.solvers import SolverRefused, budget_from_config, create_solver

SMALL = 64  # Bytes of a control message header


@dataclass
class Message:
    type: str
    src: bytes
    round: int
    payload: Dict
    size: int
    category: str


@dataclass
class RoundState:
    round: int
    attempt: int
    clients: List[bytes]
    policy: RoundClosingPolicy
    theta_prev: bytes
    predecessor: Optional[bytes]
    dispatch: float
    models: Dict[bytes, bytes] = field(default_factory=dict)
    sizes: Dict[bytes, float] = field(default_factory=dict)
    losses: Dict[bytes, float] = field(default_factory=dict)
    timer: Optional[asyncio.TimerHandle] = None
    closed: asyncio.Future = None
    aborted: bool = False
    finalized: set = field(default_factory=set)
    late_updates: List = field(default_factory=list)
    log: Dict = field(default_factory=dict)


@dataclass
class Retained:
    """
    What A(t) keeps after handing round t+1 over (fault handling): the view, to take it back if the successor fails,
    and the successor, to forward late records. Once the successor reports its round closed the view is released
    (``view = None``) and only the most recent entry is kept.
    """
    round: int
    view: Optional[PopulationView]
    successor: bytes
    attempt: int


@dataclass
class SelectionOutcome:
    ranked: List[Proposal] = field(default_factory=list)
    self_serve: bool = False
    self_clients: List[bytes] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


class CompassNode:

    def __init__(self, world, index: int, pk: bytes, kappa: int, D: float, label_counts) -> None:
        self.world = world
        self.cfg = world.config
        self.index = index
        self.pk = pk
        self.kappa = kappa
        self.D = float(D)
        self.online = False
        self.membership: Dict[bytes, Tuple[int, float]] = {}
        self.row = SimilarityRow(pk)
        self.dormant: Dict[bytes, float] = {}
        self.engine = world.make_engine(label_counts)
        self.waiters: Dict = {}
        # Client state
        self.current: Tuple[int, int] = (0, 0)
        self.current_aggregator: Optional[bytes] = None
        self.train_task: Optional[asyncio.Task] = None
        # Aggregator state
        self.view: Optional[PopulationView] = None      # Live view while this node holds it
        self.rounds: Dict[int, RoundState] = {}
        self.retained: Dict[int, Retained] = {}
        self.attempts: Dict[int, int] = {}
        self.probe_cache: Dict[Tuple, Dict[bytes, float]] = {}
        self.pending_late: Dict[int, List[StateRecord]] = {}
        self.backups: Dict[Tuple[int, int], List[bytes]] = {}
        self.buffered: Dict[int, Dict[bytes, Tuple]] = {}

    # ================================================================== utilities

    @property
    def loop(self):
        return asyncio.get_event_loop()

    def now(self) -> float:
        return self.loop.time()

    def short(self, pk: bytes) -> int:
        return self.world.index_of(pk)

    def event(self, kind: str, round: int, **details) -> None:
        self.world.log_event(kind, round, node=self.index, **details)

    def send(self, dst: bytes, type: str, round: int, payload: Dict, size: int, category: str):
        return self.world.send(self, dst, Message(type, self.pk, round, payload, int(size), category))

    async def request(self, dst: bytes, type: str, round: int, payload: Dict, size: int, category: str,
                      timeout: float):
        """
        Send a request and wait for the reply carrying the same request id; None on timeout or failure.
        """
        rid = self.world.next_rid()
        future = self.loop.create_future()
        self.waiters[rid] = future
        self.send(dst, type, round, dict(payload, rid=rid), size, category)
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.waiters.pop(rid, None)

    def reply(self, msg: Message, type: str, payload: Dict, size: int, category: str) -> None:
        self.send(msg.src, type, msg.round, dict(payload, rid=msg.payload["rid"]), size, category)

    def resolve(self, msg: Message, value) -> None:
        future = self.waiters.get(msg.payload.get("rid"))
        if future is not None and not future.done():
            future.set_result(value)

    def wait_key(self, key) -> asyncio.Future:
        future = self.waiters.get(key)
        if future is None or future.done():
            future = self.loop.create_future()
            self.waiters[key] = future
        return future

    def resolve_key(self, key, value) -> None:
        future = self.waiters.get(key)
        if future is not None and not future.done():
            future.set_result(value)

    # ================================================================== message dispatch

    def on_message(self, msg: Message, timing) -> None:
        if not self.online:
            return
        handler = getattr(self, "on_" + msg.type, None)
        if handler is None:
            raise RuntimeError("Unknown message type %s" % msg.type)
        result = handler(msg, timing)
        if asyncio.iscoroutine(result):
            self.world.spawn(result)

    # replies to ``request``
    def on_ctx_reply(self, msg, timing):
        self.resolve(msg, msg.payload)

    def on_proposal(self, msg, timing):
        self.resolve(msg, msg.payload)

    def on_probe_reply(self, msg, timing):
        self.resolve(msg, self.now())

    def on_loss_reply(self, msg, timing):
        self.resolve(msg, msg.payload)

    def on_handover_ack(self, msg, timing):
        self.resolve(msg, True)

    def on_he_answer(self, msg, timing):
        self.resolve(msg, msg.payload)

    # ================================================================== membership and similarity

    def go_online(self) -> None:
        self.online = True
        self.world.network.set_offline(self.index, False)

    def go_offline(self, reason: str) -> None:
        if not self.online:
            return
        self.online = False
        self.world.network.set_offline(self.index, True)
        if self.train_task is not None and not self.train_task.done():
            self.train_task.cancel()
        for state in self.rounds.values():
            if state.timer is not None:
                state.timer.cancel()
        self.event("offline", self.current[0], reason=reason)

    @property
    def declared_kappa(self) -> int:
        """
        The device class this node announces; without declared classes it announces nothing (one public prior, 0).
        """
        return self.kappa if self.cfg.protocol.declared_class else 0

    def cap_D(self, D: float) -> float:
        """
        Declared dataset sizes cannot be verified, so every receiver caps them at the public D_max.
        """
        return min(float(D), self.world.D_max)

    def add_member(self, pk: bytes, kappa: int, D: float) -> None:
        self.membership[pk] = (kappa, self.cap_D(D))

    def drop_member(self, pk: bytes) -> None:
        self.membership.pop(pk, None)
        if pk in self.row.row:
            self.dormant[pk] = self.row.row[pk]
            self.row.remove(pk)

    def on_join(self, msg, timing):
        self.add_member(msg.src, msg.payload["kappa"], msg.payload["D"])
        if msg.src in self.dormant and msg.src not in self.row.row:
            self.row.set(msg.src, self.dormant.pop(msg.src))

    def on_leave(self, msg, timing):
        self.drop_member(msg.src)

    def on_remove(self, msg, timing):
        self.drop_member(msg.payload["node"])

    def on_he_query(self, msg, timing):
        """
        Responder side of the CKKS protocol (once per initiator identity).
        """
        keys, cipher = self.world.public_board[msg.src]
        rng = np.random.default_rng(rng_seed(self.cfg.seed, "he", self.pk, msg.src))
        result = self.engine.respond(msg.src, cipher, keys, rng)
        self.reply(msg, "he_answer", {"cipher": result}, self.world.he_sizes["response"], "he")

    def on_he_reverse(self, msg, timing):
        """
        Reverse direction at join: the newcomer answered our public ciphertext.
        """
        self.row.set(msg.src, self.engine.decrypt(msg.payload["cipher"]))

    # ================================================================== client side

    def on_dispatch(self, msg, timing):
        round, attempt = msg.round, msg.payload["attempt"]
        if (round, attempt) <= self.current:
            return
        for key in [key for key in self.backups if key[0] < round]:
            del self.backups[key]
        if self.train_task is not None and not self.train_task.done():
            self.train_task.cancel()
            self.event("training_interrupted", self.current[0])
        self.current = (round, attempt)
        self.current_aggregator = msg.src
        self.world.set_round_capacity(self.index, round)
        if self.world.dropout(self.index, round):
            self.event("dropout", round)
            return
        self.send(msg.src, "ack", round, {"attempt": attempt}, SMALL, "control")
        if self.world.crash_now(self, round, "client"):
            return
        self.backups[(round, attempt)] = list(msg.payload.get("backups", []))
        self.train_task = self.world.spawn(self.train(round, attempt, msg.src, msg.payload["theta"]))

    async def train(self, round: int, attempt: int, aggregator: bytes, theta: bytes) -> None:
        start = self.now()
        try:
            await asyncio.sleep(self.world.compute_time(self.index, round))
        finally:  # Interrupted training still used the device
            self.world.compute_used += self.now() - start
        if not self.online or self.current != (round, attempt):
            return
        new_theta, loss = self.world.local_training(self.index, round, theta)
        payload = {"attempt": attempt, "theta": new_theta, "D": self.D, "loss": loss}
        delivery = self.send(aggregator, "update", round, payload, len(new_theta) + SMALL, "model")
        if not self.cfg.fault.client_backups or delivery is None or await delivery:
            return
        # The aggregator is unreachable: leave the update with a backup aggregator of the round
        base = hashing.model_seed(theta)
        for backup in self.backups.get((round, attempt), []):
            delivery = self.send(backup, "backup_update", round, dict(payload, base=base), len(new_theta) + SMALL,
                                 "model")
            if delivery is not None and await delivery:
                self.event("update_left_with_backup", round, backup=self.short(backup))
                return

    def on_backup_update(self, msg, timing):
        self.buffered.setdefault(msg.round, {})[msg.src] = msg.payload

    def on_loss_request(self, msg, timing):
        return self.evaluate_loss(msg)

    async def evaluate_loss(self, msg: Message) -> None:
        """
        power_of_choice: evaluate theta^(t) on the local data (costs simulated compute time) and report the loss.
        """
        await asyncio.sleep(self.world.selector.eval_time(self.index, msg.round))
        if self.online:
            loss = self.world.local_loss(self.index, msg.round, msg.payload["theta"])
            self.reply(msg, "loss_reply", {"loss": loss}, SMALL, "poc")

    def on_abort(self, msg, timing):
        state = self.rounds.get(msg.round)
        if state is not None and state.attempt == msg.payload["attempt"] and not state.closed.done():
            state.aborted = True
            if state.timer is not None:
                state.timer.cancel()
            state.closed.set_result(False)
            self.event("round_aborted", msg.round)

    # ================================================================== aggregator: running a round

    def sync_view(self, round: int) -> None:
        """
        Bring the held view's membership in line with what this node knows (joins, leaves, removals).
        """
        for pk in sorted(self.membership):
            if pk not in self.view:
                kappa, D = self.membership[pk]
                self.view.add_node(pk, kappa, D, round)
                self.world.log_membership("view_add", pk, round, by=self.index)
        for pk in self.view.members():
            if pk not in self.membership:
                self.view.remove_node(pk)
                self.world.log_membership("view_remove", pk, round, by=self.index)

    def on_handover(self, msg, timing):
        payload = msg.payload
        if self.world.crash_now(self, msg.round, "handover"):
            return
        self.reply(msg, "handover_ack", {}, SMALL, "control")
        view = pickle.loads(payload["view"])
        self.world.spawn(self.run_round(msg.round, payload["attempt"], payload["theta"], payload["clients"], view,
                                        msg.src, payload.get("late_updates", []), payload.get("backups", [])))

    async def run_round(self, round: int, attempt: int, theta_prev: bytes, clients: List[bytes],
                        view: PopulationView, predecessor: Optional[bytes], late_updates=(),
                        backups: Sequence[bytes] = ()) -> None:
        cfg = self.cfg
        self.view = view
        self.sync_view(round)
        self.world.set_round_capacity(self.index, round)
        infos = []
        for pk in clients:
            state = view.get(pk) if pk in view else None
            if state is None:  # Removed meanwhile: fall back to the class prior of its declared class
                kappa, D = self.membership.get(pk, (0, 1.0))
                prior = view.priors[kappa]
                infos.append(ClientInfo(pk, D, prior.f, prior.B, prior.sigma))
            else:
                infos.append(ClientInfo(pk, state.D, state.f, state.B, state.sigma))
        params = ClosingParams(E=cfg.train.epochs, c=cfg.train.cycles_per_sample, Z=self.world.Z, z=cfg.closing.z,
                               delta_dl=cfg.closing.delta_dl, grace=cfg.closing.grace, rho=cfg.closing.rho,
                               contention_aware=cfg.closing.contention_aware,
                               B_down_aggregator=self.world.nominal_download(self.index),
                               n_target=cfg.protocol.n, ack_factor=cfg.closing.ack_factor,
                               ack_margin=cfg.closing.ack_margin,
                               B_up_aggregator=self.world.nominal_upload(self.index),
                               down_up_ratio=cfg.env.devices.down_up_ratio)
        policy = make_policy(cfg.closing.policy, infos, self.now(), params)
        state = RoundState(round, attempt, list(clients), policy, theta_prev, predecessor, self.now(),
                           closed=self.loop.create_future(), late_updates=list(late_updates))
        self.rounds[round] = state
        self.world.round_started(self, round, attempt, clients)
        # Updates left with this node as a backup, trained from the same model, count as delivered (no measurement)
        base = hashing.model_seed(theta_prev)
        for pk, update in sorted(self.buffered.pop(round, {}).items()):
            if pk in policy.tracks and update["base"] == base:
                policy.on_ack(pk, self.now())
                policy.on_arrival(pk, self.now(), self.now(), 0.0)
                state.models[pk], state.sizes[pk], state.losses[pk] = update["theta"], self.cap_D(update["D"]), update["loss"]
                self.event("backup_update_used", round, client=self.short(pk))
        for pk in clients:
            self.send(pk, "dispatch", round, {"attempt": attempt, "theta": theta_prev, "backups": backups},
                      len(theta_prev) + SMALL + 33 * len(backups), "model")
        self.schedule(state)
        if cfg.protocol.probing.prefetch and cfg.protocol.pool_seed == "round":
            next_pool = hashing.pool(self.view.members(), round + 1, cfg.protocol.M, 0)
            for pk in next_pool:
                self.send(pk, "prefetch", round + 1, {"pool": next_pool}, SMALL + 33 * len(next_pool), "probe")
        if self.world.crash_now(self, round, "dispatch"):
            return
        ok = await state.closed
        if not ok or not self.online:
            return
        await self.after_close(state)

    def schedule(self, state: RoundState) -> None:
        if state.timer is not None:
            state.timer.cancel()
            state.timer = None
        when = state.policy.next_event()
        if when is not None and not state.closed.done():
            state.timer = self.loop.call_at(max(when, self.now()), self.on_policy_timer, state)

    def on_policy_timer(self, state: RoundState) -> None:
        state.timer = None
        if not self.online or state.closed.done():
            return
        state.policy.advance(self.now())
        self.check_closed(state)

    def check_closed(self, state: RoundState) -> None:
        if state.policy.closed and not state.closed.done():
            state.closed.set_result(True)
        else:
            self.schedule(state)

    def on_ack(self, msg, timing):
        state = self.rounds.get(msg.round)
        if state is None or state.attempt != msg.payload["attempt"]:
            return
        if state.policy.closed:
            state.policy.on_ack(msg.src, self.now())
            return
        state.policy.advance(self.now())
        state.policy.on_ack(msg.src, self.now())
        self.world.view_reset_miss(self, msg.src)
        self.check_closed(state)

    def on_update(self, msg, timing):
        state = self.rounds.get(msg.round)
        if state is None or state.attempt != msg.payload["attempt"] or state.aborted:
            return
        if not state.policy.closed:
            state.policy.advance(timing.t_last)
        status = state.policy.on_arrival(msg.src, timing.t_first, timing.t_last, timing.ingress_utilization)
        state.losses[msg.src] = msg.payload["loss"]
        if status == "accepted":
            state.models[msg.src] = msg.payload["theta"]
            state.sizes[msg.src] = self.cap_D(msg.payload["D"])
            self.check_closed(state)
        elif status == "late":
            if self.cfg.closing.late_merge.enabled:
                state.late_updates.append((msg.src, msg.payload["theta"], self.cap_D(msg.payload["D"])))
            if state.policy.closed:
                self.finalize_client(state, msg.src)
            else:
                self.check_closed(state)

    async def after_close(self, state: RoundState) -> None:
        cfg = self.cfg
        round = state.round
        policy = state.policy
        if state.timer is not None:
            state.timer.cancel()
        accepted = policy.accepted()
        models = [self.world.model_from_bytes(state.models[pk]) for pk in accepted]
        sizes = [state.sizes[pk] for pk in accepted]
        previous = self.world.model_from_bytes(state.theta_prev)
        # Optional REFL-style hook: late updates of the previous round, merged with reduced weight
        late = [u for u in state.late_updates if u[0] not in accepted and isinstance(u, tuple) and len(u) == 4]
        for pk, theta, D, _ in late:
            models.append(self.world.model_from_bytes(theta))
            sizes.append(D * cfg.closing.late_merge.weight)
        theta = self.world.model_to_bytes(fedavg(models, sizes, previous))
        # The client models and theta^(t-1) are no longer needed; holding them for every round grows without bound
        state.models.clear()
        state.theta_prev = b""

        self.view.mark_participated(state.clients, round)
        self.world.round_closed(self, state, theta)
        for pk in state.clients:
            if policy.resolvable(policy.tracks[pk], self.now()):
                self.finalize_client(state, pk)
        if state.predecessor is not None and state.predecessor != self.pk:
            self.send(state.predecessor, "round_done", round, {"attempt": state.attempt}, SMALL, "control")
        grace_end = policy.grace_end()
        if grace_end > self.now():
            self.loop.call_at(grace_end, self.finalize_remaining, state)
        else:
            self.finalize_remaining(state)
        if cfg.closing.wait_grace_before_selection and grace_end > self.now():
            await asyncio.sleep(grace_end - self.now())
        if not self.online or self.world.stopped:
            return
        await self.select_and_handover(round, theta, state)

    def finalize_client(self, state: RoundState, pk: bytes) -> None:
        if pk in state.finalized:
            return
        track = state.policy.tracks[pk]
        if not state.policy.resolvable(track, self.now()) and not state.policy.closed:
            return
        state.finalized.add(pk)
        lateness = state.policy.lateness(track)
        record = StateRecord(pk, state.round, lateness, state.policy.measurement(track), state.losses.get(pk))
        if math.isinf(lateness) and track.t_ack is None:
            self.world.view_miss(self, pk, state.round)
        self.world.client_finalized(state.round, pk, track, lateness, record)
        self.apply_or_forward(record)

    def finalize_remaining(self, state: RoundState) -> None:
        """
        At the end of the grace period: finalize every client and release the round (later arrivals are ignored, as
        they would be after the grace period anyway).
        """
        for pk in state.clients:
            if pk not in state.finalized:
                self.finalize_client(state, pk)
        if self.rounds.get(state.round) is state:
            del self.rounds[state.round]

    def apply_record(self, view: PopulationView, record: StateRecord) -> str:
        cfg = self.cfg
        status = view.apply_record(record, cfg.train.epochs, cfg.train.cycles_per_sample, self.world.Z,
                                   cfg.state.alpha, cfg.state.gamma, cfg.state.beta)
        if status != "applied":
            self.event("record_" + status, record.round, client=self.short(record.node))
        return status

    def apply_or_forward(self, record: StateRecord) -> None:
        """
        Apply a state record to the live view, or, after the handover, to the retained copy and forward it.
        """
        if self.view is not None:
            self.apply_record(self.view, record)
            return
        retained = self.retained.get(record.round + 1)
        if retained is not None:
            if retained.view is not None:
                self.apply_record(retained.view, record)
            self.send(retained.successor, "late_records", record.round, {"records": [record]}, SMALL + 64, "control")
            self.event("late_record_forwarded", record.round, client=self.short(record.node),
                       to=self.short(retained.successor))
        else:
            self.forward_downstream([record], record.round)

    def forward_downstream(self, records: List[StateRecord], round: int) -> None:
        """
        This node no longer holds the view: pass the records to whoever it handed the view to most recently.
        """
        if not self.retained:
            self.event("late_record_dropped", round, count=len(records))
            return
        latest = self.retained[max(self.retained)]
        if latest.view is not None:
            for record in records:
                self.apply_record(latest.view, record)
        self.send(latest.successor, "late_records", round, {"records": records}, SMALL + 64 * len(records), "control")

    def on_late_records(self, msg, timing):
        records = msg.payload["records"]
        if self.view is not None:
            for record in records:
                self.apply_record(self.view, record)
        else:
            self.forward_downstream(records, msg.round)

    def on_round_done(self, msg, timing):
        self.resolve_key(("round_done", msg.round, msg.payload["attempt"]), True)

    # ================================================================== selection and handover

    async def select_and_handover(self, round: int, theta: bytes, state: Optional[RoundState]) -> None:
        next_round = round + 1
        while self.online and not self.world.stopped:
            outcome = await self.world.selector.select(self, next_round, theta)
            if not self.online:
                return
            if outcome.self_serve:
                self.world.selection_done(self, next_round, outcome, None)
                attempt = self.next_attempt(next_round)
                late = self.collect_late_updates(state)
                self.world.spawn(self.run_round(next_round, attempt, theta, outcome.self_clients, self.view, None,
                                                late))
                return
            if outcome.ranked:
                if await self.handover_chain(next_round, theta, outcome, state):
                    return
            if not self.online:
                return
            self.event("selection_retry", next_round)
            await asyncio.sleep(self.cfg.fault.retry_delay)

    def next_attempt(self, round: int) -> int:
        self.attempts[round] = self.attempts.get(round, 0) + 1
        return self.attempts[round]

    def collect_late_updates(self, state: Optional[RoundState]):
        if state is None or not self.cfg.closing.late_merge.enabled:
            return []
        # Only this round's late updates (3-tuples); those inherited from the previous round were merged already
        return [(u[0], u[1], u[2], state.round) for u in state.late_updates if len(u) == 3]

    async def handover_chain(self, next_round: int, theta: bytes, outcome: SelectionOutcome,
                             state: Optional[RoundState]) -> bool:
        cfg = self.cfg.fault
        watchdog = cfg.watchdog_timeout or (self.cfg.closing.delta_dl + self.cfg.closing.grace + 300.0)
        for index, proposal in enumerate(outcome.ranked):
            if not self.online:
                return False
            attempt = self.next_attempt(next_round)
            view_bytes = pickle.dumps(self.view)
            backups = [p.aggregator for p in outcome.ranked[index + 1:index + 3]] if cfg.client_backups else []
            payload = {"theta": theta, "view": view_bytes, "clients": proposal.clients, "attempt": attempt,
                       "late_updates": self.collect_late_updates(state), "backups": backups}
            if proposal.aggregator == self.pk:
                # Selected itself as A(t+1): no transfer needed
                self.world.selection_done(self, next_round, outcome, proposal, index)
                self.world.spawn(self.run_round(next_round, attempt, theta, proposal.clients, self.view, None,
                                                payload["late_updates"], backups))
                return True
            acked = await self.request(proposal.aggregator, "handover", next_round, payload,
                                       len(theta) + len(view_bytes) + SMALL, "handover", cfg.handover_timeout)
            if not self.online:
                return False
            if not acked:
                self.event("handover_no_ack", next_round, to=self.short(proposal.aggregator), rank=index)
                continue
            self.world.selection_done(self, next_round, outcome, proposal, index)
            self.retained[next_round] = Retained(next_round, self.view, proposal.aggregator, attempt)
            self.view = None
            done = self.wait_key(("round_done", next_round, attempt))
            try:
                await asyncio.wait_for(asyncio.shield(done), watchdog)
                # The successor runs on its own copy now: keep only its address, for forwarding late records
                for old in [r for r in self.retained if r < next_round]:
                    del self.retained[old]
                if next_round in self.retained:
                    self.retained[next_round].view = None
                return True
            except asyncio.TimeoutError:
                if not self.online:
                    return False
                self.event("watchdog_timeout", next_round, aggregator=self.short(proposal.aggregator), rank=index)
                self.send(proposal.aggregator, "abort", next_round, {"attempt": attempt}, SMALL, "control")
                retained = self.retained.pop(next_round)
                self.view = retained.view  # Take the view back (late records were applied to it)
        return False

    # ------------------------------------------------------------------ candidate side

    def on_ctx_request(self, msg, timing):
        pool = msg.payload["pool"]
        expected = hashing.pool(sorted(self.membership), msg.payload["seed"], self.cfg.protocol.M, msg.payload["k"])
        if self.pk not in pool or self.pk not in expected:
            # Membership knowledge may differ slightly between nodes (churn); the request is still answered
            self.event("pool_membership_mismatch", msg.round, k=msg.payload["k"])
        r, row = self.row.report(pool)
        self.reply(msg, "ctx_reply", {"r": r, "row": row}, SMALL + 16 * len(row), "ctx")

    def on_probe(self, msg, timing):
        self.reply(msg, "probe_reply", {}, self.cfg.protocol.probing.probe_bytes, "probe")

    def on_prefetch(self, msg, timing):
        """
        Probe the predictable next pool while the current round is still running (probing.prefetch).
        """
        self.world.spawn(self.probe(msg.payload["pool"], msg.round, 0))

    async def probe(self, pool: Sequence[bytes], round: int, k: int) -> Dict[bytes, float]:
        """
        P probes to every other pool member in parallel; L_ia = median RTT (inf if every probe is lost). Measurements
        are cached per (round, k), so prefetched probes are reused and only missing members are probed.
        """
        for key in [key for key in self.probe_cache if key[0] < round - 1]:
            del self.probe_cache[key]
        cache = self.probe_cache.setdefault((round, k), {})
        P = self.cfg.protocol.probing.P
        timeout = max(1.0, 4 * self.world.L_max)
        targets = [pk for pk in pool if pk != self.pk and pk not in cache]

        async def one(pk: bytes):
            start = self.now()
            reply_time = await self.request(pk, "probe", round, {}, self.cfg.protocol.probing.probe_bytes, "probe",
                                            timeout)
            return None if reply_time is None else reply_time - start

        samples = await asyncio.gather(*(one(pk) for pk in targets for _ in range(P)))
        for i, pk in enumerate(targets):
            values = sorted(v for v in samples[i * P:(i + 1) * P] if v is not None)
            cache[pk] = float(np.median(values)) if values else math.inf
        return {pk: cache[pk] for pk in pool if pk != self.pk}

    def on_sel_request(self, msg, timing):
        return self.propose(msg)

    async def propose(self, msg: Message) -> None:
        context: SelectionContext = msg.payload["context"]
        proposal, info = await self.solve_local(context)
        if proposal is None:
            self.reply(msg, "proposal", {"decline": info}, SMALL, "propose")
        else:
            self.reply(msg, "proposal", {"proposal": proposal}, SMALL + 48 * len(proposal.clients), "propose")

    async def solve_local(self, context: SelectionContext, force: bool = False):
        """
        Probe, build N_a, solve. Returns (proposal, None) or (None, reason).
        """
        rtt = await self.probe(context.pool, context.round, context.k)
        n, L_max = context.n, context.params.L_max
        members = neighborhood_or_decline(self.pk, context.pool, rtt, L_max, context.k_nb, n,
                                          exclude=context.excluded_clients)
        if members is None:
            if not force:
                return None, "neighborhood_too_small"
            reachable = sorted((rtt[pk], pk) for pk in context.pool
                               if pk != self.pk and pk not in context.excluded_clients and math.isfinite(rtt[pk]))
            if len(reachable) < n:
                return None, "not_enough_responders"
            clients = [pk for _, pk in reachable[:n]]
            return Proposal(self.pk, clients, float("nan"), {pk: rtt[pk] for pk in clients},
                            snapshot_id=context.snapshot_id, solver="self_serve_nearest"), None
        problem = context.problem(self.pk, members, rtt)
        solver_name = self.world.solver_name
        solver = create_solver(solver_name, self.cfg.solver)
        rng = np.random.default_rng(rng_seed(self.cfg.seed, "solve", self.pk, context.round, context.k))
        try:
            result = solver.solve(problem, budget_from_config(self.cfg.solver), rng)
        except SolverRefused as exc:
            return None, "solver_refused: %s" % exc
        clients = result.ids(problem)
        proposal = Proposal(self.pk, clients, result.utility, {pk: rtt[pk] for pk in clients}, result.components,
                            context.snapshot_id, len(members), result.evaluations, result.wall_time, solver_name,
                            result.certificate)
        self.world.log_proposal(self, context, proposal)
        return proposal, None

    # ------------------------------------------------------------------ aggregator side (COMPASS selection)

    def pool_seed(self, round: int, theta: bytes):
        return round if self.cfg.protocol.pool_seed == "round" else hashing.model_seed(theta)

    async def collect(self, round: int, k: int, pool: List[bytes], seed) -> Dict[bytes, Tuple[float, Dict]]:
        timeout = self.cfg.protocol.collect_timeout

        async def ask(pk: bytes):
            if pk == self.pk:
                return pk, self.row.report(pool)
            reply = await self.request(pk, "ctx_request", round, {"pool": pool, "k": k, "seed": seed},
                                       SMALL + 33 * len(pool),
                                       "ctx", timeout)
            return pk, None if reply is None else (reply["r"], reply["row"])

        replies = await asyncio.gather(*(ask(pk) for pk in pool))
        out = {}
        for pk, report in replies:
            if report is None:
                self.world.view_miss(self, pk, round)
            else:
                self.world.view_reset_miss(self, pk)
                out[pk] = report
        return out

    def build_context(self, round: int, k: int, reports: Dict, stats: Dict) -> Optional[SelectionContext]:
        cfg = self.cfg
        tolerance = consistency_tolerance(cfg.protocol.sim_tol, cfg.protocol.sim_tol_k,
                                          cfg.he.dp.sigma if cfg.he.dp.enabled else 0.0)
        if cfg.protocol.exclude_flagged:
            reports = {pk: rep for pk, rep in reports.items()
                       if pk not in self.view or self.view.get(pk).flags < cfg.protocol.flag_threshold}
        reports = {pk: rep for pk, rep in reports.items() if pk in self.view}
        pool_sim = assemble(reports, tolerance)
        for i, j, diff in pool_sim.flags:
            self.view.add_flag(i)
            self.view.add_flag(j)
            self.event("similarity_flag", round, a=self.short(i), b=self.short(j), diff=diff)
        stats.update({"flags": len(pool_sim.flags), "one_sided_pairs": pool_sim.one_sided,
                      "dropped_unknown_pairs": [self.short(pk) for pk in pool_sim.dropped]})
        members = pool_sim.members
        if len(members) < 2:
            return None
        snapshot = self.view.snapshot(round, members, tag=(k,))
        entries = [snapshot.get(pk) for pk in members]
        params = self.world.utility_params
        eligible = [pk for pk, e in zip(members, entries)
                    if e.f >= self.world.f_min and e.B >= self.world.B_min
                    and not (cfg.protocol.forbid_repeat_aggregator and pk == self.pk)]
        excluded = [self.pk] if cfg.protocol.forbid_aggregator_as_client else []
        return SelectionContext(
            round=round, k=k, snapshot_id=snapshot.snapshot_id, pool=list(members), S=pool_sim.S, r=pool_sim.r,
            f=np.array([e.f for e in entries]), B=np.array([e.B for e in entries]),
            Q=np.array([e.Q for e in entries]), stale=np.array([round - e.tau for e in entries], dtype=float),
            net_sum=estimate_network_sum(pool_sim.r, len(self.view)), params=params, n=self.world.group_size,
            k_nb=cfg.protocol.k_nb, eligible=eligible, excluded_clients=excluded)

    async def select_compass(self, next_round: int, theta: bytes) -> SelectionOutcome:
        cfg = self.cfg
        outcome = SelectionOutcome(stats={"attempts": []})
        context = None
        seed = self.pool_seed(next_round, theta)
        for k in range(cfg.fault.max_resamples + 1):
            if not self.online:
                return outcome
            stats = {"k": k, "start": self.now()}
            pool = hashing.pool(self.view.members(), seed, cfg.protocol.M, k)
            reports = await self.collect(next_round, k, pool, seed)
            stats.update({"pool": len(pool), "responders": len(reports)})
            context = self.build_context(next_round, k, reports, stats) if self.view is not None else None
            if context is None:
                stats["eligible"] = 0
                outcome.stats["attempts"].append(stats)
                continue
            stats["eligible"] = len(context.eligible)
            proposals, declines = await self.request_proposals(context)
            valid, rejected = [], []
            for proposal in proposals:
                verdict = verify(context, proposal, cfg.protocol.verify_tol)
                (valid if verdict.ok else rejected).append(proposal)
                if not verdict.ok:
                    self.event("proposal_rejected", next_round, candidate=self.short(proposal.aggregator),
                               reason=verdict.reason)
            stats.update({"proposals": len(proposals), "declines": declines, "rejected": len(rejected),
                          "valid": len(valid), "end": self.now()})
            outcome.stats["attempts"].append(stats)
            if valid:
                outcome.ranked = rank(valid)
                return outcome
        # Last resort: serve round t+1 ourselves
        if context is not None:
            proposal, reason = await self.solve_local(context, force=True)
            if proposal is not None:
                outcome.self_serve = True
                outcome.self_clients = proposal.clients
                outcome.ranked = [proposal]
                outcome.stats["self_serve"] = proposal.solver
                return outcome
            outcome.stats["self_serve_failed"] = reason
        return outcome

    async def request_proposals(self, context: SelectionContext) -> Tuple[List[Proposal], int]:
        timeout = self.cfg.protocol.propose_timeout
        size = context.size_bytes()

        async def ask(pk: bytes):
            if pk == self.pk:
                proposal, reason = await self.solve_local(context)
                return proposal
            reply = await self.request(pk, "sel_request", context.round, {"context": context}, size, "propose",
                                       timeout)
            if reply is None or "decline" in reply:
                return None
            return reply["proposal"]

        results = await asyncio.gather(*(ask(pk) for pk in context.eligible))
        proposals = [p for p in results if p is not None]
        return proposals, len(results) - len(proposals)

    # ------------------------------------------------------------------ bootstrap

    async def bootstrap(self, theta0: bytes, view: PopulationView) -> None:
        """
        A(0): holds theta^(0) and the initial view, and runs the selection of round 1 from context collection.
        """
        self.view = view
        self.event("bootstrap", 0)
        await self.select_and_handover(0, theta0, None)
