"""
The simulated world of a COMPASS-FL run: nodes, devices, latencies, the fair-sharing network, data, models, failures,
and all logging. Nodes only learn about the world through messages and their timestamps; the ground truth is used
only to generate those timings (and by the oracle_central baseline, which is a centralized upper bound by design).

Simulated time comes from the modeled compute, transfer and latency times only (``DiscreteLoop``), never from real
GPU time.
"""
import asyncio
import copy
import io
import logging
import math
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

from dlsim.compass.community import CompassNode, Message, SMALL
from dlsim.compass.core import hashing
from dlsim.compass.core.hashing import rng_seed
from dlsim.compass.core.similarity import cosine, normalize
from dlsim.compass.core.utility import UtilityParams
from dlsim.compass.core.view import PopulationView
from dlsim.compass.data import FederatedData, build_model, evaluate_model, train_local
from dlsim.compass.env.devices import DeviceModel
from dlsim.compass.env.latency import LatencyModel
from dlsim.compass.env.network import FairNetwork, Timing
from dlsim.compass.he import CKKSParams, SimilarityEngine, serialized_size
from dlsim.compass.runlog import RunLogger

logger = logging.getLogger("CompassWorld")


def node_public_key(index: int) -> bytes:
    """
    The same deterministic keys as the Plexus simulation (peer ids start at 1), so hash orders are comparable.
    """
    from ipv8.keyvault.crypto import default_eccrypto
    key_str = chr(index + 1).encode() * 1000
    key = default_eccrypto.key_from_private_bin(b"LibNaCLSK:" + key_str[:68])
    return key.pub().key_to_bin()


class World:

    def __init__(self, config, run_dir: str) -> None:
        self.config = config
        self.seed = config.seed
        self.run_dir = run_dir
        self.log = RunLogger(run_dir)
        self.stopped = False
        self.rid = 0
        N = config.network.N
        self.N = N

        # Environment
        self.devices = DeviceModel(config.env.devices, N, self.seed)
        self.devices.c = config.train.cycles_per_sample
        kappas = [d.kappa for d in self.devices.devices]
        self.latency = LatencyModel(config.env.latency, kappas, self.seed)
        self.network = FairNetwork(N, config.network.small_message_bytes)
        for d in self.devices.devices:
            down = 1e15 if config.network.large_aggregator_downlink else d.B_down
            self.network.set_capacity(d.index, d.B_up, down)

        # Data and model
        self.data = FederatedData(config, N, self.seed)
        self.num_classes = self.data.num_classes
        self.model0 = build_model(config.data.dataset, config.model.architecture, self.num_classes, self.seed)
        self.theta0 = self.model_to_bytes(self.model0)
        self.Z = float(len(self.theta0))
        self.y_hat = [normalize(n.label_counts) for n in self.data.nodes]
        D = {i: n.size for i, n in enumerate(self.data.nodes)}
        p = config.protocol
        if p.D_max > 0:
            self.D_max = p.D_max
        elif p.D_max_quantile < 1:
            self.D_max = float(np.quantile(list(D.values()), p.D_max_quantile))
        else:
            self.D_max = math.inf
        self.priors = self.devices.priors(config.train.epochs, D, self.Z, p.declared_class, self.D_max)

        # Network-wide constants
        base_f = np.array([d.f for d in self.devices.devices])
        base_b = np.array([d.B_up for d in self.devices.devices])
        self.L_max = p.L_max if p.L_max > 0 else self.latency.percentile(p.L_max_percentile)
        self.f_min = p.f_min if p.f_min >= 0 else float(np.percentile(base_f, p.f_min_percentile))
        self.B_min = p.B_min if p.B_min >= 0 else float(np.percentile(base_b, p.B_min_percentile))
        self.f_ref = config.utility.f_ref if config.utility.f_ref > 0 else float(np.percentile(base_f, 75))
        self.utility_params = UtilityParams.from_config(config, self.f_ref, self.L_max)
        self.group_size = config.group_size

        # HE
        self.he_sizes = {"response": SMALL, "query": SMALL, "keys": 0}
        self.public_board: Dict[bytes, tuple] = {}
        self.ckks_params = CKKSParams.from_config(config.he.ckks)

        # Nodes
        self.pks = [node_public_key(i) for i in range(N)]
        self.index = {pk: i for i, pk in enumerate(self.pks)}
        self.nodes: List[CompassNode] = [
            CompassNode(self, i, self.pks[i], kappas[i], self.data.nodes[i].size, self.data.nodes[i].label_counts)
            for i in range(N)]
        if config.he.mode == "ckks":
            self._measure_he_sizes()

        from dlsim.compass.selectors import make_selector
        self.selector = make_selector(config.protocol.selector, self)
        self.solver_name = "random" if config.protocol.selector == "compass_random" else config.solver.name

        # Bookkeeping for logs
        self.traffic = defaultdict(lambda: [0, 0])          # (round, category) -> [messages, bytes]
        self.totals = defaultdict(lambda: [0, 0])
        self.round_records: Dict[int, Dict] = {}
        self.selection_records: Dict[int, Dict] = {}
        self.proposal_stats: Dict[int, Dict] = defaultdict(lambda: {"evaluations": [], "solve_time": [],
                                                                     "neighborhood": []})
        self.completed_rounds = 0
        self.compute_used = 0.0                             # Device-seconds spent training (RTA)
        self.last_aggregator: Optional[int] = None
        self.consecutive_aggregator = 0
        self.last_progress = 0.0
        self.measured_truth: Dict[int, Dict] = {}
        self.evaluator_loader = None
        self.best_accuracy = -1.0
        self.on_finish = None
        self.fired_failures: set = set()

    # ================================================================== helpers

    def index_of(self, pk: bytes) -> int:
        return self.index.get(pk, -1)

    def next_rid(self) -> int:
        self.rid += 1
        return self.rid

    @property
    def loop(self):
        return asyncio.get_event_loop()

    def now(self) -> float:
        return self.loop.time()

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)

        def done(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                self.loop.call_exception_handler({"exception": exc, "message": "task failed"})

        task.add_done_callback(done)
        return task

    def make_engine(self, label_counts) -> SimilarityEngine:
        dp = self.config.he.dp
        return SimilarityEngine(self.config.he.mode, label_counts, len(label_counts),
                                CKKSParams.from_config(self.config.he.ckks), dp.sigma if dp.enabled else 0.0)

    def model_to_bytes(self, model) -> bytes:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

    def model_from_bytes(self, data: bytes):
        model = copy.deepcopy(self.model0)
        model.load_state_dict(torch.load(io.BytesIO(data)))
        return model

    # ================================================================== transport

    def send(self, src: CompassNode, dst_pk: bytes, msg: Message):
        dst = self.nodes[self.index[dst_pk]]
        entry = self.traffic[(msg.round, msg.category)]
        entry[0] += 1
        entry[1] += msg.size
        total = self.totals[msg.category]
        total[0] += 1
        total[1] += msg.size
        if dst is src:
            now = self.now()
            self.loop.call_soon(dst.on_message, msg, Timing(True, now, now, now, 0.0, msg.size))
            return None
        return self.spawn(self._deliver(src, dst, msg))

    async def _deliver(self, src: CompassNode, dst: CompassNode, msg: Message) -> bool:
        if not src.online:
            return False
        delay = self.latency.one_way(src.index, dst.index)
        timing = await self.network.transfer(src.index, dst.index, msg.size, delay)
        if timing.ok and dst.online and not self.stopped:
            dst.on_message(msg, timing)
            return True
        return False

    # ================================================================== ground truth used to generate timings

    def set_round_capacity(self, index: int, round: int) -> None:
        up, down = self.devices.true_bandwidth(index, round)
        if self.config.network.large_aggregator_downlink:
            down = 1e15
        self.network.set_capacity(index, up, down)

    def nominal_upload(self, index: int) -> float:
        return self.devices.devices[index].B_up

    def nominal_download(self, index: int) -> float:
        if self.config.network.large_aggregator_downlink:
            return 1e15
        return self.devices.devices[index].B_down

    def compute_time(self, index: int, round: int) -> float:
        cfg = self.config.train
        return cfg.epochs * self.data.nodes[index].size * cfg.cycles_per_sample / self.devices.true_f(index, round)

    def dropout(self, index: int, round: int) -> bool:
        return self.devices.dropout(index, round)

    def local_training(self, index: int, round: int, theta: bytes):
        cfg = self.config
        if cfg.data.bypass_training:
            rng = np.random.default_rng(rng_seed(self.seed, "loss", index, round))
            mean = np.mean(self.y_hat, axis=0)
            mismatch = 1.0 - cosine(self.y_hat[index], mean)
            return theta, float(max(1e-3, (0.5 + mismatch) * math.exp(-0.01 * round) * (1 + 0.1 * rng.normal())))
        model = self.model_from_bytes(theta)
        generator = torch.Generator()
        generator.manual_seed(rng_seed(self.seed, "train", index, round) % (2 ** 63))
        torch.manual_seed(rng_seed(self.seed, "dropout_mask", index, round) % (2 ** 31))
        loader = self.data.loader(index, cfg.train.batch_size, generator)
        model, loss, _ = train_local(model, loader, cfg.train.epochs, cfg.train.lr, cfg.train.momentum,
                                     cfg.train.weight_decay, cfg.train.max_batches_per_epoch, cfg.train.device)
        return self.model_to_bytes(model), loss

    def local_loss(self, index: int, round: int, theta: bytes) -> float:
        if self.config.data.bypass_training:
            return self.local_training(index, round, theta)[1]
        from dlsim.compass.data import local_loss
        generator = torch.Generator()
        generator.manual_seed(rng_seed(self.seed, "loss_eval", index, round) % (2 ** 63))
        loader = self.data.loader(index, self.config.train.batch_size, generator)
        return local_loss(self.model_from_bytes(theta), loader, self.config.train.max_batches_per_epoch)

    # ================================================================== failures and churn

    def crash_now(self, node: CompassNode, round: int, where: str) -> bool:
        """
        Failure injection at a protocol point. Scripted failures fire once; returns True if the node crashed now.
        """
        failures = self.config.env.failures
        for number, failure in enumerate(failures.scripted):
            if failure.round != round or number in self.fired_failures or not node.online:
                continue
            if failure.target == "aggregator":
                hit = where == failure.when
            else:
                hit = where == "client" and str(node.index) == str(failure.target)
            if not hit:
                continue
            self.fired_failures.add(number)
            reason = "scripted_%s" % where
            if failure.delay > 0:
                self.loop.call_later(failure.delay, self.crash, node, failure.downtime, reason)
                return False
            self.crash(node, failure.downtime, reason)
            return True
        if where in ("client", "dispatch") and any(failures.crash_prob) and \
                self.devices.crash(node.index, round, failures.crash_prob):
            self.crash(node, failures.crash_downtime, "random_%s" % where)
            return True
        return False

    def crash(self, node: CompassNode, downtime: float, reason: str) -> None:
        if not node.online:
            return
        node.go_offline(reason)
        self.log_membership("crash", node.pk, node.current[0], reason=reason, downtime=downtime)
        if downtime > 0:
            self.loop.call_later(downtime, lambda: self.spawn(self.join(node, rejoin=True)))

    async def join(self, node: CompassNode, rejoin: bool = False) -> None:
        """
        A node (re)joins: it learns the membership out of band, announces itself, and runs the join-time similarity
        protocol in both directions.
        """
        node.go_online()
        members = [other for other in self.nodes if other.online and other is not node]
        for other in members:
            node.add_member(other.pk, other.declared_kappa, other.D)
        node.add_member(node.pk, node.declared_kappa, node.D)
        self.log_membership("rejoin" if rejoin else "join", node.pk, 0)
        for other in members:
            node.send(other.pk, "join", 0, {"kappa": node.declared_kappa, "D": node.D}, SMALL, "membership")
        await self.similarity_exchange(node, members, newcomer=True)

    async def similarity_exchange(self, node: CompassNode, members: List[CompassNode], newcomer: bool) -> None:
        if self.config.he.mode == "plaintext":
            for other in members:
                rng_a = np.random.default_rng(rng_seed(self.seed, "dp", node.index, other.index))
                node.row.set(other.pk, node.engine.direct(other.engine.y_hat, rng_a))
                if newcomer:
                    rng_b = np.random.default_rng(rng_seed(self.seed, "dp", other.index, node.index))
                    other.row.set(node.pk, other.engine.direct(node.engine.y_hat, rng_b))
            return
        # CKKS: publish keys and ciphertext once, query everybody, and (newcomer) answer everybody's public ciphertext
        self.public_board[node.pk] = (node.engine.ckks.public, node.engine.public_ciphertext())
        self.totals["he_publish"][0] += 1
        self.totals["he_publish"][1] += self.he_sizes["keys"]
        timeout = self.config.he.exchange_timeout

        async def query(other: CompassNode):
            reply = await node.request(other.pk, "he_query", 0, {}, self.he_sizes["query"], "he", timeout)
            if reply is not None and reply["cipher"] is not None:
                node.row.set(other.pk, node.engine.decrypt(reply["cipher"]))

        await asyncio.gather(*(query(other) for other in members))
        if newcomer:
            for other in members:
                if other.pk not in self.public_board:
                    continue
                keys, cipher = self.public_board[other.pk]
                rng = np.random.default_rng(rng_seed(self.seed, "he", node.pk, other.pk))
                result = node.engine.respond(other.pk, cipher, keys, rng)
                if result is not None:
                    node.send(other.pk, "he_reverse", 0, {"cipher": result}, self.he_sizes["response"], "he")

    def _measure_he_sizes(self) -> None:
        probe = self.nodes[0].engine
        other = self.nodes[1].engine
        cipher = probe.public_ciphertext()
        response = other.ckks.respond(cipher, probe.ckks.public, other.y_hat, np.random.default_rng(0))
        self.he_sizes = {"query": serialized_size(cipher) + SMALL, "response": serialized_size(response) + SMALL,
                         "keys": serialized_size(probe.ckks.public.public_key) +
                         serialized_size(probe.ckks.public.galois_keys) + serialized_size(cipher)}

    # ================================================================== view-side bookkeeping done by aggregators

    def view_miss(self, node: CompassNode, pk: bytes, round: int) -> None:
        view = node.view
        if view is None or pk not in view:
            return
        count = view.record_miss(pk)
        if count >= self.config.fault.max_missed:
            view.remove_node(pk)
            node.drop_member(pk)
            self.log_membership("removed_after_misses", pk, round, by=node.index, misses=count)
            for member in sorted(node.membership):
                if member != node.pk:
                    node.send(member, "remove", round, {"node": pk}, SMALL, "membership")

    def view_reset_miss(self, node: CompassNode, pk: bytes) -> None:
        if node.view is not None and pk in node.view:
            node.view.reset_miss(pk)

    # ================================================================== logging hooks

    def log_event(self, kind: str, round: int, **details) -> None:
        self.log.write("events", dict(time=self.now(), kind=kind, round=round, **details))

    def log_membership(self, kind: str, pk: bytes, round: int, **details) -> None:
        self.log.write("membership", dict(time=self.now(), kind=kind, node=self.index_of(pk), round=round, **details))

    def log_proposal(self, node: CompassNode, context, proposal) -> None:
        stats = self.proposal_stats[context.round]
        stats["evaluations"].append(proposal.evaluations)
        stats["solve_time"].append(proposal.solve_time)
        stats["neighborhood"].append(proposal.neighborhood_size)
        self.log.write("proposals", {"round": context.round, "k": context.k, "candidate": node.index,
                                     "neighborhood": proposal.neighborhood_size,
                                     "clients": [self.index_of(pk) for pk in proposal.clients],
                                     "utility": proposal.utility, "components": proposal.components,
                                     "evaluations": proposal.evaluations, "solve_time": proposal.solve_time,
                                     "solver": proposal.solver, "certificate": proposal.certificate})

    def selection_done(self, node: CompassNode, round: int, outcome, proposal, rank_index: int = 0) -> None:
        attempts = outcome.stats.get("attempts", [])
        record = {
            "selected_by": node.index,
            "resamples": max(0, len(attempts) - 1),
            "self_serve": outcome.self_serve,
            "handover_rank": rank_index,
            "candidates": sum(a.get("pool", 0) for a in attempts),
            "responders": sum(a.get("responders", 0) for a in attempts),
            "eligible": sum(a.get("eligible", 0) for a in attempts),
            "proposals": sum(a.get("proposals", 0) for a in attempts),
            "declines": sum(a.get("declines", 0) for a in attempts),
            "rejected": sum(a.get("rejected", 0) for a in attempts),
            "flags": sum(a.get("flags", 0) for a in attempts),
            "selection_duration": (attempts[-1].get("end", self.now()) - attempts[0]["start"]) if attempts else 0.0,
            "selector": self.config.protocol.selector,
        }
        if proposal is not None:
            record.update({"utility": proposal.utility, "components": proposal.components,
                           "winner_neighborhood": proposal.neighborhood_size})
        elif outcome.ranked:
            record.update({"utility": outcome.ranked[0].utility, "components": outcome.ranked[0].components})
        stats = self.proposal_stats.get(round)
        if stats and stats["evaluations"]:
            record.update({"solver_evaluations_total": int(sum(stats["evaluations"])),
                           "solver_evaluations_max": int(max(stats["evaluations"])),
                           "solver_time_total": float(sum(stats["solve_time"])),
                           "solver_time_max": float(max(stats["solve_time"]))})
        record.update(outcome.stats.get("extra", {}))
        self.selection_records[round] = record
        self.log.write("selections", dict(round=round, attempts=attempts, **{k: v for k, v in record.items()}))

    def round_started(self, node: CompassNode, round: int, attempt: int, clients: List[bytes]) -> None:
        if node.index == self.last_aggregator and round > 1:
            self.consecutive_aggregator += 1
        record = {"round": round, "attempt": attempt, "aggregator": node.index,
                  "same_aggregator_as_previous": node.index == self.last_aggregator,
                  "consecutive_aggregator_total": self.consecutive_aggregator,
                  "clients": [self.index_of(pk) for pk in clients], "dispatch_time": self.now(),
                  "policy": self.config.closing.policy, "client_detail": {}}
        self.round_records[round] = record
        self.last_aggregator = node.index

    def round_closed(self, node: CompassNode, state, theta: bytes) -> None:
        record = self.round_records.get(state.round, {})
        policy = state.policy
        now = self.now()
        record.update({"close_time": now, "duration": now - state.dispatch, "accepted": len(policy.accepted()),
                       "membership": len(node.view) if node.view is not None else None})
        for pk, track in policy.tracks.items():
            record["client_detail"][str(self.index_of(pk))] = {
                "deadline": track.deadline, "raw_deadline": track.raw_deadline, "k": track.k, "t_ack": track.t_ack,
                "t_last": track.t_last, "aggregated": track.aggregated, "status": track.status}
        record["selection"] = self.selection_records.get(state.round, {})
        record["traffic"] = {cat: {"messages": self.traffic[(state.round, cat)][0],
                                   "bytes": self.traffic[(state.round, cat)][1]}
                             for cat in ("ctx", "propose", "probe", "model", "handover", "control", "poc")}
        record["estimate_error"] = self.estimate_errors(node.view, state.clients)
        record["compute_total"] = self.compute_used
        record["bytes_total"] = sum(v[1] for v in self.totals.values())
        interval = self.config.run.accuracy_interval
        if interval and state.round % interval == 0:
            record["accuracy"], record["test_loss"] = self.evaluate(theta)
        self.completed_rounds += 1
        self.last_progress = now
        print("Round %d closed at t=%.2f by node %d (%d/%d aggregated%s)" % (
            state.round, now, node.index, len(policy.accepted()), len(state.clients),
            ", acc=%.4f" % record["accuracy"] if record.get("accuracy") is not None else ""))
        rounds = self.config.run.rounds
        max_time = self.config.run.max_sim_time
        if (rounds and self.completed_rounds >= rounds) or (max_time and now >= max_time):
            self.finish()

    def client_finalized(self, round: int, pk: bytes, track, lateness: float, record) -> None:
        rec = self.round_records.get(round)
        if rec is None:
            return
        detail = rec["client_detail"].setdefault(str(self.index_of(pk)), {})
        detail.update({"lateness": lateness, "measured": record.measurement is not None,
                       "censored": bool(record.measurement is not None and record.measurement.censored),
                       "t_last": track.t_last, "status": track.status})
        if "close_time" in rec and all("lateness" in d for d in rec["client_detail"].values()):
            self._write_round(round)

    def _write_round(self, round: int) -> None:
        rec = self.round_records.pop(round, None)
        if rec is None:
            return
        details = list(rec["client_detail"].values())
        rec["on_time"] = sum(1 for d in details if d.get("aggregated"))
        rec["late"] = sum(1 for d in details if not d.get("aggregated") and d.get("t_last") is not None)
        rec["missing"] = sum(1 for d in details if d.get("t_last") is None)
        measured = [d for d in details if d.get("measured")]
        rec["censored_fraction"] = (sum(1 for d in measured if d.get("censored")) / len(measured)) if measured else 0.0
        self.log.write("rounds", rec)

    def estimate_errors(self, view: Optional[PopulationView], clients: List[bytes]) -> Dict:
        if view is None:
            return {}

        def errors(nodes):
            ef, eb = [], []
            for pk in nodes:
                if pk not in view or view.get(pk).m == 0:
                    continue
                device = self.devices.devices[self.index_of(pk)]
                ef.append(abs(view.get(pk).f - device.f) / device.f)
                eb.append(abs(view.get(pk).B - device.B_up) / device.B_up)
            return (float(np.mean(ef)) if ef else None, float(np.mean(eb)) if eb else None, len(ef))

        f_round, b_round, n_round = errors(clients)
        f_all, b_all, n_all = errors(view.members())
        return {"f_rel_err_clients": f_round, "B_rel_err_clients": b_round, "f_rel_err_measured": f_all,
                "B_rel_err_measured": b_all, "measured_nodes": n_all}

    def evaluate(self, theta: bytes):
        if self.config.data.bypass_training or self.data.testset is None:
            return None, None
        if self.evaluator_loader is None:
            self.evaluator_loader = self.data.test_loader()
        model = self.model_from_bytes(theta)
        accuracy, loss = evaluate_model(model, self.evaluator_loader, self.config.run.accuracy_device)
        if self.config.run.store_best_model and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(self.run_dir, "best.model"))
        return accuracy, loss

    # ================================================================== lifecycle

    def finish(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        for round in sorted(self.round_records):
            self._write_round(round)
        self.log.write("summary", {"time": self.now(), "completed_rounds": self.completed_rounds,
                                   "compute_total": self.compute_used,
                                   "held_state": self.held_state(),
                                   "traffic_totals": {k: {"messages": v[0], "bytes": v[1]}
                                                      for k, v in sorted(self.totals.items())},
                                   "he_stats": self.he_stats()})
        self.log.flush()
        if self.on_finish:
            self.on_finish()

    def held_state(self) -> Dict:
        """
        Per-round state still held by the nodes at the end (a check that it does not grow with the run length).
        """
        return {"rounds": sum(len(n.rounds) for n in self.nodes),
                "retained": sum(len(n.retained) for n in self.nodes),
                "retained_views": sum(1 for n in self.nodes for r in n.retained.values() if r.view is not None),
                "probe_cache": sum(len(n.probe_cache) for n in self.nodes),
                "model_bytes_in_rounds": sum(len(m) for n in self.nodes for s in n.rounds.values()
                                             for m in s.models.values())}

    def he_stats(self) -> Dict:
        totals = defaultdict(float)
        for node in self.nodes:
            for key, value in node.engine.stats.items():
                totals[key] += value
        return dict(totals)

    def check_liveness(self) -> None:
        if self.stopped:
            return
        if self.now() - self.last_progress > self.config.run.liveness_check_interval:
            self.finish()
            raise RuntimeError("Liveness violated: no round completed in %.0f simulated seconds"
                               % self.config.run.liveness_check_interval)
        self.loop.call_later(self.config.run.liveness_check_interval / 4, self.check_liveness)

    async def start(self) -> None:
        cfg = self.config
        initial = cfg.network.initially_active or self.N
        active = self.nodes[:initial]
        for node in active:
            node.go_online()
            for other in active:
                node.add_member(other.pk, other.declared_kappa, other.D)
            self.log_membership("initial", node.pk, 0)
        for node in active:
            await self.similarity_exchange(node, [o for o in active if o is not node], newcomer=False)
        for time_offset, count in cfg.env.failures.joins:
            reserve = [n for n in self.nodes if not n.online]
            for node in reserve[:int(count)]:
                self.loop.call_later(float(time_offset), lambda n=node: self.spawn(self.join(n)))
        self.apply_availability_traces()
        view = PopulationView(self.priors, cfg.state.Q0)
        for node in active:
            view.add_node(node.pk, node.declared_kappa, node.cap_D(node.D), 0)
        members = view.members()
        seed = 1 if cfg.protocol.pool_seed == "round" else hashing.model_seed(self.theta0)
        bootstrap_pk = hashing.pool(members, seed, cfg.protocol.M)[0]
        self.log_event("bootstrap", 0, node=self.index_of(bootstrap_pk))
        self.last_progress = self.now()
        self.loop.call_later(cfg.run.liveness_check_interval / 4, self.check_liveness)
        await self.nodes[self.index[bootstrap_pk]].bootstrap(self.theta0, view)

    def apply_availability_traces(self) -> None:
        """
        Optional FedScale availability traces (as in Plexus: a pickle {device: {"active": [t...], "inactive": [t...],
        "finish_time": T}}): every node follows the trace of a seeded random device; going inactive is a crash
        (detected by aggregators through missed responses), becoming active again is a rejoin.
        """
        path = self.config.env.failures.availability_traces
        if not path:
            return
        with open(path, "rb") as handle:
            traces = pickle.load(handle)
        rng = np.random.default_rng(rng_seed(self.seed, "availability"))
        keys = sorted(traces)
        chosen = [keys[int(i)] for i in rng.choice(len(keys), size=self.N, replace=False)]
        for node, key in zip(self.nodes, chosen):
            trace = traces[key]
            for t in trace.get("inactive", []):
                self.loop.call_later(float(t), self.crash, node, 0.0, "availability_trace")
            for t in trace.get("active", []):
                if t > 0:
                    self.loop.call_later(float(t), lambda n=node: None if n.online else
                                         self.spawn(self.join(n, rejoin=True)))
        self.devices.stats["availability_traces"] = {"path": path, "devices": len(keys), "sampled": self.N}

    def environment_stats(self) -> Dict:
        return {"devices": {k: v for k, v in self.devices.stats.items() if k != "fedscale_sampled_ids"},
                "fedscale_sampled_ids": self.devices.stats.get("fedscale_sampled_ids"),
                "latency": self.latency.stats, "data": self.data.stats,
                "constants": {"L_max": self.L_max, "f_min": self.f_min, "B_min": self.B_min, "f_ref": self.f_ref,
                              "Z_bytes": self.Z, "group_size": self.group_size},
                "nodes": [{"index": d.index, "kappa": d.kappa, "f": d.f, "B_up": d.B_up, "B_down": d.B_down,
                           "D": self.data.nodes[d.index].size, "city": self.latency.cities[d.index],
                           "fedscale_id": d.trace_id} for d in self.devices.devices],
                "he_sizes": self.he_sizes}
