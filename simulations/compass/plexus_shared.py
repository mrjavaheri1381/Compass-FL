"""
The ``plexus`` baseline: the ORIGINAL Plexus sampling and aggregation logic (``dlsim/plexus/community.py``, unchanged)
running in the shared COMPASS-FL simulator, i.e. with the same devices (FedScale profiles), latencies (WonderNetwork
RTTs, one-way = L/2 unless ``plexus_compat``), fair-sharing network, data partition, initial model, timing model
(E D c / f) and logging as every other selector.

Only the environment hooks are replaced:
- model transfers go through the shared ``FairNetwork`` instead of the original bandwidth scheduler;
- local training takes the shared simulated time and trains on the shared partition;
- IPv8 pings use the shared mean one-way latencies.
Plexus' own rules are kept: hash-ordered samples of available peers, aggregators of sample t+1 sorted by bandwidth,
success_fraction / aggregation_timeout closing, and uniform (unweighted) model averaging.
"""
import asyncio
import json
import os
import pickle
from asyncio import ensure_future
from base64 import b64encode
from binascii import hexlify
from random import Random
from typing import Dict, List, Optional

import numpy as np
import torch

from ipv8.configuration import ConfigBuilder
from ipv8_service import IPv8
from simulation.simulation_endpoint import SimulationEndpoint

from dlsim.compass.core.hashing import rng_seed
from dlsim.compass.core.similarity import cosine
from dlsim.compass.data import FederatedData, build_model, evaluate_model, train_local
from dlsim.compass.env.devices import DeviceModel
from dlsim.compass.env.latency import LatencyModel
from dlsim.compass.env.network import FairNetwork
from dlsim.compass.runlog import RunLogger
from dlsim.core import NodeMembershipChange
from dlsim.core.models import serialize_model, unserialize_model
from dlsim.core.session_settings import DFLSettings, LearningSettings, SessionSettings
from dlsim.plexus.community import PlexusCommunity
from dlsim.util.eva.result import TransferResult


class PlexusSharedCommunity(PlexusCommunity):
    world = None  # Set by PlexusWorld

    async def eva_send_model(self, round, model, type, population_view, peer):
        serialized_model = serialize_model(model)
        serialized_view = pickle.dumps(population_view)
        self.bw_out_stats["bytes"]["model"] += len(serialized_model)
        self.bw_out_stats["bytes"]["view"] += len(serialized_view)
        self.bw_out_stats["num"]["model"] += 1
        self.bw_out_stats["num"]["view"] += 1
        data = serialized_model + serialized_view
        info = json.dumps({"round": round, "type": type, "model_data_len": len(serialized_model)}).encode()
        return await self.world.transfer(self, peer, info, data, round, type)

    async def send_aggregated_model_to_participants(self, participants, model, sample_index):
        self.world.plexus_sample(sample_index, participants)
        return await super().send_aggregated_model_to_participants(participants, model, sample_index)

    async def aggregator_complete_round(self, model_round: int, index: int):
        contributors = list(self.aggregations[index].incoming_trained_models.keys())
        self.world.plexus_aggregated(self, model_round, contributors)
        await super().aggregator_complete_round(model_round, index)


class PlexusWorld:

    def __init__(self, config, run_dir: str) -> None:
        if config.data.dataset not in ("cifar10", "femnist"):
            raise ValueError("The Plexus port supports the datasets of the original code (cifar10, femnist)")
        self.config = config
        self.seed = config.seed
        self.log = RunLogger(run_dir)
        self.run_dir = run_dir
        self.N = config.network.N
        self.devices = DeviceModel(config.env.devices, self.N, self.seed)
        self.devices.c = config.train.cycles_per_sample
        kappas = [d.kappa for d in self.devices.devices]
        self.latency = LatencyModel(config.env.latency, kappas, self.seed)
        self.network = FairNetwork(self.N, config.network.small_message_bytes)
        for d in self.devices.devices:
            down = 1e15 if config.network.large_aggregator_downlink else d.B_down
            self.network.set_capacity(d.index, d.B_up, down)
        # FL baseline (Plexus' --fix-aggregator): the node with the lowest median RTT to the others is a fixed server
        # with unlimited bandwidth, never drops out, and aggregates every round
        self.server: Optional[int] = None
        if config.protocol.selector == "fl_server":
            rtt = self.latency.rtt
            medians = [float(np.median(np.delete(rtt[i], i))) for i in range(self.N)]
            self.server = int(np.argmin(medians))
            self.network.set_capacity(self.server, 1e15, 1e15)
        self.data = FederatedData(config, self.N, self.seed)
        self.model0 = build_model(config.data.dataset, config.model.architecture, self.data.num_classes, self.seed)
        self.nodes: List[IPv8] = []
        self.overlays: List[PlexusSharedCommunity] = []
        self.index: Dict[bytes, int] = {}
        self.completed: Dict[int, Dict] = {}
        self.samples: Dict[int, List[int]] = {}
        self.stopped = False
        self.compute_used = 0.0
        self.bytes_total = 0
        self.last_close = 0.0
        self.on_finish = None
        self.test_loader = None

    def spawn(self, coro):
        task = asyncio.ensure_future(coro)

        def done(t):
            if not t.cancelled() and t.exception() is not None:
                asyncio.get_event_loop().call_exception_handler({"exception": t.exception()})

        task.add_done_callback(done)
        return task

    def now(self) -> float:
        return asyncio.get_event_loop().time()

    # ------------------------------------------------------------------ setup (as simulations/plexus/plexus_simulation)

    async def start(self) -> None:
        cfg = self.config
        PlexusSharedCommunity.world = self
        for peer_id in range(1, self.N + 1):
            builder = ConfigBuilder().clear_keys().clear_overlays()
            key_material = b64encode(b"LibNaCLSK:%s" % (chr(peer_id).encode() * 1000)[:68]).decode()
            builder.add_key_from_bin("my peer", key_material, file_path=os.path.join(self.run_dir, "ec%d.pem" % peer_id))
            builder.add_overlay("PlexusSharedCommunity", "my peer", [], [], {}, [])
            instance = IPv8(builder.finalize(), endpoint_override=SimulationEndpoint(),
                            extra_communities={"PlexusSharedCommunity": PlexusSharedCommunity})
            await instance.start()
            overlay = instance.overlays[0]
            overlay.max_peers = -1
            overlay.my_peer.address = overlay.endpoint.wan_address
            overlay.my_estimated_wan = overlay.endpoint.wan_address
            overlay.cancel_pending_task("_check_tasks")
            self.nodes.append(instance)
            self.overlays.append(overlay)
        peers = [o.my_peer for o in self.overlays]
        for i, overlay in enumerate(self.overlays):
            overlay.peers_list = peers
            self.index[overlay.my_peer.public_key.key_to_bin()] = i
            for j, other in enumerate(self.overlays):
                overlay.endpoint.latencies[other.endpoint.wan_address] = self.latency.mean_one_way(i, j)

        pc = cfg.baselines.plexus
        settings = SessionSettings(
            work_dir=self.run_dir, dataset=cfg.data.dataset,
            learning=LearningSettings(cfg.train.lr, cfg.train.momentum, cfg.train.weight_decay, cfg.train.batch_size, 1),
            participants=[hexlify(o.my_peer.public_key.key_to_bin()).decode() for o in self.overlays],
            all_participants=[hexlify(o.my_peer.public_key.key_to_bin()).decode() for o in self.overlays],
            target_participants=self.N,
            dfl=DFLSettings(sample_size=cfg.protocol.n, num_aggregators=pc.num_aggregators,
                            success_fraction=pc.success_fraction, ping_timeout=pc.ping_timeout,
                            inactivity_threshold=1000,
                            aggregation_timeout=pc.aggregation_timeout or cfg.closing.delta_dl,
                            fixed_aggregator=None if self.server is None
                            else self.overlays[self.server].my_peer.public_key.key_to_bin()),
            model=cfg.model.architecture, is_simulation=True, bypass_training=True)
        bws = {o.my_peer.public_key.key_to_bin(): int(self.devices.devices[i].B_up) for i, o in enumerate(self.overlays)}
        for i, overlay in enumerate(self.overlays):
            overlay.setup(settings)
            overlay.settings.bypass_training = cfg.data.bypass_training
            overlay.model_manager.model = unserialize_model(serialize_model(self.model0), cfg.data.dataset,
                                                            architecture=cfg.model.architecture)
            overlay.model_manager.train = self._train_hook(i, overlay)
            overlay.other_nodes_bws = bws
            overlay.aggregate_complete_callback = lambda r, m, i=i: self.on_aggregate(i, r, m)
            overlay.start()

        self.last_progress = self.now()
        asyncio.get_event_loop().call_later(cfg.run.liveness_check_interval / 4, self.check_liveness)

        # Round 1: as the original Plexus simulation, a seeded random sample of the active nodes starts training
        rand = Random(self.seed)
        activated = rand.sample(self.nodes, min(len(self.nodes), cfg.protocol.n))
        self.samples[1] = sorted(self.index[n.overlays[0].my_peer.public_key.key_to_bin()] for n in activated)
        for node in activated:
            overlay = node.overlays[0]
            overlay.received_aggregated_model(overlay.my_peer, 1, overlay.model_manager.model)
        for overlay in self.overlays:
            overlay.peers_first_round = [n.overlays[0].my_peer.public_key.key_to_bin() for n in activated]

    def _train_hook(self, index: int, overlay):
        async def train():
            round = overlay.train_sample_estimate
            overlay.model_manager.model_trainer.is_training = True
            cfg = self.config
            if cfg.baselines.plexus.dropout and index != self.server and self.devices.dropout(index, round):
                # Silent for this round, as a dropped-out client of the other selectors: it never delivers
                await asyncio.get_event_loop().create_future()
            f = self.devices.true_f(index, round)
            start = self.now()
            try:
                await asyncio.sleep(cfg.train.epochs * self.data.nodes[index].size * cfg.train.cycles_per_sample / f)
            finally:
                self.compute_used += self.now() - start
            if not cfg.data.bypass_training:
                generator = torch.Generator()
                generator.manual_seed(rng_seed(self.seed, "train", index, round) % (2 ** 63))
                loader = self.data.loader(index, cfg.train.batch_size, generator)
                train_local(overlay.model_manager.model, loader, cfg.train.epochs, cfg.train.lr, cfg.train.momentum,
                            cfg.train.weight_decay, cfg.train.max_batches_per_epoch, cfg.train.device)
            overlay.model_manager.model_trainer.is_training = False
            return self.data.nodes[index].size
        return train

    # ------------------------------------------------------------------ transport

    async def transfer(self, sender, peer, info: bytes, data: bytes, round: int, type: str) -> bool:
        src = self.index[sender.my_peer.public_key.key_to_bin()]
        dst = self.index[peer.public_key.key_to_bin()]
        receiver = self.overlays[dst]
        if not receiver.is_active:
            return False
        self.network.set_offline(dst, False)
        self.bytes_total += len(data) + len(info)
        timing = await self.network.transfer(src, dst, len(data) + len(info), self.latency.one_way(src, dst))
        if timing.ok and receiver.is_active and not self.stopped:
            ensure_future(receiver.on_receive(TransferResult(sender.my_peer, info, data, 0)))
            return True
        return False

    # ------------------------------------------------------------------ logging

    def plexus_aggregated(self, overlay, model_round: int, contributors: List[bytes]) -> None:
        if model_round in self.completed:
            return
        index = self.index[overlay.my_peer.public_key.key_to_bin()]
        self.completed[model_round] = {"round": model_round, "aggregator": index, "close_time": self.now(),
                                       "contributors": sorted(self.index[pk] for pk in contributors),
                                       "clients": self.samples.get(model_round),
                                       "selector": self.config.protocol.selector}

    def plexus_sample(self, sample_index: int, participants: List[bytes]) -> None:
        if sample_index not in self.samples:
            self.samples[sample_index] = sorted(self.index[pk] for pk in participants)

    async def on_aggregate(self, index: int, round: int, model):
        record = self.completed.get(round)
        if record is None or record.get("logged"):
            return
        record["clients"] = self.samples.get(round)
        record["logged"] = True
        record["duration"] = record["close_time"] - self.last_close
        self.last_close = record["close_time"]
        record["compute_total"] = self.compute_used
        record["bytes_total"] = self.bytes_total
        self.last_progress = self.now()
        interval = self.config.run.accuracy_interval
        if interval and round % interval == 0 and not self.config.data.bypass_training:
            if self.test_loader is None:
                self.test_loader = self.data.test_loader()
            record["accuracy"], record["test_loss"] = evaluate_model(model, self.test_loader)
        self.log.write("rounds", {k: v for k, v in record.items() if k != "logged"})
        print("Round %d closed at t=%.2f by node %d (plexus)" % (round, record["close_time"], record["aggregator"]))
        logged = len([r for r in self.completed.values() if r.get("logged")])
        max_time = self.config.run.max_sim_time
        if (self.config.run.rounds and logged >= self.config.run.rounds) or (max_time and self.now() >= max_time):
            self.finish()

    def check_liveness(self) -> None:
        if self.stopped:
            return
        if self.now() - self.last_progress > self.config.run.liveness_check_interval:
            self.finish()
            raise RuntimeError("Liveness violated: Plexus completed no round in %.0f simulated seconds"
                               % self.config.run.liveness_check_interval)
        asyncio.get_event_loop().call_later(self.config.run.liveness_check_interval / 4, self.check_liveness)

    def finish(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        self.log.write("summary", {"time": self.now(),
                                   "completed_rounds": len([r for r in self.completed.values() if r.get("logged")]),
                                   "compute_total": self.compute_used,
                                   "traffic_totals": {"all": {"bytes": self.bytes_total}}})
        self.log.flush()
        if self.on_finish:
            self.on_finish()

    def environment_stats(self) -> Dict:
        return {"devices": {k: v for k, v in self.devices.stats.items() if k != "fedscale_sampled_ids"},
                "latency": self.latency.stats, "data": self.data.stats, "selector": self.config.protocol.selector,
                "fl_server": self.server}
