"""
Selection strategies (task E). A selector decides, at the aggregator A(t), the aggregator and the clients of round t+1.
All of them run inside the same protocol (hash pool, round closing, handover with fallback, logging), so only the
decision differs:

- ``compass``          the full distributed protocol of the paper (context collection, probing, local solves,
                        verification) -- ``CompassNode.select_compass``;
- ``compass_random``   the same protocol with the ``random`` solver (isolates the effect of informed selection);
- ``power_of_choice``  pool members evaluate the loss of theta^(t) on their data (costs a model download plus
                        ``eval_fraction`` of an epoch of compute); the clients are the n' highest losses;
- ``oort_like``        statistical utility D_i sqrt(loss_i^2) times a system-speed penalty (T_pref / T_i)^alpha,
                        with a fraction of the clients explored among never-measured nodes (simplified Oort);
- ``oracle_central``   a centralized solve of the same objective over the whole network with full knowledge (true
                        f, B, RTT, similarities); the selection step itself is free, everything else is simulated;
- ``fedcs``            FedCS (Nishio & Yonetani, ICC'19): greedily add the client that increases the estimated round
                        time the least (computation in parallel, uploads scheduled one after another), as long as the
                        round fits the deadline T_round; at most n' clients, fewer if the deadline binds;
- ``plexus``           the original Plexus logic, run by ``simulations/compass/plexus_shared.py``;
- ``fl_server``        the FL reference of the Plexus paper: the same logic with a fixed server of unlimited bandwidth.

For the baselines (power_of_choice, oort_like), the aggregator is the eligible pool member with the smallest hash; the
next eligible members in hash order are the fallbacks.
"""
import asyncio
import math
from typing import Dict, List, Optional

import numpy as np

from dlsim.compass.community import SMALL, SelectionOutcome
from dlsim.compass.core import hashing
from dlsim.compass.core.context import Proposal
from dlsim.compass.core.estimators import predict_completion
from dlsim.compass.core.hashing import rng_seed
from dlsim.compass.core.neighborhood import neighborhood
from dlsim.compass.core.similarity import cosine
from dlsim.compass.core.utility import SelectionProblem, evaluate
from dlsim.compass.solvers import Budget, create_solver


class Selector:
    name = "base"

    def __init__(self, world) -> None:
        self.world = world
        self.config = world.config

    async def select(self, node, next_round: int, theta: bytes) -> SelectionOutcome:
        raise NotImplementedError


class CompassSelector(Selector):
    name = "compass"

    async def select(self, node, next_round: int, theta: bytes) -> SelectionOutcome:
        return await node.select_compass(next_round, theta)


def _eligible(world, view, members: List[bytes], exclude: Optional[bytes] = None) -> List[bytes]:
    out = []
    for pk in members:
        if pk not in view or pk == exclude:
            continue
        state = view.get(pk)
        if state.f >= world.f_min and state.B >= world.B_min:
            out.append(pk)
    return out


def _fallback_ranking(world, node, pool: List[bytes], responders: List[bytes], choose,
                      min_size: Optional[int] = None) -> List[Proposal]:
    """
    Proposals for the eligible responders in hash order: (aggregator, choose(aggregator)). A proposal needs exactly n'
    clients, or between ``min_size`` and n' when given (FedCS may select fewer).
    """
    forbid = node.pk if world.config.protocol.forbid_repeat_aggregator else None
    ranked = []
    for aggregator in _eligible(world, node.view, [pk for pk in pool if pk in responders], forbid)[:3]:
        clients = choose(aggregator)
        low = world.group_size if min_size is None else min_size
        if low <= len(clients) <= world.group_size:
            ranked.append(Proposal(aggregator, clients, float("nan"), {}, solver=world.config.protocol.selector))
    return ranked


class PowerOfChoiceSelector(Selector):
    name = "power_of_choice"

    async def select(self, node, next_round: int, theta: bytes) -> SelectionOutcome:
        world, cfg = self.world, self.config
        outcome = SelectionOutcome(stats={"attempts": []})
        seed = node.pool_seed(next_round, theta)
        for k in range(cfg.fault.max_resamples + 1):
            start = node.now()
            pool = hashing.pool(node.view.members(), seed, cfg.protocol.M, k)
            losses = await self.collect_losses(node, next_round, pool, theta)
            responders = [pk for pk in pool if pk in losses]
            excluded = {node.pk} if cfg.protocol.forbid_aggregator_as_client else set()

            def choose(aggregator: bytes) -> List[bytes]:
                options = [pk for pk in responders if pk != aggregator and pk not in excluded]
                options.sort(key=lambda pk: (-losses[pk], pk))
                return sorted(options[:world.group_size])

            ranked = _fallback_ranking(world, node, pool, responders, choose)
            outcome.stats["attempts"].append({"k": k, "start": start, "end": node.now(), "pool": len(pool),
                                              "responders": len(responders), "eligible": len(ranked),
                                              "proposals": len(ranked)})
            if ranked:
                outcome.ranked = ranked
                return outcome
        return outcome

    async def collect_losses(self, node, round: int, pool: List[bytes], theta: bytes) -> Dict[bytes, float]:
        timeout = self.config.protocol.collect_timeout + self.config.closing.delta_dl

        async def ask(pk: bytes):
            if pk == node.pk:
                await asyncio.sleep(self.eval_time(node.index, round))
                return pk, self.world.local_loss(node.index, round, theta)
            reply = await node.request(pk, "loss_request", round, {"theta": theta}, len(theta) + SMALL, "poc", timeout)
            return pk, None if reply is None else reply["loss"]

        results = await asyncio.gather(*(ask(pk) for pk in pool))
        return {pk: loss for pk, loss in results if loss is not None}

    def eval_time(self, index: int, round: int) -> float:
        return self.config.baselines.power_of_choice.eval_fraction * \
            self.world.compute_time(index, round) / self.config.train.epochs


class OortSelector(Selector):
    """
    Simplifications relative to Oort (Lai et al., OSDI'21): no pacer that adapts the preferred duration, no
    confidence-interval based exploitation, no blacklisting; the statistical utility uses the training loss each client
    reported with its last update (recorded in the view by its aggregator), and the system term uses the predicted
    completion time mu_i from the view estimates. Exploration picks the fastest never-measured pool members.
    """
    name = "oort_like"

    async def select(self, node, next_round: int, theta: bytes) -> SelectionOutcome:
        world, cfg = self.world, self.config
        o = cfg.baselines.oort
        outcome = SelectionOutcome(stats={"attempts": []})
        seed = node.pool_seed(next_round, theta)
        view = node.view
        E, c, Z = cfg.train.epochs, cfg.train.cycles_per_sample, world.Z
        for k in range(cfg.fault.max_resamples + 1):
            pool = [pk for pk in hashing.pool(view.members(), seed, cfg.protocol.M, k) if pk in view]
            mu = {pk: predict_completion(E, view.get(pk).D, c, Z, view.get(pk).f, view.get(pk).B) for pk in pool}
            t_pref = o.t_pref or float(np.median(list(mu.values()))) if mu else 1.0
            excluded = {node.pk} if cfg.protocol.forbid_aggregator_as_client else set()

            def choose(aggregator: bytes) -> List[bytes]:
                options = [pk for pk in pool if pk != aggregator and pk not in excluded]
                explored = [pk for pk in options if view.get(pk).loss is not None]
                fresh = sorted([pk for pk in options if view.get(pk).loss is None], key=lambda pk: (mu[pk], pk))

                def utility(pk):
                    state = view.get(pk)
                    stat = state.D * math.sqrt(state.loss ** 2)
                    return stat * (t_pref / mu[pk]) ** o.alpha if mu[pk] > t_pref else stat

                explored.sort(key=lambda pk: (-utility(pk), pk))
                n = world.group_size
                n_explore = min(len(fresh), int(round(o.explore * n)))
                chosen = explored[:n - n_explore]
                chosen += fresh[:n - len(chosen)]
                if len(chosen) < n:
                    chosen += [pk for pk in explored if pk not in chosen][:n - len(chosen)]
                return sorted(chosen)

            ranked = _fallback_ranking(world, node, pool, pool, choose)
            outcome.stats["attempts"].append({"k": k, "start": node.now(), "end": node.now(), "pool": len(pool),
                                              "responders": len(pool), "eligible": len(ranked),
                                              "proposals": len(ranked)})
            if ranked:
                outcome.ranked = ranked
                return outcome
        return outcome


def fedcs_schedule(compute: Dict[bytes, float], upload: Dict[bytes, float], n: int, t_round: float) -> List[bytes]:
    """
    FedCS client selection (Nishio & Yonetani 2019, Algorithm 2) with a cap of n clients. The round time of a set is
    the time at which the last upload ends when every client computes in parallel and the uploads go one after
    another in order of compute completion. Repeatedly add the client with the smallest increase of the round time;
    a client that would exceed t_round is discarded. Ties by key.
    """
    def round_time(members):
        t = 0.0
        for pk in sorted(members, key=lambda pk: (compute[pk], pk)):
            t = max(t, compute[pk]) + upload[pk]
        return t

    chosen, remaining, current = [], sorted(compute), 0.0
    while remaining and len(chosen) < n:
        best = min(remaining, key=lambda pk: (round_time(chosen + [pk]) - current, pk))
        remaining.remove(best)
        value = round_time(chosen + [best])
        if value <= t_round:
            chosen.append(best)
            current = value
    return sorted(chosen)


class FedCSSelector(Selector):
    """
    Capacities and bandwidths are the measured estimates of the population view (FedCS asks clients for them; in
    COMPASS-FL nothing is self-reported). If no client fits the deadline, the fastest single client is chosen.
    """
    name = "fedcs"

    async def select(self, node, next_round: int, theta: bytes) -> SelectionOutcome:
        world, cfg = self.world, self.config
        fc = cfg.baselines.fedcs
        outcome = SelectionOutcome(stats={"attempts": []})
        seed = node.pool_seed(next_round, theta)
        view = node.view
        E, c, Z = cfg.train.epochs, cfg.train.cycles_per_sample, world.Z
        for k in range(cfg.fault.max_resamples + 1):
            pool = [pk for pk in hashing.pool(view.members(), seed, cfg.protocol.M, k) if pk in view]
            compute = {pk: E * view.get(pk).D * c / view.get(pk).f for pk in pool}
            upload = {pk: Z / view.get(pk).B for pk in pool}
            mu = sorted(compute[pk] + upload[pk] for pk in pool)
            t_round = fc.t_round or (fc.t_round_factor * float(np.median(mu)) if mu else 0.0)
            excluded = {node.pk} if cfg.protocol.forbid_aggregator_as_client else set()

            def choose(aggregator: bytes) -> List[bytes]:
                options = [pk for pk in pool if pk != aggregator and pk not in excluded]
                chosen = fedcs_schedule({pk: compute[pk] for pk in options}, upload, world.group_size, t_round)
                if not chosen and options:
                    chosen = [min(options, key=lambda pk: (compute[pk] + upload[pk], pk))]
                return chosen

            ranked = _fallback_ranking(world, node, pool, pool, choose, min_size=1)
            outcome.stats["attempts"].append({"k": k, "start": node.now(), "end": node.now(), "pool": len(pool),
                                              "responders": len(pool), "eligible": len(ranked),
                                              "proposals": len(ranked), "t_round": t_round})
            if ranked:
                outcome.ranked = ranked
                return outcome
        return outcome


class OracleSelector(Selector):
    """
    Upper bound for the cost of removing the server: one solver run per candidate aggregator over the whole current
    membership, with the true capacities, upload bandwidths, RTTs and similarities, and Q and staleness from the view.
    """
    name = "oracle_central"

    async def select(self, node, next_round: int, theta: bytes) -> SelectionOutcome:
        world, cfg = self.world, self.config
        oc = cfg.baselines.oracle
        view = node.view
        members = view.members()
        idx = {pk: world.index_of(pk) for pk in members}
        devices = world.devices.devices
        candidates = [pk for pk in members if devices[idx[pk]].f >= world.f_min and devices[idx[pk]].B_up >= world.B_min]
        if cfg.protocol.forbid_repeat_aggregator:
            candidates = [pk for pk in candidates if pk != node.pk]
        if oc.candidates == "topk":
            candidates.sort(key=lambda pk: (-devices[idx[pk]].f * devices[idx[pk]].B_up, pk))
            candidates = sorted(candidates[:oc.topk])
        Y = {pk: world.y_hat[idx[pk]] for pk in members}
        r = {pk: sum(float(np.dot(Y[pk], Y[q])) for q in members) for pk in members}
        net_sum = float(sum(r.values()))
        excluded = [node.pk] if cfg.protocol.forbid_aggregator_as_client else []
        k_nb = oc.k_nb or len(members)
        proposals = []
        solver = create_solver(world.solver_name, cfg.solver)
        for a in candidates:
            rtt = {pk: float(world.latency.rtt[idx[a], idx[pk]]) for pk in members}
            nb = neighborhood(a, members, rtt, world.L_max, k_nb, exclude=excluded)
            if len(nb) < world.group_size:
                continue
            ordered = sorted(nb)
            problem = SelectionProblem(
                ids=ordered, n=world.group_size, f=np.array([devices[idx[pk]].f for pk in ordered]),
                L=np.array([rtt[pk] for pk in ordered]),
                stale=np.array([next_round - view.get(pk).tau for pk in ordered], dtype=float),
                Q=np.array([view.get(pk).Q for pk in ordered]), r=np.array([r[pk] for pk in ordered]),
                S=np.array([[float(np.dot(Y[p], Y[q])) for q in ordered] for p in ordered]), net_sum=net_sum,
                params=world.utility_params, aggregator=a)
            rng = np.random.default_rng(rng_seed(cfg.seed, "oracle", a, next_round))
            result = solver.solve(problem, Budget(max_evals=oc.max_evals), rng)
            clients = result.ids(problem)
            proposals.append(Proposal(a, clients, result.utility, {pk: rtt[pk] for pk in clients}, result.components,
                                      neighborhood_size=len(nb), evaluations=result.evaluations,
                                      solve_time=result.wall_time, solver="oracle:" + world.solver_name))
        proposals.sort(key=lambda p: (-p.utility, p.aggregator))
        outcome = SelectionOutcome(ranked=proposals[:3], stats={"attempts": [
            {"k": 0, "start": node.now(), "end": node.now(), "pool": len(members), "responders": len(members),
             "eligible": len(candidates), "proposals": len(proposals)}]})
        return outcome


SELECTORS = {cls.name: cls for cls in (CompassSelector, PowerOfChoiceSelector, OortSelector, FedCSSelector,
                                        OracleSelector)}


def make_selector(name: str, world) -> Selector:
    if name == "compass_random":
        return CompassSelector(world)
    if name in ("plexus", "fl_server"):
        raise ValueError("The %s selector runs the original Plexus protocol (see simulations/compass/runner.py)" % name)
    return SELECTORS[name](world)
