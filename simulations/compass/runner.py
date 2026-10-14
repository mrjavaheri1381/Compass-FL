"""
Drives one simulation: builds the world (or the Plexus port), runs the discrete-event loop until the stop condition,
and writes the run metadata.
"""
import asyncio
import logging
import os

from simulation.discrete_loop import DiscreteLoop

from dlsim.compass.runlog import base_meta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_simulation(config, run_dir: str) -> None:
    logging.basicConfig(level=getattr(logging, config.run.log_level), format="%(levelname)s:%(name)s:%(message)s")
    loop = DiscreteLoop()
    asyncio.set_event_loop(loop)
    try:
        if config.protocol.selector in ("plexus", "fl_server"):
            from simulations.compass.plexus_shared import PlexusWorld
            world = PlexusWorld(config, run_dir)
        else:
            from dlsim.compass.env.world import World
            world = World(config, run_dir)
        world.log.write_config(config.to_yaml())
        meta = base_meta(config.seed, ROOT)
        meta["environment"] = world.environment_stats()
        meta["deviations"] = deviations(config)
        world.log.write_meta(meta)
        world.on_finish = loop.stop
        world.spawn(world.start())
        loop.run_forever()
    finally:
        asyncio._set_running_loop(None)  # DiscreteLoop does not reset it
        if "world" in locals():
            world.log.close()


def deviations(config) -> list:
    """
    Settings that deviate from the paper's model and must be reported with the results.
    """
    out = []
    if config.train.max_batches_per_epoch:
        out.append("train.max_batches_per_epoch=%d caps the SGD work per epoch (simulated time still follows "
                   "E D c / f)" % config.train.max_batches_per_epoch)
    if config.data.bypass_training:
        out.append("data.bypass_training: no SGD; losses are synthetic")
    if config.protocol.aggregator_trains:
        out.append("protocol.aggregator_trains: the aggregator also trains (not in the paper)")
    if config.network.large_aggregator_downlink:
        out.append("network.large_aggregator_downlink: downlinks never bind")
    if config.env.latency.plexus_compat:
        out.append("env.latency.plexus_compat: the full RTT is used as the one-way delay")
    if config.closing.wait_grace_before_selection:
        out.append("closing.wait_grace_before_selection: option (a)")
    if config.solver.budget.max_time:
        out.append("solver.budget.max_time > 0 makes solver results depend on wall-clock time")
    return out
