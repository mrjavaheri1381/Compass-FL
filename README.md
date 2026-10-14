# Plexus: Practical Federated Learning without a Server
Repository for the source code of our paper *[Practical Federated Learning without a Server](https://arxiv.org/pdf/2302.13837)* published in the [Proceedings of the 5th Workshop on Machine Learning and Systems (EuroMLSys)](https://euromlsys.eu/#).

## Abstract

Federated Learning (FL) enables end-user devices to collaboratively train ML models without sharing raw data, thereby preserving data privacy.
In FL, a central parameter server coordinates the learning process by iteratively aggregating the trained models received from clients. 
Yet, deploying a central server is not always feasible due to hardware unavailability, infrastructure constraints, or operational costs.
We present Plexus, a fully decentralized FL system for large networks that operates without the drawbacks originating from having a central server.
Plexus distributes the responsibilities of model aggregation and sampling among participating nodes while avoiding network-wide coordination.
We evaluate Plexus using realistic traces for compute speed, pairwise latency and network capacity.
Our experiments on three common learning tasks and with up to 1000 nodes empirically show that Plexus reduces time-to-accuracy by 1.4-1.6x, communication volume by 15.8-292x and training resources needed for convergence by 30.5-77.9x compared to conventional decentralized learning algorithms.

## Installation

Start by cloning the repository recursively (since Plexus depends on the PyIPv8 networking library):

```
git clone https://github.com/sacs-epfl/plexus --recursive
```

Install the required dependencies (preferably in a virtual environment to avoid conflicts with existing libraries):

```
pip install -r requirements.txt
```

In our paper, we evaluate Plexus using the CIFAR-10, CelebA and FEMNIST datasets.
For CIFAR-10 we use `torchvision`. The FEMNIST and CelebA datasets have to be downloaded manually and we refer the reader to the [decentralizepy framework](https://github.com/sacs-epfl/decentralizepy) that uses the same datasets.

## Running

You can run Plexus for different datasets using the following commands:

```
# CIFAR-10
python3 -u simulations/plexus/cifar10.py \
    --peers 1000 \
    --sample-size 13 \
    --accuracy-logging-interval 10 \
    --success-fraction 0.8 \
    --duration 180000 \
    --bypass-model-transfers \
    --seed 42 \
    --log-level INFO

# CelebA
python3 -u simulations/plexus/celeba.py \
    --peers 500 \
    --sample-size 13 \
    --accuracy-logging-interval 10 \
    --success-fraction 0.8 \
    --duration 180000 \
    --bypass-model-transfers \
    --seed 42 \
    --log-level INFO

# FEMNIST
python3 -u simulations/plexus/femnist.py \
    --peers 355 \
    --sample-size 13 \
    --accuracy-logging-interval 10 \
    --success-fraction 0.8 \
    --duration 720000 \
    --bypass-model-transfers \
    --seed 42 \
    --log-level INFO
```

You can modify the path to the datasets with the `--dataset-base-path` argument.
You can speed up the experiment execution by enabling CUDA by passing `--train-device-name "cuda:0" --accuracy-device-name "cuda:0"`.
Since the experiment outputs quite a lot of logging, we also recommend to increase the log level for longer experiments.

To run the Federated Learning (FL) baseline, it suffices to pass the `--fix-aggregator` flag to the above commands, which will select a random peer to act as server and sets the bandwidth of this server to infinity.
The D-PSGD baseline can be run as follows:

```
# CIFAR-10
python3 -u simulations/dl/cifar10.py \
    --peers 1000 \
    --seed 42 \
    --duration 180000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --dl-round-timeout 60 \
    --topology k-regular \
    --k 10 \
    --log-level INFO

# CelebA
python3 -u simulations/dl/celeba.py \
    --peers 500 \
    --seed 42 \
    --duration 180000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --dl-round-timeout 60 \
    --topology k-regular \
    --k 10 \
    --log-level INFO

# FEMNIST
python3 -u simulations/dl/femnist.py \
    --peers 355 \
    --seed 42 \
    --duration 720000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --dl-round-timeout 60 \
    --topology k-regular \
    --k 10 \
    --log-level INFO
```

While the above commands use a `k-regular` topology, Plexus supports the `ring`, `k-regular` and `exp-one-peer` topologies.
For the `k-regular` topology, the `k` argument is required which indicates the number of incoming and outgoing edges in the topology.

The Gossip Learning (GL) baseline can be run as follows:

```
# CIFAR-10
python3 -u simulations/gl/cifar10.py \
    --peers 1000 \
    --seed 42 \
    --duration 180000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --accuracy-logging-interval-is-in-sec \
    --log-level INFO

# CelebA
python3 -u simulations/gl/celeba.py \
    --peers 500 \
    --seed 42 \
    --duration 180000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --accuracy-logging-interval-is-in-sec \
    --log-level INFO

# FEMNIST
python3 -u simulations/gl/femnist.py \
    --peers 355 \
    --seed 42 \
    --duration 720000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --accuracy-logging-interval-is-in-sec \
    --log-level INFO
```

We provide several scripts in the `scripts` directory to help running these experiments.
For example, `run_parallel_with_seeds.sh` can run the above experiments in parallel for different seeds.

## COMPASS-FL

COMPASS-FL replaces Plexus' hash-only selection with a joint, multi-criteria, privacy-aware selection of the aggregator
and the clients of every round (paper `main.tex`). The implementation lives in `dlsim/compass` (protocol and logic),
`simulations/compass` (runner and Plexus port), `configs/` (presets), `tests/` and `run.py`.
`AUDIT.md` records the audit against the paper and every design decision taken with the author.

### Running

```bash
pip install -r requirements.txt
python run.py --config configs/smoke.yaml                    # a few rounds with real SGD (CIFAR-10 in ~/dfl-data)
python run.py --config configs/smoke_bypass.yaml             # protocol only, no SGD (seconds)
python run.py --config configs/default.yaml solver=tabu selector=compass weights.comm=0.3 weights.comp=0.1 seed=3
python run.py --config configs/power_of_choice.yaml          # one preset per baseline
python run.py --config configs/default.yaml --print-config   # show the fully resolved configuration
python -m pytest                                             # unit + end-to-end tests (~3 minutes)
python scripts/plot_compass.py runs/compass/seed42 runs/plexus/seed42 --out plots/
python scripts/bench_ckks.py --K 10 62 100                   # CKKS cost, sizes and error vs K
```

Paper experiments are sweeps (variants x seeds) described in `experiments/sweeps/*.yaml`:

```bash
python experiments/sweep.py experiments/sweeps/weights.yaml --jobs 32 train.device=cuda run.accuracy_device=cuda
python experiments/aggregate.py runs/sweeps/weights           # mean ± 95% CI, TTA/CTA/RTA, table, heatmap, curves
```

`sweep.py` validates every run first, runs them in parallel (one process each, log in `<run dir>.log`), skips
complete runs and restarts incomplete ones, so it can be re-launched after an interruption. `aggregate.py` writes
`per_run.csv`, `summary.csv`, `table.md`, `heatmap.png` (change vs the reference variant) and accuracy against
simulated time, communication volume and training compute (the three axes of the Plexus paper). Each round record
carries the cumulative `bytes_total` and `compute_total` (device-seconds of training) for CTA and RTA.

Overrides are dotted keys (`closing.policy=over_selection`, `env.latency.city_assignment=concentrated`, ...); short
aliases exist for `solver`, `selector`, `weights.*`, `N`, `M`, `n`, `k_nb`, `rounds`, `policy`, `form`, `he`.
Unknown keys and invalid values fail loudly. `run.py` fixes `PYTHONHASHSEED`, and one `seed` drives every random
choice (devices, cities, partition, model initialization, jitter, fluctuations, failures, training, solvers), so the
same config and seed reproduce the same selection log. Outputs go to `runs/<run.name>/seed<seed>/`:
`config.resolved.yaml`, `meta.json` (seed, git commit, FedScale sampling and filtering, latency-matrix cleaning,
device classes and priors, data partition, deviations from the paper's model), `rounds.jsonl` (one record per round:
aggregator, clients, every Phi and U of the winning proposal, candidates/eligible/proposals/rejected/fallbacks, solver
evaluations and time, per-client deadline d_i, arrival, lateness and censoring, on-time/late/missing counts,
selection traffic, f/B estimate error, accuracy), `selections.jsonl`, `proposals.jsonl`, `membership.jsonl`,
`events.jsonl` and `summary.jsonl`.

### Architecture

```
run.py ── simulations/compass/runner.py ── World (dlsim/compass/env/world.py) ── DiscreteLoop (simulated clock)
                                             ├── env/devices.py   FedScale profiles -> classes, priors, fluctuation
                                             ├── env/latency.py   WonderNetwork RTTs (cleaned) + access delay + jitter
                                             ├── env/network.py   per-node up/down links, max-min fair sharing
                                             ├── data.py          seeded partitions, models, local SGD
                                             └── CompassNode x N (dlsim/compass/community.py)   the protocol
                                                   ├── selectors.py      compass | compass_random | baselines
                                                   ├── closing.py        RoundClosingPolicy (predictive, ...)
                                                   ├── he.py             plaintext | CKKS similarity protocol
                                                   └── core/             pure logic: utility, context + verifier,
                                                       solvers/          neighborhood, view + estimators, hashing
```

`dlsim/compass/core` and `dlsim/compass/solvers` are pure (no asyncio, no simulator) and carry most of the unit tests.
One round at A(t): dispatch theta^(t-1) to C(t) -> acknowledgements and updates feed the closing policy -> close ->
FedAvg weighted by D_i over non-stragglers -> tau_i <- t for all of C(t), state records (Q, sigma, f, B) -> selection
of t+1 on an immutable view snapshot (context collection, probing, local solves, verification) -> handover with
watchdog and next-ranked fallback. Late arrivals within the grace period become state records applied after the
snapshot and forwarded to A(t+1) if the handover already happened. The `plexus` selector runs the original
`dlsim/plexus` logic unchanged on the same environment (`simulations/compass/plexus_shared.py`).

### Adding a solver or a selector

- **Solver**: create `dlsim/compass/solvers/my_solver.py` with a class deriving from `Solver`, decorated with
  `@register("my_solver")`, implementing `solve(problem, budget, rng) -> SolverResult`; evaluate sets only through
  `dlsim.compass.core.utility.evaluate` (or `IncrementalState`), count evaluations with `Counter`, and import the module
  in `dlsim/compass/solvers/__init__.py`. It is then available as `solver=my_solver` (and validated by the config).
- **Selector**: subclass `Selector` in `dlsim/compass/selectors.py` with `async select(node, next_round, theta) ->
  SelectionOutcome` (a ranked list of `Proposal`s; the protocol does the handover, fallbacks and logging), add it to
  `SELECTORS` and to `SELECTORS` in `dlsim/compass/config.py`, and add a preset in `configs/`.

### Paper parameters and their configuration keys

| Paper | Config key | Paper | Config key |
|---|---|---|---|
| N | `network.N` | M | `protocol.M` |
| n | `protocol.n` | k_nb | `protocol.k_nb` |
| L_max (RTT) | `protocol.L_max` or `protocol.L_max_percentile` | f_min, B_min | `protocol.f_min`, `protocol.B_min` (or `*_percentile`) |
| f_ref | `utility.f_ref` (0 = 75th percentile of f) | Delta tau_cap | `utility.dtau_cap` |
| w = (comp, comm, data, fair, adh) | `utility.weights.*` (> 0, sum 1) | w_c, w_r | `utility.w_c`, `utility.w_r` |
| utility form | `utility.form` (multiplicative, additive) | epsilon | `utility.eps` |
| alpha, gamma (Q) | `state.alpha`, `state.gamma` | beta, Q(0) | `state.beta`, `state.Q0` |
| delta_dl (cap) | `closing.delta_dl` | grace g | `closing.grace` |
| z | `closing.z` | rho (censoring) | `closing.rho` |
| closing rule | `closing.policy` | over-selection o | `closing.over_selection` |
| E | `train.epochs` | c | `train.cycles_per_sample` |
| Z | measured from the model (bytes) | P | `protocol.probing.P` |
| R_s, I_max | `solver.R_s`, `solver.I_max` | solver | `solver.name` |
| selector | `protocol.selector` | hash seed (round / model hash) | `protocol.pool_seed` |
| HE mode, DP noise | `he.mode`, `he.dp.enabled`, `he.dp.sigma` | CKKS parameters | `he.ckks.*` |
| f_bar, B_bar, sigma_bar (public priors) | `env.devices.*`, `protocol.declared_class` | D_max | `protocol.D_max`, `protocol.D_max_quantile` |
| data partition, alpha | `data.partition`, `data.alpha` | model | `model.architecture` |
| seed | `seed` | | |

### Modelling assumptions and simplifications (report with the results)

- Device ground truth is sampled from the FedScale profiles in `data/fedscale_traces` (500,000 devices: compute in
  ms per sample, one bandwidth value). The bandwidth is used as the upload rate; download = 4 x upload
  (`env.devices.down_up_ratio`). Classes are quantile tiers of a combined compute/bandwidth score. With c = 1,
  f is in samples per second.
- RTTs come from the WonderNetwork matrix in `data/latencies.txt` (227 cities) after cleaning (mean of both directions,
  else the valid one; cities with an invalid pair are excluded; statistics in `meta.json`), plus a per-class access
  delay and log-normal jitter. Messages take L/2 one way (`env.latency.plexus_compat` uses L, as the original Plexus
  simulation did). The original Plexus simulation trains 5 mini-batches per round, our shared timing model trains E
  epochs, so round times differ from the original while the Plexus port picks the same participants.
- Messages smaller than `network.small_message_bytes` (probes, acknowledgements, selection context) skip bandwidth
  sharing; model and view transfers are flows on max-min fair links.
- CKKS: flooding is realized as a fresh Enc_pk(0) plus slot-domain Gaussian noise of std `he.ckks.flood_sigma`; CKKS
  operations take no simulated time (their wall time is logged); the keys are published once per node.
- `oort_like` simplifies Oort: no pacer, no confidence-interval exploitation, no blacklist; the statistical utility
  uses each client's last reported training loss (written into the view by its aggregator) and the system term uses
  the predicted completion time mu_i; exploration picks the fastest never-measured pool members.
- `power_of_choice` uses the pool as the candidate set; each member downloads theta^(t) and evaluates its loss
  (`baselines.power_of_choice.eval_fraction` of an epoch of compute). With `data.bypass_training`, losses are synthetic.
- `oracle_central` solves the same objective over the whole membership with the true f, B, RTTs and similarities; only
  the selection step is free.
- The aggregator does not train (`protocol.aggregator_trains` exists as an option).

## Reference

If you find our work useful, you can cite us as follows:

```
@inproceedings{devos2025plexus,
  title={Practical Federated Learning without a Server},
  author={Dhasade, Akash and Kermarrec, Anne-Marie and Lavoie, Erick and Pouwelse, Johan and Sharma, Rishi and de Vos, Martijn},
  booktitle={Proceedings of the 5th Workshop on Machine Learning and Systems},
  year={2025}
}
```
