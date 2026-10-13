# COMPASS-FL — project notes for Claude Code

This repository implements COMPASS-FL (paper: "COMPASS-FL: Configurable Multi-Criteria Privacy-Aware Joint Client and
Aggregator Selection in Serverless Federated Learning", by Mohammadreza Javaheri). It was forked from Plexus
(serverless FL). The author writes in Persian; answer in Persian unless asked otherwise. Code, comments and docs are
in English.

## Where things are

- `AUDIT.md`: the audit against the paper, **every design decision taken with the author** (Q1–Q14, RQ1–RQ6, P2-7),
  the Phase-2 status (section 7), and the findings for the paper (7.1). Read it before changing protocol behavior.
- `README.md`, section "COMPASS-FL": how to run, the architecture, how to add a solver or selector, and the mapping
  from paper parameters to config keys.
- `dlsim/compass/`: `core/` (pure logic: utility, view, estimators, verifier, neighborhood, hashing), `solvers/`,
  `env/` (devices from FedScale, WonderNetwork latencies, fair-sharing network, `world.py`), `community.py` (the node
  protocol), `closing.py`, `selectors.py`, `he.py`, `config.py`, `data.py`, `runlog.py`.
- `simulations/compass/`: `runner.py` and `plexus_shared.py` (the original Plexus logic on the shared environment).
- `run.py` + `configs/*.yaml` is the only entry point. `tests/` has 135 pytest tests (about 3 minutes).
- The original Plexus code (`dlsim/plexus`, `simulations/plexus`, ...) is kept unchanged as the baseline reference.

## Commands

```bash
git clone --recursive ...            # pyipv8 is a submodule
pip install -r requirements.txt
python -m pytest                      # must pass before any experiment
python run.py --config configs/smoke.yaml                         # a few rounds with real SGD (CIFAR-10 in ~/dfl-data)
python run.py --config configs/default.yaml train.device=cuda run.accuracy_device=cuda seed=1
python scripts/plot_compass.py runs/<name>/seed<seed> ... --out plots/
```

## Status (end of the first session, 2026-10-11)

Phase 1 (audit) and Phase 2 (implementation) are done: every checklist item is implemented and tested. Verified:

- Determinism, with and without real training.
- Real learning: CIFAR-10 IID, 20 nodes, 30 rounds, accuracy 29% → 48%.
- The Plexus port picks the same participants as the original Plexus simulation.
- N=100 without SGD runs 10 rounds in 28 s.
- No experiments were run for the paper.

## Open items before the paper's experiments (waiting for the author)

1. **Straggler calibration: done** (AUDIT 7.4): class_rmse sigma prior and the acknowledgement deadline.
2. **Self-declared values: done** (AUDIT 7.5): no declared device class (one public prior), declared D capped at D_max.
3. **Open defaults.**
   - `protocol.forbid_repeat_aggregator`: A(t) selecting itself and then crashing is a single point of failure (AUDIT 7.1, I-2).
   - The Plexus baseline's `aggregation_timeout`: the original code stalls under realistic timings (I-4).
4. **Experiment design.** The paper's Experiments section is empty. Agree on the datasets and α, N/n/M/k_nb,
   baselines, ablations, seeds and rounds; then write one config per experiment.
5. **Missing tooling.**
   - A sweep runner (configs × seeds in parallel, on the GPU server).
   - Aggregation across seeds (mean ± CI, paper tables).
   - An offline solver benchmark (greedy_swap vs genetic vs exact on small instances, as the paper claims). This needs
     problem instances to be dumped, which is not done yet.
6. **Data.**
   - FEMNIST and Shakespeare are not downloaded (LEAF, under `~/leaf/<dataset>/data/{train,test}`).
   - CIFAR-100 needs `data.download=true` (ask before downloading).
   - CIFAR-10 must be present in `~/dfl-data`.
7. **Commits and remotes.** Commits go to `https://github.com/mrjavaheri1381/Compass-FL` (remote `mine` on the
   author's laptop; `origin` there is the upstream Plexus repository and must not be pushed to).
