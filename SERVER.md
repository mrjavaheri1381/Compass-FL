# Running COMPASS-FL on the GPU server (g21)

Read this after `git pull` on the server. It covers the environment, the datasets, what to run, and the errors that
were already met (and fixed) on the laptop. Project background is in `CLAUDE.md`; design decisions in `AUDIT.md`.

## 1. Environment

```bash
git pull --recurse-submodules          # pyipv8 is a submodule; first time: git submodule update --init
pip install -r requirements.txt        # torch must be the CUDA build on the server
python -m pytest                       # 147 tests, ~5 min; must pass before any experiment
```

- The server runs Python 3.14. The original Plexus code needs an event loop set before it starts (see section 5);
  `run.py` and the tests already handle this.
- `run.py` re-executes itself with `PYTHONHASHSEED=0` (determinism). Always start runs through `run.py` or
  `experiments/sweep.py`.
- GPU: add `train.device=cuda run.accuracy_device=cuda` to every run. The protocol simulation itself is single-threaded
  CPU work, so throughput comes from running many runs in parallel (`--jobs`), not from the GPU.

## 2. Datasets

All datasets live under `$HOME` (or `data.base_path=<dir>`).

### CIFAR-10 → `~/dfl-data/cifar-10-batches-py`

```bash
python -c "import os, torchvision; torchvision.datasets.CIFAR10(os.path.expanduser('~/dfl-data'), download=True)"
```

### FEMNIST (LEAF) → `~/leaf/femnist/`

Two layouts are needed, with the same users:

| path | used by | format |
|---|---|---|
| `~/leaf/femnist/data/train` | COMPASS-FL (`run.py`) | JSON files with `users`, `num_samples`, `user_data` |
| `~/leaf/femnist/data/test` | COMPASS-FL and the original Plexus code | LEAF test split |
| `~/leaf/femnist/per_user_data/train` | original Plexus code only | one JSON file per user |

**Option A (fast): copy from the laptop**, where it is already built (about 730 MB):

```bash
scp -r ~/leaf/femnist g21:~/leaf/          # run on the laptop
```

**Option B: build it on the server** (about 1.6 GB download, 15 GB scratch, 1–2 hours). These are exactly the
parameters used on the laptop:

```bash
git clone https://github.com/TalwalkarLab/leaf.git ~/leaf-src
cd ~/leaf-src/data/femnist
sed -i 's/Image\.ANTIALIAS/Image.LANCZOS/' preprocess/data_to_json.py     # Pillow >= 10 removed ANTIALIAS
nohup ./preprocess.sh -s niid --sf 0.1 -k 0 -t sample --tf 0.8 --smplseed 1 --spltseed 1 > ~/leaf-src/prep.log 2>&1 &
# when it has finished (prep.log ends with "calculating JSON file checksums", data/train and data/test are not empty):
cd <repo>
mkdir -p ~/leaf/femnist/data
cp -r ~/leaf-src/data/femnist/data/test ~/leaf/femnist/data/test
python scripts/leaf_per_user.py ~/leaf-src/data/femnist/data/train ~/leaf/femnist/per_user_data/train --min-samples 20
cp -r ~/leaf/femnist/per_user_data/train ~/leaf/femnist/data/train
```

Expected result (laptop): 375 writers sampled, 9 dropped for having fewer than 20 train samples, **366 train users,
65k train samples, 16.5k test samples, 62 labels**. N = 355 (as in the Plexus paper) needs at least 355 users.

Notes:
- `preprocess.sh` uses `wget` for the raw NIST files. If it is missing, download them yourself into
  `~/leaf-src/data/femnist/data/raw_data/` and unzip them there (by_class.zip and by_write.zip from
  `https://s3.amazonaws.com/nist-srd/SD19/`).
- Interrupted runs: LEAF skips finished steps (cached `.pkl` files in `data/intermediate`), but delete any *empty*
  `all_data`, `sampled_data`, `rem_user_data`, `train`, `test` folders before rerunning, or later steps fail with
  `IndexError: list index out of range` in `split_data.py`.
- `--min-samples 20` is required: the original Plexus trainer uses batch size 20 with `drop_last=True`, so a user with
  fewer samples has no batch and its training crashes with `RuntimeError: coroutine raised StopIteration`.
- `~/leaf-src` can be deleted afterwards.

CIFAR-100 (`data.download=true`) and Shakespeare are not set up; ask the author before downloading them.

## 3. What to run

Every experiment is a sweep file in `experiments/sweeps/` (variants × seeds). `sweep.py` checks every run's config
first, runs them in parallel, skips completed runs (they have `summary.jsonl`) and restarts incomplete ones, so after
an interruption just run the same command again. Use `tmux` or `nohup` for long sweeps.

```bash
python experiments/sweep.py <sweep.yaml> --jobs <J> [--out <dir>] [--only v1 v2] [--dry-run] [overrides...]
python experiments/aggregate.py <out dir>/<sweep name>      # table.md, heatmap.png, accuracy curves, CSVs
```

Pick `--jobs` from a short pilot (CPU cores and GPU memory; each run needs about 1 GB of GPU memory).

| order | sweep | what | runs |
|---|---|---|---|
| 1 | `quick_data_weight.yaml` | w_data ∈ {0.05, 0.2, 0.6}, 100 rounds, CIFAR-10 α=0.1, N=100 | 9 |
| 2 | pilot | `main.yaml --only compass plexus run.max_sim_time=18000 seeds` → measure wall time per run | few |
| 3 | `main.yaml` | E1: COMPASS vs all baselines, CIFAR-10, N=1000, 50 sim-h, 5 seeds; once per α (see the file) | 45 per setting |
| 4 | `main_femnist.yaml` | E1 on FEMNIST, N=355, 200 sim-h | 45 |
| 5 | `factorial.yaml` | E2: selection × round closing | 20 |
| 6 | `weights.yaml`, `cohesion.yaml`, `closing.yaml` | E3, E4, E8 ablations, N=100 | 18 / 15 per α / 18 |

Examples:

```bash
python experiments/sweep.py experiments/sweeps/quick_data_weight.yaml --jobs 9 train.device=cuda run.accuracy_device=cuda
python experiments/sweep.py experiments/sweeps/main.yaml --jobs 32 --out runs/sweeps/cifar10_a0.1 data.alpha=0.1 train.device=cuda run.accuracy_device=cuda
python experiments/aggregate.py runs/sweeps/cifar10_a0.1/main
```

Baselines (all on the same simulated environment): `compass`, `compass_random`, `plexus` (port of the original
logic), `plexus_paper` (s=13), `fl_server` (fixed server with unlimited bandwidth, as Plexus' `--fix-aggregator`),
`power_of_choice`, `oort_like`, `fedcs`, `oracle_central`. Plexus-family variants use `success_fraction=0.8`; with 1.0
one dropped client holds the round until the timeout.

Interactive single runs: `python run_interactive.py` asks for the algorithm, dataset and parameters (or a saved
preset from `configs/presets/`, e.g. `--preset compass_femnist_n14`), then shows the log and a progress bar.

Single runs: `python run.py --config configs/default.yaml <key=value ...> --run-dir runs/<name>`;
`--print-config` shows the resolved configuration.

## 4. Outputs

- Each run: `<out>/<sweep>/<variant>/seed<s>/` with `rounds.jsonl` (per round: accuracy, duration, clients,
  stragglers, traffic, cumulative `bytes_total` and `compute_total`), `summary.jsonl`, `meta.json`,
  `config.resolved.yaml`; the run's stdout is in `seed<s>.log` next to it.
- `runs/` is git-ignored. To bring results back, copy the `aggregate/` folders (small) or the whole sweep folder:
  `rsync -a g21:<repo>/runs/sweeps/ runs/sweeps/`.

## 5. Original Plexus code (only to reproduce the Plexus paper)

`simulations/plexus/*.py` is the unchanged original simulator, with its own environment (no dropout, RTT used as a
one-way delay), so its numbers are **not** comparable with ours; the paper's Plexus baseline is the `plexus` selector
above. Run it from the repository root like this (plain `python3 simulations/plexus/femnist.py` fails with
`ModuleNotFoundError` and, on Python ≥ 3.12, `There is no current event loop`):

```bash
PYTHONPATH=.:pyipv8 python3 -u -c "import asyncio, runpy, sys; asyncio.set_event_loop(asyncio.new_event_loop()); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" simulations/plexus/femnist.py --peers 355 --sample-size 13 --accuracy-logging-interval 10 --success-fraction 0.8 --duration 720000 --bypass-model-transfers --seed 42 --log-level INFO --train-device-name cuda:0 --accuracy-device-name cuda:0
```

Results go to `data/n_<peers>_<dataset>_...dfl/` (`accuracies.csv` has accuracy in percent). Verified on the laptop:
355 peers, 5 rounds, no errors.

## 6. Known errors and their fixes

| error | cause | fix |
|---|---|---|
| `FileNotFoundError: .../leaf/femnist/data/test` | FEMNIST not installed | section 2 |
| `FileNotFoundError: .../leaf/femnist/per_user_data/train` | original Plexus needs the per-user layout | `scripts/leaf_per_user.py` (section 2) |
| `RuntimeError: coroutine raised StopIteration` (original Plexus, FEMNIST) | users with < 20 train samples | rebuild with `--min-samples 20` |
| `AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'` (LEAF) | Pillow ≥ 10 | the `sed` line in section 2 |
| `IndexError: list index out of range` in LEAF `split_data.py` | empty folders from a failed run | delete the empty folders, rerun |
| `ModuleNotFoundError: simulations` / `ipv8`, `There is no current event loop` | original Plexus started directly | command in section 5 |
| a sweep run `FAILED` | see `<run dir>.log` | fix, rerun the same sweep command (only unfinished runs restart) |
