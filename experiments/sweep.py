"""
Run a sweep (variants x seeds) of COMPASS-FL simulations in parallel, one process per run.

    python experiments/sweep.py experiments/sweeps/weights.yaml --jobs 32
    python experiments/sweep.py experiments/sweeps/weights.yaml --jobs 32 train.device=cuda run.accuracy_device=cuda
    python experiments/sweep.py experiments/sweeps/weights.yaml --dry-run        # print the commands only

A sweep file is YAML:

    name: weights                       # output under <out>/<name>/<variant>/seed<seed>
    config: configs/default.yaml        # one path or a list (relative to the repository root)
    common: [data.alpha=0.1, run.rounds=0, run.max_sim_time=180000]
    seeds: [1, 2, 3]
    reference: uniform                  # the variant others are compared with (used by aggregate.py)
    variants:
      uniform: []
      data: [weights.comp=0.1, weights.comm=0.1, weights.data=0.6, weights.fair=0.1, weights.adh=0.1]

Override precedence: config file < common < variant < command-line overrides. A run is complete when its directory
holds summary.jsonl; complete runs are skipped (use --force to redo them), incomplete ones are restarted from scratch
(a run cannot be resumed mid-way). Each run's stdout and stderr go to <run dir>.log. Every run is checked with
--print-config before anything starts, so a typo in an override fails the sweep immediately.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_sweep(path):
    with open(path) as handle:
        spec = yaml.safe_load(handle)
    for key in ("name", "config", "variants"):
        if key not in spec:
            sys.exit("%s: missing key %r" % (path, key))
    configs = spec["config"] if isinstance(spec["config"], list) else [spec["config"]]
    spec["config"] = [c if os.path.isabs(c) else os.path.join(ROOT, c) for c in configs]
    spec.setdefault("common", [])
    spec.setdefault("seeds", [1])
    spec.setdefault("reference", next(iter(spec["variants"])))
    if spec["reference"] not in spec["variants"]:
        sys.exit("%s: reference %r is not a variant" % (path, spec["reference"]))
    return spec


def build_runs(spec, out, extra, only=None):
    runs = []
    for variant, overrides in spec["variants"].items():
        if only and variant not in only:
            continue
        for seed in spec["seeds"]:
            run_dir = os.path.join(out, spec["name"], variant, "seed%d" % seed)
            args = [sys.executable, os.path.join(ROOT, "run.py")]
            for config in spec["config"]:
                args += ["--config", config]
            args += list(spec["common"]) + list(overrides or []) + ["seed=%d" % seed] + list(extra)
            runs.append({"variant": variant, "seed": seed, "dir": run_dir, "args": args})
    return runs


def is_complete(run_dir):
    return os.path.exists(os.path.join(run_dir, "summary.jsonl"))


def check(run):
    """Resolve the config without running, so bad overrides fail before the sweep starts."""
    result = subprocess.run(run["args"] + ["--print-config"], cwd=ROOT, capture_output=True, text=True,
                            env=dict(os.environ, PYTHONHASHSEED="0"))
    if result.returncode != 0:
        sys.exit("Invalid run %s/seed%d:\n%s" % (run["variant"], run["seed"], result.stderr[-2000:]))


def execute(run):
    if os.path.exists(run["dir"]):
        shutil.rmtree(run["dir"])
    os.makedirs(os.path.dirname(run["dir"]), exist_ok=True)
    start = time.time()
    with open(run["dir"] + ".log", "w") as log:
        code = subprocess.call(run["args"] + ["--run-dir", run["dir"]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONHASHSEED="0"))
    return run, code, time.time() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", help="sweep YAML file")
    parser.add_argument("overrides", nargs="*", help="extra key=value overrides applied to every run")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", default=os.path.join(ROOT, "runs", "sweeps"))
    parser.add_argument("--only", nargs="+", help="run only these variants")
    parser.add_argument("--force", action="store_true", help="redo complete runs too")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = parser.parse_args()

    spec = load_sweep(args.sweep)
    runs = build_runs(spec, args.out, args.overrides, args.only)
    todo = [r for r in runs if args.force or not is_complete(r["dir"])]
    print("Sweep %s: %d runs, %d complete, %d to run with %d parallel jobs"
          % (spec["name"], len(runs), len(runs) - len(todo), len(todo), args.jobs), flush=True)
    if args.dry_run:
        for run in todo:
            print(" ".join(run["args"] + ["--run-dir", run["dir"]]))
        return
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(check, todo))
    os.makedirs(os.path.join(args.out, spec["name"]), exist_ok=True)
    shutil.copy(args.sweep, os.path.join(args.out, spec["name"], "sweep.yaml"))

    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(execute, run) for run in todo]
        for done, future in enumerate(as_completed(futures), 1):
            run, code, elapsed = future.result()
            ok = code == 0 and is_complete(run["dir"])
            status = "ok" if ok else "FAILED (exit %d, see %s.log)" % (code, run["dir"])
            if status != "ok":
                failed.append(run)
            print("[%d/%d] %s/seed%d %s in %.0f min" % (done, len(todo), run["variant"], run["seed"], status,
                                                        elapsed / 60), flush=True)
    if failed:
        sys.exit("%d runs failed: %s" % (len(failed), ", ".join("%s/seed%d" % (r["variant"], r["seed"])
                                                                   for r in failed)))
    print("All runs complete. Aggregate with: python experiments/aggregate.py %s"
          % os.path.join(args.out, spec["name"]))


if __name__ == "__main__":
    main()
