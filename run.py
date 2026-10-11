"""
Run one COMPASS-FL simulation from a config file plus dotted overrides.

    python run.py --config configs/smoke.yaml
    python run.py --config configs/default.yaml solver=tabu selector=compass weights.comm=0.3 weights.comp=0.1 seed=3

Outputs go to ``<run.out_dir>/<run.name>/seed<seed>`` (or ``--run-dir``): the resolved config, meta.json (seed, git
commit, environment statistics) and the JSONL logs described in ``dlsim/compass/runlog.py``.
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    # A fixed hash seed makes iteration over sets of bytes reproducible
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    for path in (ROOT, os.path.join(ROOT, "pyipv8")):
        if path not in sys.path:
            sys.path.insert(0, path)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", default=[], help="YAML config (repeatable; later files win)")
    parser.add_argument("--run-dir", default=None, help="Output directory (default: out_dir/name/seed<seed>)")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. solver=tabu utility.form=additive")
    args = parser.parse_args()

    from dlsim.compass.config import load_config
    config = load_config(args.config, args.overrides)
    if args.print_config:
        print(config.to_yaml())
        return
    run_dir = args.run_dir or os.path.join(config.run.out_dir, config.run.name, "seed%d" % config.seed)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

    from simulations.compass.runner import run_simulation
    run_simulation(config, run_dir)
    print("Run finished; logs in %s" % run_dir)


if __name__ == "__main__":
    main()
