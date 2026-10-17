"""
Interactive launcher for one experiment: asks for the algorithm, the dataset and the other parameters (Enter keeps the
default shown in brackets), checks the configuration, then runs it, streams the run's log and keeps a progress bar
on the last line: round x of y (or simulated time of the budget), status, and the latest / best test accuracy.

    python run_interactive.py
    python run_interactive.py --defaults      # accept every default without asking (quick check)

Everything is passed to run.py as ordinary overrides, so the printed command reproduces the run exactly. Outputs go to
runs/interactive/<algorithm>_<dataset>_<time>/ (rounds.jsonl, summary.jsonl, ...); the full log is saved next to it
as <same name>.log.
"""
import argparse
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")

ALGORITHMS = [
    ("compass", "COMPASS-FL: the full protocol (multi-criteria selection, local solves, verification)"),
    ("compass_random", "COMPASS-FL protocol with random client selection"),
    ("plexus", "Plexus (original logic, on the same simulated environment)"),
    ("fl_server", "FL with a fixed server of unlimited bandwidth (the Plexus paper's FL reference)"),
    ("power_of_choice", "Power-of-Choice: the clients with the highest loss"),
    ("oort_like", "Oort (simplified): statistical utility x system-speed penalty"),
    ("fedcs", "FedCS: greedy deadline-based selection"),
    ("oracle_central", "Central solver with full knowledge (upper bound)"),
]
DATASETS = [
    ("cifar10", "CIFAR-10, 10 classes (Dirichlet or IID split)"),
    ("femnist", "FEMNIST (LEAF), 62 classes, natural split by writer"),
    ("cifar100", "CIFAR-100, 100 classes (downloaded on first use)"),
    ("shakespeare", "Shakespeare (LEAF), next-character prediction"),
]
ROUND_LINE = re.compile(r"Round (\d+) closed at t=([\d.]+) by node (\d+) \((\d+)/(\d+) aggregated(?:, acc=([\d.]+))?\)")


# ---------------------------------------------------------------------- questions

class Asker:
    def __init__(self, use_defaults: bool) -> None:
        self.use_defaults = use_defaults

    def _input(self, prompt: str) -> str:
        if self.use_defaults:
            print(prompt)
            return ""
        try:
            return input(prompt).strip()
        except EOFError:
            return ""

    def choice(self, title: str, options, default: str) -> str:
        print("\n" + title)
        for number, (name, text) in enumerate(options, 1):
            print("  %d) %-16s %s" % (number, name, text))
        names = [name for name, _ in options]
        while True:
            answer = self._input("  choice [%s]: " % default)
            if not answer:
                return default
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return names[int(answer) - 1]
            if answer in names:
                return answer
            print("  -> type a number from 1 to %d or a name" % len(options))

    def value(self, prompt: str, default, kind=str, check=None, hint: str = ""):
        while True:
            answer = self._input("  %s%s [%s]: " % (prompt, (" (%s)" % hint) if hint else "", default))
            if not answer:
                return default
            try:
                value = kind(answer)
            except ValueError:
                print("  -> not a valid %s" % kind.__name__)
                continue
            if check is not None:
                problem = check(value)
                if problem:
                    print("  -> " + problem)
                    continue
            return value

    def yes(self, prompt: str, default: bool) -> bool:
        while True:
            answer = self._input("  %s [%s]: " % (prompt, "Y/n" if default else "y/N")).lower()
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False


def positive(value):
    return None if value > 0 else "must be > 0"


def unit(value):
    return None if 0 < value <= 1 else "must be in (0, 1]"


def leaf_users(dataset: str) -> int:
    path = os.path.join(HOME, "leaf", dataset, "data", "train")
    if not os.path.isdir(path):
        return 0
    count = 0
    for name in os.listdir(path):
        if name.endswith(".json"):
            with open(os.path.join(path, name)) as handle:
                count += len(json.load(handle)["users"])
    return count


def ask_everything(asker: Asker):
    """Returns (overrides, description, rounds, budget_s)."""
    o = {}
    algorithm = asker.choice("1. Algorithm", ALGORITHMS, "compass")
    o["protocol.selector"] = algorithm
    dataset = asker.choice("2. Dataset", DATASETS, "cifar10")
    o["data.dataset"] = dataset

    print("\n3. Data and network")
    if dataset in ("cifar10", "cifar100"):
        if not os.path.isdir(os.path.join(HOME, "dfl-data")) and dataset == "cifar10":
            print("  ! ~/dfl-data not found: CIFAR-10 must be in ~/dfl-data (see SERVER.md)")
        if dataset == "cifar100":
            o["data.download"] = "true" if asker.yes("download CIFAR-100 to ~/dfl-data if missing", True) else "false"
        partition = asker.value("partition", "dirichlet", str,
                                lambda v: None if v in ("dirichlet", "iid") else "dirichlet or iid", "dirichlet/iid")
        o["data.partition"] = partition
        if partition == "dirichlet":
            o["data.alpha"] = asker.value("Dirichlet alpha (smaller = more non-IID)", 0.1, float, positive)
        N_default = 100
    else:
        users = leaf_users(dataset)
        if users == 0:
            print("  ! ~/leaf/%s/data/train not found: the dataset is not installed (see SERVER.md)" % dataset)
        else:
            print("  %d users available" % users)
        o["data.partition"] = "natural"
        N_default = min(355, users) if users else 355
    N = asker.value("number of nodes N", N_default, int, lambda v: None if v >= 4 else "must be >= 4")
    o["network.N"] = N
    n = asker.value("clients per round n", min(10, N - 2), int,
                    lambda v: None if 2 <= v <= N - 2 else "must be between 2 and N - 2")
    o["protocol.n"] = n

    print("\n4. Algorithm parameters (%s)" % algorithm)
    if algorithm in ("compass", "compass_random", "power_of_choice", "oort_like", "fedcs"):
        M = asker.value("candidate pool size M", min(max(30, n + 1), N), int,
                        lambda v: None if n + 1 <= v <= N else "must be between n + 1 and N")
        o["protocol.M"] = M
    if algorithm in ("compass", "compass_random", "oracle_central"):
        k_default = min(20, o.get("protocol.M", N) - 1)
        o["protocol.k_nb"] = asker.value("neighborhood size k_nb", k_default, int,
                                         lambda v: None if v >= n else "must be >= n")
    if algorithm in ("compass", "oracle_central"):
        o["solver.name"] = asker.value("solver", "greedy_swap", str, None,
                                       "greedy_swap/exact/tabu/simulated_annealing/genetic/lns/random")
        if asker.yes("change the utility weights (default 0.2 each)", False):
            while True:
                weights = {}
                for name, label in (("comp", "computation"), ("comm", "communication"), ("data", "data"),
                                    ("fair", "fairness"), ("adh", "deadline adherence")):
                    weights[name] = asker.value("w_%s (%s)" % (name, label), 0.2, float, positive)
                total = sum(weights.values())
                if abs(total - 1.0) < 1e-6:
                    break
                if asker.yes("the weights sum to %.3f, not 1: normalize them" % total, True):
                    weights = {k: v / total for k, v in weights.items()}
                    break
            for name, value in weights.items():
                o["utility.weights.%s" % name] = round(value, 6)
            w_c = asker.value("w_c (cohesion vs representativeness inside the data term, w_r = 1 - w_c)", 0.5,
                              float, lambda v: None if 0 <= v <= 1 else "must be in [0, 1]")
            o["utility.w_c"], o["utility.w_r"] = w_c, round(1 - w_c, 6)
    if algorithm in ("compass", "compass_random", "power_of_choice", "oort_like", "fedcs", "oracle_central"):
        o["closing.policy"] = asker.value("round closing policy", "predictive", str,
                                          lambda v: None if v in ("predictive", "wait_all", "fixed_deadline",
                                                                  "over_selection") else "unknown policy",
                                          "predictive/wait_all/fixed_deadline/over_selection")
    if algorithm in ("plexus", "fl_server"):
        o["baselines.plexus.success_fraction"] = asker.value(
            "success fraction (share of updates the aggregator waits for)", 0.8, float, unit)
    if algorithm == "fedcs":
        o["baselines.fedcs.t_round_factor"] = asker.value("deadline = factor x median predicted time", 2.0, float,
                                                          positive)

    print("\n5. Run length and training")
    rounds = asker.value("rounds (0 = use a simulated-time budget instead)", 100, int,
                         lambda v: None if v >= 0 else "must be >= 0")
    o["run.rounds"] = rounds
    budget = 0.0
    if rounds == 0:
        budget = asker.value("simulated-time budget in hours", 50.0, float, positive) * 3600
        o["run.max_sim_time"] = budget
    o["run.accuracy_interval"] = asker.value("evaluate the test accuracy every ... rounds", 5, int,
                                             lambda v: None if v >= 0 else "must be >= 0")
    o["train.epochs"] = asker.value("local epochs E", 1, int, positive)
    o["seed"] = asker.value("seed", 1, int)
    device = "cuda" if cuda_available() else "cpu"
    device = asker.value("device", device, str, lambda v: None if v in ("cpu", "cuda", "mps") else "cpu, cuda or mps")
    o["train.device"] = o["run.accuracy_device"] = device
    if asker.yes("skip real training (protocol only, much faster, no accuracy)", False):
        o["data.bypass_training"] = "true"
        o["run.accuracy_interval"] = 0
    description = "%s on %s" % (algorithm, dataset)
    return o, description, rounds, budget


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------- running

class Progress:
    """A progress bar kept on the last terminal line while the log scrolls above it."""

    def __init__(self, rounds: int, budget: float, tty: bool) -> None:
        self.rounds, self.budget, self.tty = rounds, budget, tty
        self.round = 0
        self.sim_time = 0.0
        self.status = "starting"
        self.acc = None
        self.best = None
        self.aggregated = ""
        self.start = time.time()
        self.first_round_wall = None

    def fraction(self) -> float:
        if self.rounds:
            return min(1.0, self.round / self.rounds)
        if self.budget:
            return min(1.0, self.sim_time / self.budget)
        return 0.0

    def line(self) -> str:
        width = max(10, min(30, shutil.get_terminal_size((100, 20)).columns - 90))
        done = int(round(self.fraction() * width))
        bar = "#" * done + "-" * (width - done)
        if self.rounds:
            where = "round %d/%d" % (self.round, self.rounds)
        else:
            where = "round %d, sim %.1f/%.0f h" % (self.round, self.sim_time / 3600, self.budget / 3600)
        acc = "acc -" if self.acc is None else "acc %.2f%% (best %.2f%%)" % (100 * self.acc, 100 * self.best)
        elapsed = time.time() - self.start
        eta = ""
        if self.first_round_wall is not None and 0 < self.fraction() < 1:
            spent = time.time() - self.first_round_wall
            eta = " | ETA %s" % clock(spent / self.fraction() * (1 - self.fraction()))
        agg = (" | %s aggregated" % self.aggregated) if self.aggregated else ""
        return "[%s] %3.0f%% | %s | %s | %s%s | %s%s" % (bar, 100 * self.fraction(), where, self.status, acc, agg,
                                                       clock(elapsed), eta)

    def draw(self) -> None:
        if self.tty:
            sys.stdout.write("\r\033[K" + self.line()[:shutil.get_terminal_size((200, 20)).columns - 1])
            sys.stdout.flush()

    def log(self, text: str) -> None:
        if self.tty:
            sys.stdout.write("\r\033[K" + text + "\n")
        else:
            print(text)
        self.draw()

    def round_closed(self, match) -> None:
        self.round = int(match.group(1))
        self.sim_time = float(match.group(2))
        self.aggregated = "%s/%s" % (match.group(4), match.group(5))
        if match.group(6) is not None:
            self.acc = float(match.group(6))
            self.best = self.acc if self.best is None else max(self.best, self.acc)
        if self.first_round_wall is None:
            self.first_round_wall = time.time()
        self.status = "running"


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return "%dh%02dm" % (seconds // 3600, seconds % 3600 // 60) if seconds >= 3600 else \
        "%dm%02ds" % (seconds // 60, seconds % 60)


def run(command, run_dir: str, progress: Progress) -> int:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    log_file = open(run_dir + ".log", "w")  # Next to the run directory: run.py recreates the directory itself
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    lines: "queue.Queue[str]" = queue.Queue()

    def reader():
        for raw in process.stdout:
            lines.put(raw.rstrip("\n"))
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    progress.status = "setting up (data, devices, similarities)"
    progress.draw()
    try:
        while True:
            try:
                text = lines.get(timeout=0.5)
            except queue.Empty:
                progress.draw()
                continue
            if text is None:
                break
            log_file.write(text + "\n")
            log_file.flush()
            match = ROUND_LINE.search(text)
            if match:
                progress.round_closed(match)
            elif "Traceback" in text or "Error" in text:
                progress.status = "error"
            progress.log(text)
    except KeyboardInterrupt:
        process.terminate()
        progress.status = "interrupted"
        progress.log("Interrupted: the run was stopped (partial logs are kept).")
    code = process.wait()
    log_file.close()
    return code


def summarize(run_dir: str) -> None:
    path = os.path.join(run_dir, "rounds.jsonl")
    if not os.path.exists(path):
        return
    rounds = [json.loads(line) for line in open(path) if line.strip()]
    if not rounds:
        return
    accs = [r["accuracy"] for r in rounds if r.get("accuracy") is not None]
    selected = sum(len(r.get("clients") or []) for r in rounds)
    aggregated = sum(r.get("accepted", len(r.get("contributors") or [])) for r in rounds)
    durations = [r["duration"] for r in rounds if r.get("duration") is not None]
    print("\nSummary")
    print("  rounds completed      %d" % len(rounds))
    print("  simulated time        %.2f h" % (rounds[-1]["close_time"] / 3600))
    if accs:
        print("  final / best accuracy %.2f%% / %.2f%%" % (100 * accs[-1], 100 * max(accs)))
    if durations:
        print("  mean round duration   %.1f s" % (sum(durations) / len(durations)))
    if selected:
        print("  stragglers            %.1f%% of the selected clients" % (100 * (1 - aggregated / selected)))
    if rounds[-1].get("bytes_total") is not None:
        print("  communication         %.2f GiB" % (rounds[-1]["bytes_total"] / 2 ** 30))
    print("  outputs               %s" % run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--defaults", action="store_true", help="accept every default without asking")
    args = parser.parse_args()
    asker = Asker(args.defaults)

    print("COMPASS-FL experiment launcher (Enter keeps the value in brackets)")
    overrides, description, rounds, budget = ask_everything(asker)
    name = "%s_%s_%s" % (overrides["protocol.selector"], overrides["data.dataset"], time.strftime("%Y%m%d-%H%M%S"))
    run_dir = os.path.join(ROOT, "runs", "interactive", name)
    command = [sys.executable, os.path.join(ROOT, "run.py"), "--config", os.path.join(ROOT, "configs", "default.yaml")]
    command += ["%s=%s" % (key, value) for key, value in overrides.items()]

    check = subprocess.run(command + ["--print-config"], cwd=ROOT, capture_output=True, text=True)
    if check.returncode != 0:
        sys.exit("\nInvalid configuration:\n" + check.stderr[-2000:])
    print("\nRun: %s, %s" % (description, ("%d rounds" % rounds) if rounds else "%.0f simulated hours" % (budget / 3600)))
    print("Command (to reproduce):\n  " + " ".join(shlex.quote(c) for c in command + ["--run-dir", run_dir]))
    if not asker.yes("start", True):
        return
    os.makedirs(os.path.dirname(run_dir), exist_ok=True)
    progress = Progress(rounds, budget, sys.stdout.isatty())
    code = run(command + ["--run-dir", run_dir], run_dir, progress)
    progress.status = "finished" if code == 0 else "failed (exit %d)" % code
    progress.draw()
    print()
    summarize(run_dir)
    if code != 0:
        print("  the run failed: see %s.log" % run_dir)
        sys.exit(code)


if __name__ == "__main__":
    main()
