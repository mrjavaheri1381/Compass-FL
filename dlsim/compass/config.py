"""
Configuration of COMPASS-FL runs.

One tree of dataclasses holds every parameter of the paper and every choice between alternatives (selector, solver,
utility form, round-closing policy, latency model, ...). A run is described by a YAML file and dotted command-line
overrides::

    python run.py --config configs/default.yaml solver=tabu selector=compass utility.weights.comm=0.3 seed=3

``load_config`` merges the YAML file(s) into the defaults, applies the overrides, and validates the result. Unknown
keys are rejected so that a typo never silently falls back to a default.
"""
import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import yaml

COMPONENTS = ("comp", "comm", "data", "fair", "strag")

# Short aliases accepted on the command line: whole-key aliases (``solver=tabu`` means ``solver.name=tabu``) and
# prefix aliases (``weights.comm=0.3`` means ``utility.weights.comm=0.3``).
ALIASES = {
    "solver": "solver.name",
    "selector": "protocol.selector",
    "N": "network.N",
    "M": "protocol.M",
    "n": "protocol.n",
    "k_nb": "protocol.k_nb",
    "rounds": "run.rounds",
    "policy": "closing.policy",
    "form": "utility.form",
    "he": "he.mode",
}
PREFIX_ALIASES = {
    "weights": "utility.weights",
}


@dataclass
class RunConfig:
    name: str = "compass"
    out_dir: str = "runs"
    rounds: int = 10                  # Stop after this many completed rounds (0 = no limit)
    max_sim_time: float = 0           # Stop after this much simulated time in seconds (0 = no limit)
    accuracy_interval: int = 1        # Evaluate the test accuracy every this many rounds (0 = never)
    accuracy_device: str = "cpu"
    log_level: str = "WARNING"
    store_best_model: bool = False
    liveness_check_interval: float = 3600  # Simulated seconds without a completed round before aborting


@dataclass
class DataConfig:
    dataset: str = "cifar10"          # cifar10, cifar100, femnist, shakespeare
    partition: str = "dirichlet"      # dirichlet, iid, natural (femnist / shakespeare)
    alpha: float = 0.5                # Dirichlet concentration
    min_samples: int = 10             # Every node gets at least this many samples (dirichlet)
    base_path: Optional[str] = None   # Defaults to $HOME (datasets under $HOME/dfl-data or $HOME/leaf)
    download: bool = False            # Allow torchvision to download missing datasets
    bypass_training: bool = False     # Skip SGD; label counts and D_i are still drawn from the same partitioner
    synthetic_train_size: int = 50000  # Size of the synthetic label pool when training is bypassed
    num_classes: int = 0              # 0 = derived from the dataset


@dataclass
class ModelConfig:
    architecture: Optional[str] = None  # None = the default model of the dataset (e.g. gnlenet for cifar10)


@dataclass
class TrainConfig:
    epochs: int = 1                   # E, local epochs
    batch_size: int = 20
    lr: float = 0.002
    momentum: float = 0.9
    weight_decay: float = 0.0
    max_batches_per_epoch: int = 0    # >0 caps real SGD work (smoke/test only; logged as a deviation)
    cycles_per_sample: float = 1.0    # c; with c = 1, f is expressed in samples per second
    device: str = "cpu"


@dataclass
class NetworkConfig:
    N: int = 20                       # Number of nodes
    initially_active: int = 0         # 0 = all N nodes are active at t = 0; the others join later (churn)
    small_message_bytes: int = 4096   # Messages below this size skip bandwidth sharing (latency + size/rate only)
    large_aggregator_downlink: bool = False  # Option: make every node's downlink non-binding


@dataclass
class DevicesConfig:
    source: str = "fedscale"          # fedscale (sample real profiles) or uniform (synthetic, for tests)
    trace_path: str = "data/fedscale_traces"
    min_ms_per_sample: float = 0      # Optional filter on the FedScale profiles (0 = off)
    max_ms_per_sample: float = 0
    min_kbps: float = 0
    max_kbps: float = 0
    num_classes: int = 3
    cut_points: List[float] = field(default_factory=lambda: [1 / 3, 2 / 3])  # Quantiles of the combined score
    down_up_ratio: float = 4.0        # Assumption: download = ratio * upload (FedScale has one bandwidth value)
    sigma_rel: float = 0.2            # sigma_bar_k = sigma_rel * class mean of mu
    fluct_cv: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.1])  # Per-round log-normal CV, per class
    slowdown_prob: float = 0.02       # Probability of a sudden slowdown of a node in a round
    slowdown_factor: List[float] = field(default_factory=lambda: [3.0, 5.0])  # Uniform range of the slowdown
    dropout: List[float] = field(default_factory=lambda: [0.05, 0.02, 0.01])  # Per-round silence prob., per class
    # Uniform source (tests): ranges of the true values
    uniform_f: List[float] = field(default_factory=lambda: [10.0, 60.0])        # samples/s
    uniform_bup: List[float] = field(default_factory=lambda: [2e5, 2e6])        # bytes/s


@dataclass
class LatencyConfig:
    model: str = "wondernetwork"      # wondernetwork (city RTT matrix) or uniform (tests)
    matrix_path: str = "data/latencies.txt"
    city_assignment: str = "round_robin"  # round_robin (Plexus) or concentrated
    regions: int = 3                  # concentrated: number of regions
    region_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    cities_per_region: int = 5
    access_delay_ms: List[float] = field(default_factory=lambda: [30.0, 15.0, 5.0])  # Per class (low -> high tier)
    jitter_cv: float = 0.1            # Log-normal jitter of every one-way delay (and hence of every probe)
    plexus_compat: bool = False       # Use the full RTT as the one-way delay (as the original Plexus simulation)
    uniform_rtt: List[float] = field(default_factory=lambda: [0.02, 0.3])  # uniform model: RTT range in seconds


@dataclass
class ScriptedFailure:
    round: int = 1
    target: str = "aggregator"        # aggregator, or a node index
    when: str = "dispatch"            # dispatch (after dispatching the round), handover (on receiving it), client
    delay: float = 0.0                # Crash this many seconds after the trigger (e.g. mid-round)
    downtime: float = 0               # Seconds offline (0 = permanently)


@dataclass
class FailuresConfig:
    crash_prob: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # Per class, per round of activity
    crash_downtime: float = 0         # Seconds a crashed node stays offline (0 = permanently)
    scripted: List[ScriptedFailure] = field(default_factory=list)
    joins: List[List[float]] = field(default_factory=list)  # [[time, count], ...] joins from the inactive reserve
    availability_traces: Optional[str] = None  # Optional FedScale availability pickle


@dataclass
class EnvConfig:
    devices: DevicesConfig = field(default_factory=DevicesConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    failures: FailuresConfig = field(default_factory=FailuresConfig)


@dataclass
class ProbingConfig:
    P: int = 3                        # Probes per pool member
    prefetch: bool = False            # Probe the next pool during the current round (round-index hashing only)
    probe_bytes: int = 64


@dataclass
class ProtocolConfig:
    selector: str = "compass"         # compass, compass_random, plexus, power_of_choice, oort_like, oracle_central
    n: int = 4                        # Clients per round
    M: int = 10                       # Pool size
    k_nb: int = 8                     # Neighborhood size
    pool_seed: str = "round"          # round or model_hash
    L_max: float = 0.0                # RTT threshold in seconds; 0 = use L_max_percentile
    L_max_percentile: float = 75.0    # Percentile of the realized pairwise RTT distribution
    f_min: float = -1.0               # Absolute eligibility threshold; < 0 = use f_min_percentile
    f_min_percentile: float = 20.0
    B_min: float = -1.0
    B_min_percentile: float = 20.0
    aggregator_trains: bool = False   # The paper's aggregator does not train
    forbid_repeat_aggregator: bool = False
    forbid_aggregator_as_client: bool = False
    verify_tol: float = 1e-9          # Absolute tolerance of the verifier on U
    sim_tol: float = 1e-3             # CKKS error tolerance for |S_ij - S_ji|
    sim_tol_k: float = 3.0            # tolerance = sim_tol + k * sigma_dp * sqrt(2)
    exclude_flagged: bool = False     # Exclude nodes with flagged similarity reports from the pool
    flag_threshold: int = 3
    collect_timeout: float = 5.0
    propose_timeout: float = 30.0
    probing: ProbingConfig = field(default_factory=ProbingConfig)


@dataclass
class WeightsConfig:
    comp: float = 0.2
    comm: float = 0.2
    data: float = 0.2
    fair: float = 0.2
    strag: float = 0.2


@dataclass
class UtilityConfig:
    form: str = "multiplicative"      # multiplicative or additive
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    w_c: float = 0.5                  # Cohesion exponent inside Phi_data
    w_r: float = 0.5                  # Representativeness exponent inside Phi_data
    f_ref: float = 0.0                # Reference capacity (samples/s); 0 = 80th percentile of the class priors
    dtau_cap: float = 20.0
    eps: float = 1e-3


@dataclass
class StateConfig:
    alpha: float = 0.3                # Q smoothing
    gamma: float = 0.1                # Q lateness penalty (1/s)
    beta: float = 0.2                 # Floor of the f, B, sigma step size
    Q0: float = 1.0


@dataclass
class LateMergeConfig:
    enabled: bool = False             # REFL-style merging of late updates into the next round (hook only)
    weight: float = 0.5


@dataclass
class ClosingConfig:
    policy: str = "predictive"        # predictive, wait_all, fixed_deadline, over_selection
    z: float = 2.0                    # Safety margin in standard deviations
    delta_dl: float = 600.0           # Absolute cap on the round duration from dispatch (s)
    grace: float = 60.0               # g, grace period after d_i for recording late arrivals
    over_selection: float = 0.3       # o
    contention_aware: bool = True     # Upload term Z / min(B_i, B_down(a) / k_i)
    rho: float = 0.9                  # Censoring threshold on the aggregator's ingress utilization
    wait_grace_before_selection: bool = False  # Option (a): wait for the grace period before selecting
    late_merge: LateMergeConfig = field(default_factory=LateMergeConfig)


@dataclass
class BudgetConfig:
    max_evals: int = 0                # 0 = unlimited
    max_time: float = 0.0             # Wall-clock seconds; 0 = unlimited (> 0 makes runs nondeterministic)


@dataclass
class ExactConfig:
    max_combinations: int = 2_000_000
    on_too_large: str = "error"       # error or warn (warn falls back to greedy_swap)


@dataclass
class TabuConfig:
    iterations: int = 200
    tenure: int = 5
    patience: int = 50


@dataclass
class AnnealingConfig:
    iterations: int = 2000
    t0: float = 0.1                   # Initial temperature on log U
    cooling: float = 0.995


@dataclass
class GeneticConfig:
    population: int = 20
    generations: int = 20
    elitism: int = 2
    mutation_rate: float = 0.3
    local_search_depth: int = 3
    tournament_size: int = 3


@dataclass
class LNSConfig:
    iterations: int = 100
    destroy: int = 2                  # q
    rebuild: str = "greedy"           # greedy or exact


@dataclass
class SolverConfig:
    name: str = "greedy_swap"
    R_s: int = 10                     # seed pairs of greedy_swap (sensitivity: experiments/greedy_restarts.py)
    I_max: int = 50
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    exact: ExactConfig = field(default_factory=ExactConfig)
    tabu: TabuConfig = field(default_factory=TabuConfig)
    annealing: AnnealingConfig = field(default_factory=AnnealingConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    lns: LNSConfig = field(default_factory=LNSConfig)


@dataclass
class CKKSConfig:
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    scale_bits: int = 40
    flood_sigma: float = 1e-7         # Std. of the slot-domain flooding noise added by the responder


@dataclass
class DPConfig:
    enabled: bool = False
    sigma: float = 0.01


@dataclass
class HEConfig:
    mode: str = "plaintext"           # plaintext (direct computation) or ckks (real protocol)
    ckks: CKKSConfig = field(default_factory=CKKSConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    exchange_timeout: float = 600.0


@dataclass
class FaultConfig:
    max_resamples: int = 2
    max_missed: int = 3
    client_backups: bool = False
    handover_timeout: float = 30.0    # A(t+1) must acknowledge the handover within this time
    watchdog_timeout: float = 0.0     # A(t+1) must report round completion; 0 = delta_dl + grace + 300
    retry_delay: float = 30.0         # Delay before retrying a selection that found nobody


@dataclass
class PowerOfChoiceConfig:
    eval_fraction: float = 0.3        # Loss evaluation costs this fraction of one local epoch


@dataclass
class OortConfig:
    alpha: float = 2.0                # System-speed penalty exponent
    t_pref: float = 0.0               # Preferred round time; 0 = median predicted time of the pool
    explore: float = 0.2              # Fraction of the clients chosen among never-measured nodes


@dataclass
class OracleConfig:
    candidates: str = "all"           # all or topk
    topk: int = 10
    k_nb: int = 0                     # 0 = every reachable node
    max_evals: int = 0


@dataclass
class PlexusConfig:
    num_aggregators: int = 1
    success_fraction: float = 1.0
    aggregation_timeout: float = 0.0  # 0 = closing.delta_dl (the same cap the other selectors get)
    ping_timeout: float = 5.0


@dataclass
class BaselinesConfig:
    power_of_choice: PowerOfChoiceConfig = field(default_factory=PowerOfChoiceConfig)
    oort: OortConfig = field(default_factory=OortConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    plexus: PlexusConfig = field(default_factory=PlexusConfig)


@dataclass
class CompassConfig:
    seed: int = 42
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    state: StateConfig = field(default_factory=StateConfig)
    closing: ClosingConfig = field(default_factory=ClosingConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    he: HEConfig = field(default_factory=HEConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @property
    def group_size(self) -> int:
        """
        n', the number of clients a proposal contains: n, or ceil((1 + o) n) under over-selection.
        """
        if self.closing.policy == "over_selection":
            return int(math.ceil((1 + self.closing.over_selection) * self.protocol.n - 1e-9))
        return self.protocol.n


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------------- building

def _build(cls, data: Any, path: str):
    if not dataclasses.is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError("%s must be a mapping, got %r" % (path or "<root>", data))
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError("Unknown config key(s) under %s: %s" % (path or "<root>", unknown))
    kwargs = {}
    for name, value in data.items():
        f = fields[name]
        sub_path = "%s.%s" % (path, name) if path else name
        ftype = f.type if not isinstance(f.type, str) else _resolve_type(f.type)
        if dataclasses.is_dataclass(ftype):
            kwargs[name] = _build(ftype, value, sub_path)
        elif name == "scripted":
            kwargs[name] = [_build(ScriptedFailure, item, sub_path) for item in (value or [])]
        else:
            kwargs[name] = _coerce(ftype, value, sub_path)
    return cls(**kwargs)


def _coerce(ftype, value, path: str):
    """
    Convert scalars to the declared type (YAML reads 1e12 as a string, for example); fail loudly otherwise.
    """
    if value is None:
        return value
    try:
        if ftype is float and not isinstance(value, bool):
            return float(value)
        if ftype is int and not isinstance(value, bool):
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("not an integer")
            return int(float(value)) if isinstance(value, str) else int(value)
        if ftype is bool:
            if isinstance(value, str):
                if value.lower() in ("true", "yes", "1", "on"):
                    return True
                if value.lower() in ("false", "no", "0", "off"):
                    return False
                raise ValueError("not a boolean")
            return bool(value)
        if ftype is str:
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Invalid value %r for %s (%s)" % (value, path, exc))
    return value


def _resolve_type(name: str):
    return globals().get(name, name)


def _merge(base: Dict, update: Dict) -> Dict:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "scripted":
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text: str):
    if "=" not in text:
        raise ConfigError("Override %r is not of the form key=value" % text)
    key, raw = text.split("=", 1)
    key = key.strip()
    head, _, rest = key.partition(".")
    if not rest and head in ALIASES:
        key = ALIASES[head]
    elif rest and head in PREFIX_ALIASES:
        key = PREFIX_ALIASES[head] + "." + rest
    value = yaml.safe_load(raw) if raw.strip() != "" else ""
    return key, value


def apply_override(tree: Dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = tree
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ConfigError("Unknown config key %s" % key)
        node = node[part]
    if parts[-1] not in node:
        raise ConfigError("Unknown config key %s" % key)
    node[parts[-1]] = value


def load_config(paths: Sequence[str] = (), overrides: Sequence[str] = ()) -> CompassConfig:
    """
    Defaults <- each YAML file in order <- each key=value override; then validate.
    A YAML file may contain ``base: other.yaml`` (relative to itself) to inherit from another preset.
    """
    import os
    tree = CompassConfig().to_dict()
    for path in paths:
        tree = _merge(tree, _read_yaml_with_base(path))
    for text in overrides:
        key, value = parse_override(text)
        apply_override(tree, key, value)
    config = _build(CompassConfig, tree, "")
    validate(config)
    return config


def _read_yaml_with_base(path: str) -> Dict:
    import os
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    base = data.pop("base", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        data = _merge(_read_yaml_with_base(base_path), data)
    return data


def config_from_dict(data: Dict) -> CompassConfig:
    config = _build(CompassConfig, _merge(CompassConfig().to_dict(), data), "")
    validate(config)
    return config


# ---------------------------------------------------------------------- validation

SELECTORS = ("compass", "compass_random", "plexus", "power_of_choice", "oort_like", "oracle_central")
POLICIES = ("predictive", "wait_all", "fixed_deadline", "over_selection")


def validate_weights(weights: Dict[str, float], tolerance: float = 1e-6) -> None:
    """
    The paper requires w_m > 0 for every criterion and sum_m w_m = 1.
    """
    missing = [name for name in COMPONENTS if name not in weights]
    if missing:
        raise ConfigError("Missing utility weights: %s" % missing)
    if any(not weights[name] > 0 for name in COMPONENTS):
        raise ConfigError("Utility weights must be strictly positive (use a small value to de-emphasize): %s"
                          % weights)
    total = sum(weights[name] for name in COMPONENTS)
    if abs(total - 1.0) > tolerance:
        raise ConfigError("Utility weights must sum to 1 (got %.6f): %s" % (total, weights))


def validate(config: CompassConfig) -> None:
    from dlsim.compass.solvers import available_solvers  # Late import: solvers import this module

    p, u, c = config.protocol, config.utility, config.closing
    validate_weights(dataclasses.asdict(u.weights))
    if u.w_c < 0 or u.w_r < 0 or u.w_c + u.w_r <= 0:
        raise ConfigError("w_c, w_r must be >= 0 with w_c + w_r > 0 (got %s, %s)" % (u.w_c, u.w_r))
    if u.form not in ("multiplicative", "additive"):
        raise ConfigError("utility.form must be multiplicative or additive")
    if not 0 < u.eps < 1:
        raise ConfigError("utility.eps must be in (0, 1)")
    if u.dtau_cap <= 0:
        raise ConfigError("utility.dtau_cap must be > 0")
    if p.selector not in SELECTORS:
        raise ConfigError("protocol.selector must be one of %s" % (SELECTORS,))
    if config.solver.name not in available_solvers():
        raise ConfigError("solver.name must be one of %s" % (available_solvers(),))
    if p.n < 2:
        raise ConfigError("protocol.n must be >= 2")
    if p.k_nb < config.group_size:
        raise ConfigError("protocol.k_nb (%d) must be >= the group size n' (%d)" % (p.k_nb, config.group_size))
    if p.M < config.group_size + 1:
        raise ConfigError("protocol.M (%d) must be >= n' + 1 (%d)" % (p.M, config.group_size + 1))
    if p.M > config.network.N:
        raise ConfigError("protocol.M (%d) cannot exceed network.N (%d)" % (p.M, config.network.N))
    if p.pool_seed not in ("round", "model_hash"):
        raise ConfigError("protocol.pool_seed must be round or model_hash")
    if p.probing.P < 1:
        raise ConfigError("protocol.probing.P must be >= 1")
    if c.policy not in POLICIES:
        raise ConfigError("closing.policy must be one of %s" % (POLICIES,))
    if c.z <= 0 or c.delta_dl <= 0 or c.grace < 0 or not 0 < c.rho <= 1 or c.over_selection < 0:
        raise ConfigError("Invalid closing parameters: %s" % c)
    s = config.state
    if not 0 < s.alpha < 1 or s.gamma <= 0 or not 0 < s.beta < 1 or not 0 < s.Q0 <= 1:
        raise ConfigError("Invalid state parameters: %s" % s)
    if config.he.mode not in ("plaintext", "ckks"):
        raise ConfigError("he.mode must be plaintext or ckks")
    d = config.env.devices
    if d.source not in ("fedscale", "uniform"):
        raise ConfigError("env.devices.source must be fedscale or uniform")
    if len(d.cut_points) != d.num_classes - 1 or sorted(d.cut_points) != list(d.cut_points):
        raise ConfigError("env.devices.cut_points must be %d increasing quantiles" % (d.num_classes - 1))
    for name in ("fluct_cv", "dropout"):
        if len(getattr(d, name)) != d.num_classes:
            raise ConfigError("env.devices.%s needs one value per class (%d)" % (name, d.num_classes))
    lat = config.env.latency
    if lat.model not in ("wondernetwork", "uniform"):
        raise ConfigError("env.latency.model must be wondernetwork or uniform")
    if lat.city_assignment not in ("round_robin", "concentrated"):
        raise ConfigError("env.latency.city_assignment must be round_robin or concentrated")
    if len(lat.access_delay_ms) != d.num_classes:
        raise ConfigError("env.latency.access_delay_ms needs one value per class")
    if len(config.env.failures.crash_prob) != d.num_classes:
        raise ConfigError("env.failures.crash_prob needs one value per class")
    if config.network.initially_active < 0 or config.network.initially_active > config.network.N:
        raise ConfigError("network.initially_active must be in [0, N]")
    if config.train.epochs < 1:
        raise ConfigError("train.epochs (E) must be >= 1")
    if config.baselines.oracle.candidates not in ("all", "topk"):
        raise ConfigError("baselines.oracle.candidates must be all or topk")
    if config.solver.exact.on_too_large not in ("error", "warn"):
        raise ConfigError("solver.exact.on_too_large must be error or warn")
    if config.solver.lns.rebuild not in ("greedy", "exact"):
        raise ConfigError("solver.lns.rebuild must be greedy or exact")
