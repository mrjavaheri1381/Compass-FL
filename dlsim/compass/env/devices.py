"""
Device model (task G, G2; decision Q10).

Ground truth per node is sampled directly from the FedScale device profiles shipped with Plexus
(``data/fedscale_traces``: compute in ms per sample and one bandwidth value in kbps per device). FedScale bandwidth is
used as the upload rate, and download = ``down_up_ratio`` x upload (an assumption). Device classes are quantile tiers
of a combined score (mean of the z-scored log compute speed and log upload bandwidth); class 0 is the slowest tier.
The public class priors are the class medians of the sampled values; sigma_bar_k = sigma_rel x the class mean of mu.

On top of the base values, every (node, round) draws a multiplicative log-normal fluctuation (CV per class) and, with
a small probability, a sudden slowdown. All draws are keyed by (seed, node, round), so they do not depend on event
order. Units: f in samples/s x c (c = cycles per sample), bandwidth in bytes/s.
"""
import math
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dlsim.compass.core.estimators import predict_completion
from dlsim.compass.core.hashing import rng_seed
from dlsim.compass.core.view import ClassPrior


@dataclass
class Device:
    index: int
    kappa: int
    f: float          # Base capacity (c x samples/s)
    B_up: float       # Base upload bandwidth, bytes/s
    B_down: float
    ms_per_sample: float = 0.0
    kbps: float = 0.0
    trace_id: Optional[int] = None


def _lognormal_factor(rng: np.random.Generator, cv: float) -> float:
    if cv <= 0:
        return 1.0
    sigma = math.sqrt(math.log(1 + cv * cv))
    return float(rng.lognormal(-0.5 * sigma * sigma, sigma))


class DeviceModel:

    def __init__(self, config, N: int, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.c = 1.0
        self.stats: Dict = {}
        rng = np.random.default_rng(rng_seed(seed, "devices"))
        if config.source == "fedscale":
            raw = self._sample_fedscale(N, rng)
        else:
            raw = [(float(rng.uniform(*config.uniform_f)), float(rng.uniform(*config.uniform_bup)), 0.0, 0.0, None)
                   for _ in range(N)]
        self.devices: List[Device] = []
        kappas = self._classes([f for f, *_ in raw], [b for _, b, *_ in raw])
        for index, ((f, b_up, ms, kbps, trace_id), kappa) in enumerate(zip(raw, kappas)):
            self.devices.append(Device(index, kappa, f, b_up, b_up * config.down_up_ratio, ms, kbps, trace_id))
        self.stats["class_sizes"] = [sum(1 for d in self.devices if d.kappa == k) for k in range(config.num_classes)]

    # ------------------------------------------------------------------ sampling

    def _sample_fedscale(self, N: int, rng: np.random.Generator):
        cfg = self.config
        with open(cfg.trace_path, "rb") as handle:
            traces = pickle.load(handle)
        keys = sorted(traces)
        kept = []
        for key in keys:
            ms = float(traces[key]["computation"])
            kbps = float(traces[key]["communication"])
            if ms <= 0 or kbps <= 0:
                continue
            if cfg.min_ms_per_sample and ms < cfg.min_ms_per_sample:
                continue
            if cfg.max_ms_per_sample and ms > cfg.max_ms_per_sample:
                continue
            if cfg.min_kbps and kbps < cfg.min_kbps:
                continue
            if cfg.max_kbps and kbps > cfg.max_kbps:
                continue
            kept.append(key)
        if len(kept) < N:
            raise ValueError("Only %d FedScale profiles left after filtering, need %d" % (len(kept), N))
        chosen = sorted(int(x) for x in rng.choice(len(kept), size=N, replace=False))
        chosen_keys = [kept[i] for i in chosen]
        rng.shuffle(chosen_keys)  # Node index order is independent of the trace order
        self.stats.update({
            "fedscale_profiles_total": len(keys),
            "fedscale_profiles_after_filter": len(kept),
            "fedscale_profiles_sampled": N,
            "fedscale_filters": {"min_ms_per_sample": cfg.min_ms_per_sample, "max_ms_per_sample": cfg.max_ms_per_sample,
                                 "min_kbps": cfg.min_kbps, "max_kbps": cfg.max_kbps,
                                 "dropped_nonpositive": sum(1 for k in keys if traces[k]["computation"] <= 0
                                                            or traces[k]["communication"] <= 0)},
            "fedscale_sampled_ids": [int(k) for k in chosen_keys],
        })
        out = []
        for key in chosen_keys:
            ms = float(traces[key]["computation"])
            kbps = float(traces[key]["communication"])
            # As in Plexus: kbps * 1024 / 8 bytes per second
            out.append((self.c * 1000.0 / ms, kbps * 1024.0 / 8.0, ms, kbps, int(key)))
        return out

    def _classes(self, f: List[float], b: List[float]) -> List[int]:
        logf, logb = np.log(np.asarray(f)), np.log(np.asarray(b))

        def z(x):
            std = x.std()
            return (x - x.mean()) / std if std > 0 else np.zeros_like(x)

        score = 0.5 * z(logf) + 0.5 * z(logb)
        cuts = [float(np.quantile(score, q)) for q in self.config.cut_points]
        self.stats["class_cut_scores"] = cuts
        return [int(sum(1 for cut in cuts if s > cut)) for s in score]

    # ------------------------------------------------------------------ priors

    def priors(self, E: int, D: Dict[int, float], Z: float, declared: bool = True,
               D_max: float = math.inf) -> Dict[int, ClassPrior]:
        """
        Public class priors: median f and B_up of the class. Without declared classes (``declared=False``) there is a
        single prior, key 0, pooled over all devices. The protocol predicts with the capped size min(D, D_max), the
        devices run on their true D. sigma_bar_k is either
        - ``rel``:        sigma_rel x the class mean of mu, or
        - ``class_rmse``: the root-mean-square error of the prior prediction inside the class,
                          sqrt(mean_i (mu_i(true f_i, B_i) - mu_i(f_bar, B_bar))^2), combined with the per-round
                          fluctuation of the class (CV x mean mu), i.e. the actual uncertainty of a node that has not
                          been measured yet. Wide classes thus get wide margins.
        """
        out = {}
        groups = range(self.config.num_classes) if declared else [None]
        for kappa in groups:
            members = [d for d in self.devices if kappa is None or d.kappa == kappa]
            if not members:
                continue
            f_med = float(np.median([d.f for d in members]))
            b_med = float(np.median([d.B_up for d in members]))
            mus = np.array([predict_completion(E, min(D[d.index], D_max), self.c, Z, f_med, b_med)
                            for d in members])
            if self.config.sigma_prior == "class_rmse":
                true_mus = np.array([predict_completion(E, D[d.index], self.c, Z, d.f, d.B_up) for d in members])
                rmse = float(np.sqrt(np.mean((true_mus - mus) ** 2)))
                cv = float(np.sqrt(np.mean([self.config.fluct_cv[d.kappa] ** 2 for d in members])))
                fluctuation = cv * float(np.mean(true_mus))
                sigma = float(np.sqrt(rmse ** 2 + fluctuation ** 2))
            else:
                sigma = self.config.sigma_rel * float(np.mean(mus))
            out[0 if kappa is None else kappa] = ClassPrior(f_med, b_med, sigma)
        self.stats["priors"] = {k: {"f": p.f, "B": p.B, "sigma": p.sigma} for k, p in out.items()}
        self.stats["sigma_prior"] = self.config.sigma_prior
        self.stats["declared_class"] = declared
        return out

    # ------------------------------------------------------------------ per-round truth

    def _round_rng(self, kind: str, node: int, round: int) -> np.random.Generator:
        return np.random.default_rng(rng_seed(self.seed, kind, node, round))

    def slowdown(self, node: int, round: int) -> float:
        rng = self._round_rng("slowdown", node, round)
        if rng.random() < self.config.slowdown_prob:
            low, high = self.config.slowdown_factor
            return float(rng.uniform(low, high))
        return 1.0

    def true_f(self, node: int, round: int) -> float:
        device = self.devices[node]
        cv = self.config.fluct_cv[device.kappa]
        return device.f * _lognormal_factor(self._round_rng("fluct_f", node, round), cv) / self.slowdown(node, round)

    def true_bandwidth(self, node: int, round: int):
        """
        (upload, download) in bytes/s for a round; one fluctuation factor scales both directions.
        """
        device = self.devices[node]
        cv = self.config.fluct_cv[device.kappa]
        factor = _lognormal_factor(self._round_rng("fluct_b", node, round), cv) / self.slowdown(node, round)
        return device.B_up * factor, device.B_down * factor

    def dropout(self, node: int, round: int) -> bool:
        return self._round_rng("dropout", node, round).random() < self.config.dropout[self.devices[node].kappa]

    def crash(self, node: int, round: int, crash_prob: List[float]) -> bool:
        return self._round_rng("crash", node, round).random() < crash_prob[self.devices[node].kappa]
