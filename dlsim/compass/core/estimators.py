"""
Capacity, bandwidth, completion-time and deadline-adherence estimators (paper Sec. 3.3).

    T_cmp = t_first - t_ack,  T_up = t_last - t_first
    f_hat = E D c / T_cmp,    B_hat = Z / T_up
    mu    = E D c / f + Z / B                         (predicted t_last - t_ack)
    beta_i = max(beta, 1 / (m + 1))
    sigma^2 <- (1 - beta_i) sigma^2 + beta_i (delta_hat - mu)^2   (mu before f, B are updated)
    x <- (1 - beta_i) x + beta_i x_hat for x in {f, B};  m += 1
    Q <- (1 - alpha) Q + alpha exp(-gamma l)          (l = inf contributes 0)

Units: Z in bytes and B in bytes/s; with c = 1, f is in samples/s.
"""
import math
from dataclasses import dataclass
from typing import Optional


def step_size(beta: float, m: int) -> float:
    return max(beta, 1.0 / (m + 1))


def predict_completion(E: int, D: float, c: float, Z: float, f: float, B: float,
                       upload_rate: Optional[float] = None) -> float:
    """
    mu = E D c / f + Z / B. ``upload_rate`` replaces B in the upload term (contention-aware deadline, RQ1).
    """
    rate = B if upload_rate is None else upload_rate
    if f <= 0 or rate <= 0:
        return math.inf
    return E * D * c / f + Z / rate


def contention_rate(B_i: float, B_down_agg: float, k: int) -> float:
    """
    Upload rate under ingress contention: min(B_i, B_down(a) / k), with k >= 1 concurrent uploads.
    """
    return min(B_i, B_down_agg / max(1, k))


@dataclass
class Measurement:
    t_ack: float
    t_first: float
    t_last: float
    censored: bool = False      # Aggregator ingress was saturated during the upload (RQ5: time-averaged >= rho)

    @property
    def T_cmp(self) -> float:
        return self.t_first - self.t_ack

    @property
    def T_up(self) -> float:
        return self.t_last - self.t_first

    @property
    def delta_hat(self) -> float:
        return self.t_last - self.t_ack


def measure(E: int, D: float, c: float, Z: float, m: Measurement):
    """
    (f_hat, B_hat); None for a quantity whose interval is not positive.
    """
    f_hat = E * D * c / m.T_cmp if m.T_cmp > 0 else None
    B_hat = Z / m.T_up if m.T_up > 0 else None
    return f_hat, B_hat


def update_q(Q: float, lateness: float, alpha: float, gamma: float) -> float:
    contribution = 0.0 if math.isinf(lateness) else math.exp(-gamma * max(0.0, lateness))
    return (1 - alpha) * Q + alpha * contribution


def ema(old: float, new: float, step: float) -> float:
    return (1 - step) * old + step * new
