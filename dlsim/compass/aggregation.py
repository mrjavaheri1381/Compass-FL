"""
FedAvg weighted by dataset size (paper Eq. fedavg): theta = sum_i D_i theta_i / sum_i D_i over the clients that were
not declared stragglers; theta^(t) = theta^(t-1) when there are none.
"""
import copy
from typing import List, Optional, Sequence

import torch
from torch import nn


def fedavg(models: Sequence[nn.Module], sizes: Sequence[float], previous: Optional[nn.Module]) -> nn.Module:
    if not models:
        if previous is None:
            raise ValueError("No updates and no previous model")
        return copy.deepcopy(previous)
    total = float(sum(sizes))
    if total <= 0:
        raise ValueError("Dataset sizes must be positive")
    weights = [float(s) / total for s in sizes]
    with torch.no_grad():
        result = copy.deepcopy(models[0])
        state = result.state_dict()
        for key, value in state.items():
            if torch.is_floating_point(value):
                state[key] = sum(w * m.state_dict()[key] for w, m in zip(weights, models))
        result.load_state_dict(state)
    return result
