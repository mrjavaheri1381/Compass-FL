"""
Structured run outputs (task H).

Each run directory contains:
- ``config.resolved.yaml``  the fully resolved configuration;
- ``meta.json``             seed, git commit (and dirty flag), start time, environment statistics (FedScale sampling
                            and filtering, latency-matrix cleaning, device classes and priors, data partition);
- ``rounds.jsonl``          one record per completed round (see ``RoundLog``);
- ``selections.jsonl``      one record per selection attempt (pool, responders, proposals, verdicts, ...);
- ``proposals.jsonl``       one record per candidate solve (neighborhood, solver evaluations and time, utility);
- ``membership.jsonl``      every membership change (join, leave, removal after max_missed misses);
- ``events.jsonl``          protocol events (handover fallbacks, stale late records, flags, failures, ...).
"""
import json
import math
import os
import subprocess
import time
from typing import Any, Dict, Optional


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):  # numpy scalars
        return _clean(value.item())
    return value


def git_info(path: str) -> Dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=path, capture_output=True, text=True,
                                    check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:  # pragma: no cover
        return {"commit": None, "dirty": None}


class RunLogger:

    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.handles: Dict[str, Any] = {}

    def path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def write(self, stream: str, record: Dict) -> None:
        handle = self.handles.get(stream)
        if handle is None:
            handle = open(self.path(stream + ".jsonl"), "a")
            self.handles[stream] = handle
        handle.write(json.dumps(_clean(record), sort_keys=True) + "\n")

    def write_meta(self, meta: Dict) -> None:
        with open(self.path("meta.json"), "w") as handle:
            json.dump(_clean(meta), handle, indent=2, sort_keys=True)

    def write_config(self, text: str) -> None:
        with open(self.path("config.resolved.yaml"), "w") as handle:
            handle.write(text)

    def flush(self) -> None:
        for handle in self.handles.values():
            handle.flush()

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles = {}


def base_meta(seed: int, repo: str) -> Dict:
    return {"seed": seed, "git": git_info(repo), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED")}
