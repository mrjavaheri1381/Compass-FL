import json
import os

import numpy as np
import pytest

from dlsim.compass.config import load_config
from dlsim.compass.data import FederatedData, build_model, dirichlet_partition, iid_partition


def test_dirichlet_partition_is_seeded_and_complete():
    labels = np.arange(5000) % 10
    a = dirichlet_partition(labels, 20, 0.3, np.random.default_rng(1), min_samples=5)
    b = dirichlet_partition(labels, 20, 0.3, np.random.default_rng(1), min_samples=5)
    c = dirichlet_partition(labels, 20, 0.3, np.random.default_rng(2), min_samples=5)
    assert all((x == y).all() for x, y in zip(a, b))
    assert any(len(x) != len(y) or (x != y).any() for x, y in zip(a, c))
    assert sorted(np.concatenate(a).tolist()) == list(range(5000))
    assert min(len(p) for p in a) >= 5
    assert sorted(np.concatenate(iid_partition(100, 7, np.random.default_rng(0))).tolist()) == list(range(100))


def test_bypass_partition_gives_label_counts():
    config = load_config(overrides=["data.bypass_training=true", "network.N=10", "data.alpha=0.1"])
    data = FederatedData(config, 10, seed=3)
    assert len(data.nodes) == 10 and data.stats["D_total"] == config.data.synthetic_train_size
    assert all(n.label_counts.sum() == n.size for n in data.nodes)


def test_leaf_hooks(tmp_path):
    """FEMNIST and Shakespeare use their natural per-user partitions (LEAF JSON)."""
    for dataset, x, y in [("femnist", [[0.0] * 784] * 3, [1, 5, 5]), ("shakespeare", ["a" * 80] * 3, ["b", "c", "b"])]:
        for split in ("train", "test"):
            root = tmp_path / "leaf" / dataset / "data" / split
            root.mkdir(parents=True)
            users = {"u%d" % i: {"x": x, "y": y} for i in range(4)}
            (root / "all.json").write_text(json.dumps({"users": list(users), "user_data": users}))
        config = load_config(overrides=["data.dataset=%s" % dataset, "data.partition=natural", "network.N=3",
                                        "protocol.M=3", "protocol.n=2", "protocol.k_nb=2",
                                        "data.base_path=%s" % tmp_path])
        data = FederatedData(config, 3, seed=0)
        assert [n.size for n in data.nodes] == [3, 3, 3]
        model = build_model(dataset, None, data.num_classes, seed=0)
        batch = next(iter(data.loader(0, 2, __import__("torch").Generator())))
        assert model(batch[0]).shape[-1] == (62 if dataset == "femnist" else 80)


def test_models_build():
    assert build_model("cifar10", "resnet8", 10, 0)(__import__("torch").zeros(2, 3, 32, 32)).shape == (2, 10)
    assert build_model("cifar100", None, 100, 0)(__import__("torch").zeros(2, 3, 32, 32)).shape == (2, 100)
