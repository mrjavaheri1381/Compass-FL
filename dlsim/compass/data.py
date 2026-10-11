"""
Data, models and local training for COMPASS-FL runs (task G).

- Partitions are seeded from the run seed: Dirichlet(alpha) label partitioning or IID for CIFAR-10/100, and natural
  per-user partitions for the LEAF datasets (FEMNIST, Shakespeare hooks). With ``bypass_training`` the same partitioner
  runs on a synthetic balanced label pool, so label counts and D_i are realistic without loading images.
- Models: the dataset default (GNLeNet for CIFAR, the FEMNIST CNN) or a small GroupNorm ResNet (``resnet8``), or a
  character LSTM for Shakespeare.
- Training: E local epochs of SGD with cross-entropy (works for logits and log-probabilities alike).
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from dlsim.compass.core.hashing import rng_seed

logger = logging.getLogger("CompassData")

NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "femnist": 62, "shakespeare": 80}
LEAF_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"


# ---------------------------------------------------------------------- partitioning

def dirichlet_partition(labels: np.ndarray, N: int, alpha: float, rng: np.random.Generator,
                        min_samples: int = 1, max_tries: int = 100) -> List[np.ndarray]:
    """
    For each class, split its samples over the N nodes with proportions ~ Dir(alpha). Redrawn until every node holds
    at least ``min_samples`` samples.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    for _ in range(max_tries):
        parts: List[List[int]] = [[] for _ in range(N)]
        for cls in classes:
            idx = np.flatnonzero(labels == cls)
            rng.shuffle(idx)
            proportions = rng.dirichlet([alpha] * N)
            cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
            for node, chunk in enumerate(np.split(idx, cuts)):
                parts[node].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_samples:
            return [np.array(sorted(p), dtype=int) for p in parts]
    raise ValueError("Could not give every node %d samples with alpha=%g; lower data.min_samples" % (min_samples, alpha))


def iid_partition(size: int, N: int, rng: np.random.Generator) -> List[np.ndarray]:
    idx = rng.permutation(size)
    return [np.array(sorted(part), dtype=int) for part in np.array_split(idx, N)]


# ---------------------------------------------------------------------- datasets

@dataclass
class NodeData:
    indices: Optional[np.ndarray]
    label_counts: np.ndarray
    size: int


class FederatedData:
    """
    The training data of all nodes and the test set.
    """

    def __init__(self, config, N: int, seed: int) -> None:
        self.config = config
        self.dataset = config.data.dataset
        self.num_classes = config.data.num_classes or NUM_CLASSES[self.dataset]
        self.base = config.data.base_path or os.environ.get("HOME", ".")
        self.rng = np.random.default_rng(rng_seed(seed, "partition"))
        self.trainset: Optional[Dataset] = None
        self.testset: Optional[Dataset] = None
        self.stats: Dict = {}
        self.nodes: List[NodeData] = []
        if self.dataset in ("cifar10", "cifar100"):
            self._vision(N)
        elif self.dataset in ("femnist", "shakespeare"):
            self._leaf(N)
        else:
            raise ValueError("Unknown dataset %s" % self.dataset)
        sizes = [n.size for n in self.nodes]
        self.stats.update({"dataset": self.dataset, "partition": config.data.partition, "alpha": config.data.alpha,
                           "D_min": int(min(sizes)), "D_median": float(np.median(sizes)), "D_max": int(max(sizes)),
                           "D_total": int(sum(sizes))})

    # -- CIFAR-10/100
    def _vision(self, N: int) -> None:
        cfg = self.config.data
        if cfg.bypass_training:
            labels = np.arange(cfg.synthetic_train_size) % self.num_classes
        else:
            self.trainset, self.testset = self._load_cifar()
            labels = np.asarray(self.trainset.targets)
        if cfg.partition == "dirichlet":
            parts = dirichlet_partition(labels, N, cfg.alpha, self.rng, cfg.min_samples)
        elif cfg.partition == "iid":
            parts = iid_partition(len(labels), N, self.rng)
        else:
            raise ValueError("Partition %s is not available for %s" % (cfg.partition, self.dataset))
        for part in parts:
            counts = np.bincount(labels[part], minlength=self.num_classes)
            self.nodes.append(NodeData(None if cfg.bypass_training else part, counts, len(part)))

    def _load_cifar(self):
        import torchvision
        import torchvision.transforms as T
        root = os.path.join(self.base, "dfl-data")
        if self.dataset == "cifar10":
            mean, std, cls = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201), torchvision.datasets.CIFAR10
        else:
            mean, std, cls = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), torchvision.datasets.CIFAR100
        train_t = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
        test_t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        download = self.config.data.download
        return (cls(root=root, train=True, download=download, transform=train_t),
                cls(root=root, train=False, download=download, transform=test_t))

    # -- LEAF (natural partitions)
    def _leaf(self, N: int) -> None:
        cfg = self.config.data
        if cfg.partition != "natural":
            raise ValueError("%s uses its natural per-user partition (data.partition = natural)" % self.dataset)
        root = os.path.join(self.base, "leaf", self.dataset, "data")
        train = read_leaf_dir(os.path.join(root, "train"))
        users = sorted(train)
        if len(users) < N:
            raise ValueError("%s has %d users, fewer than N = %d" % (self.dataset, len(users), N))
        chosen = [users[int(i)] for i in sorted(self.rng.choice(len(users), size=N, replace=False))]
        xs, ys, self.nodes = [], [], []
        offset = 0
        for user in chosen:
            x, y = self._leaf_tensors(train[user])
            xs.append(x)
            ys.append(y)
            idx = np.arange(offset, offset + len(y))
            offset += len(y)
            self.nodes.append(NodeData(None if cfg.bypass_training else idx,
                                       np.bincount(y.numpy(), minlength=self.num_classes), len(y)))
        if not cfg.bypass_training:
            self.trainset = TensorDataset(torch.cat(xs), torch.cat(ys))
            test = read_leaf_dir(os.path.join(root, "test"))
            tx, ty = zip(*(self._leaf_tensors(test[u]) for u in chosen if u in test))
            self.testset = TensorDataset(torch.cat(tx), torch.cat(ty))
        self.stats["leaf_users"] = len(users)

    def _leaf_tensors(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dataset == "femnist":
            x = torch.tensor(data["x"], dtype=torch.float32).reshape(-1, 1, 28, 28)
            y = torch.tensor(data["y"], dtype=torch.long)
        else:
            x = torch.tensor([[max(0, LEAF_LETTERS.find(ch)) for ch in s] for s in data["x"]], dtype=torch.long)
            y = torch.tensor([max(0, LEAF_LETTERS.find(ch)) for ch in data["y"]], dtype=torch.long)
        return x, y

    # -- access
    def loader(self, node: int, batch_size: int, generator: torch.Generator) -> DataLoader:
        subset = Subset(self.trainset, self.nodes[node].indices.tolist())
        return DataLoader(subset, batch_size=batch_size, shuffle=True, generator=generator)

    def test_loader(self, batch_size: int = 512) -> DataLoader:
        return DataLoader(self.testset, batch_size=batch_size, shuffle=False)


def read_leaf_dir(path: str) -> Dict[str, Dict]:
    users: Dict[str, Dict] = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".json"):
            with open(os.path.join(path, name)) as handle:
                content = json.load(handle)
            users.update(content["user_data"])
    return users


# ---------------------------------------------------------------------- models

class BasicBlock(nn.Module):
    def __init__(self, inp: int, out: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(2, out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(2, out)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out:
            self.shortcut = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.GroupNorm(2, out))

    def forward(self, x):
        y = F.relu(self.n1(self.conv1(x)))
        return F.relu(self.n2(self.conv2(y)) + self.shortcut(x))


class ResNet8(nn.Module):
    """
    A small ResNet (3 stages of one basic block) with GroupNorm, so FedAvg of parameters is well defined.
    """

    def __init__(self, num_classes: int = 10, width: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.GroupNorm(2, width), nn.ReLU())
        self.layers = nn.Sequential(BasicBlock(width, width, 1), BasicBlock(width, 2 * width, 2),
                                    BasicBlock(2 * width, 4 * width, 2))
        self.fc = nn.Linear(4 * width, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class CharLSTM(nn.Module):
    def __init__(self, vocab: int = 80, embed: int = 8, hidden: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        return self.fc(out[:, -1, :])


def build_model(dataset: str, architecture: Optional[str], num_classes: int, seed: int) -> nn.Module:
    torch.manual_seed(rng_seed(seed, "model") % (2 ** 31))
    architecture = architecture or {"cifar10": "gnlenet", "cifar100": "gnlenet", "femnist": "cnn",
                                    "shakespeare": "lstm"}[dataset]
    if architecture == "gnlenet":
        from dlsim.core.models.cifar10 import GNLeNet
        return GNLeNet(input_channel=3, output=num_classes, model_input=(32, 32))
    if architecture == "resnet8":
        return ResNet8(num_classes)
    if architecture == "cnn" and dataset == "femnist":
        from dlsim.core.models.femnist import CNN
        return CNN()
    if architecture == "lstm":
        return CharLSTM(vocab=num_classes)
    raise ValueError("Unknown architecture %s for dataset %s" % (architecture, dataset))


# ---------------------------------------------------------------------- training and evaluation

def train_local(model: nn.Module, loader: DataLoader, epochs: int, lr: float, momentum: float, weight_decay: float,
                max_batches: int = 0, device: str = "cpu") -> Tuple[nn.Module, float, int]:
    """
    E local epochs of SGD. Returns (model, mean training loss, number of samples processed).
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    total_loss, samples = 0.0, 0
    for _ in range(epochs):
        for batch, (x, y) in enumerate(loader):
            if max_batches and batch >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(y)
            samples += len(y)
    model.to("cpu")
    return model, total_loss / max(1, samples), samples


def evaluate_model(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Tuple[float, float]:
    model.to(device)
    model.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += float(F.cross_entropy(out, y, reduction="sum"))
            correct += int((out.argmax(dim=1) == y).sum())
            total += len(y)
    model.to("cpu")
    return correct / max(1, total), loss / max(1, total)


def local_loss(model: nn.Module, loader: DataLoader, max_batches: int = 0) -> float:
    model.eval()
    loss, total = 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(loader):
            if max_batches and batch >= max_batches:
                break
            loss += float(F.cross_entropy(model(x), y, reduction="sum"))
            total += len(y)
    return loss / max(1, total)
