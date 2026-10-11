"""
Hash-based candidate pool (paper Sec. 4.2): M^(t) = TopM_i H(i || t), the M nodes with the smallest hash values.

The resample index k (Q5) gives M^(t,k) = TopM_i H(i || t || k); k = 0 is the regular pool. With ``pool_seed =
model_hash`` the round index is replaced by an unpredictable shared seed (a hash of the current model).
"""
import hashlib
from typing import Iterable, List, Optional, Union

Seed = Union[int, bytes]


def node_hash(node_id: bytes, seed: Seed, k: int = 0) -> bytes:
    seed_bytes = seed if isinstance(seed, bytes) else str(int(seed)).encode()
    message = b"compass|" + bytes(node_id) + b"|" + seed_bytes
    if k:
        message += b"|" + str(int(k)).encode()
    return hashlib.sha256(message).digest()


def ordered(nodes: Iterable[bytes], seed: Seed, k: int = 0) -> List[bytes]:
    return sorted(nodes, key=lambda node: (node_hash(node, seed, k), node))


def pool(nodes: Iterable[bytes], seed: Seed, M: int, k: int = 0) -> List[bytes]:
    """
    TopM by hash, returned in hash order.
    """
    return ordered(nodes, seed, k)[:M]


def model_seed(model_bytes: bytes) -> bytes:
    return hashlib.sha256(model_bytes).digest()


def rng_seed(*parts) -> int:
    """
    A stable 64-bit seed from any parts (bytes, str, int); used for per-(node, round) generators.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else str(part).encode())
        digest.update(b"|")
    return int.from_bytes(digest.digest()[:8], "big")
