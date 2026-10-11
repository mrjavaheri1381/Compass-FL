"""
Privacy-preserving similarity S_ij = y_i_hat^T y_j_hat (paper Sec. 4.8).

Two modes:
- ``plaintext`` (default for FL simulation): S_ij is computed directly. It equals the CKKS result up to the CKKS error.
- ``ckks``: the real two-party protocol, implemented with TenSEAL's low-level SEAL API (``tenseal.sealapi``):
    1. initiator i encodes y_i_hat in the first K slots and sends c_i = Enc_{pk_i}(y_i_hat);
    2. responder j multiplies slot-wise by its plaintext y_j_hat, sums the K slots with ceil(log2 K) rotations (Galois
       keys of i), multiplies by the mask e_1, and re-randomizes by adding a fresh encryption of zero plus slot-domain
       flooding noise of std ``flood_sigma``; optional DP noise is added to slot 0;
    3. i decrypts slot 0.
  Without the e_1 mask the other slots would hold partial sums of y_i^k y_j^k (``mask=False`` exists for the leakage
  test only).

Optional DP noise (``dp_sigma``) perturbs the returned similarity in both modes.
"""
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dlsim.compass.core.similarity import normalize


@dataclass(frozen=True)
class CKKSParams:
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: Tuple[int, ...] = (60, 40, 40, 60)
    scale_bits: int = 40
    flood_sigma: float = 1e-7

    @classmethod
    def from_config(cls, ckks_config) -> "CKKSParams":
        return cls(ckks_config.poly_modulus_degree, tuple(ckks_config.coeff_mod_bit_sizes), ckks_config.scale_bits,
                   ckks_config.flood_sigma)


def _seal():
    try:
        import tenseal.sealapi as seal
    except ImportError as exc:  # pragma: no cover
        raise ImportError("he.mode = ckks requires TenSEAL (pip install tenseal)") from exc
    return seal


@lru_cache(maxsize=4)
def _context(params: CKKSParams):
    seal = _seal()
    parms = seal.EncryptionParameters(seal.SCHEME_TYPE.CKKS)
    parms.set_poly_modulus_degree(params.poly_modulus_degree)
    parms.set_coeff_modulus(seal.CoeffModulus.Create(params.poly_modulus_degree, list(params.coeff_mod_bit_sizes)))
    return seal.SEALContext(parms, True, seal.SEC_LEVEL_TYPE.TC128)


def rotation_steps(K: int) -> List[int]:
    return [1 << i for i in range(max(1, math.ceil(math.log2(max(K, 2)))))]


def serialized_size(obj) -> int:
    """
    Size in bytes of a SEAL object (written to a temporary file; used for byte accounting and the benchmark).
    """
    handle, path = tempfile.mkstemp()
    os.close(handle)
    try:
        obj.save(path)
        return os.path.getsize(path)
    finally:
        os.remove(path)


@dataclass
class PublicKeys:
    """
    pk_i and the Galois keys evk_i of a node; published once at join.
    """
    public_key: object
    galois_keys: object
    size_bytes: int = 0


class CKKSNode:
    """
    The CKKS key material and protocol steps of one node.
    """

    def __init__(self, params: CKKSParams, K: int) -> None:
        seal = _seal()
        self.params = params
        self.K = K
        self.context = _context(params)
        self.encoder = seal.CKKSEncoder(self.context)
        self.evaluator = seal.Evaluator(self.context)
        self.slots = self.encoder.slot_count()
        if K > self.slots:
            raise ValueError("K = %d exceeds the %d CKKS slots" % (K, self.slots))
        self.scale = 2.0 ** params.scale_bits

        start = time.perf_counter()
        keygen = seal.KeyGenerator(self.context)
        self._secret_key = keygen.secret_key()
        public_key = seal.PublicKey()
        keygen.create_public_key(public_key)
        galois_keys = seal.GaloisKeys()
        elements = self.context.key_context_data().galois_tool().get_elts_from_steps(rotation_steps(K))
        keygen.create_galois_keys(elements, galois_keys)
        self.keygen_time = time.perf_counter() - start
        self.public = PublicKeys(public_key, galois_keys)
        self._decryptor = seal.Decryptor(self.context, self._secret_key)
        self._encryptor = seal.Encryptor(self.context, public_key)

    def _pad(self, values: Sequence[float]) -> List[float]:
        values = [float(v) for v in values]
        return values + [0.0] * (self.slots - len(values))

    def encrypt(self, y_hat: Sequence[float]):
        """
        Step 1 (initiator): c_i = Enc_{pk_i}(y_i_hat).
        """
        seal = _seal()
        plain = seal.Plaintext()
        self.encoder.encode(self._pad(y_hat), self.scale, plain)
        cipher = seal.Ciphertext()
        self._encryptor.encrypt(plain, cipher)
        return cipher

    def respond(self, cipher, initiator: PublicKeys, y_hat: Sequence[float], rng: np.random.Generator,
                mask: bool = True, dp_sigma: float = 0.0):
        """
        Step 2 (responder j): Enc(y_i_hat^T y_j_hat) with slot rotations, e_1 mask and flooding re-randomization.
        The received ciphertext is not modified.
        """
        seal = _seal()
        ev = self.evaluator
        plain = seal.Plaintext()
        self.encoder.encode(self._pad(y_hat), cipher.parms_id(), self.scale, plain)
        result = seal.Ciphertext()
        ev.multiply_plain(cipher, plain, result)  # Out-of-place: the received ciphertext stays untouched
        ev.rescale_to_next_inplace(result)
        for step in rotation_steps(self.K):
            rotated = seal.Ciphertext()
            ev.rotate_vector(result, step, initiator.galois_keys, rotated)
            ev.add_inplace(result, rotated)
        if mask:
            e1 = seal.Plaintext()
            self.encoder.encode(self._pad([1.0]), result.parms_id(), self.scale, e1)
            ev.multiply_plain_inplace(result, e1)
            ev.rescale_to_next_inplace(result)
        # Re-randomization: a fresh encryption of zero under pk_i, plus flooding (and optional DP) noise
        zero = seal.Ciphertext()
        seal.Encryptor(self.context, initiator.public_key).encrypt_zero(result.parms_id(), zero)
        zero.scale = result.scale
        noise = rng.normal(0.0, self.params.flood_sigma, self.slots) if self.params.flood_sigma > 0 \
            else np.zeros(self.slots)
        if dp_sigma > 0:
            noise[0] += rng.normal(0.0, dp_sigma)
        noise_plain = seal.Plaintext()
        self.encoder.encode(list(noise), result.parms_id(), result.scale, noise_plain)
        ev.add_plain_inplace(zero, noise_plain)
        ev.add_inplace(result, zero)
        return result

    def decrypt_slots(self, cipher) -> np.ndarray:
        seal = _seal()
        plain = seal.Plaintext()
        self._decryptor.decrypt(cipher, plain)
        return np.array(self.encoder.decode_double(plain))

    def decrypt_similarity(self, cipher) -> float:
        """
        Step 3 (initiator): S_ij ~ Dec_{sk_i}(c_{i,j}), slot 0.
        """
        return float(self.decrypt_slots(cipher)[0])


class SimilarityEngine:
    """
    The similarity computation a node performs, in either mode.
    """

    def __init__(self, mode: str, label_counts: Sequence[float], K: int, params: Optional[CKKSParams] = None,
                 dp_sigma: float = 0.0) -> None:
        if mode not in ("plaintext", "ckks"):
            raise ValueError("Unknown HE mode %s" % mode)
        self.mode = mode
        self.y_hat = normalize(label_counts)
        self.K = K
        self.dp_sigma = dp_sigma
        self.ckks: Optional[CKKSNode] = CKKSNode(params or CKKSParams(), K) if mode == "ckks" else None
        self.answered: set = set()  # Initiators this node already answered (each identity is answered once)
        self.stats = {"encrypt": 0.0, "respond": 0.0, "decrypt": 0.0, "num_respond": 0, "num_decrypt": 0,
                      "refused": 0}
        self._ciphertext = None

    # --- plaintext mode
    def direct(self, other_y_hat: Sequence[float], rng: np.random.Generator) -> float:
        value = float(np.dot(self.y_hat, other_y_hat))
        if self.dp_sigma > 0:
            value += float(rng.normal(0.0, self.dp_sigma))
        return value

    # --- ckks mode
    def public_ciphertext(self):
        if self._ciphertext is None:
            start = time.perf_counter()
            self._ciphertext = self.ckks.encrypt(self.y_hat)
            self.stats["encrypt"] += time.perf_counter() - start
        return self._ciphertext

    def respond(self, initiator, cipher, keys: PublicKeys, rng: np.random.Generator):
        """
        Returns None if this initiator was already answered (one projection per identity).
        """
        if initiator in self.answered:
            self.stats["refused"] += 1
            return None
        self.answered.add(initiator)
        start = time.perf_counter()
        result = self.ckks.respond(cipher, keys, self.y_hat, rng, dp_sigma=self.dp_sigma)
        self.stats["respond"] += time.perf_counter() - start
        self.stats["num_respond"] += 1
        return result

    def decrypt(self, cipher) -> float:
        start = time.perf_counter()
        value = self.ckks.decrypt_similarity(cipher)
        self.stats["decrypt"] += time.perf_counter() - start
        self.stats["num_decrypt"] += 1
        return value
