"""
Benchmark of the CKKS similarity protocol (paper Sec. 4.8) for a range of K.

Reports key generation, encryption, responder evaluation and decryption times, the sizes of the public keys, the
query ciphertext and the response ciphertext, and the absolute error versus the plaintext cosine similarity.

    python scripts/bench_ckks.py --K 2 10 62 100 --trials 20 --out bench_ckks.csv
"""
import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlsim.compass.core.similarity import normalize  # noqa: E402
from dlsim.compass.he import CKKSNode, CKKSParams, serialized_size  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, nargs="+", default=[2, 10, 62, 100])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--poly-modulus-degree", type=int, default=8192)
    parser.add_argument("--coeff-mod-bit-sizes", type=int, nargs="+", default=[60, 40, 40, 60])
    parser.add_argument("--scale-bits", type=int, default=40)
    parser.add_argument("--flood-sigma", type=float, default=1e-7)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration of the label vectors")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    params = CKKSParams(args.poly_modulus_degree, tuple(args.coeff_mod_bit_sizes), args.scale_bits, args.flood_sigma)
    rng = np.random.default_rng(args.seed)
    rows = []
    for K in args.K:
        initiator, responder = CKKSNode(params, K), CKKSNode(params, K)
        times = {"encrypt": [], "respond": [], "decrypt": []}
        errors = []
        query = response = None
        for _ in range(args.trials):
            y_i = normalize(rng.multinomial(500, rng.dirichlet([args.alpha] * K)) + 1e-9)
            y_j = normalize(rng.multinomial(500, rng.dirichlet([args.alpha] * K)) + 1e-9)
            start = time.perf_counter()
            query = initiator.encrypt(y_i)
            times["encrypt"].append(time.perf_counter() - start)
            start = time.perf_counter()
            response = responder.respond(query, initiator.public, y_j, rng)
            times["respond"].append(time.perf_counter() - start)
            start = time.perf_counter()
            value = initiator.decrypt_similarity(response)
            times["decrypt"].append(time.perf_counter() - start)
            errors.append(abs(value - float(np.dot(y_i, y_j))))
        row = {
            "K": K,
            "rotations": len(__import__("dlsim.compass.he", fromlist=["rotation_steps"]).rotation_steps(K)),
            "keygen_s": initiator.keygen_time,
            "encrypt_ms": 1000 * statistics.median(times["encrypt"]),
            "respond_ms": 1000 * statistics.median(times["respond"]),
            "decrypt_ms": 1000 * statistics.median(times["decrypt"]),
            "public_key_bytes": serialized_size(initiator.public.public_key),
            "galois_keys_bytes": serialized_size(initiator.public.galois_keys),
            "query_bytes": serialized_size(query),
            "response_bytes": serialized_size(response),
            "abs_error_median": statistics.median(errors),
            "abs_error_max": max(errors),
        }
        rows.append(row)
        print(", ".join("%s=%s" % (key, ("%.4g" % value) if isinstance(value, float) else value)
                        for key, value in row.items()))

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
