"""
Random instances shared by the tests.
"""
import numpy as np

from dlsim.compass.core.similarity import normalize
from dlsim.compass.core.utility import SelectionProblem, UtilityParams


def random_label_vectors(rng, count, classes=10, alpha=0.5):
    return [rng.multinomial(200, rng.dirichlet([alpha] * classes)) + (rng.random(classes) < 0.05) for _ in range(count)]


def random_problem(seed, k=10, n=4, form="multiplicative", w_c=0.5, w_r=0.5, N=50, weights=None):
    rng = np.random.default_rng(seed)
    vectors = [normalize(v) for v in random_label_vectors(rng, N)]
    Y = np.array(vectors)
    S_full = Y @ Y.T
    r_full = S_full.sum(axis=1)
    idx = sorted(rng.choice(N, size=k, replace=False).tolist())
    params = UtilityParams.create(form, weights, w_c, w_r, f_ref=40.0, L_max=0.4, dtau_cap=10.0, eps=1e-3)
    return SelectionProblem(ids=idx, n=n, f=rng.uniform(5, 60, k), L=rng.uniform(0.01, 0.39, k),
                            stale=rng.integers(1, 15, k), Q=rng.uniform(0.2, 1.0, k), r=r_full[idx],
                            S=S_full[np.ix_(idx, idx)], net_sum=float(r_full.sum()), params=params, aggregator=-1), Y, idx
