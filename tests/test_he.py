import numpy as np
import pytest

from dlsim.compass.core.similarity import cosine, normalize

tenseal = pytest.importorskip("tenseal")

from dlsim.compass.he import CKKSNode, CKKSParams, SimilarityEngine  # noqa: E402


@pytest.fixture(scope="module")
def nodes():
    params = CKKSParams()
    return CKKSNode(params, K=10), CKKSNode(params, K=10)


def test_ckks_similarity_matches_plaintext(nodes):
    """Test 14: decrypted similarity ~ plaintext."""
    initiator, responder = nodes
    rng = np.random.default_rng(0)
    for _ in range(5):
        y_i, y_j = rng.multinomial(300, rng.dirichlet([0.5] * 10)), rng.multinomial(300, rng.dirichlet([0.5] * 10))
        cipher = initiator.encrypt(normalize(y_i))
        result = responder.respond(cipher, initiator.public, normalize(y_j), rng)
        assert initiator.decrypt_similarity(result) == pytest.approx(cosine(y_i, y_j), abs=1e-4)


def test_mask_zeroes_other_slots_and_unmasked_slots_leak(nodes):
    """Test 14 / F6: without e_1 the other slots hold partial sums that reveal y_j; with it they decrypt to ~0."""
    initiator, responder = nodes
    rng = np.random.default_rng(1)
    y_i = normalize(rng.random(10) + 0.1)
    y_j = normalize(rng.random(10) + 0.1)
    products = y_i * y_j
    cipher = initiator.encrypt(y_i)

    unmasked = initiator.decrypt_slots(responder.respond(cipher, initiator.public, y_j, rng, mask=False))
    # After the rotate-and-sum, slot k holds sum_{l=k}^{k+15} y_i^l y_j^l: consecutive differences give single
    # products, from which the initiator (who knows y_i) recovers y_j.
    partial = [products[k:k + 16].sum() for k in range(10)]
    assert np.allclose(unmasked[:10], partial, atol=1e-4)
    recovered = np.array([unmasked[k] - unmasked[k + 1] for k in range(9)]) / y_i[:9]
    assert np.allclose(recovered, y_j[:9], atol=1e-3)

    masked = initiator.decrypt_slots(responder.respond(cipher, initiator.public, y_j, rng, mask=True))
    assert masked[0] == pytest.approx(products.sum(), abs=1e-4)
    assert np.abs(masked[1:]).max() < 1e-5


def test_each_identity_answered_once():
    engine_i = SimilarityEngine("ckks", [3, 1, 0, 2, 5, 0, 0, 1, 2, 4], K=10)
    engine_j = SimilarityEngine("ckks", [1, 0, 2, 2, 1, 3, 0, 0, 1, 1], K=10)
    rng = np.random.default_rng(0)
    cipher = engine_i.public_ciphertext()
    first = engine_j.respond("i", cipher, engine_i.ckks.public, rng)
    assert first is not None and engine_j.respond("i", cipher, engine_i.ckks.public, rng) is None
    assert engine_i.decrypt(first) == pytest.approx(cosine([3, 1, 0, 2, 5, 0, 0, 1, 2, 4], [1, 0, 2, 2, 1, 3, 0, 0, 1, 1]),
                                                   abs=1e-4)


def test_plaintext_mode_and_dp_noise():
    a = SimilarityEngine("plaintext", [1, 2, 3], K=3)
    b = SimilarityEngine("plaintext", [3, 2, 1], K=3)
    assert a.direct(b.y_hat, np.random.default_rng(0)) == pytest.approx(cosine([1, 2, 3], [3, 2, 1]))
    noisy = SimilarityEngine("plaintext", [1, 2, 3], K=3, dp_sigma=0.05)
    values = [noisy.direct(b.y_hat, np.random.default_rng(s)) for s in range(200)]
    assert np.std(values) == pytest.approx(0.05, rel=0.25)
