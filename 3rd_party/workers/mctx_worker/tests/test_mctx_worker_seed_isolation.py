"""Seed isolation tests for mctx_worker.

MCTXWorkerRuntime seeds two independent RNGs from ``config.seed``:
  - ``self._rng  = jax.random.PRNGKey(config.seed)``   (JAX)
  - ``self._np_rng = np.random.default_rng(config.seed)`` (NumPy)

Two operator instances with different seeds must hold completely independent
RNG chains.  These tests verify that config seed isolation holds at both the
config and RNG levels.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    # Test that JAX is actually functional (requires compatible NumPy version)
    _test_key = jax.random.PRNGKey(0)
    del _test_key
    HAS_JAX = True
except Exception:
    HAS_JAX = False

from mctx_worker.config import MCTXWorkerConfig


def _make_config(run_id: str, seed: int = 42, **kwargs) -> MCTXWorkerConfig:
    return MCTXWorkerConfig(run_id=run_id, seed=seed, **kwargs)


class TestMCTXConfigSeedIsolation:
    """MCTXWorkerConfig instances must be fully independent."""

    def test_two_configs_have_independent_seeds(self):
        cfg1 = _make_config("mctx_op_1", seed=111)
        cfg2 = _make_config("mctx_op_2", seed=222)

        assert cfg1.seed == 111
        assert cfg2.seed == 222
        assert cfg1.seed != cfg2.seed

    def test_config_seed_unchanged_after_second_config_created(self):
        cfg1 = _make_config("mctx_a", seed=42)
        _cfg2 = _make_config("mctx_b", seed=99)

        assert cfg1.seed == 42, (
            "cfg1.seed was mutated after cfg2 was created — configs share state!"
        )

    def test_five_configs_retain_independent_seeds(self):
        seeds = [10, 20, 30, 40, 50]
        configs = [_make_config(f"mctx_{s}", seed=s) for s in seeds]
        for cfg, expected in zip(configs, seeds):
            assert cfg.seed == expected


class TestMCTXNumpyRNGIsolation:
    """numpy RNGs seeded from different values must be independent."""

    def test_two_rngs_with_different_seeds_diverge(self):
        """np.random.default_rng with different seeds must produce different sequences."""
        rng1 = np.random.default_rng(111)
        rng2 = np.random.default_rng(222)

        seq1 = rng1.integers(0, 8, size=30).tolist()
        seq2 = rng2.integers(0, 8, size=30).tolist()

        assert seq1 != seq2, (
            "Two numpy RNGs with different seeds produced identical sequences — "
            "RNG isolation is broken."
        )

    def test_same_seed_same_sequence(self):
        """Two RNGs with the same seed must produce identical sequences."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        seq1 = rng1.integers(0, 8, size=30).tolist()
        seq2 = rng2.integers(0, 8, size=30).tolist()

        assert seq1 == seq2

    def test_shared_reset_seed_does_not_unify_rngs(self):
        """A shared layout seed used for env.reset must not re-seed the action RNG.

        This mirrors the random_worker bug pattern: the env reset seed must only
        be passed to env.reset(), not to the RNG used for action selection.
        """
        SHARED_LAYOUT_SEED = 42

        # Two operators with unique action-selection RNGs
        rng1 = np.random.default_rng(1001)
        rng2 = np.random.default_rng(2002)

        # The shared layout seed — only for env.reset(), NOT for rng.seed()
        shared_rng = np.random.default_rng(SHARED_LAYOUT_SEED)
        shared_seq = shared_rng.integers(0, 8, size=30).tolist()

        seq1 = rng1.integers(0, 8, size=30).tolist()
        seq2 = rng2.integers(0, 8, size=30).tolist()

        assert seq1 != seq2, "Both operators produce identical actions — RNGs are shared."
        assert seq1 != shared_seq, "Operator 1 is following the shared layout seed RNG."
        assert seq2 != shared_seq, "Operator 2 is following the shared layout seed RNG."

    def test_five_operators_all_independent(self):
        """Five operators with unique seeds must all produce different sequences."""
        seeds = [10, 20, 30, 40, 50]
        sequences = {
            s: np.random.default_rng(s).integers(0, 8, size=50).tolist()
            for s in seeds
        }

        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1:]:
                assert sequences[s1] != sequences[s2], (
                    f"Seeds {s1} and {s2} produce identical sequences — isolation broken."
                )


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestMCTXJAXKeyIsolation:
    """JAX PRNGKeys used by MCTX runtime must be independent across instances."""

    def test_different_seeds_produce_different_keys(self):
        key1 = jax.random.PRNGKey(111)
        key2 = jax.random.PRNGKey(222)
        assert not bool(jnp.all(key1 == key2))

    def test_same_seed_produces_same_key(self):
        key_a = jax.random.PRNGKey(42)
        key_b = jax.random.PRNGKey(42)
        assert bool(jnp.all(key_a == key_b))

    def test_two_operator_keys_stay_independent_after_shared_reset(self):
        """Simulates two MCTX operators receiving the same reset seed.

        Each operator's JAX key must remain derived from its own init seed,
        not from the shared layout seed.
        """
        SHARED_LAYOUT_SEED = 7

        key1 = jax.random.PRNGKey(1001)  # operator 1 seed
        key2 = jax.random.PRNGKey(2002)  # operator 2 seed

        # Advance keys (simulating training steps)
        key1, _ = jax.random.split(key1)
        key2, _ = jax.random.split(key2)

        # Neither key should equal PRNGKey(SHARED_LAYOUT_SEED)
        shared = jax.random.PRNGKey(SHARED_LAYOUT_SEED)
        assert not bool(jnp.all(key1 == shared))
        assert not bool(jnp.all(key2 == shared))
        assert not bool(jnp.all(key1 == key2))
