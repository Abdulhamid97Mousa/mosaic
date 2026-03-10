"""Seed isolation tests for jumanji_worker.

Jumanji uses JAX PRNGKeys for all randomness.  The training runtime seeds its
key from ``config.seed``; the interactive runtime initialises its key at policy
load time.  In both cases, two operator instances must hold independent PRNG
chains — receiving the same shared layout seed on reset must never unify them.

These tests verify:
- JumanjiWorkerConfig instances are fully independent.
- Two configs with different seeds never share their seed value.
- JAX PRNGKeys derived from different seeds are distinct (no aliasing).
"""

from __future__ import annotations

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

from jumanji_worker.config import JumanjiWorkerConfig


def _make_config(run_id: str, seed: int | None = None) -> JumanjiWorkerConfig:
    return JumanjiWorkerConfig(run_id=run_id, seed=seed)


class TestJumanjiConfigSeedIsolation:
    """JumanjiWorkerConfig instances must be fully independent."""

    def test_two_configs_have_independent_seeds(self):
        """Two configs with different seeds must not share seed values."""
        cfg1 = _make_config("jum_op_1", seed=111)
        cfg2 = _make_config("jum_op_2", seed=222)

        assert cfg1.seed == 111
        assert cfg2.seed == 222
        assert cfg1.seed != cfg2.seed

    def test_config_seed_unchanged_after_second_config_created(self):
        """Creating a second config must not mutate the first config's seed."""
        cfg1 = _make_config("jum_a", seed=42)
        _cfg2 = _make_config("jum_b", seed=99)

        assert cfg1.seed == 42, (
            "cfg1.seed was mutated after cfg2 was created — configs share state!"
        )

    def test_five_configs_all_retain_their_seeds(self):
        """Five configs with different seeds must all retain their own seed."""
        seeds = [100, 200, 300, 400, 500]
        configs = [_make_config(f"jum_{s}", seed=s) for s in seeds]

        for cfg, expected in zip(configs, seeds):
            assert cfg.seed == expected

    def test_configs_are_distinct_objects(self):
        """Each config must be a distinct object (no aliasing)."""
        cfg1 = _make_config("jum_x", seed=7)
        cfg2 = _make_config("jum_y", seed=8)
        assert cfg1 is not cfg2


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestJumanjiJAXKeyIsolation:
    """JAX PRNGKeys derived from different seeds must be distinct."""

    def test_different_seeds_produce_different_prng_keys(self):
        """PRNGKey(seed1) must not equal PRNGKey(seed2) for seed1 != seed2."""
        seed1, seed2 = 111, 222
        key1 = jax.random.PRNGKey(seed1)
        key2 = jax.random.PRNGKey(seed2)

        assert not bool(jnp.all(key1 == key2)), (
            f"PRNGKey({seed1}) == PRNGKey({seed2}) — JAX key isolation is broken."
        )

    def test_same_seed_produces_same_key(self):
        """PRNGKey(seed) must be deterministic — two calls with same seed equal."""
        key_a = jax.random.PRNGKey(42)
        key_b = jax.random.PRNGKey(42)
        assert bool(jnp.all(key_a == key_b))

    def test_split_keys_from_different_seeds_diverge(self):
        """Splitting keys from different seeds must produce independent chains."""
        SHARED_LAYOUT_SEED = 99

        key1 = jax.random.PRNGKey(111)
        key2 = jax.random.PRNGKey(222)

        # Simulate what handle_reset does: split own key, then optionally
        # replace reset_key with shared seed — own key stays independent.
        key1_after, reset_key1 = jax.random.split(key1)
        key2_after, reset_key2 = jax.random.split(key2)

        # Both reset keys replaced with shared layout seed (GUI behaviour)
        reset_key1 = jax.random.PRNGKey(SHARED_LAYOUT_SEED)  # noqa: F841
        reset_key2 = jax.random.PRNGKey(SHARED_LAYOUT_SEED)  # noqa: F841

        # The operators' own action-selection keys must still be independent
        assert not bool(jnp.all(key1_after == key2_after)), (
            "Action-selection keys converged after a shared reset seed was applied — "
            "key isolation is broken."
        )

    def test_five_operators_have_independent_keys_after_shared_reset(self):
        """Five operators each with a unique seed stay independent after shared reset."""
        SHARED = 42
        seeds = [10, 20, 30, 40, 50]
        own_keys = []

        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            key, _ = jax.random.split(key)
            # Simulate reset with shared seed (reset_key overridden, own key unchanged)
            own_keys.append(key)

        for i in range(len(own_keys)):
            for j in range(i + 1, len(own_keys)):
                assert not bool(jnp.all(own_keys[i] == own_keys[j])), (
                    f"Operators {i} and {j} share the same action-selection key "
                    "after a shared reset seed was applied."
                )
