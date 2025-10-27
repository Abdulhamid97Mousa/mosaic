"""Test suite for refactored BDI agent (bdi_agent.py).

This tests the pure-Python BDI implementation with no legacy imports.
"""

import pytest
import logging
from pathlib import Path

from ..core.bdi_agent import BDIRLAgent, AgentHandle
from ..core.agent import create_agent
from ..adapters import FrozenLakeAdapter
from ..algorithms import QLearningAgent

LOGGER = logging.getLogger(__name__)


class TestBDIRLAgentInstantiation:
    """Test BDIRLAgent class instantiation and attribute initialization."""

    def test_create_agent_basic(self):
        """Test basic agent instantiation with defaults."""
        agent = BDIRLAgent(jid="test@localhost", password="secret")
        
        assert agent is not None
        assert agent.jid == "test@localhost"
        assert agent.adapter is not None
        assert agent.rl_agent is not None
        assert agent.runtime is not None
        assert isinstance(agent.adapter, FrozenLakeAdapter)
        assert isinstance(agent.rl_agent, QLearningAgent)

    def test_create_agent_with_custom_adapter(self):
        """Test agent instantiation with custom adapter."""
        adapter = FrozenLakeAdapter(map_size="4x4")
        agent = BDIRLAgent(jid="test@localhost", password="secret", adapter=adapter)
        
        assert agent.adapter is adapter
        # FrozenLakeAdapter has map_size and _env attributes after __post_init__
        if isinstance(agent.adapter, FrozenLakeAdapter):
            assert hasattr(agent.adapter, "map_size")
            assert hasattr(agent.adapter, "_env")
            assert agent.adapter.map_size == "4x4"

    def test_agent_initial_state(self):
        """Test agent has proper initial state."""
        agent = BDIRLAgent(jid="test@localhost", password="secret")
        
        assert agent.episode_count == 0
        assert agent.episode_steps == 0
        assert agent.episode_rewards == []
        assert agent.success_history == []
        assert agent.current_state == 0

    def test_agent_has_methods(self):
        """Test agent has expected methods."""
        agent = BDIRLAgent(jid="test@localhost", password="secret")
        
        # Should have lifecycle methods
        assert hasattr(agent, "start")
        assert hasattr(agent, "stop")
        assert hasattr(agent, "setup")
        
        # Should have training methods
        assert hasattr(agent, "run_episode")
        
        # Should have action methods
        assert hasattr(agent, "execute_action")
        assert hasattr(agent, "cache_policy")
        assert hasattr(agent, "load_cached_policy")

    def test_agent_handle_creation(self):
        """Test AgentHandle wrapping."""
        agent = BDIRLAgent(jid="test@localhost", password="secret")
        handle = AgentHandle(agent=agent, jid="test@localhost", password="secret")
        
        assert handle.agent is agent
        assert handle.jid == "test@localhost"
        assert handle.started is False

    def test_factory_function_basic(self):
        """Test create_agent factory function."""
        handle = create_agent(jid="test@localhost", password="secret")
        
        assert handle is not None
        assert handle.jid == "test@localhost"
        assert handle.agent is not None
        assert isinstance(handle.agent, BDIRLAgent)


class TestBDIRLAgentQLearningIntegration:
    """Test integration between BDI agent and Q-Learning."""

    def test_agent_has_qlearning_components(self):
        """Test that agent properly initializes Q-Learning."""
        agent = BDIRLAgent(jid="test@localhost", password="secret")
        
        assert agent.rl_agent is not None
        assert agent.runtime is not None
        assert hasattr(agent.rl_agent, "select_action")
        assert hasattr(agent.rl_agent, "update")
        assert hasattr(agent.runtime, "train")

    def test_agent_epsilon_tracking(self):
        """Test that agent tracks epsilon for exploration."""
        agent = BDIRLAgent(jid="test@localhost", password="secret")
        
        initial_epsilon = agent.rl_agent.epsilon
        assert initial_epsilon > 0
        assert initial_epsilon <= 1.0

    def test_agent_policy_caching(self):
        """Test policy caching infrastructure."""
        agent = BDIRLAgent(jid="test@localhost", password="secret")
        
        # Should have empty cache initially
        assert isinstance(agent.cached_policies, dict)
        assert len(agent.cached_policies) == 0
        
        # Should have cache_policy method
        assert hasattr(agent, "cache_policy")


class TestBDIRLAgentAdapterIntegration:
    """Test integration between BDI agent and environment adapter."""

    def test_agent_has_adapter(self):
        """Test that agent has environment adapter."""
        agent = BDIRLAgent(jid="test@localhost", password="secret")
        
        assert agent.adapter is not None
        assert hasattr(agent.adapter, "reset")
        assert hasattr(agent.adapter, "step")

    def test_agent_action_execution(self):
        """Test agent can execute actions through adapter."""
        agent = BDIRLAgent(jid="test@localhost", password="secret")
        
        # Should have execute_action method
        assert hasattr(agent, "execute_action")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
