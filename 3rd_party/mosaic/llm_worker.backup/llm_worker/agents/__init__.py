from ..client import create_llm_client

from ..prompt_builder import create_prompt_builder
from .base import BaseAgent
from .chain_of_thought import ChainOfThoughtAgent
from .custom import CustomAgent
from .dummy import DummyAgent
from .few_shot import FewShotAgent
from .naive import NaiveAgent
from .robust_naive import RobustNaiveAgent
from .robust_cot import RobustCoTAgent


class AgentFactory:
    """Factory class for creating agents based on configuration.

    The `AgentFactory` class is responsible for initializing the appropriate agent type
    based on the provided configuration, which includes setting up the LLM client and
    prompt builder.
    """

    def __init__(self, config):
        """Initialize the AgentFactory with configuration settings.

        Args:
            config (omegaconf.DictConfig): Configuration object containing settings for the agent and client.
        """
        self.config = config

    def create_agent(self):
        """Create an agent instance based on the agent type specified in the configuration.

        The function uses the `config.agent.type` attribute to determine which agent to create.
        It supports several agent types, including Naive, Chain-of-Thought, Self-Refine, Dummy,
        and Custom agents.

        Returns:
            Agent: An instance of the selected agent type, configured with the client and prompt builder.

        Raises:
            ValueError: If an unknown agent type is specified in the configuration.
        """
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)

        if self.config.agent.type == "naive":
            return NaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "cot":
            return ChainOfThoughtAgent(client_factory, prompt_builder, config=self.config)
        elif self.config.agent.type == "dummy":
            return DummyAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "custom":
            return CustomAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "few_shot":
            return FewShotAgent(client_factory, prompt_builder, self.config.agent.max_icl_history)
        elif self.config.agent.type == "robust_naive":
            return RobustNaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "robust_cot":
            return RobustCoTAgent(client_factory, prompt_builder, config=self.config)

        else:
            raise ValueError(f"Unknown agent type: {self.config.agent}")


def create_agent(agent_type, client_factory, prompt_builder, config=None):
    """Factory function to create an agent by type.

    Args:
        agent_type: Type of agent to create ('naive', 'cot', 'robust_naive', etc.).
        client_factory: Factory function that creates LLM client instances.
        prompt_builder: Prompt builder for constructing messages.
        config: Optional configuration object for agents that need it.

    Returns:
        Agent instance of the selected type.
    """
    agent_type = agent_type.lower()

    if agent_type == "naive":
        return NaiveAgent(client_factory, prompt_builder)
    elif agent_type == "cot":
        return ChainOfThoughtAgent(client_factory, prompt_builder, config=config)
    elif agent_type == "dummy":
        return DummyAgent(client_factory, prompt_builder)
    elif agent_type == "custom":
        return CustomAgent(client_factory, prompt_builder)
    elif agent_type == "few_shot":
        max_icl_history = getattr(config, "max_icl_history", 16) if config else 16
        return FewShotAgent(client_factory, prompt_builder, max_icl_history)
    elif agent_type == "robust_naive":
        return RobustNaiveAgent(client_factory, prompt_builder)
    elif agent_type == "robust_cot":
        return RobustCoTAgent(client_factory, prompt_builder, config=config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
