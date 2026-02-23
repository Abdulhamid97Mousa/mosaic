# Here we should have an environment manager function that can be used to instantiate
# environments with the correct wrappers.
import logging
from typing import Any, Optional

from gym import spaces

from llm_worker.environments.env_wrapper import EnvWrapper

_logger = logging.getLogger(__name__)


def make_env(env_name, task, config, render_mode=None):
    """Create an environment instance with the appropriate wrapper based on the environment name.

    Args:
        env_name (str): The name of the environment to create.
        task (str): The specific task within the environment.
        config (dict): Configuration settings for the environment.
        render_mode (str, optional): Rendering mode for the environment. Defaults to None.

    Returns:
        EnvWrapper: A wrapped environment instance.

    Raises:
        ValueError: If the environment name is not recognized.
    """
    if env_name == "nle":
        from llm_worker.environments.nle.nle_env import make_nle_env

        base_env = make_nle_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "minihack":
        from llm_worker.environments.minihack.minihack_env import make_minihack_env

        base_env = make_minihack_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "babyai":
        from llm_worker.environments.babyai_text.babyai_env import make_babyai_env

        base_env = make_babyai_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "minigrid":
        from llm_worker.environments.minigrid.minigrid_env import make_minigrid_env

        base_env = make_minigrid_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "crafter":
        from llm_worker.environments.crafter.crafter_env import make_crafter_env

        base_env = make_crafter_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "textworld":
        from llm_worker.environments.textworld.textworld_env import make_textworld_env

        base_env = make_textworld_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "babaisai":
        from llm_worker.environments.babaisai.babaisai_env import make_babaisai_env

        base_env = make_babaisai_env(env_name, task, config, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return EnvWrapper(base_env, env_name, task)


# =============================================================================
# MultiGrid helpers (MOSAIC extension) — used by runtime.py
# =============================================================================

def generate_multigrid_description(
    obs: Any,
    agent_id: int,
    env: Any,
    observation_mode: str = "visible_teammates",
    agent_direction: int = 0,
    carrying: Optional[str] = None,
) -> str:
    """Generate text description from MultiGrid observation.

    Delegates to llm_worker.environments.multigrid.observations.
    """
    from llm_worker.environments.multigrid import observations

    if observation_mode == "egocentric":
        return observations.describe_observation_egocentric(
            obs, agent_direction, carrying
        )
    else:  # visible_teammates
        agent_team = agent_id // 2  # Soccer: 0,1=team0, 2,3=team1
        visible_teammates = observations.extract_visible_teammates(
            env, agent_id, agent_team
        )
        return observations.describe_observation_with_teammates(
            obs, agent_id, visible_teammates, agent_direction, carrying
        )


def get_multigrid_instruction_prompt(
    agent_id: int,
    env_id: str,
    coordination_level: int = 1,
    role: Optional[str] = None,
) -> str:
    """Get MultiGrid instruction prompt for the given coordination level.

    Delegates to llm_worker.environments.multigrid.prompts.
    """
    from llm_worker.environments.multigrid import prompts

    team = agent_id // 2 if "Soccer" in env_id else agent_id

    if coordination_level == 1:
        return prompts.get_instruction_prompt_level1(agent_id, team, env_id)
    elif coordination_level == 2:
        return prompts.get_instruction_prompt_level2(agent_id, team, env_id)
    elif coordination_level == 3:
        return prompts.get_instruction_prompt_level3(
            agent_id, team, role or "forward", env_id
        )
    else:
        _logger.warning(f"Unknown coordination_level {coordination_level}, using Level 1")
        return prompts.get_instruction_prompt_level1(agent_id, team, env_id)


class Strings(spaces.Space):
    """A custom Gym space for managing discrete string-based actions."""

    def __init__(self, values, seed=None):
        super().__init__((len(values),), str, seed)
        self._dict = {value: i for i, value in enumerate(values)}
        self._values = values

    def sample(self):
        return self.np_random.choice(self._values)

    def map(self, action):
        return self._dict[action]

    def contains(self, value):
        return value in self._dict

    def __iter__(self):
        return self._values.__iter__()
