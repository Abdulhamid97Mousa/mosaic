"""MiniGrid Text Clean Language Wrapper.

This wrapper transforms MiniGrid observations into text-based observations
suitable for LLM agents. It converts the visual grid observation into
natural language descriptions.
"""

import gymnasium as gym
from PIL import Image

# MiniGrid action space - text commands that map to discrete actions
MINIGRID_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]


class MiniGridTextCleanLangWrapper(gym.Wrapper):
    """Wrapper that converts MiniGrid observations to text for LLM agents.

    This wrapper:
    1. Converts image observations to PIL images for optional VLM use
    2. Provides text descriptions via obs["text"]["long_term_context"]
    3. Maps text actions to discrete action indices
    4. Provides a default action for invalid LLM outputs
    """

    def __init__(self, env, vlm=False, **kwargs):
        """Initialize the wrapper.

        Args:
            env: The MiniGrid environment to wrap
            vlm: Whether to include image for vision-language models
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__(env)
        self.language_action_space = MINIGRID_ACTION_SPACE[:]
        self._mission = None
        self.progression = 0.0
        self._vlm = vlm

    @property
    def max_steps(self):
        """Maximum steps allowed in the environment."""
        return self.env.unwrapped.max_steps

    @property
    def default_action(self):
        """Default action when LLM returns invalid output."""
        return "go forward"

    def get_text_action(self, action):
        """Convert discrete action to text."""
        return self.language_action_space[action.value]

    def get_prompt(self, obs, infos):
        """Generate text prompt and optional image from observation.

        Args:
            obs: The raw observation dict
            infos: The info dict with 'descriptions' key

        Returns:
            Tuple of (prompt_text, pil_image)
        """
        # Get rendered image for VLM
        try:
            image = Image.fromarray(
                self.env.unwrapped.get_pov_render(tile_size=16)
            ).convert("RGB")
        except Exception:
            # Fallback if POV render not available
            image = None

        # Format descriptions into prompt
        descriptions = infos.get("descriptions", ["You see nothing special."])

        def _form_prompt(description_list):
            # Remove "You see " prefix for cleaner output
            return "\n".join([d.replace("You see ", "") for d in description_list])

        prompt = _form_prompt(descriptions)
        return prompt, image

    def reset(self, **kwargs):
        """Reset the environment and return text observation.

        Returns:
            Tuple of (observation_dict, info_dict)
        """
        obs, info = self.env.reset(**kwargs)
        prompt, image = self.get_prompt(obs, info)

        self._mission = obs.get("mission", "navigate the environment")
        self.progression = 0.0

        # Structure observation for LLM agent
        # long_term_context: spatial descriptions of visible objects
        # short_term_context: status updates (empty for MiniGrid)
        obs["text"] = {"long_term_context": prompt, "short_term_context": ""}
        obs["image"] = image

        return obs, info

    def step(self, action):
        """Execute action and return text observation.

        Args:
            action: Text action string (e.g., "go forward")

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)

        Raises:
            ValueError: If the action is not valid.
        """
        # Convert text action to discrete index (case-insensitive matching)
        action_lower = action.lower().strip() if isinstance(action, str) else ""
        try:
            action_int = self.language_action_space.index(action_lower)
        except ValueError:
            raise ValueError(
                f"Invalid action: '{action}'. "
                f"Valid actions are: {self.language_action_space}"
            )

        obs, reward, terminated, truncated, info = self.env.step(action_int)

        # Track progression (1.0 if goal reached)
        if reward > 0:
            self.progression = 1.0

        prompt, image = self.get_prompt(obs, info)
        obs["text"] = {"long_term_context": prompt, "short_term_context": ""}
        obs["image"] = image

        return obs, reward, terminated, truncated, info

    def get_stats(self):
        """Get environment statistics."""
        return {"mission": self._mission, "progression": self.progression}
