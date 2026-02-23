from .history import HistoryPromptBuilder, Message

import warnings

def create_prompt_builder(config):
    """
    Creates an instance of a prompt builder based on the provided configuration.
    This function initializes a prompt builder by extracting relevant configuration
    parameters. It can be extended or modified to support different types of prompt
    builders beyond just the HistoryPromptBuilder.
    Args:
        config (Config): An object containing configuration settings, which must
            include the following keys:
            - max_text_history (int): Maximum number of text history entries to retain.
            - max_image_history (int): Maximum number of image history entries to retain.
            - max_cot_history (int): Maximum number of chain-of-thought history entries to retain.
    Returns:
        PromptBuilder: An instance of a prompt builder configured with the specified
            history limits and any additional parameters defined in the config.
    """

    max_history = config.get("max_history", None)
    if max_history is not None:
        warnings.warn("The 'max_history' parameter is deprecated. Please use 'max_text_history' instead.")
    
    max_text_history = max_history
    if max_text_history is None:
        max_text_history = config.max_text_history

    return HistoryPromptBuilder(
        max_text_history=max_text_history,
        max_image_history=config.max_image_history,
        max_cot_history=config.max_cot_history,
    )


def create_simple_prompt_builder(
    max_text_history=16,
    max_image_history=0,
    max_cot_history=1,
    system_prompt=None,
):
    """Create a simple prompt builder with default settings.

    Args:
        max_text_history: Maximum text history entries.
        max_image_history: Maximum image history entries.
        max_cot_history: Maximum chain-of-thought history entries.
        system_prompt: Optional system prompt.

    Returns:
        HistoryPromptBuilder instance.
    """
    return HistoryPromptBuilder(
        max_text_history=max_text_history,
        max_image_history=max_image_history,
        max_cot_history=max_cot_history,
        system_prompt=system_prompt,
    )
