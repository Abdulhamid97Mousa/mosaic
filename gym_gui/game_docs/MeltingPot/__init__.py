"""Melting Pot multi-agent environment documentation.

Melting Pot is a suite of test scenarios for multi-agent reinforcement learning
developed by Google DeepMind. It assesses generalization to novel social situations
involving both familiar and unfamiliar individuals.

Repository: https://github.com/google-deepmind/meltingpot
Paper: https://arxiv.org/abs/2211.13746
Shimmy: https://shimmy.farama.org/environments/meltingpot/

Key Features:
- 50+ multi-agent substrates (game scenarios)
- 256+ unique test scenarios
- Up to 16 simultaneous agents
- Social interactions: cooperation, competition, deception, trust, reciprocation
- Parallel stepping paradigm (all agents act simultaneously)
- Compatible with PettingZoo API through Shimmy wrapper

NOTE: Linux/macOS only (Windows NOT supported)
"""

from __future__ import annotations

from gym_gui.game_docs.MeltingPot.CollaborativeCooking import COLLABORATIVE_COOKING_HTML
from gym_gui.game_docs.MeltingPot.CleanUp import CLEAN_UP_HTML
from gym_gui.game_docs.MeltingPot.CommonsHarvest import COMMONS_HARVEST_HTML
from gym_gui.game_docs.MeltingPot.Territory import TERRITORY_HTML
from gym_gui.game_docs.MeltingPot.KingOfTheHill import KING_OF_THE_HILL_HTML
from gym_gui.game_docs.MeltingPot.PrisonersDilemma import PRISONERS_DILEMMA_HTML
from gym_gui.game_docs.MeltingPot.StagHunt import STAG_HUNT_HTML
from gym_gui.game_docs.MeltingPot.AllelopathicHarvest import ALLELOPATHIC_HARVEST_HTML


# Export individual substrate documentation
MELTINGPOT_COLLABORATIVE_COOKING_HTML = COLLABORATIVE_COOKING_HTML
MELTINGPOT_CLEAN_UP_HTML = CLEAN_UP_HTML
MELTINGPOT_COMMONS_HARVEST_HTML = COMMONS_HARVEST_HTML
MELTINGPOT_TERRITORY_HTML = TERRITORY_HTML
MELTINGPOT_KING_OF_THE_HILL_HTML = KING_OF_THE_HILL_HTML
MELTINGPOT_PRISONERS_DILEMMA_HTML = PRISONERS_DILEMMA_HTML
MELTINGPOT_STAG_HUNT_HTML = STAG_HUNT_HTML
MELTINGPOT_ALLELOPATHIC_HARVEST_HTML = ALLELOPATHIC_HARVEST_HTML


__all__ = [
    "MELTINGPOT_COLLABORATIVE_COOKING_HTML",
    "MELTINGPOT_CLEAN_UP_HTML",
    "MELTINGPOT_COMMONS_HARVEST_HTML",
    "MELTINGPOT_TERRITORY_HTML",
    "MELTINGPOT_KING_OF_THE_HILL_HTML",
    "MELTINGPOT_PRISONERS_DILEMMA_HTML",
    "MELTINGPOT_STAG_HUNT_HTML",
    "MELTINGPOT_ALLELOPATHIC_HARVEST_HTML",
]
