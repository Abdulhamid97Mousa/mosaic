"""Game documentation for Griddly single-agent environments."""

from __future__ import annotations

import importlib

from gym_gui.game_docs.Griddly.single_agent.Bait import GRIDDLY_BAIT_HTML
from gym_gui.game_docs.Griddly.single_agent.Bait_With_Keys import GRIDDLY_BAIT_WITH_KEYS_HTML
from gym_gui.game_docs.Griddly.single_agent.Butterflies_and_Spiders import GRIDDLY_BUTTERFLIES_AND_SPIDERS_HTML
from gym_gui.game_docs.Griddly.single_agent.Clusters import GRIDDLY_CLUSTERS_HTML
from gym_gui.game_docs.Griddly.single_agent.Cook_Me_Pasta import GRIDDLY_COOK_ME_PASTA_HTML
from gym_gui.game_docs.Griddly.single_agent.Doggo import GRIDDLY_DOGGO_HTML
from gym_gui.game_docs.Griddly.single_agent.Drunk_Dwarf import GRIDDLY_DRUNK_DWARF_HTML
from gym_gui.game_docs.Griddly.single_agent.Eyeball import GRIDDLY_EYEBALL_HTML
from gym_gui.game_docs.Griddly.single_agent.Labyrinth import GRIDDLY_LABYRINTH_HTML
from gym_gui.game_docs.Griddly.single_agent.Partially_Observable_Bait import GRIDDLY_PO_BAIT_HTML
from gym_gui.game_docs.Griddly.single_agent.Partially_Observable_Clusters import GRIDDLY_PO_CLUSTERS_HTML
from gym_gui.game_docs.Griddly.single_agent.Partially_Observable_Cook_Me_Pasta import GRIDDLY_PO_COOK_ME_PASTA_HTML
from gym_gui.game_docs.Griddly.single_agent.Partially_Observable_Labyrinth import GRIDDLY_PO_LABYRINTH_HTML
from gym_gui.game_docs.Griddly.single_agent.Partially_Observable_Zelda import GRIDDLY_PO_ZELDA_HTML
from gym_gui.game_docs.Griddly.single_agent.Partially_Observable_Zen_Puzzle import GRIDDLY_PO_ZEN_PUZZLE_HTML
from gym_gui.game_docs.Griddly.single_agent.Random_butterflies import GRIDDLY_RANDOM_BUTTERFLIES_HTML
from gym_gui.game_docs.Griddly.single_agent.Sokoban import GRIDDLY_SOKOBAN_HTML
from gym_gui.game_docs.Griddly.single_agent.Spider_Nest import GRIDDLY_SPIDER_NEST_HTML
from gym_gui.game_docs.Griddly.single_agent.Spiders import GRIDDLY_SPIDERS_HTML
from gym_gui.game_docs.Griddly.single_agent.Zelda import GRIDDLY_ZELDA_HTML
from gym_gui.game_docs.Griddly.single_agent.Zelda_Sequential import GRIDDLY_ZELDA_SEQUENTIAL_HTML
from gym_gui.game_docs.Griddly.single_agent.Zen_Puzzle import GRIDDLY_ZEN_PUZZLE_HTML

# Directory names with dashes are not valid Python identifiers; use importlib.
_sokoban_2 = importlib.import_module("gym_gui.game_docs.Griddly.single_agent.Sokoban_-_2")
GRIDDLY_SOKOBAN_2_HTML: str = _sokoban_2.GRIDDLY_SOKOBAN_2_HTML

_po_sokoban_2 = importlib.import_module(
    "gym_gui.game_docs.Griddly.single_agent.Partially_Observable_Sokoban_-_2"
)
GRIDDLY_PO_SOKOBAN_2_HTML: str = _po_sokoban_2.GRIDDLY_PO_SOKOBAN_2_HTML

__all__ = [
    "GRIDDLY_ZELDA_HTML",
    "GRIDDLY_ZELDA_SEQUENTIAL_HTML",
    "GRIDDLY_PO_ZELDA_HTML",
    "GRIDDLY_SOKOBAN_HTML",
    "GRIDDLY_SOKOBAN_2_HTML",
    "GRIDDLY_PO_SOKOBAN_2_HTML",
    "GRIDDLY_CLUSTERS_HTML",
    "GRIDDLY_PO_CLUSTERS_HTML",
    "GRIDDLY_BAIT_HTML",
    "GRIDDLY_BAIT_WITH_KEYS_HTML",
    "GRIDDLY_PO_BAIT_HTML",
    "GRIDDLY_ZEN_PUZZLE_HTML",
    "GRIDDLY_PO_ZEN_PUZZLE_HTML",
    "GRIDDLY_LABYRINTH_HTML",
    "GRIDDLY_PO_LABYRINTH_HTML",
    "GRIDDLY_COOK_ME_PASTA_HTML",
    "GRIDDLY_PO_COOK_ME_PASTA_HTML",
    "GRIDDLY_SPIDERS_HTML",
    "GRIDDLY_SPIDER_NEST_HTML",
    "GRIDDLY_BUTTERFLIES_AND_SPIDERS_HTML",
    "GRIDDLY_RANDOM_BUTTERFLIES_HTML",
    "GRIDDLY_EYEBALL_HTML",
    "GRIDDLY_DRUNK_DWARF_HTML",
    "GRIDDLY_DOGGO_HTML",
]
