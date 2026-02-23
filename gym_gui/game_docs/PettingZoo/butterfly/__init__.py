"""Documentation for PettingZoo Butterfly environments."""
from __future__ import annotations


def get_butterfly_overview_html() -> str:
    """Generate Butterfly overview HTML documentation."""
    return (
        "<h3>PettingZoo: Butterfly Environments</h3>"
        "<p>Butterfly environments are challenging scenarios created by Farama using Pygame "
        "with visual Atari-style spaces. They require a high degree of coordination and "
        "emergent behavior learning.</p>"
        "<h4>Installation</h4>"
        "<pre><code>pip install 'pettingzoo[butterfly]'</code></pre>"
        "<h4>Available Environments</h4>"
        "<ul>"
        "<li><strong>Cooperative Pong:</strong> Two paddles keep a ball in play</li>"
        "<li><strong>Knights Archers Zombies:</strong> Defend against zombie waves</li>"
        "<li><strong>Pistonball:</strong> Coordinated pistons move a ball</li>"
        "</ul>"
        "<h4>Characteristics</h4>"
        "<ul>"
        "<li>Visual observations (RGB images)</li>"
        "<li>Highly cooperative - requires learning emergent behaviors</li>"
        "<li>Challenging to train optimal policies</li>"
        "</ul>"
    )


BUTTERFLY_OVERVIEW_HTML = get_butterfly_overview_html()

__all__ = ["BUTTERFLY_OVERVIEW_HTML", "get_butterfly_overview_html"]
