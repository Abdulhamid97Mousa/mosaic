"""Documentation for Overcooked Cramped Room layout."""
from __future__ import annotations


def get_cramped_room_html(env_id: str) -> str:
    """Generate Cramped Room HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>A compact kitchen where two chefs must navigate tight spaces while preparing and delivering onion soups. The primary challenge is <strong>collision avoidance</strong> in shared confined spaces.</p>"
        "<h4>Layout Characteristics</h4>"
        "<ul>"
        "<li><strong>Difficulty:</strong> ⭐⭐ (Basic)</li>"
        "<li><strong>Challenge:</strong> Low-level motion coordination</li>"
        "<li><strong>SP-XP Gap:</strong> Almost negligible (generalizes well to new partners)</li>"
        "<li><strong>Best for:</strong> Learning basic coordination, curriculum starting point</li>"
        "</ul>"
        "<h4>Coordination Requirements</h4>"
        "<ul>"
        "<li>Spatial awareness of partner's position</li>"
        "<li>Path planning to minimize collisions</li>"
        "<li>Implicit task division (gathering vs delivery)</li>"
        "<li>⚠ Tight walkways require careful navigation</li>"
        "</ul>"
        "<h4>Optimal Strategies</h4>"
        "<ol>"
        "<li><strong>Implicit Task Division:</strong> One agent gathers onions, other handles plating/delivery</li>"
        "<li><strong>Sequential Workflow:</strong> Agents take turns using shared walkways</li>"
        "<li><strong>Opportunistic Cooperation:</strong> Whoever is closest handles each task</li>"
        "</ol>"
        "<h4>Common Failure Modes</h4>"
        "<ul>"
        "<li>Head-on collisions in narrow passages</li>"
        "<li>Blocking doorways or critical paths</li>"
        "<li>Resource conflicts (both accessing same station)</li>"
        "</ul>"
        "<h4>Performance Benchmarks (400 timesteps)</h4>"
        "<ul>"
        "<li>Random: ~0-20 points (0-1 soup)</li>"
        "<li>Self-Play (PPO): ~200-220 points (10-11 soups)</li>"
        "<li>Human-Human: ~180-250 points (9-12 soups)</li>"
        "<li>Expert: ~260-300 points (13-15 soups)</li>"
        "</ul>"
        "<h4>Research Context</h4>"
        "<p>Cramped Room is the foundational layout from Carroll et al. (NeurIPS 2019). Even self-play agents collaborate effectively here, making it ideal for testing basic coordination mechanisms. Success indicates functional motion planning and collision avoidance.</p>"
        "<p><strong>Key Insight:</strong> Low SP-XP gap means agents trained here generalize well to new partners, unlike more complex layouts.</p>"
        "<p>See: <a href=\"https://github.com/HumanCompatibleAI/overcooked_ai\">Overcooked-AI Repository</a> | <a href=\"https://arxiv.org/abs/1910.05789\">Research Paper</a></p>"
    )


# For backward compatibility
OVERCOOKED_CRAMPED_ROOM_HTML = get_cramped_room_html("overcooked/cramped_room")

__all__ = ["OVERCOOKED_CRAMPED_ROOM_HTML", "get_cramped_room_html"]
