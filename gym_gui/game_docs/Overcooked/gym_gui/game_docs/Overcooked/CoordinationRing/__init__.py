"""Documentation for Overcooked Coordination Ring layout."""
from __future__ import annotations


def get_coordination_ring_html(env_id: str) -> str:
    """Generate Coordination Ring HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>A circular kitchen with <strong>two equally viable strategies</strong> (clockwise vs counter-clockwise movement). Partners must implicitly agree on the same convention to avoid collisions.</p>"
        "<h4>Layout Characteristics</h4>"
        "<ul>"
        "<li><strong>Difficulty:</strong> ⭐⭐⭐⭐ (Advanced)</li>"
        "<li><strong>Challenge:</strong> Protocol alignment without communication</li>"
        "<li><strong>SP-XP Gap:</strong> ⚠ SIGNIFICANT (self-play agents fail with new partners!)</li>"
        "<li><strong>Best for:</strong> Testing zero-shot coordination</li>"
        "</ul>"
        "<h4>The Coordination Dilemma</h4>"
        "<p>Agents must travel between bottom-left (onions) and top-right (dishes/serving) corners. There are two routes:</p>"
        "<ul>"
        "<li><strong>Clockwise:</strong> Bottom-left → Right → Top → Top-right</li>"
        "<li><strong>Counter-clockwise:</strong> Bottom-left → Up → Left → Top-right</li>"
        "</ul>"
        "<p>⚠ <strong>Problem:</strong> If agents use DIFFERENT routes, they collide head-on, creating gridlock!</p>"
        "<h4>Coordination Requirements</h4>"
        "<ul>"
        "<li>Implicitly agree on same traffic flow direction</li>"
        "<li>Detect partner's chosen convention early</li>"
        "<li>Adapt to partner's strategy (don't rigidly enforce your own)</li>"
        "<li>Maintain consistent convention throughout episode</li>"
        "</ul>"
        "<h4>Why Self-Play Fails</h4>"
        "<p>Self-play populations converge to ONE arbitrary convention (e.g., always clockwise). When paired with agents from different populations using the opposite convention, complete coordination failure occurs.</p>"
        "<h4>Common Failure Modes</h4>"
        "<ul>"
        "<li>⚠⚠⚠ <strong>Different conventions:</strong> Agent trained clockwise meets counter-clockwise partner → constant collisions, near-zero performance</li>"
        "<li>Rigid commitment: Agent refuses to adapt to partner's chosen route</li>"
        "<li>Oscillation: Agents keep switching routes, never establishing stable protocol</li>"
        "</ul>"
        "<h4>Performance Benchmarks (400 timesteps)</h4>"
        "<ul>"
        "<li>Self-play (same population): ~200-220 points (10-11 soups)</li>"
        "<li>⚠ Self-play (cross-play): ~20-80 points (1-4 soups) - THE SP-XP GAP!</li>"
        "<li>Adaptive cross-play agent: ~180-220 points (9-11 soups)</li>"
        "<li>Human-human: ~200-240 points (10-12 soups)</li>"
        "</ul>"
        "<h4>Research Context</h4>"
        "<p>From Carroll et al. (NeurIPS 2019), Coordination Ring is <strong>THE defining benchmark for zero-shot coordination</strong> because it exposes the arbitrariness problem in self-play training.</p>"
        "<p><strong>Key Finding:</strong> \"Standard RL methods create agents that coordinate with themselves, not with humans. The arbitrary convention problem reveals this brittleness.\"</p>"
        "<p><strong>Real-World Analogy:</strong> Like driving on left vs right side of the road - both work perfectly, but only if everyone agrees.</p>"
        "<p>See: <a href=\"https://bair.berkeley.edu/blog/2019/10/21/coordination/\">BAIR Blog</a> | <a href=\"https://arxiv.org/abs/1910.05789\">Research Paper</a></p>"
    )


# For backward compatibility
OVERCOOKED_COORDINATION_RING_HTML = get_coordination_ring_html("overcooked/coordination_ring")

__all__ = ["OVERCOOKED_COORDINATION_RING_HTML", "get_coordination_ring_html"]
