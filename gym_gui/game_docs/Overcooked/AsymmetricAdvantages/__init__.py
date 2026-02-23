"""Documentation for Overcooked Asymmetric Advantages layout."""
from __future__ import annotations


def get_asymmetric_advantages_html(env_id: str) -> str:
    """Generate Asymmetric Advantages HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>An asymmetric kitchen where agents have different access to resources. Success requires recognizing positional advantages and specializing in complementary roles.</p>"
        "<h4>Layout Characteristics</h4>"
        "<ul>"
        "<li><strong>Difficulty:</strong> ⭐⭐⭐ (Intermediate)</li>"
        "<li><strong>Challenge:</strong> Role specialization based on position</li>"
        "<li><strong>SP-XP Gap:</strong> Almost negligible (adaptable strategies)</li>"
        "<li><strong>Best for:</strong> Testing strategic role division</li>"
        "</ul>"
        "<h4>The Asymmetry</h4>"
        "<p>One agent is positioned near onion dispenser (fast gathering), while the other is near dishes/serving area (fast delivery). Optimal performance requires exploiting these positional strengths.</p>"
        "<h4>Coordination Requirements</h4>"
        "<ul>"
        "<li>Recognize positional advantages and disadvantages</li>"
        "<li>Choose complementary roles (gathering vs delivery)</li>"
        "<li>Avoid symmetric load balancing (inefficient)</li>"
        "<li>Adapt if partner chooses unexpected role</li>"
        "</ul>"
        "<h4>Optimal Strategy: Positional Specialization</h4>"
        "<ul>"
        "<li><strong>Agent A (near onions):</strong> Focus on gathering all 3 onions and placing in pot</li>"
        "<li><strong>Agent B (near dishes):</strong> Wait for cooking, plate soup, deliver to serving area</li>"
        "<li><strong>Result:</strong> Minimal travel distance, maximum efficiency</li>"
        "</ul>"
        "<h4>Common Failure Modes</h4>"
        "<ul>"
        "<li>Ignoring asymmetry (both agents do same tasks)</li>"
        "<li>Wrong specialization (agent far from onions tries to gather)</li>"
        "<li>No specialization at all (random task assignment)</li>"
        "</ul>"
        "<h4>Performance Benchmarks (400 timesteps)</h4>"
        "<ul>"
        "<li>Symmetric strategy: ~100-140 points (5-7 soups)</li>"
        "<li>Weak specialization: ~160-200 points (8-10 soups)</li>"
        "<li>Strong specialization: ~220-260 points (11-13 soups)</li>"
        "<li>Expert: ~280-320 points (14-16 soups)</li>"
        "</ul>"
        "<h4>Research Context</h4>"
        "<p>Tests whether agents can adapt strategies to environmental structure. Self-play agents successfully discover specialization, and the low SP-XP gap indicates strategies are robust across partners.</p>"
        "<p><strong>Key Insight:</strong> Unlike arbitrary conventions (Coordination Ring), there's one clear optimal strategy, making cross-play successful.</p>"
        "<p>See: <a href=\"https://github.com/HumanCompatibleAI/overcooked_ai\">Overcooked-AI Repository</a> | <a href=\"https://arxiv.org/abs/1910.05789\">Research Paper</a></p>"
    )


# For backward compatibility
OVERCOOKED_ASYMMETRIC_ADVANTAGES_HTML = get_asymmetric_advantages_html("overcooked/asymmetric_advantages")

__all__ = ["OVERCOOKED_ASYMMETRIC_ADVANTAGES_HTML", "get_asymmetric_advantages_html"]
