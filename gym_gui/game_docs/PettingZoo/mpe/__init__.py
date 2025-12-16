"""Documentation for PettingZoo MPE (Multi-Particle Environments)."""
from __future__ import annotations


def get_mpe_overview_html() -> str:
    """Generate MPE overview HTML documentation."""
    return (
        "<h3>PettingZoo: Multi-Particle Environments (MPE)</h3>"
        "<p>MPE is a set of communication-oriented environments where particle agents can move, "
        "communicate, see each other, push each other, and interact with fixed landmarks.</p>"
        "<p><strong>Note:</strong> MPE has been moved to the MPE2 package and will be removed "
        "from PettingZoo in a future release.</p>"
        "<h4>Installation</h4>"
        "<pre><code>pip install 'pettingzoo[mpe]'</code></pre>"
        "<h4>Environment Types</h4>"
        "<ul>"
        "<li><strong>Adversarial:</strong> Simple Adversary, Simple Crypto, Simple Push, "
        "Simple Tag, Simple World Comm (red vs green agents)</li>"
        "<li><strong>Cooperative:</strong> Simple Reference, Simple Speaker Listener, "
        "Simple Spread (agents work together)</li>"
        "</ul>"
        "<h4>Key Concepts</h4>"
        "<ul>"
        "<li><strong>Landmarks:</strong> Static circular features affecting rewards based on agent distance</li>"
        "<li><strong>Visibility:</strong> Agents observe relative positions of visible entities</li>"
        "<li><strong>Communication:</strong> Some agents can broadcast messages to others</li>"
        "<li><strong>Distances:</strong> Agents/landmarks spawn randomly from -1 to 1 on the map</li>"
        "</ul>"
        "<h4>Action Space</h4>"
        "<p><strong>Discrete (default):</strong> 5 actions - [no_action, move_left, move_right, "
        "move_down, move_up] plus optional communication</p>"
        "<p><strong>Continuous:</strong> Velocity vector [0.0, 1.0] in each cardinal direction</p>"
        "<h4>Termination</h4>"
        "<p>Games terminate after <code>max_cycles</code> steps (default: 25).</p>"
    )


MPE_OVERVIEW_HTML = get_mpe_overview_html()

__all__ = ["MPE_OVERVIEW_HTML", "get_mpe_overview_html"]
