"""Documentation for PettingZoo SISL environments."""
from __future__ import annotations


def get_sisl_overview_html() -> str:
    """Generate SISL overview HTML documentation."""
    return (
        "<h3>PettingZoo: SISL Environments</h3>"
        "<p>SISL (Stanford Intelligent Systems Laboratory) environments are cooperative "
        "multi-agent benchmark environments originally released as part of "
        "'Cooperative multi-agent control using deep reinforcement learning.'</p>"
        "<h4>Installation</h4>"
        "<pre><code>pip install 'pettingzoo[sisl]'</code></pre>"
        "<h4>Available Environments</h4>"
        "<ul>"
        "<li><strong>Multiwalker:</strong> Bipedal robots carry a package together</li>"
        "<li><strong>Pursuit:</strong> Pursuers cooperate to catch evaders</li>"
        "<li><strong>Waterworld:</strong> Agents gather food while avoiding poison</li>"
        "</ul>"
        "<h4>Citation</h4>"
        "<pre><code>@inproceedings{gupta2017cooperative,\n"
        "  title={Cooperative multi-agent control using deep reinforcement learning},\n"
        "  author={Gupta, Jayesh K and Egorov, Maxim and Kochenderfer, Mykel},\n"
        "  booktitle={AAMAS},\n"
        "  year={2017}\n"
        "}</code></pre>"
    )


SISL_OVERVIEW_HTML = get_sisl_overview_html()

__all__ = ["SISL_OVERVIEW_HTML", "get_sisl_overview_html"]
