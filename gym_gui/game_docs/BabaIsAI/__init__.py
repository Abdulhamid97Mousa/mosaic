"""Documentation for BabaIsAI puzzle environment.

BabaIsAI is an AI-friendly version of "Baba Is You" puzzle game where agents
manipulate rules by pushing word blocks. Excellent for testing LLM reasoning
and compositional generalization.

Paper: Cloos et al. (2024). "Baba Is AI: Break the Rules to Beat the Benchmark"
       ICML 2024 Workshop on LLMs and Cognition
Repository: https://github.com/nacloos/baba-is-ai
"""

from __future__ import annotations


def get_babaisai_overview_html() -> str:
    """Generate BabaIsAI overview HTML documentation."""
    return (
        "<h3>BabaIsAI: Rule Manipulation Puzzle Environment</h3>"
        "<p>An AI-friendly version of the puzzle game <strong>Baba Is You</strong> "
        "where agents manipulate rules by pushing word blocks. The game challenges "
        "agents to understand rule semantics, plan rule manipulation sequences, and "
        "demonstrate compositional generalization.</p>"
        "<h4>Installation</h4>"
        "<pre><code>pip install git+https://github.com/nacloos/baba-is-ai</code></pre>"
        "<h4>Game Mechanics</h4>"
        "<p>Rules are active when word blocks are aligned horizontally in the form "
        "<code>&lt;object&gt; IS &lt;property&gt;</code>:</p>"
        "<ul>"
        "<li><code>[BABA] [IS] [YOU]</code> → You control Baba</li>"
        "<li><code>[FLAG] [IS] [WIN]</code> → Touch flag to win</li>"
        "<li><code>[WALL] [IS] [STOP]</code> → Walls block movement</li>"
        "</ul>"
        "<p>You can <strong>break and create rules</strong> by pushing blocks:</p>"
        "<ul>"
        "<li>Push <code>[STOP]</code> away from <code>[WALL] [IS]</code> → walls no longer block</li>"
        "<li>Push <code>[WIN]</code> next to <code>[KEY] [IS]</code> → key becomes win condition</li>"
        "</ul>"
        "<h4>Objects & Properties</h4>"
        "<table border='1' cellpadding='5'>"
        "<tr><th>Objects</th><th>Properties</th></tr>"
        "<tr><td>baba (white triangle), wall, key, door, ball</td>"
        "<td>you, win, stop, push, defeat</td></tr>"
        "</table>"
        "<h4>Actions</h4>"
        "<p>Discrete action space: <code>[UP, DOWN, LEFT, RIGHT, IDLE]</code></p>"
        "<h4>Why It's Interesting for LLMs</h4>"
        "<ul>"
        "<li><strong>Rule Semantics:</strong> LLMs must understand what rules mean</li>"
        "<li><strong>Planning:</strong> Multi-step rule manipulation sequences</li>"
        "<li><strong>Compositional Generalization:</strong> Same objects, different solutions</li>"
        "<li><strong>Benchmark Result:</strong> LLMs struggle significantly on this task</li>"
        "</ul>"
        "<h4>Research Paper</h4>"
        "<p><em>Baba Is AI: Break the Rules to Beat the Benchmark</em><br>"
        "Cloos, Jens, Naim, Kuo, Cases, Barbu, Cueva (2024)<br>"
        "ICML 2024 Workshop on LLMs and Cognition</p>"
    )


def get_babaisai_env_html(env_id: str = "two_room-break_stop-make_win") -> str:
    """Generate specific BabaIsAI environment HTML documentation."""
    return (
        f"<h3>BabaIsAI: {env_id}</h3>"
        "<h4>Environment Details</h4>"
        "<ul>"
        "<li><strong>Import:</strong> <code>import baba; env = baba.make(f'env/{env_id}')</code></li>"
        "<li><strong>Actions:</strong> Discrete (5 actions: UP, DOWN, LEFT, RIGHT, IDLE)</li>"
        "<li><strong>Observation:</strong> Text description of object positions + RGB image</li>"
        "<li><strong>API Type:</strong> Old Gym API (obs, reward, done, info)</li>"
        "</ul>"
        "<h4>Text Observation Format</h4>"
        "<pre><code>Active rules:\n"
        "baba is you\n"
        "flag is win\n"
        "wall is stop\n\n"
        "Objects on the map:\n"
        "key 2 steps to the right and 1 step up\n"
        "door 3 steps down\n"
        "rule `flag` 1 step to the left</code></pre>"
        "<h4>Usage</h4>"
        "<pre><code>import baba\n\n"
        "# List all available environments\n"
        "print(baba.make('env/*').keys())\n\n"
        "# Create environment with render_mode\n"
        f"env = baba.make('env/{env_id}', render_mode='rgb_array')\n"
        "obs = env.reset()\n\n"
        "# Step with action\n"
        "obs, reward, done, info = env.step(action)\n\n"
        "# Render (no mode argument needed)\n"
        "img = env.render()</code></pre>"
        "<h4>Example Environments</h4>"
        "<ul>"
        "<li><code>two_room-break_stop-make_win</code> - Break STOP rule, create WIN rule</li>"
        "<li><code>one_room-open_door</code> - Open door with key</li>"
        "<li><code>one_room-push_to_win</code> - Push object to win</li>"
        "</ul>"
    )


BABAISAI_OVERVIEW_HTML = get_babaisai_overview_html()
BABAISAI_ENV_HTML = get_babaisai_env_html()

__all__ = [
    "BABAISAI_OVERVIEW_HTML",
    "BABAISAI_ENV_HTML",
    "get_babaisai_overview_html",
    "get_babaisai_env_html",
]
