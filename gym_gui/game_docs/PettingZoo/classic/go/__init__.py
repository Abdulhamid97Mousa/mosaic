"""Documentation for PettingZoo Go."""
from __future__ import annotations


def get_go_html(board_size: int = 19) -> str:
    """Generate Go HTML documentation for a specific board size.

    Args:
        board_size: Board size (9, 13, or 19). Default is 19.
    """
    size_desc = {
        9: "9×9 board for faster games and learning (81 positions)",
        13: "13×13 board for intermediate play (169 positions)",
        19: "19×19 standard board for full gameplay (361 positions)",
    }
    desc = size_desc.get(board_size, size_desc[19])
    action_space = board_size * board_size + 1  # positions + pass

    return (
        "<h3>PettingZoo: Go (go_v5)</h3>"
        "<p>The ancient game of Go, one of the most complex board games. Players place "
        "stones on intersections to surround territory and capture opponent stones. "
        "Uses the Minigo implementation for game logic.</p>"
        "<h4>Environment Details</h4>"
        "<ul>"
        "<li><strong>Import:</strong> <code>from pettingzoo.classic import go_v5</code></li>"
        "<li><strong>API Type:</strong> AEC (turn-based)</li>"
        "<li><strong>Parallel API:</strong> Yes</li>"
        "<li><strong>Agents:</strong> ['black_0', 'white_0']</li>"
        f"<li><strong>Action Shape:</strong> Discrete({action_space})</li>"
        f"<li><strong>Observation Shape:</strong> ({board_size}, {board_size}, 17)</li>"
        "<li><strong>Observation Values:</strong> [0, 1]</li>"
        "</ul>"
        "<h4>Board Sizes</h4>"
        "<ul>"
        "<li><strong>9×9:</strong> Fast games, good for beginners and testing (82 actions)</li>"
        "<li><strong>13×13:</strong> Intermediate, balanced complexity (170 actions)</li>"
        "<li><strong>19×19:</strong> Standard tournament size, full complexity (362 actions)</li>"
        "</ul>"
        "<h4>Observation Space</h4>"
        "<p>N×N×17 binary tensor (N = board size, AlphaGo-style):</p>"
        "<ul>"
        "<li><strong>Channels 0-7:</strong> Current player's stone positions (8 history frames)</li>"
        "<li><strong>Channels 8-15:</strong> Opponent's stone positions (8 history frames)</li>"
        "<li><strong>Channel 16:</strong> Color indicator (all 1s for black, all 0s for white)</li>"
        "</ul>"
        "<h4>Action Space</h4>"
        f"<p>Discrete({action_space}) - place a stone at intersection (row × N + col) or pass:</p>"
        "<ul>"
        f"<li><strong>Actions 0 to {board_size * board_size - 1}:</strong> Place stone at board position</li>"
        f"<li><strong>Action {board_size * board_size}:</strong> Pass (end turn without placing)</li>"
        "</ul>"
        "<p>Game ends after two consecutive passes, triggering scoring.</p>"
        "<h4>Legal Actions Mask</h4>"
        "<p>The <code>observation['action_mask']</code> indicates legal moves. Illegal moves include:</p>"
        "<ul>"
        "<li>Occupied intersections</li>"
        "<li>Suicide moves (placing in position with no liberties)</li>"
        "<li>Ko violations (recreating previous board state)</li>"
        "</ul>"
        "<h4>Game Rules</h4>"
        "<ul>"
        "<li><strong>Capture:</strong> Stones with no liberties (empty adjacent points) are removed</li>"
        "<li><strong>Ko Rule:</strong> Cannot recreate the previous board position</li>"
        "<li><strong>Komi:</strong> Points given to white for going second (default 7.5)</li>"
        "<li><strong>Scoring:</strong> Chinese rules - territory + captured stones</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<table>"
        "<tr><th>Winner</th><th>Loser</th><th>Draw</th></tr>"
        "<tr><td>+1</td><td>-1</td><td>0</td></tr>"
        "</table>"
        "<p>Rewards given at game end (after two passes and scoring).</p>"
        "<h4>Usage (AEC)</h4>"
        "<pre><code>from pettingzoo.classic import go_v5\n\n"
        "env = go_v5.env(board_size=19, komi=7.5, render_mode='human')\n"
        "env.reset(seed=42)\n\n"
        "for agent in env.agent_iter():\n"
        "    observation, reward, termination, truncation, info = env.last()\n\n"
        "    if termination or truncation:\n"
        "        action = None\n"
        "    else:\n"
        "        mask = observation['action_mask']\n"
        "        action = env.action_space(agent).sample(mask)\n\n"
        "    env.step(action)\n"
        "env.close()</code></pre>"
        "<h4>Configuration</h4>"
        "<ul>"
        "<li><code>board_size</code>: 9, 13, or 19 (default 19)</li>"
        "<li><code>komi</code>: Points given to white (default 7.5)</li>"
        "</ul>"
        "<h4>Human Control Mode</h4>"
        "<p>In the GUI, click on an intersection to place a stone. Black plays first. "
        "Use the Pass button to pass your turn. The game ends after two consecutive passes.</p>"
        "<h4>Version History</h4>"
        "<ul>"
        "<li><strong>v5:</strong> Current version (1.13.0)</li>"
        "<li><strong>v4:</strong> Fixed observation bugs and memory leaks (1.10.0)</li>"
        "<li><strong>v3:</strong> Added ko rule and proper scoring (1.6.0)</li>"
        "</ul>"
        "<p>Docs: <a href='https://pettingzoo.farama.org/environments/classic/go/'>PettingZoo Go</a></p>"
    )


# For backward compatibility
GO_HTML = get_go_html(19)

__all__ = ["GO_HTML", "get_go_html"]
