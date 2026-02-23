"""Documentation for Jumanji GraphColoring environment.

Graph Coloring is a combinatorial optimization problem where the goal is
to assign colors to graph nodes such that no adjacent nodes share the same color.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_graph_coloring_html(env_id: str = "GraphColoring-v1") -> str:
    """Generate GraphColoring HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Graph Coloring is a classic combinatorial optimization problem. Given a graph,
the goal is to assign colors to nodes such that no two adjacent nodes (connected
by an edge) share the same color, while minimizing the number of colors used.
</p>

<h4>Problem Definition</h4>
<ul>
    <li><strong>Input</strong>: An undirected graph G = (V, E)</li>
    <li><strong>Goal</strong>: Assign colors to each vertex</li>
    <li><strong>Constraint</strong>: Adjacent vertices must have different colors</li>
    <li><strong>Objective</strong>: Minimize the number of colors used (chromatic number)</li>
</ul>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>adjacency</strong>: <code>Box(shape=(N, N), dtype=bool)</code> - Graph adjacency matrix</li>
    <li><strong>colors</strong>: <code>Box(shape=(N,), dtype=int32)</code> - Current color assignments (-1 = uncolored)</li>
    <li><strong>num_colors</strong>: <code>int</code> - Number of available colors</li>
    <li><strong>action_mask</strong>: <code>Box(shape=(N*C,), dtype=bool)</code> - Valid actions</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(N * C)</code> - Assign a color to a node:</p>
<ul>
    <li>Action = node_id * num_colors + color_id</li>
    <li>Each action assigns a specific color to a specific node</li>
</ul>

<h4>Applications</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Domain</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Application</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scheduling</td><td style="border: 1px solid #ddd; padding: 8px;">Exam scheduling (no conflicts)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Register Allocation</td><td style="border: 1px solid #ddd; padding: 8px;">Compiler optimization</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Frequency Assignment</td><td style="border: 1px solid #ddd; padding: 8px;">Radio/cellular networks</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Map Coloring</td><td style="border: 1px solid #ddd; padding: 8px;">Adjacent regions different colors</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Sudoku</td><td style="border: 1px solid #ddd; padding: 8px;">Can be modeled as graph coloring</td></tr>
</table>

<h4>Rewards</h4>
<ul>
    <li><strong>-1</strong>: Creating a conflict (coloring adjacent nodes same color)</li>
    <li><strong>+1</strong>: Successfully coloring a node without conflicts</li>
    <li><strong>Bonus</strong>: Using fewer colors than the baseline</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination (Success)</strong>: All nodes colored without conflicts</li>
    <li><strong>Termination (Failure)</strong>: Conflict created with no valid resolution</li>
    <li><strong>Truncation</strong>: Maximum steps reached</li>
</ul>

<h4>Graph Types</h4>
<ul>
    <li><strong>Random Graphs</strong>: Erdos-Renyi random graphs</li>
    <li><strong>Planar Graphs</strong>: Can be colored with 4 colors (Four Color Theorem)</li>
    <li><strong>Complete Graphs</strong>: Need N colors for N nodes</li>
    <li><strong>Bipartite Graphs</strong>: Always 2-colorable</li>
</ul>

<h4>Algorithms</h4>
<ul>
    <li><strong>Greedy</strong>: Color nodes in order, use first available color</li>
    <li><strong>DSatur</strong>: Color node with highest saturation degree first</li>
    <li><strong>Backtracking</strong>: Try colors, backtrack on conflicts</li>
</ul>

<h4>Complexity</h4>
<p>
Graph coloring is <strong>NP-complete</strong>. Determining if a graph can be
colored with k colors (k >= 3) is NP-complete. Finding the chromatic number
is NP-hard.
</p>

<h4>Controls</h4>
<p>
Graph Coloring uses a complex action space (node + color selection) that is
best suited for mouse/touch interaction with a color palette.
</p>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Graph_coloring" target="_blank">Wikipedia: Graph Coloring</a></li>
</ul>
"""


GRAPH_COLORING_HTML = get_graph_coloring_html()

__all__ = [
    "get_graph_coloring_html",
    "GRAPH_COLORING_HTML",
]
