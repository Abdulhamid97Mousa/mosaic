"""Documentation for Jumanji MMST environment.

MMST (Multi-agent Minimum Spanning Tree) is a graph optimization
problem where agents must collaboratively build a minimum spanning tree.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_mmst_html(env_id: str = "MMST-v0") -> str:
    """Generate MMST HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
The Multi-agent Minimum Spanning Tree (MMST) problem involves
constructing a minimum spanning tree on a weighted graph. The
objective is to connect all nodes with minimum total edge weight.
</p>

<h4>Problem Description</h4>
<ul>
    <li>Given: A weighted undirected graph</li>
    <li>Goal: Select edges to form a spanning tree</li>
    <li>Constraint: Tree must connect all nodes</li>
    <li>Objective: Minimize total edge weight</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>nodes</strong>: Node positions/features</li>
    <li><strong>edges</strong>: Edge weights and connections</li>
    <li><strong>selected_edges</strong>: Edges in current tree</li>
    <li><strong>action_mask</strong>: Valid edges to add</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(num_edges)</code> - Select edge to add to tree</p>

<h4>Rewards</h4>
<ul>
    <li><strong>Negative edge weight</strong>: Cost of adding edge</li>
    <li><strong>Bonus</strong>: Completing a valid spanning tree</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Termination</strong>: Spanning tree complete (n-1 edges)</li>
</ul>

<h4>Classical Algorithms</h4>
<ul>
    <li><strong>Prim's Algorithm</strong>: Grow tree from single node</li>
    <li><strong>Kruskal's Algorithm</strong>: Sort edges, add if no cycle</li>
    <li><strong>Boruvka's Algorithm</strong>: Parallel edge selection</li>
</ul>

<h4>Applications</h4>
<ul>
    <li>Network design</li>
    <li>Circuit design</li>
    <li>Cluster analysis</li>
    <li>Image segmentation</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Minimum_spanning_tree" target="_blank">Minimum Spanning Tree (Wikipedia)</a></li>
</ul>
"""


MMST_HTML = get_mmst_html()

__all__ = ["get_mmst_html", "MMST_HTML"]
