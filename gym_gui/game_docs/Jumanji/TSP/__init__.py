"""Documentation for Jumanji TSP environment.

TSP (Traveling Salesman Problem) is a classic optimization problem
where the goal is to find the shortest route visiting all cities.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_tsp_html(env_id: str = "TSP-v1") -> str:
    """Generate TSP HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
The Traveling Salesman Problem (TSP) is one of the most famous NP-hard
optimization problems. Given a list of cities and distances between them,
find the shortest possible route that visits each city exactly once
and returns to the starting city.
</p>

<h4>Problem Description</h4>
<ul>
    <li>Visit all cities exactly once</li>
    <li>Return to the starting city</li>
    <li>Minimize total travel distance</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>city_positions</strong>: Coordinates of all cities</li>
    <li><strong>current_city</strong>: Salesman's current location</li>
    <li><strong>visited_mask</strong>: Cities already visited</li>
    <li><strong>first_city</strong>: Starting city (for return)</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(num_cities)</code> - Select next city to visit</p>

<h4>Rewards</h4>
<ul>
    <li><strong>Negative distance</strong>: Cost of travel to next city</li>
    <li>Goal: Minimize total tour length</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Termination</strong>: All cities visited and returned to start</li>
</ul>

<h4>Classical Heuristics</h4>
<ul>
    <li><strong>Nearest Neighbor</strong>: Always visit closest unvisited city</li>
    <li><strong>Christofides</strong>: 1.5-approximation algorithm</li>
    <li><strong>2-Opt</strong>: Improve tour by swapping edges</li>
    <li><strong>Lin-Kernighan</strong>: Advanced local search</li>
</ul>

<h4>Modern Approaches</h4>
<ul>
    <li><strong>Attention Models</strong>: Transformer-based neural networks</li>
    <li><strong>Graph Neural Networks</strong>: Learn city relationships</li>
    <li><strong>Reinforcement Learning</strong>: Learn construction policy</li>
</ul>

<h4>Problem Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Variant</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Symmetric TSP</td><td style="border: 1px solid #ddd; padding: 8px;">Distance A→B equals B→A</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Asymmetric TSP</td><td style="border: 1px solid #ddd; padding: 8px;">Directed distances</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">TSP with Time Windows</td><td style="border: 1px solid #ddd; padding: 8px;">Visit during time slots</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Prize-Collecting TSP</td><td style="border: 1px solid #ddd; padding: 8px;">Optional city visits</td></tr>
</table>

<h4>Applications</h4>
<ul>
    <li>Route planning and logistics</li>
    <li>Circuit board drilling</li>
    <li>DNA sequencing</li>
    <li>Telescope positioning</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Travelling_salesman_problem" target="_blank">TSP (Wikipedia)</a></li>
</ul>
"""


TSP_HTML = get_tsp_html()

__all__ = ["get_tsp_html", "TSP_HTML"]
