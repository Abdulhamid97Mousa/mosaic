"""Documentation for Jumanji Knapsack environment.

The 0/1 Knapsack problem is a classic combinatorial optimization problem
where items must be selected to maximize value while respecting capacity.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_knapsack_html(env_id: str = "Knapsack-v1") -> str:
    """Generate Knapsack HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
The 0/1 Knapsack Problem is a classic NP-hard combinatorial optimization
problem. Given a set of items with weights and values, select which items
to include in a knapsack to maximize total value without exceeding capacity.
</p>

<h4>Problem Description</h4>
<ul>
    <li>Given: Items with weights w<sub>i</sub> and values v<sub>i</sub></li>
    <li>Given: Knapsack with capacity W</li>
    <li>Goal: Maximize total value of selected items</li>
    <li>Constraint: Total weight must not exceed W</li>
    <li>Binary: Each item is either taken (1) or left (0)</li>
</ul>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>weights</strong>: Weight of each item</li>
    <li><strong>values</strong>: Value of each item</li>
    <li><strong>remaining_capacity</strong>: Space left in knapsack</li>
    <li><strong>selected</strong>: Binary mask of selected items</li>
    <li><strong>action_mask</strong>: Items that still fit</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(num_items)</code> - Select item to add:</p>
<ul>
    <li>Each action adds an item to the knapsack</li>
    <li>Items that don't fit are masked as invalid</li>
</ul>

<h4>Rewards</h4>
<ul>
    <li><strong>Immediate</strong>: Value of selected item</li>
    <li>Goal: Maximize total collected value</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: No more items fit in remaining capacity</li>
    <li><strong>Termination</strong>: All items have been considered</li>
</ul>

<h4>Solution Approaches</h4>
<ul>
    <li><strong>Greedy (Value/Weight)</strong>: Select by value density</li>
    <li><strong>Dynamic Programming</strong>: O(nW) pseudo-polynomial</li>
    <li><strong>Branch and Bound</strong>: Exact solution with pruning</li>
    <li><strong>Reinforcement Learning</strong>: Learn selection policy</li>
</ul>

<h4>Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Variant</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0/1 Knapsack</td><td style="border: 1px solid #ddd; padding: 8px;">Take or leave each item</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Bounded</td><td style="border: 1px solid #ddd; padding: 8px;">Multiple copies of items available</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Unbounded</td><td style="border: 1px solid #ddd; padding: 8px;">Infinite copies available</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Multi-Knapsack</td><td style="border: 1px solid #ddd; padding: 8px;">Multiple knapsacks</td></tr>
</table>

<h4>Applications</h4>
<ul>
    <li>Resource allocation</li>
    <li>Portfolio optimization</li>
    <li>Cargo loading</li>
    <li>Budget allocation</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Knapsack_problem" target="_blank">Knapsack Problem (Wikipedia)</a></li>
</ul>
"""


KNAPSACK_HTML = get_knapsack_html()

__all__ = [
    "get_knapsack_html",
    "KNAPSACK_HTML",
]
