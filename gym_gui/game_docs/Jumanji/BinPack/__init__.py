"""Documentation for Jumanji BinPack environment.

BinPack is a classic combinatorial optimization problem where items of
varying sizes must be packed into a minimum number of fixed-capacity bins.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_binpack_html(env_id: str = "BinPack-v2") -> str:
    """Generate BinPack HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
The Bin Packing Problem is a classic NP-hard combinatorial optimization problem.
The objective is to pack a set of items with varying sizes into the minimum
number of fixed-capacity bins.
</p>

<h4>Problem Description</h4>
<ul>
    <li>Given: A set of items with known sizes</li>
    <li>Given: Bins with fixed capacity</li>
    <li>Goal: Pack all items using minimum number of bins</li>
    <li>Constraint: No bin can exceed its capacity</li>
</ul>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>items</strong>: Item sizes to be packed</li>
    <li><strong>bins</strong>: Current bin capacities/utilization</li>
    <li><strong>action_mask</strong>: Valid placement actions</li>
</ul>

<h4>Action Space</h4>
<p><code>MultiDiscrete([obs_num_ems, max_num_items])</code> - Two-part placement decision:</p>
<ul>
    <li><strong>Dimension 0</strong>: EMS (Empty Maximal Space) index — which bin slot to use (0 to obs_num_ems-1)</li>
    <li><strong>Dimension 1</strong>: Item index — which item to place (0 to max_num_items-1)</li>
    <li>Invalid placements (overflow or occupied) are masked via <code>action_mask</code></li>
</ul>

<h4>Rewards</h4>
<ul>
    <li><strong>Negative reward</strong>: For each new bin opened</li>
    <li><strong>Zero</strong>: For placing item in existing bin with space</li>
    <li>Goal: Minimize total bins used (maximize reward)</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: All items have been packed</li>
</ul>

<h4>Strategies</h4>
<ul>
    <li><strong>First Fit</strong>: Place in first bin with space</li>
    <li><strong>Best Fit</strong>: Place in bin that minimizes remaining space</li>
    <li><strong>First Fit Decreasing</strong>: Sort by size, then first fit</li>
    <li><strong>Learned Heuristics</strong>: RL can discover better policies</li>
</ul>

<h4>Applications</h4>
<ul>
    <li>Container loading and logistics</li>
    <li>Cloud VM resource allocation</li>
    <li>Memory management</li>
    <li>Cutting stock problems</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Bin_packing_problem" target="_blank">Bin Packing Problem (Wikipedia)</a></li>
</ul>
"""


BINPACK_HTML = get_binpack_html()

__all__ = [
    "get_binpack_html",
    "BINPACK_HTML",
]
