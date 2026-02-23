"""Documentation for Jumanji FlatPack environment.

FlatPack is a 2D bin packing problem where rectangular items must be
placed on a surface without overlap, optimizing space utilization.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_flatpack_html(env_id: str = "FlatPack-v0") -> str:
    """Generate FlatPack HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
FlatPack is a 2D rectangular bin packing problem. The agent must place
rectangular items onto a 2D surface, minimizing wasted space while
avoiding overlaps.
</p>

<h4>Problem Description</h4>
<ul>
    <li>Given: Rectangular items with width and height</li>
    <li>Given: A 2D surface with fixed dimensions</li>
    <li>Goal: Place all items with minimum wasted space</li>
    <li>Constraint: Items cannot overlap</li>
</ul>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>grid</strong>: 2D occupancy grid of the surface</li>
    <li><strong>current_item</strong>: Dimensions of item to place</li>
    <li><strong>remaining_items</strong>: Queue of upcoming items</li>
    <li><strong>action_mask</strong>: Valid placement positions</li>
</ul>

<h4>Action Space</h4>
<p><code>MultiDiscrete</code> - Select position and rotation:</p>
<ul>
    <li><strong>x, y</strong>: Grid position for item placement</li>
    <li><strong>rotation</strong>: 0 or 90 degree rotation</li>
</ul>

<h4>Rewards</h4>
<ul>
    <li><strong>Positive</strong>: For successfully placing an item</li>
    <li><strong>Bonus</strong>: For efficient packing (less wasted space)</li>
    <li><strong>Negative</strong>: For invalid placements (masked)</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination (Success)</strong>: All items placed</li>
    <li><strong>Termination (Failure)</strong>: Cannot fit remaining items</li>
</ul>

<h4>Strategies</h4>
<ul>
    <li><strong>Bottom-Left</strong>: Place as far bottom-left as possible</li>
    <li><strong>Skyline</strong>: Track surface profile, place in valleys</li>
    <li><strong>Guillotine</strong>: Recursive partitioning</li>
    <li><strong>MaxRects</strong>: Track maximal free rectangles</li>
</ul>

<h4>Applications</h4>
<ul>
    <li>Sheet metal cutting</li>
    <li>Textile cutting</li>
    <li>Circuit board layout</li>
    <li>Warehouse floor planning</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Rectangle_packing" target="_blank">Rectangle Packing (Wikipedia)</a></li>
</ul>
"""


FLATPACK_HTML = get_flatpack_html()

__all__ = [
    "get_flatpack_html",
    "FLATPACK_HTML",
]
