"""Documentation for Jumanji MultiCVRP environment.

MultiCVRP extends CVRP with multiple vehicles that must coordinate
to serve all customers efficiently.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_multi_cvrp_html(env_id: str = "MultiCVRP-v0") -> str:
    """Generate MultiCVRP HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Multi-Vehicle Capacitated Vehicle Routing Problem (MultiCVRP) extends
the classic CVRP to multiple vehicles. Each vehicle has its own capacity
and must coordinate with others to serve all customers efficiently.
</p>

<h4>Problem Description</h4>
<ul>
    <li>Multiple vehicles with individual capacities</li>
    <li>All customers must be served by exactly one vehicle</li>
    <li>Vehicles start and end at the depot</li>
    <li>Minimize total distance across all vehicles</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>customer_locations</strong>: Coordinates of customers</li>
    <li><strong>customer_demands</strong>: Demand at each customer</li>
    <li><strong>vehicle_capacities</strong>: Remaining capacity per vehicle</li>
    <li><strong>vehicle_positions</strong>: Current location of each vehicle</li>
    <li><strong>visited_mask</strong>: Customers already served</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete</code> - Select next customer for current vehicle</p>

<h4>Rewards</h4>
<ul>
    <li><strong>Negative distance</strong>: Travel cost</li>
    <li>Goal: Minimize total fleet distance</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Termination</strong>: All customers served, vehicles at depot</li>
</ul>

<h4>Challenges</h4>
<ul>
    <li>Workload balancing across vehicles</li>
    <li>Coordination to avoid redundant routes</li>
    <li>Capacity constraints per vehicle</li>
</ul>

<h4>Applications</h4>
<ul>
    <li>Fleet management</li>
    <li>Last-mile delivery</li>
    <li>Ride-sharing optimization</li>
    <li>Emergency response routing</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
</ul>
"""


MULTI_CVRP_HTML = get_multi_cvrp_html()

__all__ = ["get_multi_cvrp_html", "MULTI_CVRP_HTML"]
