"""Documentation for Jumanji CVRP environment.

CVRP (Capacitated Vehicle Routing Problem) is a classic optimization
problem where a vehicle must visit all customers while respecting capacity.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_cvrp_html(env_id: str = "CVRP-v1") -> str:
    """Generate CVRP HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
The Capacitated Vehicle Routing Problem (CVRP) is a classic NP-hard
optimization problem. A vehicle with limited capacity must visit all
customers, each with a demand, starting and ending at a depot, while
minimizing total travel distance.
</p>

<h4>Problem Description</h4>
<ul>
    <li>Vehicle starts at depot with full capacity</li>
    <li>Must visit all customers to fulfill their demands</li>
    <li>Return to depot when capacity is exhausted</li>
    <li>Minimize total distance traveled</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>customer_locations</strong>: Coordinates of all customers</li>
    <li><strong>customer_demands</strong>: Demand at each customer</li>
    <li><strong>vehicle_capacity</strong>: Current remaining capacity</li>
    <li><strong>visited_mask</strong>: Which customers have been served</li>
    <li><strong>current_position</strong>: Vehicle's current location</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(num_customers + 1)</code> - Select next destination:</p>
<ul>
    <li><strong>0</strong>: Return to depot</li>
    <li><strong>1-N</strong>: Visit customer N</li>
</ul>

<h4>Rewards</h4>
<ul>
    <li><strong>Negative distance</strong>: Penalty for travel distance</li>
    <li>Goal: Minimize total distance (maximize reward)</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Termination</strong>: All customers served and returned to depot</li>
</ul>

<h4>Solution Approaches</h4>
<ul>
    <li><strong>Nearest Neighbor</strong>: Greedy closest customer</li>
    <li><strong>Clarke-Wright Savings</strong>: Merge routes for savings</li>
    <li><strong>Or-Opt</strong>: Local search improvement</li>
    <li><strong>Attention Models</strong>: Neural network approaches</li>
</ul>

<h4>Applications</h4>
<ul>
    <li>Delivery route optimization</li>
    <li>Waste collection routing</li>
    <li>Field service scheduling</li>
    <li>School bus routing</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Vehicle_routing_problem" target="_blank">Vehicle Routing Problem (Wikipedia)</a></li>
</ul>
"""


CVRP_HTML = get_cvrp_html()

__all__ = ["get_cvrp_html", "CVRP_HTML"]
