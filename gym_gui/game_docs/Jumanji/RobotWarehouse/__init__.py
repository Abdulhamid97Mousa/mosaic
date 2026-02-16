"""Documentation for Jumanji RobotWarehouse environment.

RobotWarehouse simulates autonomous robots in a warehouse environment
coordinating to pick up and deliver items.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_robot_warehouse_html(env_id: str = "RobotWarehouse-v0") -> str:
    """Generate RobotWarehouse HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Robot Warehouse (RWARE) simulates a team of robots in a warehouse
environment. Robots must coordinate to pick up shelves containing
requested items and deliver them to workstations.
</p>

<h4>Environment Description</h4>
<ul>
    <li>Grid-based warehouse with shelves and aisles</li>
    <li>Multiple robots that can move and carry shelves</li>
    <li>Workstations where items are requested</li>
    <li>Robots must avoid collisions</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>warehouse_grid</strong>: Layout with shelves and obstacles</li>
    <li><strong>robot_positions</strong>: Location of all robots</li>
    <li><strong>robot_carrying</strong>: Which robots are carrying shelves</li>
    <li><strong>requests</strong>: Current delivery requests</li>
</ul>

<h4>Action Space</h4>
<p><code>MultiDiscrete([5] * num_agents)</code> - Each robot selects one of 5 actions:</p>
<ul>
    <li><strong>0</strong>: No-op (stay)</li>
    <li><strong>1</strong>: Move forward</li>
    <li><strong>2</strong>: Turn left</li>
    <li><strong>3</strong>: Turn right</li>
    <li><strong>4</strong>: Toggle load (pick up / put down shelf)</li>
</ul>
<p><em>All robots act simultaneously each step.</em></p>

<h4>Rewards</h4>
<ul>
    <li><strong>+1</strong>: Delivering a requested item</li>
    <li><strong>-0.5</strong>: Robot collision</li>
    <li><strong>-0.01</strong>: Step cost (encourages efficiency)</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Termination</strong>: All requests fulfilled</li>
    <li><strong>Truncation</strong>: Maximum steps reached</li>
</ul>

<h4>Challenges</h4>
<ul>
    <li><strong>Coordination</strong>: Avoid robot collisions</li>
    <li><strong>Path Planning</strong>: Navigate congested aisles</li>
    <li><strong>Task Allocation</strong>: Assign requests to robots</li>
    <li><strong>Traffic Management</strong>: Prevent deadlocks</li>
</ul>

<h4>Applications</h4>
<ul>
    <li>Amazon-style warehouse automation</li>
    <li>Multi-robot path planning (MAPF)</li>
    <li>Cooperative multi-agent RL</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://github.com/semitable/robotic-warehouse" target="_blank">RWARE Environment</a></li>
</ul>
"""


ROBOT_WAREHOUSE_HTML = get_robot_warehouse_html()

__all__ = ["get_robot_warehouse_html", "ROBOT_WAREHOUSE_HTML"]
