"""Documentation for Jumanji JobShop environment.

JobShop scheduling is a classic NP-hard optimization problem where jobs
consisting of multiple operations must be scheduled on machines to
minimize makespan (total completion time).

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_jobshop_html(env_id: str = "JobShop-v0") -> str:
    """Generate JobShop HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
The Job Shop Scheduling Problem (JSSP) is a classic NP-hard combinatorial
optimization problem. Multiple jobs, each consisting of a sequence of
operations, must be scheduled on a set of machines to minimize the
total completion time (makespan).
</p>

<h4>Problem Description</h4>
<ul>
    <li>Given: A set of jobs, each with ordered operations</li>
    <li>Given: A set of machines, each operation requires a specific machine</li>
    <li>Given: Processing time for each operation</li>
    <li>Goal: Schedule all operations to minimize makespan</li>
    <li>Constraints: Operation precedence within jobs, machine capacity</li>
</ul>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>operations</strong>: Job-operation matrix with processing times</li>
    <li><strong>machine_assignments</strong>: Which machine each operation uses</li>
    <li><strong>schedule</strong>: Current partial schedule</li>
    <li><strong>action_mask</strong>: Valid scheduling actions</li>
</ul>

<h4>Action Space</h4>
<p><code>MultiDiscrete([num_jobs+1] * num_machines)</code> - Assign a job to each machine simultaneously:</p>
<ul>
    <li>One decision per machine (shape = num_machines)</li>
    <li><strong>0 to num_jobs-1</strong>: Schedule next operation of that job on the machine</li>
    <li><strong>num_jobs</strong>: No-op (machine idles this step)</li>
    <li>Operations must respect precedence within each job</li>
</ul>

<h4>Rewards</h4>
<ul>
    <li><strong>Sparse</strong>: Negative makespan at episode end</li>
    <li><strong>Dense (optional)</strong>: Incremental rewards for progress</li>
    <li>Goal: Minimize total completion time</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: All operations scheduled</li>
</ul>

<h4>Scheduling Strategies</h4>
<ul>
    <li><strong>SPT</strong>: Shortest Processing Time first</li>
    <li><strong>LPT</strong>: Longest Processing Time first</li>
    <li><strong>FIFO</strong>: First In, First Out</li>
    <li><strong>EDD</strong>: Earliest Due Date</li>
    <li><strong>Critical Path</strong>: Schedule based on critical operations</li>
</ul>

<h4>Applications</h4>
<ul>
    <li>Manufacturing scheduling</li>
    <li>Cloud computing job scheduling</li>
    <li>Project management</li>
    <li>Compiler optimization (instruction scheduling)</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Job-shop_scheduling" target="_blank">Job Shop Scheduling (Wikipedia)</a></li>
</ul>
"""


JOBSHOP_HTML = get_jobshop_html()

__all__ = [
    "get_jobshop_html",
    "JOBSHOP_HTML",
]
