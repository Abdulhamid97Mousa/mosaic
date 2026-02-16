"""Shared HTML fragments for SMAC / SMACv2 documentation."""

SMAC_CTDE_HTML = """
<h4>CTDE Paradigm (Centralised Training, Decentralised Execution)</h4>
<p>
During <b>training</b>, each agent can access the <em>global state</em> vector
(unit positions, health, shields for ALL units on both teams).  This enables
centralised critics (QMIX, MAPPO).  During <b>execution</b>, each agent sees
only its local partial observation (nearby units within sight range).
</p>
"""

SMAC_ACTIONS_HTML = """
<h4>Action Space (Discrete)</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 6px;">Index</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Notes</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">0</td><td style="border: 1px solid #ddd; padding: 6px;">NO-OP</td><td style="border: 1px solid #ddd; padding: 6px;">Dead agents only</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">1</td><td style="border: 1px solid #ddd; padding: 6px;">STOP</td><td style="border: 1px solid #ddd; padding: 6px;">Hold position</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">2</td><td style="border: 1px solid #ddd; padding: 6px;">MOVE NORTH</td><td style="border: 1px solid #ddd; padding: 6px;"></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">3</td><td style="border: 1px solid #ddd; padding: 6px;">MOVE SOUTH</td><td style="border: 1px solid #ddd; padding: 6px;"></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">4</td><td style="border: 1px solid #ddd; padding: 6px;">MOVE EAST</td><td style="border: 1px solid #ddd; padding: 6px;"></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">5</td><td style="border: 1px solid #ddd; padding: 6px;">MOVE WEST</td><td style="border: 1px solid #ddd; padding: 6px;"></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">6+</td><td style="border: 1px solid #ddd; padding: 6px;">ATTACK ENEMY i</td><td style="border: 1px solid #ddd; padding: 6px;">One per enemy in sight</td></tr>
</table>
<p><b>Action Masking:</b> Invalid actions are masked at each timestep.
Dead agents can only NO-OP.  Attack actions are only available for enemies within range.</p>
"""

SMAC_OBS_HTML = """
<h4>Observation Space</h4>
<p>Each agent's local observation includes (for visible units within sight range):</p>
<ul>
    <li><b>Allied units:</b> distance, relative x/y, health, shield, unit type</li>
    <li><b>Enemy units:</b> distance, relative x/y, health, shield, unit type</li>
    <li><b>Self features:</b> own health, shield, unit type, (optionally) own position</li>
    <li><b>Optional:</b> pathing grid (terrain walkability), terrain height</li>
</ul>
"""

SMAC_REWARD_HTML = """
<h4>Reward Structure</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 6px;">Mode</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>Shaped</b> (default)</td>
        <td style="border: 1px solid #ddd; padding: 6px;">
            Reward for each timestep based on damage dealt and received.
            +200 bonus for winning the battle.
        </td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>Sparse</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;">
            +1 for winning the battle, 0 otherwise.
        </td>
    </tr>
</table>
<p><b>Cooperative:</b> All agents receive the same shared team reward.</p>
"""
