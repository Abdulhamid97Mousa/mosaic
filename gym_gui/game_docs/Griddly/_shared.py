"""Shared HTML fragments for Griddly game documentation."""

from __future__ import annotations

_TH = 'style="border: 1px solid #ddd; padding: 8px;"'
_TD = 'style="border: 1px solid #ddd; padding: 8px;"'
_TABLE = 'style="width:100%; border-collapse: collapse; margin: 10px 0;"'
_HDR = 'style="background-color: #f0f0f0;"'
_HDR_GREEN = 'style="background-color: #e8f5e9;"'

KEYBOARD_HTML = """
<h4>Keyboard Controls</h4>
<table {table}>
    <tr {hdr}>
        <th {th}>Key</th>
        <th {th}>Action</th>
        <th {th}>ID</th>
    </tr>
    <tr><td {td}>↑ or W</td><td {td}>UP</td><td {td}><strong>1</strong></td></tr>
    <tr><td {td}>↓ or S</td><td {td}>DOWN</td><td {td}><strong>2</strong></td></tr>
    <tr><td {td}>← or A</td><td {td}>LEFT</td><td {td}><strong>3</strong></td></tr>
    <tr><td {td}>→ or D</td><td {td}>RIGHT</td><td {td}><strong>4</strong></td></tr>
    <tr><td {td}><em>(no key)</em></td><td {td}>NOOP</td><td {td}><strong>0</strong></td></tr>
</table>
""".format(table=_TABLE, hdr=_HDR_GREEN, th=_TH, td=_TD)

BACKEND_HTML = """
<h4>Platform Details</h4>
<table {table}>
    <tr {hdr}>
        <th {th}>Property</th>
        <th {th}>Value</th>
    </tr>
    <tr><td {td}>Backend</td><td {td}>C++ with Vulkan GPU rendering</td></tr>
    <tr><td {td}>Throughput</td><td {td}>30 000+ FPS (headless training)</td></tr>
    <tr><td {td}>Interface</td><td {td}>Gymnasium (via MOSAIC compatibility wrapper)</td></tr>
    <tr><td {td}>Stepping</td><td {td}>Single-agent, turn-based</td></tr>
</table>
""".format(table=_TABLE, hdr=_HDR, th=_TH, td=_TD)

REFERENCE_HTML = """
<h4>References</h4>
<ul>
    <li><a href="https://github.com/Bam4d/Griddly" target="_blank">Griddly GitHub Repository</a></li>
    <li><a href="https://griddly.readthedocs.io" target="_blank">Griddly Documentation</a></li>
    <li><a href="https://griddly.readthedocs.io/en/latest/getting-started/gdy/index.html" target="_blank">GDY Language Reference</a></li>
</ul>
"""

__all__ = ["KEYBOARD_HTML", "BACKEND_HTML", "REFERENCE_HTML"]
