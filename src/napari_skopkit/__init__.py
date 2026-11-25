"""Run scikit-ops ops from napari.

Exposes the Ops panel, which lists every op skop can discover and runs the
chosen one in its own Appose environment -- off the GUI thread, with progress
and cancellation, and with its outputs added to the viewer as the layer types
their roles call for; the Workflows panel, its twin for workflows; and the
Environments panel, which shows and builds the environments ops run in.
"""

from __future__ import annotations

from ._environments import EnvironmentsPanel
from ._panel import OpsPanel, WorkflowsPanel

__all__ = ["EnvironmentsPanel", "OpsPanel", "WorkflowsPanel"]
