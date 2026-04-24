"""Agent loop and trajectory tracing for the CHD evaluation framework.

Public exports:

- :class:`~episodic_log.agent.loop.AgentLoop` — drives the tool-use agent loop.
- :class:`~episodic_log.agent.trace.AgentTrace` — full trajectory record for one run.
- :class:`~episodic_log.agent.trace.ToolCallRecord` — single tool invocation record.
"""

from episodic_log.agent.loop import AgentLoop
from episodic_log.agent.trace import AgentTrace, ToolCallRecord

__all__ = ["AgentLoop", "AgentTrace", "ToolCallRecord"]
