"""Tools layer for the CHD evaluation agent.

Public exports:

- :func:`~episodic_log.tools.session_tools.make_session_tools` — factory that
  returns session-bound tool callables.
- :data:`~episodic_log.tools.session_tools.TOOL_SCHEMAS` — OpenAI-format tool
  schema list for both ``grep_memory`` and ``load_turn``.
"""

from episodic_log.tools.session_tools import TOOL_SCHEMAS, make_session_tools

__all__ = ["make_session_tools", "TOOL_SCHEMAS"]
