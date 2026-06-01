---
name: Allow uv run python commands without asking
description: User wants all uv run python commands to execute without permission prompts
type: feedback
---

Always run `uv run python` commands without asking for permission. These are safe read-only operations in the project context.

**Why:** User finds the permission prompts disruptive during iterative development and testing.

**How to apply:** Any `uv run python`, `uv run pytest`, or similar uv-based commands should just execute.
