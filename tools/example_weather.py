"""
Example plugin tool for bridge.py.
Drop any .py file in tools/ that exports TOOL_DEFINITION, SAFE, and execute(inp).
Restart bridge.py to load it.
"""

TOOL_DEFINITION = {
    "name": "get_time",
    "description": "Get the current UTC time.",
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}

SAFE = True  # True = auto-execute, False = requires operator confirmation

def execute(inp: dict) -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
