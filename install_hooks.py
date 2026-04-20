"""Install Claude hooks for jarvis-memory. Run: python3 ~/Jarvis/jarvis-memory/install_hooks.py"""
import json
import os

hooks_dir = os.path.expanduser("~/.claude")
os.makedirs(hooks_dir, exist_ok=True)

hooks = {
    "hooks": {
        "SessionStart": [
            {
                "type": "command",
                "command": "python3 ~/Jarvis/jarvis-memory/hooks/session_start.py",
                "timeout": 5000,
            }
        ],
        "Stop": [
            {
                "type": "command",
                "command": "python3 ~/Jarvis/jarvis-memory/hooks/session_stop.py",
                "timeout": 5000,
            }
        ],
    }
}

hooks_path = os.path.join(hooks_dir, "hooks.json")
with open(hooks_path, "w") as f:
    json.dump(hooks, f, indent=2)

print(f"Hooks configured at {hooks_path}")
