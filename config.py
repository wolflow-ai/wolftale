"""
wolftale.config
---------------
Loads environment variables from .env file at repo root.
Import this at the top of any file that needs the API key.

Usage:
    from wolftale.config import ANTHROPIC_API_KEY

The .env file is gitignored and lives only on your machine.
Never hardcode keys. Never commit .env.
"""

import os
from pathlib import Path

# Load .env file if it exists — do this before reading os.environ
_env_path = Path(__file__).parent / ".env"

if _env_path.exists():
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

# Expose key as a named constant — fails loudly if missing
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "\n\nANTHROPIC_API_KEY not found.\n"
        "Create a .env file in the repo root with:\n\n"
        "  ANTHROPIC_API_KEY=your-key-here\n\n"
        "See .env.example for the template.\n"
    )
