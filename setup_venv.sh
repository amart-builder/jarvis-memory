#!/bin/bash
# Create a fresh venv for jarvis-memory at ~/Atlas/jarvis-memory
# Run: cd ~/Atlas/jarvis-memory && bash setup_venv.sh

set -e

echo "=== Setting up jarvis-memory venv ==="

# Remove broken venv if it exists
if [ -d ".venv" ]; then
    echo "Removing broken .venv..."
    rm -rf .venv
fi

# Create fresh venv
python3 -m venv .venv
echo "✓ Created .venv"

# Install dependencies
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
echo "✓ Installed dependencies"

# Verify
.venv/bin/python -c "import neo4j; import mcp; print('✓ All imports OK')"

echo ""
echo "=== Done ==="
echo "venv python: $(pwd)/.venv/bin/python3"
echo ""
echo "Next: restart Claude Desktop to pick up the new config."
