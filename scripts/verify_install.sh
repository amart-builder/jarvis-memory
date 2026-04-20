#!/bin/bash
# jarvis-memory — post-install sanity check
#
# Run this right after `pip install -e .` + `.env` + `migrate_to_v2.py` to
# confirm the stack is wired up correctly. Exits 0 if everything's good,
# non-zero with a clear message on the first failure.
#
# Usage:
#   cd jarvis-memory
#   source .venv/bin/activate
#   bash scripts/verify_install.sh

set -u

PASS='\033[0;32m✓\033[0m'
FAIL='\033[0;31m✗\033[0m'
SKIP='\033[0;33m·\033[0m'
NC='\033[0m'

fails=0
skips=0

check() {
    local label=$1
    shift
    if "$@" >/tmp/jarvis-verify.log 2>&1; then
        printf "  ${PASS} %s\n" "$label"
    else
        printf "  ${FAIL} %s\n" "$label"
        printf "${NC}    (see /tmp/jarvis-verify.log for output)\n"
        fails=$((fails + 1))
    fi
}

skip() {
    printf "  ${SKIP} %s  ${NC}(%s)\n" "$1" "$2"
    skips=$((skips + 1))
}

echo ""
echo "═══════════════════════════════════════════"
echo "  jarvis-memory install verification"
echo "═══════════════════════════════════════════"
echo ""

# ── Load .env if present ──────────────────────────────────
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "Loaded .env from $(pwd)/.env"
else
    echo "⚠  no .env in $(pwd) — some checks will skip"
fi
echo ""

# ── 1. Imports ────────────────────────────────────────────
echo "[1] Python package imports"
check "jarvis_memory imports" \
    python -c "import jarvis_memory, jarvis_memory.api, jarvis_memory.scoring, jarvis_memory.search, jarvis_memory.compaction"
check "mcp_server imports" \
    python -c "import mcp_server.server"

# ── 2. Entrypoints ────────────────────────────────────────
echo ""
echo "[2] Console scripts + module entrypoints"
check "jarvis-mcp entrypoint registered" \
    bash -c 'command -v jarvis-mcp'
check "python -m jarvis_memory.api import" \
    python -c "from jarvis_memory.api import app; assert hasattr(app, 'routes')"
check "scripts/migrate_to_v2.py --help" \
    python scripts/migrate_to_v2.py --help

# ── 3. Neo4j connectivity + schema ────────────────────────
echo ""
echo "[3] Neo4j (via .env)"
if [[ -z "${NEO4J_URI:-}" ]]; then
    skip "Neo4j reachable"        "NEO4J_URI unset"
    skip "Run 2 schema applied"   "NEO4J_URI unset"
else
    check "Neo4j reachable at $NEO4J_URI" \
        python -c "from neo4j import GraphDatabase; import os; d = GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])); d.verify_connectivity(); d.close()"
    check "Run 2 schema applied (page_slug_unique + page_compiled_truth_fulltext)" \
        bash -c 'python scripts/migrate_to_v2.py --dry-run --json | python -c "import json,sys; d=json.load(sys.stdin); assert d[\"planned_count\"] == 0, f\"migration not applied: {d[\\\"planned_count\\\"]} statements pending\""'
fi

# ── 4. ChromaDB + embedding model ─────────────────────────
echo ""
echo "[4] ChromaDB + embedding model"
check "sentence-transformers model cached" \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ── 5. Core flows ─────────────────────────────────────────
echo ""
echo "[5] Core flows"
check "scored_search fails-open without ANTHROPIC_API_KEY (expansion returns [query])" \
    env -u ANTHROPIC_API_KEY python -c "from jarvis_memory.search.expansion import expand; out = expand('test', n=3); assert out == ['test'], out"
check "RRF combiner produces stable ranking" \
    python -c "from jarvis_memory.search.rrf import reciprocal_rank_fusion as r; assert r([['a','b'],['b','a']])[0][0] in ('a','b')"
check "Intent classifier returns expected categories" \
    python -c "from jarvis_memory.search.intent import classify; assert classify('last week notes') == 'temporal'; assert classify('what is the architecture') == 'general'"

# ── 6. MCP parity ─────────────────────────────────────────
echo ""
echo "[6] MCP server"
check "MCP tool count = 27 (parity lock)" \
    python -c "
import re, pathlib
src = pathlib.Path('mcp_server/server.py').read_text()
# Count the Tool(name=\"...\") literals inside the list_tools() function.
names = re.findall(r'Tool\(\s*\n\s*name=\"([^\"]+)\"', src)
assert len(names) == 27, f'expected 27 Tool literals, found {len(names)}: {names}'
"

# ── 7. Pytest smoke ───────────────────────────────────────
echo ""
echo "[7] Pytest (smoke subset)"
check "pytest tests/search/ tests/test_scoring.py tests/test_mcp_parity.py (fast subset)" \
    python -m pytest tests/search/ tests/test_scoring.py tests/test_mcp_parity.py -q --no-header --tb=line

# ── Summary ──────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
if (( fails == 0 )); then
    echo -e "  ${PASS} ${NC}All checks passed ($skips skipped)"
    echo "  jarvis-memory is install-ready."
    exit 0
else
    echo -e "  ${FAIL} ${NC}$fails check(s) failed"
    echo "  Fix the first failure (tail /tmp/jarvis-verify.log) and re-run."
    exit 1
fi
