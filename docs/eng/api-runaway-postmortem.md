# jarvis-memory API runaway — postmortem (2026-04-28)

## Incident

The `jarvis_memory.api` process on the Mac Mini (PID 1603, started Sun 2026-04-26 17:00 PT) consumed **~75 hours of CPU time over 1 day 17 hours of wall time**, pegging two cores at 191.9%. The process was unresponsive to HTTP requests (`curl /health` timed out) but the kernel kept feeding it CPU. It had to be SIGKILLed.

**User-visible impact:**
- Production `jarvis-memory` API was hung — OpenClaw memory writes from production sessions silently failed
- `chromadb_count` reported 0 embeddings on `/health` (vector search was effectively disabled)
- The runaway also slowed down concurrent LongMemEval benchmark runs on the same Mini by ~4× (workers competed for CPU)

## Why we don't have proof of the trigger

The launchd plist `com.atlas.memclawz-api` redirected stdout/stderr to `/tmp/jarvis-memory-api.{log,err}`. By the time we noticed the runaway, those files had been overwritten / lost. **No log trail exists for the failed run.**

This is itself a finding. Fix #3 below addresses it.

## Strongest hypothesis (subagent diagnosis, 2026-04-28)

`jarvis_memory/compaction.py:229-250` — the `daily_digest` Pass-2 deduplication loop calls `self._embed_store.search(query=content, limit=5)` **once per remaining memory in a tight Python `for` loop**. `EmbeddingStore.search()` runs an on-the-fly sentence-transformer encode of the query string for every call (`embeddings.py:122-132`).

When ChromaDB has 0 embeddings (which it did — see "User-visible impact" above), `health_check()` still returns True (it just calls `count()` which returns 0 cleanly). So the gate passes, and every loop iteration does a forward pass through MiniLM-L6-v2 against a near-empty index.

With 295 active episodes pulled per `daily_digest` (`limit=500`), that's hundreds of CPU-bound model invocations per cron run, plus an inner `for s in similar` linear-scan loop that repeats the work. **191.9% CPU on 2 cores is the textbook sentence-transformer signature.**

**Why the runaway lived in the API process, not the cron:** unclear without logs. Possibilities:
- An API endpoint indirectly triggered `daily_digest`-equivalent behavior (less likely — none directly call it)
- The cron `scripts/run_compaction.py` somehow used the API's in-process embedding store via a shared singleton (very implausible)
- A different hot loop entirely, with the same CPU signature, lives somewhere I haven't looked

**Confidence:** ~70%. The signature matches; the trigger pathway is plausible but unproven.

## Alternative hypotheses (ranked)

1. **Sync `engine.compact_session` blocking the event loop + hook re-fire pile-up** (`api.py:522-541`). The endpoint is declared `async def` but calls a sync method that holds the event loop until completion. If `session_stop.py` hooks fire faster than compactions complete, requests pile up. Doesn't explain 75h of CPU on its own (compaction is hash-only, no embeddings) but explains the unresponsive `/health`.

2. **Sentence-transformer model load failing repeatedly via a retry loop.** `EmbeddingStore.__init__` catches exceptions and sets `_available=False`, so this would only happen if `_get_embed_store()` is called per-request and the constructor re-runs. Currently it's a singleton, so this is unlikely unless something nullifies `_embed_store`.

3. **Bug we haven't found.** The fact that we have no logs means we can't rule out something completely different.

## Containment (already deployed, 2026-04-28)

| # | Safeguard | Where | What it catches |
|---|---|---|---|
| C1 | **CPU watchdog** running every 60s. Restarts the API if the process tree exceeds 150% CPU for 5 consecutive ticks. | `~/Atlas/bin/jarvis-memory-api-watchdog.sh` + `com.atlas.jarvis-memory-api-watchdog.plist` (Mini) | ANY runaway, regardless of cause. Symptom-level fix. |
| C2 | **Persistent log files** at `~/Atlas/brain/logs/jarvis-memory-api.{log,err}` (was `/tmp/`, which wipes on reboot) | Edit to `com.atlas.memclawz-api.plist` | Evidence trail for next runaway |
| C3 | **Watchdog tick log** at `~/Atlas/brain/logs/jarvis-memory-api-watchdog.log` (per-minute CPU stamp) | Watchdog script writes this | Continuous monitoring; `tail -f` to spot trouble live |

C1 alone makes the exact failure mode (75h-of-CPU runaway) impossible to recur for more than ~5 minutes.

## Recommended causal fixes (proposed; review before applying)

### F1 — Batch the embedding-search loop (highest impact)

**File:** `jarvis_memory/compaction.py:229-250`
**Current:** per-memory `embed_store.search(query=content, limit=5)` calls in a Python `for` loop.
**Replace with:** single batched encode (`SentenceTransformer.encode(list, batch_size=64)`) followed by a vectorized cosine-similarity matrix in NumPy.
**Add:** wall-clock guard at the top of every compaction phase:
```python
import time
phase_start = time.monotonic()
# ...
for ...:
    if time.monotonic() - phase_start > 120:
        logger.warning("daily_digest phase exceeded 120s wall-clock; aborting")
        break
```

**Risk:** medium — changes dedup algorithm. Needs unit tests verifying dedup decisions match prior behavior on a fixed corpus.

### F2 — True fire-and-forget compact_session

**File:** `jarvis_memory/api.py:522-541`
**Current:**
```python
async def v1_compact_session_compat(req):
    engine = CompactionEngine(...)
    result = engine.compact_session(req.session_id)  # SYNC, blocks event loop
    return {"status": "ok", "result": result}
```
**Proposed:**
```python
async def v1_compact_session_compat(req, background_tasks: BackgroundTasks):
    background_tasks.add_task(_do_compact, req.session_id)
    return {"status": "accepted", "session_id": req.session_id}
```
Plus a per-`session_id` debounce: skip if a compaction for this session ran in the last 5 min (in-memory dict, evict old entries).

Plus the missing Neo4j index:
```cypher
CREATE INDEX EpisodicNode_source_description IF NOT EXISTS
  FOR (n:EpisodicNode) ON (n.source_description)
```

**Risk:** medium-high — changes API contract (response shape, status code 200→202 effectively). The `session_stop.py` hook that calls this endpoint would need to handle the new shape. Needs hook-side changes too.

### F3 — Structured logging to file (lowest risk)

**File:** `jarvis_memory/api.py` startup
**Add:** Python `logging` configured to write to a real rotating file (`~/Atlas/brain/logs/jarvis-memory-api.app.log`) at INFO level. Don't rely on stdout-to-launchd alone.

**Add:** `/metrics` endpoint exposing event-loop lag (sample `time.monotonic()` from a background task; report jitter).

**Add:** `--timeout-keep-alive 5 --limit-concurrency 32` flags to the uvicorn invocation in `start-jarvis-memory-api.sh`, so a single blocked handler can't wedge the whole server.

**Risk:** low. Pure observability + uvicorn tuning. No semantic change.

## Order of application

1. F3 (lowest risk, biggest evidence-trail upside)
2. F1 (highest causal value; needs the most testing)
3. F2 (defer until F1 lands and the API contract change is coordinated with `hooks/session_stop.py`)

## Open questions

- **Has the daily compaction cron been failing silently?** The plist `com.atlas.jarvis-compact-daily` exists. We have no log of its last successful run. The 0-embedding ChromaDB state suggests embeddings have been broken for a while.
- **Was ChromaDB always at 0 embeddings, or did it lose data?** If lost, that's a separate incident.
- **Does `scripts/run_compaction.py` have its own log path, or also `/tmp/`?** Mirrors the API issue.

We should answer these as a follow-up, but they're not blocking the immediate Phase 6 / Phase 10 work.
