"""Stage 5 v2 Phase 8 — typed observation extraction for LongMemEval.

Per Atlas's spec (`docs/eval/stage5-v2-spec.md` Phase 8) and Codex's
2026-04-28 review: at adapter ingestion, call gpt-4o-mini per session
to extract a small list of TYPED, CONCISE observations (NOT free-form
evidence packets — that approach was Phase 3, which net-regressed and
got reverted).

Why this is different from Phase 3:
  * Phase 3 dumped raw user-turn snippets into the prompt — too noisy,
    distracted reasoning-heavy questions.
  * Phase 8 produces structured rows: ``{type, key, value, date, details}``.
    The model sees compact facts ("user's 5K PB: 26:30 on 2023-07-30")
    instead of paragraph-long quotes.

Schema:
    observations: list of records, each with:
      - type: one of {event, fact, preference, update}
      - key: short lowercase topic, e.g. "5k_pb", "yoga_class", "coffee_temp"
      - value: short verbatim quote OR a number/state — never a paraphrase
      - date: ISO date (YYYY-MM-DD) if anchored in the text, else null
      - details: optional ≤80-char extra context, else empty string

Cost: ~$0.0005 per session × ~3000 sessions per 500q dataset
≈ $1.50 for full extraction. Cached on Neo4j after first ingest.

Determinism: temperature=0, seed=42, max_tokens capped per session.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_EXTRACTION_MODEL_DEFAULT = "gpt-4o-mini"
_MAX_TOKENS_DEFAULT = 800  # ~30 obs × ~25 tokens each is plenty
_RUN_SEED = 42

_VALID_TYPES = {"event", "fact", "preference", "update"}
_MAX_KEY_LEN = 60
_MAX_VALUE_LEN = 200
_MAX_DETAILS_LEN = 80
_MAX_OBSERVATIONS_PER_SESSION = 30

# Bump when EXTRACTION_PROMPT_TEMPLATE changes in a way that should
# invalidate cached extractions. The cache key includes this string,
# so a bump just means future runs miss the cache and re-extract.
PROMPT_VERSION = "phase8-v1"


# OMEGA-style strict prompt — verbatim only, typed, concise.
EXTRACTION_PROMPT_TEMPLATE = """You are an extractor. Your job is to read a chat session between a user
and an assistant, and produce a small list of typed, verbatim observations.

OUTPUT a JSON object with one key "observations". Each observation is:
{{
  "type": "event" | "fact" | "preference" | "update",
  "key":  short lowercase topic (snake_case), max {max_key} chars
  "value": short verbatim quote OR a numeric/state value, max {max_value} chars
  "date": ISO date "YYYY-MM-DD" if anchored in the text, else null
  "details": optional brief context, max {max_details} chars (use "" if none)
}}

Type meanings:
  - event:      something that happened ("user attended yoga class #5")
  - fact:       a stable user attribute ("user_age = 32")
  - preference: a stated like/dislike ("prefers iced coffee")
  - update:     a value that REPLACED an earlier value ("5K PB: 27:45 → 26:30")

Strict rules:
  1. Extract VERBATIM. Do NOT infer, summarize, or paraphrase. If the
     user did not state it explicitly, do not include it.
  2. Quote short snippets in "value" — exact phrases, not invented prose.
  3. If the same observation appears multiple times, output it only once.
  4. Numbers, ordinals ("my 5th session"), and dates are gold — capture
     them exactly as stated.
  5. Skip assistant-only content unless it relays a fact the user just
     stated and the assistant confirmed.
  6. Skip greetings, pleasantries, hypotheticals, and questions.
  7. Cap at {max_obs} observations per session. If more would qualify,
     keep the most specific (numeric/dated/ordinal) ones.

Session date (treat as the "today" for this conversation): {session_date}

--- BEGIN SESSION ---
{session_text}
--- END SESSION ---

Output ONLY the JSON object. No prose, no markdown, no preamble.
"""


@dataclass(frozen=True)
class Observation:
    """One typed observation extracted from a session."""
    type: str
    key: str
    value: str
    date: Optional[str]
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "key": self.key,
            "value": self.value,
            "date": self.date,
            "details": self.details,
        }

    def render_line(self) -> str:
        """Render as a single line for prompt context.

        Format examples:
          event: yoga class #5 [2023-06-12] — vinyasa style
          fact:  user_age = 32 [2023-04-11]
          pref:  coffee_temperature = iced [2023-05-01]
          update: 5K_PB: 27:45 → 26:30 [2023-07-30]
        """
        date_part = f" [{self.date}]" if self.date else ""
        details_part = f" — {self.details}" if self.details else ""
        return f"- {self.type}: {self.key} = {self.value}{date_part}{details_part}"


class ExtractionCache:
    """SQLite-backed cache for typed observation extractions.

    Why this exists: Phase 8 makes ~50 gpt-4o-mini calls per question
    (one per session). At 104q × 50 sessions × ~2-4s/call, that's
    ~70-150 minutes of API time per run. Cached, the second run drops
    to ~1-2 min — extraction becomes ~free.

    Cache key = sha256(prompt_version + model + session_date + session_text).
    Bumping ``PROMPT_VERSION`` invalidates all cached entries naturally.

    Concurrency: WAL mode + connection-per-call lets the 4-worker
    parallel harness share the cache safely. Each get/put opens a
    short-lived connection; SQLite's WAL handles concurrent readers
    plus a single writer at a time.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False is safe because we never share a single
        # connection across threads — we open per call. The flag just
        # silences SQLite's defensive thread-affinity check.
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Explicit busy timeout — under ~16 concurrent extraction threads
        # writing small JSON blobs, transient locks are expected. WAL allows
        # concurrent readers + 1 writer; busy_timeout keeps any second
        # writer waiting up to 30s instead of raising OperationalError.
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extractions (
                    cache_key TEXT PRIMARY KEY,
                    observations_json TEXT NOT NULL,
                    n_observations INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @staticmethod
    def make_key(*, session_text: str, session_date: str,
                 prompt_version: str, model: str) -> str:
        """Deterministic SHA-256 over the inputs that affect the extraction."""
        h = hashlib.sha256()
        for part in (prompt_version, model, session_date or "", session_text):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> Optional[list["Observation"]]:
        """Return cached observations as Observation instances, or None on miss."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT observations_json FROM extractions WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            with self._stats_lock:
                self._misses += 1
            return None
        with self._stats_lock:
            self._hits += 1
        try:
            payload = json.loads(row[0])
        except (TypeError, ValueError):
            logger.warning("cache: corrupt entry for key=%s; treating as miss", key[:12])
            return None
        out: list[Observation] = []
        for d in payload:
            obs = _coerce_observation(d) if isinstance(d, dict) else None
            if obs is not None:
                out.append(obs)
        return out

    def put(self, key: str, observations: list["Observation"], *,
            model: str, prompt_version: str, session_date: str) -> None:
        payload = json.dumps([obs.to_dict() for obs in observations])
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO extractions
                   (cache_key, observations_json, n_observations,
                    model, prompt_version, session_date)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (key, payload, len(observations), model, prompt_version,
                 session_date or ""),
            )
            conn.commit()

    @property
    def hits(self) -> int:
        with self._stats_lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._stats_lock:
            return self._misses

    def stats_summary(self) -> str:
        h, m = self.hits, self.misses
        total = h + m
        rate = (100 * h / total) if total else 0.0
        return f"cache: {h} hits / {m} misses ({rate:.1f}% hit rate)"


def build_extraction_prompt(session_text: str, session_date: str) -> str:
    """Format the extraction prompt for one session."""
    return EXTRACTION_PROMPT_TEMPLATE.format(
        session_text=session_text,
        session_date=session_date or "unknown",
        max_key=_MAX_KEY_LEN,
        max_value=_MAX_VALUE_LEN,
        max_details=_MAX_DETAILS_LEN,
        max_obs=_MAX_OBSERVATIONS_PER_SESSION,
    )


def parse_extraction_response(raw: str) -> list[Observation]:
    """Parse and validate the JSON response from the extractor.

    Defensive: handles model returning extra prose around JSON, or
    individual rows missing fields. Drops invalid rows rather than
    raising, since one bad observation shouldn't kill a whole session.
    """
    if not raw or not raw.strip():
        return []

    # Strip code-fence wrappers the model sometimes adds.
    txt = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, re.DOTALL)
    if fence_match:
        txt = fence_match.group(1).strip()

    # Find the outermost JSON object.
    first_brace = txt.find("{")
    last_brace = txt.rfind("}")
    if first_brace < 0 or last_brace <= first_brace:
        logger.debug("extractor: no JSON object found in response")
        return []
    txt = txt[first_brace : last_brace + 1]

    try:
        payload = json.loads(txt)
    except json.JSONDecodeError as e:
        logger.debug("extractor: JSON parse failed: %s", e)
        return []

    rows = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []

    out: list[Observation] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        obs = _coerce_observation(row)
        if obs is not None:
            out.append(obs)
        if len(out) >= _MAX_OBSERVATIONS_PER_SESSION:
            break
    return out


def _coerce_observation(row: dict[str, Any]) -> Optional[Observation]:
    """Validate one row from the model. Returns None if not coercible."""
    obs_type = str(row.get("type", "")).strip().lower()
    if obs_type not in _VALID_TYPES:
        return None

    key = str(row.get("key", "")).strip()
    if not key:
        return None
    key = key[:_MAX_KEY_LEN]

    value = row.get("value")
    if value is None:
        return None
    value = str(value).strip()[:_MAX_VALUE_LEN]
    if not value:
        return None

    date_raw = row.get("date")
    date: Optional[str] = None
    if isinstance(date_raw, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_raw.strip()):
        date = date_raw.strip()

    details = str(row.get("details", "") or "").strip()[:_MAX_DETAILS_LEN]

    return Observation(type=obs_type, key=key, value=value, date=date, details=details)


def extract_observations(
    *,
    session_text: str,
    session_date: str,
    client: Any = None,
    model: Optional[str] = None,
    cache: Optional[ExtractionCache] = None,
) -> list[Observation]:
    """Extract typed observations from one session via gpt-4o-mini.

    Args:
        session_text: full role-prefixed session text (post format_session_text).
        session_date: ISO date string for the session (for prompt context).
        client: pre-built OpenAI client. If None, builds one from env.
        model: model id; defaults to gpt-4o-mini. Overridable for tests.
        cache: optional ExtractionCache. On hit, skips OpenAI entirely.
               On miss, calls OpenAI and stores the result.

    Returns:
        List of Observation records. Empty list on failure (graceful
        degradation — Phase 8 is additive, not load-bearing).
    """
    if not session_text or not session_text.strip():
        return []

    model_id = model or os.getenv("JARVIS_LME_EXTRACT_MODEL", _EXTRACTION_MODEL_DEFAULT)

    cache_key: Optional[str] = None
    if cache is not None:
        cache_key = ExtractionCache.make_key(
            session_text=session_text,
            session_date=session_date,
            prompt_version=PROMPT_VERSION,
            model=model_id,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    if client is None:
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai SDK not installed; observation extraction disabled")
            return []
        # Same timeout discipline as the answerer (postmortem 2026-04-28):
        # CLOSE_WAIT on OpenAI's edge has wedged us before. 60s is plenty
        # for a single session-extraction call.
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=60.0,
            max_retries=2,
        )

    prompt = build_extraction_prompt(session_text, session_date)
    try:
        resp = client.chat.completions.create(
            model=model_id,
            max_tokens=_MAX_TOKENS_DEFAULT,
            temperature=0,
            seed=_RUN_SEED,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        # API errored — DO NOT cache. A retry on the next run might
        # produce a different (or non-empty) result, and we want that
        # path to actually retry rather than serve a stale-error.
        logger.warning("cache: extraction call failed (graceful skip, not cached): %s", e)
        return []

    observations = parse_extraction_response(raw)

    # API succeeded — cache the result, including empty lists. Empty is
    # a deterministic outcome of "the model returned no usable rows for
    # this prompt"; serving it from cache on re-runs preserves accuracy
    # parity. NOT caching empty results would let OpenAI's per-call
    # variance leak into benchmark scores.
    if cache is not None and cache_key is not None:
        try:
            cache.put(
                cache_key,
                observations,
                model=model_id,
                prompt_version=PROMPT_VERSION,
                session_date=session_date,
            )
        except Exception as e:  # noqa: BLE001
            # Cache write failures are non-fatal — we still return the obs.
            logger.warning("cache: write failed (non-fatal): %s", e)

    return observations


def extract_observations_batch(
    *,
    sessions: list[tuple[str, str, str]],
    client: Any = None,
    model: Optional[str] = None,
    max_workers: int = 4,
    cache: Optional[ExtractionCache] = None,
) -> dict[str, list[Observation]]:
    """Extract observations for many sessions in parallel.

    Args:
        sessions: list of ``(session_id, session_text, session_date)`` tuples.
        client: optional pre-built OpenAI client. If None, builds one.
        model: model id; defaults to gpt-4o-mini.
        max_workers: thread-pool size. 4 is the sweet spot for OpenAI rate
                     limits + a single-session timeout of 60s — at 16 in-
                     flight requests per Mini host (4 adapter workers each
                     with this 4-way pool) we stay well under TPM caps.

    Returns:
        Map of ``session_id -> list[Observation]``. Sessions whose
        extraction failed (empty/timeout) get an empty list — extraction
        is additive, never load-bearing.

    Why thread-pool not async: matches the rest of the adapter's blocking
    style (Neo4j driver is sync, Chroma is sync). Threads avoid having
    to thread an event loop through every call site.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not sessions:
        return {}

    # Build one client per call (cheap) so callers without an OpenAI
    # client don't have to construct one.
    if client is None:
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai SDK not installed; batch extraction disabled")
            return {sid: [] for sid, _, _ in sessions}
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=60.0,
            max_retries=2,
        )

    out: dict[str, list[Observation]] = {sid: [] for sid, _, _ in sessions}

    def _one(item: tuple[str, str, str]) -> tuple[str, list[Observation]]:
        sid, text, date = item
        return sid, extract_observations(
            session_text=text,
            session_date=date,
            client=client,
            model=model,
            cache=cache,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, item) for item in sessions]
        for fut in as_completed(futures):
            try:
                sid, obs = fut.result()
                out[sid] = obs
            except Exception as e:  # noqa: BLE001
                logger.warning("batch extraction worker failed: %s", e)

    return out
