"""Memory type classification — heuristic + LLM fallback.

v2 additions:
  - Code line filtering (skip code before classification)
  - Disambiguation (problem + resolution → outcome)
  - Confidence scoring (keyword match density)
  - Basic sentiment detection (positive/negative/neutral)
  - detailed=True mode returns all metadata

Backward-compatible: default behavior returns just the type string.
"""
from __future__ import annotations

import re
import logging
from typing import Literal, Optional, Union

from .config import CLASSIFIER_MODEL

logger = logging.getLogger(__name__)

# All supported memory types with descriptions
MEMORY_TYPES: dict[str, str] = {
    # Core types
    "fact": "A factual statement about a person, project, system, or concept",
    "decision": "A choice or decision that was made",
    "preference": "A user preference or style choice",
    "procedure": "Steps to accomplish something, a workflow or process",
    "relationship": "Information about a person, contact, or organizational relationship",
    "event": "Something that happened at a specific time",
    "insight": "A learned lesson, pattern, or strategic insight",
    # Action cycle types
    "intention": "An agent or user intends to do something in the future",
    "plan": "A structured plan, spec, or roadmap",
    "commitment": "A promise or commitment to deliver something",
    "action": "An action that was taken or is being taken",
    "outcome": "The result or consequence of an action",
    "cancellation": "Something that was cancelled or abandoned",
    # Knowledge types
    "goal": "A stated objective or target",
    "constraint": "A limitation, requirement, or boundary condition",
    "hypothesis": "An unverified belief or theory",
    "observation": "Something noticed or observed without interpretation",
    "question": "An open question that needs answering",
    "answer": "A response to a previously recorded question",
    "correction": "A correction to previously held information",
    "meta": "Information about the memory system itself",
}

# Keyword heuristics for fast classification (checked in order)
_KEYWORD_MAP: dict[str, list[str]] = {
    "decision": ["decided", "decision", "chose", "agreed", "resolved", "ruling", "approved", "rejected"],
    "preference": ["prefers", "preference", "likes", "wants", "favorite", "style", "rather"],
    "procedure": ["how to", "steps to", "process", "workflow", "recipe", "guide", "instructions"],
    "relationship": ["is a", "works at", "phone", "email", "contact", "reports to", "married", "founder of"],
    "event": ["deployed", "launched", "happened", "completed", "shipped", "released", "ipo", "merged", "broke"],
    "insight": ["learned", "lesson", "insight", "realized", "pattern", "takeaway", "key finding"],
    "goal": ["goal", "objective", "target", "milestone", "aim to", "plan to achieve"],
    "constraint": ["constraint", "limitation", "must not", "cannot", "requirement", "blocked by", "depends on"],
    "commitment": ["committed", "promised", "will deliver", "guaranteed", "pledged", "deadline"],
    "plan": ["roadmap", "plan", "spec", "architecture", "design doc", "phase 1", "phase 2"],
    "correction": ["correction", "actually", "was wrong", "updated", "revised", "turns out"],
    "question": ["question", "wondering", "unclear", "need to find out", "investigate"],
    "cancellation": ["cancelled", "abandoned", "dropped", "no longer", "deprecated", "killed"],
    "outcome": ["result", "outcome", "succeeded", "failed", "produced", "yielded"],
    "action": ["doing", "working on", "implementing", "building", "fixing", "deploying"],
    "observation": ["noticed", "observed", "saw that", "appears to be", "seems like"],
    "intention": ["intend to", "planning to", "going to", "will start", "next step"],
}

# ── v2: Code line filtering ──────────────────────────────────────────

_CODE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*[$>#]\s"),               # Shell prompts
    re.compile(r"^\s*(import |from .+ import)"),  # Python imports
    re.compile(r"^\s*(def |class |async def )"),  # Function/class defs
    re.compile(r"^\s*```"),                    # Code fences
    re.compile(r"^\s*[a-zA-Z_]\w*\s*=\s*"),   # Variable assignments
    re.compile(r"^\s*(if |elif |else:|for |while |try:|except |finally:)"),  # Control flow
    re.compile(r"^\s*return\s"),               # Return statements
    re.compile(r"^\s*#\s"),                    # Comments
    re.compile(r"^\s*//\s"),                   # JS/TS comments
    re.compile(r'^\s*"[^"]*":\s'),            # JSON keys
    re.compile(r"^\s*\{|\}\s*$"),             # Bare braces
]


def _filter_code_lines(text: str) -> str:
    """Remove code lines from text before classification.

    Preserves prose lines for better keyword matching.
    """
    lines = text.split("\n")
    prose_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Check alpha ratio — low alpha = likely code
        alpha_count = sum(1 for c in stripped if c.isalpha())
        if len(stripped) > 5 and alpha_count / len(stripped) < 0.4:
            continue
        # Check against code patterns
        if any(p.match(line) for p in _CODE_PATTERNS):
            continue
        prose_lines.append(line)
    return "\n".join(prose_lines)


# ── v2: Disambiguation ───────────────────────────────────────────────

_RESOLUTION_KEYWORDS = [
    "fixed", "resolved", "solved", "solution", "workaround", "patched",
    "now works", "working now", "the fix", "root cause was",
]


def _disambiguate(mem_type: str, text_lower: str) -> str:
    """Refine classification with disambiguation rules."""
    # Problem with resolution → outcome (it's a fix, not an active problem)
    if mem_type in ("question", "hypothesis"):
        if any(kw in text_lower for kw in _RESOLUTION_KEYWORDS):
            return "outcome"
    return mem_type


# ── v2: Confidence scoring ───────────────────────────────────────────

def _compute_confidence(text_lower: str, mem_type: str) -> float:
    """Compute classification confidence based on keyword match density."""
    keywords = _KEYWORD_MAP.get(mem_type, [])
    if not keywords:
        return 0.3  # No keywords = low confidence

    match_count = sum(1 for kw in keywords if kw in text_lower)
    # Scale: 1 match = 0.3, 2 = 0.5, 3 = 0.7, 5+ = 1.0
    confidence = min(1.0, match_count / 5.0 + 0.1)
    return max(0.3, confidence)


# ── v2: Sentiment detection ──────────────────────────────────────────

_POSITIVE_WORDS = {
    "pride", "joy", "happy", "love", "breakthrough", "solved", "success",
    "great", "excellent", "perfect", "amazing", "wonderful", "excited",
    "improved", "better", "progress", "achievement", "celebrate", "win",
    "working", "shipped", "deployed", "fixed", "accomplished", "proud",
    "grateful", "thankful", "awesome", "fantastic",
}

_NEGATIVE_WORDS = {
    "bug", "error", "crash", "fail", "broken", "issue", "stuck", "problem",
    "frustrated", "confused", "lost", "wrong", "bad", "terrible", "awful",
    "slow", "blocked", "regression", "revert", "rollback", "outage",
    "incident", "panic", "stressed", "worried", "anxious", "angry",
    "disappointing", "annoying", "painful",
}


def _detect_sentiment(text_lower: str) -> str:
    """Simple sentiment detection."""
    words = set(text_lower.split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


# ── Main API ─────────────────────────────────────────────────────────

def classify_heuristic(text: str) -> Optional[str]:
    """Fast keyword-based classification.

    Returns the first matching type, or None if no keywords match.
    """
    text_lower = text.lower()
    for mem_type, keywords in _KEYWORD_MAP.items():
        if any(kw in text_lower for kw in keywords):
            return mem_type
    return None


def classify_with_llm(text: str, model: str = CLASSIFIER_MODEL) -> str:
    """LLM-based classification for ambiguous memories."""
    import anthropic

    client = anthropic.Anthropic()
    type_list = ", ".join(MEMORY_TYPES.keys())

    response = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": (
                f"Classify this memory into exactly one type: {type_list}\n\n"
                f"Memory: {text[:500]}\n\n"
                "Respond with ONLY the type name, nothing else."
            ),
        }],
    )

    result = response.content[0].text.strip().lower()
    if result in MEMORY_TYPES:
        return result

    logger.warning(f"LLM returned unrecognized type '{result}', falling back to 'fact'")
    return "fact"


def classify_memory(
    text: str,
    use_llm: bool = False,
    detailed: bool = False,
) -> Union[str, dict]:
    """Classify a memory into one of the supported types.

    v2: Added detailed mode with confidence + sentiment.

    Args:
        text: The memory content to classify.
        use_llm: Whether to use LLM fallback for ambiguous cases.
        detailed: If True, returns dict with type, confidence, sentiment.

    Returns:
        String (type name) if detailed=False.
        Dict with type, confidence, sentiment if detailed=True.
    """
    # v2: Filter code lines before classification
    clean_text = _filter_code_lines(text)
    text_lower = (clean_text or text).lower()

    # Step 1: Keyword heuristic
    result = classify_heuristic(clean_text or text)

    # Step 2: LLM fallback
    if result is None and use_llm:
        try:
            result = classify_with_llm(text)
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")

    if result is None:
        result = "fact"

    # v2: Disambiguation
    result = _disambiguate(result, text_lower)

    if not detailed:
        return result

    # v2: Compute confidence and sentiment
    confidence = _compute_confidence(text_lower, result)
    sentiment = _detect_sentiment(text_lower)

    return {
        "type": result,
        "confidence": round(confidence, 2),
        "sentiment": sentiment,
    }


# ── Run 1: three-layer routing classifier ────────────────────────────
#
# Spec: brain/projects/jarvis-memory/plans/runs/2026-04-20-eval-harness-and-routing/spec.md §5-6.
# Routing doc: brain/MEMORY_PROTOCOL.md §"Three-layer routing rule".
#
# Non-blocking advisory. detect_layer() returns the predicted layer +
# confidence, and conversation.py turns confidence > 0.7 predictions of
# non-world-knowledge into a WARNING log line on the write path. Writes
# still persist; this is a signal, not a gate.

Layer = Literal["world_knowledge", "agent_operations", "session_ephemeral"]

# episode_type hints. These are high-confidence signals (explicit caller
# intent) that short-circuit the keyword scan.
_WORLD_KNOWLEDGE_TYPES: set[str] = {
    "decision", "fact", "plan", "completion", "outcome", "milestone",
    "correction", "event", "meeting", "handoff", "commitment", "insight",
    "relationship", "observation", "action", "intention", "procedure",
    "answer", "goal", "constraint", "cancellation", "hypothesis",
    "question", "meta",
}
_AGENT_OPS_TYPES: set[str] = {"preference", "config", "guideline"}
_SESSION_TYPES: set[str] = {"ephemeral", "session", "transcript"}

# Regex and phrase heuristics — keyword scans on lowercased content.
# Patterns per layer, each scored at 1.0 per match (confidence floors at 0.3).

_AGENT_OPS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\buser (prefers?|likes?|wants?|hates?|dislikes?)\b"),
    re.compile(r"\balex (prefers?|wants?|likes?|hates?|dislikes?)\b"),
    re.compile(r"\b(always|never) (do|use|respond|reply|format|include|add|mention)\b"),
    re.compile(r"\bdefault(s|\s+to)\b"),
    re.compile(r"\b(tool|response|output|formatting) (rule|config|settings?)\b"),
    re.compile(r"\bin (this|every) (session|conversation), (always|never|do not|don't)\b"),
    re.compile(r"\bclaude (should|must|needs to|always|never)\b"),
    re.compile(r"\bset up [\w\- ]{1,40} for (claude|cursor|codex|openclaw)\b"),
    re.compile(r"\b(preferences?|configuration|guideline)s? (for|about|on)\b"),
    re.compile(r"\b(system prompt|system instruction|instruction tuning)\b"),
    re.compile(r"\bauto[- ]?memory\b"),
    re.compile(r"\.claude/settings\.json"),
]

_SESSION_EPHEMERAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(this|current) (conversation|chat|session)\b"),
    re.compile(r"\bjust now\b"),
    re.compile(r"\b(what|as) i (just )?said\b"),
    re.compile(r"\b(as|like) (we|i) (just )?(said|mentioned|discussed)\b"),
    re.compile(r"\b(earlier|a moment ago|above) (in this|in the) (chat|conversation|thread)\b"),
    re.compile(r"\[temp\]"),
    re.compile(r"\[ephemeral\]"),
    re.compile(r"\bin this thread\b"),
]

# World-knowledge positive markers. Matched for completeness (raise
# confidence of a world_knowledge prediction) but never required — the
# default bucket is world_knowledge regardless.
_WORLD_KNOWLEDGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\[(decision|fact|plan|correction|meeting|completion|handoff|milestone)\]", re.IGNORECASE),
    re.compile(r"\bdecided to\b"),
    re.compile(r"\bchose\b"),
    re.compile(r"\bshipped\b"),
    re.compile(r"\bdeployed\b"),
    re.compile(r"\bmerged\b"),
    re.compile(r"\blaunched\b"),
    re.compile(r"\b(?:\$|€|£)\d+[\d,.]*\b"),  # money amounts
    re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),   # ISO dates
]

# Pronoun-heaviness signal for session_ephemeral. If the text is pronoun-
# dense without a proper noun / concrete entity, that's a tell.
_PRONOUNS: set[str] = {
    "i", "me", "my", "mine",
    "we", "us", "our", "ours",
    "you", "your", "yours",
    "it", "this", "that", "these", "those",
    "he", "she", "his", "her", "them", "their",
}
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")


def _count_matches(patterns: list[re.Pattern], text: str) -> int:
    """Sum matches across a list of patterns on the given text."""
    total = 0
    for pat in patterns:
        total += len(pat.findall(text))
    return total


def _confidence_from_counts(hits: int) -> float:
    """Map raw hit count → confidence in [0.0, 1.0].

    0 hits → 0.0. 1 → ~0.55. 2 → ~0.75. 3+ → 0.85+. Caps at 0.95 so
    callers never see a perfectly-certain classifier.
    """
    if hits <= 0:
        return 0.0
    # 1 - 0.45 * 0.55^hits → smooth saturating curve.
    return min(0.95, 1.0 - 0.45 * (0.55 ** hits))


def detect_layer(content: str, episode_type: Optional[str] = None) -> tuple[Layer, float]:
    """Classify a prospective write into one of three routing layers.

    Layers
    ------
    world_knowledge
        Facts, decisions, plans, events, and other durable observations
        about the world (people, companies, deals, code, infra). The
        default bucket — jarvis-memory's raison d'être.

    agent_operations
        Agent-side configuration: user preferences, response-formatting
        rules, always/never directives, tool config. Belongs in Claude's
        ``.auto-memory/`` or an OpenClaw equivalent, NOT in jarvis.

    session_ephemeral
        References to the current conversation ("this chat", "just now",
        pronoun-heavy with no concrete entity, or an explicit ``[TEMP]``
        tag). Should stay in the session window and not be persisted.

    Args:
        content: The candidate episode body.
        episode_type: Optional explicit type from the caller. Accepts
            ``None``; the classifier handles it the same way as an unknown
            type (keyword heuristics only).

    Returns:
        Tuple of ``(layer, confidence)``. Confidence is in [0.0, 0.95];
        0.0 always means ``world_knowledge`` (default bucket). Confidence
        > 0.7 for a non-default layer is the threshold the recorder uses
        to emit a WARNING.
    """
    # Null-safe content: detect_layer MUST NOT raise on empty input — the
    # write-path hook calls us before persisting and any exception would
    # break save_episode.
    if content is None:
        content = ""
    text = content if isinstance(content, str) else str(content)
    text_lower = text.lower()

    # Normalize episode_type.
    et = (episode_type or "").strip().lower() or None

    # ── 1. Explicit episode_type hint is the strongest signal ────────
    if et in _AGENT_OPS_TYPES:
        # Still scan content for agreement; a known ops type earns a
        # high base confidence that keyword hits add to.
        hits = _count_matches(_AGENT_OPS_PATTERNS, text_lower)
        return "agent_operations", max(0.85, _confidence_from_counts(hits + 1))
    if et in _SESSION_TYPES:
        hits = _count_matches(_SESSION_EPHEMERAL_PATTERNS, text_lower)
        return "session_ephemeral", max(0.85, _confidence_from_counts(hits + 1))
    if et in _WORLD_KNOWLEDGE_TYPES:
        wk_hits = _count_matches(_WORLD_KNOWLEDGE_PATTERNS, text_lower)
        return "world_knowledge", _confidence_from_counts(wk_hits + 1)

    # ── 2. Keyword scan per layer ────────────────────────────────────
    ops_hits = _count_matches(_AGENT_OPS_PATTERNS, text_lower)
    session_hits = _count_matches(_SESSION_EPHEMERAL_PATTERNS, text_lower)
    wk_hits = _count_matches(_WORLD_KNOWLEDGE_PATTERNS, text_lower)

    ops_conf = _confidence_from_counts(ops_hits)
    session_conf = _confidence_from_counts(session_hits)
    wk_conf = _confidence_from_counts(wk_hits)

    # ── 3. Pronoun-heavy boost for session_ephemeral ─────────────────
    # If the content is short, pronoun-dense, and lacks a proper noun,
    # it's more likely session_ephemeral.
    tokens = [t for t in re.findall(r"[a-zA-Z']+", text_lower) if t]
    if 3 <= len(tokens) <= 40:
        pronoun_count = sum(1 for t in tokens if t in _PRONOUNS)
        pronoun_ratio = pronoun_count / len(tokens) if tokens else 0.0
        has_proper_noun = bool(_PROPER_NOUN.search(text))
        if pronoun_ratio >= 0.25 and not has_proper_noun:
            session_conf = max(session_conf, 0.75)

    # ── 4. Pick winner ────────────────────────────────────────────────
    # Non-default layers must beat the default by a clear margin; this is
    # what keeps false positives low. If agent_ops and session both fire,
    # pick the higher-confidence one.
    best_layer: Layer = "world_knowledge"
    best_conf = max(wk_conf, 0.0)
    if ops_conf >= session_conf and ops_conf >= 0.5:
        best_layer = "agent_operations"
        best_conf = ops_conf
    elif session_conf > ops_conf and session_conf >= 0.5:
        best_layer = "session_ephemeral"
        best_conf = session_conf

    # Short, ambiguous content → fall back to world_knowledge / low conf.
    if best_layer == "world_knowledge":
        return "world_knowledge", round(wk_conf, 2)
    return best_layer, round(best_conf, 2)


# ── Run 2: entity-reference extraction ───────────────────────────────
#
# Spec: brain/projects/jarvis-memory/plans/runs/2026-04-20-entity-layer/spec.md.
#
# ``extract_entity_references`` is a lightweight companion to
# ``graph.extract_typed_edges``. Typed edges are *specific* (WORKS_AT,
# INVESTED_IN, ...) while entity refs are just "this episode talks about
# X" — used by ``record_episode`` to guarantee every referenced entity
# has an ambient Page, even if no typed edge fires.
#
# Returns a list of ``EntityRef`` that callers use to seed Page creation.
# Deterministic, no LLM.

from dataclasses import dataclass  # noqa: E402 (module-level imports already done above; keep local alias scope clean)


@dataclass(frozen=True)
class EntityRef:
    """A mention of an entity in episode content.

    ``slug`` is the canonical Page key; ``domain`` is a best-effort
    category; ``display`` preserves the surface form for logging.
    """

    slug: str
    domain: str
    display: str


# Domain hints from phrase context. Ordered; first match wins.
# Patterns are applied *around* a proper noun match to pick a domain.
_DOMAIN_CONTEXT_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(company|startup|firm|fund|LLC|Inc\.?|Corp\.?)\b", re.IGNORECASE), "company"),
    (re.compile(r"\b(project|initiative|codebase|repo|service)\b", re.IGNORECASE), "project"),
    (re.compile(r"\b(person|founder|CEO|CTO|engineer|advisor|investor)\b", re.IGNORECASE), "person"),
    (re.compile(r"\b(concept|pattern|framework|methodology|approach)\b", re.IGNORECASE), "concept"),
    (re.compile(r"\b(infrastructure|database|server|cluster|system)\b", re.IGNORECASE), "system"),
]


def _guess_domain(noun: str, surrounding_window: str) -> str:
    """Infer a domain for a detected proper noun from its sentence context."""
    for pattern, domain in _DOMAIN_CONTEXT_HINTS:
        if pattern.search(surrounding_window):
            return domain
    # Heuristic: single capitalized word = likely person or place.
    if " " not in noun:
        return "person" if noun[0].isupper() and len(noun) <= 12 else "topic"
    return "topic"


def extract_entity_references(
    content: str,
    episode_type: Optional[str] = None,
) -> list[EntityRef]:
    """Extract entity references from an episode body.

    Uses the same proper-noun pattern as ``graph._extract_proper_nouns``
    but exposes a stable public surface. Filters via ``detect_layer``:

    * If layer == ``session_ephemeral`` with confidence > 0.7, we suppress
      extraction (the content is probably a pronoun-heavy chat fragment
      about the current session, not durable world knowledge).
    * If layer == ``agent_operations`` with confidence > 0.7, we also
      suppress extraction (preferences/config don't need Page timelines).

    Args:
        content: Episode body.
        episode_type: Optional caller-supplied type.

    Returns:
        Deduplicated list of ``EntityRef``. Order is stable: by slug.
        Empty list on empty / ambiguous input.
    """
    if not content:
        return []

    # Gate: if detect_layer strongly predicts non-world-knowledge, skip.
    try:
        layer, conf = detect_layer(content, episode_type)
        if layer != "world_knowledge" and conf > 0.7:
            return []
    except Exception:  # noqa: BLE001 — advisory only
        pass

    # Import locally to avoid a circular import at module load
    # (graph.py imports pages; classifier is loaded very early).
    from .graph import _extract_proper_nouns  # type: ignore
    from .pages import slugify  # type: ignore

    nouns = _extract_proper_nouns(content)
    if not nouns:
        return []

    seen: set[str] = set()
    refs: list[EntityRef] = []
    for noun, start in nouns:
        slug = slugify(noun)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        # Grab ~60 chars around the mention for domain inference.
        left = max(0, start - 30)
        right = min(len(content), start + len(noun) + 30)
        window = content[left:right]
        domain = _guess_domain(noun, window)
        refs.append(EntityRef(slug=slug, domain=domain, display=noun))

    return sorted(refs, key=lambda r: r.slug)
