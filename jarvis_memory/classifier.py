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
from typing import Optional, Union

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
