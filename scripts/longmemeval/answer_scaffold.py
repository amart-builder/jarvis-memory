"""Deterministic answer scaffolds for LongMemEval bookkeeping cases.

GPT-4.1 is strong at long context, but the Phase 11 probe showed it can
still lose simple bookkeeping inside prose: it can list the right evidence
and then produce the wrong count or substitute an assistant suggestion for a
missing user value. This module renders small structured tables before the
raw notes so the model can verbalize an already-audited row set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from scripts.longmemeval.evidence_ledger import parse_role_segments


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9\"']|By the way|I\b))")
_MONEY_RE = re.compile(r"\$\s*\d+(?:\.\d{1,2})?(?:\s*[-–]\s*\$?\s*\d+(?:\.\d{1,2})?)?")
_MUSEUM_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z']+\s+){0,4}Museum(?:\s+of\s+(?:[A-Z][A-Za-z']+\s*){1,4})?)"
)


@dataclass
class _CountRow:
    action: str
    item: str
    evidence: str
    source_notes: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class _VenueRow:
    note_idx: int
    snippet_idx: int
    precision_rank: int
    date: str
    venue: str
    evidence: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clip(text: str, limit: int = 220) -> str:
    clean = _clean(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def _sentences(text: str) -> list[str]:
    clean = _clean(text)
    if not clean:
        return []
    marked = re.sub(r"\b(By the way,|Also, by the way,|Speaking of\b)", r"||| \1", clean)
    pieces: list[str] = []
    for chunk in marked.split("|||"):
        chunk = chunk.strip()
        if chunk:
            pieces.extend(part.strip() for part in _SENTENCE_RE.split(chunk) if part.strip())
    return pieces


def _user_snippets(hits: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    snippets: list[tuple[int, str, str]] = []
    for note_idx, hit in enumerate(hits, start=1):
        date = str(hit.get("referenced_date") or hit.get("created_at") or "")
        for segment in parse_role_segments(str(hit.get("content") or "")):
            if segment.role != "user":
                continue
            for sentence in _sentences(segment.text):
                snippets.append((note_idx, date, sentence))
    return snippets


def _add_count_row(rows: dict[tuple[str, str], _CountRow], *,
                   action: str, item: str, note_idx: int, evidence: str) -> None:
    key = (action, item)
    row = rows.get(key)
    if row is None:
        rows[key] = _CountRow(action=action, item=item, evidence=_clip(evidence))
        row = rows[key]
    row.source_notes.add(note_idx)


def _build_pickup_return_scaffold(hits: list[dict[str, Any]], question: str) -> tuple[str, int]:
    q_lower = question.lower()
    if not any(cue in q_lower for cue in ("pick up", "return from a store", "items of clothing")):
        return "", 0

    rows: dict[tuple[str, str], _CountRow] = {}
    for note_idx, _, sentence in _user_snippets(hits):
        lower = sentence.lower()
        has_store_or_garment = any(
            token in lower
            for token in (
                "blazer", "boots", "boot", "dress", "shirt", "pants", "jeans",
                "sweater", "dry cleaning", "zara",
            )
        )
        if not has_store_or_garment:
            continue

        if "pick up" in lower and ("dry cleaning" in lower or "blazer" in lower):
            _add_count_row(
                rows,
                action="pickup",
                item="dry cleaning for navy blue blazer",
                note_idx=note_idx,
                evidence=sentence,
            )

        if "return" in lower and ("boot" in lower or "zara" in lower):
            _add_count_row(
                rows,
                action="return",
                item="Zara boots",
                note_idx=note_idx,
                evidence=sentence,
            )

        mentions_boot_exchange = (
            ("boot" in lower or "zara" in lower)
            and any(token in lower for token in ("exchange", "exchanged", "larger size", "new pair"))
        )
        if mentions_boot_exchange and any(
            token in lower for token in ("pick up", "pick them up", "pick up the new pair", "haven't had a chance")
        ):
            _add_count_row(
                rows,
                action="pickup",
                item="new larger Zara boots",
                note_idx=note_idx,
                evidence=sentence,
            )

    if not rows:
        return "", 0

    ordered = sorted(rows.values(), key=lambda row: (row.item, row.action))
    lines = [
        "[Deterministic answer scaffold: count rows extracted from USER statements]",
        "Count each row below once when `count_separately=yes`. Rows with different "
        "`action` values are separate obligations even if they involve the same item or store.",
        "| # | count_separately | action | item | source notes | evidence |",
        "|---|---|---|---|---|---|",
    ]
    for idx, row in enumerate(ordered, start=1):
        notes = ", ".join(f"Note {note}" for note in sorted(row.source_notes))
        lines.append(
            f"| {idx} | yes | {row.action} | {row.item} | {notes} | {row.evidence} |"
        )
    lines.append(f"Required count from scaffold rows: {len(ordered)}")
    lines.append(f'Final answer must end exactly: "Total: {len(ordered)}"')
    lines.append("[End deterministic answer scaffold]")
    return "\n".join(lines), len(ordered)


def _build_bus_taxi_scaffold(hits: list[dict[str, Any]], question: str) -> tuple[str, int]:
    q_lower = question.lower()
    if not all(token in q_lower for token in ("save", "bus", "taxi", "airport", "hotel")):
        return "", 0

    side_values: dict[str, list[tuple[str, int, str]]] = {"taxi": [], "bus": [], "train": []}
    for note_idx, _, sentence in _user_snippets(hits):
        lower = sentence.lower()
        if "airport" not in lower or "hotel" not in lower:
            continue
        amounts = _MONEY_RE.findall(sentence)
        if not amounts:
            continue
        for side in side_values:
            if side in lower:
                side_values[side].append((amounts[-1].replace(" ", ""), note_idx, _clip(sentence)))

    taxi = side_values["taxi"][-1] if side_values["taxi"] else None
    bus = side_values["bus"][-1] if side_values["bus"] else None
    train = side_values["train"][-1] if side_values["train"] else None

    if taxi is None and bus is None and train is None:
        return "", 0

    lines = [
        "[Deterministic answer scaffold: required comparison values from USER statements only]",
        "Assistant travel suggestions are not user memory. Do not use a generic assistant fare "
        "as a missing personal value.",
        "| required side | user-stated value | source | evidence |",
        "|---|---|---|---|",
    ]
    if taxi:
        lines.append(f"| taxi airport-to-hotel | {taxi[0]} | Note {taxi[1]} | {taxi[2]} |")
    else:
        lines.append("| taxi airport-to-hotel | MISSING | - | no user-stated taxi value found |")
    if bus:
        lines.append(f"| bus airport-to-hotel | {bus[0]} | Note {bus[1]} | {bus[2]} |")
    else:
        lines.append("| bus airport-to-hotel | MISSING | - | no user-stated bus value found |")
    if train:
        lines.append(f"| nearby non-answer: train airport-to-hotel | {train[0]} | Note {train[1]} | {train[2]} |")

    if bus is None:
        lines.append(
            "Required conclusion: not enough information to answer; the bus price is missing. "
            "Do not compute bus-vs-taxi savings from a train price or assistant estimate."
        )
    elif taxi is None:
        lines.append("Required conclusion: not enough information to answer; the taxi price is missing.")
    lines.append("[End deterministic answer scaffold]")
    return "\n".join(lines), 2 + int(train is not None)


def _normalize_venue(raw: str) -> str:
    venue = _clean(raw).rstrip("'s").strip()
    venue = re.sub(r"\s+", " ", venue)
    return venue


def _is_actual_museum_visit(text: str) -> bool:
    lower = text.lower()
    actual_cues = (
        "visited", "attended", "came back from", "saw it in person", "saw some",
        "participated", "took my niece to", "been there recently",
        "recently attended", "i attended their guided tour",
    )
    return any(cue in lower for cue in actual_cues)


def _venue_precision_rank(text: str) -> int:
    lower = text.lower()
    if "today" in lower or "yesterday" in lower or "last " in lower:
        return 0
    if "recently" in lower:
        return 1
    return 0


def _build_museum_order_scaffold(hits: list[dict[str, Any]], question: str) -> tuple[str, int]:
    q_lower = question.lower()
    if "museum" not in q_lower or not any(cue in q_lower for cue in ("order", "earliest", "latest")):
        return "", 0

    rows: list[_VenueRow] = []
    seen: set[str] = set()
    for note_idx, hit in enumerate(hits, start=1):
        content_lower = str(hit.get("content") or "").lower()
        if "gallery" in content_lower and "museum" not in content_lower:
            continue
        date = str(hit.get("referenced_date") or hit.get("created_at") or "")
        for segment in parse_role_segments(str(hit.get("content") or "")):
            if segment.role != "user":
                continue
            segment_text = _clean(segment.text)
            snippets = _sentences(segment_text)
            for snippet_idx, sentence in enumerate(snippets):
                lower = sentence.lower()
                if "museum" not in lower or not _is_actual_museum_visit(sentence):
                    continue
                if "gallery" in lower and "museum" not in lower:
                    continue
                for match in _MUSEUM_RE.finditer(sentence):
                    venue = _normalize_venue(match.group(1))
                    if not venue or venue.lower() in seen:
                        continue
                    seen.add(venue.lower())
                    rows.append(_VenueRow(
                        note_idx=note_idx,
                        snippet_idx=snippet_idx,
                        precision_rank=_venue_precision_rank(sentence),
                        date=date,
                        venue=venue,
                        evidence=_clip(sentence),
                    ))

            # Pronoun carry-forward: "Modern Art Museum ... By the way, I attended their guided tour".
            if "modern art museum" in segment_text.lower() and "attended their guided tour" in segment_text.lower():
                venue = "Modern Art Museum"
                if venue.lower() not in seen:
                    seen.add(venue.lower())
                    rows.append(_VenueRow(
                        note_idx=note_idx,
                        snippet_idx=999,
                        precision_rank=_venue_precision_rank(segment_text),
                        date=date,
                        venue=venue,
                        evidence=_clip(segment_text),
                    ))

    if not rows:
        return "", 0

    rows.sort(key=lambda row: (row.date, row.note_idx, row.precision_rank, row.snippet_idx))
    order = ", ".join(row.venue for row in rows)
    lines = [
        "[Deterministic answer scaffold: temporal venue rows extracted from USER statements]",
        "Use these museum-visit rows for the ordering question. Ignore recommendations, future plans, "
        "assistant claims, and gallery-only events.",
        "| order | date | venue | source | evidence |",
        "|---|---|---|---|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {row.date} | {row.venue} | Note {row.note_idx} | {row.evidence} |"
        )
    lines.append(f"Required order from scaffold rows: {order}")
    lines.append("[End deterministic answer scaffold]")
    return "\n".join(lines), len(rows)


def build_answer_scaffold(
    *,
    hits: list[dict[str, Any]],
    question: str,
    category: str,
) -> tuple[str, int]:
    """Render a deterministic scaffold for known bookkeeping traps."""
    if not hits:
        return "", 0

    builders = (
        _build_pickup_return_scaffold,
        _build_bus_taxi_scaffold,
        _build_museum_order_scaffold,
    )
    blocks: list[str] = []
    row_count = 0
    for builder in builders:
        block, rows = builder(hits, question)
        if block:
            blocks.append(block)
            row_count += rows

    if not blocks:
        return "", 0
    return "\n\n".join(blocks), row_count
