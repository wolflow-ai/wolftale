"""
wolftale.gate
-------------
Gate layer: pure function over conversation turn text.

No model calls. No network I/O. No state.
Input:  a turn string and its index in the conversation.
Output: GateDecision — extract | edge | skip

Latency target: < 20ms

Decision logic (in order):

  1. Continuation check — exits early if turn is mid-thought
  2. Ephemeral suppression — flags transient claims
  3. High-stakes escalation — "remember this", "important", etc. → edge
  4. Preference detection — durable user preferences
  5. Assertion detection — facts about the user
     5a. Compound form assertions (ASSERTION_PATTERN)
     5b. Proper noun assertions: "I am Chris", "I'm Messina"
         (PROPER_NOUN_ASSERTION_PATTERN — v2 addition)
  6. Commitment detection — time-bound intentions
  7. Default — skip if no storable signal found

Edge escalation triggers:
  - Assertion detected AND ephemeral suppressor present (unusual combination)
  - Multiple conflicting signal types in same turn
  - Explicit high-stakes language ("important", "remember this", "critical")

Changelog:
  v2 — Added PROPER_NOUN_ASSERTION_PATTERN check in step 5b.
       'I am Chris', 'I'm Messina' now route to extract correctly.
       Previously these fell through to skip because ASSERTION_PATTERN
       only matched compound forms ('I am a', 'I work at', etc.).
"""

from typing import List
from .models import GateDecision, TurnSignal
from .patterns import (
    BARE_CONTINUATION_PATTERN,
    EXPLICIT_CONTINUATION_PATTERN,
    PRONOUN_OPENER_PATTERN,
    PREFERENCE_PATTERN,
    ASSERTION_PATTERN,
    PROPER_NOUN_ASSERTION_PATTERN,
    COMMITMENT_PATTERN,
    EPHEMERAL_PATTERN,
    NAMED_ENTITY_BOOST_PATTERN,
)

# Explicit high-stakes markers — user flagging something as important
HIGH_STAKES_PATTERN = __import__('re').compile(
    r'\b(?:remember this|important|critical|don\'t forget|make sure you know|'
    r'this matters|keep in mind|never forget)\b',
    __import__('re').IGNORECASE
)


def evaluate(turn: str, turn_index: int = 0) -> GateDecision:
    """
    Evaluate a conversation turn and decide whether to extract a memory claim.

    Parameters
    ----------
    turn : str
        The raw text of a single conversation turn (user side only).
    turn_index : int
        Position of this turn in the conversation. Used for provenance.

    Returns
    -------
    GateDecision
        decision: "extract" | "edge" | "skip"
        signals: all detected signals
        reason: human-readable explanation
        signal_count: total signals detected
    """
    turn_stripped = turn.strip()
    signals: List[TurnSignal] = []

    # ------------------------------------------------------------------
    # 1. Continuation check — exit early, nothing to store
    # ------------------------------------------------------------------
    if BARE_CONTINUATION_PATTERN.match(turn_stripped):
        return _decision(
            "skip", signals,
            "Bare continuation detected — turn is a command or acknowledgement, not an assertion."
        )

    if EXPLICIT_CONTINUATION_PATTERN.search(turn_stripped):
        signals.append(_signal(
            EXPLICIT_CONTINUATION_PATTERN.search(turn_stripped).group(),
            "continuation"
        ))
        return _decision(
            "skip", signals,
            "Explicit continuation detected — turn builds on prior context, not a new assertion."
        )

    if PRONOUN_OPENER_PATTERN.match(turn_stripped):
        signals.append(_signal(turn_stripped.split()[0], "continuation"))
        # Pronoun opener is weak — don't exit yet, check for assertions too

    # ------------------------------------------------------------------
    # 2. Ephemeral detection — note but don't skip yet
    # ------------------------------------------------------------------
    ephemeral_matches = EPHEMERAL_PATTERN.findall(turn_stripped)
    ephemeral_detected = len(ephemeral_matches) > 0
    for match in ephemeral_matches:
        signals.append(_signal(match, "ephemeral"))

    # ------------------------------------------------------------------
    # 3. High-stakes check — escalate to edge path
    # ------------------------------------------------------------------
    high_stakes_match = HIGH_STAKES_PATTERN.search(turn_stripped)
    if high_stakes_match:
        signals.append(_signal(high_stakes_match.group(), "assertion"))
        return _decision(
            "edge", signals,
            f"High-stakes marker detected ('{high_stakes_match.group()}') — escalating to edge extraction path."
        )

    # ------------------------------------------------------------------
    # 4. Preference detection
    # ------------------------------------------------------------------
    preference_matches = _find_all(PREFERENCE_PATTERN, turn_stripped, "preference")
    signals.extend(preference_matches)

    # ------------------------------------------------------------------
    # 5a. Assertion detection — compound forms
    # ------------------------------------------------------------------
    assertion_matches = _find_all(ASSERTION_PATTERN, turn_stripped, "assertion")
    signals.extend(assertion_matches)

    # ------------------------------------------------------------------
    # 5b. Proper noun assertion detection (v2)
    # "I am Chris", "I'm Messina", "I am Chris Messina"
    # Kept as a separate check so it can be tuned independently.
    # ------------------------------------------------------------------
    proper_noun_match = PROPER_NOUN_ASSERTION_PATTERN.search(turn_stripped)
    if proper_noun_match:
        signals.append(_signal(proper_noun_match.group().strip(), "assertion"))
        # Merge into assertion_matches so the decision logic below sees it.
        assertion_matches = assertion_matches + [
            _signal(proper_noun_match.group().strip(), "assertion")
        ]

    # ------------------------------------------------------------------
    # 6. Commitment detection
    # ------------------------------------------------------------------
    commitment_matches = _find_all(COMMITMENT_PATTERN, turn_stripped, "commitment")
    signals.extend(commitment_matches)

    # ------------------------------------------------------------------
    # 7. Decision logic
    # ------------------------------------------------------------------
    has_preference  = len(preference_matches) > 0
    has_assertion   = len(assertion_matches) > 0
    has_commitment  = len(commitment_matches) > 0
    has_continuation = any(s["signal_type"] == "continuation" for s in signals)

    # Assertion + ephemeral is unusual — escalate
    if has_assertion and ephemeral_detected:
        return _decision(
            "edge", signals,
            "Assertion detected alongside ephemeral marker — unusual combination, escalating to edge path."
        )

    # Clear preference — store it
    if has_preference and not has_continuation:
        return _decision(
            "extract", signals,
            "Preference signal detected — extracting durable user preference."
        )

    # Clear assertion — store it
    if has_assertion and not has_continuation:
        return _decision(
            "extract", signals,
            "Assertion signal detected — extracting user fact."
        )

    # Commitment — store with short TTL flag
    if has_commitment and not has_continuation:
        return _decision(
            "extract", signals,
            "Commitment signal detected — extracting time-bound intention."
        )

    # Nothing storable found
    return _decision(
        "skip", signals,
        "No storable signal detected — turn appears to be a question, filler, or mid-thought."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_all(pattern, text: str, signal_type: str) -> List[TurnSignal]:
    """Find all non-overlapping matches and return as TurnSignal list."""
    seen = set()
    results = []
    for match in pattern.finditer(text):
        value = match.group().lower()
        if value not in seen:
            seen.add(value)
            results.append(_signal(match.group(), signal_type))
    return results


def _signal(value: str, signal_type: str) -> TurnSignal:
    return {"value": value, "signal_type": signal_type}


def _decision(decision: str, signals: List[TurnSignal], reason: str) -> GateDecision:
    return {
        "decision":     decision,
        "signals":      signals,
        "reason":       reason,
        "signal_count": len(signals),
    }
