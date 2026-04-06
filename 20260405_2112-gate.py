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
  3. Preference detection — durable user preferences
  4. Assertion detection — facts about the user
  5. Commitment detection — time-bound intentions
  6. Default — skip if no storable signal found

Edge escalation triggers:
  - Assertion detected AND ephemeral suppressor present (unusual combination)
  - Multiple conflicting signal types in same turn
  - Explicit high-stakes language ("important", "remember this", "critical")
"""

from typing import List
from .models import GateDecision, TurnSignal
from .patterns import (
    BARE_CONTINUATION_PATTERN,
    EXPLICIT_CONTINUATION_PATTERN,
    PRONOUN_OPENER_PATTERN,
    PREFERENCE_PATTERN,
    ASSERTION_PATTERN,
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
        # If no assertion found downstream, will skip

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
    # 5. Assertion detection
    # ------------------------------------------------------------------
    assertion_matches = _find_all(ASSERTION_PATTERN, turn_stripped, "assertion")
    signals.extend(assertion_matches)

    # ------------------------------------------------------------------
    # 6. Commitment detection
    # ------------------------------------------------------------------
    commitment_matches = _find_all(COMMITMENT_PATTERN, turn_stripped, "commitment")
    signals.extend(commitment_matches)

    # ------------------------------------------------------------------
    # 7. Decision logic
    # ------------------------------------------------------------------
    has_preference = len(preference_matches) > 0
    has_assertion = len(assertion_matches) > 0
    has_commitment = len(commitment_matches) > 0
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
            f"Preference signal detected — extracting durable user preference."
        )

    # Clear assertion — store it
    if has_assertion and not has_continuation:
        return _decision(
            "extract", signals,
            f"Assertion signal detected — extracting user fact."
        )

    # Commitment — store with short TTL flag
    if has_commitment and not has_continuation:
        return _decision(
            "extract", signals,
            f"Commitment signal detected — extracting time-bound intention."
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
        "decision": decision,
        "signals": signals,
        "reason": reason,
        "signal_count": len(signals),
    }
