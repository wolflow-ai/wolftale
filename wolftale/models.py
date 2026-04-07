"""
wolftale.models
---------------
Typed output structures for all Wolftale layers.

All public-facing return values are plain dicts conforming to these shapes.
TypedDicts are used for editor support and documentation — not enforced at runtime.

Layer map:
  gate.py        → GateDecision
  extractor.py   → ExtractionResult, ClaimRecord
  store.py       → StoreResult
  retrieval.py   → List[ClaimRecord]
"""

from typing import TypedDict, List, Optional


# ---------------------------------------------------------------------------
# Gate layer
# ---------------------------------------------------------------------------

class TurnSignal(TypedDict):
    """A single detected signal in a conversation turn."""
    value: str          # The matched text
    signal_type: str    # "preference" | "assertion" | "commitment" | "ephemeral" | "continuation"


class GateDecision(TypedDict):
    """
    Return type of gate.evaluate(turn: str) -> GateDecision.

    decision:
        "extract"  — turn contains a storable claim, run standard extractor
        "edge"     — turn contains a high-stakes or contradictory claim, run RLM path
        "skip"     — turn contains nothing worth storing

    signals:
        All signals detected in the turn, regardless of decision.

    reason:
        Human-readable explanation of why this decision was made.
    """
    decision: str               # "extract" | "edge" | "skip"
    signals: List[TurnSignal]   # All detected signals
    reason: str                 # Why this decision was made
    signal_count: int           # Total signals detected


# ---------------------------------------------------------------------------
# Claim record — the unit of memory (Option B: assertion with metadata)
# ---------------------------------------------------------------------------

class ClaimRecord(TypedDict):
    """
    A single stored memory unit.

    This is the core data structure of Wolftale.
    Every memory is an assertion with provenance — not a raw quote,
    not a summary, but a discrete, revisable, confidence-weighted claim.

    original_confidence:
        Set at extraction time. Never mutated. Used as the ceiling for
        retrieval reinforcement — repeated retrieval can restore a claim
        to its original strength, but not beyond it. This keeps extraction
        quality and retrieval frequency as independent signals.
    """
    claim: str                      # The extracted assertion in plain language
    confidence: float               # 0.0 - 1.0. Decays over time; bumped by retrieval.
    original_confidence: float      # Set at extraction. Ceiling for reinforcement. Never mutated.
    domain: str                     # "preference" | "identity" | "commitment" |
                                    # "technical" | "relational" | "ephemeral" | "other"
    source_turn: int                # Turn index this claim was extracted from
    timestamp: str                  # ISO 8601 — when this claim was stored
    supersedes: List[str]           # IDs of claims this one replaces
    retrieved_count: int            # How many times this claim has been retrieved
    last_retrieved: Optional[str]   # ISO 8601 — last retrieval timestamp, None if never
    extraction_path: str            # "standard" | "edge" — which extractor produced this
    id: str                         # UUID — unique identifier for this claim


# ---------------------------------------------------------------------------
# Extraction layer
# ---------------------------------------------------------------------------

class ExtractionResult(TypedDict):
    """
    Return type of extractor.extract(turn: str, gate: GateDecision) -> ExtractionResult.

    success:
        True if a claim was extracted. False if the model could not
        identify a storable claim despite the gate saying extract.

    claim:
        The extracted ClaimRecord. None if success is False.

    raw_response:
        The raw model response, preserved for debugging.
    """
    success: bool
    claim: Optional[ClaimRecord]
    raw_response: str
    extraction_path: str            # "standard" | "edge"


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------

class ConflictInfo(TypedDict):
    """Details of a detected conflict with an existing claim."""
    existing_id: str        # ID of the conflicting claim
    existing_claim: str     # Text of the conflicting claim
    similarity: float       # Cosine similarity score
    conflict_type: str      # "duplicate" | "contradiction" | "update"


class StoreResult(TypedDict):
    """
    Return type of store.write(claim: ClaimRecord) -> StoreResult.

    action:
        "stored"       — claim was written to the store
        "deduplicated" — claim was too similar to an existing one, skipped
        "superseded"   — claim replaced one or more existing claims
        "flagged"      — contradiction detected, needs resolution (edge path)
    """
    action: str                         # "stored" | "deduplicated" | "superseded" | "flagged"
    claim_id: Optional[str]             # ID of the stored claim, None if deduplicated
    conflicts: List[ConflictInfo]       # Any conflicts detected, empty if none
    reason: str                         # Human-readable explanation
