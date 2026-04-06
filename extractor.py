"""
wolftale.extractor
------------------
Claim extractor: takes a conversation turn and a GateDecision,
calls the Claude API, and returns a structured ClaimRecord.

Two paths:
  standard — single prompt, fast, cheap. Runs when gate says "extract".
  edge     — stubbed in v1. Runs when gate says "edge". Will become
             an RLM loop in a future build.

The prompt is the primary engineering surface here.
Claude decides what the claim is, what domain it belongs to,
and how confident it is — based entirely on the prompt we write.

Design notes:
  - Always returns ExtractionResult, even on failure.
  - Failure modes: model returns malformed JSON, model says no storable
    claim exists despite gate signal, API error.
  - raw_response is always preserved for debugging.
  - ClaimRecord IDs are UUIDs generated here, not by the store.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import anthropic

from .config import ANTHROPIC_API_KEY
from .models import GateDecision, ClaimRecord, ExtractionResult

# ---------------------------------------------------------------------------
# Claude client — initialized once at import time
# ---------------------------------------------------------------------------

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-haiku-4-5-20251001"   # Fast and cheap — this runs on every storable turn

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a memory extraction assistant. Your job is to read a single conversation turn and extract one discrete, storable fact about the user.

You must return a JSON object and nothing else. No preamble, no explanation, no markdown fences.

The JSON must have exactly these fields:

{
  "extractable": true or false,
  "claim": "A single, self-contained assertion about the user in plain English. Third person. E.g. 'Chris prefers dark theme UIs.' or 'Chris works at Wolflow.'",
  "confidence": 0.0 to 1.0,
  "domain": one of: "preference" | "identity" | "commitment" | "technical" | "relational" | "ephemeral" | "other"
}

Domain definitions:
  preference  — how the user likes things to work (durable, personal)
  identity    — who the user is, where they live, what they do (durable, factual)
  commitment  — something the user intends to do (time-bound, may expire)
  technical   — tools, languages, systems the user works with
  relational  — people, teams, organisations in the user's life
  ephemeral   — time-sensitive, unlikely to be relevant tomorrow
  other       — storable but doesn't fit above categories

Rules:
  - If the turn contains no storable fact about the user, set extractable to false and omit the other fields.
  - Extract only facts about the USER, not about the world in general.
  - Write the claim in third person, as a complete sentence.
  - Be specific. 'Chris prefers directness' is better than 'The user has preferences.'
  - Confidence reflects how certain you are this is a durable, accurate claim.
    High (0.8-1.0): clear, direct, unambiguous statement.
    Medium (0.5-0.79): implied or slightly ambiguous.
    Low (0.2-0.49): inferred, uncertain, or possibly ephemeral.

Signal hints (from gate layer — use to inform your extraction):
{signal_hints}
"""

USER_PROMPT = """Conversation turn:
\"\"\"{turn}\"\"\"

Extract the storable claim. Return JSON only."""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract(turn: str, gate: GateDecision, source_turn: int = 0) -> ExtractionResult:
    """
    Extract a ClaimRecord from a conversation turn.

    Parameters
    ----------
    turn : str
        The raw conversation turn text.
    gate : GateDecision
        The gate layer decision for this turn. Used to inform the prompt
        and determine which extraction path to use.
    source_turn : int
        Turn index in the conversation — stored as provenance in ClaimRecord.

    Returns
    -------
    ExtractionResult
        success: bool
        claim: ClaimRecord or None
        raw_response: str
        extraction_path: str
    """
    path = gate["decision"]  # "extract" or "edge"

    if path == "edge":
        return _extract_edge(turn, gate, source_turn)
    else:
        return _extract_standard(turn, gate, source_turn)


# ---------------------------------------------------------------------------
# Standard path — single prompt
# ---------------------------------------------------------------------------

def _extract_standard(turn: str, gate: GateDecision, source_turn: int) -> ExtractionResult:
    signal_hints = _format_signal_hints(gate)
    system = SYSTEM_PROMPT.replace("{signal_hints}", signal_hints)
    user = USER_PROMPT.replace("{turn}", turn)

    try:
        response = _client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = response.content[0].text.strip()
    except Exception as e:
        return _failure(str(e), "standard")

    return _parse_response(raw, source_turn, "standard")


# ---------------------------------------------------------------------------
# Edge path — stubbed in v1
# Will become an RLM loop: extract → verify → resolve conflicts → store
# ---------------------------------------------------------------------------

def _extract_edge(turn: str, gate: GateDecision, source_turn: int) -> ExtractionResult:
    """
    Edge path stub. Currently falls back to standard extraction
    but flags the result as edge-path for downstream handling.

    In a future build this will:
      1. Run standard extraction
      2. Query the store for potential conflicts
      3. Run a second Claude call to resolve the conflict
      4. Return a higher-confidence ClaimRecord with supersedes populated
    """
    # Run standard extraction for now
    result = _extract_standard(turn, gate, source_turn)

    # Re-tag as edge path so the store knows to treat it carefully
    if result["claim"]:
        result["claim"]["extraction_path"] = "edge"
        # Boost confidence slightly — edge path was triggered by explicit user emphasis
        result["claim"]["confidence"] = min(1.0, result["claim"]["confidence"] + 0.1)
    result["extraction_path"] = "edge"

    return result


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_response(raw: str, source_turn: int, path: str) -> ExtractionResult:
    """Parse the model's JSON response into an ExtractionResult."""
    # Strip markdown fences — Haiku sometimes wraps JSON despite instructions
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop first line (```json or ```) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return _failure(f"JSON parse error. Raw response: {raw}", path)

    if not data.get("extractable", False):
        return {
            "success": False,
            "claim": None,
            "raw_response": raw,
            "extraction_path": path,
        }

    # Validate required fields
    required = {"claim", "confidence", "domain"}
    missing = required - data.keys()
    if missing:
        return _failure(f"Missing fields: {missing}. Raw: {raw}", path)

    claim: ClaimRecord = {
        "claim": str(data["claim"]),
        "confidence": float(data["confidence"]),
        "domain": str(data["domain"]),
        "source_turn": source_turn,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "supersedes": [],
        "retrieved_count": 0,
        "last_retrieved": None,
        "extraction_path": path,
        "id": str(uuid.uuid4()),
    }

    return {
        "success": True,
        "claim": claim,
        "raw_response": raw,
        "extraction_path": path,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_signal_hints(gate: GateDecision) -> str:
    """Format gate signals as readable hints for the extraction prompt."""
    if not gate["signals"]:
        return "No specific signals detected."
    lines = [f"  - {s['signal_type']}: '{s['value']}'" for s in gate["signals"]]
    return "\n".join(lines)


def _failure(reason: str, path: str) -> ExtractionResult:
    """Return a failed ExtractionResult with the reason preserved."""
    return {
        "success": False,
        "claim": None,
        "raw_response": reason,
        "extraction_path": path,
    }
