"""
wolftale.retrieval
------------------
Retrieval layer: query-conditioned search with conflict surfacing.

Owns:
  retrieve(query, domain_hints, top_k)  → RetrievalResult

This is the read-side entry point for the pipeline. It sits between
the store and whatever generates a response (the LLM, the demo REPL,
or eventually an MCP tool).

Responsibilities:
  1. Call store.search() with caller-supplied domain hints (two-pass inside store)
  2. Detect conflicts in the returned claim set
  3. Surface conflicts transparently — tag them, don't hide them
  4. Return a RetrievalResult the caller can inject directly into a context window

What retrieval does NOT do:
  - Infer domains from query text (caller's responsibility — Wolfprompt does this)
  - Resolve conflicts (LLM's responsibility — it has more context than we do)
  - Make API calls (pure Python + store.search())

Conflict detection at retrieval time:
  Two claims in the same result set conflict if:
    - They share a domain AND
    - Their embeddings are similar (same structural shape) AND
    - They were not deduplicated at write time (meaning the store accepted both)
  
  Detection is embedding-free here — we compare claim IDs against the
  store's conflict metadata. Each ClaimRecord written with action "flagged"
  has its conflicts recorded in StoreResult. We don't re-embed at read time.

  Instead: cross-reference the result set against each claim's domain.
  If two claims share a domain and both appear in the result set, surface
  them as a conflicting pair. The LLM sees both and can reason over them.

Context window budget:
  top_k=3 is the default. At ~20 tokens per claim, 3 claims = ~60 tokens.
  Well within budget for any model. Caller can raise top_k if needed.

RetrievalResult shape:
  {
    "claims": List[ClaimRecord],       # All retrieved claims, ranked by relevance
    "conflicts": List[ConflictPair],   # Pairs of claims that conflict with each other
    "domain_hints": List[str],         # What domains were searched (for tracing)
    "query": str,                      # Original query (for tracing)
    "used_fallback": bool,             # Whether broad fallback was triggered
  }

ConflictPair shape:
  {
    "claim_a": ClaimRecord,
    "claim_b": ClaimRecord,
    "domain": str,                     # The shared domain where conflict lives
  }
"""

from typing import List, Optional, TypedDict

import numpy as np

from .models import ClaimRecord
from . import store


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

class ConflictPair(TypedDict):
    """Two claims in the result set that conflict with each other."""
    claim_a: ClaimRecord
    claim_b: ClaimRecord
    domain: str


class RetrievalResult(TypedDict):
    """
    Return type of retrieve().

    claims:       All retrieved claims, ranked by relevance (store order).
    conflicts:    Pairs of claims that share a domain — surfaced for the caller.
    domain_hints: Domains passed in by the caller (empty list = broad search).
    query:        Original query string, preserved for tracing.
    used_fallback: True if the domain-conditioned pass returned nothing and
                   the broad fallback was used instead.
    """
    claims: List[ClaimRecord]
    conflicts: List[ConflictPair]
    domain_hints: List[str]
    query: str
    used_fallback: bool


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    domain_hints: Optional[List[str]] = None,
    top_k: int = 3,
) -> RetrievalResult:
    """
    Retrieve relevant claims for a query.

    Parameters
    ----------
    query : str
        The user's query or conversation turn. Embedded by store.search().
    domain_hints : List[str] or None
        Domains to search first. If None, goes straight to broad search.
        Caller supplies these — typically from Wolfprompt classification.
        Valid values: "preference" | "identity" | "commitment" |
                      "technical" | "relational" | "ephemeral" | "other"
    top_k : int
        Maximum claims to return. Default 3 (~60 tokens, safe for any model).

    Returns
    -------
    RetrievalResult
        claims: ranked list of ClaimRecords
        conflicts: any same-domain pairs detected in the result set
        domain_hints: echoed back for tracing
        query: echoed back for tracing
        used_fallback: whether the broad fallback fired
    """
    hints = domain_hints or []

    # ------------------------------------------------------------------
    # Pass 1: domain-conditioned search (if hints supplied)
    # ------------------------------------------------------------------
    used_fallback = False

    if hints:
        claims = store.search(query, domains=hints, top_k=top_k)
        if not claims:
            # Domain search returned nothing — fall back to broad
            claims = store.search(query, domains=None, top_k=top_k)
            used_fallback = True
    else:
        # No hints — broad search immediately
        claims = store.search(query, domains=None, top_k=top_k)

    # ------------------------------------------------------------------
    # Conflict detection across result set
    # ------------------------------------------------------------------
    conflicts = _detect_conflicts(claims)

    return {
        "claims": claims,
        "conflicts": conflicts,
        "domain_hints": hints,
        "query": query,
        "used_fallback": used_fallback,
    }


def format_for_context(result: RetrievalResult) -> str:
    """
    Format a RetrievalResult as a context block for injection into a prompt.

    Produces a compact, readable block the LLM can reason over.
    Conflicting claims are grouped and flagged so the LLM sees the tension
    rather than having one silently dropped.

    Example output:

        [Memory]
        - Chris prefers dark theme UIs. (preference, confidence: 0.95)
        - Chris is based in St. Louis, Missouri. (identity, confidence: 0.90)

        [Conflicting signals — reason over these]
        - preference: "Chris prefers dark theme UIs." vs "Chris prefers light theme UIs."

    Parameters
    ----------
    result : RetrievalResult
        Output of retrieve().

    Returns
    -------
    str
        A formatted string ready for insertion into a system prompt or
        user turn prefix.
    """
    if not result["claims"]:
        return "[Memory]\nNo relevant memories found.\n"

    lines = ["[Memory]"]
    for claim in result["claims"]:
        conf = f"{claim['confidence']:.2f}"
        lines.append(f"- {claim['claim']} ({claim['domain']}, confidence: {conf})")

    if result["conflicts"]:
        lines.append("")
        lines.append("[Conflicting signals — reason over these]")
        for pair in result["conflicts"]:
            lines.append(
                f"- {pair['domain']}: \"{pair['claim_a']['claim']}\" "
                f"vs \"{pair['claim_b']['claim']}\""
            )

    if result["used_fallback"]:
        lines.append("")
        lines.append("[Note: domain-conditioned search returned nothing — broad fallback used]")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_conflicts(claims: List[ClaimRecord]) -> List[ConflictPair]:
    """
    Detect conflicting pairs within a retrieved claim set.

    Two claims conflict if:
      1. They share a domain (necessary — different domains can't conflict)
      2. Their embeddings are similar (sufficient — they're describing the same thing differently)

    Domain alone is not enough. "Chris prefers dark theme UIs" and "Chris prefers
    direct communication" are both preference claims but describe orthogonal things —
    flagging them as conflicting would be noise. "Chris prefers dark theme UIs" and
    "Chris prefers light theme UIs" are geometrically close and genuinely compete.

    Similarity threshold: store.CONFLICT_THRESHOLD (0.75). Same threshold used at
    write time — consistent definition of "similar enough to conflict" across the pipeline.

    Cost: embeds each claim once per retrieval call. Bounded by top_k (default 3),
    so at most 3 embeddings. Each embedding is ~1ms. Acceptable latency.

    Returns all conflicting pairs found. If three claims share a domain and are
    mutually similar, returns all three pairs: (A,B), (A,C), (B,C).
    """
    if len(claims) < 2:
        return []

    # Embed all claims in one batch — more efficient than one at a time
    texts = [c["claim"] for c in claims]
    embeddings = [store._embed(t) for t in texts]

    conflicts: List[ConflictPair] = []

    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            # Condition 1: same domain
            if claims[i]["domain"] != claims[j]["domain"]:
                continue

            # Condition 2: similar embeddings (cosine similarity)
            vec_a = embeddings[i]
            vec_b = embeddings[j]
            similarity = float(
                np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
            )

            if similarity >= store.CONFLICT_THRESHOLD:
                conflicts.append({
                    "claim_a": claims[i],
                    "claim_b": claims[j],
                    "domain": claims[i]["domain"],
                })

    return conflicts
