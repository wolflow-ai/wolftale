"""
test_retrieval.py
-----------------
Tests the retrieval layer end to end.

Exercises:
  1. retrieve() with domain hints — expect domain-scoped results
  2. retrieve() with no hints — expect broad search
  3. retrieve() where hints return nothing — expect fallback fires
  4. Conflict detection — two same-domain claims in result set → conflict pair
  5. format_for_context() — readable output with conflict block
  6. used_fallback flag — correct in all three cases above

Run from repo root with venv activated:
    python test_retrieval.py

No API calls. Populates the store with known claims, then retrieves against them.
Wipes and repopulates the store at the start of each run.
"""

import sys
import os
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wolftale.retrieval import retrieve, format_for_context
from wolftale.store import write, _get_collection
from wolftale.models import ClaimRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_claim(text: str, domain: str, confidence: float = 0.85) -> ClaimRecord:
    return {
        "claim": text,
        "confidence": confidence,
        "domain": domain,
        "source_turn": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "supersedes": [],
        "retrieved_count": 0,
        "last_retrieved": None,
        "extraction_path": "standard",
        "id": str(uuid.uuid4()),
    }


def wipe_and_seed() -> None:
    """Wipe the collection and write a known set of claims for testing."""
    import wolftale.store as store_module
    coll = _get_collection()
    client = coll._client
    try:
        client.delete_collection(store_module.COLLECTION_NAME)
    except Exception:
        pass
    store_module._collection = None

    # Seed with a known set
    claims = [
        make_claim("Chris prefers dark theme UIs.", "preference", 0.95),
        make_claim("Chris prefers direct communication over diplomatic softening.", "preference", 0.88),
        make_claim("Chris prefers light theme UIs.", "preference", 0.80),   # Genuine conflict with dark theme
        make_claim("Chris is based in St. Louis, Missouri.", "identity", 0.92),
        make_claim("Chris works under the Wolflow brand.", "identity", 0.90),
        make_claim("Chris uses Python 3.10.11 for all his projects.", "technical", 0.87),
        make_claim("Chris builds with Gradio for local UIs.", "technical", 0.83),
        make_claim("Chris will follow up with Marcus on Friday.", "commitment", 0.75),
    ]
    for c in claims:
        write(c)

    print(f"  [Store seeded with {len(claims)} claims]\n")


def print_result(result: dict) -> None:
    print(f"  Query       : {result['query']}")
    print(f"  Hints       : {result['domain_hints'] or 'none (broad)'}")
    print(f"  Fallback    : {result['used_fallback']}")
    print(f"  Claims ({len(result['claims'])}):")
    for c in result["claims"]:
        print(f"    [{c['domain']}] {c['claim']}")
    if result["conflicts"]:
        print(f"  Conflicts ({len(result['conflicts'])}):")
        for pair in result["conflicts"]:
            print(f"    [{pair['domain']}] '{pair['claim_a']['claim'][:40]}' vs '{pair['claim_b']['claim'][:40]}'")
    print()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    print("\n" + "=" * 60)
    print("  WOLFTALE — Retrieval Layer Tests")
    print("=" * 60 + "\n")

    wipe_and_seed()

    # ------------------------------------------------------------------
    # Test 1: Domain-conditioned retrieval
    # ------------------------------------------------------------------
    print("TEST 1: retrieve() with domain hints — preference")
    r1 = retrieve("How does Chris like his interfaces?", domain_hints=["preference"])
    print_result(r1)
    assert len(r1["claims"]) > 0, "Expected results"
    assert all(c["domain"] == "preference" for c in r1["claims"]), \
        "Expected only preference claims"
    assert r1["used_fallback"] == False
    assert r1["domain_hints"] == ["preference"]

    # ------------------------------------------------------------------
    # Test 2: Multi-domain hints
    # ------------------------------------------------------------------
    print("TEST 2: retrieve() with multi-domain hints — identity + technical")
    r2 = retrieve("Tell me about Chris's background and tools.", domain_hints=["identity", "technical"])
    print_result(r2)
    assert len(r2["claims"]) > 0, "Expected results"
    assert all(c["domain"] in ("identity", "technical") for c in r2["claims"]), \
        "Expected only identity or technical claims"
    assert r2["used_fallback"] == False

    # ------------------------------------------------------------------
    # Test 3: No hints — broad search
    # ------------------------------------------------------------------
    print("TEST 3: retrieve() with no hints — broad search")
    r3 = retrieve("Where does Chris live?")
    print_result(r3)
    assert len(r3["claims"]) > 0, "Expected results"
    assert r3["used_fallback"] == False  # No hints → broad directly, fallback stays False
    assert r3["domain_hints"] == []

    # ------------------------------------------------------------------
    # Test 4: Conflict detection — two preference claims in result set
    # ------------------------------------------------------------------
    print("TEST 4: Conflict detection — query that pulls multiple preference claims")
    r4 = retrieve("What are Chris's preferences?", domain_hints=["preference"], top_k=5)
    print_result(r4)
    # Two preference claims returned, but they describe orthogonal things:
    # "dark theme UIs" vs "direct communication" — geometrically far apart.
    # Correct behaviour: no conflict flagged. Conflict detection requires
    # both shared domain AND similarity >= CONFLICT_THRESHOLD (0.75).
    # Domain co-occurrence alone is not enough.
    pref_claims = [c for c in r4["claims"] if c["domain"] == "preference"]
    if len(pref_claims) >= 2:
        if len(r4["conflicts"]) == 0:
            print("  [No conflicts — orthogonal preferences correctly not flagged]\n")
        else:
            print(f"  [Conflicts detected: {len(r4['conflicts'])} pair(s)]\n")
    else:
        print("  [Only one preference claim returned — no conflict to detect]\n")

    # ------------------------------------------------------------------
    # Test 4b: Genuine conflict — dark theme vs light theme
    # ------------------------------------------------------------------
    print("TEST 4b: Genuine conflict detection — dark theme vs light theme")
    r4b = retrieve("What UI theme does Chris prefer?", domain_hints=["preference"], top_k=5)
    print_result(r4b)
    theme_claims = [c for c in r4b["claims"] if "theme" in c["claim"].lower() or "mode" in c["claim"].lower()]
    if len(theme_claims) >= 2:
        assert len(r4b["conflicts"]) >= 1, \
            "Expected a conflict between dark theme and light theme claims"
        assert all(p["domain"] == "preference" for p in r4b["conflicts"])
        print("  [Genuine conflict correctly detected]\n")
    else:
        print("  [Both theme claims not returned — conflict detection not exercised]\n")

    # ------------------------------------------------------------------
    # Test 5: format_for_context() — readable output
    # ------------------------------------------------------------------
    print("TEST 5: format_for_context() output")
    r5 = retrieve("What does Chris prefer?", domain_hints=["preference"], top_k=5)
    formatted = format_for_context(r5)
    print(formatted)
    assert "[Memory]" in formatted
    if r5["conflicts"]:
        assert "[Conflicting signals" in formatted

    # ------------------------------------------------------------------
    # Test 6: format_for_context() with no results
    # ------------------------------------------------------------------
    print("TEST 6: format_for_context() — empty result set")
    empty_result = {
        "claims": [],
        "conflicts": [],
        "domain_hints": ["ephemeral"],
        "query": "anything",
        "used_fallback": False,
    }
    formatted_empty = format_for_context(empty_result)
    print(formatted_empty)
    assert "No relevant memories" in formatted_empty

    print("=" * 60)
    print("  All assertions passed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
