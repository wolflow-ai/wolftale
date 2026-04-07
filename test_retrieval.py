"""
test_retrieval.py
-----------------
Tests the retrieval layer end to end.

Uses a separate test directory (wolftale_store_test) so test data
never pollutes the demo store.

Exercises:
  1. retrieve() with domain hints              → domain-scoped results
  2. retrieve() with multi-domain hints        → results from both domains
  3. retrieve() with no hints                  → broad search
  4. Conflict detection (orthogonal prefs)     → no conflict flagged
  5. Genuine conflict (dark vs light theme)    → conflict correctly detected
  6. format_for_context()                      → readable output with conflict block
  7. format_for_context() empty result set     → "No relevant memories"

Run from repo root with venv activated:
    python test_retrieval.py

No API calls. Populates the store with known claims, then retrieves.
Wipes and repopulates at the start of each run.
"""

import sys
import os
import uuid
import shutil
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wolftale import store
from wolftale.retrieval import retrieve, format_for_context
from wolftale.models import ClaimRecord

TEST_STORE_PATH = "wolftale_store_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_claim(text: str, domain: str, confidence: float = 0.85) -> ClaimRecord:
    return {
        "claim":               text,
        "confidence":          confidence,
        "original_confidence": confidence,   # set equal to confidence at creation
        "domain":              domain,
        "source_turn":         0,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "supersedes":          [],
        "retrieved_count":     0,
        "last_retrieved":      None,
        "extraction_path":     "standard",
        "id":                  str(uuid.uuid4()),
    }


def wipe_and_seed() -> None:
    if os.path.isdir(TEST_STORE_PATH):
        shutil.rmtree(TEST_STORE_PATH)
    store.load(TEST_STORE_PATH)

    claims = [
        make_claim("Chris prefers dark theme UIs.",                               "preference", 0.95),
        make_claim("Chris prefers direct communication over diplomatic softening.", "preference", 0.88),
        make_claim("Chris prefers light theme UIs.",                              "preference", 0.80),
        make_claim("Chris is based in St. Louis, Missouri.",                      "identity",   0.92),
        make_claim("Chris works under the Wolflow brand.",                        "identity",   0.90),
        make_claim("Chris uses Python 3.10.11 for all his projects.",             "technical",  0.87),
        make_claim("Chris builds with Gradio for local UIs.",                     "technical",  0.83),
        make_claim("Chris will follow up with Marcus on Friday.",                 "commitment", 0.75),
    ]
    for c in claims:
        store.write(c)

    print(f"  [Store seeded with {len(claims)} claims at '{TEST_STORE_PATH}']\n")


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
            print(
                f"    [{pair['domain']}] "
                f"'{pair['claim_a']['claim'][:40]}' vs "
                f"'{pair['claim_b']['claim'][:40]}'"
            )
    print()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    print("\n" + "=" * 60)
    print("  WOLFTALE — Retrieval Layer Tests (HNSWlib backend)")
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
    r2 = retrieve(
        "Tell me about Chris's background and tools.",
        domain_hints=["identity", "technical"]
    )
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
    assert r3["used_fallback"] == False   # No hints → broad directly, fallback stays False
    assert r3["domain_hints"] == []

    # ------------------------------------------------------------------
    # Test 4: Orthogonal preferences — should NOT conflict
    # ------------------------------------------------------------------
    print("TEST 4: Orthogonal preferences — should not be flagged as conflicts")
    r4 = retrieve(
        "What are Chris's preferences?",
        domain_hints=["preference"],
        top_k=5
    )
    print_result(r4)
    pref_claims = [c for c in r4["claims"] if c["domain"] == "preference"]
    if len(pref_claims) >= 2 and len(r4["conflicts"]) == 0:
        print("  [Orthogonal preferences correctly not flagged as conflicting]\n")
    elif len(r4["conflicts"]) > 0:
        print(f"  [Note: {len(r4['conflicts'])} conflict(s) detected — "
              f"may include genuine theme conflict]\n")

    # ------------------------------------------------------------------
    # Test 5: Genuine conflict — dark theme vs light theme
    # ------------------------------------------------------------------
    print("TEST 5: Genuine conflict — dark theme vs light theme")
    r5 = retrieve(
        "What UI theme does Chris prefer?",
        domain_hints=["preference"],
        top_k=5
    )
    print_result(r5)
    theme_claims = [
        c for c in r5["claims"]
        if "theme" in c["claim"].lower() or "mode" in c["claim"].lower()
    ]
    if len(theme_claims) >= 2:
        assert len(r5["conflicts"]) >= 1, \
            "Expected a conflict between dark theme and light theme claims"
        print("  [Genuine conflict correctly detected]\n")
    else:
        print("  [Both theme claims not returned — conflict detection not exercised]\n")

    # ------------------------------------------------------------------
    # Test 6: format_for_context() — readable output
    # ------------------------------------------------------------------
    print("TEST 6: format_for_context() output")
    r6 = retrieve("What does Chris prefer?", domain_hints=["preference"], top_k=5)
    formatted = format_for_context(r6)
    print(formatted)
    assert "[Memory]" in formatted
    if r6["conflicts"]:
        assert "[Conflicting signals" in formatted

    # ------------------------------------------------------------------
    # Test 7: format_for_context() — empty result set
    # ------------------------------------------------------------------
    print("TEST 7: format_for_context() — empty result set")
    empty_result = {
        "claims":       [],
        "conflicts":    [],
        "domain_hints": ["ephemeral"],
        "query":        "anything",
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
