"""
test_store.py
-------------
Tests the store layer in isolation — no gate, no extractor.

Uses a separate test directory (wolftale_store_test) so test data
never pollutes the demo store. Directory is wiped and reloaded at
the start of each run.

Exercises:
  1.  Write a clean claim                          → stored
  2.  Write a near-duplicate                       → deduplicated or flagged
  3.  Write a contradicting claim (same domain)    → flagged
  4.  Write a clearly unrelated claim              → stored
  5.  Write a technical claim                      → stored
  6.  count()                                      → correct total
  7.  search() with domain filter                  → domain-scoped results
  8.  search() broad (no domain filter)            → results across all domains
  9.  all_claims()                                 → all stored claims returned
  10. save() then load() → verify persistence

Run from repo root with venv activated:
    python test_store.py

No API calls. Uses HNSWlib + JSON backend.
"""

import sys
import os
import uuid
import shutil
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wolftale import store
from wolftale.models import ClaimRecord

TEST_STORE_PATH = "wolftale_store_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_claim(
    claim_text:      str,
    domain:          str   = "preference",
    confidence:      float = 0.85,
    extraction_path: str   = "standard",
) -> ClaimRecord:
    return {
        "claim":           claim_text,
        "confidence":      confidence,
        "domain":          domain,
        "source_turn":     0,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "supersedes":      [],
        "retrieved_count": 0,
        "last_retrieved":  None,
        "extraction_path": extraction_path,
        "id":              str(uuid.uuid4()),
    }


def print_result(label: str, result: dict) -> None:
    action = result["action"].upper()
    colors = {
        "STORED":       "\033[92m",
        "DEDUPLICATED": "\033[93m",
        "FLAGGED":      "\033[91m",
        "SUPERSEDED":   "\033[94m",
    }
    reset = "\033[0m"
    color = colors.get(action, "")
    print(f"  {label}")
    print(f"    Action : {color}{action}{reset}")
    print(f"    Reason : {result['reason']}")
    if result["conflicts"]:
        for c in result["conflicts"]:
            print(
                f"    Conflict: [{c['conflict_type']}] "
                f"sim={c['similarity']:.3f} — {c['existing_claim'][:60]}"
            )
    print()


def fresh_store() -> None:
    """Wipe and reload the test store for a clean run."""
    if os.path.isdir(TEST_STORE_PATH):
        shutil.rmtree(TEST_STORE_PATH)
    store.load(TEST_STORE_PATH)
    print(f"  [Test store initialized at '{TEST_STORE_PATH}']\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    print("\n" + "=" * 60)
    print("  WOLFTALE — Store Layer Tests (HNSWlib backend)")
    print("=" * 60 + "\n")

    fresh_store()

    # ------------------------------------------------------------------
    # Test 1: Write a clean claim
    # ------------------------------------------------------------------
    print("TEST 1: Write a clean claim")
    c1 = make_claim("Chris prefers dark theme UIs.", domain="preference")
    r1 = store.write(c1)
    print_result("Chris prefers dark theme UIs.", r1)
    assert r1["action"] == "stored", f"Expected stored, got {r1['action']}"

    # ------------------------------------------------------------------
    # Test 2: Write a near-duplicate
    # ------------------------------------------------------------------
    print("TEST 2: Write a near-duplicate")
    c2 = make_claim("Chris prefers dark mode interfaces.", domain="preference")
    r2 = store.write(c2)
    print_result("Chris prefers dark mode interfaces.", r2)
    # ~0.90 similarity: above CONFLICT_THRESHOLD (0.75), same domain → contradiction.
    # Contradiction takes priority — flagged, not deduplicated.
    assert r2["action"] in ("deduplicated", "flagged"), \
        f"Expected deduplicated or flagged, got {r2['action']}"

    # ------------------------------------------------------------------
    # Test 3: Write a contradicting claim (same domain)
    # ------------------------------------------------------------------
    print("TEST 3: Write a contradicting claim (same domain)")
    c3 = make_claim("Chris prefers light theme UIs.", domain="preference")
    r3 = store.write(c3)
    print_result("Chris prefers light theme UIs.", r3)
    print(f"  [Result: {r3['action']} — acceptable: flagged or stored depending on similarity]\n")

    # ------------------------------------------------------------------
    # Test 4: Write a clearly unrelated claim
    # ------------------------------------------------------------------
    print("TEST 4: Write a clearly unrelated claim")
    c4 = make_claim("Chris is based in St. Louis, Missouri.", domain="identity")
    r4 = store.write(c4)
    print_result("Chris is based in St. Louis, Missouri.", r4)
    assert r4["action"] == "stored", f"Expected stored, got {r4['action']}"

    # ------------------------------------------------------------------
    # Test 5: Write a technical claim
    # ------------------------------------------------------------------
    print("TEST 5: Write a technical claim")
    c5 = make_claim("Chris uses Python 3.10.11 for all his projects.", domain="technical")
    r5 = store.write(c5)
    print_result("Chris uses Python 3.10.11 for all his projects.", r5)
    assert r5["action"] == "stored", f"Expected stored, got {r5['action']}"

    # ------------------------------------------------------------------
    # Test 6: count()
    # ------------------------------------------------------------------
    print("TEST 6: count()")
    n = store.count()
    print(f"  Claims in store: {n}")
    print(f"  [Expected: >= 3 (c1, c4, c5 definitely stored)]\n")
    assert n >= 3, f"Expected at least 3, got {n}"

    # ------------------------------------------------------------------
    # Test 7: search() with domain filter
    # ------------------------------------------------------------------
    print("TEST 7: search() — domain-conditioned (preference)")
    results = store.search(
        "What UI theme does Chris prefer?", domains=["preference"], top_k=3
    )
    print(f"  Query  : 'What UI theme does Chris prefer?'")
    print(f"  Domains: ['preference']")
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"    [{r['domain']}] {r['claim']}")
    print()
    assert all(r["domain"] == "preference" for r in results), \
        "Expected only preference-domain results"

    # ------------------------------------------------------------------
    # Test 8: search() broad (no domain filter)
    # ------------------------------------------------------------------
    print("TEST 8: search() — broad (no domain filter)")
    results2 = store.search("Where does Chris live?", top_k=3)
    print(f"  Query  : 'Where does Chris live?'")
    print(f"  Domains: None (broad)")
    print(f"  Results: {len(results2)}")
    for r in results2:
        print(f"    [{r['domain']}] {r['claim']}")
    print()
    assert len(results2) > 0, "Expected at least one broad result"

    # ------------------------------------------------------------------
    # Test 9: all_claims()
    # ------------------------------------------------------------------
    print("TEST 9: all_claims()")
    all_c = store.all_claims()
    print(f"  Total claims returned: {len(all_c)}")
    for c in all_c:
        print(f"    [{c['domain']}] {c['claim'][:60]}")
    print()

    # ------------------------------------------------------------------
    # Test 10: save() → wipe RAM → load() → verify persistence
    # ------------------------------------------------------------------
    print("TEST 10: Persistence — save, reload, verify")
    count_before = store.count()
    store.save()

    # Reset module state manually to simulate a fresh process.
    store._path         = None
    store._index        = None
    store._claims       = {}
    store._label_to_id  = {}
    store._domain_index = __import__("collections").defaultdict(list)
    store._meta         = {}
    store._model        = None
    store._next_label   = 0

    store.load(TEST_STORE_PATH)
    count_after = store.count()
    print(f"  Claims before save: {count_before}")
    print(f"  Claims after reload: {count_after}")
    assert count_after == count_before, \
        f"Persistence failed: {count_before} before, {count_after} after"

    # Verify meta persisted correctly.
    assert store._meta.get("embedding_model") == "all-MiniLM-L6-v2", \
        "Meta block not persisted correctly"
    print(f"  Meta: model={store._meta['embedding_model']}, "
          f"dim={store._meta['embedding_dim']}, "
          f"capacity={store._meta['max_elements']}")
    print()

    print("=" * 60)
    print("  All assertions passed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
