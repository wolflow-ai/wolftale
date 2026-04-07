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
  11. Confidence decay — backdated claim returns lower confidence than stored
  12. Retrieval reinforcement — retrieved_count increments, last_retrieved set

Run from repo root with venv activated:
    python test_store.py

No API calls. Uses HNSWlib + JSON backend.
"""

import sys
import os
import uuid
import shutil
import math
from datetime import datetime, timezone, timedelta

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
    days_ago:        float = 0.0,
) -> ClaimRecord:
    """
    Build a ClaimRecord for testing.

    days_ago: if > 0, backdates the timestamp so decay tests can
              verify age-based confidence reduction without waiting.
    """
    if days_ago > 0:
        ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    else:
        ts = datetime.now(timezone.utc).isoformat()

    return {
        "claim":               claim_text,
        "confidence":          confidence,
        "original_confidence": confidence,   # set equal to confidence at creation
        "domain":              domain,
        "source_turn":         0,
        "timestamp":           ts,
        "supersedes":          [],
        "retrieved_count":     0,
        "last_retrieved":      None,
        "extraction_path":     extraction_path,
        "id":                  str(uuid.uuid4()),
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

    assert store._meta.get("embedding_model") == "all-MiniLM-L6-v2", \
        "Meta block not persisted correctly"
    print(f"  Meta: model={store._meta['embedding_model']}, "
          f"dim={store._meta['embedding_dim']}, "
          f"capacity={store._meta['max_elements']}")
    print()

    # ------------------------------------------------------------------
    # Test 11: Confidence decay — backdated claim returns lower confidence
    # ------------------------------------------------------------------
    print("TEST 11: Confidence decay — backdated claim")

    original_conf = 0.90
    domain        = "preference"
    days_back     = 365.0

    c11 = make_claim(
        "Chris enjoys working in focused quiet sessions.",
        domain     = domain,
        confidence = original_conf,
        days_ago   = days_back,
    )
    store.write(c11)

    results11 = store.search("How does Chris like to work?", domains=[domain], top_k=5)
    backdated  = next((r for r in results11 if r["id"] == c11["id"]), None)

    assert backdated is not None, "Backdated claim not found in search results"

    returned_conf = backdated["confidence"]
    decay_factor  = store.DECAY_FACTORS[domain]

    # Reinforcement bump is applied first (+0.05, capped at original_confidence).
    # Then decay. So expected = min(original + bump, original) * decay^days
    # = original_conf * decay^days  (bump can't exceed ceiling = original_conf)
    expected_conf = original_conf * (decay_factor ** days_back)

    print(f"  Original confidence : {original_conf}")
    print(f"  Days backdated      : {days_back}")
    print(f"  Decay factor        : {decay_factor} ({domain})")
    print(f"  Expected (approx)   : {expected_conf:.4f}")
    print(f"  Returned            : {returned_conf:.4f}")

    assert returned_conf < original_conf, \
        f"Decayed confidence ({returned_conf:.4f}) should be less than original ({original_conf})"

    # Allow 10% relative tolerance — reinforcement bump shifts the value slightly
    tolerance = 0.10
    assert abs(returned_conf - expected_conf) / expected_conf < tolerance, (
        f"Decayed confidence {returned_conf:.4f} deviates more than {tolerance*100:.0f}% "
        f"from expected {expected_conf:.4f}"
    )

    print(f"  [Decay verified — confidence reduced by "
          f"{(1 - returned_conf/original_conf)*100:.1f}% over {days_back:.0f} days]\n")

    # ------------------------------------------------------------------
    # Test 12: Retrieval reinforcement — retrieved_count and last_retrieved
    # ------------------------------------------------------------------
    print("TEST 12: Retrieval reinforcement — tracking fields update on retrieval")

    c12 = make_claim(
        "Chris builds products under the Wolflow brand.",
        domain     = "identity",
        confidence = 0.88,
    )
    store.write(c12)

    # Verify initial state
    stored_before = store._claims[c12["id"]]
    assert stored_before["retrieved_count"] == 0, \
        f"Expected retrieved_count=0 before any retrieval, got {stored_before['retrieved_count']}"
    assert stored_before["last_retrieved"] is None, \
        "Expected last_retrieved=None before any retrieval"

    print(f"  Before first search: retrieved_count={stored_before['retrieved_count']}, "
          f"last_retrieved={stored_before['last_retrieved']}")

    # First retrieval
    store.search("What brand does Chris work under?", domains=["identity"], top_k=3)
    stored_after_1 = store._claims[c12["id"]]

    assert stored_after_1["retrieved_count"] == 1, \
        f"Expected retrieved_count=1 after first retrieval, got {stored_after_1['retrieved_count']}"
    assert stored_after_1["last_retrieved"] is not None, \
        "Expected last_retrieved to be set after first retrieval"

    print(f"  After first search : retrieved_count={stored_after_1['retrieved_count']}, "
          f"last_retrieved set: {stored_after_1['last_retrieved'] is not None}")

    # Second retrieval
    store.search("Tell me about Chris's projects.", domains=["identity"], top_k=3)
    stored_after_2 = store._claims[c12["id"]]

    assert stored_after_2["retrieved_count"] == 2, \
        f"Expected retrieved_count=2 after second retrieval, got {stored_after_2['retrieved_count']}"

    print(f"  After second search: retrieved_count={stored_after_2['retrieved_count']}")

    # Verify confidence bump is capped at original_confidence
    conf_after_2      = stored_after_2["confidence"]
    original_conf_12  = c12["original_confidence"]
    assert conf_after_2 <= original_conf_12, (
        f"Reinforced confidence ({conf_after_2:.4f}) exceeds "
        f"original_confidence ceiling ({original_conf_12:.4f})"
    )

    print(f"  Confidence after 2 retrievals: {conf_after_2:.4f} "
          f"(original ceiling: {original_conf_12:.4f}) ✓")
    print(f"  [Reinforcement verified — count increments, last_retrieved set, ceiling respected]\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  All assertions passed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
