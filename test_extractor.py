"""
test_extractor.py
-----------------
Tests the gate + extractor pipeline end to end.

Run from repo root with venv activated:
    python test_extractor.py

This makes real API calls to Claude (Haiku).
Each test case costs a small number of tokens.
Expect ~1-2 seconds per case.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wolftale.gate import evaluate
from wolftale.extractor import extract


def print_result(turn: str, gate_decision: dict, result: dict) -> None:
    print(f"\n  Turn     : {turn}")
    print(f"  Gate     : {gate_decision['decision'].upper()} — {gate_decision['reason']}")

    if not result["success"]:
        print(f"  Extract  : FAILED — {result['raw_response']}")
        return

    claim = result["claim"]
    print(f"  Claim    : {claim['claim']}")
    print(f"  Domain   : {claim['domain']}")
    print(f"  Confidence: {claim['confidence']}")
    print(f"  Path     : {claim['extraction_path']}")
    print(f"  ID       : {claim['id']}")


def run_tests():
    print("\n" + "="*60)
    print("  WOLFTALE — Gate + Extractor Pipeline Tests")
    print("="*60)

    test_cases = [
        "I prefer dark theme UIs",
        "I'm based in St. Louis",
        "I've been building under the Wolflow brand",
        "I'll follow up with Marcus on Friday",
        "Remember this — I work at Wolflow",
        "I tend to work better late at night",
        "I use Python 3.10 for all my projects",
        "My project is called Wolftale",
    ]

    skipped = 0
    extracted = 0

    for i, turn in enumerate(test_cases):
        gate = evaluate(turn, turn_index=i)

        if gate["decision"] == "skip":
            print(f"\n  Turn     : {turn}")
            print(f"  Gate     : SKIP — {gate['reason']}")
            skipped += 1
            continue

        result = extract(turn, gate, source_turn=i)
        print_result(turn, gate, result)
        if result["success"]:
            extracted += 1

    print(f"\n{'='*60}")
    print(f"  Results: {extracted} extracted, {skipped} skipped, "
          f"{len(test_cases) - extracted - skipped} failed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_tests()
