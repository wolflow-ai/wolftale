"""
test_gate.py
------------
Simple REPL test for the Wolftale gate layer.

Run from repo root with venv activated:
    python test_gate.py

Tests the five main decision branches:
  - preference  → extract
  - question    → skip
  - commitment  → extract
  - continuation → skip
  - high-stakes → edge
  
Then drops into an interactive loop so you can type your own turns
and see what the gate decides.
"""

import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wolftale.gate import evaluate


def print_decision(turn: str, result: dict) -> None:
    decision = result["decision"].upper()
    reason = result["reason"]
    signals = result["signals"]

    # Color coding via ANSI — works in PowerShell and most terminals
    colors = {"EXTRACT": "\033[92m", "EDGE": "\033[93m", "SKIP": "\033[90m"}
    reset = "\033[0m"
    color = colors.get(decision, "")

    print(f"\n  Turn    : {turn}")
    print(f"  Decision: {color}{decision}{reset}")
    print(f"  Reason  : {reason}")
    if signals:
        print(f"  Signals : {[s['signal_type'] + ':' + s['value'] for s in signals]}")
    else:
        print(f"  Signals : none")
    print()


def run_smoke_tests() -> None:
    print("\n" + "="*60)
    print("  WOLFTALE — Gate Layer Smoke Tests")
    print("="*60)

    test_cases = [
        ("I prefer dark theme UIs",                     "expect: EXTRACT — preference"),
        ("What time is it?",                            "expect: SKIP    — question, no signal"),
        ("I'll follow up with Marcus on Friday",        "expect: EXTRACT — commitment"),
        ("Continue from where we left off",             "expect: SKIP    — continuation"),
        ("Remember this — I work at Wolflow",           "expect: EDGE    — high-stakes"),
        ("I'm based in St. Louis",                      "expect: EXTRACT — assertion"),
        ("I tend to work better late at night",         "expect: EXTRACT — preference"),
        ("Yes",                                         "expect: SKIP    — bare continuation"),
        ("I need to finish the proposal by Friday",     "expect: EXTRACT — commitment"),
        ("I've been building under the Wolflow brand",  "expect: EXTRACT — assertion"),
    ]

    passed = 0
    for turn, note in test_cases:
        result = evaluate(turn)
        print_decision(f"{turn}  [{note}]", result)
        passed += 1

    print(f"  {passed}/{len(test_cases)} cases ran without error.\n")


def run_interactive() -> None:
    print("="*60)
    print("  Interactive mode — type a conversation turn.")
    print("  Gate will tell you what it decides.")
    print("  Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")

    turn_index = 0
    while True:
        try:
            turn = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if turn.lower() in ("quit", "exit", "q"):
            print("  Exiting.")
            break

        if not turn:
            continue

        result = evaluate(turn, turn_index=turn_index)
        print_decision(turn, result)
        turn_index += 1


if __name__ == "__main__":
    run_smoke_tests()
    run_interactive()
