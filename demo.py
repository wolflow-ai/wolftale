"""
wolftale.demo
-------------
End-to-end REPL demo. All six layers running together:

  Gate → Extractor → Store → Retrieval → Format → Generate

Type a conversation turn. The demo shows:
  1. Gate decision — extract / edge / skip
  2. Extracted claim (if any) and store result
  3. Retrieved memories relevant to your turn
  4. Claude's response, informed by those memories

This is not a chatbot. It is a pipeline visibility tool.
Each turn is processed independently — there is no multi-turn
conversation history sent to Claude. The memory system IS the
persistence mechanism. What Claude knows about you comes from
Wolftale, not from a rolling context window.

System prompt design:
  - Does not assume the user's name or any prior facts
  - Instructs Claude to use the [Memory] block if relevant
  - Instructs Claude to be honest when memory is empty or irrelevant
  - The user establishes their own identity through conversation

Domain hint inference:
  - Lightweight keyword matching against Wolftale's domain taxonomy
  - Not Wolfprompt — purpose-built for the assertion signal problem
  - Placeholder for semantic routing when that layer is ready
  - Infers 1-2 likely domains from the query; falls back to broad search

Commands:
  quit / exit / q   — exit the demo
  /store            — show all claims currently in the store
  /clear            — wipe the store (fresh start)
  /help             — show commands
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import anthropic

from wolftale.config import ANTHROPIC_API_KEY
from wolftale.gate import evaluate
from wolftale.extractor import extract
from wolftale import store
from wolftale.retrieval import retrieve, format_for_context

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a personal assistant with access to a memory system.

Before each response, you receive a [Memory] block containing facts about the user that have been extracted from prior conversations. These are things the user has told you — preferences, identity facts, commitments, technical context.

Use the memory block to inform your response when it is relevant. If the memory contains conflicting signals (marked as [Conflicting signals]), acknowledge the conflict honestly rather than picking one arbitrarily.

If the memory block is empty or irrelevant to the question, say so and answer as best you can from the conversation.

Do not invent facts about the user that are not in the memory block. Do not assume you know the user's name unless it appears in memory.

Be direct. Be useful. Be honest about what you know and don't know."""

# ---------------------------------------------------------------------------
# Domain hint inference
# ---------------------------------------------------------------------------
# Purpose-built for Wolftale's domain taxonomy.
# Not Wolfprompt — different problem, different vocabulary.
# Keyword matching is the wedge; semantic routing is the proper solution.

_DOMAIN_KEYWORDS = {
    "preference": [
        "prefer", "like", "love", "hate", "enjoy", "avoid", "style",
        "theme", "color", "format", "tone", "approach", "habit",
    ],
    "identity": [
        "who", "where", "live", "based", "from", "work", "job", "role",
        "company", "name", "background", "career", "team",
    ],
    "technical": [
        "use", "build", "code", "language", "tool", "library", "framework",
        "python", "gradio", "chroma", "model", "api", "stack", "version",
    ],
    "commitment": [
        "plan", "will", "going to", "intend", "follow up", "deadline",
        "meeting", "schedule", "next", "friday", "tomorrow", "week",
    ],
    "relational": [
        "team", "colleague", "manager", "client", "friend", "partner",
        "marcus", "report", "stakeholder",
    ],
}


def _infer_domain_hints(query: str) -> list:
    """
    Infer likely domains from a query string using keyword matching.

    Returns up to 2 domains. If nothing matches, returns [] (broad search).
    Caller (retrieve()) handles the broad fallback.
    """
    query_lower = query.lower()
    scores = {}

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in query_lower)
        if hits > 0:
            scores[domain] = hits

    # Return top 2 by hit count
    ranked = sorted(scores, key=lambda d: scores[d], reverse=True)
    return ranked[:2]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(turn: str) -> dict:
    """
    Run a single turn through the full Wolftale pipeline.

    Returns a dict with all intermediate results for display.
    """
    # Step 1: Gate
    gate_result = evaluate(turn)

    # Step 2: Extract + Store (only if gate says to)
    extraction_result = None
    store_result = None

    if gate_result["decision"] in ("extract", "edge"):
        extraction_result = extract(turn, gate_result)
        if extraction_result["success"]:
            store_result = store.write(extraction_result["claim"])

    # Step 3: Retrieve
    domain_hints = _infer_domain_hints(turn)
    retrieval_result = retrieve(turn, domain_hints=domain_hints or None)

    # Step 4: Format memory block
    memory_block = format_for_context(retrieval_result)

    # Step 5: Generate
    user_message = f"{memory_block}\nUser: {turn}"

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    assistant_reply = response.content[0].text.strip()

    return {
        "gate": gate_result,
        "extraction": extraction_result,
        "store": store_result,
        "retrieval": retrieval_result,
        "memory_block": memory_block,
        "reply": assistant_reply,
        "domain_hints": domain_hints,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

GRAY   = "\033[90m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _print_pipeline(result: dict) -> None:
    gate = result["gate"]
    extraction = result["extraction"]
    store_res = result["store"]
    retrieval = result["retrieval"]

    # Gate
    decision_colors = {"extract": GREEN, "edge": YELLOW, "skip": GRAY}
    color = decision_colors.get(gate["decision"], RESET)
    print(f"\n{GRAY}── Gate ─────────────────────────────────────────{RESET}")
    print(f"  Decision : {color}{gate['decision'].upper()}{RESET}")
    print(f"  Reason   : {gate['reason']}")

    # Extraction + Store
    if extraction is not None:
        print(f"\n{GRAY}── Extractor ────────────────────────────────────{RESET}")
        if extraction["success"]:
            claim = extraction["claim"]
            print(f"  Claim    : {claim['claim']}")
            print(f"  Domain   : {claim['domain']}  Confidence: {claim['confidence']:.2f}  Path: {claim['extraction_path']}")
            if store_res:
                action_colors = {
                    "stored": GREEN, "deduplicated": GRAY,
                    "flagged": YELLOW, "superseded": CYAN,
                }
                sc = action_colors.get(store_res["action"], RESET)
                print(f"  Store    : {sc}{store_res['action'].upper()}{RESET} — {store_res['reason']}")
        else:
            print(f"  {GRAY}Extraction failed — gate said extract but model found nothing storable{RESET}")

    # Retrieval
    print(f"\n{GRAY}── Retrieval ────────────────────────────────────{RESET}")
    hints_display = result["domain_hints"] or ["(broad)"]
    print(f"  Hints    : {hints_display}  Fallback: {retrieval['used_fallback']}")
    if retrieval["claims"]:
        for c in retrieval["claims"]:
            print(f"  Memory   : [{c['domain']}] {c['claim']}")
    else:
        print(f"  {GRAY}No memories retrieved{RESET}")
    if retrieval["conflicts"]:
        for pair in retrieval["conflicts"]:
            print(f"  {YELLOW}Conflict : [{pair['domain']}] '{pair['claim_a']['claim'][:45]}' vs '{pair['claim_b']['claim'][:45]}'{RESET}")

    # Response
    print(f"\n{GRAY}── Response ─────────────────────────────────────{RESET}")
    print(f"{BOLD}{result['reply']}{RESET}\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_store() -> None:
    """Show all claims currently in the store."""
    claims = store.all_claims()
    print(f"\n{GRAY}── Store ({len(claims)} claims) ──────────────────────────{RESET}")
    if not claims:
        print(f"  {GRAY}Empty{RESET}")
    else:
        for c in claims:
            flag = f"{YELLOW}[flagged]{RESET} " if c.get("extraction_path") == "edge" else ""
            print(f"  {flag}[{c['domain']}] {c['claim']}  {GRAY}(conf: {c['confidence']:.2f}){RESET}")
    print()


def _cmd_clear() -> None:
    """Wipe the store."""
    import wolftale.store as store_module
    coll = store._get_collection()
    client = coll._client
    try:
        client.delete_collection(store_module.COLLECTION_NAME)
    except Exception:
        pass
    store_module._collection = None
    print(f"\n  {GREEN}Store cleared.{RESET}\n")


def _cmd_help() -> None:
    print(f"""
{GRAY}Commands:{RESET}
  /store    — show all claims in the store
  /clear    — wipe the store (fresh start)
  /help     — show this message
  quit / exit / q — exit
""")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def run() -> None:
    print(f"""
{BOLD}Wolftale Demo{RESET}
Personal memory layer — full pipeline

Gate → Extractor → Store → Retrieval → Generate

Type anything. The system will remember what matters.
Type /help for commands. Type quit to exit.
""")

    turn_index = 0

    while True:
        try:
            raw = input(f"{CYAN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q"):
            print("  Exiting.")
            break

        if raw == "/store":
            _cmd_store()
            continue

        if raw == "/clear":
            _cmd_clear()
            continue

        if raw == "/help":
            _cmd_help()
            continue

        try:
            result = _run_pipeline(raw)
            _print_pipeline(result)
        except Exception as e:
            print(f"\n  {RED}Pipeline error: {e}{RESET}\n")

        turn_index += 1


if __name__ == "__main__":
    run()
