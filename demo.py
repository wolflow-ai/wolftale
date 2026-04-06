"""
wolftale.demo
-------------
Full pipeline REPL: gate → extractor → store → retrieval → Claude generation.

Not a chatbot. No rolling conversation history.
Memory IS the persistence mechanism.

Startup
-------
    python demo.py

The store loads explicitly from STORE_PATH on startup.
On clean exit (/quit) or Ctrl+C, the store saves back to disk.
The card is only touched at those two moments. Everything between is RAM.

Commands
--------
    /store    show all claims in the store
    /clear    wipe the store and start fresh (asks for confirmation)
    /save     save the store to disk mid-session
    /help     show commands
    /quit     save and exit
"""

import atexit
import signal
import sys
import os

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anthropic
from wolftale import store, gate, extractor, retrieval
from wolftale.config import ANTHROPIC_API_KEY

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STORE_PATH   = "wolftale_store"
DEMO_MODEL   = "claude-sonnet-4-20250514"
MAX_TOKENS   = 1024

# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ---------------------------------------------------------------------------
# System prompt — does not assume user identity.
# The user establishes their own name through conversation;
# the extractor stores it as a ClaimRecord.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a thoughtful AI assistant with access to a structured memory layer.

Before each response, you will receive a [Memory] block containing facts, preferences, and commitments the user has established in prior conversations. Use these to personalize your responses. If conflicting signals appear, acknowledge the tension and ask for clarification rather than silently picking one.

You do not know who the user is until they tell you. Do not assume a name or identity.

Keep responses clear and direct. When you draw on a memory, do so naturally — not mechanically.
"""

# ---------------------------------------------------------------------------
# Startup and shutdown
# ---------------------------------------------------------------------------

def _startup() -> None:
    """Load the store and register shutdown handlers."""
    print("\n  Wolftale")
    print("  ─────────────────────────────────────")
    print(f"  Loading store from: {STORE_PATH}")
    store.load(STORE_PATH)
    n = store.count()
    model = store._meta.get("embedding_model", "unknown")
    print(f"  {n} claim{'s' if n != 1 else ''} loaded  ·  model: {model}")
    print("  Type /help for commands. Ctrl+C to exit safely.")
    print("  ─────────────────────────────────────\n")

    # Register save on normal interpreter exit.
    atexit.register(_shutdown)

    # Register save on Ctrl+C (SIGINT).
    signal.signal(signal.SIGINT, _handle_sigint)


def _shutdown() -> None:
    """Save the store to disk. Called on clean exit and Ctrl+C."""
    print("\n  Saving store...")
    try:
        store.save()
        n = store.count()
        print(f"  Saved. {n} claim{'s' if n != 1 else ''} persisted to {STORE_PATH}.")
    except Exception as e:
        print(f"  Save failed: {e}")


def _handle_sigint(sig, frame) -> None:
    """Handle Ctrl+C gracefully — save then exit."""
    # atexit handlers run on sys.exit(), so _shutdown() fires automatically.
    print()  # newline after ^C
    sys.exit(0)


# ---------------------------------------------------------------------------
# Domain hint inference
# Lightweight keyword matching against Wolftale's domain taxonomy.
# Placeholder for semantic routing — deliberately simple.
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS = {
    "preference":  ["prefer", "like", "love", "hate", "enjoy", "theme", "style",
                    "approach", "habit", "tend", "usually", "always", "never"],
    "identity":    ["am", "name", "live", "based", "from", "work", "background",
                    "career", "role", "founder", "location", "city"],
    "commitment":  ["will", "plan", "going to", "intend", "follow up", "finish",
                    "deadline", "by friday", "next week", "today"],
    "technical":   ["python", "code", "tool", "library", "framework", "model",
                    "api", "database", "version", "stack", "language", "build"],
    "relational":  ["team", "colleague", "friend", "client", "partner", "manager",
                    "he said", "she said", "they", "our", "together"],
}

def _infer_domain_hints(text: str) -> list:
    text_lower = text.lower()
    hints = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            hints.append(domain)
    return hints[:2]   # cap at 2 domains per query


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(user_turn: str, memory_block: str) -> str:
    """
    Call Claude with the user's turn and a memory context block.
    Returns the response text.
    """
    messages = [
        {
            "role": "user",
            "content": f"{memory_block}\n\nUser: {user_turn}"
        }
    ]
    response = _client.messages.create(
        model      = DEMO_MODEL,
        max_tokens = MAX_TOKENS,
        system     = SYSTEM_PROMPT,
        messages   = messages,
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Pipeline — one turn
# ---------------------------------------------------------------------------

def _process_turn(turn: str, turn_index: int) -> None:
    """
    Run the full pipeline for a single user turn:
        gate → extractor → store → retrieval → generation → print
    """
    # 1. Gate
    gate_decision = gate.evaluate(turn, turn_index=turn_index)
    gate_label    = gate_decision["decision"].upper()

    # 2. Extractor (only if gate says extract or edge)
    store_action = None
    if gate_decision["decision"] in ("extract", "edge"):
        result = extractor.extract(turn, gate_decision, source_turn=turn_index)
        if result["success"]:
            store_result = store.write(result["claim"])
            store_action = store_result["action"]

    # 3. Retrieval
    hints        = _infer_domain_hints(turn)
    ret_result   = retrieval.retrieve(turn, domain_hints=hints or None, top_k=3)
    memory_block = retrieval.format_for_context(ret_result)

    # 4. Generation
    response = _generate(turn, memory_block)

    # 5. Print response
    print(f"\n  Assistant: {response}\n")

    # 6. Trace line (compact)
    flags = []
    if gate_label != "SKIP":
        flags.append(f"gate:{gate_label}")
    if store_action:
        flags.append(f"stored:{store_action}")
    if ret_result["conflicts"]:
        flags.append(f"conflicts:{len(ret_result['conflicts'])}")
    if flags:
        print(f"  [{' · '.join(flags)}]\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_store() -> None:
    claims = store.all_claims()
    if not claims:
        print("\n  Store is empty.\n")
        return
    print(f"\n  Store — {len(claims)} claim{'s' if len(claims) != 1 else ''}:")
    for c in claims:
        conf = f"{c['confidence']:.2f}"
        print(f"    [{c['domain']}] ({conf}) {c['claim']}")
    print()


def _cmd_clear() -> None:
    confirm = input("  Clear all claims? This cannot be undone. (yes/no): ").strip().lower()
    if confirm == "yes":
        store.load.__globals__['_claims'].clear()
        # Reload as empty store to reset index and derived structures.
        import shutil
        store_path = store._path
        shutil.rmtree(store_path, ignore_errors=True)
        store.load(store_path)
        print("  Store cleared.\n")
    else:
        print("  Cancelled.\n")


def _cmd_save() -> None:
    store.save()
    print(f"  Saved. {store.count()} claims persisted.\n")


def _cmd_help() -> None:
    print("""
  Commands:
    /store   — show all stored claims
    /clear   — wipe the store (asks for confirmation)
    /save    — save to disk now
    /help    — show this
    /quit    — save and exit
    Ctrl+C   — save and exit
""")


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def main() -> None:
    _startup()
    turn_index = 0

    while True:
        try:
            user_input = input("  You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() == "/quit":
            break
        elif user_input.lower() == "/store":
            _cmd_store()
            continue
        elif user_input.lower() == "/clear":
            _cmd_clear()
            continue
        elif user_input.lower() == "/save":
            _cmd_save()
            continue
        elif user_input.lower() == "/help":
            _cmd_help()
            continue

        _process_turn(user_input, turn_index)
        turn_index += 1

    # Clean exit — atexit fires _shutdown() automatically.
    sys.exit(0)


if __name__ == "__main__":
    main()
