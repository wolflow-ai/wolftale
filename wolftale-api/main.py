"""
wolftale-api · main.py
----------------------
FastAPI backend for the Wolftale cloud demo.

Session model:
  - UUID token in client localStorage → passed as X-Session-ID header
  - Each session gets its own HNSWlib store under stores/{session_id}/
  - Stores older than SESSION_TTL_DAYS are wiped by a background cleanup task
  - No auth, no accounts — the session token IS the identity

Endpoints:
  POST /api/chat          — run one turn through the full pipeline
  GET  /api/memories      — return all claims for a session
  GET  /api/export        — download claims.json for a session
  DELETE /api/memories    — wipe a session's store
  GET  /api/health        — liveness check

CORS: configured for tale.wolflow.ai + localhost dev
"""

import os
import uuid
import shutil
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Wolftale pipeline imports
# We expect the wolftale package to be on PYTHONPATH (installed or symlinked)
# ---------------------------------------------------------------------------

from wolftale import store as wt_store
from wolftale import gate as wt_gate
from wolftale import extractor as wt_extractor
from wolftale import retrieval as wt_retrieval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION_TTL_DAYS   = 7          # wipe sessions older than this
STORES_DIR         = Path("stores")
DEMO_MODEL         = "claude-sonnet-4-20250514"
MAX_TOKENS         = 1024
CLEANUP_INTERVAL_S = 3600       # run cleanup every hour

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY not set")

_claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ---------------------------------------------------------------------------
# Session store registry
# One store module instance is not enough — we need per-session isolation.
# Solution: store state is module-level in wolftale.store, so we manage
# load/save ourselves and keep a simple in-memory set of loaded sessions.
# For a single-process deployment this is sufficient.
# ---------------------------------------------------------------------------

_loaded_sessions: set[str] = set()
_session_locks: dict[str, asyncio.Lock] = {}

def _session_path(session_id: str) -> str:
    return str(STORES_DIR / session_id)

def _get_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]

def _ensure_session(session_id: str) -> None:
    """Load the session store if not already loaded."""
    path = _session_path(session_id)
    wt_store.load(path)
    _loaded_sessions.add(session_id)

def _save_session(session_id: str) -> None:
    """Persist the current in-memory store state for a session."""
    wt_store.save()

# ---------------------------------------------------------------------------
# Domain hint inference (lightweight keyword matching)
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

def _infer_domain_hints(text: str) -> list[str]:
    text_lower = text.lower()
    hints = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            hints.append(domain)
    return hints[:2]

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a thoughtful AI assistant with access to a structured memory layer.

Before each response, you will receive a [Memory] block containing facts, preferences, and commitments the user has established in prior conversations. Use these to personalize your responses naturally — not mechanically. Don't say "according to my memory" — just respond as someone who knows you.

If conflicting signals appear in the memory block, acknowledge the tension and ask for clarification rather than silently picking one.

You do not know who the user is until they tell you. Do not assume a name or identity.

Keep responses clear, direct, and warm. Two to four sentences is usually enough unless the question warrants more depth."""

def _generate(turn: str, memory_block: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"{memory_block}\n\nUser: {turn}"
        }
    ]
    response = _claude.messages.create(
        model=DEMO_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text.strip()

# ---------------------------------------------------------------------------
# Session cleanup background task
# ---------------------------------------------------------------------------

async def _cleanup_old_sessions():
    """Remove store directories older than SESSION_TTL_DAYS."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_S)
        if not STORES_DIR.exists():
            continue
        cutoff = datetime.now(timezone.utc) - timedelta(days=SESSION_TTL_DAYS)
        for entry in STORES_DIR.iterdir():
            if not entry.is_dir():
                continue
            mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                _loaded_sessions.discard(entry.name)
                _session_locks.pop(entry.name, None)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    STORES_DIR.mkdir(exist_ok=True)
    task = asyncio.create_task(_cleanup_old_sessions())
    yield
    task.cancel()

app = FastAPI(title="Wolftale API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tale.wolflow.ai",
        "https://clewismessina.com",
        "http://localhost:5173",
        "http://localhost:4173",
    ],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    gate_decision: str          # "extract" | "edge" | "skip"
    gate_reason: str
    extracted: bool             # True if a claim was successfully extracted
    claim: dict | None          # The ClaimRecord if extracted, else None
    store_action: str | None    # "stored" | "deduplicated" | "flagged" | "superseded" | None
    memory_block: str           # The formatted memory context used for generation
    conflicts: list             # Any conflicts detected at retrieval time
    memory_count: int           # Total claims in this session's store

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_session(session_id: str | None) -> str:
    """Validate or reject a session ID. Returns the session ID."""
    if not session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header required")
    # Basic UUID format check — reject obviously bad values
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    return session_id

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    x_session_id: str | None = Header(default=None),
):
    session_id = _validate_session(x_session_id)
    turn = body.message.strip()
    if not turn:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    lock = _get_lock(session_id)
    async with lock:
        # Load session store (idempotent)
        _ensure_session(session_id)

        # 1. Gate
        gate_decision = wt_gate.evaluate(turn)

        # 2. Extract + store
        extracted = False
        claim_record = None
        store_action = None

        if gate_decision["decision"] in ("extract", "edge"):
            result = wt_extractor.extract(turn, gate_decision)
            if result["success"]:
                store_result = wt_store.write(result["claim"])
                extracted = True
                claim_record = result["claim"]
                store_action = store_result["action"]
                # Persist immediately after write
                _save_session(session_id)

        # 3. Retrieve
        hints = _infer_domain_hints(turn)
        ret_result = wt_retrieval.retrieve(turn, domain_hints=hints or None, top_k=3)
        memory_block = wt_retrieval.format_for_context(ret_result)

        # 4. Generate
        response_text = _generate(turn, memory_block)

        # Serialize claim for response (remove internal _label field)
        claim_out = None
        if claim_record:
            claim_out = {k: v for k, v in claim_record.items() if k != "_label"}

        return ChatResponse(
            response=response_text,
            gate_decision=gate_decision["decision"],
            gate_reason=gate_decision["reason"],
            extracted=extracted,
            claim=claim_out,
            store_action=store_action,
            memory_block=memory_block,
            conflicts=[
                {
                    "claim_a": pair["claim_a"]["claim"],
                    "claim_b": pair["claim_b"]["claim"],
                    "domain": pair["domain"],
                }
                for pair in ret_result["conflicts"]
            ],
            memory_count=wt_store.count(),
        )


@app.get("/api/memories")
async def get_memories(
    x_session_id: str | None = Header(default=None),
):
    session_id = _validate_session(x_session_id)
    lock = _get_lock(session_id)
    async with lock:
        _ensure_session(session_id)
        claims = wt_store.all_claims()
        # Strip internal _label field
        clean = [{k: v for k, v in c.items() if k != "_label"} for c in claims]
        return {"claims": clean, "count": len(clean)}


@app.get("/api/export")
async def export_memories(
    x_session_id: str | None = Header(default=None),
):
    """Return the raw claims.json for download."""
    session_id = _validate_session(x_session_id)
    lock = _get_lock(session_id)
    async with lock:
        _ensure_session(session_id)
        wt_store.save()
        claims_path = Path(_session_path(session_id)) / "claims.json"
        if not claims_path.exists():
            raise HTTPException(status_code=404, detail="No memories found for this session")
        return FileResponse(
            path=str(claims_path),
            media_type="application/json",
            filename="wolftale_memories.json",
        )


@app.delete("/api/memories")
async def clear_memories(
    x_session_id: str | None = Header(default=None),
):
    session_id = _validate_session(x_session_id)
    lock = _get_lock(session_id)
    async with lock:
        path = _session_path(session_id)
        if os.path.isdir(path):
            shutil.rmtree(path)
        _loaded_sessions.discard(session_id)
        # Reload as fresh empty store
        _ensure_session(session_id)
        return {"cleared": True}
