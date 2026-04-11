# Wolftale

*Personal memory layer for the Wolflow ecosystem.*

**Live demo: [tale.wolflow.ai](https://tale.wolflow.ai)**

---

## What It Is

Wolftale is a persistent, portable, personal memory layer. It stores what a user has established across conversations — preferences, facts, commitments, ongoing projects — and retrieves the right subset at query time to inform generation.

It is infrastructure, not a companion. A companion sits on top of a memory layer. Wolftale *is* the memory layer.

> The memory of a relationship should belong to the person in it.

---

## The Problem It Solves

Language models have no memory between sessions. Every conversation starts blank. Wolftale gives the model a structured, queryable model of the user that persists across time — small enough to be portable, fast enough to be practical, and smart enough to revise rather than just append.

Centralised AI memory platforms are converging on the right architecture. The problem is that your memory lives in their cloud, under their terms, accessible only through their client. Wolftale's answer is a different architecture: the memory store is a file you own, not a service you subscribe to.

---

## Where It Fits

Wolftale is part of the [Wolflow](https://wolflow.ai) ecosystem:

| Tool | What it solves |
|---|---|
| Wolfprompt | What does this prompt *mean*? (signal extraction, routing) |
| Wolftrain | How do we *shape* a model? (fine-tuning) |
| Wolfmerge | How do multiple models *compose*? (orchestration) |
| Wolfkit | How do we *prepare* the material? (document processing) |
| **Wolftale** | **What does this system *remember*?** |

---

## Design Principles

**Ownership before everything.** The memory store is a file, not a service. It lives on hardware the user controls.

**Portability is physical.** The target persistence medium is a microSDXC card. The entire memory layer fits on a 1GB card with room to spare.

**Detail is relational, not intrinsic.** Information becomes worth storing only in relation to a goal or decision. Not every sentence is a memory.

**Memory is a model of the user, not a log.** Claims can be superseded, confidence decays over time, contradictions are surfaced — not silently appended.

**Small footprint by design.** At 5,000 entries with 384-dimension embeddings, the vector index is ~7–8MB. The card is the persistence medium; RAM is the search medium.

---

## Architecture

```
Conversation turn
       │
       ▼
┌──────────────────┐
│   1. Gate Layer  │  Pattern matching — no API calls, <20ms
└─────────┬────────┘
          │ extract / edge / skip
          ▼
┌──────────────────┐
│   2. Extractor   │  Single Claude Haiku prompt (standard) or RLM loop (edge — stubbed)
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│   3. Store       │  HNSWlib + JSON sidecar. Conflict detection. Confidence decay.
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│   4. Retrieval   │  Query-conditioned. Conflict surfacing. Injects into context.
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│   5. Generation  │  Claude Sonnet. Memory block injected into every turn.
└──────────────────┘
```

---

## The Unit of Memory

Every memory is a **ClaimRecord** — an assertion with provenance:

```json
{
  "claim": "Chris prefers directness over diplomatic softening.",
  "confidence": 0.85,
  "original_confidence": 0.85,
  "domain": "preference",
  "source_turn": 42,
  "timestamp": "2026-04-05T22:00:00Z",
  "supersedes": [],
  "retrieved_count": 3,
  "last_retrieved": "2026-04-09T14:22:00Z",
  "extraction_path": "standard",
  "id": "uuid-here"
}
```

Not a raw quote. Not a summary. A discrete, revisable, confidence-weighted claim.

`original_confidence` is set at extraction and never mutated — it is the ceiling for retrieval reinforcement. Retrieval frequency cannot push a claim above its extraction-time strength.

---

## Confidence Decay

Claims lose confidence over time unless reinforced by retrieval.

```
current_confidence = confidence × decay_factor ^ days_since_stored
```

| Domain | Decay Factor | Half-life |
|---|---|---|
| identity | 0.9995 | ~4 years |
| preference | 0.9990 | ~2 years |
| technical | 0.9985 | ~15 months |
| relational | 0.9980 | ~1 year |
| commitment | 0.9900 | ~70 days |
| ephemeral | 0.9500 | ~14 days |

Each retrieval bumps confidence +0.05, capped at `original_confidence`. Decay is applied at read time, not write time — the store retains reinforced state; the caller sees the realistic current value.

---

## Repository Structure

```
wolftale/
  wolftale/           ← Python package
    __init__.py
    config.py         ← Loads .env from repo root
    models.py         ← TypedDict shapes: ClaimRecord, GateDecision, etc.
    patterns.py       ← All compiled regex patterns for the gate layer
    gate.py           ← Pure function. evaluate(turn) → GateDecision
    extractor.py      ← Claude Haiku. extract(turn, gate) → ExtractionResult
    store.py          ← HNSWlib + JSON. load/save/write/search/count
    retrieval.py      ← retrieve() + format_for_context()
  api/
    main.py           ← FastAPI backend. Per-session stores. Five endpoints.
    requirements.txt
    render.yaml       ← Render deployment config (persistent disk)
  ui/
    src/
      Wolftale.jsx    ← React SPA. Idle screen, chat, memory panel, export.
      main.jsx
    index.html
    vite.config.js
    package.json
    render.yaml       ← Render static site config
  demo.py             ← Local REPL. Runs full pipeline interactively.
  test_gate.py        ← 10 smoke tests + interactive mode
  test_store.py       ← 12 tests including decay and reinforcement
  test_retrieval.py   ← 7 tests including conflict detection
  test_extractor.py   ← Gate + extractor end-to-end (makes real API calls)
  requirements.txt    ← Root venv requirements
  .env                ← API key (gitignored, never commit)
```

---

## Build State

| Layer | File | Status |
|---|---|---|
| Models | `wolftale/models.py` | ✅ Complete |
| Patterns | `wolftale/patterns.py` | ✅ Complete (v2) |
| Gate | `wolftale/gate.py` | ✅ Complete |
| Extractor | `wolftale/extractor.py` | ✅ Complete (edge path stubbed) |
| Store | `wolftale/store.py` | ✅ Complete — HNSWlib + JSON |
| Retrieval | `wolftale/retrieval.py` | ✅ Complete |
| API | `api/main.py` | ✅ Complete |
| UI | `ui/src/Wolftale.jsx` | ✅ Complete |
| Confidence Decay | `wolftale/store.py` | ✅ Complete |
| RLM Edge Path | `wolftale/extractor.py` | 🔲 Stubbed — deferred |
| MCP Exposure | — | 🔲 Planned |
| microSDXC Validation | — | 🔲 Planned |

---

## Local Development

**Prerequisites:** Python 3.10.11, venv active, `ANTHROPIC_API_KEY` in `.env`, Node.js installed.

**Terminal 1 — API:**
```powershell
cd C:\Users\clewi\wolftale
.\venv\Scripts\Activate.ps1
cd api
uvicorn main:app --reload --port 8000
```

**Terminal 2 — UI:**
```powershell
cd C:\Users\clewi\wolftale\ui
npm run dev
```

Open **http://localhost:5173**.

The UI needs a `.env.local` file in the `ui/` directory:
```
VITE_API_URL=http://localhost:8000
```

**Running tests:**
```powershell
# From repo root with venv active
python test_gate.py        # No API calls
python test_store.py       # No API calls
python test_retrieval.py   # No API calls
python test_extractor.py   # Makes real API calls to Claude Haiku
```

**Local REPL demo:**
```powershell
python demo.py
```

---

## Deployment

The live demo runs on Render:

- **API** — Web Service, `api/` directory, persistent disk at `/app/stores`
- **UI** — Static Site, `ui/` directory, custom domain `tale.wolflow.ai`

See `api/render.yaml` and `ui/render.yaml` for configuration. Set `ANTHROPIC_API_KEY` manually in the Render dashboard — it is intentionally absent from `render.yaml`.

---

## Session Model (Cloud Demo)

- UUID generated on first visit, stored in `localStorage`
- Passed as `X-Session-ID` header on every request
- Each session gets its own isolated store under `api/stores/{session_id}/`
- Sessions are wiped after 7 days
- Export your `claims.json` at any time — it is your memory, portable to any machine

---

## Part of the NLP Learning Path

Wolftale is the capstone project of a structured NLP learning path:

| Project | Contribution to Wolftale |
|---|---|
| P1 — Tokenization & Embeddings | How claims become vectors |
| P2 — Vector Databases | How the memory store works |
| P3 — RAG | How retrieved claims inject into generation |
| P4 — Entity Extraction | How named entities are pulled from turns |
| P6 — Recursive Language Models | The edge extraction path for high-stakes claims |
| P7 — Wolfprompt | The architecture pattern: gate, compiled patterns, typed outputs |
| P8/P9 — MCP | How Wolftale will expose retrieval as a composable tool |

---

*Memory is story. Store it like it matters.*

---

*Chris Messina · [Wolflow](https://wolflow.ai) · 2026*
