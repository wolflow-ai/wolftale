# Wolftale

*Personal memory layer for the Wolflow ecosystem.*

---

## What It Is

Wolftale is a persistent, portable, personal memory layer. It stores what a user has established across conversations — preferences, facts, commitments, ongoing projects — and retrieves the right subset at query time to inform generation.

It is infrastructure, not a companion. A companion sits on top of a memory layer. Wolftale *is* the memory layer.

---

## The Problem It Solves

Language models have no memory between sessions. Every conversation starts blank. Wolftale gives the model a structured, queryable model of the user that persists across time — small enough to be portable, fast enough to be practical, smart enough to revise rather than just append.

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

**Detail is relational, not intrinsic.** Information becomes worth storing only in relation to a goal or decision. Not every sentence is a memory.

**Perfection is the enemy of good enough.** Extraction cost scales with the value of what's being extracted. Most claims need one fast API call. High-stakes claims get more.

**Memory is a model of the user, not a log.** Claims can be superseded, confidence decays over time, contradictions are resolved — not silently appended.

**Small footprint, portable by design.** The memory store is designed to fit in RAM. The card is the persistence medium; RAM is the search medium.

---

## Architecture

```
Conversation turn
       │
       ▼
┌─────────────────┐
│  1. Gate Layer  │  Pattern matching — no API calls, <20ms
└────────┬────────┘
         │ extract / edge
         ▼
┌─────────────────┐
│  2. Extractor   │  Single Claude prompt (standard) or RLM loop (edge)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Store       │  ChromaDB → HNSWlib. Conflict detection before write.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Retrieval   │  Query-conditioned. Injects relevant claims into context.
└─────────────────┘
```

---

## The Unit of Memory

Every memory is a **ClaimRecord** — an assertion with provenance:

```json
{
  "claim": "Chris prefers directness over diplomatic softening.",
  "confidence": 0.85,
  "domain": "preference",
  "source_turn": 42,
  "timestamp": "2026-04-05T22:00:00Z",
  "supersedes": [],
  "retrieved_count": 0,
  "last_retrieved": null,
  "extraction_path": "standard",
  "id": "uuid-here"
}
```

Not a raw quote. Not a summary. A discrete, revisable, confidence-weighted claim.

---

## Build State

| Layer | File | Status |
|---|---|---|
| Models | `models.py` | ✅ Complete |
| Patterns | `patterns.py` | ✅ Complete |
| Gate | `gate.py` | ✅ Complete |
| Extractor | `extractor.py` | 🔲 Next |
| Store | `store.py` | 🔲 Planned |
| Retrieval | `retrieval.py` | 🔲 Planned |
| Demo | `demo.py` | 🔲 Planned |

---

## Environment

- Python 3.10.11
- Windows / PowerShell
- Requires: `anthropic`, `chromadb`, `sentence-transformers`
- API key: `ANTHROPIC_API_KEY` in environment

---

## Quick Start

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Test the gate layer
python -c "
from wolftale.gate import evaluate
print(evaluate('I prefer dark theme UIs'))
print(evaluate('What time is it?'))
"
```

---

## Part of the NLP Learning Path

Wolftale is the capstone project of a structured NLP learning path covering tokenization, vector databases, RAG, entity extraction, recursive language models, prompt routing, and MCP. Every prior project feeds into it.

*Memory is story. Store it like it matters.*

---

*Chris Messina · Wolflow · 2026*
