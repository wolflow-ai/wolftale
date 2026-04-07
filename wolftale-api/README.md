# Wolftale API

FastAPI backend for the Wolftale cloud demo. Runs the full pipeline:
gate → extractor → store (HNSWlib + JSON, per-session) → retrieval → Claude generation.

## Local development

```bash
# From wolftale repo root — wolftale package must be importable
cd wolftale-api

# Create venv and install
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

# Set API key
# Add ANTHROPIC_API_KEY=your-key to a .env file, or set in environment

# Run
uvicorn main:app --reload --port 8000
```

API available at http://localhost:8000
Docs at http://localhost:8000/docs

## Session model

- Client generates a UUID on first visit, stores in localStorage as `wolftale_session_id`
- Every request passes it as `X-Session-ID` header
- Each session gets its own store under `stores/{session_id}/`
- Sessions are wiped after 7 days of inactivity (background task)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/chat | Run a turn through the full pipeline |
| GET | /api/memories | List all claims for a session |
| GET | /api/export | Download claims.json |
| DELETE | /api/memories | Clear all memories for a session |
| GET | /api/health | Liveness check |

## Deploying to Render

1. Push to GitHub
2. Create a new Web Service on Render, point at this directory
3. Render will detect `render.yaml` and configure automatically
4. Set `ANTHROPIC_API_KEY` in Render environment variables (marked `sync: false` in render.yaml — set manually in dashboard)
5. The persistent disk at `/app/stores` keeps session stores across deploys

## Directory structure

```
wolftale-api/
  main.py          — FastAPI app
  requirements.txt
  render.yaml      — Render deployment config
  stores/          — Per-session HNSWlib stores (created at runtime)
    {session-uuid}/
      index.bin
      claims.json
```

## CORS

Configured for:
- https://tale.wolflow.ai
- https://clewismessina.com
- http://localhost:5173 (Vite dev)
- http://localhost:4173 (Vite preview)

Add additional origins in `main.py` → `CORSMiddleware` → `allow_origins`.

## Note on concurrency

The wolftale store module uses module-level singletons. This backend is designed
for single-process deployment (one Render instance). Each session is protected
by an asyncio.Lock to prevent concurrent writes to the same store. If you need
to scale horizontally, the store module would need to be refactored to support
multiple concurrent stores without module-level state.
