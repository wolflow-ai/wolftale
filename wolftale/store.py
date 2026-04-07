"""
wolftale.store
--------------
Portable memory store: HNSWlib vector index + JSON sidecar.

Architecture
------------
Two files, one directory. Designed to live on a microSDXC card.

    wolftale_store/
        index.bin      ← HNSWlib vector index (vectors only)
        claims.json    ← { "_meta": {...}, "claims": {uuid: ClaimRecord} }

The card is cold storage. RAM is the search medium.
load() reads both files into RAM. Every operation after that is in-memory.
save() writes both files back to disk. Only called explicitly.

In-memory state (four structures)
----------------------------------
    _path          str                     store directory path
    _index         hnswlib.Index           vector index
    _claims        dict[str, ClaimRecord]  uuid → ClaimRecord
    _label_to_id   dict[int, str]          hnswlib label → uuid
    _domain_index  dict[str, list[int]]    domain → [hnswlib labels]

_label_to_id and _domain_index are derived from _claims on load.
They are not persisted — claims.json is the single source of truth.
The index holds vectors; the JSON holds everything else.

Self-describing stores
----------------------
Every store carries its own specification in the _meta block:
embedding model name, vector dimensions, index capacity, schema version.
A store loaded on any machine knows exactly what it was built with.
No external config. No assumptions.

Switching models requires re-embedding all claims (store.migrate()).
Vectors from different models live in incompatible geometric spaces.
load() detects model mismatches and raises rather than producing
garbage results silently.

Resizing capacity requires rebuilding the index (store.resize()).
A warning fires at 80% capacity so you act before hitting the ceiling.

Public interface
----------------
    load(path)                              read files into RAM
    save()                                  write RAM back to disk
    write(claim)        → StoreResult       add claim to in-memory index
    search(query, ...)  → List[ClaimRecord] search in-memory index
    count()             → int               number of stored claims
    all_claims()        → List[ClaimRecord] all ClaimRecords
    migrate(new_model)                      re-embed all claims with new model
    resize(new_max)                         rebuild index with higher capacity

Conflict classification (no API call — structural only)
--------------------------------------------------------
    same domain + similarity >= CONFLICT_THRESHOLD (0.75)  → contradiction
    similarity >= DUPLICATE_THRESHOLD (0.92)               → duplicate
    different domain + similarity >= CONFLICT_THRESHOLD    → update
    below CONFLICT_THRESHOLD                               → clear

Contradiction takes priority over duplicate: antonyms share sentence
structure and score very high similarity. Silently deduplicating them
would swallow a real conflict.

Confidence decay
----------------
Claims lose confidence over time unless reinforced by retrieval.
Decay is applied at read time (not write time) — the stored record
keeps its current confidence; the caller sees the decayed value.

    current_confidence = confidence * decay_factor ^ days_since_stored

decay_factor is domain-specific (see DECAY_FACTORS). Preferences decay
slower than commitments; identity facts decay slowest of all.

Retrieval reinforcement
-----------------------
Each retrieval bumps confidence by REINFORCEMENT_BUMP (0.05), capped at
original_confidence. This mutates the stored record — retrieval tracking
is persistent state. The bump is applied before decay so the caller
sees: (stored_confidence + bump, decayed) — reinforced but realistic.

The cap at original_confidence is intentional: retrieval frequency cannot
push a claim above its extraction-time strength. Extraction quality and
retrieval frequency are independent signals. A weakly-extracted claim
(confidence 0.60) that gets retrieved often should not end up competing
with a strongly-extracted claim (confidence 0.90). The ceiling preserves
that distinction.
"""

import json
import os
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

from .models import ClaimRecord, StoreResult, ConflictInfo


# ---------------------------------------------------------------------------
# Default configuration — used only when creating a new store.
# Existing stores read their own _meta block; these values are never
# applied to a store that already exists on disk.
# ---------------------------------------------------------------------------

DEFAULT_META = {
    "embedding_model":  "all-MiniLM-L6-v2",
    "embedding_dim":    384,
    "max_elements":     10_000,
    "ef_construction":  200,
    "M":                16,
    "schema_version":   "1",
}

# Conflict thresholds
DUPLICATE_THRESHOLD = 0.92   # similarity above which a claim is a duplicate
CONFLICT_THRESHOLD  = 0.75   # similarity above which same-domain pair is contradiction
MIN_SCORE           = 0.40   # minimum similarity for search results
PRE_FILTER_K        = 5      # candidates pulled at write time for conflict detection

# Capacity warning threshold
CAPACITY_WARNING_RATIO = 0.80

# ---------------------------------------------------------------------------
# Confidence decay — domain-specific decay factors.
#
# Formula: current_confidence = confidence * decay_factor ^ days_since_stored
#
# Half-lives are approximate (derived from: ln(0.5) / ln(decay_factor)):
#   identity:   ~1,400 days (~4 years)   — who someone is changes rarely
#   preference: ~700 days (~2 years)     — preferences shift slowly
#   technical:  ~460 days (~15 months)   — tools change with projects
#   relational: ~345 days (~1 year)      — relationships evolve
#   commitment: ~70 days                 — intentions are time-bound
#   ephemeral:  ~14 days                 — explicitly transient
#   other:      ~345 days                — default mid-range
# ---------------------------------------------------------------------------

DECAY_FACTORS = {
    "identity":   0.9995,
    "preference": 0.9990,
    "technical":  0.9985,
    "relational": 0.9980,
    "other":      0.9980,
    "commitment": 0.9900,
    "ephemeral":  0.9500,
}

# Retrieval reinforcement — how much confidence is restored per retrieval.
# Small bump: retrieval should slow decay, not override it.
# Capped at original_confidence — see module docstring for rationale.
REINFORCEMENT_BUMP = 0.05

# File names inside the store directory.
INDEX_FILE  = "index.bin"
CLAIMS_FILE = "claims.json"


# ---------------------------------------------------------------------------
# In-memory state — module-level singletons.
# All None until load() is called.
# ---------------------------------------------------------------------------

_path:         Optional[str]                      = None
_index:        Optional[hnswlib.Index]            = None
_claims:       Dict[str, ClaimRecord]             = {}
_label_to_id:  Dict[int, str]                     = {}   # hnswlib int → uuid
_domain_index: Dict[str, List[int]]               = defaultdict(list)  # domain → [labels]
_meta:         Dict                               = {}
_model:        Optional[SentenceTransformer]      = None
_next_label:   int                               = 0     # monotonically increasing


# ---------------------------------------------------------------------------
# Public lifecycle interface
# ---------------------------------------------------------------------------

def load(path: str) -> None:
    """
    Load the store from disk into RAM.

    If the directory does not exist, creates it and initializes an empty store
    with DEFAULT_META. On subsequent loads, reads _meta from claims.json and
    initializes the index with the stored parameters.

    Parameters
    ----------
    path : str
        Directory path for the store. This is what goes on the card.

    Raises
    ------
    ValueError
        If the store on disk was built with a different embedding model
        than what is currently configured — detected via _meta comparison.
        Prevents silently mixing incompatible vector spaces.
    RuntimeError
        If claims.json or index.bin exists but is unreadable or corrupted.
    """
    global _path, _index, _claims, _label_to_id, _domain_index, _meta, _model, _next_label

    _path = path

    if not os.path.isdir(path):
        # New store — initialize empty with defaults.
        os.makedirs(path, exist_ok=True)
        _meta = dict(DEFAULT_META)
        _meta["created_at"] = _now()
        _claims = {}
        _index = _new_index(_meta)
        _model = _load_model(_meta["embedding_model"])
        _label_to_id = {}
        _domain_index = defaultdict(list)
        _next_label = 0
        return

    # Existing store — read claims.json first to get _meta.
    claims_path = os.path.join(path, CLAIMS_FILE)
    index_path  = os.path.join(path, INDEX_FILE)

    if not os.path.isfile(claims_path):
        # Directory exists but no claims.json — treat as a fresh empty store.
        # This happens when the directory was created but never written to
        # (e.g. a failed first request). Safe to initialize fresh.
        _meta = dict(DEFAULT_META)
        _meta["created_at"] = _now()
        _claims = {}
        _index = _new_index(_meta)
        _model = _load_model(_meta["embedding_model"])
        _label_to_id = {}
        _domain_index = defaultdict(list)
        _next_label = 0
        return

    with open(claims_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _meta   = data.get("_meta", dict(DEFAULT_META))
    _claims = data.get("claims", {})

    # Initialize model from stored spec.
    _model = _load_model(_meta["embedding_model"])

    # Initialize index from stored parameters.
    _index = _new_index(_meta)
    if os.path.isfile(index_path):
        _index.load_index(index_path, max_elements=_meta["max_elements"])

    # Rebuild derived structures from _claims.
    _label_to_id  = {}
    _domain_index = defaultdict(list)
    _next_label   = 0

    for uuid, claim in _claims.items():
        label = claim.get("_label")
        if label is not None:
            _label_to_id[label] = uuid
            _domain_index[claim["domain"]].append(label)
            if label >= _next_label:
                _next_label = label + 1

    _capacity_check()


def save() -> None:
    """
    Write the in-memory store back to disk.

    Saves both files atomically (write to temp, rename).
    Call at session end or on clean interrupt.

    Raises
    ------
    RuntimeError
        If load() has not been called.
    """
    _assert_loaded()

    index_path  = os.path.join(_path, INDEX_FILE)
    claims_path = os.path.join(_path, CLAIMS_FILE)

    # Save index.
    if _index.get_current_count() > 0:
        _index.save_index(index_path)
    else:
        # Empty index — write nothing (load() handles missing index.bin).
        if os.path.isfile(index_path):
            os.remove(index_path)

    # Save claims + meta.
    payload = {"_meta": _meta, "claims": _claims}
    tmp_path = claims_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, claims_path)  # atomic on POSIX and Windows


# ---------------------------------------------------------------------------
# Public store interface
# ---------------------------------------------------------------------------

def write(claim: ClaimRecord) -> StoreResult:
    """
    Add a ClaimRecord to the in-memory store, with conflict detection.

    Does NOT write to disk. Call save() to persist.

    Steps:
        1. Embed the claim text.
        2. Search for similar existing claims.
        3. Classify any conflicts.
        4. Skip if duplicate; write otherwise.

    Returns
    -------
    StoreResult
        action  : "stored" | "deduplicated" | "superseded" | "flagged"
        claim_id: UUID of stored claim (None if deduplicated)
        conflicts: List[ConflictInfo]
        reason  : human-readable explanation
    """
    _assert_loaded()
    _capacity_check()

    embedding = _embed(claim["claim"])
    conflicts: List[ConflictInfo] = []

    # Search for conflicts only if store is non-empty.
    if _index.get_current_count() > 0:
        k = min(PRE_FILTER_K, _index.get_current_count())
        labels, distances = _index.knn_query([embedding], k=k)

        for label, distance in zip(labels[0], distances[0]):
            similarity = _dist_to_sim(distance)
            if similarity < CONFLICT_THRESHOLD:
                continue

            existing_uuid  = _label_to_id.get(int(label))
            if existing_uuid is None:
                continue
            existing_claim = _claims.get(existing_uuid)
            if existing_claim is None:
                continue

            conflict_type = _classify_conflict(
                similarity, claim["domain"], existing_claim["domain"]
            )
            conflicts.append({
                "existing_id":    existing_uuid,
                "existing_claim": existing_claim["claim"],
                "similarity":     round(similarity, 4),
                "conflict_type":  conflict_type,
            })

    # Duplicate → skip write entirely.
    duplicate = next((c for c in conflicts if c["conflict_type"] == "duplicate"), None)
    if duplicate:
        return {
            "action":    "deduplicated",
            "claim_id":  None,
            "conflicts": conflicts,
            "reason": (
                f"Claim too similar to existing "
                f"'{duplicate['existing_claim'][:60]}' "
                f"(similarity={duplicate['similarity']:.2f}). Skipped."
            ),
        }

    # Assign HNSWlib label and store.
    global _next_label
    label = _next_label
    _next_label += 1

    _index.add_items([embedding], [label])
    claim["_label"] = label          # store label in record for rebuild on load
    _claims[claim["id"]] = claim
    _label_to_id[label] = claim["id"]
    _domain_index[claim["domain"]].append(label)

    # Determine result action.
    has_contradiction = any(c["conflict_type"] == "contradiction" for c in conflicts)
    has_update        = any(c["conflict_type"] == "update"        for c in conflicts)

    if has_contradiction:
        return {
            "action":    "flagged",
            "claim_id":  claim["id"],
            "conflicts": conflicts,
            "reason": (
                f"Claim stored but flagged — potential contradiction with "
                f"{len(conflicts)} existing claim(s). Review recommended."
            ),
        }

    if has_update:
        return {
            "action":    "superseded",
            "claim_id":  claim["id"],
            "conflicts": conflicts,
            "reason": (
                f"Claim stored. Similar claim(s) in store — may be an update. "
                f"{len(conflicts)} related claim(s) flagged."
            ),
        }

    return {
        "action":    "stored",
        "claim_id":  claim["id"],
        "conflicts": [],
        "reason":    "Claim stored. No conflicts detected.",
    }


def search(
    query:     str,
    domains:   Optional[List[str]] = None,
    top_k:     int                 = 3,
    min_score: float               = MIN_SCORE,
) -> List[ClaimRecord]:
    """
    Search the in-memory store for claims relevant to a query.

    Two-pass strategy:
        Pass 1 — domain-conditioned: if domains supplied, filter to those domains.
        Pass 2 — broad fallback: if pass 1 returns nothing above min_score,
                 search across all domains.

    Each returned ClaimRecord has had retrieval reinforcement applied to
    the stored record (retrieved_count incremented, last_retrieved updated,
    confidence bumped up to original_confidence), then confidence decay
    applied for the returned value (confidence reduced based on age).

    The caller sees decayed confidence. The store retains reinforced state.

    Parameters
    ----------
    query    : str
    domains  : optional list of domain strings to search within
    top_k    : max results to return
    min_score: minimum cosine similarity to include

    Returns
    -------
    List[ClaimRecord]
        Sorted by relevance (highest similarity first). Empty if nothing qualifies.
        Confidence values reflect decay since storage.
    """
    _assert_loaded()

    if _index.get_current_count() == 0:
        return []

    embedding = _embed(query)

    # Pass 1 — domain-conditioned.
    if domains:
        results = _search_domains(embedding, domains, top_k, min_score)
        if results:
            return results

    # Pass 2 — broad fallback (also the direct path when domains=None).
    return _search_broad(embedding, top_k, min_score)


def count() -> int:
    """Return the number of claims currently in the store."""
    _assert_loaded()
    return len(_claims)


def all_claims() -> List[ClaimRecord]:
    """
    Return all ClaimRecords. Useful for /store command and debugging.
    Not intended for production retrieval — use search() instead.
    Confidence values are as stored (not decayed).
    """
    _assert_loaded()
    return list(_claims.values())


# ---------------------------------------------------------------------------
# Migration and resize
# ---------------------------------------------------------------------------

def migrate(new_model_name: str) -> None:
    """
    Re-embed all claims with a new embedding model and rebuild the index.

    Switching models is not a config change — vectors from different models
    live in incompatible geometric spaces. This operation:
        1. Loads the new model to get its output dimension.
        2. Re-embeds all claim texts.
        3. Rebuilds the index from scratch with the new dimension.
        4. Updates _meta.
        5. Does NOT save — call save() explicitly after.

    Parameters
    ----------
    new_model_name : str
        HuggingFace model name, e.g. "all-mpnet-base-v2"
    """
    global _model, _index, _label_to_id, _domain_index, _meta, _next_label
    _assert_loaded()

    print(f"  Loading new model: {new_model_name}")
    new_model = SentenceTransformer(new_model_name)
    sample    = new_model.encode(["test"], convert_to_numpy=True)
    new_dim   = sample.shape[1]

    print(f"  New dimension: {new_dim}. Re-embedding {len(_claims)} claims...")
    _model = new_model

    new_meta = dict(_meta)
    new_meta["embedding_model"] = new_model_name
    new_meta["embedding_dim"]   = new_dim

    new_index = _new_index(new_meta)
    new_label_to_id  = {}
    new_domain_index = defaultdict(list)
    new_next_label   = 0

    for uuid, claim in _claims.items():
        embedding = _embed(claim["claim"])
        label     = new_next_label
        new_next_label += 1
        new_index.add_items([embedding], [label])
        claim["_label"] = label
        new_label_to_id[label]  = uuid
        new_domain_index[claim["domain"]].append(label)

    _index        = new_index
    _label_to_id  = new_label_to_id
    _domain_index = new_domain_index
    _next_label   = new_next_label
    _meta         = new_meta

    print(f"  Migration complete. Call save() to persist.")


def resize(new_max_elements: int) -> None:
    """
    Rebuild the index with a higher capacity ceiling.

    HNSWlib does not support in-place resize. This rebuilds the index
    from existing vectors. Does NOT save — call save() explicitly after.

    Parameters
    ----------
    new_max_elements : int
        New capacity ceiling. Must be >= current count().
    """
    global _index, _meta
    _assert_loaded()

    current = count()
    if new_max_elements < current:
        raise ValueError(
            f"new_max_elements ({new_max_elements}) must be >= current count ({current})."
        )

    print(f"  Rebuilding index: {_meta['max_elements']} → {new_max_elements} capacity...")
    new_meta = dict(_meta)
    new_meta["max_elements"] = new_max_elements

    new_index = _new_index(new_meta)
    for label, uuid in _label_to_id.items():
        claim     = _claims[uuid]
        embedding = _embed(claim["claim"])
        new_index.add_items([embedding], [label])

    _index = new_index
    _meta  = new_meta
    print(f"  Resize complete. Call save() to persist.")


# ---------------------------------------------------------------------------
# Confidence decay and retrieval reinforcement
# ---------------------------------------------------------------------------

def _apply_reinforcement(claim: ClaimRecord) -> None:
    """
    Increment retrieval tracking and bump confidence for a retrieved claim.

    Mutates _claims[claim['id']] directly — retrieval tracking is persistent
    state that should be saved to disk at session end.

    The confidence bump is capped at original_confidence. This is the key
    design constraint: retrieval frequency can restore a claim to its
    extraction-time strength, but not exceed it. A weakly-extracted claim
    (0.60) that is retrieved often will not compete with a strongly-extracted
    claim (0.90) — their extraction quality remains the distinguishing signal.

    For claims from older stores that predate original_confidence, we fall
    back to capping at 1.0 to avoid KeyError.
    """
    stored = _claims.get(claim["id"])
    if stored is None:
        return

    ceiling = stored.get("original_confidence", 1.0)
    stored["confidence"]      = min(ceiling, stored["confidence"] + REINFORCEMENT_BUMP)
    stored["retrieved_count"] = stored.get("retrieved_count", 0) + 1
    stored["last_retrieved"]  = _now()


def _apply_decay(claim: ClaimRecord) -> ClaimRecord:
    """
    Return a copy of the claim with confidence decayed based on age.

    Does NOT mutate the stored record — decay is applied at read time only.
    The store retains the reinforced confidence; the caller sees the realistic
    current value.

    Days since storage is computed from claim['timestamp']. If the timestamp
    is missing or unparseable (e.g. claims from very old stores), decay is
    skipped and the stored confidence is returned unchanged.

    For claims from older stores that predate original_confidence, we default
    the ceiling to 1.0 so the function still works correctly.
    """
    import copy

    result = copy.copy(claim)

    try:
        stored_at = datetime.fromisoformat(claim["timestamp"])
        # Make timezone-aware if naive (older records may lack tz info)
        if stored_at.tzinfo is None:
            stored_at = stored_at.replace(tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - stored_at).total_seconds() / 86400.0
    except (KeyError, ValueError):
        return result   # Can't compute age — return as-is

    if days <= 0:
        return result   # Brand new claim — no decay yet

    decay_factor = DECAY_FACTORS.get(claim.get("domain", "other"), DECAY_FACTORS["other"])
    decayed = claim["confidence"] * (decay_factor ** days)

    # Never decay below a small floor — a stored claim is still evidence of something.
    result["confidence"] = max(0.05, round(decayed, 4))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_loaded() -> None:
    if _path is None:
        raise RuntimeError(
            "Store not loaded. Call store.load(path) before any other operation."
        )


def _capacity_check() -> None:
    if _meta and _index is not None:
        current  = _index.get_current_count()
        capacity = _meta.get("max_elements", DEFAULT_META["max_elements"])
        if current >= capacity * CAPACITY_WARNING_RATIO:
            warnings.warn(
                f"Wolftale store is at {current}/{capacity} capacity "
                f"({100*current/capacity:.0f}%). Consider calling store.resize().",
                stacklevel=3,
            )


def _new_index(meta: dict) -> hnswlib.Index:
    """Create and initialize a new HNSWlib index from a meta block."""
    idx = hnswlib.Index(space="cosine", dim=meta["embedding_dim"])
    idx.init_index(
        max_elements   = meta["max_elements"],
        ef_construction= meta["ef_construction"],
        M              = meta["M"],
    )
    idx.set_ef(50)   # ef at query time — higher = more accurate, slower
    return idx


def _load_model(model_name: str) -> SentenceTransformer:
    """Load the sentence-transformers model. Cached at module level after load()."""
    global _model
    if _model is None or getattr(_model, "_model_name", None) != model_name:
        _model = SentenceTransformer(model_name)
        _model._model_name = model_name   # tag for cache check
    return _model


def _embed(text: str) -> List[float]:
    """Embed a single string. Returns a list of floats."""
    return _model.encode(text, convert_to_numpy=True).tolist()


def _dist_to_sim(distance: float) -> float:
    """Convert HNSWlib cosine distance to cosine similarity."""
    return 1.0 - (distance / 2.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_conflict(
    similarity:      float,
    incoming_domain: str,
    existing_domain: str,
) -> str:
    """
    Classify a detected conflict based on similarity score and domain.

    Contradiction takes priority over duplicate: antonyms share sentence
    structure and score very high similarity (e.g., 'dark theme' vs 'light
    theme' scores ~0.96). Silently deduplicating them would swallow a real
    conflict.

    Rules (in priority order):
        same domain + similarity >= CONFLICT_THRESHOLD  → contradiction
        similarity >= DUPLICATE_THRESHOLD               → duplicate
        different domain + similarity >= CONFLICT_THRESHOLD → update
    """
    if incoming_domain == existing_domain and similarity >= CONFLICT_THRESHOLD:
        return "contradiction"
    if similarity >= DUPLICATE_THRESHOLD:
        return "duplicate"
    return "update"


def _search_domains(
    embedding: List[float],
    domains:   List[str],
    top_k:     int,
    min_score: float,
) -> List[ClaimRecord]:
    """
    Search within specific domains, apply reinforcement and decay, return results.

    HNSWlib has no metadata filtering — we over-fetch from the full index
    and post-filter by domain. At 10,000 entries this is negligible.
    """
    # Collect all labels that belong to the requested domains.
    valid_labels = set()
    for domain in domains:
        valid_labels.update(_domain_index.get(domain, []))

    if not valid_labels:
        return []

    # Over-fetch to ensure we have enough after domain filtering.
    k = min(top_k * 5, _index.get_current_count())
    labels, distances = _index.knn_query([embedding], k=k)

    scored = []
    for label, distance in zip(labels[0], distances[0]):
        if int(label) not in valid_labels:
            continue
        similarity = _dist_to_sim(distance)
        if similarity < min_score:
            continue
        uuid  = _label_to_id.get(int(label))
        claim = _claims.get(uuid)
        if claim:
            _apply_reinforcement(claim)           # mutates stored record
            scored.append((similarity, _apply_decay(claim)))   # returns decayed copy
        if len(scored) >= top_k:
            break

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def _search_broad(
    embedding: List[float],
    top_k:     int,
    min_score: float,
) -> List[ClaimRecord]:
    """Search across all domains, apply reinforcement and decay, return results."""
    k = min(top_k, _index.get_current_count())
    labels, distances = _index.knn_query([embedding], k=k)

    scored = []
    for label, distance in zip(labels[0], distances[0]):
        similarity = _dist_to_sim(distance)
        if similarity < min_score:
            continue
        uuid  = _label_to_id.get(int(label))
        claim = _claims.get(uuid)
        if claim:
            _apply_reinforcement(claim)           # mutates stored record
            scored.append((similarity, _apply_decay(claim)))   # returns decayed copy

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]
