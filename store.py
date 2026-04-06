"""
wolftale.store
--------------
Store layer: ChromaDB wrapper with conflict detection and domain-conditioned search.

Owns:
  write(claim)              → StoreResult
  search(query, ...)        → List[ClaimRecord]
  count()                   → int
  all_claims()              → List[ClaimRecord]

Write path:
  1. Embed the incoming claim text
  2. Query existing store for similar claims (top_k=5, cosine similarity)
  3. Classify each similar claim: duplicate | contradiction | update | clear
  4. If duplicate (similarity >= DUPLICATE_THRESHOLD): skip, return deduplicated
  5. If contradiction or update: flag in StoreResult, write anyway
  6. If clear: write normally

Conflict classification (no API call — structural only):
  - same domain + similarity >= CONFLICT_THRESHOLD (0.75): contradiction — write and flag
    (takes priority over duplicate: antonyms share sentence structure and score very high;
    silently deduplicating them would swallow a real conflict)
  - similarity >= DUPLICATE_THRESHOLD (0.92): duplicate — skip write
  - different domain + similarity >= CONFLICT_THRESHOLD (0.75): update — write and flag lightly
  - below CONFLICT_THRESHOLD: clear — write normally

Search path (two-pass):
  Pass 1 — domain-conditioned: query against claims in the specified or inferred domains
  Pass 2 — broad fallback: if pass 1 returns nothing above MIN_SCORE, search all domains
  Returns top_k results, sorted by score descending.

Serialization:
  Full ClaimRecord stored as JSON blob in 'record' metadata field.
  Filterable fields duplicated as flat metadata: domain, confidence, timestamp, extraction_path.
  ChromaDB persistent directory — wedge approach, proven in P2/P3.

Design notes:
  - ChromaDB client is initialized once at import time via _get_collection().
  - sentence-transformers model loaded once — same model used across the pipeline.
  - All metadata values must be str, int, or float for ChromaDB. Lists serialized to JSON.
  - supersedes field in ClaimRecord is List[str] — stored as JSON string in metadata.
  - IDs in ChromaDB are the ClaimRecord UUIDs.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from .models import ClaimRecord, StoreResult, ConflictInfo

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = "wolftale_store"          # Persistent ChromaDB directory (relative to repo root)
COLLECTION_NAME = "wolftale_claims"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Same model used across the pipeline

DUPLICATE_THRESHOLD = 0.92   # Cosine similarity — skip write, return deduplicated
CONFLICT_THRESHOLD = 0.75    # Cosine similarity — write but flag
MIN_SCORE = 0.40             # Minimum score for search results to be considered relevant
PRE_FILTER_K = 5             # How many candidates to pull for conflict detection at write time

# ---------------------------------------------------------------------------
# Lazy singletons — initialized on first use, not at import
# ---------------------------------------------------------------------------

_client: Optional[chromadb.PersistentClient] = None
_collection = None
_model: Optional[SentenceTransformer] = None


def _get_collection():
    """Initialize and return the ChromaDB collection. Singleton."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine distance, not L2
        )
    return _collection


def _get_model() -> SentenceTransformer:
    """Load and return the embedding model. Singleton."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _embed(text: str) -> List[float]:
    """Embed a single string. Returns a list of floats for ChromaDB."""
    return _get_model().encode(text, convert_to_numpy=True).tolist()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _claim_to_metadata(claim: ClaimRecord) -> dict:
    """
    Flatten a ClaimRecord into ChromaDB-compatible metadata.

    ChromaDB metadata values must be str, int, or float.
    Lists and None values are not allowed — serialize them.

    Two representations:
      - 'record': full JSON blob for lossless deserialization
      - flat fields: domain, confidence, timestamp, extraction_path
        duplicated as top-level keys for WHERE filtering
    """
    return {
        "record": json.dumps(claim),           # Full record — deserialize this
        "domain": claim["domain"],             # For domain filtering
        "confidence": claim["confidence"],     # For confidence filtering
        "timestamp": claim["timestamp"],       # For recency sorting
        "extraction_path": claim["extraction_path"],
    }


def _metadata_to_claim(metadata: dict) -> ClaimRecord:
    """Deserialize a ClaimRecord from ChromaDB metadata."""
    return json.loads(metadata["record"])


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def write(claim: ClaimRecord) -> StoreResult:
    """
    Write a ClaimRecord to the store, with conflict detection.

    Steps:
      1. Embed the claim text
      2. Query store for similar claims
      3. Classify conflicts
      4. Return deduplicated if duplicate found; otherwise write and return result

    Parameters
    ----------
    claim : ClaimRecord
        The claim to store. Must have a valid UUID in claim['id'].

    Returns
    -------
    StoreResult
        action: "stored" | "deduplicated" | "superseded" | "flagged"
        claim_id: UUID of the stored claim (None if deduplicated)
        conflicts: List of ConflictInfo for any detected conflicts
        reason: Human-readable explanation
    """
    collection = _get_collection()
    embedding = _embed(claim["claim"])

    # ------------------------------------------------------------------
    # Step 1: Check for conflicts
    # ------------------------------------------------------------------
    conflicts: List[ConflictInfo] = []

    existing_count = collection.count()
    if existing_count > 0:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(PRE_FILTER_K, existing_count),
            include=["metadatas", "distances"],
        )

        if results["ids"] and results["ids"][0]:
            for i, existing_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # ChromaDB cosine space returns distance (0=identical, 2=opposite)
                # Convert to similarity: similarity = 1 - (distance / 2)
                similarity = 1.0 - (distance / 2.0)

                if similarity < CONFLICT_THRESHOLD:
                    continue  # Not similar enough to matter

                existing_metadata = results["metadatas"][0][i]
                existing_claim = _metadata_to_claim(existing_metadata)

                conflict_type = _classify_conflict(
                    similarity, claim["domain"], existing_claim["domain"]
                )

                conflicts.append({
                    "existing_id": existing_id,
                    "existing_claim": existing_claim["claim"],
                    "similarity": round(similarity, 4),
                    "conflict_type": conflict_type,
                })

    # ------------------------------------------------------------------
    # Step 2: Act on conflicts
    # ------------------------------------------------------------------

    # Duplicate found — skip write entirely
    duplicate = next((c for c in conflicts if c["conflict_type"] == "duplicate"), None)
    if duplicate:
        return {
            "action": "deduplicated",
            "claim_id": None,
            "conflicts": conflicts,
            "reason": (
                f"Claim too similar to existing '{duplicate['existing_claim'][:60]}...' "
                f"(similarity={duplicate['similarity']:.2f}). Skipped."
            ),
        }

    # ------------------------------------------------------------------
    # Step 3: Write the claim
    # ------------------------------------------------------------------
    collection.add(
        ids=[claim["id"]],
        embeddings=[embedding],
        metadatas=[_claim_to_metadata(claim)],
    )

    # Determine action and reason based on what conflicts were found
    has_contradiction = any(c["conflict_type"] == "contradiction" for c in conflicts)
    has_update = any(c["conflict_type"] == "update" for c in conflicts)

    if has_contradiction:
        return {
            "action": "flagged",
            "claim_id": claim["id"],
            "conflicts": conflicts,
            "reason": (
                f"Claim stored but flagged — potential contradiction with "
                f"{len(conflicts)} existing claim(s). Review recommended."
            ),
        }

    if has_update:
        return {
            "action": "superseded",
            "claim_id": claim["id"],
            "conflicts": conflicts,
            "reason": (
                f"Claim stored. Similar claim(s) in store — may be an update. "
                f"{len(conflicts)} related claim(s) flagged."
            ),
        }

    return {
        "action": "stored",
        "claim_id": claim["id"],
        "conflicts": [],
        "reason": "Claim stored. No conflicts detected.",
    }


def search(
    query: str,
    domains: Optional[List[str]] = None,
    top_k: int = 3,
    min_score: float = MIN_SCORE,
) -> List[ClaimRecord]:
    """
    Search the store for claims relevant to a query.

    Two-pass strategy:
      Pass 1 — domain-conditioned: if domains specified, filter to those domains.
               Returns top_k results above min_score.
      Pass 2 — broad fallback: if pass 1 returns nothing above threshold,
               search across all domains. Returns top_k results above min_score.

    The caller (retrieval layer) decides which domains to pass.
    This function executes the search; it does not infer domain from the query.

    Parameters
    ----------
    query : str
        The query string to embed and search against.
    domains : List[str] or None
        If provided, search only within these domains first.
        If None, skip to broad search immediately.
    top_k : int
        Maximum number of results to return.
    min_score : float
        Minimum cosine similarity for a result to be included.

    Returns
    -------
    List[ClaimRecord]
        Sorted by relevance (highest similarity first).
        Empty list if nothing above threshold.
    """
    collection = _get_collection()
    existing_count = collection.count()

    if existing_count == 0:
        return []

    query_embedding = _embed(query)

    # ------------------------------------------------------------------
    # Pass 1 — domain-conditioned search
    # ------------------------------------------------------------------
    if domains:
        results = _query_with_domains(
            collection, query_embedding, domains, top_k, existing_count
        )
        claims = _filter_by_score(results, min_score)
        if claims:
            return claims

    # ------------------------------------------------------------------
    # Pass 2 — broad fallback (all domains)
    # ------------------------------------------------------------------
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, existing_count),
        include=["metadatas", "distances"],
    )

    return _filter_by_score(results, min_score)


def count() -> int:
    """Return the total number of claims in the store."""
    return _get_collection().count()


def all_claims() -> List[ClaimRecord]:
    """
    Return all claims in the store. Useful for debugging and demo.
    Not intended for production retrieval — use search() instead.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    return [_metadata_to_claim(m) for m in results["metadatas"]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_conflict(
    similarity: float,
    incoming_domain: str,
    existing_domain: str,
) -> str:
    """
    Classify a detected conflict based on similarity and domain.

    Rules:
      same domain + similarity >= CONFLICT_THRESHOLD  → contradiction
        (takes priority over duplicate — a claim that contradicts something
        you already believe is more important to keep than a true duplicate.
        Same-domain antonyms score very high similarity due to shared sentence
        structure; silently deduplicating them would swallow a real conflict.)
      similarity >= DUPLICATE_THRESHOLD               → duplicate
      different domain + similarity >= CONFLICT_THRESHOLD → update
    """
    if incoming_domain == existing_domain and similarity >= CONFLICT_THRESHOLD:
        return "contradiction"
    if similarity >= DUPLICATE_THRESHOLD:
        return "duplicate"
    return "update"


def _query_with_domains(
    collection,
    query_embedding: List[float],
    domains: List[str],
    top_k: int,
    existing_count: int,
) -> dict:
    """
    Query the collection filtered to specific domains.
    ChromaDB WHERE clause: {"domain": {"$in": domains}}
    """
    where = {"domain": {"$in": domains}} if len(domains) > 1 else {"domain": domains[0]}
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, existing_count),
        where=where,
        include=["metadatas", "distances"],
    )


def _filter_by_score(results: dict, min_score: float) -> List[ClaimRecord]:
    """
    Convert ChromaDB query results to ClaimRecords, filtering by min_score.
    Sorts by similarity descending.
    """
    if not results["ids"] or not results["ids"][0]:
        return []

    scored = []
    for i, _ in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        similarity = 1.0 - (distance / 2.0)
        if similarity >= min_score:
            claim = _metadata_to_claim(results["metadatas"][0][i])
            scored.append((similarity, claim))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [claim for _, claim in scored]
