"""
wolftale
--------
Personal memory layer for the Wolflow ecosystem.

Stores what a user has established across conversations — preferences,
facts, commitments, ongoing projects — and retrieves the right subset
at query time to inform generation.

Current build state: Gate layer complete (models, patterns, gate).
Extractor, store, retrieval, and demo layers to follow.

Entry points (grow as layers are added):
  from wolftale.gate import evaluate
  from wolftale.models import ClaimRecord, GateDecision
"""

__version__ = "0.1.0"
__author__ = "Chris Messina"
