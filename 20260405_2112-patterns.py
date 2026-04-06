"""
wolftale.patterns
-----------------
All detection patterns used by the gate layer.

Isolated here so pattern tuning never requires touching gate logic.
Patterns are compiled once at import time.

Five signal types:
  PREFERENCE    — "I prefer", "I like", "I tend to" — durable user preferences
  ASSERTION     — "I am", "I work at", "I use" — facts about the user
  COMMITMENT    — "I'll", "I plan to" — time-bound intentions (short TTL)
  EPHEMERAL     — "today", "right now", "this week" — likely not worth storing
  CONTINUATION  — "continue", "go on", pronoun openers — mid-thought, skip

Design notes:
  - Patterns are ordered from most specific to least specific within each group
  - Ephemeral signals act as suppressors — if detected alongside an assertion,
    they lower confidence rather than triggering a skip outright
  - Continuation signals are checked first in gate.py — they exit early
"""

import re

# ---------------------------------------------------------------------------
# Preference signals
# Durable statements about how the user likes things to work
# ---------------------------------------------------------------------------

PREFERENCE_PATTERN = re.compile(
    r'\b(?:'
    r'I prefer|I like|I love|I hate|I dislike|I enjoy|I avoid|'
    r'I tend to|I usually|I generally|I typically|I always|I never|'
    r'my preference|my style|my approach|my habit|'
    r'works better for me|works best for me|'
    r'I find that|I\'ve found|I\'ve noticed|'
    r'I\'m more of a|I\'m a [a-z]+ person|'
    r'I do better with|I work better'
    r')\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Assertion signals
# Factual statements about who the user is, what they have, what they do
# ---------------------------------------------------------------------------

ASSERTION_PATTERN = re.compile(
    r'\b(?:'
    r'I am|I\'m|I was|I\'ve been|'
    r'I work|I worked|I\'ve worked|'
    r'I live|I\'m based|I\'m located|'
    r'I have|I\'ve got|I own|'
    r'I use|I\'ve used|I\'m using|'
    r'I build|I built|I\'ve built|I\'m building|'
    r'I run|I manage|I lead|I founded|I started|'
    r'my name|my company|my team|my project|my business|'
    r'my background|my experience|my career|my role|my job|'
    r'I study|I\'m studying|I\'m learning|I\'ve learned|'
    r'I speak|I know|I understand'
    r')\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Commitment signals
# Time-bound intentions — store with short TTL flag
# ---------------------------------------------------------------------------

COMMITMENT_PATTERN = re.compile(
    r'\b(?:'
    r'I\'ll|I will|I\'m going to|I am going to|'
    r'I plan to|I\'m planning to|I intend to|'
    r'I need to|I have to|I must|I should|'
    r'I want to|I\'d like to|I hope to|'
    r'I\'m working on|I\'m trying to'
    r')\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Ephemeral signals
# Temporal markers that suggest the claim is transient, not durable
# Used as confidence suppressors, not hard skips
# ---------------------------------------------------------------------------

EPHEMERAL_PATTERN = re.compile(
    r'\b(?:'
    r'today|tonight|this morning|this afternoon|this evening|'
    r'right now|at the moment|currently|these days|'
    r'this week|this month|this year|'
    r'just now|a moment ago|earlier today|'
    r'temporarily|for now|for the moment|short.?term'
    r')\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Continuation signals
# Signals that this turn is mid-thought — not a new assertion
# Gate exits early when these are detected
# ---------------------------------------------------------------------------

BARE_CONTINUATION_PATTERN = re.compile(
    r'^(?:'
    r'continue|go on|keep going|proceed|next|more|'
    r'and\??|elaborate|expand|finish|complete it|'
    r'finish it|go ahead|carry on|yes|no|ok|okay|sure|right|got it'
    r')[\.\!\?]?$',
    re.IGNORECASE
)

EXPLICIT_CONTINUATION_PATTERN = re.compile(
    r'\b(?:'
    r'continue (?:from|with|where)|following (?:up|on)|'
    r'building on (?:that|this)|based on (?:that|this|what you said)|'
    r'as (?:mentioned|discussed|noted)|referring back to|'
    r'in addition to (?:that|this)|like I said|as I said'
    r')\b',
    re.IGNORECASE
)

PRONOUN_OPENER_PATTERN = re.compile(
    r'^(?:he|she|it|they|them|his|her|its|their|this|that|these|those)\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Named entity signals (user-adjacent)
# Proper nouns likely referring to people, places, or projects in the user's life
# These alone don't trigger extraction but boost confidence when alongside assertions
# ---------------------------------------------------------------------------

NAMED_ENTITY_BOOST_PATTERN = re.compile(
    r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'  # Title-cased multi-word phrases
)
