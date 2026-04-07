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

Changelog:
  v2 — Added proper noun assertion patterns:
       'I am [Name]' and 'I'm [Name]' now trigger ASSERTION_PATTERN.
       Previously only compound forms ('I am a', 'I work at') were matched,
       so bare identity declarations like 'I am Chris' fell through to skip.

       Implementation: a two-part approach.
         Part 1 — negative lookahead in ASSERTION_PATTERN excludes the
                  article forms (a, an, the) and lowercase continuations
                  so 'I am a developer' still routes to the existing
                  compound match, not here.
         Part 2 — PROPER_NOUN_ASSERTION_PATTERN is a separate compiled
                  pattern that gate.py checks explicitly for the
                  'I am [Capital]' / 'I'm [Capital]' case.
                  Kept separate to avoid making ASSERTION_PATTERN
                  unreadable and to allow independent tuning.
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
#
# v2 note: bare proper noun forms ('I am Chris', 'I'm Chris') are handled
# by PROPER_NOUN_ASSERTION_PATTERN below. This pattern covers compound forms
# and role/location/tool assertions.
# ---------------------------------------------------------------------------

ASSERTION_PATTERN = re.compile(
    r'\b(?:'
    # Specific compound forms — declarative, not exploratory
    r'I am a|I am an|I\'m a|I\'m an|'                          # "I'm a developer" not "I'm thinking"
    r'I was a|I was an|I\'ve been a|I\'ve been an|'            # "I was a teacher"
    r'I am based|I\'m based|I am located|I\'m located|'
    r'I work at|I work for|I work with|I worked at|I\'ve worked at|'
    r'I live in|I live at|I live near|'
    r'I have a|I have an|I\'ve got a|I own a|I own an|'
    r'I use |I\'ve used|I\'m using|'                            # trailing space prevents "I used to"
    r'I build|I built|I\'ve built|I\'m building|'
    r'I run |I manage|I lead|I founded|I started|'             # trailing space prevents "I running"
    r'my name is|my name\'s|'
    r'my company|my team|my project|my business|'
    r'my background|my experience|my career|my role|my job|'
    r'I study|I\'m studying|I\'m learning|I\'ve learned|'
    r'I speak |I know how|'
    r'I\'ve been working|I\'ve been building|I\'ve been using|I\'ve been learning'
    r')\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Proper noun assertion pattern (v2)
#
# Catches bare identity declarations not covered by ASSERTION_PATTERN:
#   "I am Chris"       → matches
#   "I'm Chris"        → matches
#   "I am Messina"     → matches
#   "I'm thinking"     → does NOT match (lowercase after I am/I'm)
#   "I am a developer" → does NOT match (article 'a' excluded)
#   "I am an engineer" → does NOT match (article 'an' excluded)
#   "I am based"       → does NOT match (covered by ASSERTION_PATTERN)
#
# How it works:
#   Matches "I am" or "I'm" followed by one or more Title-Cased words.
#   Negative lookahead excludes articles (a, an, the) and the specific
#   compound openings already handled above ('based', 'located', 'a\b',
#   'an\b') to prevent double-matching.
#
# Title-case heuristic: [A-Z][a-zA-Z]+ catches proper nouns in normal
# sentence position. It won't catch ALL-CAPS names or names at the start
# of a sentence (where everything is capitalised) — acceptable for v1.
#
# gate.py checks this pattern alongside ASSERTION_PATTERN and treats a
# match as signal_type "assertion" with the same downstream behaviour.
# ---------------------------------------------------------------------------

PROPER_NOUN_ASSERTION_PATTERN = re.compile(
    r'\b(?:I am|I\'m)\s+'           # "I am" or "I'm" followed by whitespace
    r'(?!'                           # negative lookahead — exclude:
        r'a\b|an\b|the\b|'          #   articles
        r'based|located|'           #   compound forms already in ASSERTION_PATTERN
        r'not\b|just\b|'            #   negations and fillers ("I'm not sure")
        r'still\b|also\b|'          #   continuations ("I'm still learning")
        r'so\b|very\b|'             #   intensifiers
        r'going|trying|working|'    #   progressive verb forms ("I'm going to")
        r'thinking|wondering|'      #   exploratory phrases
        r'not\s|just\s'             #   with following space
    r')'
    r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',   # one or more Title-Cased words
    re.UNICODE
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
