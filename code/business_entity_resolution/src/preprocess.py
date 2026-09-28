"""
preprocess.py — Text normalisation for Business Entity Resolution.

Driven by EDA findings:
- S2 ~19% ALL-CAPS → lowercase everything
- Devanagari/Tamil/Kannada/Gurmukhi/Bengali in S2/S3 → NFKC normalise
- Legal suffix abbreviations (Ltd, Pvt, Corp…) → expand to canonical form
- Address abbreviations (Rd, St, Ave…) → expand to full form
- French legal forms (SARL, SASU, SAS) → keep lowercased for test set

Usage:
    from preprocess import normalize_name, normalize_address, tokenize
"""

import re
import unicodedata

# ── Legal suffix expansion ────────────────────────────────────────────────────
# Maps abbreviated/variant form → canonical lowercase form.
# Applied as whole-word substitutions (word-boundary anchored).
# Order matters for multi-token patterns: longer patterns first.
_NAME_EXPANSIONS: list[tuple[str, str]] = [
    # Multi-token first
    (r"\bpvt\.?\s+ltd\.?\b",       "private limited"),
    (r"\bprivate\s+ltd\.?\b",      "private limited"),
    (r"\bpvt\.?\s+limited\b",      "private limited"),
    (r"\bpte\.?\s+ltd\.?\b",       "private limited"),
    # Single-token suffixes
    (r"\bllp\b",                   "llp"),
    (r"\bllc\b",                   "llc"),
    (r"\bltd\.?\b",                "limited"),
    (r"\bcorp\.?\b",               "corporation"),
    (r"\binc\.?\b",                "incorporated"),
    (r"\bpvt\.?\b",                "private"),
    (r"\bco\.?\b",                 "company"),
    (r"\bintl\.?\b",               "international"),
    (r"\bmfg\.?\b",                "manufacturing"),
    (r"\bassoc\.?\b",              "associates"),
    (r"\bbros\.?\b",               "brothers"),
    (r"\bservices\b",              "services"),
    # French legal forms (test set)
    (r"\bsarl\b",                  "sarl"),
    (r"\bsasu\b",                  "sasu"),
    (r"\bsas\b",                   "sas"),
    (r"\bsa\b",                    "sa"),
    (r"\bsce\b",                   "sce"),
    # Indian forms
    (r"\bprivate\s+limited\b",     "private limited"),
]

# Pre-compile with IGNORECASE
_NAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), r) for p, r in _NAME_EXPANSIONS
]

# ── Address abbreviation expansion ───────────────────────────────────────────
_ADDR_EXPANSIONS: list[tuple[str, str]] = [
    (r"\brd\.?\b",    "road"),
    (r"\bst\.?\b",    "street"),
    (r"\bave\.?\b",   "avenue"),
    (r"\bblvd\.?\b",  "boulevard"),
    (r"\bdr\.?\b",    "drive"),
    (r"\bln\.?\b",    "lane"),
    (r"\bct\.?\b",    "court"),
    (r"\bpl\.?\b",    "place"),
    (r"\bsq\.?\b",    "square"),
    (r"\bhwy\.?\b",   "highway"),
    (r"\bfwy\.?\b",   "freeway"),
    (r"\bpkwy\.?\b",  "parkway"),
    (r"\bexpy\.?\b",  "expressway"),
    (r"\bno\.?\b",    "number"),
    (r"\bapt\.?\b",   "apartment"),
    (r"\bste\.?\b",   "suite"),
    (r"\bflr\.?\b",   "floor"),
    (r"\bfl\.?\b",    "floor"),
    (r"\bn\.?\b",     "north"),
    (r"\bs\.?\b",     "south"),
    (r"\be\.?\b",     "east"),
    (r"\bw\.?\b",     "west"),
    (r"\bnw\.?\b",    "northwest"),
    (r"\bne\.?\b",    "northeast"),
    (r"\bsw\.?\b",    "southwest"),
    (r"\bse\.?\b",    "southeast"),
]

_ADDR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), r) for p, r in _ADDR_EXPANSIONS
]

# Strip only ASCII punctuation/symbols — never Unicode combining marks.
# \w with re.UNICODE strips Devanagari vowel signs (category M), so we
# explicitly enumerate only ASCII punctuation to remove.
_PUNCT_RE = re.compile(r'[!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~]')
# Collapse whitespace
_WS_RE = re.compile(r"\s+")


def _nfkc(text: str) -> str:
    """Apply Unicode NFKC normalisation."""
    return unicodedata.normalize("NFKC", text)


def normalize_name(text: str) -> str:
    """
    Normalise a business name for comparison.

    Steps:
    1. NFKC Unicode normalisation
    2. Lowercase
    3. Legal suffix expansion (Pvt Ltd → private limited, Corp → corporation …)
    4. Strip punctuation (keep alphanumeric, space, hyphen)
    5. Collapse whitespace and strip
    """
    if not text or not text.strip():
        return ""

    t = _nfkc(text).lower()

    # Apply legal suffix expansions
    for pattern, replacement in _NAME_PATTERNS:
        t = pattern.sub(replacement, t)

    # Strip punctuation, collapse whitespace
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def normalize_address(text: str) -> str:
    """
    Normalise a business address for comparison.

    Steps:
    1. NFKC Unicode normalisation
    2. Lowercase
    3. Street-type abbreviation expansion (Rd → road, St → street …)
    4. Strip punctuation (keep alphanumeric, space, hyphen)
    5. Collapse whitespace and strip
    """
    if not text or not text.strip():
        return ""

    t = _nfkc(text).lower()

    # Expand address abbreviations
    for pattern, replacement in _ADDR_PATTERNS:
        t = pattern.sub(replacement, t)

    # Strip punctuation, collapse whitespace
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def tokenize(text: str) -> frozenset[str]:
    """
    Return a frozenset of whitespace-split tokens from a normalised string.
    Used for Jaccard similarity computation.
    Tokens shorter than 2 characters are dropped (noise reduction).
    """
    if not text:
        return frozenset()
    return frozenset(tok for tok in text.split() if len(tok) >= 2)


def jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity between two normalised strings."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
