"""Closed, versioned normalization for interest-rate index display labels."""

from __future__ import annotations

import re

from pyspark.sql import Column
from pyspark.sql import functions as F

from spark_parser._text_patterns import (
    UNICODE_EDGE_WHITESPACE_PATTERN,
    UNICODE_WHITESPACE_PATTERN,
)

# Tenors are normalized only to form a comparison key. The complete key must still resolve through
# the approved catalog below, so a phrase such as ``Unknown Index 12M`` remains a parse error.
_MONTH_TENOR_PATTERN = (
    r"(?<![\p{L}\p{N}_.,])([1-9][0-9]*)[ -]?(?:months?|mos?|m)"
    r"(?![\p{L}\p{N}_])"
)
_YEAR_TENOR_PATTERN = (
    r"(?<![\p{L}\p{N}_.,])([1-9][0-9]*)[ -]?(?:years?|yrs?|y)"
    r"(?![\p{L}\p{N}_])"
)
_DAY_TENOR_PATTERN = (
    r"(?<![\p{L}\p{N}_.,])([1-9][0-9]*)[ -](?:days?)"
    r"(?![\p{L}\p{N}_])"
)
_WEEK_TENOR_PATTERN = (
    r"(?<![\p{L}\p{N}_.,])([1-9][0-9]*)[ -](?:weeks?)"
    r"(?![\p{L}\p{N}_])"
)

# Python builds the small immutable lookup catalog at import time; Spark performs every row-level
# comparison natively. These ASCII-only patterns deliberately mirror the Java/Spark expressions
# above for code-owned aliases.
_PYTHON_TENOR_RULES = (
    (re.compile(r"(?<![a-z0-9_.,])([1-9][0-9]*)[ -]?(?:months?|mos?|m)(?![a-z0-9_])"), r"\1-month"),
    (re.compile(r"(?<![a-z0-9_.,])([1-9][0-9]*)[ -]?(?:years?|yrs?|y)(?![a-z0-9_])"), r"\1-year"),
    (re.compile(r"(?<![a-z0-9_.,])([1-9][0-9]*)[ -](?:days?)(?![a-z0-9_])"), r"\1-day"),
    (re.compile(r"(?<![a-z0-9_.,])([1-9][0-9]*)[ -](?:weeks?)(?![a-z0-9_])"), r"\1-week"),
)


def _canonical_tenors(amounts: tuple[int, ...], unit: str) -> tuple[str, ...]:
    return tuple(f"{amount}-{unit}" for amount in amounts)


_CANONICAL_INTEREST_RATE_INDEXES = (
    "SOFR Term - Overnight",
    *(f"SOFR Term - {tenor}" for tenor in _canonical_tenors((1, 3, 6, 12), "Month")),
    *(
        f"{tenor} Constant Maturity Treasury (CMT)"
        for tenor in _canonical_tenors((1, 2, 3, 5, 7, 10, 30), "Year")
    ),
    "BSBY - Overnight",
    *(f"BSBY - {tenor}" for tenor in _canonical_tenors((1, 3, 6, 12), "Month")),
    "Ameribor - Overnight",
    "Ameribor - 1-Week",
    *(f"Ameribor - {tenor}" for tenor in _canonical_tenors((1, 3, 6), "Month")),
    *(f"Ameribor - {tenor}" for tenor in _canonical_tenors((1, 2), "Year")),
    *(f"Ameribor - {tenor} Average" for tenor in _canonical_tenors((30, 90), "Day")),
    "Ameribor - Derived 30T",
    "Ameribor - Derived 90T",
    "Prime",
    *(f"{tenor} LIBOR" for tenor in _canonical_tenors((1, 2, 3, 6, 9, 12), "Month")),
    "LIBOR - 1-Year (Daily)",
    *(f"Treasury - {tenor}" for tenor in _canonical_tenors((3, 6), "Month")),
    *(f"Treasury - {tenor}" for tenor in _canonical_tenors((1, 2, 3, 5, 7, 10, 30), "Year")),
    "Treasury Avg - 12-Month",
    *(f"USD Swap {tenor}" for tenor in _canonical_tenors((1, 5, 10), "Year")),
    *(f"Freddie Mac - {tenor}" for tenor in _canonical_tenors((1, 3, 6, 12), "Month")),
    *(f"FHLB - {tenor}" for tenor in _canonical_tenors((1, 2, 3, 5, 7, 10), "Year")),
    "SOFR",
    *(f"SOFR {tenor}" for tenor in _canonical_tenors((1, 12), "Month")),
    "SOFR 30-Day Average",
    "SOFR 180-Day Average",
    "SOFR - 30-Day",
    "RCF 6-Month",
    "RCF 12-Month",
)


def _normalize_catalog_key(value: str) -> str:
    """Mirror the runtime comparison rules for code-owned ASCII catalog entries."""
    comparable = " ".join(value.lower().split())
    comparable = re.sub(r" *- *", "-", comparable)
    for pattern, replacement in _PYTHON_TENOR_RULES:
        comparable = pattern.sub(replacement, comparable)
    return comparable


def _build_interest_rate_index_catalog() -> tuple[tuple[str, str], ...]:
    """Build a collision-checked lookup from canonical values and approved exact aliases."""
    catalog: dict[str, str] = {}

    def register(alias: str, canonical: str) -> None:
        if canonical not in _CANONICAL_INTEREST_RATE_INDEXES:
            raise ValueError(f"Unknown canonical interest-rate index: {canonical!r}.")
        key = _normalize_catalog_key(alias)
        existing = catalog.get(key)
        if existing is not None and existing != canonical:
            raise ValueError(
                f"Interest-rate index alias {alias!r} resolves to both "
                f"{existing!r} and {canonical!r}."
            )
        catalog[key] = canonical

    for canonical in _CANONICAL_INTEREST_RATE_INDEXES:
        register(canonical, canonical)

    # CMT shorthand is semantically explicit, so all approved CMT tenors share this compact form.
    for amount in (1, 2, 3, 5, 7, 10, 30):
        register(
            f"{amount}-Year CMT",
            f"{amount}-Year Constant Maturity Treasury (CMT)",
        )

    # Opaque source/vendor codes are exact aliases. Token boundaries keep the generic tenor rules
    # from rewriting them, so only these complete keys can resolve their source-specific meaning.
    exact_aliases = {
        "TSFR6M": "SOFR Term - 6-Month",
        "SOFR Avg - 30 days": "SOFR 30-Day Average",
        "SOFR30": "SOFR - 30-Day",
        "SOFR30A": "SOFR 30-Day Average",
        "SOFR180A": "SOFR 180-Day Average",
        "RCF6M": "RCF 6-Month",
        "RCF12M": "RCF 12-Month",
    }
    for alias, canonical in exact_aliases.items():
        register(alias, canonical)

    return tuple(sorted(catalog.items()))


_INTEREST_RATE_INDEX_CATALOG = _build_interest_rate_index_catalog()


def _comparison_key(value: Column) -> Column:
    """Return the case-insensitive, tenor-normalized full-value lookup key."""
    comparable = F.regexp_replace(F.lower(value), UNICODE_WHITESPACE_PATTERN, " ")
    comparable = F.regexp_replace(comparable, UNICODE_EDGE_WHITESPACE_PATTERN, "")
    comparable = F.regexp_replace(comparable, r" *- *", "-")
    comparable = F.regexp_replace(comparable, _MONTH_TENOR_PATTERN, "$1-month")
    comparable = F.regexp_replace(comparable, _YEAR_TENOR_PATTERN, "$1-year")
    comparable = F.regexp_replace(comparable, _DAY_TENOR_PATTERN, "$1-day")
    return F.regexp_replace(comparable, _WEEK_TENOR_PATTERN, "$1-week")


def format_interest_rate_index_v1(value: Column) -> Column:
    """Return a canonical approved interest-rate label, or null for an unknown value."""
    pairs: list[Column] = []
    for alias, canonical in _INTEREST_RATE_INDEX_CATALOG:
        pairs.extend((F.lit(alias), F.lit(canonical)))
    return F.element_at(F.create_map(*pairs), _comparison_key(value))
