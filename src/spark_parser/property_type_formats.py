"""Versioned property-type normalization for loan collateral labels."""

from __future__ import annotations

import re

from pyspark.sql import Column
from pyspark.sql import functions as F

from spark_parser._spark_columns import string_map_lookup
from spark_parser._text_patterns import (
    UNICODE_EDGE_WHITESPACE_PATTERN,
    UNICODE_WHITESPACE_PATTERN,
)

_UNICODE_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_ASCII_DASHES = "-" * len(_UNICODE_DASHES)
_PYTHON_DASH_TRANSLATION = str.maketrans({dash: "-" for dash in _UNICODE_DASHES})

# Only these values may be emitted by ordinary catalog lookup. Mixed-use values form a separate,
# deliberately structured family: ``Mixed Use`` or ``Mixed Use - <preserved descriptor>``.
_CANONICAL_PROPERTY_TYPES = (
    "Single Family",
    "Townhouse",
    "Duplex",
    "PUD",
    "Two-Unit",
    "Three-Unit",
    "Four-Unit",
    "Manufactured Housing",
    "Coop",
    "Condominium",
    "Farm Land",
    "Crops or Livestock",
    "Machinery & Equipment",
    "Equipment",
    "Operating Equipment",
    "Technology Equipment",
    "Apartment - Garden",
    "Apartment - High-Rise",
    "Apartment - Low-Rise",
    "Apartment",
    "Senior Housing - Assisted Living",
    "Senior Housing",
    "Student Housing",
    "Hotel",
    "Industrial - Warehouse",
    "Industrial - Manufacturing",
    "Industrial",
    "Office - Medical Office",
    "Office",
    "Retail - Regional Mall",
    "Retail - Strip",
    "Retail",
    "Entertainment",
    "Parking",
    "Restaurant",
    "Self Storage",
    "Other",
    "Multifamily",
    "Multifamily - LIHTC",
    "Mixed Use",
)

# Meaning-changing source/vendor vocabulary stays explicit. Case, Unicode whitespace, dash style,
# and spacing around common separators are handled by the comparison key instead of being repeated
# here. Ambiguous values such as ``2-4 Unit``, ``Storage``, and ``RE/Equipment`` are intentionally
# absent so they follow the configured parse-error policy.
_PROPERTY_TYPE_ALIASES = (
    # Residential form and ownership.
    ("SFR", "Single Family"),
    ("Single Family Residential", "Single Family"),
    ("Single Family Attached", "Single Family"),
    ("Single-Family Attached", "Single Family"),
    ("Single Family Detached", "Single Family"),
    ("Single-Family Detached", "Single Family"),
    ("Single Family SemiDetached", "Single Family"),
    ("Townhome", "Townhouse"),
    ("Town House/Row House", "Townhouse"),
    ("Planned Unit Development", "PUD"),
    ("dPUD", "PUD"),
    ("PUD-Attached", "PUD"),
    ("PUD-Detached", "PUD"),
    ("MH", "Manufactured Housing"),
    ("Manufactured Home", "Manufactured Housing"),
    ("Mobile Home", "Manufactured Housing"),
    ("CO-OP", "Coop"),
    ("Condominimum", "Condominium"),
    ("Condo", "Condominium"),
    ("Condo (Stories Unknown)", "Condominium"),
    ("Non-Warrantable Condo", "Condominium"),
    ("NonWarrantable Condo", "Condominium"),
    ("Condo Low Rise (<= 4 Stories; Single Unit)", "Condominium"),
    ("Condo High Rise (> 4 Stories; single unit)", "Condominium"),
    # Exact residential unit counts remain distinct from Duplex and from range labels.
    ("2 Unit", "Two-Unit"),
    ("Two Unit", "Two-Unit"),
    ("2 Family Home", "Two-Unit"),
    ("3 Unit", "Three-Unit"),
    ("Three Unit", "Three-Unit"),
    ("3 Family Home", "Three-Unit"),
    ("4 Unit", "Four-Unit"),
    ("Four Unit", "Four-Unit"),
    ("4 Family Home", "Four-Unit"),
    # Multifamily and housing programs.
    ("Multi-Family", "Multifamily"),
    ("Multi Family", "Multifamily"),
    ("MultiFamily", "Multifamily"),
    ("MULTIFAMLY", "Multifamily"),
    ("Multifamily-Over 20", "Multifamily"),
    ("Multifamily Over 20", "Multifamily"),
    ("Multi Family LIHTC", "Multifamily - LIHTC"),
    ("Multifamily LIHTC", "Multifamily - LIHTC"),
    ("Apartments", "Apartment"),
    ("Multifamily - Student", "Student Housing"),
    ("Health Care - Independent Living", "Senior Housing"),
    ("Assisted Living", "Senior Housing - Assisted Living"),
    # Industrial, office, and retail.
    ("Warehouse", "Industrial - Warehouse"),
    ("Industrial Warehouse", "Industrial - Warehouse"),
    ("Industrial/Warehouse", "Industrial - Warehouse"),
    ("Industrial Office-Warehouse", "Industrial - Warehouse"),
    ("Industrial Flex", "Industrial"),
    ("Industrial-Flex", "Industrial"),
    ("Multi-Tenant Flex", "Industrial"),
    ("Medical Office", "Office - Medical Office"),
    ("Office-Single Tenant", "Office"),
    ("Office Single Tenant", "Office"),
    ("Multi-Tenant Office", "Office"),
    ("OFFMIDRISE", "Office"),
    ("Retail-Regional Mall", "Retail - Regional Mall"),
    ("Retail Regional Mall", "Retail - Regional Mall"),
    ("Retail-Anchored", "Retail"),
    ("Retail Anchored", "Retail"),
    ("Retail-Unanchored", "Retail"),
    ("Retail Unanchored", "Retail"),
    ("Retail-Other", "Retail"),
    ("Retail-Mall", "Retail"),
    ("Grocery Store", "Retail"),
    # Agriculture and other explicit special-purpose values.
    ("AG RE", "Farm Land"),
    ("Ag Real Estate", "Farm Land"),
    ("Crops & Livestock", "Crops or Livestock"),
    ("Crops and Livestock", "Crops or Livestock"),
    ("Other-Parking Garage", "Parking"),
    ("Parking Garage", "Parking"),
    ("Oth_Comm", "Other"),
)

_TOKEN_CHARACTER_CLASS = r"[\p{L}\p{N}_]"
_MIXED_USE_BODY = (
    rf"(?<!{_TOKEN_CHARACTER_CLASS})mixed(?: +| *- *)use"
    rf"(?!{_TOKEN_CHARACTER_CLASS})"
)
_MIXED_USE_DETECTION_PATTERN = rf"(?i){_MIXED_USE_BODY}"
_MIXED_USE_CAPTURE_PATTERN = rf"(?is)^(.*?){_MIXED_USE_BODY}(.*)$"
_MIXED_FRAGMENT_EDGE_DELIMITERS = r"^[-_/,:; ]+|[-_/,:; ]+$"
_MULTIFAMILY_FRAGMENT_PATTERN = (
    rf"(?<!{_TOKEN_CHARACTER_CLASS})multi(?:[ -]?family|famly)"
    rf"(?!{_TOKEN_CHARACTER_CLASS})"
)
_MULTIFAMILY_LIHTC_FRAGMENT_PATTERN = (
    rf"(?<!{_TOKEN_CHARACTER_CLASS})multifamily +lihtc"
    rf"(?!{_TOKEN_CHARACTER_CLASS})"
)
_FIVE_OR_MORE_UNITS_PATTERN = r"^(?:[5-9]|[1-9][0-9]+)[ -]?units?$"
_LEADING_MIXED_WRAPPERS_PATTERN = r"^(?:\( *)+"
_TRAILING_MIXED_WRAPPERS_PATTERN = r"(?: *\))+$"
_FIRST_LETTER_PATTERN = r"\p{L}"
_FIRST_LETTER_CAPTURE_PATTERN = r"(?s)^([^\p{L}]*)(\p{L})(.*)$"
_MIXED_FRAGMENT_EXCEPTIONS = (
    ("lihtc", "LIHTC"),
    ("pud", "PUD"),
    ("sfr", "SFR"),
    ("mh", "MH"),
)


def _complete_token_pattern(value: str) -> str:
    """Return a case-insensitive full-token pattern for a mixed-use descriptor."""
    return (
        rf"(?i)(?<!{_TOKEN_CHARACTER_CLASS})"
        rf"{value}(?!{_TOKEN_CHARACTER_CLASS})"
    )


def _normalize_catalog_key(value: str) -> str:
    """Mirror the runtime comparison rules for the code-owned ASCII catalog."""
    comparable = " ".join(value.translate(_PYTHON_DASH_TRANSLATION).lower().split())
    for separator in ("-", "/", "&", ","):
        comparable = re.sub(rf" *{re.escape(separator)} *", separator, comparable)
    return comparable


def _build_property_type_catalog() -> tuple[tuple[str, str], ...]:
    """Build a collision-checked lookup from canonical values and approved aliases."""
    catalog: dict[str, str] = {}

    def register(alias: str, canonical: str) -> None:
        if canonical not in _CANONICAL_PROPERTY_TYPES:
            raise ValueError(f"Unknown canonical property type: {canonical!r}.")
        key = _normalize_catalog_key(alias)
        existing = catalog.get(key)
        if existing is not None and existing != canonical:
            raise ValueError(
                f"Property-type alias {alias!r} resolves to both "
                f"{existing!r} and {canonical!r}."
            )
        catalog[key] = canonical

    for canonical in _CANONICAL_PROPERTY_TYPES:
        register(canonical, canonical)
    for alias, canonical in _PROPERTY_TYPE_ALIASES:
        register(alias, canonical)
    return tuple(sorted(catalog.items()))


_PROPERTY_TYPE_CATALOG = _build_property_type_catalog()


def _comparison_key(value: Column) -> Column:
    """Return a case-insensitive, punctuation-preserving full-value lookup key."""
    comparable = F.translate(F.lower(value), _UNICODE_DASHES, _ASCII_DASHES)
    comparable = F.regexp_replace(comparable, UNICODE_WHITESPACE_PATTERN, " ")
    comparable = F.regexp_replace(comparable, UNICODE_EDGE_WHITESPACE_PATTERN, "")
    for separator in ("-", "/", "&", ","):
        comparable = F.regexp_replace(
            comparable,
            rf" *{re.escape(separator)} *",
            separator,
        )
    return comparable


def _clean_mixed_fragment(value: Column) -> Column:
    """Trim delimiters from a fragment whose whitespace was already normalized."""
    return F.regexp_replace(value, _MIXED_FRAGMENT_EDGE_DELIMITERS, "")


def _unwrap_mixed_component(value: Column) -> Column:
    """Remove balanced enclosing parentheses while retaining separate or unmatched groups."""
    leading = F.regexp_extract(value, _LEADING_MIXED_WRAPPERS_PATTERN, 0)
    trailing = F.regexp_extract(value, _TRAILING_MIXED_WRAPPERS_PATTERN, 0)
    opening_count = F.length(F.regexp_replace(leading, " ", ""))
    closing_count = F.length(F.regexp_replace(trailing, " ", ""))
    interior = F.regexp_replace(
        value,
        _LEADING_MIXED_WRAPPERS_PATTERN + "|" + _TRAILING_MIXED_WRAPPERS_PATTERN,
        "",
    )

    def scan(state: Column, character: Column) -> Column:
        depth = state["depth"] + (
            F.when(character == "(", 1).when(character == ")", -1).otherwise(0)
        )
        return F.struct(
            depth.alias("depth"),
            F.least(state["minimum"], depth).alias("minimum"),
        )

    def unwrap(state: Column) -> Column:
        # A wrapper must enclose the entire component. The minimum interior nesting depth
        # distinguishes ``((Office))`` from ``(Office) & (Retail)`` and handles nested groups.
        balanced = (state["depth"] == closing_count) & (state["minimum"] >= 0)
        layers = F.when(
            balanced, F.least(opening_count, closing_count, state["minimum"])
        ).otherwise(0)
        wrapper_pattern = F.concat(
            F.lit(r"^(?:\( *){"), layers, F.lit(r"}|(?: *\)){"), layers, F.lit(r"}$")
        )
        return F.trim(F.regexp_replace(value, wrapper_pattern, ""))

    return F.when(
        (opening_count > 0) & (closing_count > 0),
        F.aggregate(
            F.regexp_extract_all(interior, F.lit(r"(?s)."), 0),
            F.struct(opening_count.alias("depth"), opening_count.alias("minimum")),
            scan,
            unwrap,
        ),
    ).otherwise(value)


def _format_mixed_fragment(value: Column) -> Column:
    """Sentence-capitalize each retained hyphen component in a Mixed Use descriptor."""
    normalized = F.lower(value)
    normalized = F.regexp_replace(
        normalized,
        _MULTIFAMILY_FRAGMENT_PATTERN,
        "multifamily",
    )
    normalized = F.regexp_replace(
        normalized,
        _MULTIFAMILY_LIHTC_FRAGMENT_PATTERN,
        "multifamily-lihtc",
    )
    parts = F.split(normalized, r" *-+ *", -1)

    def format_component(part: Column) -> Column:
        """Remove cosmetic wrapping parentheses and uppercase the first actual letter."""
        unwrapped = _unwrap_mixed_component(part)
        return F.when(
            unwrapped.rlike(_FIRST_LETTER_PATTERN),
            F.concat(
                F.regexp_extract(unwrapped, _FIRST_LETTER_CAPTURE_PATTERN, 1),
                F.upper(
                    F.regexp_extract(unwrapped, _FIRST_LETTER_CAPTURE_PATTERN, 2)
                ),
                F.regexp_extract(unwrapped, _FIRST_LETTER_CAPTURE_PATTERN, 3),
            ),
        ).otherwise(unwrapped)

    formatted = F.concat_ws(
        " - ",
        F.filter(
            F.transform(parts, format_component),
            lambda part: F.length(part) > F.lit(0),
        ),
    )
    for source, target in _MIXED_FRAGMENT_EXCEPTIONS:
        formatted = F.regexp_replace(formatted, _complete_token_pattern(source), target)
    return formatted


def _mixed_use_candidate(value: Column) -> tuple[Column, Column]:
    """Return whether a value is Mixed Use and its reordered, descriptor-preserving output."""
    normalized = F.translate(value, _UNICODE_DASHES, _ASCII_DASHES)
    normalized = F.regexp_replace(normalized, UNICODE_WHITESPACE_PATTERN, " ")
    normalized = F.regexp_replace(normalized, UNICODE_EDGE_WHITESPACE_PATTERN, "")
    is_mixed_use = normalized.rlike(_MIXED_USE_DETECTION_PATTERN)

    prefix = _format_mixed_fragment(
        _clean_mixed_fragment(
            F.regexp_extract(normalized, _MIXED_USE_CAPTURE_PATTERN, 1)
        )
    )
    suffix = _format_mixed_fragment(
        _clean_mixed_fragment(
            F.regexp_extract(normalized, _MIXED_USE_CAPTURE_PATTERN, 2)
        )
    )
    descriptor = (
        F.when(
            (F.length(prefix) > F.lit(0)) & (F.length(suffix) > F.lit(0)),
            F.concat(prefix, F.lit(" - "), suffix),
        )
        .when(F.length(prefix) > F.lit(0), prefix)
        .otherwise(suffix)
    )
    candidate = F.when(
        F.length(descriptor) > F.lit(0),
        F.concat(F.lit("Mixed Use - "), descriptor),
    ).otherwise(F.lit("Mixed Use"))
    return is_mixed_use, candidate


def format_property_type_v1(value: Column) -> Column:
    """Return a canonical property type, a structured Mixed Use value, or null if unknown."""
    key = _comparison_key(value)
    catalog_candidate = string_map_lookup(_PROPERTY_TYPE_CATALOG, key)
    ordinary_candidate = F.coalesce(
        catalog_candidate,
        F.when(key.rlike(_FIVE_OR_MORE_UNITS_PATTERN), F.lit("Multifamily")),
    )
    is_mixed_use, mixed_use_candidate = _mixed_use_candidate(value)
    return F.when(is_mixed_use, mixed_use_candidate).otherwise(ordinary_candidate)
