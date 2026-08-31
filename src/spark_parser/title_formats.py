"""Spark-native, versioned title-formatting profiles."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

# This is an intentionally closed vocabulary. Expanding it changes canonical output for existing
# inputs, so additions belong in a new versioned business-title profile.
_TITLE_BUSINESS_V1_EXCEPTIONS = (
    ("fhlb", "FHLB"),
    ("p&i", "P&I"),
    ("ust", "UST"),
    ("rcf", "RCF"),
    ("cmt", "CMT"),
)

# Letters, numbers, and underscores keep an exception embedded in an identifier from being
# mistaken for a standalone business term. Punctuation such as parentheses, slashes, apostrophes,
# and hyphens remains a valid display-text boundary.
_TITLE_TOKEN_CHARACTER_CLASS = r"[\p{L}\p{N}_]"
_BOUNDED_INTEGER_COMPONENT_PATTERN = r"(?<![\p{L}\p{N}_.,])[0-9]+$"


def _capitalize_numeric_hyphen_suffixes(value: Column) -> Column:
    """Capitalize a letter after a hyphen whose preceding component is an integer."""
    parts = F.split(value, "-", -1)
    # Spark array positions are one-based while transform indices are zero-based. Padding with an
    # empty component lets every part inspect its predecessor without ever requesting index zero.
    padded_parts = F.concat(F.array(F.lit("")), parts)
    formatted_parts = F.transform(
        parts,
        lambda part, index: F.when(
            F.element_at(padded_parts, index + F.lit(1)).rlike(_BOUNDED_INTEGER_COMPONENT_PATTERN)
            & part.rlike(r"^\p{Ll}"),
            F.concat(
                F.upper(F.substring(part, 1, 1)),
                F.substring(part, 2, 2_147_483_647),
            ),
        ).otherwise(part),
    )
    # concat_ws turns a null array into an empty string. Guard null explicitly so this profile has
    # exactly the same null contract as Spark's ordinary lower/initcap title expression.
    return F.when(value.isNull(), value).otherwise(F.concat_ws("-", formatted_parts))


def format_title_business_v1(value: Column) -> Column:
    """Apply ordinary title casing, bounded numeric hyphens, and frozen business exceptions."""
    formatted = _capitalize_numeric_hyphen_suffixes(F.initcap(F.lower(value)))
    for source, target in _TITLE_BUSINESS_V1_EXCEPTIONS:
        pattern = (
            rf"(?i)(?<!{_TITLE_TOKEN_CHARACTER_CLASS})"
            rf"{source}(?!{_TITLE_TOKEN_CHARACTER_CLASS})"
        )
        formatted = F.regexp_replace(formatted, pattern, target)
    return formatted
