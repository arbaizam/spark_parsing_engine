"""Spark-native, versioned title-formatting profiles."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

# These closed vocabularies are part of the versioned output contract. Any addition changes
# canonical output for existing inputs and requires explicit regression coverage and release notes.
_TITLE_BUSINESS_V1_EXCEPTIONS = (
    ("fhlb", "FHLB"),
    ("p&i", "P&I"),
    ("ust", "UST"),
    ("rcf", "RCF"),
    ("cmt", "CMT"),
)
_TITLE_BUSINESS_V1_FREQUENCY_ALIASES = (
    ("semiannual", "Semi-Annual"),
    ("semiannually", "Semi-Annual"),
    ("semi-annually", "Semi-Annual"),
    ("biannual", "Bi-Annual"),
    ("biannually", "Bi-Annual"),
    ("bi-annually", "Bi-Annual"),
    ("biweekly", "Bi-Weekly"),
    ("bimonthly", "Bi-Monthly"),
)
_TITLE_BUSINESS_V1_PRE_HYPHEN_ALIASES = (("yrs", "Years"),)

# Letters, numbers, and underscores keep an exception embedded in an identifier from being
# mistaken for a standalone business term. Punctuation such as parentheses, slashes, apostrophes,
# and hyphens remains a valid display-text boundary.
_TITLE_TOKEN_CHARACTER_CLASS = r"[\p{L}\p{N}_]"
_INTEGER_TIME_UNIT_PATTERN = (
    rf"(?i)(?<![\p{{L}}\p{{N}}_.,])([0-9]+) +(years|months)"
    rf"(?!{_TITLE_TOKEN_CHARACTER_CLASS})"
)


def _complete_token_pattern(value: str) -> str:
    """Return a case-insensitive pattern for one complete business-title token."""
    return (
        rf"(?i)(?<!{_TITLE_TOKEN_CHARACTER_CLASS})"
        rf"{value}(?!{_TITLE_TOKEN_CHARACTER_CLASS})"
    )


def _capitalize_hyphen_suffixes(value: Column) -> Column:
    """Capitalize the first lowercase letter in every component after an ASCII hyphen."""
    parts = F.split(value, "-", -1)
    formatted_parts = F.transform(
        parts,
        lambda part, index: F.when(
            (index > F.lit(0)) & part.rlike(r"^\p{Ll}"),
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
    """Apply title casing, business hyphen rules, and frozen complete-token replacements."""
    formatted = F.initcap(F.lower(value))
    for source, target in _TITLE_BUSINESS_V1_PRE_HYPHEN_ALIASES:
        formatted = F.regexp_replace(formatted, _complete_token_pattern(source), target)
    formatted = F.regexp_replace(
        formatted,
        _INTEGER_TIME_UNIT_PATTERN,
        "$1-$2",
    )
    formatted = _capitalize_hyphen_suffixes(formatted)
    for source, target in (
        *_TITLE_BUSINESS_V1_FREQUENCY_ALIASES,
        *_TITLE_BUSINESS_V1_EXCEPTIONS,
    ):
        formatted = F.regexp_replace(formatted, _complete_token_pattern(source), target)
    return formatted
