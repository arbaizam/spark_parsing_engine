"""Shared, PySpark-free regular expressions for Unicode text normalization."""

from __future__ import annotations

# Unicode's White_Space property is deliberately enumerated. Java/Spark ``\s`` does not cover the
# full set consistently across supported runtimes, while these source-data rules must not change
# with the host regex engine's shorthand behavior.
UNICODE_WHITESPACE_CLASS = (
    r"[\u0009-\u000D\u0020\u0085\u00A0\u1680"
    r"\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]"
)
UNICODE_WHITESPACE_PATTERN = UNICODE_WHITESPACE_CLASS + "+"
UNICODE_EDGE_WHITESPACE_PATTERN = (
    "^" + UNICODE_WHITESPACE_CLASS + "+|" + UNICODE_WHITESPACE_CLASS + "+$"
)
UNICODE_LIST_DELIMITER_PATTERN = UNICODE_WHITESPACE_CLASS + "*," + UNICODE_WHITESPACE_CLASS + "*"
