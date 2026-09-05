"""Shared Spark column-expression primitives."""

from collections.abc import Iterable

from pyspark.sql import Column
from pyspark.sql import functions as F


def literal_column(name: str) -> Column:
    """Resolve one top-level column literally, including dots and embedded backticks."""
    # Bare ``F.col(name)`` treats dots as nested-field separators. Spark SQL identifier quoting
    # keeps the whole authored name literal; embedded backticks are escaped by doubling them.
    return F.col(f"`{name.replace('`', '``')}`")


def string_map_lookup(entries: Iterable[tuple[str, str]], key: Column) -> Column:
    """Look up a string in a small code-owned catalog using a native Spark map."""
    pairs: list[Column] = []
    for source, target in entries:
        pairs.extend((F.lit(source), F.lit(target)))
    return F.element_at(F.create_map(*pairs), key)
