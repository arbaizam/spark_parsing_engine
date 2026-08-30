"""Shared Spark column-expression primitives."""

from pyspark.sql import Column
from pyspark.sql import functions as F


def literal_column(name: str) -> Column:
    """Resolve one top-level column literally, including dots and embedded backticks."""
    # Bare ``F.col(name)`` treats dots as nested-field separators. Spark SQL identifier quoting
    # keeps the whole authored name literal; embedded backticks are escaped by doubling them.
    return F.col(f"`{name.replace('`', '``')}`")
