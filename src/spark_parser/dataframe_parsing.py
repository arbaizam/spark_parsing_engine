"""Lazy parsed and audit projections over one shared Spark plan."""

from __future__ import annotations

from collections.abc import Sequence

from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def _column(name: str):
    """Resolve one top-level column by its literal name."""
    return F.col(f"`{name.replace('`', '``')}`")


class DataFrameParsing:
    """Expose silver and audit projections built from the same lazy plan."""

    def __init__(
        self,
        evaluated: DataFrame,
        *,
        parsed_columns: Sequence[tuple[str, str]],
        key_columns: Sequence[str],
        result_columns: Sequence[str],
        warnings: Sequence[str] = (),
    ) -> None:
        self._evaluated = evaluated
        self._parsed_columns = tuple(parsed_columns)
        self._key_columns = tuple(key_columns)
        self._result_columns = tuple(result_columns)
        self._warnings = tuple(warnings)

    @property
    def key_columns(self) -> tuple[str, ...]:
        """Return row-identity fields used by :attr:`results_df`."""
        return self._key_columns

    @property
    def result_columns(self) -> tuple[str, ...]:
        """Return parser audit/identity fields in output order."""
        return self._result_columns

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return recoverable issues found while binding the input schema."""
        return self._warnings

    @property
    def parsed_df(self) -> DataFrame:
        """Return only configured silver columns in configuration order."""
        return self._evaluated.select(
            *[
                _column(internal_name).alias(column_name)
                for column_name, internal_name in self._parsed_columns
            ]
        )

    @property
    def results_df(self) -> DataFrame:
        """Return row keys followed by nested parser audit metadata."""
        return self._evaluated.select(
            *[_column(name) for name in (*self._key_columns, *self._result_columns)]
        )

    def persist(self, storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK):
        """Persist the shared plan when both projections will be materialized."""
        self._evaluated.persist(storage_level)
        return self

    def unpersist(self, *, blocking: bool = False):
        """Release a plan previously persisted through :meth:`persist`."""
        self._evaluated.unpersist(blocking=blocking)
        return self
