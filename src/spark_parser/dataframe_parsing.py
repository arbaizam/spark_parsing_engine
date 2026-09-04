"""Lazy parsed and audit projections over one shared Spark plan.

The runtime builds expensive normalization/parsing expressions once. This wrapper exposes a clean
target projection and a separate audit projection without duplicating the expression-building
logic or forcing an action. Configured audit keys share their final parsed target values with the
target projection. Callers choose when Spark materializes either view.
"""

from __future__ import annotations

from collections.abc import Sequence

from pyspark import StorageLevel
from pyspark.sql import DataFrame

from spark_parser._spark_columns import literal_column as _column


class DataFrameParsing:
    """Expose target and audit projections built from the same lazy Spark plan.

    Accessing properties performs only logical ``select`` operations. Data is evaluated by later
    actions such as ``collect``, ``write``, or ``count``. Persist this wrapper before materializing
    both projections when recomputing the shared parser plan would be expensive.
    """

    def __init__(
        self,
        evaluated: DataFrame,
        *,
        parsed_columns: Sequence[tuple[str, str]],
        key_columns: Sequence[str],
        result_columns: Sequence[str],
        warnings: Sequence[str] = (),
        error_mode: str = "configured",
    ) -> None:
        """Store immutable projection metadata around the runtime's evaluated logical plan."""
        # Convert every caller-owned sequence to a tuple so later mutation cannot silently alter
        # column order or result identity.
        self._evaluated = evaluated
        self._parsed_columns = tuple(parsed_columns)
        self._key_columns = tuple(key_columns)
        self._result_columns = tuple(result_columns)
        self._warnings = tuple(warnings)
        self._error_mode = error_mode

    @property
    def error_mode(self) -> str:
        """Return the execution policy independently of the authored configuration."""
        return self._error_mode

    @property
    def key_columns(self) -> tuple[str, ...]:
        """Return public row-identity fields used by :attr:`results_df`."""
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
        """Return configured targets in order, followed by parse errors in collection mode.

        Internal UUID-backed names prevent collisions while the plan is built. This final select
        restores the public target names promised by the configuration. Collection mode appends
        ``<prefix>_parse_errors`` so errors remain attached even when parsed row keys change.
        """
        return self._evaluated.select(
            *[
                _column(internal_name).alias(column_name)
                for column_name, internal_name in self._parsed_columns
            ]
        )

    @property
    def results_df(self) -> DataFrame:
        """Return target-mapped row keys followed by audit and configuration identity metadata."""
        return self._evaluated.select(
            *[_column(name) for name in (*self._key_columns, *self._result_columns)]
        )

    def persist(
        self,
        storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK_DESER,
    ) -> DataFrameParsing:
        """Persist the shared plan and return ``self`` for fluent orchestration.

        Persistence remains lazy: Spark stores partitions only after the next action. Because the
        complete evaluated parser plan is persisted, that action can surface a fail-mode error from
        a target that its selected projection would otherwise let Spark prune.
        Databricks serverless compute does not support DataFrame cache APIs; on that platform Spark
        raises its native unsupported-operation error and callers should omit this optional step.
        """
        self._evaluated.persist(storage_level)
        return self

    def unpersist(self, *, blocking: bool = False) -> DataFrameParsing:
        """Release cached partitions and return ``self``.

        Non-blocking release is the Spark default and is appropriate for most job cleanup. Select
        blocking cleanup only when subsequent resource-sensitive work must wait for eviction.
        Databricks serverless compute rejects this API just as it rejects persistence; callers must
        omit both cache operations there and the native platform error is propagated if invoked.
        """
        self._evaluated.unpersist(blocking=blocking)
        return self
