"""Build native Spark expressions that turn bronze strings into typed target values.

This module constructs a lazy logical plan; it does not collect rows or use Python/pandas UDFs.
Keeping transformations inside Spark SQL lets Catalyst optimize the plan and allows the same code
to scale from local tests to Databricks jobs. Most helpers return :class:`Column` expressions rather
than immediate Python values, which is the central concept a maintainer should keep in mind.
"""

from __future__ import annotations

import json
import re
import warnings as python_warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from pyspark.errors import AnalysisException, PySparkException
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from spark_parser._spark_columns import literal_column as _column
from spark_parser._text_patterns import (
    UNICODE_EDGE_WHITESPACE_PATTERN,
    UNICODE_LIST_DELIMITER_PATTERN,
    UNICODE_WHITESPACE_PATTERN,
)
from spark_parser.address_formats import (
    format_address_us_v1,
    format_county,
    format_state_us,
    format_zip,
)
from spark_parser.data_types import SparkDataType
from spark_parser.dataframe_parsing import DataFrameParsing
from spark_parser.defaults import BUILTIN_DATETIME_FORMAT_SHAPES
from spark_parser.enums import (
    COMPLEX_PARSER_TYPES,
    NUMERIC_PARSER_TYPES,
    BinaryEncoding,
    ChildErrorMode,
    ComplexInputFormat,
    ParseErrorMode,
    ParserType,
    StringFormat,
)
from spark_parser.exceptions import SchemaValidationError, SchemaWarning
from spark_parser.interest_rate_formats import format_interest_rate_index_v1
from spark_parser.models import (
    ColumnParser,
    NestedValueParser,
    ParserConfig,
    ParserOptions,
    StructFieldParser,
    iter_parser_options,
    needs_spark_boolean_overlap_check,
)
from spark_parser.serializer import ParserConfigSerializer
from spark_parser.title_formats import format_title_business_v1
from spark_parser.version import __version__

# Public audit schema. Define it explicitly so an empty audit array has exactly the same type as a
# populated one.
PARSE_RESULT_STRUCT = T.StructType(
    [
        T.StructField("source_column_name", T.StringType(), True),
        T.StructField("target_column_name", T.StringType(), True),
        T.StructField("parser_type", T.StringType(), True),
        T.StructField("expected_data_type", T.StringType(), True),
        T.StructField("original_value", T.StringType(), True),
        T.StructField("parsed_value", T.StringType(), True),
        T.StructField("changed", T.BooleanType(), True),
        T.StructField("effective", T.BooleanType(), True),
        T.StructField("actions_applied", T.ArrayType(T.StringType(), True), True),
        T.StructField("options", T.MapType(T.StringType(), T.StringType(), True), True),
        T.StructField("error", T.StringType(), True),
        T.StructField("nested_error_paths", T.ArrayType(T.StringType(), True), True),
        T.StructField(
            "nested_default_on_null_paths",
            T.ArrayType(T.StringType(), True),
            True,
        ),
        T.StructField(
            "nested_zero_invalidated_paths",
            T.ArrayType(T.StringType(), True),
            True,
        ),
    ]
)
PARSE_RESULT_ARRAY = T.ArrayType(PARSE_RESULT_STRUCT, containsNull=True)

# Numeric child values are decoded through strict JSON to obtain ANSI-safe casts. This pattern
# rejects partial tokens and non-JSON spellings before Spark attempts the typed conversion.
_JSON_NUMBER_PATTERN = r"\A-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\z"
# Bronze numeric input deliberately accepts a small set of useful non-JSON spellings before they
# are normalized above: a leading plus, leading zeroes, a leading decimal point, or a trailing
# decimal point. Requiring at least one digit prevents punctuation-only values such as ``.`` and
# ``-.e2`` from being rewritten into a valid zero.
_BRONZE_NUMBER_PATTERN = r"\A[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\z"
# Match the same padded standard alphabet accepted by ``base64.b64decode(validate=True)`` during
# compilation. Spark's native decoder deliberately ignores whitespace and missing padding, so it
# needs this lexical guard to keep runtime values on the same contract.
_BASE64_PATTERN = (
    r"\A(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\z"
)

# ``PERMISSIVE`` gives a null result instead of throwing for malformed containers. The runtime can
# then apply the user's explicit fail/null/default policy in a consistent way.
_STRICT_JSON_OPTIONS = {
    "mode": "PERMISSIVE",
    "allowComments": "false",
    "allowSingleQuotes": "false",
    "allowUnquotedFieldNames": "false",
    "allowNumericLeadingZeros": "false",
    "allowNonNumericNumbers": "false",
}
# Audit values must be stable, printable strings. Explicit formats prevent session settings from
# changing the JSON representation of dates and timestamps.
_AUDIT_JSON_OPTIONS = {
    "ignoreNullFields": "false",
    "dateFormat": "yyyy-MM-dd",
    "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
    "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
}

# Spark names these formats differently from the public enum. Keeping the translation in one place
# prevents ordinary parsing and compiler-validated default literals from drifting apart.
_BINARY_FORMATS = {
    BinaryEncoding.BASE64: "base64",
    BinaryEncoding.HEX: "hex",
    BinaryEncoding.UTF8: "utf-8",
}


@dataclass(frozen=True)
class _ColumnRuntimePlan:
    """Generated internal names and immutable config for one top-level column.

    Staging each major transformation under a unique name keeps projection depth constant for wide
    configurations and avoids collisions with user columns.
    """

    config: ColumnParser
    source_missing: bool
    normalized_name: str
    candidate_name: str
    post_parse_name: str
    post_zero_name: str
    final_name: str


@dataclass(frozen=True)
class _ComplexDecode:
    """Result of decoding one raw JSON/delimited container before recursive child parsing."""

    value: Column
    failed: Column


def _quoted_identifier(name: str) -> str:
    """Quote one top-level name for safe interpolation into a Spark SQL expression."""
    return f"`{name.replace('`', '``')}`"


def _is_well_formed_unicode(value: str) -> bool:
    """Return whether a public name can cross Py4J's UTF-8 boundary safely."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _ascii_resolver_key(value: str, *, case_sensitive: bool) -> str:
    """Return the exact Catalyst resolver key for already-verified ASCII names."""
    return value if case_sensitive else value.lower()


def _spark_error_class(exc: PySparkException) -> str:
    """Return Spark's structured condition across supported PySpark releases."""
    get_condition = getattr(exc, "getCondition", None)
    condition = get_condition() if get_condition is not None else exc.getErrorClass()
    return str(condition or "")


def _resolver_is_case_sensitive(df: DataFrame) -> bool:
    """Detect the active Catalyst resolver without reading a restricted SQL setting.

    Some managed Spark environments apply ``spark.sql.caseSensitive`` while denying callers
    access to the setting itself. Dropping an upper-case spelling from a one-column lower-case
    projection asks Catalyst the question directly: the field is removed by the case-insensitive
    resolver and retained by the case-sensitive resolver. Inspecting ``schema`` performs analysis
    without manufacturing an expected error or executing a Spark job.
    """
    lower_name = "__spark_parser_case_probe"
    upper_name = lower_name.upper()
    probe = df.sparkSession.range(0).select(
        F.lit(None).cast("string").alias(lower_name),
    )
    return bool(probe.drop(upper_name).schema.fields)


_SAFE_JSON_PATH_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _static_json_path_segment(name: str) -> str:
    """Return one unambiguous path segment for a configured struct field."""
    if _SAFE_JSON_PATH_FIELD.fullmatch(name):
        return f".{name}"
    # JSON escaping handles backslashes, quotes, and control characters. The path convention uses
    # single-quoted bracket notation, so apostrophes need one additional escape.
    escaped = json.dumps(name, ensure_ascii=False)[1:-1].replace("'", r"\'")
    return f"['{escaped}']"


def _resolve_input_fields(
    df: DataFrame,
    names: Sequence[str],
    *,
    case_sensitive: bool,
) -> tuple[dict[str, T.StructField], list[str], list[str]]:
    """Resolve top-level fields through Spark and classify ambiguous or missing references.

    Catalyst's case-insensitive resolver is Java ``equalsIgnoreCase``, which neither Python
    ``lower`` nor ``casefold`` reproduces for all Unicode. A public, metadata-only ``select`` uses
    the session's real resolver without an action or private JVM coupling. Exact case-sensitive and
    ASCII-only comparisons stay in Python; Unicode fallback usually takes one combined analysis,
    with individual analysis reserved for an already-invalid schema.
    """
    unique_names = tuple(dict.fromkeys(names))
    if not unique_names:
        return {}, [], []
    input_fields = tuple(df.schema.fields)
    if case_sensitive or (
        all(name.isascii() for name in unique_names)
        and all(field.name.isascii() for field in input_fields)
    ):
        grouped: dict[str, list[T.StructField]] = {}
        for field in input_fields:
            grouped.setdefault(
                _ascii_resolver_key(field.name, case_sensitive=case_sensitive),
                [],
            ).append(field)
        resolved: dict[str, T.StructField] = {}
        ambiguous: list[str] = []
        missing: list[str] = []
        for name in unique_names:
            matches = grouped.get(
                _ascii_resolver_key(name, case_sensitive=case_sensitive),
                [],
            )
            if len(matches) > 1:
                ambiguous.append(name)
            elif matches:
                resolved[name] = matches[0]
            else:
                missing.append(name)
        return resolved, sorted(ambiguous), sorted(missing)

    try:
        selected_fields = df.select(*(_column(name) for name in unique_names)).schema.fields
    except AnalysisException:
        resolved = {}
        ambiguous = []
        missing = []
        for name in unique_names:
            try:
                resolved[name] = df.select(_column(name)).schema.fields[0]
            except AnalysisException as exc:
                if _spark_error_class(exc).startswith("AMBIGUOUS_REFERENCE"):
                    ambiguous.append(name)
                else:
                    missing.append(name)
        return resolved, sorted(ambiguous), sorted(missing)

    return (
        dict(zip(unique_names, selected_fields, strict=True)),
        [],
        [],
    )


def _resolver_duplicates(
    df: DataFrame,
    names: Sequence[str],
    *,
    case_sensitive: bool,
) -> list[str]:
    """Return aliases that collide under this Spark session's active identifier resolver."""
    if len(names) < 2:
        return []
    if case_sensitive or all(name.isascii() for name in names):
        grouped: dict[str, list[str]] = {}
        for name in names:
            grouped.setdefault(
                _ascii_resolver_key(name, case_sensitive=case_sensitive),
                [],
            ).append(name)
        return sorted({name for group in grouped.values() if len(group) > 1 for name in group})
    probe = df.sparkSession.range(1).select(
        *(F.lit(None).cast("string").alias(name) for name in names)
    )
    unique_names = tuple(dict.fromkeys(names))
    try:
        _ = probe.select(*(_column(name) for name in unique_names)).schema
        return []
    except AnalysisException:
        collisions: list[str] = []
        for name in unique_names:
            try:
                _ = probe.select(_column(name)).schema
            except AnalysisException:
                collisions.append(name)
        return sorted(collisions)


def _reserved_output_conflicts(
    df: DataFrame,
    names: Sequence[str],
    *,
    case_sensitive: bool,
) -> list[str]:
    """Return generated output aliases that collide with existing input fields."""
    input_names = tuple(df.columns)
    if case_sensitive or all(name.isascii() for name in (*input_names, *names)):
        input_keys = {
            _ascii_resolver_key(name, case_sensitive=case_sensitive) for name in input_names
        }
        return sorted(
            name
            for name in names
            if _ascii_resolver_key(name, case_sensitive=case_sensitive) in input_keys
        )
    probe = df.select(
        "*",
        *(F.lit(None).cast("string").alias(name) for name in names),
    )
    try:
        _ = probe.select(*(_column(name) for name in names)).schema
        return []
    except AnalysisException:
        conflicts: list[str] = []
        for name in names:
            try:
                _ = probe.select(_column(name)).schema
            except AnalysisException:
                conflicts.append(name)
        return sorted(conflicts)


class SparkDataFrameParser:
    """Build lazy target and audit projections using only native Spark expressions.

    The compiler guarantees option consistency. This class binds that contract to a concrete input
    schema, creates collision-resistant internal columns, and exposes the final projections through
    :class:`DataFrameParsing`.
    """

    def __init__(self) -> None:
        """Create the stateless serializer reused for lineage and audit option rendering."""
        self._serializer = ParserConfigSerializer()

    def parse_dataframe(
        self,
        df: DataFrame,
        config: ParserConfig,
        *,
        key_columns: Sequence[str],
        on_missing_source: str = "fail",
        column_prefix: str = "spark_parser",
    ) -> DataFrameParsing:
        """Build lazy parsed and row-level audit projections.

        Existing configured source fields must occur exactly once and have Spark ``string`` type.
        A missing source fails binding unless ``on_missing_source='warn'`` explicitly requests a
        recoverable warning and typed null/default substitution.
        Invalid non-null values raise during the first Spark action unless their parser explicitly
        selects ``on_parse_error: null``, ``default``, or string-only ``preserve``.
        """
        if not isinstance(config, ParserConfig):
            raise TypeError("config must be a ParserConfig.")
        if not isinstance(column_prefix, str) or not column_prefix:
            raise ValueError("column_prefix must be a non-empty string.")
        if not _is_well_formed_unicode(column_prefix):
            raise ValueError("column_prefix must contain well-formed Unicode.")
        normalized_keys, schema_warnings, missing_sources = self._validate_schema(
            df,
            config,
            key_columns,
            on_missing_source,
            column_prefix,
        )

        # Generate every internal name up front. UUID-backed names prevent conflicts with unusual
        # bronze schemas and let later stages refer to expressions without nesting the entire plan.
        plans = tuple(
            self._runtime_plan(
                column_config,
                source_missing=column_config.source_column_name in missing_sources,
            )
            for column_config in config.columns
        )
        # Stage 1: apply common whitespace/null normalization to every configured source in one
        # projection. ``withColumns`` avoids a deep chain of one-column projections on wide loads.
        working = self._with_columns_checked(
            df,
            {
                plan.normalized_name: self._normalized_value(
                    self._source_value(plan),
                    plan.config.parser,
                )
                for plan in plans
            },
            config,
        )
        # Stage 2: build type-specific candidates. Complex candidates are staged as one carrier
        # struct so their value and three diagnostic arrays share one expression subtree. A
        # candidate may be null either because normalized input was null or because conversion
        # failed; the next stage distinguishes those cases.
        candidate_columns: dict[str, Column] = {}
        for plan in plans:
            candidate = self._parse_candidate(
                _column(plan.normalized_name),
                plan.config,
            )
            candidate_columns[plan.candidate_name] = (
                candidate
                if plan.config.parser.parser_type in COMPLEX_PARSER_TYPES
                else candidate.getField("value")
            )
        working = self._with_columns_checked(working, candidate_columns, config)
        # Stage 3: apply the configured fail/null/default/string-preserve behavior only to non-null
        # input that failed conversion.
        working = self._with_columns_checked(
            working,
            {
                plan.post_parse_name: self._resolve_parse_error(
                    self._candidate_value(plan),
                    self._parse_failed(plan),
                    self._source_value(plan),
                    plan.config,
                )
                for plan in plans
            },
            config,
        )
        # Stage 4: optionally invalidate numeric zero after parse-error resolution.
        working = self._with_columns_checked(
            working,
            {plan.post_zero_name: self._post_zero_value(plan) for plan in plans},
            config,
        )
        # Stage 5: enforce final nullability and apply default_on_null where required.
        working = self._with_columns_checked(
            working,
            {plan.final_name: self._final_value(plan) for plan in plans},
            config,
        )
        parsed_columns = [(plan.config.target_column_name, plan.final_name) for plan in plans]
        audit_structs = [self._audit_for_plan(plan) for plan in plans if plan.config.parser.audit]

        parse_results_name = f"{column_prefix}_parse_results"
        config_name = f"{column_prefix}_config"
        engine_version_name = f"{column_prefix}_engine_version"
        # Casting the empty-array branch is essential: otherwise Spark infers array<void>, and the
        # audit table schema would depend on whether any columns happened to enable auditing.
        parse_results = (F.array(*audit_structs) if audit_structs else F.array()).cast(
            PARSE_RESULT_ARRAY
        )
        working = self._with_columns_checked(
            working,
            {
                parse_results_name: parse_results,
                config_name: F.struct(
                    F.lit(config.parser_config_id).alias("id"),
                    F.lit(config.version).alias("version"),
                    F.lit(self._serializer.content_hash(config)).alias("content_hash"),
                ),
                engine_version_name: F.lit(__version__),
            },
            config,
        )
        # Drop bronze and temporary staging columns from the shared plan. Row keys are retained only
        # for results_df; parsed_df later selects the target internal names and restores aliases.
        working = working.select(
            *[_column(name) for name in normalized_keys],
            *[_column(plan.final_name) for plan in plans],
            _column(parse_results_name),
            _column(config_name),
            _column(engine_version_name),
        )
        return DataFrameParsing(
            working,
            parsed_columns=parsed_columns,
            key_columns=normalized_keys,
            result_columns=(parse_results_name, config_name, engine_version_name),
            warnings=schema_warnings,
        )

    @classmethod
    def _with_columns_checked(
        cls,
        df: DataFrame,
        columns: dict[str, Column],
        config: ParserConfig,
    ) -> DataFrame:
        """Add one projection stage and explain Spark's fixed-point analyzer exhaustion.

        ``withColumns`` performs metadata-only analysis but can exhaust Spark's configurable
        resolution-iteration budget for very deeply nested higher-order expressions. Translate the
        otherwise opaque Py4J/JVM failure while leaving every unrelated analysis error untouched.
        """
        try:
            return df.withColumns(columns)
        except Exception as exc:
            cls._raise_resource_exhaustion(df, config, exc)
            raise

    @classmethod
    def _raise_resource_exhaustion(
        cls,
        df: DataFrame,
        config: ParserConfig,
        exc: Exception,
    ) -> None:
        """Translate analyzer/driver resource exhaustion from any metadata-analysis boundary."""
        message = str(exc)
        iteration_exhausted = (
            "Max iterations (" in message and "reached for batch Resolution" in message
        )
        stack_exhausted = "java.lang.StackOverflowError" in message
        if not iteration_exhausted and not stack_exhausted:
            return
        maximum_depth = max(cls._data_type_nesting(column.data_type) for column in config.columns)
        if stack_exhausted:
            raise SchemaValidationError(
                "Spark exhausted the driver JVM thread stack while resolving the combined input "
                "and parser plan. The maximum configured complex nesting depth is "
                f"{maximum_depth}. Increase the driver JVM thread-stack size before Spark starts "
                "or reduce parser nesting. No Spark data action was started."
            ) from exc
        try:
            setting = df.sparkSession.conf.get("spark.sql.analyzer.maxIterations", "100")
            setting_description = f"spark.sql.analyzer.maxIterations={setting}"
        except PySparkException:
            # Managed Spark runtimes can enforce this limit while withholding the setting. It is
            # explanatory context only; never mask the analyzer failure with a second access error.
            setting_description = "the runtime's spark.sql.analyzer.maxIterations limit"
        raise SchemaValidationError(
            "Spark could not resolve the combined input and parser plan within "
            f"{setting_description}. The maximum configured complex nesting depth is "
            f"{maximum_depth}. Increase spark.sql.analyzer.maxIterations for intentionally deep "
            "schemas or reduce parser nesting. No Spark data action was started."
        ) from exc

    @classmethod
    def _data_type_nesting(cls, data_type: SparkDataType) -> int:
        """Return the greatest number of complex containers on one datatype path."""
        if data_type.parser_type is ParserType.ARRAY:
            assert data_type.element_type is not None
            return 1 + cls._data_type_nesting(data_type.element_type)
        if data_type.parser_type is ParserType.MAP:
            assert data_type.key_type is not None and data_type.value_type is not None
            return 1 + max(
                cls._data_type_nesting(data_type.key_type),
                cls._data_type_nesting(data_type.value_type),
            )
        if data_type.parser_type is ParserType.STRUCT:
            return 1 + max(
                (cls._data_type_nesting(field.data_type) for field in data_type.fields),
                default=0,
            )
        return 0

    @classmethod
    def _case_sensitivity_for_schema(cls, df: DataFrame, config: ParserConfig) -> bool:
        """Probe Catalyst's resolver and translate metadata-analysis resource failures."""
        try:
            return _resolver_is_case_sensitive(df)
        except Exception as exc:
            cls._raise_resource_exhaustion(df, config, exc)
            raise

    def _validate_schema(
        self,
        df: DataFrame,
        config: ParserConfig,
        key_columns: Sequence[str],
        on_missing_source: str,
        column_prefix: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...], frozenset[str]]:
        """Validate the bronze schema and return normalized result keys plus recoverable warnings.

        Schema inspection is metadata-only and does not trigger a Spark action. Missing configured
        sources are recoverable only through an explicit warn policy; ambiguous or non-string
        sources are not recoverable and fail immediately.
        """
        if not isinstance(on_missing_source, str):
            raise TypeError("on_missing_source must be 'fail' or 'warn'.")
        if on_missing_source not in {"fail", "warn"}:
            raise ValueError("on_missing_source must be 'fail' or 'warn'.")
        if key_columns is None:
            raise SchemaValidationError("key_columns is required and must be supplied explicitly.")
        if isinstance(key_columns, (str, bytes)):
            raise TypeError("key_columns must be a sequence, not a string.")
        try:
            normalized_keys = tuple(key_columns)
        except TypeError as exc:
            raise TypeError("key_columns must be a sequence of column names.") from exc
        if not normalized_keys:
            raise SchemaValidationError("key_columns must contain at least one column name.")
        invalid_keys = [name for name in normalized_keys if not isinstance(name, str) or not name]
        if invalid_keys:
            raise SchemaValidationError("key_columns must contain only non-empty strings.")
        if any(not _is_well_formed_unicode(name) for name in normalized_keys):
            raise SchemaValidationError("key_columns must contain well-formed Unicode.")

        case_sensitive = self._case_sensitivity_for_schema(df, config)
        self._validate_spark_boolean_overlap(df, config)
        self._validate_custom_datetime_policy(df, config)
        configured_names = {column.source_column_name for column in config.columns}
        configured_output_names = {column.target_column_name for column in config.columns}
        if any(
            not _is_well_formed_unicode(name) for name in configured_names | configured_output_names
        ):
            raise SchemaValidationError(
                "Configured source and target column names must contain well-formed Unicode."
            )
        resolved_sources, ambiguous, missing = _resolve_input_fields(
            df,
            sorted(configured_names),
            case_sensitive=case_sensitive,
        )
        if ambiguous:
            raise SchemaValidationError(f"Configured input columns are ambiguous: {ambiguous}.")
        schema_warnings: list[str] = []
        if missing:
            if on_missing_source == "fail":
                raise SchemaValidationError(
                    f"Configured source columns are missing: {missing}. Pass "
                    "on_missing_source='warn' only when typed null/default substitution is "
                    "intentional."
                )
            message = (
                "Configured source columns are missing and will produce typed null/default "
                f"values: {missing}."
            )
            schema_warnings.append(message)
            python_warnings.warn(message, SchemaWarning, stacklevel=3)
        non_string = {
            name: resolved_sources[name].dataType.simpleString()
            for name in sorted(configured_names - set(missing))
            if not isinstance(resolved_sources[name].dataType, T.StringType)
        }
        if non_string:
            raise SchemaValidationError(
                f"Configured bronze columns must have Spark string type; found {non_string}."
            )

        output_names = (
            f"{column_prefix}_parse_results",
            f"{column_prefix}_config",
            f"{column_prefix}_engine_version",
        )
        conflicts = _reserved_output_conflicts(
            df,
            output_names,
            case_sensitive=case_sensitive,
        )
        if conflicts:
            raise SchemaValidationError(
                f"Input contains reserved parser output columns: {conflicts}."
            )
        target_names = tuple(column.target_column_name for column in config.columns)
        target_conflicts = _resolver_duplicates(
            df,
            target_names,
            case_sensitive=case_sensitive,
        )
        if target_conflicts:
            raise SchemaValidationError(
                "Configured target columns collide under Spark's active identifier resolver: "
                f"{target_conflicts}."
            )
        duplicate_keys = _resolver_duplicates(
            df,
            normalized_keys,
            case_sensitive=case_sensitive,
        )
        if duplicate_keys:
            raise SchemaValidationError(f"key_columns contains duplicates: {duplicate_keys}.")
        _, ambiguous_keys, missing_keys = _resolve_input_fields(
            df,
            normalized_keys,
            case_sensitive=case_sensitive,
        )
        if missing_keys:
            raise SchemaValidationError(f"key_columns are missing: {missing_keys}.")
        if ambiguous_keys:
            raise SchemaValidationError(f"key_columns are ambiguous: {ambiguous_keys}.")
        result_key_conflicts = _resolver_duplicates(
            df,
            (*normalized_keys, *output_names),
            case_sensitive=case_sensitive,
        )
        result_key_conflicts = sorted(set(normalized_keys) & set(result_key_conflicts))
        if result_key_conflicts:
            raise SchemaValidationError(
                f"key_columns conflict with parser result columns: {result_key_conflicts}."
            )
        self._validate_nested_resolver_contracts(
            df,
            config,
            case_sensitive=case_sensitive,
        )
        return normalized_keys, tuple(schema_warnings), frozenset(missing)

    @classmethod
    def _validate_nested_resolver_contracts(
        cls,
        df: DataFrame,
        config: ParserConfig,
        *,
        case_sensitive: bool,
    ) -> None:
        """Reject nested struct aliases that Spark would resolve ambiguously at runtime."""
        collision_cache: dict[tuple[str, ...], list[str]] = {}

        def collisions(names: tuple[str, ...]) -> list[str]:
            if names not in collision_cache:
                collision_cache[names] = _resolver_duplicates(
                    df,
                    names,
                    case_sensitive=case_sensitive,
                )
            return collision_cache[names]

        for column in config.columns:
            for options in iter_parser_options(column.parser):
                if options.parser_type is not ParserType.STRUCT:
                    continue
                source_names = tuple(field.source_field_name for field in options.field_parsers)
                if any(not _is_well_formed_unicode(name) for name in source_names):
                    raise SchemaValidationError(
                        "Nested struct source fields must contain well-formed Unicode."
                    )
                source_conflicts = collisions(source_names)
                if source_conflicts:
                    raise SchemaValidationError(
                        "Nested struct source fields collide under Spark's active identifier "
                        f"resolver for target column {column.target_column_name!r}: "
                        f"{source_conflicts}."
                    )
                target_names = tuple(field.target_field_name for field in options.field_parsers)
                if any(not _is_well_formed_unicode(name) for name in target_names):
                    raise SchemaValidationError(
                        "Nested struct target fields must contain well-formed Unicode."
                    )
                target_conflicts = collisions(target_names)
                if target_conflicts:
                    raise SchemaValidationError(
                        "Nested struct target fields collide under Spark's active identifier "
                        f"resolver for target column {column.target_column_name!r}: "
                        f"{target_conflicts}."
                    )

    @classmethod
    def _validate_custom_datetime_policy(cls, df: DataFrame, config: ParserConfig) -> None:
        """Reject unsafe custom datetime parsing before Spark can bypass parser error policy."""
        custom_formats = sorted(
            {
                datetime_format
                for column in config.columns
                for options in iter_parser_options(column.parser)
                if options.parser_type
                in {ParserType.DATE, ParserType.TIMESTAMP, ParserType.TIMESTAMP_NTZ}
                for datetime_format in options.formats
                if datetime_format not in BUILTIN_DATETIME_FORMAT_SHAPES
            }
        )
        if not custom_formats:
            return
        try:
            policy_setting = df.sparkSession.conf.get(
                "spark.sql.legacy.timeParserPolicy",
                "EXCEPTION",
            )
        except PySparkException as exc:
            raise SchemaValidationError(
                "Custom datetime formats require access to "
                "spark.sql.legacy.timeParserPolicy so the parser can verify CORRECTED mode. "
                "The active Spark runtime withholds that setting; use built-in datetime formats "
                "or enable the setting for this workload."
            ) from exc
        policy = (policy_setting or "EXCEPTION").upper()
        if policy != "CORRECTED":
            raise SchemaValidationError(
                "Custom datetime formats require "
                "spark.sql.legacy.timeParserPolicy=CORRECTED so malformed values follow the "
                f"configured parser error policy; current policy is {policy!r}. Custom formats: "
                f"{custom_formats}."
            )
        probe_session = cls._metadata_probe_session(df, time_parser_policy=policy)
        for datetime_format in custom_formats:
            try:
                # ``schema`` stops at analysis, where Spark has not yet compiled the formatter.
                # Public ``inputFiles`` reaches optimization without executing a Spark job, so an
                # invalid extension pattern fails here instead of escaping on_parse_error later.
                probe = probe_session.range(0).select(
                    F.try_to_timestamp(F.lit(""), F.lit(datetime_format)).alias("parsed")
                )
                probe.inputFiles()
            except PySparkException as exc:
                raise SchemaValidationError(
                    f"Custom datetime format {datetime_format!r} is invalid for the active "
                    "Spark runtime."
                ) from exc

    @classmethod
    def _validate_spark_boolean_overlap(cls, df: DataFrame, config: ParserConfig) -> None:
        """Validate non-ASCII case-insensitive vocabularies with Spark, without a data job.

        Python and Spark may use different Unicode tables, and Spark releases can differ even on
        the same JVM. ``from_json`` requires a foldable schema expression and evaluates that schema
        during analysis. Selecting between two valid DDL strings therefore exposes Spark's own
        normalization result through metadata alone, even when optimizer constant folding is
        disabled and even for empty DataFrames or nested Boolean parsers with empty containers.
        """
        checked: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for column in config.columns:
            for options in iter_parser_options(column.parser):
                if options.parser_type is not ParserType.BOOLEAN or not (
                    needs_spark_boolean_overlap_check(
                        options.true_values,
                        options.false_values,
                        options.boolean_case_sensitive,
                    )
                ):
                    continue
                signature = (options.true_values, options.false_values)
                if signature in checked:
                    continue
                checked.add(signature)
                true_values = F.array(
                    *(F.lower(F.lit(value)) for value in options.true_values)
                )
                false_values = F.array(
                    *(F.lower(F.lit(value)) for value in options.false_values)
                )
                overlap = F.arrays_overlap(true_values, false_values)
                schema_ddl = F.when(overlap, F.lit("STRUCT<overlap: INT>")).otherwise(
                    F.lit("STRUCT<disjoint: INT>")
                )
                probe_type = (
                    df.sparkSession.range(0)
                    .select(F.from_json(F.lit("{}"), schema_ddl).alias("validated"))
                    .schema["validated"]
                    .dataType
                )
                field_names = (
                    probe_type.fieldNames() if isinstance(probe_type, T.StructType) else []
                )
                if field_names == ["overlap"]:
                    raise SchemaValidationError(
                        "Boolean true_values and false_values overlap under the active Spark "
                        "runtime's case-insensitive Unicode normalization for target column "
                        f"{column.target_column_name!r}. Use disjoint token sets or enable "
                        "boolean_case_sensitive. No Spark data action was started."
                    )
                if field_names != ["disjoint"]:
                    raise SchemaValidationError(
                        "Spark did not return the expected metadata shape while validating "
                        "case-insensitive Boolean vocabularies. Ensure the built-in from_json "
                        "function is not shadowed, then retry. No Spark data action was started."
                    )

    @classmethod
    def _metadata_probe_session(
        cls,
        df: DataFrame,
        *,
        time_parser_policy: str,
    ) -> SparkSession:
        """Return an isolated session whose optimizer can evaluate configuration constants.

        Callers may legitimately exclude Catalyst rules or lower optimizer iteration limits for
        their data plans. Metadata validation must neither inherit those settings nor mutate them,
        so classic Spark uses a fresh SQL session with constant folding restored. Spark Connect
        does not expose ``newSession`` or arbitrary optimizer settings; there the caller session is
        safe only after a known-invalid control proves that optimization is actually evaluating
        foldable expressions. Neither path executes a Spark data job.
        """
        caller_session = df.sparkSession
        new_session = getattr(caller_session, "newSession", None)
        if not callable(new_session):
            probe_session = caller_session
        else:
            try:
                probe_session = new_session()
            except PySparkException as exc:
                if _spark_error_class(exc) not in {
                    "JVM_ATTRIBUTE_NOT_SUPPORTED",
                    "NOT_IMPLEMENTED",
                }:
                    raise
                probe_session = caller_session
            else:
                try:
                    probe_session.conf.set("spark.sql.optimizer.excludedRules", "")
                    probe_session.conf.set("spark.sql.optimizer.maxIterations", "100")
                    probe_session.conf.set(
                        "spark.sql.legacy.timeParserPolicy",
                        time_parser_policy,
                    )
                except PySparkException as exc:
                    if _spark_error_class(exc) not in {
                        "CANNOT_MODIFY_CONFIG",
                        "CONFIG_NOT_AVAILABLE",
                    }:
                        raise
                    probe_session = caller_session

        control = probe_session.range(0).select(
            F.try_to_timestamp(F.lit(""), F.lit("invalid[")).alias("validated")
        )
        try:
            control.inputFiles()
        except PySparkException as exc:
            condition = _spark_error_class(exc)
            if condition.startswith("INVALID_DATETIME_PATTERN") or (
                not condition and "Illegal pattern character" in str(exc)
            ):
                return probe_session
            raise
        raise SchemaValidationError(
            "Custom datetime formats cannot be validated because the active Spark optimizer did "
            "not evaluate a foldable metadata probe. Restore Catalyst constant folding or use a "
            "built-in datetime format. No Spark data action was started."
        )

    @staticmethod
    def _runtime_plan(
        column_config: ColumnParser,
        *,
        source_missing: bool,
    ) -> _ColumnRuntimePlan:
        """Allocate collision-resistant internal column names for one configured output."""
        token = uuid4().hex
        return _ColumnRuntimePlan(
            config=column_config,
            source_missing=source_missing,
            normalized_name=f"__spark_parser_normalized_{token}",
            candidate_name=f"__spark_parser_candidate_{token}",
            post_parse_name=f"__spark_parser_post_parse_{token}",
            post_zero_name=f"__spark_parser_post_zero_{token}",
            final_name=f"__spark_parser_final_{token}",
        )

    @staticmethod
    def _source_value(plan: _ColumnRuntimePlan) -> Column:
        """Return the literal bronze source or a typed null for a missing configured column."""
        return (
            F.lit(None).cast("string")
            if plan.source_missing
            else _column(plan.config.source_column_name)
        )

    def _normalization_state(
        self,
        source: Column,
        options: ParserOptions,
    ) -> tuple[Column, Column, Column]:
        """Build reusable normalization expressions and audit flags for one source value.

        The returned tuple contains the whitespace-normalized value plus independent flags for
        empty-to-null and null-marker matches. Keeping the flags separate lets audit construction
        explain exactly why the final value changed.
        """
        # The compiler resolves outer complex parsers to collapse_whitespace=false. Trust that
        # invariant here so the rule has one executable source of truth.
        whitespace_normalized = (
            F.regexp_replace(source, UNICODE_WHITESPACE_PATTERN, " ")
            if options.collapse_whitespace
            else source
        )
        if options.trim_whitespace:
            whitespace_normalized = F.regexp_replace(
                whitespace_normalized,
                UNICODE_EDGE_WHITESPACE_PATTERN,
                "",
            )
        # Coalesce each Boolean predicate to false so SQL three-valued null logic never leaks into
        # the audit ``changed`` calculation.
        empty_to_null = F.coalesce(
            (whitespace_normalized == "") & F.lit(options.empty_is_null),
            F.lit(False),
        )
        marker_match = self._null_marker_match(whitespace_normalized, options)
        return whitespace_normalized, empty_to_null, marker_match

    def _normalized_value(self, source: Column, options: ParserOptions) -> Column:
        """Apply common normalization in documented order and return a string expression."""
        whitespace_normalized, empty_to_null, marker_match = self._normalization_state(
            source,
            options,
        )
        json_null = self._json_null_match(whitespace_normalized, options)
        # Empty and the exact complex JSON literal ``null`` take precedence over configurable null
        # markers. The precedence is mirrored in the ordered audit actions.
        return (
            F.when(empty_to_null | json_null, F.lit(None).cast("string"))
            .when(
                marker_match & F.lit(options.replace_null_markers),
                F.lit(None).cast("string"),
            )
            .otherwise(whitespace_normalized)
        )

    @staticmethod
    def _json_null_match(value: Column, options: ParserOptions) -> Column:
        """Recognize only the exact lowercase JSON null literal for complex containers."""
        if options.parser_type not in COMPLEX_PARSER_TYPES:
            return F.lit(False)
        return F.coalesce(value == F.lit("null"), F.lit(False))

    @staticmethod
    def _candidate_value(plan: _ColumnRuntimePlan) -> Column:
        """Read the typed value from a scalar column or staged complex carrier."""
        candidate = _column(plan.candidate_name)
        return (
            candidate.getField("value")
            if plan.config.parser.parser_type in COMPLEX_PARSER_TYPES
            else candidate
        )

    @classmethod
    def _parse_failed(cls, plan: _ColumnRuntimePlan) -> Column:
        """Identify non-null normalized input whose typed candidate is null."""
        return _column(plan.normalized_name).isNotNull() & cls._candidate_value(plan).isNull()

    @staticmethod
    def _zero_invalidated(plan: _ColumnRuntimePlan) -> Column:
        """Return a null-safe flag for a successfully parsed numeric zero that is disallowed."""
        options = plan.config.parser
        if options.parser_type in NUMERIC_PARSER_TYPES and not options.zero_is_valid:
            return F.coalesce(
                _column(plan.post_parse_name) == F.lit(0).cast(plan.config.expected_data_type),
                F.lit(False),
            )
        return F.lit(False)

    def _post_zero_value(self, plan: _ColumnRuntimePlan) -> Column:
        """Convert disallowed numeric zero to a correctly typed null."""
        if (
            plan.config.parser.parser_type in NUMERIC_PARSER_TYPES
            and not plan.config.parser.zero_is_valid
        ):
            return F.when(
                self._zero_invalidated(plan),
                F.lit(None).cast(plan.config.expected_data_type),
            ).otherwise(_column(plan.post_parse_name))
        return _column(plan.post_parse_name)

    @staticmethod
    def _default_on_null_applied(plan: _ColumnRuntimePlan) -> Column:
        """Return whether final nullability requires assigning ``default_on_null``."""
        return (
            _column(plan.post_zero_name).isNull()
            if not plan.config.parser.is_nullable
            else F.lit(False)
        )

    def _final_value(self, plan: _ColumnRuntimePlan) -> Column:
        """Apply the final nullability contract after parsing and zero handling."""
        options = plan.config.parser
        if not options.is_nullable:
            return F.when(
                self._default_on_null_applied(plan),
                self._typed_value_literal(
                    options.default_on_null,
                    plan.config.data_type,
                    options,
                ),
            ).otherwise(_column(plan.post_zero_name))
        return _column(plan.post_zero_name)

    def _audit_for_plan(self, plan: _ColumnRuntimePlan) -> Column:
        """Rebuild explanatory flags and assemble one top-level audit struct expression."""
        # These expressions intentionally mirror normalization rather than reading only the final
        # value. Two different actions can lead to the same null, and audit consumers need the cause.
        source = self._source_value(plan)
        whitespace_normalized, empty_to_null, marker_match = self._normalization_state(
            source,
            plan.config.parser,
        )
        json_null = self._json_null_match(whitespace_normalized, plan.config.parser)
        return self._audit_struct(
            plan.config,
            source=source,
            parsed=_column(plan.final_name),
            empty_to_null=empty_to_null,
            marker_replaced=(
                marker_match
                & F.lit(plan.config.parser.replace_null_markers)
                & ~empty_to_null
                & ~json_null
            ),
            json_null=json_null,
            parse_failed=self._parse_failed(plan),
            zero_invalidated=self._zero_invalidated(plan),
            default_on_null_applied=self._default_on_null_applied(plan),
            source_missing=F.lit(plan.source_missing),
            normalized=_column(plan.normalized_name),
            candidate=self._candidate_value(plan),
            nested_error_paths=self._candidate_diagnostic(plan, "error_paths"),
            nested_default_on_null_paths=self._candidate_diagnostic(
                plan,
                "default_on_null_paths",
            ),
            nested_zero_invalidated_paths=self._candidate_diagnostic(
                plan,
                "zero_invalidated_paths",
            ),
        )

    def _candidate_diagnostic(self, plan: _ColumnRuntimePlan, field_name: str) -> Column:
        """Read a staged complex path array; scalar candidates have no nested paths."""
        if plan.config.parser.parser_type not in COMPLEX_PARSER_TYPES:
            return self._empty_error_paths()
        return _column(plan.candidate_name).getField(field_name)

    def _parse_candidate(
        self,
        normalized: Column,
        column_config: ColumnParser,
    ) -> Column:
        """Build the initial typed candidate and all nested diagnostic arrays for one column."""
        if column_config.parser.parser_type in COMPLEX_PARSER_TYPES:
            emit_diagnostics = column_config.parser.audit
            decoded = self._decode_complex(
                normalized,
                column_config.data_type,
                column_config.parser,
            )
            nested = self._parse_nested_complex(
                decoded.value,
                column_config.data_type,
                column_config.parser,
                F.lit("$"),
                column_config,
                emit_diagnostics=emit_diagnostics,
            )
            empty_paths = self._empty_error_paths()
            # Guard the complete state, not only its value. A descendant map_from_entries can throw
            # DUPLICATED_MAP_KEY and must not be evaluated after the outer validator has already
            # classified the raw container as invalid.
            guarded_state = F.when(
                decoded.failed,
                self._parse_state(
                    F.lit(None).cast(column_config.expected_data_type),
                    F.lit(False),
                    empty_paths,
                    empty_paths,
                    empty_paths,
                ),
            ).otherwise(nested)

            def build_candidate(state: Column) -> Column:
                if not emit_diagnostics:
                    return self._parse_state(
                        state.getField("value"),
                        F.lit(False),
                        empty_paths,
                        empty_paths,
                        empty_paths,
                    )
                return self._parse_state(
                    state.getField("value"),
                    F.lit(False),
                    F.concat(
                        state.getField("error_paths"),
                        self._paths_when(decoded.failed, F.lit("$")),
                    ),
                    state.getField("default_on_null_paths"),
                    state.getField("zero_invalidated_paths"),
                )

            return self._bind_once(guarded_state, build_candidate)
        # Top-level and nested leaves deliberately use the same conversion helper. This is a
        # contract invariant: a token such as ``1d`` must not succeed or fail solely because it is
        # located at a different nesting depth.
        empty_paths = self._empty_error_paths()
        return self._parse_state(
            self._parse_scalar_candidate(
                normalized,
                column_config.data_type,
                column_config.parser,
            ),
            F.lit(False),
            empty_paths,
            empty_paths,
            empty_paths,
        )

    def _parse_scalar_candidate(
        self,
        normalized: Column,
        data_type: SparkDataType,
        options: ParserOptions,
    ) -> Column:
        """Build a scalar candidate expression; invalid non-null input returns typed null."""
        parser_type = data_type.parser_type
        if parser_type is ParserType.STRING:
            return self._parse_string_candidate(normalized, options.string_format)
        if parser_type in NUMERIC_PARSER_TYPES:
            # Nested values arrive as strings after JSON decoding. Normalize safe alternate numeric
            # spellings, then wrap the token in a tiny JSON object so from_json performs an
            # ANSI-independent typed conversion.
            json_number = F.regexp_replace(normalized, r"^\+", "")
            json_number = F.regexp_replace(json_number, r"^(-?)0+(?=\d)", "$1")
            json_number = F.regexp_replace(json_number, r"^\.", "0.")
            json_number = F.regexp_replace(json_number, r"^-\.", "-0.")
            json_number = F.regexp_replace(json_number, r"\.$", "")
            json_number = F.regexp_replace(json_number, r"\.(?=[eE])", "")
            decode_type = "short" if parser_type is ParserType.BYTE else data_type.canonical
            parsed = F.from_json(
                F.concat(F.lit('{"value":'), json_number, F.lit("}")),
                f"struct<value:{decode_type}>",
                _STRICT_JSON_OPTIONS,
            ).getField("value")
            # ``from_json`` may accept a valid prefix in some runtime combinations. The explicit
            # full-token regex prevents wrapper injection and trailing garbage from slipping through.
            parsed = F.when(
                normalized.rlike(_BRONZE_NUMBER_PATTERN) & json_number.rlike(_JSON_NUMBER_PATTERN),
                parsed,
            ).otherwise(F.lit(None).cast(data_type.canonical))
            if parser_type is ParserType.BYTE:
                # Spark's JSON decoder accepts 128..255 as unsigned bytes and then wraps them into
                # negative signed values. Decode through short and narrow only after an explicit
                # signed-byte range check so overflow remains a parse error.
                return F.when(
                    parsed.between(-128, 127),
                    parsed.cast("byte"),
                ).otherwise(F.lit(None).cast("byte"))
            if parser_type in {ParserType.FLOAT, ParserType.DOUBLE}:
                return F.when(
                    F.isnan(parsed) | (F.abs(parsed) == F.lit(float("inf"))),
                    F.lit(None).cast(data_type.canonical),
                ).otherwise(parsed)
            return parsed
        if parser_type is ParserType.BINARY:
            parsed = F.try_to_binary(
                normalized,
                F.lit(_BINARY_FORMATS[options.binary_encoding]),
            )
            if options.binary_encoding is BinaryEncoding.BASE64:
                return F.when(
                    normalized.rlike(_BASE64_PATTERN),
                    parsed,
                ).otherwise(F.lit(None).cast(data_type.canonical))
            return parsed
        if parser_type is ParserType.BOOLEAN:
            # Exact/ASCII overlap was rejected by the compiler, and non-ASCII overlap was checked
            # once at DataFrame binding with this runtime's Unicode tables.
            comparable = normalized if options.boolean_case_sensitive else F.lower(normalized)
            true_values = (
                options.true_values
                if options.boolean_case_sensitive
                else tuple(F.lower(F.lit(value)) for value in options.true_values)
            )
            false_values = (
                options.false_values
                if options.boolean_case_sensitive
                else tuple(F.lower(F.lit(value)) for value in options.false_values)
            )
            parsed = (
                F.when(comparable.isin(*true_values), F.lit(True))
                .when(comparable.isin(*false_values), F.lit(False))
                .otherwise(F.lit(None).cast("boolean"))
            )
            return parsed
        if parser_type in {ParserType.DATE, ParserType.TIMESTAMP, ParserType.TIMESTAMP_NTZ}:
            # Try formats in author-supplied order and select the first successful result. Built-in
            # formats are shape-guarded so Spark 3.5's EXCEPTION time-parser policy cannot
            # turn a harmless format mismatch into a job-level SparkUpgradeException.
            if parser_type in {ParserType.DATE, ParserType.TIMESTAMP_NTZ}:
                candidates = [
                    self._timestamp_ntz_candidate(normalized, datetime_format)
                    for datetime_format in options.formats
                ]
                parsed = F.coalesce(*candidates)
                # A date represents the calendar fields the author supplied, not the instant those
                # fields denote in a session time zone. Parse through TimestampNTZ before narrowing
                # so an explicit input offset cannot move the value to an adjacent day.
                return parsed.cast("date") if parser_type is ParserType.DATE else parsed
            candidates = [
                self._timestamp_candidate(normalized, datetime_format)
                for datetime_format in options.formats
            ]
            return F.coalesce(*candidates)
        raise ValueError(f"Unsupported parser type: {parser_type.value}.")

    @staticmethod
    def _parse_string_candidate(
        normalized: Column,
        string_format: StringFormat | None,
    ) -> Column:
        """Apply one closed string-format profile and reject future unimplemented enum members."""
        if string_format is None:
            return normalized
        if string_format is StringFormat.LOWER:
            return F.lower(normalized)
        if string_format is StringFormat.UPPER:
            return F.upper(normalized)
        if string_format is StringFormat.TITLE:
            # Unlike Pascal casing, title casing intentionally retains the normalized spaces.
            # Lowercasing first makes output deterministic for mixed-case bronze values.
            return F.initcap(F.lower(normalized))
        if string_format is StringFormat.TITLE_BUSINESS_V1:
            return format_title_business_v1(normalized)
        if string_format is StringFormat.INTEREST_RATE_INDEX_V1:
            return format_interest_rate_index_v1(normalized)
        if string_format is StringFormat.PASCAL:
            return F.regexp_replace(
                F.initcap(F.lower(normalized)),
                UNICODE_WHITESPACE_PATTERN,
                "",
            )
        if string_format is StringFormat.ADDRESS_US_V1:
            return format_address_us_v1(normalized)
        if string_format is StringFormat.COUNTY:
            return format_county(normalized)
        if string_format is StringFormat.STATE_US:
            return format_state_us(normalized)
        if string_format is StringFormat.ZIP:
            return format_zip(normalized)
        raise ValueError(f"Unsupported string format: {string_format.value}.")

    @staticmethod
    def _timestamp_candidate(normalized: Column, datetime_format: str) -> Column:
        """Parse one timestamp format after any known full-token safety guard."""
        parsed = F.try_to_timestamp(normalized, F.lit(datetime_format))
        shape = BUILTIN_DATETIME_FORMAT_SHAPES.get(datetime_format)
        if shape is None:
            # DataFrame binding requires CORRECTED timeParserPolicy for custom patterns because the
            # compiler cannot safely infer a complete regex for Spark's full pattern language.
            return parsed
        return F.when(normalized.rlike(shape), parsed).otherwise(F.lit(None).cast("timestamp"))

    @classmethod
    def _timestamp_ntz_candidate(cls, normalized: Column, datetime_format: str) -> Column:
        """Parse one local timestamp only after the tolerant timestamp probe succeeds."""
        probed = cls._timestamp_candidate(normalized, datetime_format)
        # to_timestamp_ntz is not itself a try-function. CaseWhen short-circuiting keeps it away
        # from lexically invalid input and invalid calendar values rejected by the probe.
        return F.when(
            probed.isNotNull(),
            F.to_timestamp_ntz(normalized, F.lit(datetime_format)),
        ).otherwise(F.lit(None).cast("timestamp_ntz"))

    def _decode_complex(
        self,
        normalized: Column,
        data_type: SparkDataType,
        options: ParserOptions,
    ) -> _ComplexDecode:
        """Decode one complete complex value and separately flag malformed syntax.

        Spark's permissive JSON reader accepts a valid prefix and can ignore trailing content. A
        small Catalyst-native boundary scan therefore verifies that the source contains exactly one
        complete array or object token; ``from_json`` remains responsible for JSON grammar and type
        validation. Keeping both checks as column expressions avoids Python UDFs and driver-side
        collection.
        """
        if (
            options.parser_type is ParserType.ARRAY
            and options.input_format is ComplexInputFormat.DELIMITED
        ):
            assert options.delimiter is not None
            # ``re.escape`` makes the configured delimiter literal; characters such as ``|`` or
            # ``.`` must not become regular-expression operators.
            return _ComplexDecode(
                F.split(normalized, re.escape(options.delimiter), -1),
                F.lit(False),
            )

        value = F.from_json(
            normalized,
            self._raw_type_ddl(data_type, options),
            _STRICT_JSON_OPTIONS,
        )
        wrapper = F.from_json(
            F.concat(F.lit('{"value":'), normalized, F.lit("}")),
            "map<string,string>",
            _STRICT_JSON_OPTIONS,
        )
        wrapper_keys = F.map_keys(wrapper)
        wrapper_valid = F.coalesce(
            (F.size(wrapper_keys) == F.lit(1))
            & (F.size(F.array_distinct(wrapper_keys)) == F.lit(1))
            & F.array_contains(wrapper_keys, "value"),
            F.lit(False),
        )
        complete_token = self._is_complete_json_container(normalized, options.parser_type)
        invalid = value.isNull() | ~wrapper_valid | ~complete_token
        if options.parser_type is ParserType.STRUCT:
            # A malformed object may decode as a non-null all-null struct in PERMISSIVE mode.
            # Decoding the same object as a map gives us a schema-independent syntax probe and
            # also exposes duplicate source fields, whose winner would otherwise vary by runtime.
            object_probe = F.from_json(
                normalized,
                "map<string,string>",
                _STRICT_JSON_OPTIONS,
            )
            object_keys = F.map_keys(object_probe)
            duplicate_object_fields = F.size(object_keys) != F.size(F.array_distinct(object_keys))
            invalid = (
                invalid | object_probe.isNull() | F.coalesce(duplicate_object_fields, F.lit(False))
            )
        if options.parser_type is ParserType.MAP:
            # Spark can retain duplicate JSON map keys in its internal map representation, but a
            # later map reconstruction may fail. Classify the entire container before that point.
            keys = F.map_keys(value)
            duplicate_keys = F.size(keys) != F.size(F.array_distinct(keys))
            invalid = invalid | F.coalesce(duplicate_keys, F.lit(False))
        return _ComplexDecode(value, normalized.isNotNull() & invalid)

    @staticmethod
    def _is_complete_json_container(normalized: Column, parser_type: ParserType) -> Column:
        """Return whether text is exactly one balanced JSON array or object token.

        This deliberately checks only structural token boundaries. Complete quoted strings are
        removed before scanning so their braces and brackets do not affect nesting. Spark's strict
        ``from_json`` calls validate matching delimiters, escapes, scalar grammar, and the requested
        schema. The accumulator is one integer rather than a struct so recursively nested parser
        plans remain compact enough for Spark code generation and executor transport.
        """
        expected_opener = "[" if parser_type is ParserType.ARRAY else "{"
        # Java's regex engine recursively evaluates an alternation inside a star. Strip JSON escape
        # pairs first so the remaining quoted runs need only one iterative character-class star;
        # this keeps ordinary long string fields from consuming one JVM stack frame per character.
        without_escapes = F.regexp_replace(normalized, r"\\.", "")
        without_strings = F.regexp_replace(without_escapes, r'"[^"]*"', "")
        delimiters = F.regexp_replace(without_strings, r"[^\[\]\{\}]", "")

        def advance(depth: Column, delimiter: Column) -> Column:
            # ``-1`` means the outer container has finished; any later structural delimiter
            # changes it to the permanently invalid ``-2`` state.
            return (
                F.when(depth < F.lit(0), F.lit(-2))
                .when(
                    depth == F.lit(0),
                    F.when(delimiter == F.lit(expected_opener), F.lit(1)).otherwise(F.lit(-2)),
                )
                .when(delimiter.isin("[", "{"), depth + F.lit(1))
                .when(depth == F.lit(1), F.lit(-1))
                .otherwise(depth - F.lit(1))
            )

        final_depth = F.aggregate(F.split(delimiters, ""), F.lit(0), advance)
        return F.coalesce(final_depth == F.lit(-1), F.lit(False))

    def _parse_nested_complex(
        self,
        raw: Column,
        data_type: SparkDataType,
        options: ParserOptions,
        path: Column,
        root_config: ColumnParser,
        *,
        emit_diagnostics: bool,
    ) -> Column:
        """Dispatch a decoded complex value to the matching recursive container parser."""
        parser_type = data_type.parser_type
        if parser_type is ParserType.ARRAY:
            return self._parse_nested_array(
                raw,
                data_type,
                options,
                path,
                root_config,
                emit_diagnostics=emit_diagnostics,
            )
        if parser_type is ParserType.STRUCT:
            return self._parse_nested_struct(
                raw,
                data_type,
                options,
                path,
                root_config,
                emit_diagnostics=emit_diagnostics,
            )
        if parser_type is ParserType.MAP:
            return self._parse_nested_map(
                raw,
                data_type,
                options,
                path,
                root_config,
                emit_diagnostics=emit_diagnostics,
            )
        raise ValueError(f"Expected a complex parser type, found {parser_type.value}.")

    def _parse_nested_value(
        self,
        raw: Column,
        nested: NestedValueParser | StructFieldParser,
        path: Column,
        root_config: ColumnParser,
        *,
        child_error_mode: ChildErrorMode | None = None,
        emit_diagnostics: bool,
    ) -> Column:
        """Parse one child value, apply its owning error policy/default, and retain error paths."""
        if nested.data_type.is_complex:
            normalized = self._normalized_value(raw, nested.parser)
            decoded = self._decode_complex(
                normalized,
                nested.data_type,
                nested.parser,
            )
            state = self._parse_nested_complex(
                decoded.value,
                nested.data_type,
                nested.parser,
                path,
                root_config,
                emit_diagnostics=emit_diagnostics,
            )
            empty_paths = self._empty_error_paths()
            guarded_state = F.when(
                decoded.failed,
                self._parse_state(
                    F.lit(None).cast(nested.expected_data_type),
                    F.lit(False),
                    empty_paths,
                    empty_paths,
                    empty_paths,
                ),
            ).otherwise(state)

            def resolve_complex(child_state: Column) -> Column:
                value = self._resolve_nested_parse_error(
                    child_state.getField("value"),
                    decoded.failed,
                    raw,
                    nested,
                    path,
                    root_config,
                    child_error_mode,
                )
                value, default_applied = self._apply_nested_default(value, nested)
                if not emit_diagnostics:
                    return self._parse_state(
                        value,
                        decoded.failed,
                        empty_paths,
                        empty_paths,
                        empty_paths,
                    )
                # A DROP owner discards a failed child after this function returns. Do not report a
                # default computed only inside that discarded temporary record.
                default_applied = default_applied & ~(
                    decoded.failed & F.lit(child_error_mode is ChildErrorMode.DROP)
                )
                # Error paths are diagnostic history, not just unresolved failures. Preserve
                # handled child paths even if the parent later defaults or drops the value.
                return self._parse_state(
                    value,
                    decoded.failed,
                    F.concat(
                        child_state.getField("error_paths"),
                        self._paths_when(decoded.failed, path),
                    ),
                    F.concat(
                        child_state.getField("default_on_null_paths"),
                        self._paths_when(default_applied, path),
                    ),
                    child_state.getField("zero_invalidated_paths"),
                )

            return self._bind_once(guarded_state, resolve_complex)

        normalized = self._normalized_value(raw, nested.parser)
        candidate = self._parse_scalar_candidate(normalized, nested.data_type, nested.parser)
        failed = normalized.isNotNull() & candidate.isNull()
        value = self._resolve_nested_parse_error(
            candidate,
            failed,
            raw,
            nested,
            path,
            root_config,
            child_error_mode,
        )
        zero_invalidated = F.lit(False)
        if nested.parser.parser_type in NUMERIC_PARSER_TYPES and not nested.parser.zero_is_valid:
            # Child zero invalidation follows the same order as top-level parsing: after immediate
            # parse-error resolution and before the final null default.
            zero_invalidated = F.coalesce(
                value == F.lit(0).cast(nested.expected_data_type),
                F.lit(False),
            )
            value = F.when(
                zero_invalidated,
                F.lit(None).cast(nested.expected_data_type),
            ).otherwise(value)
        value, default_applied = self._apply_nested_default(value, nested)
        if not emit_diagnostics:
            empty_paths = self._empty_error_paths()
            return self._parse_state(
                value,
                failed,
                empty_paths,
                empty_paths,
                empty_paths,
            )
        default_applied = default_applied & ~(
            failed & F.lit(child_error_mode is ChildErrorMode.DROP)
        )
        return self._parse_state(
            value,
            failed,
            self._paths_when(failed, path),
            self._paths_when(default_applied, path),
            self._paths_when(zero_invalidated, path),
        )

    def _parse_nested_array(
        self,
        raw: Column,
        data_type: SparkDataType,
        options: ParserOptions,
        path: Column,
        root_config: ColumnParser,
        *,
        emit_diagnostics: bool,
    ) -> Column:
        """Parse array elements with zero-based paths and apply drop/null/distinct behavior."""
        element_parser = options.element_parser
        assert element_parser is not None

        def parse_element(element: Column, index: Column) -> Column:
            """Return a temporary record so value and diagnostics travel through one transform."""
            element_path = F.concat(path, F.lit("["), index.cast("string"), F.lit("]"))
            parsed = self._parse_nested_value(
                element,
                element_parser,
                element_path,
                root_config,
                child_error_mode=options.on_element_error,
                emit_diagnostics=emit_diagnostics,
            )
            return parsed

        records = F.transform(raw, parse_element)

        def build_array(bound_records: Column) -> Column:
            # Keep diagnostics from the original records. Filtering only the value projection
            # ensures a dropped bad element remains visible in ``nested_error_paths``.
            retained = (
                F.filter(bound_records, lambda record: ~record.getField("failed"))
                if options.on_element_error is ChildErrorMode.DROP
                else bound_records
            )
            if options.drop_null_elements:
                # Removes original nulls and values resolved to null by child policy/defaults.
                retained = F.filter(
                    retained,
                    lambda record: record.getField("value").isNotNull(),
                )
            values = F.transform(retained, lambda record: record.getField("value"))
            if options.distinct:
                values = F.array_distinct(values)
            value = F.when(
                bound_records.isNull(),
                F.lit(None).cast(data_type.canonical),
            ).otherwise(values)
            empty_paths = self._empty_error_paths()
            if not emit_diagnostics:
                return self._parse_state(
                    value,
                    F.lit(False),
                    empty_paths,
                    empty_paths,
                    empty_paths,
                )
            error_paths = F.flatten(
                F.transform(bound_records, lambda record: record.getField("error_paths"))
            )
            default_paths = F.flatten(
                F.transform(
                    bound_records,
                    lambda record: record.getField("default_on_null_paths"),
                )
            )
            zero_paths = F.flatten(
                F.transform(
                    bound_records,
                    lambda record: record.getField("zero_invalidated_paths"),
                )
            )
            return self._parse_state(
                value,
                F.lit(False),
                F.when(bound_records.isNull(), empty_paths).otherwise(error_paths),
                F.when(bound_records.isNull(), empty_paths).otherwise(default_paths),
                F.when(bound_records.isNull(), empty_paths).otherwise(zero_paths),
            )

        return self._bind_once(records, build_array)

    def _parse_nested_struct(
        self,
        raw: Column,
        data_type: SparkDataType,
        options: ParserOptions,
        path: Column,
        root_config: ColumnParser,
        *,
        emit_diagnostics: bool,
    ) -> Column:
        """Parse every configured struct field and emit fields in compiled schema order."""
        parsed_states: list[Column] = []
        for index, field in enumerate(options.field_parsers):
            field_path = F.concat(
                path,
                F.lit(_static_json_path_segment(field.target_field_name)),
            )
            parsed = self._parse_nested_value(
                raw.getField(field.source_field_name),
                field,
                field_path,
                root_config,
                emit_diagnostics=emit_diagnostics,
            )
            parsed_states.append(parsed.alias(f"field_{index}"))

        # Field values may have different data types, so bind their carriers through one struct
        # rather than an array. Each complete child subtree occurs exactly once in ``bound_fields``.
        bound_fields = F.struct(
            raw.isNull().alias("raw_is_null"),
            *parsed_states,
        )

        def build_struct(fields: Column) -> Column:
            states = [fields.getField(f"field_{index}") for index in range(len(parsed_states))]
            value = F.struct(
                *(
                    state.getField("value").alias(field.target_field_name)
                    for state, field in zip(states, options.field_parsers, strict=True)
                )
            )
            # Spark's struct() itself is non-null even when every child is null. Restore a truly
            # null struct when the decoded raw container was null.
            value = F.when(
                fields.getField("raw_is_null"),
                F.lit(None).cast(data_type.canonical),
            ).otherwise(value)
            empty_paths = self._empty_error_paths()
            if not emit_diagnostics:
                return self._parse_state(
                    value,
                    F.lit(False),
                    empty_paths,
                    empty_paths,
                    empty_paths,
                )
            errors = F.concat(*(state.getField("error_paths") for state in states))
            defaults = F.concat(
                *(state.getField("default_on_null_paths") for state in states)
            )
            zeros = F.concat(
                *(state.getField("zero_invalidated_paths") for state in states)
            )
            return self._parse_state(
                value,
                F.lit(False),
                F.when(fields.getField("raw_is_null"), empty_paths).otherwise(errors),
                F.when(fields.getField("raw_is_null"), empty_paths).otherwise(defaults),
                F.when(fields.getField("raw_is_null"), empty_paths).otherwise(zeros),
            )

        return self._bind_once(bound_fields, build_struct)

    def _parse_nested_map(
        self,
        raw: Column,
        data_type: SparkDataType,
        options: ParserOptions,
        path: Column,
        root_config: ColumnParser,
        *,
        emit_diagnostics: bool,
    ) -> Column:
        """Parse map values while preserving keys and recording key-specific error paths."""
        value_parser = options.value_parser
        assert value_parser is not None

        def parse_entry(entry: Column) -> Column:
            """Attach the key, parsed value, failure flag, and paths to one temporary entry."""
            key = entry.getField("key")
            # Let Spark's JSON encoder escape backslashes, quotes, control characters, and Unicode
            # consistently. Remove the surrounding one-element array syntax, then add the extra
            # apostrophe escape required by this package's single-quoted bracket convention.
            escaped_key = F.regexp_replace(
                F.regexp_replace(
                    F.to_json(F.array(key)),
                    r'^\["|"\]$',
                    "",
                ),
                "'",
                r"\\'",
            )
            value_path = F.concat(path, F.lit("['"), escaped_key, F.lit("']"))
            parsed = self._parse_nested_value(
                entry.getField("value"),
                value_parser,
                value_path,
                root_config,
                child_error_mode=options.on_value_error,
                emit_diagnostics=emit_diagnostics,
            )
            return F.struct(
                key.alias("key"),
                parsed.alias("parsed"),
            )

        # MapType does not promise iteration order. Sort the permissively decoded raw string pairs
        # before parsing so target construction and diagnostic path arrays are reproducible.
        records = F.transform(F.array_sort(F.map_entries(raw)), parse_entry)

        def build_map(bound_records: Column) -> Column:
            retained = (
                F.filter(
                    bound_records,
                    lambda record: ~record.getField("parsed").getField("failed"),
                )
                if options.on_value_error is ChildErrorMode.DROP
                else bound_records
            )
            if options.drop_null_values:
                retained = F.filter(
                    retained,
                    lambda record: record.getField("parsed").getField("value").isNotNull(),
                )
            # Remove helper fields before map_from_entries; only key/value pairs belong in target.
            entries = F.transform(
                retained,
                lambda record: F.struct(
                    record.getField("key").alias("key"),
                    record.getField("parsed").getField("value").alias("value"),
                ),
            )
            value = F.map_from_entries(entries)
            value = F.when(
                bound_records.isNull(),
                F.lit(None).cast(data_type.canonical),
            ).otherwise(value)
            empty_paths = self._empty_error_paths()
            if not emit_diagnostics:
                return self._parse_state(
                    value,
                    F.lit(False),
                    empty_paths,
                    empty_paths,
                    empty_paths,
                )
            error_paths = F.flatten(
                F.transform(
                    bound_records,
                    lambda record: record.getField("parsed").getField("error_paths"),
                )
            )
            default_paths = F.flatten(
                F.transform(
                    bound_records,
                    lambda record: record.getField("parsed").getField(
                        "default_on_null_paths"
                    ),
                )
            )
            zero_paths = F.flatten(
                F.transform(
                    bound_records,
                    lambda record: record.getField("parsed").getField(
                        "zero_invalidated_paths"
                    ),
                )
            )
            return self._parse_state(
                value,
                F.lit(False),
                F.when(bound_records.isNull(), empty_paths).otherwise(error_paths),
                F.when(bound_records.isNull(), empty_paths).otherwise(default_paths),
                F.when(bound_records.isNull(), empty_paths).otherwise(zero_paths),
            )

        return self._bind_once(records, build_map)

    def _resolve_nested_parse_error(
        self,
        candidate: Column,
        failed: Column,
        source: Column,
        nested: NestedValueParser | StructFieldParser,
        path: Column,
        root_config: ColumnParser,
        child_error_mode: ChildErrorMode | None,
    ) -> Column:
        """Apply the immediate owner of a nested parse failure.

        Array/map child modes override the child's direct parse mode. Struct fields have no parent
        child mode, so their own fail/null/default/string-preserve policy applies.
        """
        message = F.concat(
            F.lit(
                "Spark Parser could not parse nested value for source "
                f"{root_config.source_column_name!r} into target column "
                f"{root_config.target_column_name!r} at "
            ),
            path,
            F.lit(f" as {nested.expected_data_type}: "),
            source.cast("string"),
        )
        if child_error_mode is ChildErrorMode.FAIL:
            return F.when(
                failed,
                F.raise_error(message).cast(nested.expected_data_type),
            ).otherwise(candidate)
        if child_error_mode in {ChildErrorMode.NULL, ChildErrorMode.DROP}:
            # DROP is performed by the parent container after this method returns. At this level both
            # NULL and DROP retain a typed-null candidate and the failure flag.
            return candidate
        if child_error_mode is ChildErrorMode.PRESERVE:
            # Compilation permits this branch only for a string child. Use the original decoded
            # token—not the normalized value—so "preserve" means an exact raw-value fallback.
            return F.when(failed, source.cast("string")).otherwise(candidate)
        if nested.parser.on_parse_error is ParseErrorMode.NULL:
            return candidate
        if nested.parser.on_parse_error is ParseErrorMode.DEFAULT:
            return F.when(
                failed,
                self._typed_value_literal(
                    nested.parser.default_on_error,
                    nested.data_type,
                    nested.parser,
                ),
            ).otherwise(candidate)
        if nested.parser.on_parse_error is ParseErrorMode.PRESERVE:
            # Struct fields own their direct parse policy. As above, the compiler guarantees this
            # field's target type is string before the raw token can enter the typed struct.
            return F.when(failed, source.cast("string")).otherwise(candidate)
        return F.when(
            failed,
            F.raise_error(message).cast(nested.expected_data_type),
        ).otherwise(candidate)

    def _apply_nested_default(
        self,
        value: Column,
        nested: NestedValueParser | StructFieldParser,
    ) -> tuple[Column, Column]:
        """Apply a nested final-null default and return both value and null-safe action flag."""
        if nested.parser.is_nullable:
            return value, F.lit(False)
        applied = F.coalesce(value.isNull(), F.lit(False))
        resolved = F.when(
            applied,
            self._typed_value_literal(
                nested.parser.default_on_null,
                nested.data_type,
                nested.parser,
            ),
        ).otherwise(value)
        return resolved, applied

    def _raw_type_ddl(self, data_type: SparkDataType, options: ParserOptions) -> str:
        """Return the permissive intermediate schema used to decode raw child text.

        Leaves decode as strings so the configured leaf parser—not Spark's JSON coercion—owns
        normalization, formatting, and error policy. Complex children are recursively decoded from
        those captured strings.
        """
        if data_type.parser_type is ParserType.ARRAY:
            return "array<string>"
        if data_type.parser_type is ParserType.MAP:
            return "map<string,string>"
        fields = ",".join(
            f"{_quoted_identifier(field.source_field_name)}:string"
            for field in options.field_parsers
        )
        return f"struct<{fields}>"

    @staticmethod
    def _empty_error_paths() -> Column:
        """Return a typed empty path array so concat/flatten schemas stay consistent."""
        return F.array().cast("array<string>")

    @staticmethod
    def _parse_state(
        value: Column,
        failed: Column,
        error_paths: Column,
        default_on_null_paths: Column,
        zero_invalidated_paths: Column,
    ) -> Column:
        """Pack one parse node into the common carrier schema."""
        return F.struct(
            value.alias("value"),
            failed.alias("failed"),
            error_paths.alias("error_paths"),
            default_on_null_paths.alias("default_on_null_paths"),
            zero_invalidated_paths.alias("zero_invalidated_paths"),
        )

    @staticmethod
    def _bind_once(expression: Column, build: Callable[[Column], Column]) -> Column:
        """Bind an expression to one higher-order variable before consuming it repeatedly.

        Catalyst Columns are expression trees rather than references. Merely assigning a Column to
        a Python variable does not share it: every use embeds another copy. A singleton transform
        creates an actual Catalyst lambda variable, and ``element_at`` unwraps the one result.
        """
        return F.element_at(F.transform(F.array(expression), build), F.lit(1))

    @staticmethod
    def _paths_when(condition: Column, path: Column) -> Column:
        """Return one typed path when a null-safe audit condition occurred."""
        return F.filter(
            F.array(F.when(F.coalesce(condition, F.lit(False)), path)),
            lambda item: item.isNotNull(),
        )

    def _resolve_parse_error(
        self,
        candidate: Column,
        parse_failed: Column,
        source: Column,
        column_config: ColumnParser,
    ) -> Column:
        """Resolve a top-level parse failure according to its compiled error policy."""
        options = column_config.parser
        if options.on_parse_error is ParseErrorMode.NULL:
            return candidate
        if options.on_parse_error is ParseErrorMode.DEFAULT:
            return F.when(
                parse_failed,
                self._typed_value_literal(
                    options.default_on_error,
                    column_config.data_type,
                    options,
                ),
            ).otherwise(candidate)
        if options.on_parse_error is ParseErrorMode.PRESERVE:
            # Present configured source columns are schema-validated as strings, and compilation
            # restricts this mode to string parsers. Returning source therefore preserves the exact
            # bronze token (including its original whitespace) without weakening the target schema.
            return F.when(parse_failed, source.cast("string")).otherwise(candidate)
        message = F.concat(
            F.lit(
                f"Spark Parser could not parse source {column_config.source_column_name!r} "
                f"into target column {column_config.target_column_name!r} as "
                f"{column_config.expected_data_type}: "
            ),
            source,
        )
        # raise_error remains a lazy Spark expression. It fires only if an action materializes this
        # target value; a count whose projection prunes the value may not evaluate it.
        return F.when(
            parse_failed,
            F.raise_error(message).cast(column_config.expected_data_type),
        ).otherwise(candidate)

    def _typed_value_literal(
        self,
        value: Any,
        data_type: SparkDataType,
        options: ParserOptions,
    ) -> Column:
        """Recursively convert a compiler-validated Python default into native Spark literals."""
        if value is None:
            if not options.is_nullable and options.default_on_null is not None:
                return self._typed_value_literal(
                    options.default_on_null,
                    data_type,
                    options,
                )
            return F.lit(None).cast(data_type.canonical)
        if (
            data_type.parser_type in NUMERIC_PARSER_TYPES
            and not options.zero_is_valid
            and value == 0
        ):
            # A scalar embedded in a parent complex default still obeys its own final-value
            # contract. Route an invalid zero through that child's null/default behavior.
            return self._typed_value_literal(None, data_type, options)
        if data_type.parser_type is ParserType.DATE:
            assert isinstance(value, date) and not isinstance(value, datetime)
            # Sending Python date/datetime objects through Py4J uses host-timezone/platform
            # conversion and can fail at boundary years. Spark-owned ISO parsing is deterministic.
            return F.lit(value.isoformat()).cast(data_type.canonical)
        if data_type.parser_type in {ParserType.TIMESTAMP, ParserType.TIMESTAMP_NTZ}:
            assert isinstance(value, datetime)
            # Naive text is interpreted in spark.sql.session.timeZone, while an authored offset is
            # preserved for TimestampType. TimestampNTZ compilation rejects offsets.
            return F.lit(value.isoformat(sep=" ")).cast(data_type.canonical)
        if data_type.parser_type is ParserType.BINARY:
            return F.try_to_binary(F.lit(value), F.lit(_BINARY_FORMATS[options.binary_encoding]))
        if data_type.parser_type is ParserType.ARRAY:
            element_type = data_type.element_type
            element_parser = options.element_parser
            assert element_type is not None and element_parser is not None
            items = [
                self._typed_value_literal(
                    item,
                    element_type,
                    element_parser.parser,
                )
                for item in value
            ]
            # Empty arrays need an explicit cast; otherwise Spark infers array<void>.
            return (F.array(*items) if items else F.array()).cast(data_type.canonical)
        if data_type.parser_type is ParserType.STRUCT:
            # Follow compiled field order and aliases so positional Spark struct semantics match the
            # canonical expected datatype.
            fields = [
                self._typed_value_literal(
                    value[field.target_field_name],
                    field.data_type,
                    field.parser,
                ).alias(field.target_field_name)
                for field in options.field_parsers
            ]
            return F.struct(*fields).cast(data_type.canonical)
        if data_type.parser_type is ParserType.MAP:
            value_type = data_type.value_type
            value_parser = options.value_parser
            assert value_type is not None and value_parser is not None
            pairs: list[Column] = []
            for key in sorted(value):
                item = value[key]
                pairs.extend(
                    (
                        F.lit(key),
                        self._typed_value_literal(
                            item,
                            value_type,
                            value_parser.parser,
                        ),
                    )
                )
            # Empty maps also need an explicit cast because Spark cannot infer key/value types.
            return (F.create_map(*pairs) if pairs else F.create_map()).cast(data_type.canonical)
        return F.lit(value).cast(data_type.canonical)

    @staticmethod
    def _null_marker_match(value: Column, options: ParserOptions) -> Column:
        """Return a null-safe match expression for the effective configured null vocabulary."""
        if not options.replace_null_markers or not options.null_markers:
            return F.lit(False)
        if options.null_marker_case_sensitive:
            matched = value.isin(*options.null_markers)
        else:
            # Normalize both sides with Spark's own Unicode tables. Python and the JVM can ship
            # different Unicode versions, so pre-lowering configured literals in Python can make
            # an otherwise supported Python/PySpark pairing disagree on newer code points.
            matched = F.lower(value).isin(
                *(F.lower(F.lit(marker)) for marker in options.null_markers)
            )
        return F.coalesce(matched, F.lit(False))

    def _audit_struct(
        self,
        column_config: ColumnParser,
        *,
        source: Column,
        parsed: Column,
        empty_to_null: Column,
        marker_replaced: Column,
        json_null: Column,
        parse_failed: Column,
        zero_invalidated: Column,
        default_on_null_applied: Column,
        source_missing: Column,
        normalized: Column,
        candidate: Column,
        nested_error_paths: Column,
        nested_default_on_null_paths: Column,
        nested_zero_invalidated_paths: Column,
    ) -> Column:
        """Assemble one deterministic, human-explainable parser audit struct expression."""
        options = column_config.parser
        parse_to_null = parse_failed & F.lit(options.on_parse_error is ParseErrorMode.NULL)
        parse_default = parse_failed & F.lit(options.on_parse_error is ParseErrorMode.DEFAULT)
        parse_preserved = parse_failed & F.lit(options.on_parse_error is ParseErrorMode.PRESERVE)
        # A top-level malformed container is already represented by parse_failed. Only separately
        # label nested resolution when one or more child paths failed inside a valid container.
        nested_failed = (F.size(nested_error_paths) > 0) & ~parse_failed
        nested_defaulted = F.size(nested_default_on_null_paths) > 0
        nested_zero_invalidated = F.size(nested_zero_invalidated_paths) > 0
        if options.string_format is StringFormat.ZIP:
            zip_source_parts = F.split(normalized, UNICODE_LIST_DELIMITER_PATTERN, -1)
            zip_candidate_parts = F.split(candidate, UNICODE_LIST_DELIMITER_PATTERN, -1)
            zip_padding_flags = F.zip_with(
                zip_source_parts,
                zip_candidate_parts,
                lambda source_part, candidate_part: (
                    F.length(F.regexp_replace(source_part, r"\D", ""))
                    < F.length(F.regexp_replace(candidate_part, r"\D", ""))
                ),
            )
            zip_plus4_flags = F.zip_with(
                zip_source_parts,
                zip_candidate_parts,
                lambda source_part, candidate_part: (
                    candidate_part.contains("-")
                    & (
                        candidate_part
                        != F.regexp_replace(source_part, UNICODE_WHITESPACE_PATTERN, "")
                    )
                ),
            )
            zip_plus4 = F.coalesce(
                F.exists(zip_plus4_flags, lambda changed: changed),
                F.lit(False),
            )
            zip_padded = F.coalesce(
                F.exists(zip_padding_flags, lambda changed: changed),
                F.lit(False),
            )
        else:
            zip_plus4 = F.lit(False)
            zip_padded = F.lit(False)
        # Action order is intentional. Null placeholders are filtered in Spark so each
        # row receives only the transformations that actually occurred.
        actions = F.filter(
            F.array(
                F.when(source_missing, F.lit("source_column_missing")),
                F.when(empty_to_null, F.lit("empty_string_to_null")),
                F.when(marker_replaced, F.lit("null_marker_replaced")),
                F.when(json_null, F.lit("json_null_to_null")),
                F.when(parse_to_null, F.lit("parse_error_to_null")),
                F.when(parse_default, F.lit("parse_error_default_applied")),
                F.when(parse_preserved, F.lit("parse_error_preserved")),
                F.when(nested_failed, F.lit("nested_parse_errors_resolved")),
                F.when(nested_zero_invalidated, F.lit("nested_zero_invalidated")),
                F.when(
                    nested_defaulted,
                    F.lit("nested_default_on_null_applied"),
                ),
                F.when(zero_invalidated, F.lit("zero_invalidated")),
                F.when(default_on_null_applied, F.lit("default_on_null_applied")),
                F.when(zip_padded, F.lit("zip_padded")),
                F.when(zip_plus4, F.lit("zip_plus4_formatted")),
            ),
            lambda item: item.isNotNull(),
        )
        changed = F.coalesce(
            (
                source_missing
                | empty_to_null
                | marker_replaced
                | json_null
                | parse_to_null
                | parse_default
                | parse_preserved
                | nested_failed
                | nested_zero_invalidated
                | nested_defaulted
                | zero_invalidated
                | default_on_null_applied
                | zip_padded
                | zip_plus4
            ),
            F.lit(False),
        )
        error = (
            F.when(source_missing, F.lit("Source column is missing."))
            .when(
                parse_failed,
                F.lit(f"Value could not be parsed as {column_config.expected_data_type}."),
            )
            .when(
                nested_failed,
                F.lit("One or more nested values could not be parsed; see nested_error_paths."),
            )
        )
        # Audit storage is always printable: complex values become canonical JSON and binary values
        # become base64 regardless of the bronze input encoding.
        if options.parser_type in COMPLEX_PARSER_TYPES:
            parsed_value = F.to_json(parsed, _AUDIT_JSON_OPTIONS)
        elif options.parser_type is ParserType.BINARY:
            parsed_value = F.base64(parsed)
        else:
            parsed_value = parsed.cast("string")
        return F.struct(
            F.lit(column_config.source_column_name).alias("source_column_name"),
            F.lit(column_config.target_column_name).alias("target_column_name"),
            F.lit(options.parser_type.value).alias("parser_type"),
            F.lit(column_config.expected_data_type).alias("expected_data_type"),
            source.alias("original_value"),
            parsed_value.alias("parsed_value"),
            changed.alias("changed"),
            (~source_missing).alias("effective"),
            actions.alias("actions_applied"),
            self._options_map(options).alias("options"),
            error.alias("error"),
            nested_error_paths.alias("nested_error_paths"),
            nested_default_on_null_paths.alias("nested_default_on_null_paths"),
            nested_zero_invalidated_paths.alias("nested_zero_invalidated_paths"),
        )

    def _options_map(self, options: ParserOptions) -> Column:
        """Serialize effective parser options into a Spark ``map<string,string>`` expression."""
        payload = self._serializer.parser_mapping(
            options,
            include_audit=True,
            include_error_mode=True,
        )
        payload.setdefault("default_on_null", None)
        payload.setdefault("default_on_error", None)
        pairs: list[Column] = []
        # Stringifying values gives every audit row one consistent map type even though option values in
        # the compiled model include Booleans, lists, mappings, dates, and nulls.
        for key, value in payload.items():
            pairs.extend((F.lit(key), F.lit(self._option_text(value))))
        return F.create_map(*pairs)

    @staticmethod
    def _option_text(value: Any) -> str:
        """Render one effective option deterministically for row-level audit storage."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (list, tuple, Mapping)):
            return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"))
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)
