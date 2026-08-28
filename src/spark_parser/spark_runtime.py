"""Build native Spark expressions that turn bronze strings into typed silver values.

This module constructs a lazy logical plan; it does not collect rows or use Python/pandas UDFs.
Keeping transformations inside Spark SQL lets Catalyst optimize the plan and allows the same code
to scale from local tests to Databricks jobs. Most helpers return :class:`Column` expressions rather
than immediate Python values, which is the central concept a maintainer should keep in mind.
"""

from __future__ import annotations

import json
import re
import warnings as python_warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from spark_parser.address_formats import (
    format_address_us_v1,
    format_county,
    format_state_us,
    format_zip,
)
from spark_parser.data_types import SparkDataType
from spark_parser.dataframe_parsing import DataFrameParsing
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
from spark_parser.models import (
    ColumnParser,
    NestedValueParser,
    ParserConfig,
    ParserOptions,
    StructFieldParser,
)
from spark_parser.serializer import ParserConfigSerializer
from spark_parser.version import __version__

# Public audit schema. Define it explicitly so an empty audit array has exactly the same type as a
# populated one. Field order is contractual for consumers that persist or flatten parser results.
PARSE_RESULT_STRUCT = T.StructType(
    [
        T.StructField("source_column_name", T.StringType(), True),
        T.StructField("silver_column_name", T.StringType(), True),
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
    ]
)
PARSE_RESULT_ARRAY = T.ArrayType(PARSE_RESULT_STRUCT, containsNull=True)

# Spark's ordinary ``trim`` behavior does not cover every whitespace character handled here. The
# patterns include non-breaking space because it commonly appears in copied source data.
_WHITESPACE_PATTERN = r"[\s\u00A0]+"
_EDGE_WHITESPACE_PATTERN = r"^[\s\u00A0]+|[\s\u00A0]+$"

# Numeric child values are decoded through strict JSON to obtain ANSI-safe casts. This pattern
# rejects partial tokens and non-JSON spellings before Spark attempts the typed conversion.
_JSON_NUMBER_PATTERN = r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$"

# ``PERMISSIVE`` gives a null result instead of throwing for malformed containers. The runtime can
# then apply the user's explicit fail/null/default policy in a consistent way.
_STRICT_JSON_OPTIONS = {
    "mode": "PERMISSIVE",
    "allowComments": "false",
    "allowSingleQuotes": "false",
    "allowUnquotedFieldNames": "false",
    "allowNumericLeadingZeros": "false",
}
# Audit values must be stable, printable strings. Explicit formats prevent session settings from
# changing the JSON representation of dates and timestamps.
_AUDIT_JSON_OPTIONS = {
    "ignoreNullFields": "false",
    "dateFormat": "yyyy-MM-dd",
    "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSXXX",
    "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSS",
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
    nested_errors_name: str
    post_parse_name: str
    post_zero_name: str
    final_name: str


@dataclass(frozen=True)
class _NestedParse:
    """Value, immediate failure flag, and accumulated JSON paths for a nested parser node."""

    value: Column
    failed: Column
    error_paths: Column


@dataclass(frozen=True)
class _ComplexDecode:
    """Result of decoding one raw JSON/delimited container before recursive child parsing."""

    value: Column
    failed: Column


def _column(name: str) -> Column:
    """Resolve one top-level column literally, even when its name contains dots/backticks."""
    return F.col(f"`{name.replace('`', '``')}`")


def _quoted_identifier(name: str) -> str:
    """Quote one top-level name for safe interpolation into a Spark SQL expression."""
    return f"`{name.replace('`', '``')}`"


class SparkDataFrameParser:
    """Build lazy silver and audit projections using only native Spark expressions.

    The compiler guarantees option consistency. This class binds that contract to a concrete input
    schema, creates collision-resistant internal columns, and exposes the final projections through
    :class:`DataFrameParsing`.
    """

    def parse_dataframe(
        self,
        df: DataFrame,
        config: ParserConfig,
        *,
        key_columns: Sequence[str] | None = None,
        column_prefix: str = "spark_parser",
    ) -> DataFrameParsing:
        """Build lazy parsed and row-level audit projections.

        Existing configured source fields must occur exactly once and have
        Spark ``string`` type. A missing source emits a recoverable warning and
        produces a typed null (or ``default_on_null`` for a non-nullable target).
        Invalid non-null values raise during the first Spark action unless their
        parser explicitly selects ``on_parse_error: null`` or ``default``.
        """
        if not isinstance(config, ParserConfig):
            raise TypeError("config must be a ParserConfig.")
        if not isinstance(column_prefix, str) or not column_prefix:
            raise ValueError("column_prefix must be a non-empty string.")
        normalized_keys, schema_warnings = self._validate_schema(
            df,
            config,
            key_columns,
            column_prefix,
        )

        # Generate every internal name up front. UUID-backed names prevent conflicts with unusual
        # bronze schemas and let later stages refer to expressions without nesting the entire plan.
        plans = tuple(
            self._runtime_plan(
                column_config,
                source_missing=column_config.source_column_name not in df.columns,
            )
            for column_config in config.columns
        )
        # Stage 1: apply common whitespace/null normalization to every configured source in one
        # projection. ``withColumns`` avoids a deep chain of one-column projections on wide loads.
        working = df.withColumns(
            {
                plan.normalized_name: self._normalized_value(
                    self._source_value(plan),
                    plan.config.parser,
                )
                for plan in plans
            }
        )
        # Stage 2: build type-specific candidates and nested error paths. A candidate may be null
        # either because the normalized input was null or because conversion failed; the next stage
        # distinguishes those cases explicitly.
        candidate_columns: dict[str, Column] = {}
        for plan in plans:
            candidate, nested_errors = self._parse_candidate(
                _column(plan.normalized_name),
                plan.normalized_name,
                plan.config,
            )
            candidate_columns[plan.candidate_name] = candidate
            candidate_columns[plan.nested_errors_name] = nested_errors
        working = working.withColumns(candidate_columns)
        # Stage 3: apply fail/null/default behavior only to non-null input that failed conversion.
        working = working.withColumns(
            {
                plan.post_parse_name: self._resolve_parse_error(
                    _column(plan.candidate_name),
                    self._parse_failed(plan),
                    self._source_value(plan),
                    plan.config,
                )
                for plan in plans
            }
        )
        # Stage 4: optionally invalidate numeric zero after parse-error resolution.
        working = working.withColumns(
            {plan.post_zero_name: self._post_zero_value(plan) for plan in plans}
        )
        # Stage 5: enforce final nullability and apply default_on_null where required.
        working = working.withColumns({plan.final_name: self._final_value(plan) for plan in plans})
        parsed_columns = [(plan.config.silver_column_name, plan.final_name) for plan in plans]
        audit_structs = [self._audit_for_plan(plan) for plan in plans if plan.config.parser.audit]

        parse_results_name = f"{column_prefix}_parse_results"
        config_name = f"{column_prefix}_config"
        engine_version_name = f"{column_prefix}_engine_version"
        # Casting the empty-array branch is essential: otherwise Spark infers array<void>, and the
        # audit table schema would depend on whether any columns happened to enable auditing.
        parse_results = (F.array(*audit_structs) if audit_structs else F.array()).cast(
            PARSE_RESULT_ARRAY
        )
        serializer = ParserConfigSerializer()
        working = working.withColumns(
            {
                parse_results_name: parse_results,
                config_name: F.struct(
                    F.lit(config.parser_config_id).alias("id"),
                    F.lit(config.version).alias("version"),
                    F.lit(serializer.content_hash(config)).alias("content_hash"),
                ),
                engine_version_name: F.lit(__version__),
            }
        )
        # Drop bronze and temporary staging columns from the shared plan. Row keys are retained only
        # for results_df; parsed_df later selects the silver internal names and restores aliases.
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

    def _validate_schema(
        self,
        df: DataFrame,
        config: ParserConfig,
        key_columns: Sequence[str] | None,
        column_prefix: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Validate the bronze schema and return normalized result keys plus recoverable warnings.

        Schema inspection is metadata-only and does not trigger a Spark action. Missing configured
        sources are recoverable because the runtime can substitute a typed null/default; ambiguous
        or non-string sources are not recoverable and fail immediately.
        """
        duplicates = sorted({name for name in df.columns if df.columns.count(name) > 1})
        configured_names = {column.source_column_name for column in config.columns}
        ambiguous = sorted(configured_names & set(duplicates))
        if ambiguous:
            raise SchemaValidationError(f"Configured input columns are ambiguous: {ambiguous}.")
        missing = sorted(configured_names - set(df.columns))
        schema_warnings: list[str] = []
        if missing:
            message = (
                "Configured source columns are missing and will produce typed null/default "
                f"values: {missing}."
            )
            schema_warnings.append(message)
            python_warnings.warn(message, SchemaWarning, stacklevel=3)
        field_types = {field.name: field.dataType for field in df.schema.fields}
        non_string = {
            name: field_types[name].simpleString()
            for name in sorted(configured_names - set(missing))
            if not isinstance(field_types[name], T.StringType)
        }
        if non_string:
            raise SchemaValidationError(
                f"Configured bronze columns must have Spark string type; found {non_string}."
            )

        output_names = {
            f"{column_prefix}_parse_results",
            f"{column_prefix}_config",
            f"{column_prefix}_engine_version",
        }
        conflicts = sorted(output_names & set(df.columns))
        if conflicts:
            raise SchemaValidationError(
                f"Input contains reserved parser output columns: {conflicts}."
            )
        if key_columns is None:
            # All bronze columns provide a usable default identity for exploratory work. Production
            # callers should normally pass stable business/load keys explicitly.
            normalized_keys = tuple(df.columns)
        else:
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
        duplicate_keys = sorted(
            {name for name in normalized_keys if normalized_keys.count(name) > 1}
        )
        if duplicate_keys:
            raise SchemaValidationError(f"key_columns contains duplicates: {duplicate_keys}.")
        missing_keys = sorted(set(normalized_keys) - set(df.columns))
        if missing_keys:
            raise SchemaValidationError(f"key_columns are missing: {missing_keys}.")
        ambiguous_keys = sorted(set(normalized_keys) & set(duplicates))
        if ambiguous_keys:
            raise SchemaValidationError(f"key_columns are ambiguous: {ambiguous_keys}.")
        result_key_conflicts = sorted(set(normalized_keys) & output_names)
        if result_key_conflicts:
            raise SchemaValidationError(
                f"key_columns conflict with parser result columns: {result_key_conflicts}."
            )
        return normalized_keys, tuple(schema_warnings)

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
            nested_errors_name=f"__spark_parser_nested_errors_{token}",
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
        whitespace_normalized = (
            F.regexp_replace(source, _WHITESPACE_PATTERN, " ")
            if options.collapse_whitespace and options.parser_type not in COMPLEX_PARSER_TYPES
            else source
        )
        if options.trim_whitespace:
            whitespace_normalized = F.regexp_replace(
                whitespace_normalized,
                _EDGE_WHITESPACE_PATTERN,
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
    def _parse_failed(plan: _ColumnRuntimePlan) -> Column:
        """Identify non-null normalized input whose typed candidate is null."""
        return _column(plan.normalized_name).isNotNull() & _column(plan.candidate_name).isNull()

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
                self._typed_literal(
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
            candidate=_column(plan.candidate_name),
            nested_error_paths=_column(plan.nested_errors_name),
        )

    def _parse_candidate(
        self,
        normalized: Column,
        normalized_name: str,
        column_config: ColumnParser,
    ) -> tuple[Column, Column]:
        """Build the initial typed candidate and nested-error array for one top-level column."""
        if column_config.parser.parser_type in COMPLEX_PARSER_TYPES:
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
            )
            # A malformed outer container is reported at ``$``; recursively handled child failures
            # already carry more specific paths.
            container_errors = self._error_paths(decoded.failed, F.lit("$"))
            candidate = F.when(
                decoded.failed,
                F.lit(None).cast(column_config.expected_data_type),
            ).otherwise(nested.value)
            return candidate, F.concat(nested.error_paths, container_errors)
        if column_config.parser.parser_type in NUMERIC_PARSER_TYPES:
            # Top-level numerics can use Spark's ANSI-safe try_cast directly. The quoted generated
            # column name prevents SQL parser ambiguity.
            candidate = F.expr(
                "try_cast("
                f"{_quoted_identifier(normalized_name)} AS {column_config.expected_data_type})"
            )
            if column_config.parser.parser_type in {ParserType.FLOAT, ParserType.DOUBLE}:
                # Spark accepts NaN and infinities as floating values, but the parser contract treats
                # all non-finite values as parse errors for predictable downstream rules.
                candidate = F.when(
                    F.isnan(candidate) | (F.abs(candidate) == F.lit(float("inf"))),
                    F.lit(None).cast(column_config.expected_data_type),
                ).otherwise(candidate)
            return candidate, self._empty_error_paths()
        return (
            self._parse_scalar_candidate(
                normalized,
                column_config.data_type,
                column_config.parser,
            ),
            self._empty_error_paths(),
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
            if options.string_format is StringFormat.LOWER:
                return F.lower(normalized)
            if options.string_format is StringFormat.UPPER:
                return F.upper(normalized)
            if options.string_format is StringFormat.TITLE:
                # Unlike Pascal casing, title casing intentionally retains the normalized spaces.
                # Lowercasing first makes output deterministic for mixed-case bronze values.
                return F.initcap(F.lower(normalized))
            if options.string_format is StringFormat.PASCAL:
                return F.regexp_replace(F.initcap(F.lower(normalized)), r"\s+", "")
            if options.string_format is StringFormat.ADDRESS_US_V1:
                return format_address_us_v1(normalized)
            if options.string_format is StringFormat.COUNTY:
                return format_county(normalized)
            if options.string_format is StringFormat.STATE_US:
                return format_state_us(normalized)
            if options.string_format is StringFormat.ZIP:
                return format_zip(normalized)
            return normalized
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
            parsed = F.from_json(
                F.concat(F.lit('{"value":'), json_number, F.lit("}")),
                f"struct<value:{data_type.canonical}>",
                _STRICT_JSON_OPTIONS,
            ).getField("value")
            # ``from_json`` may accept a valid prefix in some runtime combinations. The explicit
            # full-token regex prevents wrapper injection and trailing garbage from slipping through.
            parsed = F.when(
                json_number.rlike(_JSON_NUMBER_PATTERN),
                parsed,
            ).otherwise(F.lit(None).cast(data_type.canonical))
            if parser_type in {ParserType.FLOAT, ParserType.DOUBLE}:
                return F.when(F.abs(parsed) == F.lit(float("inf")), F.lit(None)).otherwise(parsed)
            return parsed
        if parser_type is ParserType.BINARY:
            binary_format = {
                BinaryEncoding.BASE64: "base64",
                BinaryEncoding.HEX: "hex",
                BinaryEncoding.UTF8: "utf-8",
            }[options.binary_encoding]
            return F.try_to_binary(normalized, F.lit(binary_format))
        if parser_type is ParserType.BOOLEAN:
            # Compiler overlap validation guarantees a token cannot satisfy both branches.
            comparable = normalized if options.boolean_case_sensitive else F.lower(normalized)
            true_values = (
                options.true_values
                if options.boolean_case_sensitive
                else tuple(value.lower() for value in options.true_values)
            )
            false_values = (
                options.false_values
                if options.boolean_case_sensitive
                else tuple(value.lower() for value in options.false_values)
            )
            return (
                F.when(comparable.isin(*true_values), F.lit(True))
                .when(comparable.isin(*false_values), F.lit(False))
                .otherwise(F.lit(None).cast("boolean"))
            )
        if parser_type in {ParserType.DATE, ParserType.TIMESTAMP, ParserType.TIMESTAMP_NTZ}:
            # Try formats in author-supplied order and select the first successful result. All
            # functions are tolerant so ANSI mode does not bypass the configured error policy.
            if parser_type is ParserType.TIMESTAMP_NTZ:
                candidates = [
                    F.when(
                        F.try_to_timestamp(normalized, F.lit(datetime_format)).isNotNull(),
                        F.to_timestamp_ntz(normalized, F.lit(datetime_format)),
                    ).otherwise(F.lit(None).cast("timestamp_ntz"))
                    for datetime_format in options.formats
                ]
                return F.coalesce(*candidates)
            candidates = [
                F.try_to_timestamp(normalized, F.lit(datetime_format))
                for datetime_format in options.formats
            ]
            parsed = F.coalesce(*candidates)
            return parsed.cast("date") if parser_type is ParserType.DATE else parsed
        raise ValueError(f"Unsupported parser type: {parser_type.value}.")

    def _decode_complex(
        self,
        normalized: Column,
        data_type: SparkDataType,
        options: ParserOptions,
    ) -> _ComplexDecode:
        """Decode one complex value and separately flag malformed container syntax.

        The typed decode supplies raw child strings for recursive parsing. A second broad decode is
        used as a syntax validator so a malformed object cannot masquerade as a successful struct
        whose fields are all null.
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
        validation_type = (
            "array<string>" if options.parser_type is ParserType.ARRAY else "map<string,string>"
        )
        validation = F.from_json(normalized, validation_type, _STRICT_JSON_OPTIONS)
        invalid = validation.isNull()
        if options.parser_type is ParserType.MAP:
            # Spark's map construction fails on duplicate keys. Detect them while the value is still
            # an array-like decoded representation so the configured container policy can handle it.
            keys = F.map_keys(validation)
            duplicate_keys = F.size(keys) != F.size(F.array_distinct(keys))
            invalid = invalid | F.coalesce(duplicate_keys, F.lit(False))
        return _ComplexDecode(value, normalized.isNotNull() & invalid)

    def _parse_nested_complex(
        self,
        raw: Column,
        data_type: SparkDataType,
        options: ParserOptions,
        path: Column,
        root_config: ColumnParser,
    ) -> _NestedParse:
        """Dispatch a decoded complex value to the matching recursive container parser."""
        parser_type = data_type.parser_type
        if parser_type is ParserType.ARRAY:
            return self._parse_nested_array(raw, data_type, options, path, root_config)
        if parser_type is ParserType.STRUCT:
            return self._parse_nested_struct(raw, data_type, options, path, root_config)
        if parser_type is ParserType.MAP:
            return self._parse_nested_map(raw, data_type, options, path, root_config)
        raise ValueError(f"Expected a complex parser type, found {parser_type.value}.")

    def _parse_nested_value(
        self,
        raw: Column,
        nested: NestedValueParser | StructFieldParser,
        path: Column,
        root_config: ColumnParser,
        *,
        child_error_mode: ChildErrorMode | None = None,
    ) -> _NestedParse:
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
            )
            candidate = F.when(
                decoded.failed,
                F.lit(None).cast(nested.expected_data_type),
            ).otherwise(state.value)
            value = self._resolve_nested_parse_error(
                candidate,
                decoded.failed,
                raw,
                nested,
                path,
                root_config,
                child_error_mode,
            )
            value = self._apply_nested_default(value, nested, path)
            # Error paths are diagnostic history, not just unresolved failures. Preserve handled
            # child paths even when the final value becomes a default or the parent later drops it.
            return _NestedParse(
                value,
                decoded.failed,
                F.concat(state.error_paths, self._error_paths(decoded.failed, path)),
            )

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
        if nested.parser.parser_type in NUMERIC_PARSER_TYPES and not nested.parser.zero_is_valid:
            # Child zero invalidation follows the same order as top-level parsing: after immediate
            # parse-error resolution and before the final null default.
            value = F.when(
                value == F.lit(0).cast(nested.expected_data_type),
                F.lit(None).cast(nested.expected_data_type),
            ).otherwise(value)
        value = self._apply_nested_default(value, nested, path)
        return _NestedParse(value, failed, self._error_paths(failed, path))

    def _parse_nested_array(
        self,
        raw: Column,
        data_type: SparkDataType,
        options: ParserOptions,
        path: Column,
        root_config: ColumnParser,
    ) -> _NestedParse:
        """Parse array elements with zero-based paths and apply drop/null/distinct behavior."""
        assert options.element_parser is not None

        def parse_element(element: Column, index: Column) -> Column:
            """Return a temporary record so value and diagnostics travel through one transform."""
            element_path = F.concat(path, F.lit("["), index.cast("string"), F.lit("]"))
            parsed = self._parse_nested_value(
                element,
                options.element_parser,
                element_path,
                root_config,
                child_error_mode=options.on_element_error,
            )
            return F.struct(
                parsed.value.alias("value"),
                parsed.failed.alias("failed"),
                parsed.error_paths.alias("error_paths"),
            )

        records = F.transform(raw, parse_element)
        # Keep diagnostics from the original records. Filtering only the value projection ensures a
        # dropped bad element remains visible in ``nested_error_paths``.
        retained = (
            F.filter(records, lambda record: ~record.getField("failed"))
            if options.on_element_error is ChildErrorMode.DROP
            else records
        )
        if options.drop_null_elements:
            # This removes both original nulls and values resolved to null by child policy/defaults.
            retained = F.filter(retained, lambda record: record.getField("value").isNotNull())
        values = F.transform(retained, lambda record: record.getField("value"))
        if options.distinct:
            values = F.array_distinct(values)
        error_paths = F.flatten(F.transform(records, lambda record: record.getField("error_paths")))
        return _NestedParse(
            F.when(raw.isNull(), F.lit(None).cast(data_type.canonical)).otherwise(values),
            F.lit(False),
            F.when(raw.isNull(), self._empty_error_paths()).otherwise(error_paths),
        )

    def _parse_nested_struct(
        self,
        raw: Column,
        data_type: SparkDataType,
        options: ParserOptions,
        path: Column,
        root_config: ColumnParser,
    ) -> _NestedParse:
        """Parse every configured struct field and emit fields in compiled schema order."""
        parsed_fields: list[Column] = []
        errors: list[Column] = []
        for field in options.field_parsers:
            field_path = F.concat(path, F.lit("."), F.lit(field.silver_field_name))
            parsed = self._parse_nested_value(
                raw.getField(field.source_field_name),
                field,
                field_path,
                root_config,
            )
            parsed_fields.append(parsed.value.alias(field.silver_field_name))
            errors.append(parsed.error_paths)
        value = F.struct(*parsed_fields)
        # Spark's struct() itself is non-null even when every child is null. Restore a truly null
        # struct when the decoded raw container was null.
        return _NestedParse(
            F.when(raw.isNull(), F.lit(None).cast(data_type.canonical)).otherwise(value),
            F.lit(False),
            F.when(raw.isNull(), self._empty_error_paths()).otherwise(F.concat(*errors)),
        )

    def _parse_nested_map(
        self,
        raw: Column,
        data_type: SparkDataType,
        options: ParserOptions,
        path: Column,
        root_config: ColumnParser,
    ) -> _NestedParse:
        """Parse map values while preserving keys and recording key-specific error paths."""
        assert options.value_parser is not None

        def parse_entry(entry: Column) -> Column:
            """Attach the key, parsed value, failure flag, and paths to one temporary entry."""
            key = entry.getField("key")
            value_path = F.concat(path, F.lit("['"), key, F.lit("']"))
            parsed = self._parse_nested_value(
                entry.getField("value"),
                options.value_parser,
                value_path,
                root_config,
                child_error_mode=options.on_value_error,
            )
            return F.struct(
                key.alias("key"),
                parsed.value.alias("value"),
                parsed.failed.alias("failed"),
                parsed.error_paths.alias("error_paths"),
            )

        records = F.transform(F.map_entries(raw), parse_entry)
        retained = (
            F.filter(records, lambda record: ~record.getField("failed"))
            if options.on_value_error is ChildErrorMode.DROP
            else records
        )
        if options.drop_null_values:
            retained = F.filter(retained, lambda record: record.getField("value").isNotNull())
        # Remove helper fields before map_from_entries; only key/value pairs belong in silver data.
        entries = F.transform(
            retained,
            lambda record: F.struct(
                record.getField("key").alias("key"),
                record.getField("value").alias("value"),
            ),
        )
        value = F.map_from_entries(entries)
        error_paths = F.flatten(F.transform(records, lambda record: record.getField("error_paths")))
        return _NestedParse(
            F.when(raw.isNull(), F.lit(None).cast(data_type.canonical)).otherwise(value),
            F.lit(False),
            F.when(raw.isNull(), self._empty_error_paths()).otherwise(error_paths),
        )

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
        child mode, so their own fail/null/default policy applies.
        """
        message = F.concat(
            F.lit(
                "Spark Parser could not parse nested value for source "
                f"{root_config.source_column_name!r} into silver column "
                f"{root_config.silver_column_name!r} at "
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
        return F.when(
            failed,
            F.raise_error(message).cast(nested.expected_data_type),
        ).otherwise(candidate)

    def _apply_nested_default(
        self,
        value: Column,
        nested: NestedValueParser | StructFieldParser,
        path: Column,
    ) -> Column:
        """Apply a nested parser's final nullability/default contract."""
        # ``path`` is accepted for symmetry and future diagnostics. It currently has no bearing on
        # the typed literal, so delete it explicitly to document that intentional non-use.
        del path
        if nested.parser.is_nullable:
            return value
        return F.when(
            value.isNull(),
            self._typed_value_literal(
                nested.parser.default_on_null,
                nested.data_type,
                nested.parser,
            ),
        ).otherwise(value)

    def _raw_type_ddl(self, data_type: SparkDataType, options: ParserOptions) -> str:
        """Return the permissive intermediate schema used to decode raw child text.

        Leaves decode as strings so the configured leaf parser—not Spark's JSON coercion—owns
        normalization, formatting, and error policy. Complex children are recursively decoded from
        those captured strings.
        """
        if not data_type.is_complex:
            return "string"
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
        """Return a typed empty path array so concat/flatten schemas stay stable."""
        return F.array().cast("array<string>")

    def _error_paths(self, failed: Column, path: Column) -> Column:
        """Return ``[path]`` when failed, otherwise a typed empty array."""
        return F.filter(
            F.array(F.when(failed, path)),
            lambda item: item.isNotNull(),
        )

    def _resolve_parse_error(
        self,
        candidate: Column,
        parse_failed: Column,
        source: Column,
        column_config: ColumnParser,
    ) -> Column:
        """Resolve a top-level parse failure according to fail/null/default policy."""
        options = column_config.parser
        if options.on_parse_error is ParseErrorMode.NULL:
            return candidate
        if options.on_parse_error is ParseErrorMode.DEFAULT:
            return F.when(
                parse_failed,
                self._typed_literal(
                    options.default_on_error,
                    column_config.data_type,
                    options,
                ),
            ).otherwise(candidate)
        message = F.concat(
            F.lit(
                f"Spark Parser could not parse source {column_config.source_column_name!r} "
                f"into silver column {column_config.silver_column_name!r} as "
                f"{column_config.expected_data_type}: "
            ),
            source,
        )
        # raise_error remains a lazy Spark expression. It fires only if an action materializes this
        # silver value; a count whose projection prunes the value may not evaluate it.
        return F.when(
            parse_failed,
            F.raise_error(message).cast(column_config.expected_data_type),
        ).otherwise(candidate)

    def _typed_literal(
        self,
        value: Any,
        data_type: SparkDataType,
        options: ParserOptions,
    ) -> Column:
        """Compatibility wrapper for constructing a recursively typed Spark literal."""
        return self._typed_value_literal(value, data_type, options)

    def _typed_value_literal(
        self,
        value: Any,
        data_type: SparkDataType,
        options: ParserOptions,
    ) -> Column:
        """Recursively convert a compiler-validated Python default into native Spark literals."""
        if value is None:
            return F.lit(None).cast(data_type.canonical)
        if data_type.parser_type is ParserType.BINARY:
            binary_format = {
                BinaryEncoding.BASE64: "base64",
                BinaryEncoding.HEX: "hex",
                BinaryEncoding.UTF8: "utf-8",
            }[options.binary_encoding]
            return F.try_to_binary(F.lit(value), F.lit(binary_format))
        if data_type.parser_type is ParserType.ARRAY:
            assert data_type.element_type is not None and options.element_parser is not None
            items = [
                self._typed_value_literal(
                    item,
                    data_type.element_type,
                    options.element_parser.parser,
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
                    value[field.silver_field_name],
                    field.data_type,
                    field.parser,
                ).alias(field.silver_field_name)
                for field in options.field_parsers
            ]
            return F.struct(*fields).cast(data_type.canonical)
        if data_type.parser_type is ParserType.MAP:
            assert data_type.value_type is not None and options.value_parser is not None
            pairs: list[Column] = []
            for key, item in value.items():
                pairs.extend(
                    (
                        F.lit(key),
                        self._typed_value_literal(
                            item,
                            data_type.value_type,
                            options.value_parser.parser,
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
            matched = F.lower(value).isin(*(marker.lower() for marker in options.null_markers))
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
    ) -> Column:
        """Assemble one stable, human-explainable parser audit struct expression."""
        options = column_config.parser
        parse_to_null = parse_failed & F.lit(options.on_parse_error is ParseErrorMode.NULL)
        parse_default = parse_failed & F.lit(options.on_parse_error is ParseErrorMode.DEFAULT)
        # A top-level malformed container is already represented by parse_failed. Only separately
        # label nested resolution when one or more child paths failed inside a valid container.
        nested_failed = (F.size(nested_error_paths) > 0) & ~parse_failed
        is_zip = options.string_format is StringFormat.ZIP
        compact_zip = F.regexp_replace(normalized, r"\s+", "")
        zip_digit_count = F.length(F.regexp_replace(compact_zip, r"\D", ""))
        zip_plus4 = (
            F.coalesce(candidate.contains("-") & (candidate != compact_zip), F.lit(False))
            if is_zip
            else F.lit(False)
        )
        zip_padded = (
            F.coalesce(
                candidate.isNotNull()
                & (zip_digit_count < F.when(candidate.contains("-"), F.lit(9)).otherwise(F.lit(5))),
                F.lit(False),
            )
            if is_zip
            else F.lit(False)
        )
        # Action order is intentional and stable. Null placeholders are filtered in Spark so each
        # row receives only the transformations that actually occurred.
        actions = F.filter(
            F.array(
                F.when(source_missing, F.lit("source_column_missing")),
                F.when(empty_to_null, F.lit("empty_string_to_null")),
                F.when(marker_replaced, F.lit("null_marker_replaced")),
                F.when(json_null, F.lit("json_null_to_null")),
                F.when(parse_to_null, F.lit("parse_error_to_null")),
                F.when(parse_default, F.lit("parse_error_default_applied")),
                F.when(nested_failed, F.lit("nested_parse_errors_resolved")),
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
                | nested_failed
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
            F.lit(column_config.silver_column_name).alias("silver_column_name"),
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
        )

    def _options_map(self, options: ParserOptions) -> Column:
        """Serialize effective parser options into a Spark ``map<string,string>`` expression."""
        payload = ParserConfigSerializer().parser_mapping(
            options,
            include_audit=True,
            include_error_mode=True,
        )
        payload.setdefault("default_on_null", None)
        payload.setdefault("default_on_error", None)
        pairs: list[Column] = []
        # Stringifying values gives every audit row one stable map type even though option values in
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
