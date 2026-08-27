"""Native Spark-expression runtime for bronze string parsing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from spark_parser.dataframe_parsing import DataFrameParsing
from spark_parser.enums import NUMERIC_PARSER_TYPES, ParseErrorMode, ParserType, StringFormat
from spark_parser.exceptions import SchemaValidationError
from spark_parser.models import ColumnParser, ParserConfig, ParserOptions
from spark_parser.serializer import ParserConfigSerializer
from spark_parser.version import __version__

PARSE_RESULT_STRUCT = T.StructType(
    [
        T.StructField("column_name", T.StringType(), False),
        T.StructField("parser_type", T.StringType(), False),
        T.StructField("data_type", T.StringType(), False),
        T.StructField("original_value", T.StringType(), True),
        T.StructField("parsed_value", T.StringType(), True),
        T.StructField("changed", T.BooleanType(), False),
        T.StructField("effective", T.BooleanType(), False),
        T.StructField("actions_applied", T.ArrayType(T.StringType(), False), False),
        T.StructField("options", T.MapType(T.StringType(), T.StringType(), False), False),
        T.StructField("error", T.StringType(), True),
    ]
)


def _column(name: str) -> Column:
    """Resolve one top-level column by its literal name."""
    return F.col(f"`{name.replace('`', '``')}`")


def _quoted_identifier(name: str) -> str:
    """Quote one top-level name for use in a Spark SQL expression."""
    return f"`{name.replace('`', '``')}`"


class SparkDataFrameParser:
    """Parse configured bronze columns without Python or pandas UDFs."""

    def parse_dataframe(
        self,
        df: DataFrame,
        config: ParserConfig,
        *,
        key_columns: Sequence[str] | None = None,
        column_prefix: str = "spark_parser",
    ) -> DataFrameParsing:
        """Build lazy parsed and row-level audit projections.

        The input contract is deliberately strict: every configured source
        field must exist exactly once and have Spark ``string`` type. Invalid
        non-null values raise during the first Spark action unless their parser
        explicitly selects ``on_parse_error: null`` or ``default``.
        """
        if not isinstance(config, ParserConfig):
            raise TypeError("config must be a ParserConfig.")
        if not isinstance(column_prefix, str) or not column_prefix:
            raise ValueError("column_prefix must be a non-empty string.")
        normalized_keys = self._validate_schema(df, config, key_columns, column_prefix)

        working = df
        parsed_columns: list[tuple[str, str]] = []
        audit_structs: list[Column] = []
        for column_config in config.columns:
            working, final_column, audit_struct = self._apply_parser(working, column_config)
            parsed_columns.append((column_config.column_name, final_column))
            if audit_struct is not None:
                audit_structs.append(audit_struct)

        parse_results_name = f"{column_prefix}_parse_results"
        config_name = f"{column_prefix}_config"
        engine_version_name = f"{column_prefix}_engine_version"
        parse_results = (
            F.array(*audit_structs)
            if audit_structs
            else F.from_json(
                F.lit("[]"),
                T.ArrayType(PARSE_RESULT_STRUCT, containsNull=False),
            )
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
        return DataFrameParsing(
            working,
            parsed_columns=parsed_columns,
            key_columns=normalized_keys,
            result_columns=(parse_results_name, config_name, engine_version_name),
        )

    def _validate_schema(
        self,
        df: DataFrame,
        config: ParserConfig,
        key_columns: Sequence[str] | None,
        column_prefix: str,
    ) -> tuple[str, ...]:
        duplicates = sorted({name for name in df.columns if df.columns.count(name) > 1})
        configured_names = {column.column_name for column in config.columns}
        ambiguous = sorted(configured_names & set(duplicates))
        if ambiguous:
            raise SchemaValidationError(f"Configured input columns are ambiguous: {ambiguous}.")
        missing = sorted(configured_names - set(df.columns))
        if missing:
            raise SchemaValidationError(f"Configured input columns are missing: {missing}.")
        field_types = {field.name: field.dataType for field in df.schema.fields}
        non_string = {
            name: field_types[name].simpleString()
            for name in sorted(configured_names)
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
        return normalized_keys

    def _apply_parser(
        self,
        df: DataFrame,
        column_config: ColumnParser,
    ) -> tuple[DataFrame, str, Column | None]:
        options = column_config.parser
        token = uuid4().hex
        normalized_name = f"__spark_parser_normalized_{token}"
        candidate_name = f"__spark_parser_candidate_{token}"
        post_parse_name = f"__spark_parser_post_parse_{token}"
        post_zero_name = f"__spark_parser_post_zero_{token}"
        final_name = f"__spark_parser_final_{token}"

        source = _column(column_config.column_name)
        whitespace_normalized = (
            F.regexp_replace(source, r"\s+", " ") if options.collapse_whitespace else source
        )
        whitespace_normalized = (
            F.trim(whitespace_normalized) if options.trim_whitespace else whitespace_normalized
        )
        empty_to_null = F.coalesce(
            (whitespace_normalized == "") & F.lit(options.empty_is_null),
            F.lit(False),
        )
        marker_match = self._null_marker_match(whitespace_normalized, options)
        normalized = (
            F.when(empty_to_null, F.lit(None).cast("string"))
            .when(
                marker_match & F.lit(options.replace_null_markers),
                F.lit(None).cast("string"),
            )
            .otherwise(whitespace_normalized)
        )
        working = df.withColumn(normalized_name, normalized)
        working = working.withColumn(
            candidate_name,
            self._parse_candidate(
                _column(normalized_name),
                normalized_name,
                column_config,
            ),
        )

        parse_failed = _column(normalized_name).isNotNull() & _column(candidate_name).isNull()
        post_parse = self._resolve_parse_error(
            _column(candidate_name),
            parse_failed,
            source,
            column_config,
        )
        working = working.withColumn(post_parse_name, post_parse)

        zero_invalidated = F.lit(False)
        if options.parser_type in NUMERIC_PARSER_TYPES and not options.zero_is_valid:
            zero_invalidated = F.coalesce(
                _column(post_parse_name) == F.lit(0).cast(column_config.data_type),
                F.lit(False),
            )
            post_zero = F.when(
                zero_invalidated,
                F.lit(None).cast(column_config.data_type),
            ).otherwise(_column(post_parse_name))
        else:
            post_zero = _column(post_parse_name)
        working = working.withColumn(post_zero_name, post_zero)

        default_on_null_applied = F.lit(False)
        if not options.is_nullable:
            default_on_null_applied = _column(post_zero_name).isNull()
            final_value = F.when(
                default_on_null_applied,
                self._typed_literal(options.default_on_null, column_config.data_type),
            ).otherwise(_column(post_zero_name))
        else:
            final_value = _column(post_zero_name)
        working = working.withColumn(final_name, final_value)

        if not options.audit:
            return working, final_name, None
        audit_struct = self._audit_struct(
            column_config,
            source=source,
            parsed=_column(final_name),
            empty_to_null=empty_to_null,
            marker_replaced=(marker_match & F.lit(options.replace_null_markers) & ~empty_to_null),
            parse_failed=parse_failed,
            zero_invalidated=zero_invalidated,
            default_on_null_applied=default_on_null_applied,
        )
        return working, final_name, audit_struct

    def _parse_candidate(
        self,
        normalized: Column,
        normalized_name: str,
        column_config: ColumnParser,
    ) -> Column:
        options = column_config.parser
        parser_type = options.parser_type
        if parser_type is ParserType.STRING:
            if options.string_format is StringFormat.LOWER:
                return F.lower(normalized)
            if options.string_format is StringFormat.UPPER:
                return F.upper(normalized)
            if options.string_format is StringFormat.PASCAL:
                return F.regexp_replace(F.initcap(F.lower(normalized)), r"\s+", "")
            return normalized
        if parser_type in NUMERIC_PARSER_TYPES:
            return F.expr(
                f"try_cast({_quoted_identifier(normalized_name)} AS {column_config.data_type})"
            )
        if parser_type is ParserType.BOOLEAN:
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
        if parser_type in {ParserType.DATE, ParserType.TIMESTAMP}:
            candidates = [
                F.try_to_timestamp(normalized, F.lit(datetime_format))
                for datetime_format in options.formats
            ]
            parsed = F.coalesce(*candidates)
            return parsed.cast("date") if parser_type is ParserType.DATE else parsed
        raise ValueError(f"Unsupported parser type: {parser_type.value}.")

    def _resolve_parse_error(
        self,
        candidate: Column,
        parse_failed: Column,
        source: Column,
        column_config: ColumnParser,
    ) -> Column:
        options = column_config.parser
        if options.on_parse_error is ParseErrorMode.NULL:
            return candidate
        if options.on_parse_error is ParseErrorMode.DEFAULT:
            return F.when(
                parse_failed,
                self._typed_literal(options.default_on_error, column_config.data_type),
            ).otherwise(candidate)
        message = F.concat(
            F.lit(
                f"Spark Parser could not parse {column_config.column_name!r} as "
                f"{column_config.data_type}: "
            ),
            source,
        )
        return F.when(
            parse_failed,
            F.raise_error(message).cast(column_config.data_type),
        ).otherwise(candidate)

    @staticmethod
    def _typed_literal(value: Any, data_type: str) -> Column:
        return F.lit(value).cast(data_type)

    @staticmethod
    def _null_marker_match(value: Column, options: ParserOptions) -> Column:
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
        parse_failed: Column,
        zero_invalidated: Column,
        default_on_null_applied: Column,
    ) -> Column:
        options = column_config.parser
        parse_to_null = parse_failed & F.lit(options.on_parse_error is ParseErrorMode.NULL)
        parse_default = parse_failed & F.lit(options.on_parse_error is ParseErrorMode.DEFAULT)
        actions = F.filter(
            F.array(
                F.when(empty_to_null, F.lit("empty_string_to_null")),
                F.when(marker_replaced, F.lit("null_marker_replaced")),
                F.when(parse_to_null, F.lit("parse_error_to_null")),
                F.when(parse_default, F.lit("parse_error_default_applied")),
                F.when(zero_invalidated, F.lit("zero_invalidated")),
                F.when(default_on_null_applied, F.lit("default_on_null_applied")),
            ),
            lambda item: item.isNotNull(),
        )
        changed = F.coalesce(
            (
                empty_to_null
                | marker_replaced
                | parse_to_null
                | parse_default
                | zero_invalidated
                | default_on_null_applied
            ),
            F.lit(False),
        )
        error = F.when(
            parse_failed,
            F.lit(f"Value could not be parsed as {column_config.data_type}."),
        )
        return F.struct(
            F.lit(column_config.column_name).alias("column_name"),
            F.lit(options.parser_type.value).alias("parser_type"),
            F.lit(column_config.data_type).alias("data_type"),
            source.alias("original_value"),
            parsed.cast("string").alias("parsed_value"),
            changed.alias("changed"),
            F.lit(True).alias("effective"),
            actions.alias("actions_applied"),
            self._options_map(options).alias("options"),
            error.alias("error"),
        )

    def _options_map(self, options: ParserOptions) -> Column:
        payload: dict[str, Any] = {
            "trim_whitespace": options.trim_whitespace,
            "collapse_whitespace": options.collapse_whitespace,
            "empty_is_null": options.empty_is_null,
            "replace_null_markers": options.replace_null_markers,
            "null_markers": list(options.null_markers),
            "null_markers_mode": options.null_markers_mode.value,
            "null_marker_case_sensitive": options.null_marker_case_sensitive,
            "is_nullable": options.is_nullable,
            "default_on_null": options.default_on_null,
            "on_parse_error": options.on_parse_error.value,
            "default_on_error": options.default_on_error,
            "audit": options.audit,
        }
        if options.parser_type in NUMERIC_PARSER_TYPES:
            payload["zero_is_valid"] = options.zero_is_valid
        if options.parser_type is ParserType.STRING:
            payload["format"] = (
                options.string_format.value if options.string_format is not None else None
            )
        if options.parser_type in {ParserType.DATE, ParserType.TIMESTAMP}:
            payload["formats"] = list(options.formats)
        if options.parser_type is ParserType.BOOLEAN:
            payload.update(
                true_values=list(options.true_values),
                false_values=list(options.false_values),
                boolean_case_sensitive=options.boolean_case_sensitive,
            )
        pairs: list[Column] = []
        for key, value in payload.items():
            pairs.extend((F.lit(key), F.lit(self._option_text(value))))
        return F.create_map(*pairs)

    @staticmethod
    def _option_text(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (list, tuple, Mapping)):
            return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"))
        return str(value)
