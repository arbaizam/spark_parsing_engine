"""Native Spark-expression runtime for bronze string parsing."""

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

from spark_parser.address_formats import format_address_us_v1, format_county, format_zip
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
_WHITESPACE_PATTERN = r"[\s\u00A0]+"
_EDGE_WHITESPACE_PATTERN = r"^[\s\u00A0]+|[\s\u00A0]+$"
_JSON_NUMBER_PATTERN = r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$"
_STRICT_JSON_OPTIONS = {
    "mode": "PERMISSIVE",
    "allowComments": "false",
    "allowSingleQuotes": "false",
    "allowUnquotedFieldNames": "false",
    "allowNumericLeadingZeros": "false",
}
_AUDIT_JSON_OPTIONS = {
    "ignoreNullFields": "false",
    "dateFormat": "yyyy-MM-dd",
    "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSXXX",
    "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSS",
}


@dataclass(frozen=True)
class _ColumnRuntimePlan:
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
    value: Column
    failed: Column
    error_paths: Column


@dataclass(frozen=True)
class _ComplexDecode:
    value: Column
    failed: Column


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

        plans = tuple(
            self._runtime_plan(
                column_config,
                source_missing=column_config.source_column_name not in df.columns,
            )
            for column_config in config.columns
        )
        working = df.withColumns(
            {
                plan.normalized_name: self._normalized_value(
                    self._source_value(plan),
                    plan.config.parser,
                )
                for plan in plans
            }
        )
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
        working = working.withColumns(
            {plan.post_zero_name: self._post_zero_value(plan) for plan in plans}
        )
        working = working.withColumns({plan.final_name: self._final_value(plan) for plan in plans})
        parsed_columns = [(plan.config.silver_column_name, plan.final_name) for plan in plans]
        audit_structs = [self._audit_for_plan(plan) for plan in plans if plan.config.parser.audit]

        parse_results_name = f"{column_prefix}_parse_results"
        config_name = f"{column_prefix}_config"
        engine_version_name = f"{column_prefix}_engine_version"
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
        empty_to_null = F.coalesce(
            (whitespace_normalized == "") & F.lit(options.empty_is_null),
            F.lit(False),
        )
        marker_match = self._null_marker_match(whitespace_normalized, options)
        return whitespace_normalized, empty_to_null, marker_match

    def _normalized_value(self, source: Column, options: ParserOptions) -> Column:
        whitespace_normalized, empty_to_null, marker_match = self._normalization_state(
            source,
            options,
        )
        json_null = self._json_null_match(whitespace_normalized, options)
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
        if options.parser_type not in COMPLEX_PARSER_TYPES:
            return F.lit(False)
        return F.coalesce(value == F.lit("null"), F.lit(False))

    @staticmethod
    def _parse_failed(plan: _ColumnRuntimePlan) -> Column:
        return _column(plan.normalized_name).isNotNull() & _column(plan.candidate_name).isNull()

    @staticmethod
    def _zero_invalidated(plan: _ColumnRuntimePlan) -> Column:
        options = plan.config.parser
        if options.parser_type in NUMERIC_PARSER_TYPES and not options.zero_is_valid:
            return F.coalesce(
                _column(plan.post_parse_name) == F.lit(0).cast(plan.config.expected_data_type),
                F.lit(False),
            )
        return F.lit(False)

    def _post_zero_value(self, plan: _ColumnRuntimePlan) -> Column:
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
        return (
            _column(plan.post_zero_name).isNull()
            if not plan.config.parser.is_nullable
            else F.lit(False)
        )

    def _final_value(self, plan: _ColumnRuntimePlan) -> Column:
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
            container_errors = self._error_paths(decoded.failed, F.lit("$"))
            candidate = F.when(
                decoded.failed,
                F.lit(None).cast(column_config.expected_data_type),
            ).otherwise(nested.value)
            return candidate, F.concat(nested.error_paths, container_errors)
        if column_config.parser.parser_type in NUMERIC_PARSER_TYPES:
            candidate = F.expr(
                "try_cast("
                f"{_quoted_identifier(normalized_name)} AS {column_config.expected_data_type})"
            )
            if column_config.parser.parser_type in {ParserType.FLOAT, ParserType.DOUBLE}:
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
        parser_type = data_type.parser_type
        if parser_type is ParserType.STRING:
            if options.string_format is StringFormat.LOWER:
                return F.lower(normalized)
            if options.string_format is StringFormat.UPPER:
                return F.upper(normalized)
            if options.string_format is StringFormat.PASCAL:
                return F.regexp_replace(F.initcap(F.lower(normalized)), r"\s+", "")
            if options.string_format is StringFormat.ADDRESS_US_V1:
                return format_address_us_v1(normalized)
            if options.string_format is StringFormat.COUNTY:
                return format_county(normalized)
            if options.string_format is StringFormat.ZIP:
                return format_zip(normalized)
            return normalized
        if parser_type in NUMERIC_PARSER_TYPES:
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
        if (
            options.parser_type is ParserType.ARRAY
            and options.input_format is ComplexInputFormat.DELIMITED
        ):
            assert options.delimiter is not None
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
        assert options.element_parser is not None

        def parse_element(element: Column, index: Column) -> Column:
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
        retained = (
            F.filter(records, lambda record: ~record.getField("failed"))
            if options.on_element_error is ChildErrorMode.DROP
            else records
        )
        if options.drop_null_elements:
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
        assert options.value_parser is not None

        def parse_entry(entry: Column) -> Column:
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
        return F.array().cast("array<string>")

    def _error_paths(self, failed: Column, path: Column) -> Column:
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
        return self._typed_value_literal(value, data_type, options)

    def _typed_value_literal(
        self,
        value: Any,
        data_type: SparkDataType,
        options: ParserOptions,
    ) -> Column:
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
            return (F.array(*items) if items else F.array()).cast(data_type.canonical)
        if data_type.parser_type is ParserType.STRUCT:
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
            return (F.create_map(*pairs) if pairs else F.create_map()).cast(data_type.canonical)
        return F.lit(value).cast(data_type.canonical)

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
        json_null: Column,
        parse_failed: Column,
        zero_invalidated: Column,
        default_on_null_applied: Column,
        source_missing: Column,
        normalized: Column,
        candidate: Column,
        nested_error_paths: Column,
    ) -> Column:
        options = column_config.parser
        parse_to_null = parse_failed & F.lit(options.on_parse_error is ParseErrorMode.NULL)
        parse_default = parse_failed & F.lit(options.on_parse_error is ParseErrorMode.DEFAULT)
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
        payload = ParserConfigSerializer().parser_mapping(
            options,
            include_audit=True,
            include_error_mode=True,
        )
        payload.setdefault("default_on_null", None)
        payload.setdefault("default_on_error", None)
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
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)
