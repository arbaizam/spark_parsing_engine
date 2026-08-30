# Databricks notebook source
# ruff: noqa: BLE001, E402, F821, I001
# MAGIC %md
# MAGIC # Spark Parser User Guide
# MAGIC
# MAGIC Spark Parser converts load-specific string columns into explicitly typed target columns
# MAGIC using strict YAML configuration and native Spark SQL expressions.
# MAGIC
# MAGIC This guide covers the complete application workflow:
# MAGIC
# MAGIC 1. Discover parser types, arguments, and defaults.
# MAGIC 2. Author and review a parser configuration.
# MAGIC 3. Compile YAML into an immutable configuration.
# MAGIC 4. Bind the configuration to a bronze DataFrame.
# MAGIC 5. Materialize typed target and audit outputs.
# MAGIC 6. Understand nulls, defaults, parse errors, and nested containers.
# MAGIC 7. Handle missing or invalid input schemas.
# MAGIC 8. Integrate the parser into a larger ingestion pipeline.
# MAGIC
# MAGIC The package performs parsing only. Reading source files, writing bronze or target tables,
# MAGIC schema evolution, checkpoints, and orchestration belong to the surrounding integration
# MAGIC layer.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load The Package From The Repository
# MAGIC
# MAGIC This notebook is designed to run from a Databricks Git checkout. The bootstrap below puts
# MAGIC the checkout's `src` directory first on `sys.path`, ensuring the notebook exercises the
# MAGIC source currently under review instead of an older installed wheel.

# COMMAND ----------

import json
import os
import sys
import warnings
from decimal import Decimal
from pathlib import Path

root = next(
    (
        path
        for path in [Path.cwd(), *Path.cwd().parents]
        if (path / "pyproject.toml").is_file()
        and (path / "src" / "spark_parser").is_dir()
    ),
    None,
)
if root:
    src_path = os.path.normpath(root / "src")
    if src_path in sys.path:
        sys.path.remove(src_path)
    print(f"Adding source checkout to sys.path: {src_path}")
    sys.path.insert(0, src_path)

from pyspark.sql import functions as F

import spark_parser
from spark_parser import ConfigReviewReport, SchemaValidationError, SparkParserService, parser

assert root is not None, (
    "Could not locate the spark_parser repository root. Run this notebook from a Databricks "
    "Git checkout containing pyproject.toml and src/spark_parser."
)
REPO_ROOT = root

print(f"Spark Parser version: {spark_parser.__version__}")
print(f"Spark Parser package: {spark_parser.__file__}")
print(f"Spark version: {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Understand The Processing Model
# MAGIC
# MAGIC For each configured column, Spark Parser builds one native Spark expression pipeline:
# MAGIC
# MAGIC ```text
# MAGIC bronze string
# MAGIC     → whitespace normalization
# MAGIC     → empty/null-marker handling
# MAGIC     → type or format parsing
# MAGIC     → configured parse-error policy
# MAGIC     → optional numeric-zero invalidation
# MAGIC     → final nullability/default handling
# MAGIC     → typed target value
# MAGIC ```
# MAGIC
# MAGIC No action runs when `parse_dataframe()` is called. It returns a `DataFrameParsing` wrapper
# MAGIC containing two lazy projections over the same expression plan:
# MAGIC
# MAGIC - `parsed_df`: configured target columns in YAML order;
# MAGIC - `results_df`: caller-selected row keys plus parser audit and configuration identity.
# MAGIC
# MAGIC Spark evaluates parsed expressions only when an action consumes them. Actions such as
# MAGIC `display()`, `collect()`, and a target write normally do that. Do not use `count()` to test
# MAGIC a parse failure: Spark may prune an unused parsed expression, and that optimizer choice can
# MAGIC vary by runtime and plan.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Discover Parsers, Arguments, And Defaults
# MAGIC
# MAGIC The package-level `parser` object is the recommended public service. It exposes compiler,
# MAGIC runtime, serializer, metadata, and configuration-review operations. Creating
# MAGIC `SparkParserService()` directly is useful only when an application wants an independently
# MAGIC constructed service object.
# MAGIC
# MAGIC Supported top-level parser types are:
# MAGIC
# MAGIC - `string`
# MAGIC - `byte`, `short`, `integer`, and `long`
# MAGIC - `float`, `decimal`, and `double`
# MAGIC - `binary` and `boolean`
# MAGIC - `date`, `timestamp`, and `timestamp_ntz`
# MAGIC - recursive `array`, `struct`, and string-keyed `map`
# MAGIC
# MAGIC `parser.defaults()` exposes the code-owned defaults for inspection; changing the returned
# MAGIC mapping does not reconfigure the package. YAML `globals` supplies shared null-marker and
# MAGIC Boolean vocabularies. Other common and parser-specific behavior is configured on each
# MAGIC parser node so the compiled load contract remains explicit.

# COMMAND ----------

assert isinstance(parser, SparkParserService)

parser_catalog = parser.describe()
print("Parser types:", ", ".join(parser_catalog))
print()
print("Decimal parser metadata:")
print(json.dumps(parser.decimal.describe(), indent=2, default=str))

config_metadata = parser.config.describe()
print()
print("Top-level configuration arguments:")
print(", ".join(argument["name"] for argument in config_metadata["top_level_arguments"]))

detached_defaults = parser.defaults()
print()
print("Code-owned defaults:")
print(json.dumps(detached_defaults, indent=2, default=str))

# `defaults()` returns a detached copy. Mutating it cannot change the compiler's defaults.
detached_defaults["common"]["trim_whitespace"] = False
assert parser.defaults()["common"]["trim_whitespace"] is True

assert parser.normalize_data_type(" array < struct < amount : decimal(10, 2) > > ") == (
    "array<struct<amount:decimal(10,2)>>"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Author One Parser Configuration
# MAGIC
# MAGIC A configuration describes one source-to-target parsing contract. Required metadata is:
# MAGIC
# MAGIC - `parser_config_id`
# MAGIC - `parser_config_name`
# MAGIC - `version`
# MAGIC - a non-empty ordered `columns` list
# MAGIC
# MAGIC Every column declares an exact source name, unique target name, expected Spark datatype, and
# MAGIC parser. Global null and Boolean vocabularies are inherited by columns unless a column
# MAGIC explicitly replaces or extends them.
# MAGIC
# MAGIC This example deliberately demonstrates successful values, handled errors, recursive JSON,
# MAGIC source fan-out, auditing, and one source column that is not delivered by the DataFrame.

# COMMAND ----------

config_yaml = """
parser_config_id: customer_user_guide
parser_config_name: Customer User Guide
version: "1"
description: Demonstrate scalar, recursive, error, and audit behavior.
owner: Data Engineering
owner_department: Enterprise Data

globals:
  null_markers: [NA, N/A]
  null_marker_case_sensitive: false
  true_values: ["true", Y, "yes"]
  false_values: ["false", N, "no"]
  boolean_case_sensitive: false

columns:
  - source_column_name: record_id
    target_column_name: RecordId
    expected_data_type: string
    parser: string

  - source_column_name: customer_name
    target_column_name: CustomerName
    expected_data_type: string
    parser:
      type: string
      format: title
      replace_null_markers: true
      audit: true

  - source_column_name: state
    target_column_name: StateCode
    expected_data_type: string
    parser:
      type: string
      format: state_us
      on_parse_error: preserve
      audit: true

  - source_column_name: amount
    target_column_name: Amount
    expected_data_type: decimal(12,2)
    parser:
      type: decimal
      on_parse_error: null
      audit: true

  - source_column_name: quantity
    target_column_name: Quantity
    expected_data_type: integer
    parser:
      type: integer
      on_parse_error: default
      default_on_error: 0
      audit: true

  - source_column_name: active_flag
    target_column_name: IsActive
    expected_data_type: boolean
    parser:
      type: boolean
      on_parse_error: null
      audit: true

  - source_column_name: event_date
    target_column_name: EventDate
    expected_data_type: date
    parser:
      type: date
      on_parse_error: null
      audit: true

  - source_column_name: event_timestamp
    target_column_name: EventTimestamp
    expected_data_type: timestamp
    parser: timestamp

  - source_column_name: event_timestamp
    target_column_name: EventTimestampNtz
    expected_data_type: timestamp_ntz
    parser: timestamp_ntz

  - source_column_name: aliases
    target_column_name: Aliases
    expected_data_type: array<string>
    parser:
      type: array
      element_parser:
        type: string
        format: upper
      on_element_error: drop
      drop_null_elements: true
      distinct: true
      on_parse_error: default
      default_on_error: [UNKNOWN]
      audit: true

  - source_column_name: profile
    target_column_name: Profile
    expected_data_type: struct<postal_code:string,scores:array<integer>>
    parser:
      type: struct
      fields:
        - source_field_name: zip_code
          target_field_name: postal_code
          parser:
            type: string
            format: zip
            on_parse_error: null
        - source_field_name: raw_scores
          target_field_name: scores
          parser:
            type: array
            element_parser:
              type: integer
              zero_is_valid: false
              is_nullable: false
              default_on_null: -1
            on_element_error: null
      on_parse_error: default
      default_on_error:
        postal_code: "00000"
        scores: []
      audit: true

  - source_column_name: attributes
    target_column_name: Attributes
    expected_data_type: map<string,decimal(12,2)>
    parser:
      type: map
      value_parser: decimal
      on_value_error: drop
      on_parse_error: null
      audit: true

  - source_column_name: source_not_delivered
    target_column_name: MissingSourceValue
    expected_data_type: string
    parser:
      type: string
      is_nullable: false
      default_on_null: UNKNOWN
      audit: true
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Review Before Compiling
# MAGIC
# MAGIC `review_yaml()` is intended for notebooks, authoring tools, and CI reports. It catches
# MAGIC expected authoring failures and returns a `ConfigReviewReport` instead of raising. A valid
# MAGIC report contains resolved options, validation evidence, warnings, a canonical configuration,
# MAGIC and a deterministic content hash.
# MAGIC
# MAGIC Review is configuration-only. It cannot report missing DataFrame columns or invalid source
# MAGIC Spark types because no DataFrame has been supplied yet.

# COMMAND ----------

review = parser.review_yaml(config_yaml)
assert isinstance(review, ConfigReviewReport)
assert review.is_valid, review.errors
assert not review.warnings, review.warnings

print(review.to_markdown())

print()
print("Configuration review summary:")
print(json.dumps(review.summary, indent=2, default=str))

# An invalid review remains inspectable without swallowing programming errors outside compilation.
invalid_review = parser.review_yaml("parser_config_id: incomplete")
assert invalid_review.is_valid is False
assert invalid_review.errors
print()
print("Example invalid review:")
print(invalid_review.to_markdown())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compile To The Runtime Contract
# MAGIC
# MAGIC Compilation is strict and Spark-independent. It rejects duplicate YAML keys, unknown
# MAGIC arguments, incompatible parser/datatype pairs, contradictory options, invalid formats, and
# MAGIC defaults that cannot be represented by their declared Spark datatype.
# MAGIC
# MAGIC The returned `ParserConfig` is immutable and fully resolved. The runtime does not reinterpret
# MAGIC YAML defaults after compilation.

# COMMAND ----------

config = parser.compile_text(config_yaml)
config_mapping = parser.to_mapping(config)
config_hash = parser.content_hash(config)

assert config.parser_config_id == "customer_user_guide"
assert config_hash == review.summary["content_hash"]
assert config_mapping == review.resolved_config
assert parser.compile_mapping(config_mapping) == config

print(f"Config ID: {config.parser_config_id}")
print(f"Config version: {config.version}")
print(f"Configured target columns: {len(config.columns)}")
print(f"Resolved configuration SHA-256: {config_hash}")

# `canonical_json()` provides deterministic JSON for storage, comparison, and hashing.
canonical_json = parser.canonical_json(config)
assert json.loads(canonical_json) == config_mapping

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Prepare A Bronze DataFrame
# MAGIC
# MAGIC Present configured source columns must be top-level Spark `string` columns. This is an
# MAGIC intentional boundary: ingestion preserves source tokens in bronze, and Spark Parser owns
# MAGIC the explicit conversion from those strings to target types.
# MAGIC
# MAGIC Source columns may feed more than one target interpretation. In this guide,
# MAGIC `event_timestamp` becomes both `timestamp` and `timestamp_ntz`.
# MAGIC
# MAGIC Missing configured sources are recoverable. They produce typed null/default target values,
# MAGIC a warning on the returned wrapper, and an audit record when auditing is enabled.

# COMMAND ----------

bronze_schema = """
record_id string,
customer_name string,
state string,
amount string,
quantity string,
active_flag string,
event_date string,
event_timestamp string,
aliases string,
profile string,
attributes string
"""

bronze_rows = [
    (
        "customer-1",
        "  alice   smith ",
        "Illinois",
        "12.345",
        "7",
        "Y",
        "09/30/2026 12:00:00 AM",
        "2026-09-30T13:45:00",
        '[" ally ","ALLY",null]',
        '{"zip_code":"1234","raw_scores":[1,"bad",0]}',
        '{"principal":"10.125","bad":"x","empty":null}',
    ),
    (
        "customer-2",
        "n/a",
        "Mul",
        "not-a-decimal",
        "not-an-integer",
        "maybe",
        "not-a-date",
        "2026-08-30 09:15:00",
        "not-json",
        "not-json",
        "not-json",
    ),
]

bronze_df = spark.createDataFrame(bronze_rows, schema=bronze_schema)
display(bronze_df.orderBy("record_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Bind The Configuration And Build Lazy Outputs
# MAGIC
# MAGIC `key_columns` defines row identity in `results_df`. Keys are not automatically copied into
# MAGIC `parsed_df`; this guide maps `record_id` to `RecordId` explicitly because the target also
# MAGIC needs it.
# MAGIC
# MAGIC `column_prefix` reserves three result fields. The default is `spark_parser`, producing:
# MAGIC
# MAGIC - `spark_parser_parse_results`
# MAGIC - `spark_parser_config`
# MAGIC - `spark_parser_engine_version`

# COMMAND ----------

with warnings.catch_warnings(record=True) as captured_warnings:
    warnings.simplefilter("always")
    parsing = parser.parse_dataframe(
        bronze_df,
        config,
        key_columns=["record_id"],
        on_missing_source="warn",
    )

assert parsing.key_columns == ("record_id",)
assert parsing.result_columns == (
    "spark_parser_parse_results",
    "spark_parser_config",
    "spark_parser_engine_version",
)
assert parsing.warnings and "source_not_delivered" in parsing.warnings[0]

for captured in captured_warnings:
    print(f"Schema warning: {captured.message}")

target_df = parsing.parsed_df
audit_df = parsing.results_df

print("Target columns:", target_df.columns)
print("Audit/result columns:", audit_df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Inspect Typed Target Values
# MAGIC
# MAGIC `parsed_df` contains only configured target columns, in configuration order. Normal
# MAGIC successful conversions remain native Spark values: decimals stay decimals, dates stay
# MAGIC dates, timestamps stay timestamps, and complex parsers produce typed arrays, structs, and
# MAGIC maps.

# COMMAND ----------

display(target_df.orderBy("RecordId"))
target_df.printSchema()

target_rows = {
    row.RecordId: row.asDict(recursive=True)
    for row in target_df.orderBy("RecordId").collect()
}

successful = target_rows["customer-1"]
assert successful["CustomerName"] == "Alice Smith"
assert successful["StateCode"] == "IL"
assert successful["Amount"] == Decimal("12.35")
assert successful["Quantity"] == 7
assert successful["IsActive"] is True
assert successful["EventDate"].isoformat() == "2026-09-30"
assert successful["EventTimestamp"].isoformat(sep=" ") == "2026-09-30 13:45:00"
assert successful["EventTimestampNtz"].isoformat(sep=" ") == "2026-09-30 13:45:00"
assert successful["Aliases"] == ["ALLY"]
assert successful["Profile"]["postal_code"] == "01234"
assert successful["Profile"]["scores"] == [1, -1, -1]
assert successful["Attributes"] == {
    "principal": Decimal("10.13"),
    "empty": None,
}
assert successful["MissingSourceValue"] == "UNKNOWN"

handled = target_rows["customer-2"]
assert handled["CustomerName"] is None
assert handled["StateCode"] == "Mul"
assert handled["Amount"] is None
assert handled["Quantity"] == 0
assert handled["IsActive"] is None
assert handled["EventDate"] is None
assert handled["Aliases"] == ["UNKNOWN"]
assert handled["Profile"] == {"postal_code": "00000", "scores": []}
assert handled["Attributes"] is None
assert handled["MissingSourceValue"] == "UNKNOWN"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Inspect Row-Level Audit Results
# MAGIC
# MAGIC Only columns with `audit: true` emit parse-result structs. Each struct includes:
# MAGIC
# MAGIC - source and target identity;
# MAGIC - parser and expected datatype;
# MAGIC - original and printable parsed values;
# MAGIC - effective options;
# MAGIC - whether the value changed;
# MAGIC - actions applied and any error;
# MAGIC - nested error, default, and invalid-zero paths.
# MAGIC
# MAGIC Routine formatting is represented by the original value, parsed value, and options rather
# MAGIC than noisy action entries. Actions identify meaningful null/error/default behavior.

# COMMAND ----------

flattened_audit_df = audit_df.select(
    "record_id",
    "spark_parser_config",
    "spark_parser_engine_version",
    F.explode("spark_parser_parse_results").alias("parser_result"),
).select(
    "record_id",
    "spark_parser_config",
    "spark_parser_engine_version",
    "parser_result.*",
)

display(flattened_audit_df.orderBy("record_id", "target_column_name"))

audit_rows = {
    row.record_id: {
        result.target_column_name: result
        for result in row.spark_parser_parse_results
    }
    for row in audit_df.collect()
}

successful_audit = audit_rows["customer-1"]
assert successful_audit["Profile"].nested_error_paths == ["$.scores[1]"]
assert successful_audit["Profile"].nested_zero_invalidated_paths == ["$.scores[2]"]
assert successful_audit["Attributes"].nested_error_paths == ["$['bad']"]
assert successful_audit["MissingSourceValue"].effective is False
assert successful_audit["MissingSourceValue"].actions_applied == [
    "source_column_missing",
    "default_on_null_applied",
]

handled_audit = audit_rows["customer-2"]
assert handled_audit["CustomerName"].actions_applied == ["null_marker_replaced"]
assert handled_audit["StateCode"].actions_applied == ["parse_error_preserved"]
assert handled_audit["Amount"].actions_applied == ["parse_error_to_null"]
assert handled_audit["Quantity"].actions_applied == ["parse_error_default_applied"]
assert handled_audit["Aliases"].actions_applied == ["parse_error_default_applied"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Understand Nulls, Defaults, And Error Policies
# MAGIC
# MAGIC Common processing order is significant:
# MAGIC
# MAGIC 1. Normalize whitespace.
# MAGIC 2. Convert empty strings and configured null markers to null.
# MAGIC 3. Parse the normalized value.
# MAGIC 4. Apply `on_parse_error` when non-null input cannot parse.
# MAGIC 5. Optionally invalidate numeric zero.
# MAGIC 6. Apply `default_on_null` when the final value may not remain null.
# MAGIC
# MAGIC Top-level `on_parse_error` values are:
# MAGIC
# MAGIC | Mode | Result |
# MAGIC | --- | --- |
# MAGIC | `fail` | Raise when Spark materializes the failed parsed value. |
# MAGIC | `null` | Replace the failed value with null. |
# MAGIC | `default` | Use the required typed `default_on_error`. |
# MAGIC | `preserve` | Keep the normalized string token; valid only for string positions. |
# MAGIC
# MAGIC Arrays, structs, and maps also have child-error policies such as `drop`, `null`, and
# MAGIC `fail`. Their audit paths are consolidated into the owning top-level audit record.

# COMMAND ----------

fail_config = parser.compile_text(
    """
parser_config_id: user_guide_fail
parser_config_name: User Guide Fail Policy
version: "1"
columns:
  - source_column_name: raw_value
    target_column_name: ParsedValue
    expected_data_type: integer
    parser: integer
"""
)
fail_parsing = parser.parse_dataframe(
    spark.createDataFrame([("not-an-integer",)], "raw_value string"),
    fail_config,
    key_columns=["raw_value"],
)

# Building the parsed projection is lazy and exposes the configured target schema without
# evaluating the bad source value.
assert fail_parsing.parsed_df.columns == ["ParsedValue"]
try:
    fail_parsing.parsed_df.select("ParsedValue").collect()
except Exception as exc:  # Spark wraps executor failures differently across runtimes.
    print(f"Expected fail-mode materialization error: {type(exc).__name__}")
else:
    raise AssertionError("Fail-mode value materialized without raising")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Understand Recursive Complex Types
# MAGIC
# MAGIC Complex parsers are recursive and their declared Spark DDL drives the child datatype:
# MAGIC
# MAGIC - `array<T>` requires an `element_parser` for `T`;
# MAGIC - `struct<a:T,...>` requires one named field parser for every declared field;
# MAGIC - `map<string,T>` requires a `value_parser` for `T`.
# MAGIC
# MAGIC JSON is the default complex input format. Scalar arrays may instead use
# MAGIC `input_format: delimited` with an explicit literal delimiter. Struct field mappings use
# MAGIC `source_field_name` and `target_field_name`, allowing nested source-to-target renaming.
# MAGIC
# MAGIC The exhaustive reference at `examples/all_parsers.yaml` documents every complex option and
# MAGIC recursively valid combination.

# COMMAND ----------

complex_columns = [
    column
    for column in config.columns
    if column.parser.parser_type.value in {"array", "struct", "map"}
]
assert [column.target_column_name for column in complex_columns] == [
    "Aliases",
    "Profile",
    "Attributes",
]

print("Recursive target types:")
for column in complex_columns:
    print(f"- {column.target_column_name}: {column.expected_data_type}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Treat Input-Schema Validation Separately From YAML Compilation
# MAGIC
# MAGIC YAML compilation proves the authored parser contract. DataFrame binding then proves whether
# MAGIC a particular bronze schema can satisfy it.
# MAGIC
# MAGIC Fail-closed condition:
# MAGIC
# MAGIC - a configured source is missing. Pass `on_missing_source="warn"` only when substituting a
# MAGIC   warning and typed null/default is an intentional load-contract decision.
# MAGIC
# MAGIC Rejected conditions include:
# MAGIC
# MAGIC - a present configured source is not a top-level Spark string;
# MAGIC - configured or key columns are ambiguous because names are duplicated;
# MAGIC - explicit keys are missing or repeated;
# MAGIC - an input column collides with reserved parser result fields.

# COMMAND ----------

schema_guard_config = parser.compile_text(
    """
parser_config_id: user_guide_schema_guard
parser_config_name: User Guide Schema Guard
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: string
    parser: string
"""
)

try:
    parser.parse_dataframe(
        spark.range(1).selectExpr("id AS value"),
        schema_guard_config,
        key_columns=["value"],
    )
except SchemaValidationError as exc:
    assert "must have Spark string type" in str(exc)
    print(f"Expected schema validation error: {exc}")
else:
    raise AssertionError("A configured non-string source was accepted")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Use Configuration Identity For Traceability
# MAGIC
# MAGIC Every `results_df` row carries:
# MAGIC
# MAGIC - parser configuration ID;
# MAGIC - parser configuration version;
# MAGIC - SHA-256 of the fully resolved configuration;
# MAGIC - Spark Parser package version.
# MAGIC
# MAGIC This allows an integration layer to record exactly which parsing behavior produced a target
# MAGIC row without putting parser metadata into the business-facing `parsed_df`.

# COMMAND ----------

identity_row = audit_df.select(
    "spark_parser_config",
    "spark_parser_engine_version",
).first()

assert identity_row.spark_parser_config.id == config.parser_config_id
assert identity_row.spark_parser_config.version == config.version
assert identity_row.spark_parser_config.content_hash == config_hash
assert identity_row.spark_parser_engine_version == spark_parser.__version__

print("Parser identity:")
print(identity_row.spark_parser_config.asDict(recursive=True))
print(f"Engine version: {identity_row.spark_parser_engine_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Persist Only When Both Projections Need Materialization
# MAGIC
# MAGIC `parsed_df` and `results_df` are projections over one shared lazy plan. If a production job
# MAGIC materializes both, it can persist the wrapper before the first action:
# MAGIC
# MAGIC ```python
# MAGIC parsing = parser.parse_dataframe(bronze_df, config, key_columns=["record_id"]).persist()
# MAGIC try:
# MAGIC     parsing.parsed_df.write.format("delta").mode("append").saveAsTable(target_table)
# MAGIC     parsing.results_df.write.format("delta").mode("append").saveAsTable(audit_table)
# MAGIC finally:
# MAGIC     parsing.unpersist()
# MAGIC ```
# MAGIC
# MAGIC Persistence is optional and remains lazy until an action runs. Some serverless compute does
# MAGIC not expose all DataFrame cache operations, so the surrounding integration should use it only
# MAGIC when supported and when avoiding repeated evaluation materially helps.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Keep Pipeline Integration Outside The Parser
# MAGIC
# MAGIC A typical integration layer owns source and storage behavior while delegating only the
# MAGIC transformation contract to Spark Parser:
# MAGIC
# MAGIC ```python
# MAGIC # Integration-owned ingestion.
# MAGIC bronze_df = (
# MAGIC     spark.read.format(source_format)
# MAGIC     .options(**source_options)
# MAGIC     .load(source_path)
# MAGIC )
# MAGIC
# MAGIC # Parser-owned transformation.
# MAGIC config = parser.compile_path(parser_config_path)
# MAGIC parsing = parser.parse_dataframe(bronze_df, config, key_columns=business_keys)
# MAGIC
# MAGIC # Integration-owned persistence and orchestration.
# MAGIC parsing.parsed_df.write.format("delta").mode(target_mode).saveAsTable(target_table)
# MAGIC parsing.results_df.write.format("delta").mode("append").saveAsTable(audit_table)
# MAGIC ```
# MAGIC
# MAGIC Keep these concerns in the integration layer:
# MAGIC
# MAGIC - file format, delimiter, encoding, and discovery;
# MAGIC - catalog, schema, table, and Volume resolution;
# MAGIC - bronze ingestion and schema evolution;
# MAGIC - streaming checkpoints and triggers;
# MAGIC - target append, overwrite, or merge behavior;
# MAGIC - data-quality rules, orchestration, and operational notifications.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Authoring And Operational Checklist
# MAGIC
# MAGIC Before using a parser configuration in a load:
# MAGIC
# MAGIC 1. Give it clear identity, purpose, ownership, and version metadata.
# MAGIC 2. Keep every present configured bronze source as a top-level string.
# MAGIC 3. Declare exact target Spark datatypes rather than relying on inference.
# MAGIC 4. Choose every error policy deliberately; `fail` is the default.
# MAGIC 5. Enable audit only where the diagnostic value justifies its size.
# MAGIC 6. Supply stable `key_columns` for operational and audit joins.
# MAGIC 7. Review and compile the YAML before binding a DataFrame.
# MAGIC 8. Treat missing-source warnings as explicit load-contract decisions.
# MAGIC 9. Run pytest for compiler/runtime regression coverage.
# MAGIC 10. Run the Databricks system-test notebook for the real Spark boundary.
# MAGIC
# MAGIC Additional references:
# MAGIC
# MAGIC - `README.md`: complete public API and behavioral reference;
# MAGIC - `examples/all_parsers.yaml`: exhaustive YAML argument reference;
# MAGIC - `docs/spark_parser_unit_test_summary.md`: pytest inventory and commands;
# MAGIC - `docs/spark_parser_system_test_summary.md`: Databricks system-test inventory.

# COMMAND ----------

print()
print("=" * 80)
print("PASS: Spark Parser user guide completed successfully.")
print("=" * 80)
