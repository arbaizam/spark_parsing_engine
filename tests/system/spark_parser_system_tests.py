# Databricks notebook source
# ruff: noqa: BLE001, E402, F821, I001
# MAGIC %md
# MAGIC # Spark Parser System Tests
# MAGIC
# MAGIC These tests cover behavior that requires a real Databricks Spark session: native scalar
# MAGIC and complex parsing, lazy fail-mode materialization, ANSI-mode parity, generated output
# MAGIC schemas, row-level audit data, and input-schema validation. Compiler-only permutations
# MAGIC remain in the pytest suite.
# MAGIC
# MAGIC Run this notebook from a Databricks Git checkout of the repository. It reads source directly
# MAGIC from the checkout, requires no built wheel or release artifact, and writes no tables.

# COMMAND ----------

import os
import sys
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

from spark_parser import SchemaValidationError, __version__, parser


def _start(test_id, name):
    """Print one visible system-test boundary."""
    print()
    print(f"{test_id}: {name}")
    print("-" * 80)


def _expect_raises(exception_type, operation, *, contains=None):
    """Run one operation and return its expected exception."""
    try:
        operation()
    except exception_type as exc:
        if contains is not None:
            assert contains in str(exc), (
                f"Expected {exception_type.__name__} containing {contains!r}, found {exc!r}."
            )
        return exc
    raise AssertionError(f"Expected {exception_type.__name__} to be raised.")


def _rows_by_key(df, key):
    """Collect the fixed system-test fixture into a key-indexed mapping."""
    return {row[key]: row.asDict(recursive=True) for row in df.collect()}


def _audit_by_key(df, key, results_column):
    """Index each row's audit structs by target column name."""
    return {
        row[key]: {
            result.target_column_name: result
            for result in row[results_column]
        }
        for row in df.collect()
    }


assert root is not None, (
    "Could not locate the spark_parser repository root. Run this notebook from a Databricks "
    "Git checkout containing pyproject.toml and src/spark_parser."
)
REPO_ROOT = root
REFERENCE_CONFIG_PATH = REPO_ROOT / "examples" / "all_parsers.yaml"
assert REFERENCE_CONFIG_PATH.is_file(), (
    f"Parser reference configuration does not exist: {REFERENCE_CONFIG_PATH}"
)

SYSTEM_CONFIG_YAML = """
parser_config_id: databricks_system_tests
parser_config_name: Databricks System Tests
version: "1"
description: Representative real-Spark system coverage for scalar and recursive parsing.
owner: Data Engineering
owner_department: Enterprise Data

globals:
  null_markers: [NA, N/A]
  null_marker_case_sensitive: false
  true_values: ["true", Y]
  false_values: ["false", N]
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
      format: upper
      replace_null_markers: true
      audit: true

  - source_column_name: loan_status
    target_column_name: LoanStatus
    expected_data_type: string
    parser:
      type: string
      format: title
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
    expected_data_type: decimal(10,2)
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

  - source_column_name: event_date
    target_column_name: EventDate
    expected_data_type: date
    parser:
      type: date
      audit: true

  - source_column_name: event_timestamp
    target_column_name: EventTimestamp
    expected_data_type: timestamp
    parser:
      type: timestamp
      audit: true

  - source_column_name: event_timestamp
    target_column_name: EventTimestampNtz
    expected_data_type: timestamp_ntz
    parser:
      type: timestamp_ntz
      audit: true

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
    expected_data_type: map<string,decimal(10,2)>
    parser:
      type: map
      value_parser: decimal
      on_value_error: drop
      on_parse_error: null
      audit: true
"""

BRONZE_SCHEMA = """
record_id string,
customer_name string,
loan_status string,
state string,
amount string,
quantity string,
event_date string,
event_timestamp string,
aliases string,
profile string,
attributes string
"""

BRONZE_ROWS = [
    (
        "good-1",
        "  Alice   Smith  ",
        "  ACTIVE   loan ",
        "Illinois",
        "12.345",
        "7",
        "09/30/2026 12:00:00 AM",
        "09/30/2026 12:00:00 AM",
        '[" ally ","ALLY",null]',
        '{"zip_code":"1234","raw_scores":[1,"bad",0]}',
        '{"principal":"10.125","bad":"x","empty":null}',
    ),
    (
        "handled-errors-1",
        "n/a",
        "charged OFF",
        "Mul",
        "not-a-decimal",
        "not-an-integer",
        "2026-08-28",
        "2026-08-28 13:45:00",
        "not-json",
        "not-json",
        "not-json",
    ),
]

ORIGINAL_ANSI = spark.conf.get("spark.sql.ansi.enabled")
ORIGINAL_TIME_POLICY = spark.conf.get("spark.sql.legacy.timeParserPolicy")
spark.conf.set("spark.sql.ansi.enabled", "true")
spark.conf.set("spark.sql.legacy.timeParserPolicy", "EXCEPTION")

print(f"Repository root: {REPO_ROOT}")
print(f"Reference configuration: {REFERENCE_CONFIG_PATH}")
print(f"Spark version: {spark.version}")
print(f"Spark Parser version: {__version__}")

# COMMAND ----------

_start("ST-001", "Compile the shipped reference and representative system configuration")

reference_config = parser.compile_path(REFERENCE_CONFIG_PATH)
assert {column.parser.parser_type.value for column in reference_config.columns} == set(
    parser.describe()
)

review = parser.review_yaml(SYSTEM_CONFIG_YAML)
assert review.is_valid, review.errors
assert not review.warnings, review.warnings
config = parser.compile_text(SYSTEM_CONFIG_YAML)
config_hash = parser.content_hash(config)
assert len(config_hash) == 64
assert config_hash == parser.content_hash(parser.compile_text(SYSTEM_CONFIG_YAML))

print("PASS: Current parser vocabulary and representative configuration compile deterministically.")

# COMMAND ----------

bronze_df = spark.createDataFrame(BRONZE_ROWS, schema=BRONZE_SCHEMA)
strict_parsing = parser.parse_dataframe(
    bronze_df,
    config,
    key_columns=["record_id"],
    column_prefix="system_parser",
)
strict_target_rows = _rows_by_key(strict_parsing.parsed_df, "RecordId")
strict_result_rows = _rows_by_key(strict_parsing.results_df, "record_id")
strict_audit_rows = _audit_by_key(
    strict_parsing.results_df,
    "record_id",
    "system_parser_parse_results",
)

_start("ST-002", "Materialize representative scalar values under strict Spark SQL settings")

good = strict_target_rows["good-1"]
assert good["CustomerName"] == "ALICE SMITH"
assert good["LoanStatus"] == "Active Loan"
assert good["StateCode"] == "IL"
assert good["Amount"] == Decimal("12.35")
assert good["Quantity"] == 7
assert good["EventDate"].isoformat() == "2026-09-30"
assert good["EventTimestamp"].isoformat(sep=" ") == "2026-09-30 00:00:00"
assert good["EventTimestampNtz"].isoformat(sep=" ") == "2026-09-30 00:00:00"

print("PASS: Native scalar expressions retain their expected Databricks values and types.")

# COMMAND ----------

_start("ST-003", "Parse recursive arrays, structs, and maps with exact nested audit paths")

assert good["Aliases"] == ["ALLY"]
assert good["Profile"]["postal_code"] == "01234"
assert good["Profile"]["scores"] == [1, -1, -1]
assert good["Attributes"] == {"principal": Decimal("10.13"), "empty": None}

good_audit = strict_audit_rows["good-1"]
assert good_audit["Profile"].nested_error_paths == ["$.scores[1]"]
assert good_audit["Profile"].nested_default_on_null_paths == [
    "$.scores[1]",
    "$.scores[2]",
]
assert good_audit["Profile"].nested_zero_invalidated_paths == ["$.scores[2]"]
assert good_audit["Attributes"].nested_error_paths == ["$['bad']"]
assert "nested_parse_errors_resolved" in good_audit["Profile"].actions_applied
assert "nested_default_on_null_applied" in good_audit["Profile"].actions_applied
assert "nested_zero_invalidated" in good_audit["Profile"].actions_applied

print("PASS: Recursive native expressions retain nested values and JSONPath-like audit evidence.")

# COMMAND ----------

_start("ST-004", "Apply handled error policies and fail only when fail-mode data materializes")

handled = strict_target_rows["handled-errors-1"]
assert handled["CustomerName"] is None
assert handled["LoanStatus"] == "Charged Off"
assert handled["StateCode"] == "Mul"
assert handled["Amount"] is None
assert handled["Quantity"] == 0
assert handled["Aliases"] == ["UNKNOWN"]
assert handled["Profile"] == {"postal_code": "00000", "scores": []}
assert handled["Attributes"] is None

handled_audit = strict_audit_rows["handled-errors-1"]
assert handled_audit["CustomerName"].actions_applied == ["null_marker_replaced"]
assert handled_audit["StateCode"].actions_applied == ["parse_error_preserved"]
assert handled_audit["Amount"].actions_applied == ["parse_error_to_null"]
assert handled_audit["Quantity"].actions_applied == ["parse_error_default_applied"]
assert handled_audit["Aliases"].actions_applied == ["parse_error_default_applied"]
assert handled_audit["Profile"].actions_applied == ["parse_error_default_applied"]
assert handled_audit["Attributes"].actions_applied == ["parse_error_to_null"]

fail_config = parser.compile_text(
    """
parser_config_id: system_fail_policy
parser_config_name: System Fail Policy
version: "1"
columns:
  - source_column_name: raw_value
    target_column_name: FailValue
    expected_data_type: integer
    parser: integer
"""
)
fail_parsing = parser.parse_dataframe(
    spark.createDataFrame([("not-an-integer",)], "raw_value string"),
    fail_config,
    key_columns=["raw_value"],
)
try:
    fail_parsing.parsed_df.select("FailValue").collect()
except Exception as exc:  # Spark wraps executor errors differently across runtimes.
    print(f"Expected fail-mode materialization error: {type(exc).__name__}")
else:
    raise AssertionError("on_parse_error: fail did not fail when FailValue materialized")

print("PASS: Null, default, preserve, nested handling, and lazy fail behavior are enforced.")

# COMMAND ----------

_start("ST-005", "Keep handled target and audit outcomes identical across ANSI modes")

spark.conf.set("spark.sql.ansi.enabled", "false")
try:
    permissive_parsing = parser.parse_dataframe(
        bronze_df,
        config,
        key_columns=["record_id"],
        column_prefix="system_parser",
    )
    permissive_target_rows = _rows_by_key(permissive_parsing.parsed_df, "RecordId")
    permissive_result_rows = _rows_by_key(permissive_parsing.results_df, "record_id")
    assert permissive_target_rows == strict_target_rows
    assert permissive_result_rows == strict_result_rows
finally:
    spark.conf.set("spark.sql.ansi.enabled", "true")

print("PASS: Handled parser results do not depend on permissive versus ANSI-enabled casting.")

# COMMAND ----------

_start("ST-006", "Expose the ordered target, audit, and configuration identity contracts")

assert strict_parsing.parsed_df.columns == [
    "RecordId",
    "CustomerName",
    "LoanStatus",
    "StateCode",
    "Amount",
    "Quantity",
    "EventDate",
    "EventTimestamp",
    "EventTimestampNtz",
    "Aliases",
    "Profile",
    "Attributes",
]
assert strict_parsing.key_columns == ("record_id",)
assert strict_parsing.result_columns == (
    "system_parser_parse_results",
    "system_parser_config",
    "system_parser_engine_version",
)
assert strict_parsing.results_df.columns == [
    "record_id",
    "system_parser_parse_results",
    "system_parser_config",
    "system_parser_engine_version",
]

identity = strict_parsing.results_df.first()
assert identity.system_parser_config.id == config.parser_config_id
assert identity.system_parser_config.version == config.version
assert identity.system_parser_config.content_hash == config_hash
assert identity.system_parser_engine_version == __version__

audit_fields = strict_parsing.results_df.schema[
    "system_parser_parse_results"
].dataType.elementType.fieldNames()
assert audit_fields == [
    "source_column_name",
    "target_column_name",
    "parser_type",
    "expected_data_type",
    "original_value",
    "parsed_value",
    "changed",
    "effective",
    "actions_applied",
    "options",
    "error",
    "nested_error_paths",
    "nested_default_on_null_paths",
    "nested_zero_invalidated_paths",
]

print("PASS: Ordered DataFrame and row-level audit metadata match the public contract.")

# COMMAND ----------

_start("ST-007", "Explicitly represent a missing source as a warning and auditable null")

missing_config = parser.compile_text(
    """
parser_config_id: system_missing_source
parser_config_name: System Missing Source
version: "1"
columns:
  - source_column_name: source_not_delivered
    target_column_name: MissingValue
    expected_data_type: string
    parser:
      type: string
      audit: true
"""
)
missing_parsing = parser.parse_dataframe(
    spark.createDataFrame([("row-1",)], "row_id string"),
    missing_config,
    key_columns=["row_id"],
    on_missing_source="warn",
    column_prefix="system_missing",
)
assert missing_parsing.parsed_df.first().MissingValue is None
assert missing_parsing.warnings and "source_not_delivered" in missing_parsing.warnings[0]
missing_audit = missing_parsing.results_df.first().system_missing_parse_results[0]
assert missing_audit.effective is False
assert missing_audit.actions_applied == ["source_column_missing"]
assert missing_audit.error == "Source column is missing."

print("PASS: Explicitly recoverable drift stays visible in warnings and row-level audit output.")

# COMMAND ----------

_start("ST-008", "Reject unsafe input schemas before constructing parser expressions")

schema_config = parser.compile_text(
    """
parser_config_id: system_schema_guards
parser_config_name: System Schema Guards
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: string
    parser: string
"""
)

_expect_raises(
    SchemaValidationError,
    lambda: parser.parse_dataframe(
        spark.range(1).selectExpr("id AS value"),
        schema_config,
        key_columns=["value"],
    ),
    contains="must have Spark string type",
)
_expect_raises(
    SchemaValidationError,
    lambda: parser.parse_dataframe(
        spark.createDataFrame(
            [("x", "occupied")],
            "value string, system_guard_config string",
        ),
        schema_config,
        key_columns=["value"],
        column_prefix="system_guard",
    ),
    contains="reserved parser output columns",
)
_expect_raises(
    TypeError,
    lambda: parser.parse_dataframe(
        spark.sql("SELECT 'x' AS value, 1 AS duplicate, 2 AS duplicate"),
        schema_config,
    ),
    contains="key_columns",
)

print("PASS: Non-string sources, reserved outputs, and omitted explicit keys fail safely.")

# COMMAND ----------

spark.conf.set("spark.sql.ansi.enabled", ORIGINAL_ANSI)
spark.conf.set("spark.sql.legacy.timeParserPolicy", ORIGINAL_TIME_POLICY)

print()
print("=" * 80)
print("PASS: All 8 current-contract Spark Parser system tests completed.")
print("=" * 80)
