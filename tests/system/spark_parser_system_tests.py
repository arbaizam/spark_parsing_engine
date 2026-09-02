# Databricks notebook source
# ruff: noqa: BLE001, E402, F821, I001
# MAGIC %md
# MAGIC # Spark Parser System Tests
# MAGIC
# MAGIC These tests cover behavior that requires a real Databricks Spark session: native scalar
# MAGIC parsing and display profiles, lazy fail-mode materialization, ANSI-mode parity, generated
# MAGIC output schemas, row-level audit data, and input-schema validation. Compiler-only
# MAGIC permutations remain in the pytest suite.
# MAGIC
# MAGIC Run this notebook from a Databricks Git checkout of the repository. It reads source directly
# MAGIC from the checkout, requires no built wheel or release artifact, and writes no tables.

# COMMAND ----------

import os
import re
import sys
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from py4j.protocol import Py4JJavaError
from pyspark.errors import PySparkException
from pyspark.sql import functions as F

root = next(
    (
        path
        for path in [Path.cwd(), *Path.cwd().parents]
        if (path / "pyproject.toml").is_file() and (path / "src" / "spark_parser").is_dir()
    ),
    None,
)
if root is not None:
    src_path = os.path.normpath((root / "src").resolve())
    while src_path in sys.path:
        sys.path.remove(src_path)
    print(f"Adding source checkout to sys.path: {src_path}")
    sys.path.insert(0, src_path)

    # A Databricks Python process can outlive a notebook run. Remove any previously imported
    # spark_parser modules so sys.path precedence is sufficient even after a wheel-backed import.
    for module_name in tuple(sys.modules):
        if module_name == "spark_parser" or module_name.startswith("spark_parser."):
            del sys.modules[module_name]

import spark_parser as spark_parser_package
from spark_parser import SchemaValidationError, __version__, parser


SYSTEM_TEST_IDS = tuple(f"ST-{number:03d}" for number in range(1, 9))
PASSED_TEST_IDS = []
ACTIVE_TEST_ID = None


def _start(test_id, name):
    """Print one visible system-test boundary."""
    global ACTIVE_TEST_ID
    assert len(PASSED_TEST_IDS) < len(SYSTEM_TEST_IDS), (
        "All registered system tests have already passed; rerun the notebook from the setup cell."
    )
    expected_test_id = SYSTEM_TEST_IDS[len(PASSED_TEST_IDS)]
    assert test_id == expected_test_id, (
        f"System tests must run in order; expected {expected_test_id}, received {test_id}."
    )
    ACTIVE_TEST_ID = test_id
    print()
    print(f"{test_id}: {name}")
    print("-" * 80)


def _pass(test_id, message):
    """Record one successful system-test boundary before printing its PASS marker."""
    global ACTIVE_TEST_ID
    assert ACTIVE_TEST_ID == test_id, (
        f"Cannot pass {test_id}; the active system test is {ACTIVE_TEST_ID!r}."
    )
    assert test_id not in PASSED_TEST_IDS, f"System test {test_id} was already recorded as passed."
    PASSED_TEST_IDS.append(test_id)
    ACTIVE_TEST_ID = None
    print(f"PASS: {message}")


def _expect_raises(exception_type, operation, *, contains=None):
    """Run one operation and return its expected exception."""
    try:
        operation()
    except exception_type as exc:
        if contains is not None:
            assert contains in str(exc), (
                f"Expected exception containing {contains!r}, found {exc!r}."
            )
        return exc
    expected_names = ", ".join(
        candidate.__name__
        for candidate in (
            exception_type if isinstance(exception_type, tuple) else (exception_type,)
        )
    )
    raise AssertionError(f"Expected one of [{expected_names}] to be raised.")


@contextmanager
def _spark_conf_scope(settings):
    """Apply Spark SQL settings for one test and restore them on every exit path."""
    previous_values = []
    try:
        for key, value in settings.items():
            previous_values.append((key, spark.conf.get(key)))
            spark.conf.set(key, value)
        yield
    finally:
        for key, previous_value in reversed(previous_values):
            spark.conf.set(key, previous_value)


def _spark_major_minor(version):
    """Return a comparable Spark major/minor pair from a runtime version string."""
    match = re.match(r"^(\d+)\.(\d+)", version)
    assert match is not None, f"Could not parse the Spark runtime version: {version!r}."
    return int(match.group(1)), int(match.group(2))


def _rows_by_key(df, key):
    """Collect the fixed system-test fixture into a key-indexed mapping."""
    return {row[key]: row.asDict(recursive=True) for row in df.collect()}


def _audit_by_key(df, key, results_column):
    """Index each row's audit structs by target column name."""
    return {
        row[key]: {result.target_column_name: result for result in row[results_column]}
        for row in df.collect()
    }


assert root is not None, (
    "Could not locate the spark_parser repository root. Run this notebook from a Databricks "
    "Git checkout containing pyproject.toml and src/spark_parser."
)
REPO_ROOT = root
REFERENCE_CONFIG_PATH = REPO_ROOT / "examples" / "all_parsers.yaml"
EXPECTED_PACKAGE_DIRECTORY = (REPO_ROOT / "src" / "spark_parser").resolve()
STRICT_SQL_SETTINGS = {
    "spark.sql.ansi.enabled": "true",
    "spark.sql.legacy.timeParserPolicy": "EXCEPTION",
}

SYSTEM_CONFIG_YAML = """
parser_config_id: databricks_system_tests
parser_config_name: Databricks System Tests
version: "1"
description: Representative real-Spark system coverage for scalar parsing.
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

  - source_column_name: business_label
    target_column_name: BusinessLabel
    expected_data_type: string
    parser:
      type: string
      format: title_business_v1
      audit: true

  - source_column_name: rate_index
    target_column_name: RateIndex
    expected_data_type: string
    parser:
      type: string
      format: interest_rate_index_v1
      on_parse_error: preserve
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

"""

BRONZE_SCHEMA = """
record_id string,
customer_name string,
loan_status string,
business_label string,
rate_index string,
state string,
amount string,
quantity string,
event_date string,
event_timestamp string
"""

BRONZE_ROWS = [
    (
        "good-1",
        "  Alice   Smith  ",
        "  ACTIVE   loan ",
        "fhlb semi-annual 2 yrs 12-month advance",
        "SOFR Term - 12M",
        "Illinois",
        "12.345",
        "7",
        "09/30/2026 12:00:00 AM",
        "09/30/2026 12:00:00 AM",
    ),
    (
        "handled-errors-1",
        "n/a",
        "charged OFF",
        "biannually ust rcf cmt",
        "NAP",
        "Mul",
        "not-a-decimal",
        "not-an-integer",
        "2026-08-28",
        "2026-08-28 13:45:00",
    ),
]

_start("ST-001", "Compile the shipped reference and representative system configuration")

package_file = Path(spark_parser_package.__file__).resolve()
assert package_file.is_relative_to(EXPECTED_PACKAGE_DIRECTORY), (
    "System tests must import spark_parser from the repository checkout; "
    f"loaded {package_file}, expected a module under {EXPECTED_PACKAGE_DIRECTORY}."
)
assert _spark_major_minor(spark.version) >= (3, 5), (
    f"Spark Parser requires Spark 3.5 or newer; found Spark {spark.version}."
)
assert REFERENCE_CONFIG_PATH.is_file(), (
    f"Parser reference configuration does not exist: {REFERENCE_CONFIG_PATH}"
)

print(f"Repository root: {REPO_ROOT}")
print(f"Reference configuration: {REFERENCE_CONFIG_PATH}")
print(f"Imported package: {package_file}")
print(f"Spark version: {spark.version}")
print(f"Spark Parser version: {__version__}")

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

_pass(
    "ST-001",
    "Current parser vocabulary and representative configuration compile deterministically.",
)

# COMMAND ----------

_start("ST-002", "Materialize representative scalar values under strict Spark SQL settings")

with _spark_conf_scope(STRICT_SQL_SETTINGS):
    bronze_df = spark.createDataFrame(BRONZE_ROWS, schema=BRONZE_SCHEMA)
    scalar_parsing = parser.parse_dataframe(
        bronze_df,
        config,
        key_columns=["record_id"],
        column_prefix="system_parser",
    )
    scalar_schema = {
        field.name: field.dataType.simpleString()
        for field in scalar_parsing.parsed_df.schema.fields
        if field.name
        in {
            "CustomerName",
            "LoanStatus",
            "BusinessLabel",
            "RateIndex",
            "StateCode",
            "Amount",
            "Quantity",
            "EventDate",
            "EventTimestamp",
            "EventTimestampNtz",
        }
    }
    scalar_rows = _rows_by_key(
        scalar_parsing.parsed_df.select(
            "RecordId",
            "CustomerName",
            "LoanStatus",
            "BusinessLabel",
            "RateIndex",
            "StateCode",
            "Amount",
            "Quantity",
            F.col("EventDate").cast("string").alias("EventDateText"),
            F.col("EventTimestamp").cast("string").alias("EventTimestampText"),
            F.col("EventTimestampNtz").cast("string").alias("EventTimestampNtzText"),
        ),
        "RecordId",
    )

assert scalar_schema == {
    "CustomerName": "string",
    "LoanStatus": "string",
    "BusinessLabel": "string",
    "RateIndex": "string",
    "StateCode": "string",
    "Amount": "decimal(10,2)",
    "Quantity": "int",
    "EventDate": "date",
    "EventTimestamp": "timestamp",
    "EventTimestampNtz": "timestamp_ntz",
}
good = scalar_rows["good-1"]
assert good["CustomerName"] == "ALICE SMITH"
assert good["LoanStatus"] == "Active Loan"
assert good["BusinessLabel"] == "FHLB Semi-Annual 2-Years 12-Month Advance"
assert good["RateIndex"] == "SOFR Term 12-Month"
assert good["StateCode"] == "IL"
assert good["Amount"] == Decimal("12.35")
assert good["Quantity"] == 7
assert good["EventDateText"] == "2026-09-30"
assert good["EventTimestampText"] == "2026-09-30 00:00:00"
assert good["EventTimestampNtzText"] == "2026-09-30 00:00:00"

_pass("ST-002", "Native scalar expressions retain their expected Databricks values and types.")

# COMMAND ----------

_start("ST-003", "Apply versioned business display and property-type profiles")

with _spark_conf_scope(STRICT_SQL_SETTINGS):
    profile_parsing = parser.parse_dataframe(
        bronze_df,
        config,
        key_columns=["record_id"],
        column_prefix="system_parser",
    )
    profile_rows = _rows_by_key(
        profile_parsing.parsed_df.select("RecordId", "BusinessLabel", "RateIndex"),
        "RecordId",
    )
    profile_audit_rows = _audit_by_key(
        profile_parsing.results_df,
        "RecordId",
        "system_parser_parse_results",
    )
    property_config = parser.compile_text(
        """
parser_config_id: system_property_types
parser_config_name: System Property Types
version: "1"
columns:
  - source_column_name: property_type
    target_column_name: PropertyType
    expected_data_type: string
    parser:
      type: string
      format: property_type_v1
      on_parse_error: preserve
      audit: true
"""
    )
    property_parsing = parser.parse_dataframe(
        spark.createDataFrame(
            [
                ("condo", "Condominimum"),
                ("four", "4 Family Home"),
                ("lihtc", "Multifamily-LIHTC"),
                ("mixed", "Warehouse-Mixed use"),
                ("compound", "Mixed Use-MULTIFAMILY-Over 20"),
                ("unknown", "Storage"),
            ],
            "row_id string, property_type string",
        ),
        property_config,
        key_columns=["row_id"],
        column_prefix="system_property",
    )
    property_rows = _rows_by_key(property_parsing.parsed_df, "PropertyType")
    property_audit_rows = _audit_by_key(
        property_parsing.results_df,
        "row_id",
        "system_property_parse_results",
    )

assert profile_rows["good-1"]["BusinessLabel"] == (
    "FHLB Semi-Annual 2-Years 12-Month Advance"
)
assert profile_rows["good-1"]["RateIndex"] == "SOFR Term 12-Month"
assert profile_audit_rows["good-1"]["BusinessLabel"].changed is True
assert profile_audit_rows["good-1"]["BusinessLabel"].actions_applied == []
assert profile_audit_rows["good-1"]["RateIndex"].changed is True
assert profile_audit_rows["good-1"]["RateIndex"].actions_applied == []
assert profile_rows["handled-errors-1"]["BusinessLabel"] == "Bi-Annual UST RCF CMT"
assert profile_rows["handled-errors-1"]["RateIndex"] == "NAP"
assert profile_audit_rows["handled-errors-1"]["RateIndex"].actions_applied == [
    "parse_error_preserved"
]
assert set(property_rows) == {
    "Condominium",
    "Four-Unit",
    "Multifamily - LIHTC",
    "Mixed Use - Warehouse",
    "Mixed Use - Multifamily - Over 20",
    "Storage",
}
assert property_audit_rows["unknown"]["PropertyType"].actions_applied == [
    "parse_error_preserved"
]

_pass(
    "ST-003",
    "Business titles, approved indexes, property aliases, retained mixed-use descriptors, and "
    "preserved unknowns are stable.",
)

# COMMAND ----------

_start("ST-004", "Apply handled error policies and fail only when fail-mode data materializes")

with _spark_conf_scope(STRICT_SQL_SETTINGS):
    handled_parsing = parser.parse_dataframe(
        bronze_df,
        config,
        key_columns=["record_id"],
        column_prefix="system_parser",
    )
    handled = (
        handled_parsing.parsed_df.where(F.col("RecordId") == "handled-errors-1")
        .first()
        .asDict(recursive=True)
    )
    handled_result = handled_parsing.results_df.where(
        F.col("RecordId") == "handled-errors-1"
    ).first()

assert handled["CustomerName"] is None
assert handled["LoanStatus"] == "Charged Off"
assert handled["BusinessLabel"] == "Bi-Annual UST RCF CMT"
assert handled["RateIndex"] == "NAP"
assert handled["StateCode"] == "Mul"
assert handled["Amount"] is None
assert handled["Quantity"] == 0

handled_audit = {
    result.target_column_name: result for result in handled_result.system_parser_parse_results
}
assert handled_audit["CustomerName"].actions_applied == ["null_marker_replaced"]
assert handled_audit["RateIndex"].actions_applied == ["parse_error_preserved"]
assert handled_audit["StateCode"].actions_applied == ["parse_error_preserved"]
assert handled_audit["Amount"].actions_applied == ["parse_error_to_null"]
assert handled_audit["Quantity"].actions_applied == ["parse_error_default_applied"]

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
with _spark_conf_scope(STRICT_SQL_SETTINGS):
    fail_parsing = parser.parse_dataframe(
        spark.createDataFrame([("not-an-integer",)], "raw_value string"),
        fail_config,
        key_columns=["raw_value"],
    )
    fail_exception = _expect_raises(
        (Py4JJavaError, PySparkException),
        lambda: fail_parsing.parsed_df.select("FailValue").collect(),
        contains=(
            "Spark Parser could not parse source 'raw_value' into target column 'FailValue' "
            "as integer"
        ),
    )
print(f"Expected fail-mode materialization error: {type(fail_exception).__name__}")

_pass("ST-004", "Null, default, preserve, and lazy fail behavior are enforced.")

# COMMAND ----------

_start("ST-005", "Keep handled target and audit outcomes identical across ANSI modes")

with _spark_conf_scope(STRICT_SQL_SETTINGS):
    strict_mode_parsing = parser.parse_dataframe(
        bronze_df,
        config,
        key_columns=["record_id"],
        column_prefix="system_parser",
    )
    strict_target_rows = _rows_by_key(strict_mode_parsing.parsed_df, "RecordId")
    strict_result_rows = _rows_by_key(strict_mode_parsing.results_df, "RecordId")

with _spark_conf_scope(
    {
        "spark.sql.ansi.enabled": "false",
        "spark.sql.legacy.timeParserPolicy": "EXCEPTION",
    }
):
    permissive_parsing = parser.parse_dataframe(
        bronze_df,
        config,
        key_columns=["record_id"],
        column_prefix="system_parser",
    )
    permissive_target_rows = _rows_by_key(permissive_parsing.parsed_df, "RecordId")
    permissive_result_rows = _rows_by_key(permissive_parsing.results_df, "RecordId")
    assert permissive_target_rows == strict_target_rows
    assert permissive_result_rows == strict_result_rows

_pass(
    "ST-005",
    "Handled parser results do not depend on permissive versus ANSI-enabled casting.",
)

# COMMAND ----------

_start("ST-006", "Expose the ordered target, audit, and configuration identity contracts")

with _spark_conf_scope(STRICT_SQL_SETTINGS):
    contract_parsing = parser.parse_dataframe(
        bronze_df,
        config,
        key_columns=["record_id"],
        column_prefix="system_parser",
    )
    identity = contract_parsing.results_df.first()

assert contract_parsing.parsed_df.columns == [
    "RecordId",
    "CustomerName",
    "LoanStatus",
    "BusinessLabel",
    "RateIndex",
    "StateCode",
    "Amount",
    "Quantity",
    "EventDate",
    "EventTimestamp",
    "EventTimestampNtz",
]
assert contract_parsing.key_columns == ("RecordId",)
assert contract_parsing.result_columns == (
    "system_parser_parse_results",
    "system_parser_config",
    "system_parser_engine_version",
)
assert contract_parsing.results_df.columns == [
    "RecordId",
    "system_parser_parse_results",
    "system_parser_config",
    "system_parser_engine_version",
]
assert identity.RecordId in {row[0] for row in BRONZE_ROWS}

assert identity.system_parser_config.id == config.parser_config_id
assert identity.system_parser_config.version == config.version
assert identity.system_parser_config.content_hash == config_hash
assert identity.system_parser_engine_version == __version__

audit_fields = contract_parsing.results_df.schema[
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
]

_pass("ST-006", "Ordered DataFrame and row-level audit metadata match the public contract.")

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
missing_field = missing_parsing.parsed_df.schema["MissingValue"]
assert missing_parsing.parsed_df.schema.simpleString() == "struct<MissingValue:string>"
assert missing_field.dataType.simpleString() == "string"
assert missing_field.nullable is True
assert missing_parsing.parsed_df.first().MissingValue is None
assert missing_parsing.warnings and "source_not_delivered" in missing_parsing.warnings[0]
missing_audit = missing_parsing.results_df.first().system_missing_parse_results[0]
assert missing_audit.effective is False
assert missing_audit.actions_applied == ["source_column_missing"]
assert missing_audit.error == "Source column is missing."

_pass(
    "ST-007",
    "Explicitly recoverable drift stays visible in warnings and row-level audit output.",
)

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

_pass("ST-008", "Non-string sources, reserved outputs, and omitted explicit keys fail safely.")

# COMMAND ----------

assert tuple(PASSED_TEST_IDS) == SYSTEM_TEST_IDS, (
    "The final success marker requires every system test to pass in order; "
    f"recorded {PASSED_TEST_IDS!r}."
)
assert ACTIVE_TEST_ID is None, f"System test {ACTIVE_TEST_ID} did not record a PASS result."

print()
print("=" * 80)
print("PASS: All 8 current-contract Spark Parser system tests completed.")
print("=" * 80)
