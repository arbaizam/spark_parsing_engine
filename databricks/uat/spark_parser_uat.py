# Databricks notebook source
# ruff: noqa: BLE001, E402, F821
# MAGIC %md
# MAGIC # Spark Parser Databricks UAT
# MAGIC
# MAGIC Installs one wheel, validates representative bronze-to-silver behavior under ANSI mode,
# MAGIC round-trips silver and parser-audit data through Delta, and publishes a rules-engine
# MAGIC handoff contract. See the adjacent `README.md` for parameters and pass criteria.

# COMMAND ----------

# Widgets make the same notebook usable interactively and as a parameterized Databricks job task.
# Keep artifact and destination values outside source control; UAT operators supply them per run.
dbutils.widgets.text("wheel_path", "")
dbutils.widgets.text("expected_version", "0.4.0")
dbutils.widgets.text("expected_wheel_sha256", "")
dbutils.widgets.text("config_path", "")
dbutils.widgets.text("target_catalog", "")
dbutils.widgets.text("target_schema", "")
dbutils.widgets.text("table_prefix", "spark_parser_uat")
dbutils.widgets.text("run_id", "")

# Validate the wheel path before invoking %pip. A missing placeholder should fail with a clear
# message rather than being interpreted as a package name and searched on a public index.
wheel_path = dbutils.widgets.get("wheel_path").strip()
if not wheel_path.endswith(".whl"):
    raise ValueError("wheel_path must be an absolute /Volumes or /Workspace .whl path")
if not wheel_path.startswith(("/Volumes/", "/Workspace/")):
    raise ValueError("wheel_path must start with /Volumes/ or /Workspace/")

# COMMAND ----------

# ``--no-deps`` protects runtime-owned PySpark libraries from replacement. The UAT compute must
# already satisfy the package dependencies documented in README.md.
# MAGIC %pip install --no-deps --force-reinstall "$wheel_path"

# COMMAND ----------

# A clean interpreter prevents a previously imported package from masking the installed wheel.
dbutils.library.restartPython()

# COMMAND ----------

import hashlib
import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import distribution
from pathlib import Path

from pyspark.sql import functions as F

import spark_parser
from spark_parser import parser


def required_widget(name: str) -> str:
    """Read one required Databricks text widget and reject blank or whitespace-only input."""
    value = dbutils.widgets.get(name).strip()
    if not value:
        raise ValueError(f"Notebook parameter {name!r} is required")
    return value


def identifier(value: str, *, parameter: str) -> str:
    """Restrict catalog/schema/table fragments to simple, safely quoted SQL identifiers."""
    # This conservative grammar avoids dynamic SQL surprises and guarantees generated table/view
    # names need no interpretation beyond ordinary backtick quoting.
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ValueError(
            f"{parameter} must start with a letter and contain only letters, numbers, or underscores"
        )
    return value


wheel_path = required_widget("wheel_path")
expected_version = required_widget("expected_version")
expected_wheel_sha256 = dbutils.widgets.get("expected_wheel_sha256").strip().lower()
config_path = required_widget("config_path")
if expected_wheel_sha256 and not re.fullmatch(r"[0-9a-f]{64}", expected_wheel_sha256):
    raise ValueError("expected_wheel_sha256 must contain exactly 64 lowercase hexadecimal digits")
if not config_path.startswith(("/Volumes/", "/Workspace/")):
    raise ValueError("config_path must start with /Volumes/ or /Workspace/")
target_catalog = identifier(required_widget("target_catalog"), parameter="target_catalog")
target_schema = identifier(required_widget("target_schema"), parameter="target_schema")
table_prefix = identifier(required_widget("table_prefix"), parameter="table_prefix")
# Prefixing temporarily with ``r_`` lets a timestamp-style run ID pass the identifier validator.
# The generated object name already has a letter-prefixed table_prefix, so the helper prefix is
# removed after validation.
run_id = dbutils.widgets.get("run_id").strip() or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
run_id = identifier(f"r_{run_id}", parameter="run_id")[2:]

wheel_file = Path(wheel_path)
config_file = Path(config_path)
if not wheel_file.is_file():
    raise FileNotFoundError(f"Wheel does not exist: {wheel_path}")
if not config_file.is_file():
    raise FileNotFoundError(f"UAT config does not exist: {config_path}")

# COMMAND ----------

# Verify both distribution metadata and imported module metadata. This catches a stale module on
# sys.path even when pip reports that the intended wheel was installed.
installed_distribution = distribution("spark-parser")
installed_version = installed_distribution.version

# Hash the staged wheel itself, not installed files. The optional approved digest ties this run to
# the exact artifact reviewed by release engineering.
wheel_sha256 = hashlib.sha256(wheel_file.read_bytes()).hexdigest()

assert installed_version == expected_version, (
    f"Installed spark-parser {installed_version}, expected {expected_version}"
)
assert spark_parser.__version__ == expected_version
if expected_wheel_sha256:
    assert wheel_sha256 == expected_wheel_sha256, (
        f"Wheel SHA-256 {wheel_sha256} did not match {expected_wheel_sha256}"
    )

print(
    json.dumps(
        {
            "distribution": installed_distribution.metadata["Name"],
            "installed_version": installed_version,
            "module_path": spark_parser.__file__,
            "wheel_path": wheel_path,
            "wheel_sha256": wheel_sha256,
        },
        indent=2,
    )
)

# COMMAND ----------

# ANSI mode is an explicit UAT condition because permissive and ANSI-enabled runtimes can differ in
# casting behavior. Parser error policies must remain in control under the stricter setting.
spark.conf.set("spark.sql.ansi.enabled", "true")
assert spark.conf.get("spark.sql.ansi.enabled").lower() == "true"

# Review before compilation so the notebook prints the same resolved contract a human approver sees.
# The warning assertion intentionally fails UAT when ownership/audit metadata is incomplete.
config_text = config_file.read_text(encoding="utf-8")
review = parser.review_yaml(config_text)
assert review.is_valid, review.errors
assert not review.warnings, review.warnings
config = parser.compile_text(config_text)
config_hash = parser.content_hash(config)

print(review.to_markdown())

# COMMAND ----------

# All configured bronze fields are strings by contract. The explicit schema prevents Python's local
# inference from hiding a Databricks-side binding problem.
bronze_schema = """
record_id string,
customer_name string,
amount string,
quantity string,
event_date string,
aliases string,
profile string,
attributes string
"""
# Row 1 exercises successful scalar parsing plus handled child failures in valid JSON containers.
# Row 2 exercises top-level null/default policies without contaminating the successful output row.
bronze_rows = [
    (
        "good-1",
        "  Alice   Smith  ",
        "12.345",
        "7",
        "2026-08-27",
        '[" ally ","ALLY",null]',
        '{"zip_code":"1234","raw_scores":[1,"bad",3]}',
        '{"principal":"10.125","bad":"x","empty":null}',
    ),
    (
        "handled-errors-1",
        "n/a",
        "not-a-decimal",
        "not-an-integer",
        "2026-08-28",
        "not-json",
        "not-json",
        "not-json",
    ),
]
bronze_df = spark.createDataFrame(bronze_rows, schema=bronze_schema)

# Persist the shared lazy plan because both silver and audit projections are materialized below.
# Without this call, Spark could evaluate every parser expression twice.
parsing = parser.parse_dataframe(bronze_df, config, key_columns=["record_id"]).persist()
silver_df = parsing.parsed_df
audit_df = parsing.results_df.withColumnRenamed("record_id", "RecordId")

# Collecting is appropriate here only because this is a fixed two-row UAT fixture. Production loads
# must validate aggregations or write distributed DataFrames instead of collecting arbitrary data.
silver_rows = {
    row.RecordId: row.asDict(recursive=True) for row in silver_df.collect()
}
audit_rows = {
    row.RecordId: {
        result.silver_column_name: result
        for result in row.spark_parser_parse_results
    }
    for row in audit_df.collect()
}

# COMMAND ----------

# Assert exact typed Python values after Spark materialization. These checks cover the representative
# integration contract; exhaustive edge cases remain in the repository's runtime test suite.
good = silver_rows["good-1"]
assert good["CustomerName"] == "ALICE SMITH"
assert good["Amount"] == Decimal("12.35")
assert good["Quantity"] == 7
assert good["EventDate"].isoformat() == "2026-08-27"
assert good["Aliases"] == ["ALLY"]
assert good["Profile"]["postal_code"] == "01234"
assert good["Profile"]["scores"] == [1, None, 3]
assert good["Attributes"] == {"principal": Decimal("10.13"), "empty": None}

handled = silver_rows["handled-errors-1"]
assert handled["CustomerName"] is None
assert handled["Amount"] is None
assert handled["Quantity"] == 0
assert handled["EventDate"].isoformat() == "2026-08-28"
assert handled["Aliases"] == ["UNKNOWN"]
assert handled["Profile"] == {"postal_code": "00000", "scores": []}
assert handled["Attributes"] is None

# Audit assertions prove handled failures remain explainable even when a bad child is dropped or
# replaced in the final silver value.
good_audit = audit_rows["good-1"]
handled_audit = audit_rows["handled-errors-1"]
assert good_audit["Profile"].nested_error_paths == ["$.scores[1]"]
assert good_audit["Attributes"].nested_error_paths == ["$['bad']"]
assert "nested_parse_errors_resolved" in good_audit["Profile"].actions_applied
assert "nested_parse_errors_resolved" in good_audit["Attributes"].actions_applied
assert handled_audit["CustomerName"].actions_applied == ["null_marker_replaced"]
assert handled_audit["Amount"].actions_applied == ["parse_error_to_null"]
assert handled_audit["Quantity"].actions_applied == ["parse_error_default_applied"]
assert handled_audit["Aliases"].actions_applied == ["parse_error_default_applied"]
assert handled_audit["Profile"].actions_applied == ["parse_error_default_applied"]
assert handled_audit["Attributes"].actions_applied == ["parse_error_to_null"]

display(silver_df.orderBy("RecordId"))
display(audit_df.orderBy("RecordId"))

# COMMAND ----------

# Validate fail mode separately so the representative output remains writable. Selecting the actual
# parsed column prevents projection pruning from turning a lazy raise_error expression into a false
# success (a bare count may not need to evaluate the value).
fail_config = parser.compile_text(
    """
parser_config_id: databricks_fail_policy_uat
parser_config_name: Databricks Fail Policy UAT
version: "1"
columns:
  - source_column_name: raw_value
    silver_column_name: FailValue
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
except Exception as exc:  # Spark wraps executor failures differently across supported runtimes.
    # The exception class/message varies by Databricks Runtime. UAT requires the materialized action
    # to fail, while focused package tests validate the parser-generated message content.
    print(f"Expected fail-policy materialization error: {type(exc).__name__}")
else:
    raise AssertionError("on_parse_error: fail did not fail when the parsed value was materialized")

# COMMAND ----------

# The schema must already exist; this notebook intentionally does not create or alter it.
spark.sql(f"DESCRIBE SCHEMA `{target_catalog}`.`{target_schema}`").collect()

# Unique run-scoped names plus errorifexists provide a fail-closed write policy. The notebook never
# overwrites evidence from another attempt and never creates or drops the containing schema.
silver_table = f"{target_catalog}.{target_schema}.{table_prefix}_{run_id}_silver"
audit_table = f"{target_catalog}.{target_schema}.{table_prefix}_{run_id}_audit"

silver_df.write.format("delta").mode("errorifexists").saveAsTable(silver_table)
audit_df.write.format("delta").mode("errorifexists").saveAsTable(audit_table)

silver_delta_df = spark.table(silver_table)
audit_delta_df = spark.table(audit_table)


def ordered_rows(df, key: str) -> list[dict]:
    """Collect the fixed UAT fixture in deterministic key order for exact round-trip comparison."""
    return [row.asDict(recursive=True) for row in df.orderBy(key).collect()]


assert silver_delta_df.schema.json() == silver_df.schema.json()
assert audit_delta_df.schema.json() == audit_df.schema.json()
assert ordered_rows(silver_delta_df, "RecordId") == ordered_rows(silver_df, "RecordId")
assert ordered_rows(audit_delta_df, "RecordId") == ordered_rows(audit_df, "RecordId")

# COMMAND ----------

# Rules-engine handoff: typed silver rows stay separate from flattened parser diagnostics. Rules
# consume domain values from the first DataFrame and may use the second for parser lineage, handled
# errors, and routing decisions without forcing audit structs into the business schema.
rules_engine_input_df = silver_delta_df
rules_engine_parser_results_df = audit_delta_df.select(
    "RecordId",
    F.col("spark_parser_config.id").alias("parser_config_id"),
    F.col("spark_parser_config.version").alias("parser_config_version"),
    F.col("spark_parser_config.content_hash").alias("parser_config_content_hash"),
    "spark_parser_engine_version",
    F.explode("spark_parser_parse_results").alias("parser_result"),
).select(
    "RecordId",
    "parser_config_id",
    "parser_config_version",
    "parser_config_content_hash",
    "spark_parser_engine_version",
    "parser_result.*",
)

# The handoff contract requires a non-null unique row key and a single parser configuration hash.
# Cardinality also proves every configured audited column survived the explode operation.
assert rules_engine_input_df.filter(F.col("RecordId").isNull()).count() == 0
assert rules_engine_input_df.select("RecordId").distinct().count() == rules_engine_input_df.count()
assert rules_engine_parser_results_df.filter(F.col("RecordId").isNull()).count() == 0
assert rules_engine_parser_results_df.select("parser_config_content_hash").distinct().first()[0] == (
    config_hash
)
assert rules_engine_parser_results_df.count() == (
    rules_engine_input_df.count() * review.summary["audited_column_count"]
)
# Field-order validation protects consumers that write the flattened diagnostics to a declared
# schema or select by position in external orchestration code.
assert rules_engine_parser_results_df.schema.fieldNames() == [
    "RecordId",
    "parser_config_id",
    "parser_config_version",
    "parser_config_content_hash",
    "spark_parser_engine_version",
    "source_column_name",
    "silver_column_name",
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
]

rules_input_view = f"{table_prefix}_{run_id}_rules_input"
parser_results_view = f"{table_prefix}_{run_id}_parser_results"
# Temporary views make interactive handoff convenient in the current session. A later job task
# should use the reported Delta tables because temporary views do not cross task/session boundaries.
rules_engine_input_df.createOrReplaceTempView(rules_input_view)
rules_engine_parser_results_df.createOrReplaceTempView(parser_results_view)

# COMMAND ----------

# A machine-readable exit value is easier for Databricks Workflows and release evidence tooling to
# consume than notebook display output. Reaching this cell means every assertion above passed.
summary = {
    "status": "PASS",
    "ansi_enabled": spark.conf.get("spark.sql.ansi.enabled"),
    "spark_parser_version": installed_version,
    "wheel_sha256": wheel_sha256,
    "parser_config_id": config.parser_config_id,
    "parser_config_version": config.version,
    "parser_config_content_hash": config_hash,
    "bronze_row_count": bronze_df.count(),
    "silver_table": silver_table,
    "audit_table": audit_table,
    "rules_engine_input_view": rules_input_view,
    "rules_engine_parser_results_view": parser_results_view,
}
print(json.dumps(summary, indent=2))

parsing.unpersist(blocking=False)
dbutils.notebook.exit(json.dumps(summary))
