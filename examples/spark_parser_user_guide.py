# Databricks notebook source
# ruff: noqa: BLE001, E402, F821
# MAGIC %md
# MAGIC # Spark Parser 0.5 User Guide
# MAGIC
# MAGIC Spark Parser converts top-level bronze string columns into typed scalar Spark columns from a
# MAGIC strict YAML contract. This notebook covers discovery, review, compilation, parsing, audit,
# MAGIC error policies, schema validation, configuration identity, and the complex-data boundary.
# MAGIC
# MAGIC The deliberate boundary is simple: decode or flatten complex source values upstream, parse
# MAGIC scalar leaves here, and reconstruct arrays, structs, or maps downstream when needed.

# COMMAND ----------

import json
import os
import sys
from decimal import Decimal
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql import types as T

repo_root = next(
    (
        candidate
        for candidate in [Path.cwd(), *Path.cwd().parents]
        if (candidate / "pyproject.toml").is_file()
        and (candidate / "src" / "spark_parser").is_dir()
    ),
    None,
)
assert repo_root is not None, "Run this notebook from a spark_parser repository checkout."

src_path = os.path.normpath((repo_root / "src").resolve())
while src_path in sys.path:
    sys.path.remove(src_path)
sys.path.insert(0, src_path)

# A Databricks Python process can outlive a notebook run. Clear an older wheel-backed import.
for module_name in tuple(sys.modules):
    if module_name == "spark_parser" or module_name.startswith("spark_parser."):
        del sys.modules[module_name]

from spark_parser import CompilationError, SchemaValidationError, __version__, parser

print(f"Spark Parser {__version__}")
print(f"Source checkout: {src_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Discover the scalar contract
# MAGIC
# MAGIC Metadata comes from the same enums and defaults used by compilation. The reference YAML
# MAGIC contains every scalar parser option and every string-format name.

# COMMAND ----------

expected_parser_types = {
    "string",
    "byte",
    "short",
    "integer",
    "long",
    "float",
    "decimal",
    "double",
    "binary",
    "boolean",
    "date",
    "timestamp",
    "timestamp_ntz",
}
assert set(parser.describe()) == expected_parser_types

format_values = next(
    argument["allowed_values"]
    for argument in parser.string.describe()["arguments"]
    if argument["name"] == "format"
)
assert {
    "lower",
    "upper",
    "title",
    "title_business_v1",
    "pascal",
    "address_us_v1",
    "county",
    "state_us",
    "zip",
    "interest_rate_index_v1",
    "property_type_v1",
} <= set(format_values)

assert parser.normalize_data_type(" NUMERIC ( 18, 2 ) ") == "decimal(18,2)"
assert parser.normalize_data_type("TIMESTAMP_LTZ") == "timestamp"

try:
    parser.normalize_data_type("array<string>")
except CompilationError as exc:
    print(f"Expected scalar-boundary rejection: {exc}")
else:
    raise AssertionError("Complex expected_data_type unexpectedly compiled.")

reference_config = parser.compile_path(repo_root / "examples" / "all_parsers.yaml")
assert {column.parser.parser_type.value for column in reference_config.columns} == (
    expected_parser_types
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Author a representative configuration
# MAGIC
# MAGIC The existing `title` formatter is unchanged. `title_business_v1` adds only the frozen
# MAGIC `FHLB`, `P&I`, `UST`, `RCF`, and `CMT` tokens, ASCII-hyphen component capitalization, and
# MAGIC bounded integer `Years`/`Months` hyphenation, plus its closed `Yrs` and frequency aliases.
# MAGIC `interest_rate_index_v1` is a closed, fail-closed catalog. Here an unknown rate is preserved
# MAGIC explicitly rather than inferred.
# MAGIC `property_type_v1` similarly uses approved full-value aliases for ordinary property types,
# MAGIC retains LIHTC, and restructures mixed-use components instead of discarding them.

# COMMAND ----------

CONFIG_YAML = """
parser_config_id: user_guide_scalar
parser_config_name: User Guide Scalar Contract
version: "1"
description: Demonstrate scalar parsing, error handling, and audit output.
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

  - source_column_name: mailing_address
    target_column_name: MailingAddress
    expected_data_type: string
    parser: {type: string, format: address_us_v1}

  - source_column_name: county
    target_column_name: County
    expected_data_type: string
    parser: {type: string, format: county}

  - source_column_name: state
    target_column_name: StateCode
    expected_data_type: string
    parser: {type: string, format: state_us, on_parse_error: preserve, audit: true}

  - source_column_name: postal_code
    target_column_name: PostalCode
    expected_data_type: string
    parser: {type: string, format: zip, on_parse_error: null, audit: true}

  - source_column_name: balance
    target_column_name: Balance
    expected_data_type: decimal(12,2)
    parser: {type: decimal, replace_null_markers: true, on_parse_error: null, audit: true}

  - source_column_name: quantity
    target_column_name: Quantity
    expected_data_type: integer
    parser: {type: integer, on_parse_error: default, default_on_error: 0, audit: true}

  - source_column_name: active
    target_column_name: IsActive
    expected_data_type: boolean
    parser: boolean

  - source_column_name: event_date
    target_column_name: EventDate
    expected_data_type: date
    parser: date

  - source_column_name: event_timestamp
    target_column_name: EventTimestamp
    expected_data_type: timestamp
    parser: timestamp

  - source_column_name: event_timestamp
    target_column_name: EventTimestampNtz
    expected_data_type: timestamp_ntz
    parser: timestamp_ntz

  - source_column_name: encoded_value
    target_column_name: EncodedValue
    expected_data_type: binary
    parser: {type: binary, encoding: base64}
"""

review = parser.review_yaml(CONFIG_YAML)
assert review.is_valid, review.errors
assert not review.warnings, review.warnings
displayHTML(review.to_markdown())

config = parser.compile_text(CONFIG_YAML)
resolved_mapping = parser.to_mapping(config)
assert parser.compile_mapping(resolved_mapping) == config
assert json.loads(parser.canonical_json(config)) == resolved_mapping
assert len(parser.content_hash(config)) == 64

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Bind top-level bronze strings

# COMMAND ----------

bronze_schema = """
record_id string,
business_label string,
rate_index string,
mailing_address string,
county string,
state string,
postal_code string,
balance string,
quantity string,
active string,
event_date string,
event_timestamp string,
encoded_value string
"""

bronze_rows = [
    (
        "good-1",
        "fhlb semi-annual 2 yrs 12-month advance",
        "SOFR Term - 12M",
        "123 n main st apt 4b",
        "mcclain COUNTY",
        "Illinois",
        "1234",
        "12.345",
        "7",
        "Y",
        "09/30/2026 12:00:00 AM",
        "09/30/2026 12:00:00 AM",
        "SGVsbG8=",
    ),
    (
        "handled-1",
        "biannually ust rcf cmt",
        "NAP",
        "500 w oak rd",
        "cook",
        "Mul",
        "bad-zip",
        "not-a-decimal",
        "not-an-integer",
        "N",
        "2026-08-28",
        "2026-08-28 13:45:00",
        "V29ybGQ=",
    ),
]

bronze_df = spark.createDataFrame(bronze_rows, bronze_schema)
parsing = parser.parse_dataframe(
    bronze_df,
    config,
    key_columns=["record_id"],
    column_prefix="guide_parser",
)

target_rows = {
    row.RecordId: row.asDict(recursive=True)
    for row in parsing.parsed_df.orderBy("RecordId").collect()
}

assert target_rows["good-1"]["BusinessLabel"] == "FHLB Semi-Annual 2-Years 12-Month Advance"
assert target_rows["good-1"]["RateIndex"] == "SOFR Term 12-Month"
assert target_rows["good-1"]["StateCode"] == "IL"
assert target_rows["good-1"]["PostalCode"] == "01234"
assert target_rows["good-1"]["Balance"] == Decimal("12.35")
assert target_rows["good-1"]["Quantity"] == 7
assert target_rows["good-1"]["IsActive"] is True
assert target_rows["good-1"]["EncodedValue"] == b"Hello"

assert target_rows["handled-1"]["BusinessLabel"] == "Bi-Annual UST RCF CMT"
assert target_rows["handled-1"]["RateIndex"] == "NAP"
assert target_rows["handled-1"]["StateCode"] == "Mul"
assert target_rows["handled-1"]["PostalCode"] is None
assert target_rows["handled-1"]["Balance"] is None
assert target_rows["handled-1"]["Quantity"] == 0

display(parsing.parsed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect scalar audit records
# MAGIC
# MAGIC Each audited column produces one fixed-shape struct with source/target identity, original and
# MAGIC parsed text, resolved options, ordered actions, and an optional handled-error message. There
# MAGIC are no nested child-path fields in the 0.5 audit contract.

# COMMAND ----------

audit_rows = {
    row.RecordId: {result.target_column_name: result for result in row.guide_parser_parse_results}
    for row in parsing.results_df.collect()
}

handled_audit = audit_rows["handled-1"]
assert handled_audit["RateIndex"].actions_applied == ["parse_error_preserved"]
assert handled_audit["StateCode"].actions_applied == ["parse_error_preserved"]
assert handled_audit["PostalCode"].actions_applied == ["parse_error_to_null"]
assert handled_audit["Balance"].actions_applied == ["parse_error_to_null"]
assert handled_audit["Quantity"].actions_applied == ["parse_error_default_applied"]

good_audit = audit_rows["good-1"]
assert good_audit["BusinessLabel"].changed is True
assert good_audit["BusinessLabel"].actions_applied == []
assert good_audit["RateIndex"].changed is True
assert good_audit["RateIndex"].actions_applied == []
assert good_audit["PostalCode"].changed is True
assert good_audit["PostalCode"].actions_applied == ["zip_padded"]

audit_fields = parsing.results_df.schema[
    "guide_parser_parse_results"
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

display(parsing.results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Understand failure and schema boundaries
# MAGIC
# MAGIC `fail` is lazy: Spark raises when it materializes the failing expression. Present configured
# MAGIC sources must be top-level strings. `on_missing_source="warn"` is an explicit, auditable
# MAGIC exception for a source that is absent entirely.

# COMMAND ----------

fail_config = parser.compile_text(
    """
parser_config_id: guide_fail
parser_config_name: Guide Fail
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: integer
    parser: integer
"""
)

try:
    parser.parse_dataframe(
        spark.createDataFrame([("bad",)], "value string"),
        fail_config,
        key_columns=["value"],
    ).parsed_df.select("Value").collect()
except Exception as exc:
    print(f"Expected lazy parse failure: {type(exc).__name__}")
else:
    raise AssertionError("Fail-mode value unexpectedly materialized.")

try:
    parser.parse_dataframe(
        spark.range(1).selectExpr("id AS value"),
        fail_config,
        key_columns=["value"],
    )
except SchemaValidationError as exc:
    print(f"Expected schema rejection: {exc}")
else:
    raise AssertionError("Non-string source unexpectedly bound.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Decode and reconstruct complex data outside Spark Parser
# MAGIC
# MAGIC Use Spark readers or `from_json` before parsing. Flatten only the scalar leaves the parser
# MAGIC owns. After normalization, downstream code may build whatever complex target it needs.

# COMMAND ----------

payload_schema = T.StructType(
    [
        T.StructField("amount", T.StringType()),
        T.StructField("rate_index", T.StringType()),
    ]
)
payload_df = spark.createDataFrame(
    [("row-1", '{"amount":"12.30","rate_index":"Treasury - 1 yr"}')],
    "record_id string, payload_json string",
)
flat_payload_df = payload_df.withColumn(
    "payload", F.from_json("payload_json", payload_schema)
).select(
    "record_id",
    F.col("payload.amount").alias("amount"),
    F.col("payload.rate_index").alias("rate_index"),
)
assert flat_payload_df.schema.simpleString() == (
    "struct<record_id:string,amount:string,rate_index:string>"
)

# A dedicated scalar parser config can now parse amount and rate_index. A downstream stage can use
# F.struct, F.array, map_from_entries, grouping, or joins to reconstruct the target model.
display(flat_payload_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Preserve configuration identity and pipeline ownership
# MAGIC
# MAGIC `results_df` carries the config ID, author version, canonical content hash, and installed
# MAGIC engine version. Ingestion/file discovery, upstream decoding, table writes, quarantine,
# MAGIC expectations, and downstream complex reconstruction remain orchestration responsibilities.

# COMMAND ----------

identity = parsing.results_df.first()
assert identity.guide_parser_config.id == config.parser_config_id
assert identity.guide_parser_config.version == config.version
assert identity.guide_parser_config.content_hash == parser.content_hash(config)
assert identity.guide_parser_engine_version == __version__

# Persist only on Spark runtimes that support DataFrame caching, and only when both projections will
# be materialized. Databricks serverless may reject all cache APIs.
# parsing.persist()
# parsing.parsed_df.write.mode("append").saveAsTable("target_table")
# parsing.results_df.write.mode("append").saveAsTable("parser_audit_table")
# parsing.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Operational checklist
# MAGIC
# MAGIC 1. Decode and flatten complex source data before the parser boundary.
# MAGIC 2. Keep each present configured source as a top-level string.
# MAGIC 3. Declare an exact supported scalar target datatype.
# MAGIC 4. Choose every error policy deliberately; `fail` is the default.
# MAGIC 5. Use `title_business_v1`, `interest_rate_index_v1`, and `property_type_v1` only for their
# MAGIC documented domains.
# MAGIC 6. Enable audit where its diagnostic value justifies the size.
# MAGIC 7. Supply stable row keys for audit joins.
# MAGIC 8. Review and compile YAML before binding a DataFrame.
# MAGIC 9. Reconstruct complex targets downstream.
# MAGIC 10. Run unit/integration tests and the Databricks system notebook before release.
