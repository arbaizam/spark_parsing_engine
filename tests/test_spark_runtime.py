"""Small native-Spark smoke test for the first-round runtime contract."""

import os
import shutil

import pytest

from spark_parser import SparkDataFrameParser, YamlParserConfigCompiler

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    """Use an active Databricks session or a local session when Java exists."""
    active = SparkSession.getActiveSession()
    if active is not None:
        yield active
        return
    if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
        pytest.skip("A Java runtime is required for the Spark execution smoke test.")
    session = SparkSession.builder.master("local[1]").appName("spark-parser-test").getOrCreate()
    try:
        yield session
    finally:
        session.stop()


def test_native_parsing_and_nested_audit(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: smoke
parser_config_name: Smoke Test
version: "1"
globals:
  null_markers: [NA]
columns:
  - column_name: customer_name
    data_type: string
    parser:
      type: string
      format: pascal
      audit: true
  - column_name: item_count
    data_type: integer
    parser:
      type: integer
      zero_is_valid: false
      is_nullable: false
      default_on_null: -1
      audit: true
  - column_name: amount
    data_type: decimal(8,2)
    parser:
      type: decimal
      replace_null_markers: true
      audit: true
  - column_name: opened_date
    data_type: date
    parser: date
  - column_name: is_active
    data_type: boolean
    parser:
      type: boolean
      true_values: ["true", Y]
      false_values: ["false", N]
"""
    )
    bronze_df = spark.createDataFrame(
        [("row-1", " alice smith ", "0", "NA", "2026-08-27", "Y")],
        ["row_id", "customer_name", "item_count", "amount", "opened_date", "is_active"],
    )

    parsing = SparkDataFrameParser().parse_dataframe(
        bronze_df,
        config,
        key_columns=["row_id"],
    )

    parsed = parsing.parsed_df.first()
    assert parsed.customer_name == "AliceSmith"
    assert parsed.item_count == -1
    assert parsed.amount is None
    assert parsed.opened_date.isoformat() == "2026-08-27"
    assert parsed.is_active is True

    audit = parsing.results_df.first().spark_parser_parse_results
    assert [item.column_name for item in audit] == ["customer_name", "item_count", "amount"]
    assert audit[0].changed is False
    assert audit[1].parsed_value == "-1"
    assert audit[1].actions_applied == ["zero_invalidated", "default_on_null_applied"]
    assert audit[2].parsed_value is None
    assert audit[2].actions_applied == ["null_marker_replaced"]
