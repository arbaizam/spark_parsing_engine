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
  true_values: ["true", Y]
  false_values: ["false", N]
columns:
  - source_column_name: customer_name
    silver_column_name: CustomerName
    expected_data_type: string
    parser:
      type: string
      format: upper
      audit: true
  - source_column_name: nickname
    silver_column_name: Nickname
    expected_data_type: string
    parser:
      type: string
      audit: true
  - source_column_name: item_count
    silver_column_name: ItemCount
    expected_data_type: integer
    parser:
      type: integer
      zero_is_valid: false
      is_nullable: false
      default_on_null: -1
      audit: true
  - source_column_name: amount
    silver_column_name: Amount
    expected_data_type: decimal(8,2)
    parser:
      type: decimal
      replace_null_markers: true
      audit: true
  - source_column_name: opened_date
    silver_column_name: OpenedDate
    expected_data_type: date
    parser: date
  - source_column_name: is_active
    silver_column_name: IsActive
    expected_data_type: boolean
    parser:
      type: boolean
  - source_column_name: address
    silver_column_name: Address
    expected_data_type: string
    parser:
      type: string
      format: address_us_v1
      audit: true
  - source_column_name: county
    silver_column_name: County
    expected_data_type: string
    parser:
      type: string
      format: county
  - source_column_name: zip_code
    silver_column_name: ZipCode
    expected_data_type: string
    parser:
      type: string
      format: zip
      audit: true
  - source_column_name: source_not_delivered
    silver_column_name: MissingDate
    expected_data_type: date
    parser:
      type: date
      audit: true
"""
    )
    bronze_df = spark.createDataFrame(
        [
            (
                "row-1",
                " alice \t  smith ",
                " \t ",
                "0",
                "NA",
                "2026-08-27",
                "Y",
                "123 mccormick st. apt 4b",
                "mclean county",
                "123456",
            )
        ],
        [
            "row_id",
            "customer_name",
            "nickname",
            "item_count",
            "amount",
            "opened_date",
            "is_active",
            "address",
            "county",
            "zip_code",
        ],
    )

    with pytest.warns(UserWarning, match="source_not_delivered"):
        parsing = SparkDataFrameParser().parse_dataframe(
            bronze_df,
            config,
            key_columns=["row_id"],
        )

    parsed = parsing.parsed_df.first()
    assert parsed.CustomerName == "ALICE SMITH"
    assert parsed.Nickname is None
    assert parsed.ItemCount == -1
    assert parsed.Amount is None
    assert parsed.OpenedDate.isoformat() == "2026-08-27"
    assert parsed.IsActive is True
    assert parsed.Address == "123 McCormick St Apt 4B"
    assert parsed.County == "McLean County"
    assert parsed.ZipCode == "00012-3456"
    assert parsed.MissingDate is None
    assert parsing.warnings and "source_not_delivered" in parsing.warnings[0]

    audit = parsing.results_df.first().spark_parser_parse_results
    assert [item.source_column_name for item in audit] == [
        "customer_name",
        "nickname",
        "item_count",
        "amount",
        "address",
        "zip_code",
        "source_not_delivered",
    ]
    assert audit[0].changed is False
    assert audit[1].parsed_value is None
    assert audit[1].actions_applied == ["empty_string_to_null"]
    assert audit[2].parsed_value == "-1"
    assert audit[2].actions_applied == ["zero_invalidated", "default_on_null_applied"]
    assert audit[3].parsed_value is None
    assert audit[3].actions_applied == ["null_marker_replaced"]
    assert audit[4].silver_column_name == "Address"
    assert audit[5].actions_applied == ["zip_padded", "zip_plus4_formatted"]
    assert audit[6].effective is False
    assert audit[6].actions_applied == ["source_column_missing"]
    assert audit[6].error == "Source column is missing."
