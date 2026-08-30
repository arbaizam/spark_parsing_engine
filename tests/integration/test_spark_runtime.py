"""Native-Spark behavioral tests for the complete runtime contract."""

import os
import shutil
import sys

import pytest
from py4j.protocol import Py4JJavaError

from spark_parser import SchemaValidationError, SparkDataFrameParser, YamlParserConfigCompiler

pyspark = pytest.importorskip("pyspark")
from pyspark.errors import PySparkException  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark():
    """Use an active Databricks session or a local session when Java exists."""
    active = SparkSession.getActiveSession()
    if active is not None:
        yield active
        return
    if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
        if os.environ.get("SPARK_PARSER_REQUIRE_JAVA") == "1":
            pytest.fail(
                "SPARK_PARSER_REQUIRE_JAVA=1, but no Java runtime was found on PATH or JAVA_HOME"
            )
        pytest.skip("A Java runtime is required for the Spark execution smoke test.")
    # Spark defaults to a ``python3`` worker command that is not guaranteed to exist on Windows.
    # Pinning both sides to the interpreter running pytest makes local and CI execution equivalent
    # and prevents expression tests from failing for an unrelated process-launch reason.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[1]")
        .appName("spark-parser-test")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )
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
    target_column_name: CustomerName
    expected_data_type: string
    parser:
      type: string
      format: upper
      audit: true
  - source_column_name: nickname
    target_column_name: Nickname
    expected_data_type: string
    parser:
      type: string
      audit: true
  - source_column_name: item_count
    target_column_name: ItemCount
    expected_data_type: integer
    parser:
      type: integer
      zero_is_valid: false
      is_nullable: false
      default_on_null: -1
      audit: true
  - source_column_name: amount
    target_column_name: Amount
    expected_data_type: decimal(8,2)
    parser:
      type: decimal
      replace_null_markers: true
      audit: true
  - source_column_name: opened_date
    target_column_name: OpenedDate
    expected_data_type: date
    parser: date
  - source_column_name: is_active
    target_column_name: IsActive
    expected_data_type: boolean
    parser:
      type: boolean
  - source_column_name: address
    target_column_name: Address
    expected_data_type: string
    parser:
      type: string
      format: address_us_v1
      audit: true
  - source_column_name: county
    target_column_name: County
    expected_data_type: string
    parser:
      type: string
      format: county
  - source_column_name: zip_code
    target_column_name: ZipCode
    expected_data_type: string
    parser:
      type: string
      format: zip
      audit: true
  - source_column_name: source_not_delivered
    target_column_name: MissingDate
    expected_data_type: date
    parser:
      type: date
      audit: true
"""
    )
    bronze_df = spark.sql(
        """
SELECT
  'row-1' AS row_id,
  concat(' alice ', char(9), '  smith ') AS customer_name,
  concat(' ', char(9), ' ') AS nickname,
  '0' AS item_count,
  'NA' AS amount,
  '2026-08-27' AS opened_date,
  'Y' AS is_active,
  '123 mccormick st. apt 4b' AS address,
  'mclean county' AS county,
  '123-45' AS zip_code
"""
    )

    with pytest.raises(SchemaValidationError, match="Configured source columns are missing"):
        SparkDataFrameParser().parse_dataframe(
            bronze_df,
            config,
            key_columns=["row_id"],
        )
    with pytest.warns(UserWarning, match="source_not_delivered"):
        parsing = SparkDataFrameParser().parse_dataframe(
            bronze_df,
            config,
            key_columns=["row_id"],
            on_missing_source="warn",
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
    assert parsed.ZipCode == "00123-0045"
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
    assert audit[4].target_column_name == "Address"
    assert audit[5].actions_applied == ["zip_padded", "zip_plus4_formatted"]
    assert audit[6].effective is False
    assert audit[6].actions_applied == ["source_column_missing"]
    assert audit[6].error == "Source column is missing."


def test_address_county_and_zip_edge_cases_under_ansi(spark: SparkSession) -> None:
    assert spark.conf.get("spark.sql.ansi.enabled") == "true"
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: location_edges
parser_config_name: Location Edges
version: "1"
columns:
  - source_column_name: address
    target_column_name: Address
    expected_data_type: string
    parser:
      type: string
      format: address_us_v1
      audit: true
  - source_column_name: address
    target_column_name: AddressRequired
    expected_data_type: string
    parser:
      type: string
      format: address_us_v1
      is_nullable: false
      default_on_null: UNKNOWN
  - source_column_name: county
    target_column_name: County
    expected_data_type: string
    parser:
      type: string
      format: county
      on_parse_error: null
  - source_column_name: zip_code
    target_column_name: ZipCode
    expected_data_type: string
    parser:
      type: string
      format: zip
      on_parse_error: null
      audit: true
"""
    )
    bronze_df = spark.sql(
        """
SELECT 1 AS row_id, CAST(NULL AS STRING) AS address, 'County' AS county, '1234A' AS zip_code
UNION ALL
SELECT 2, '123 Center Street , Apt #4b', concat('o', char(39), 'brien county'), '12345-6789'
UNION ALL
SELECT 3, 'route66 road', 'smith-jones county', '123-45'
"""
    )

    parsing = SparkDataFrameParser().parse_dataframe(
        bronze_df,
        config,
        key_columns=["row_id"],
    )
    rows = parsing.parsed_df.collect()
    assert rows[0].Address is None
    assert rows[0].AddressRequired == "UNKNOWN"
    assert rows[0].County is None
    assert rows[0].ZipCode is None
    assert rows[1].Address == "123 Center St Apt #4B"
    assert rows[1].County == "O'Brien County"
    assert rows[1].ZipCode == "12345-6789"
    assert rows[2].Address == "Route66 Rd"
    assert rows[2].County == "Smith-Jones County"
    assert rows[2].ZipCode == "00123-0045"

    audits = [row.spark_parser_parse_results for row in parsing.results_df.collect()]
    assert audits[0][0].parsed_value is None
    assert audits[0][1].actions_applied == ["parse_error_to_null"]
    assert audits[1][1].actions_applied == []
    assert audits[2][1].actions_applied == ["zip_padded", "zip_plus4_formatted"]


def test_title_and_state_us_formats_under_ansi(spark: SparkSession) -> None:
    """Validate space-preserving title case and strict state-code normalization."""
    assert spark.conf.get("spark.sql.ansi.enabled") == "true"
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: display_and_state
parser_config_name: Display and State Formats
version: "1"
columns:
  - source_column_name: label
    target_column_name: DisplayLabel
    expected_data_type: string
    parser:
      type: string
      format: title
  - source_column_name: state
    target_column_name: StateCode
    expected_data_type: string
    parser:
      type: string
      format: state_us
      on_parse_error: null
      audit: true
  - source_column_name: state
    target_column_name: RequiredStateCode
    expected_data_type: string
    parser:
      type: string
      format: state_us
      on_parse_error: default
      default_on_error: UNKNOWN
      audit: true
  - source_column_name: state
    target_column_name: PreservedStateCode
    expected_data_type: string
    parser:
      type: string
      format: state_us
      on_parse_error: preserve
      audit: true
"""
    )
    bronze_df = spark.range(11).select(
        (F.col("id") + 1).cast("integer").alias("row_id"),
        F.element_at(
            F.array(
                F.lit("  LOAN   status "),
                F.lit("mixed CASE"),
                F.lit("account owner"),
                F.lit("district record"),
                F.lit("punctuated district"),
                F.lit("conventional abbreviation"),
                F.lit("punctuated code"),
                F.lit("conventional california"),
                F.lit("conventional north dakota"),
                F.lit("conventional west virginia"),
                F.lit("invalid state"),
            ),
            (F.col("id") + 1).cast("integer"),
        ).alias("label"),
        F.element_at(
            F.array(
                F.lit("Illinois"),
                F.lit("  new   york "),
                F.lit("il"),
                F.lit("District of Columbia"),
                F.lit("Washington, D.C."),
                F.lit("Ill."),
                F.lit("WA."),
                F.lit("Calif."),
                F.lit("N. Dak."),
                F.lit("W. Va."),
                F.lit("  Mul  "),
            ),
            (F.col("id") + 1).cast("integer"),
        ).alias("state"),
    )

    parsing = SparkDataFrameParser().parse_dataframe(
        bronze_df,
        config,
        key_columns=["row_id"],
    )
    rows = parsing.parsed_df.collect()
    assert [row.DisplayLabel for row in rows] == [
        "Loan Status",
        "Mixed Case",
        "Account Owner",
        "District Record",
        "Punctuated District",
        "Conventional Abbreviation",
        "Punctuated Code",
        "Conventional California",
        "Conventional North Dakota",
        "Conventional West Virginia",
        "Invalid State",
    ]
    assert [row.StateCode for row in rows] == [
        "IL",
        "NY",
        "IL",
        "DC",
        "DC",
        "IL",
        "WA",
        "CA",
        "ND",
        "WV",
        None,
    ]
    assert [row.RequiredStateCode for row in rows] == [
        "IL",
        "NY",
        "IL",
        "DC",
        "DC",
        "IL",
        "WA",
        "CA",
        "ND",
        "WV",
        "UNKNOWN",
    ]
    assert [row.PreservedStateCode for row in rows] == [
        "IL",
        "NY",
        "IL",
        "DC",
        "DC",
        "IL",
        "WA",
        "CA",
        "ND",
        "WV",
        "  Mul  ",
    ]

    invalid_audit = parsing.results_df.orderBy("row_id").collect()[-1].spark_parser_parse_results
    assert invalid_audit[0].actions_applied == ["parse_error_to_null"]
    assert invalid_audit[1].actions_applied == ["parse_error_default_applied"]
    assert invalid_audit[2].actions_applied == ["parse_error_preserved"]
    assert invalid_audit[2].original_value == "  Mul  "
    assert invalid_audit[2].parsed_value == "  Mul  "
    assert invalid_audit[2].changed is True


def test_territories_and_single_or_multiple_property_values(spark: SparkSession) -> None:
    """Parse scalar/list property locations while retaining a canonical string target."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: property_locations
parser_config_name: Property Locations
version: "1"
columns:
  - source_column_name: single_state
    target_column_name: SingleState
    expected_data_type: string
    parser: {type: string, format: state_us}
  - source_column_name: territory_states
    target_column_name: TerritoryStates
    expected_data_type: string
    parser: {type: string, format: state_us}
  - source_column_name: dc_and_state
    target_column_name: DcAndState
    expected_data_type: string
    parser: {type: string, format: state_us}
  - source_column_name: single_zip
    target_column_name: SingleZip
    expected_data_type: string
    parser: {type: string, format: zip}
  - source_column_name: property_zips
    target_column_name: PropertyZips
    expected_data_type: string
    parser: {type: string, format: zip, audit: true}
  - source_column_name: invalid_states
    target_column_name: InvalidStates
    expected_data_type: string
    parser: {type: string, format: state_us, on_parse_error: null}
  - source_column_name: invalid_zips
    target_column_name: InvalidZips
    expected_data_type: string
    parser: {type: string, format: zip, on_parse_error: null}
"""
    )
    bronze_df = spark.range(1).select(
        F.lit("Puerto Rico").alias("single_state"),
        F.lit(
            "American Samoa, Guam, Northern Mariana Islands, Puerto Rico, U.S. Virgin Islands"
        ).alias("territory_states"),
        F.lit("Washington, D.C., Illinois").alias("dc_and_state"),
        F.lit("1234").alias("single_zip"),
        F.lit("1234, 67890").alias("property_zips"),
        F.lit("IL, Mul").alias("invalid_states"),
        F.lit("12345, nope").alias("invalid_zips"),
    )

    row = (
        SparkDataFrameParser()
        .parse_dataframe(bronze_df, config, key_columns=["single_state"])
        .parsed_df.first()
    )
    assert row.SingleState == "PR"
    assert row.TerritoryStates == "AS, GU, MP, PR, VI"
    assert row.DcAndState == "DC, IL"
    assert row.SingleZip == "01234"
    assert row.PropertyZips == "01234, 67890"
    assert row.InvalidStates is None
    assert row.InvalidZips is None
    audit = (
        SparkDataFrameParser()
        .parse_dataframe(bronze_df, config, key_columns=["single_state"])
        .results_df.first()
        .spark_parser_parse_results[0]
    )
    assert audit.actions_applied == ["zip_padded"]


def test_zip_rejects_ambiguous_compact_six_to_eight_digit_values(
    spark: SparkSession,
) -> None:
    """Require an explicit comma or a complete nine-digit ZIP+4 representation."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: zip_shapes
parser_config_name: ZIP Shapes
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: string
    parser: {type: string, format: zip, on_parse_error: null}
"""
    )
    tokens = ["123456", "1234567", "12345678", "123456789"]
    rows = (
        SparkDataFrameParser()
        .parse_dataframe(
            spark.range(len(tokens)).select(
                F.col("id").cast("integer").alias("row_id"),
                F.element_at(
                    F.array(*(F.lit(token) for token in tokens)),
                    (F.col("id") + 1).cast("integer"),
                ).alias("value"),
            ),
            config,
            key_columns=["row_id"],
        )
        .parsed_df.orderBy("row_id")
        .collect()
    )
    assert [row.Value for row in rows] == [None, None, None, "12345-6789"]


def test_nested_string_error_policies_can_preserve_raw_tokens(spark: SparkSession) -> None:
    """Preserve invalid formatted strings inside structs, arrays, and maps without losing paths."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: nested_preserve
parser_config_name: Nested Preserve
version: "1"
columns:
  - source_column_name: profile
    target_column_name: Profile
    expected_data_type: struct<state:string>
    parser:
      type: struct
      fields:
        - source_field_name: state
          target_field_name: state
          parser: {type: string, format: state_us, on_parse_error: preserve}
      audit: true
  - source_column_name: states
    target_column_name: States
    expected_data_type: array<string>
    parser:
      type: array
      element_parser: {type: string, format: state_us}
      on_element_error: preserve
      audit: true
  - source_column_name: state_map
    target_column_name: StateMap
    expected_data_type: map<string,string>
    parser:
      type: map
      value_parser: {type: string, format: state_us}
      on_value_error: preserve
      audit: true
"""
    )
    bronze_df = spark.range(1).select(
        F.lit(1).cast("integer").alias("row_id"),
        F.lit('{"state":"Mul"}').alias("profile"),
        F.lit('["Illinois","Mul","ny"]').alias("states"),
        F.lit('{"home":"Illinois","other":"Mul"}').alias("state_map"),
    )

    parsing = SparkDataFrameParser().parse_dataframe(bronze_df, config, key_columns=["row_id"])
    row = parsing.parsed_df.first()
    assert row.Profile.state == "Mul"
    assert row.States == ["IL", "Mul", "NY"]
    assert row.StateMap == {"home": "IL", "other": "Mul"}

    audits = {
        item.target_column_name: item
        for item in parsing.results_df.first().spark_parser_parse_results
    }
    assert audits["Profile"].nested_error_paths == ["$.state"]
    assert audits["States"].nested_error_paths == ["$[1]"]
    assert audits["StateMap"].nested_error_paths == ["$['other']"]
    assert all(
        item.actions_applied == ["nested_parse_errors_resolved"]
        for item in audits.values()
    )


def test_datetime_defaults_accept_iso_and_us_12_hour_timestamp(spark: SparkSession) -> None:
    """Keep documented date and timestamp defaults working under production-style ANSI mode."""
    assert spark.conf.get("spark.sql.ansi.enabled") == "true"
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: default_date_formats
parser_config_name: Default Date Formats
version: "1"
columns:
  - source_column_name: event_date
    target_column_name: EventDate
    expected_data_type: date
    parser:
      type: date
      on_parse_error: null
  - source_column_name: event_timestamp
    target_column_name: EventTimestamp
    expected_data_type: timestamp
    parser:
      type: timestamp
      on_parse_error: null
  - source_column_name: event_timestamp
    target_column_name: EventTimestampNtz
    expected_data_type: timestamp_ntz
    parser:
      type: timestamp_ntz
      on_parse_error: null
"""
    )
    bronze_df = spark.range(6).select(
        (F.col("id") + 1).cast("integer").alias("row_id"),
        F.element_at(
            F.array(
                F.lit("2026-09-29 01:02:03"),
                F.lit("09/30/2026 8:08 AM"),
                F.lit("09/30/2026 08:08 AM"),
                F.lit("06/11/2026 8:08:17 PM"),
                F.lit("06/11/2026 08:08:17 PM"),
                # A bare slash date remains invalid. Accepting it would silently guess whether the
                # source uses month/day or day/month ordering.
                F.lit("09/10/2026"),
            ),
            (F.col("id") + 1).cast("integer"),
        ).alias("event_date"),
        F.element_at(
            F.array(
                F.lit("2026-09-29 01:02:03"),
                F.lit("09/30/2026 8:08 AM"),
                F.lit("09/30/2026 08:08 AM"),
                F.lit("06/11/2026 8:08:17 PM"),
                F.lit("06/11/2026 08:08:17 PM"),
                F.lit("09/10/2026"),
            ),
            (F.col("id") + 1).cast("integer"),
        ).alias("event_timestamp"),
    )

    rows = (
        SparkDataFrameParser()
        .parse_dataframe(bronze_df, config, key_columns=["row_id"])
        .parsed_df.orderBy("row_id")
        .collect()
    )
    assert rows[0].EventDate.isoformat() == "2026-09-29"
    assert rows[1].EventDate.isoformat() == "2026-09-30"
    assert rows[2].EventDate.isoformat() == "2026-09-30"
    assert rows[3].EventDate.isoformat() == "2026-06-11"
    assert rows[4].EventDate.isoformat() == "2026-06-11"
    assert rows[5].EventDate is None
    assert str(rows[0].EventTimestamp) == "2026-09-29 01:02:03"
    assert str(rows[1].EventTimestamp) == "2026-09-30 08:08:00"
    assert str(rows[2].EventTimestamp) == "2026-09-30 08:08:00"
    assert str(rows[3].EventTimestamp) == "2026-06-11 20:08:17"
    assert str(rows[4].EventTimestamp) == "2026-06-11 20:08:17"
    assert rows[5].EventTimestamp is None
    assert str(rows[0].EventTimestampNtz) == "2026-09-29 01:02:03"
    assert str(rows[1].EventTimestampNtz) == "2026-09-30 08:08:00"
    assert str(rows[2].EventTimestampNtz) == "2026-09-30 08:08:00"
    assert str(rows[3].EventTimestampNtz) == "2026-06-11 20:08:17"
    assert str(rows[4].EventTimestampNtz) == "2026-06-11 20:08:17"
    assert rows[5].EventTimestampNtz is None


def test_iso_timestamp_defaults_cover_fractional_and_offset_input(spark: SparkSession) -> None:
    """Parse modern ISO timestamps while keeping timestamp_ntz strictly timezone-free."""
    previous_timezone = spark.conf.get("spark.sql.session.timeZone")
    previous_time_parser_policy = spark.conf.get("spark.sql.legacy.timeParserPolicy")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "EXCEPTION")
    try:
        config = YamlParserConfigCompiler().compile_text(
            """
parser_config_id: iso_timestamp_formats
parser_config_name: ISO Timestamp Formats
version: "1"
columns:
  - source_column_name: value
    target_column_name: EventDate
    expected_data_type: date
    parser: {type: date, on_parse_error: null}
  - source_column_name: value
    target_column_name: EventTimestamp
    expected_data_type: timestamp
    parser: {type: timestamp, on_parse_error: null}
  - source_column_name: value
    target_column_name: EventTimestampNtz
    expected_data_type: timestamp_ntz
    parser: {type: timestamp_ntz, on_parse_error: null}
"""
        )
        bronze_df = spark.range(3).select(
            (F.col("id") + 1).cast("integer").alias("row_id"),
            F.element_at(
                F.array(
                    F.lit("2026-09-30T12:34:56.123456"),
                    F.lit("2026-09-30T12:34:56Z"),
                    F.lit("2026-09-30T12:34:56-05:00"),
                ),
                (F.col("id") + 1).cast("integer"),
            ).alias("value"),
        )

        parsed_df = SparkDataFrameParser().parse_dataframe(
            bronze_df,
            config,
            key_columns=["row_id"],
        ).parsed_df
        # Render timestamps inside Spark. Collecting TimestampType as a Python datetime uses the
        # host process timezone, which is separate from spark.sql.session.timeZone and would make
        # this assertion test the workstation rather than the parser expression.
        rows = (
            parsed_df.orderBy("row_id").select(
                "EventDate",
                F.date_format("EventTimestamp", "yyyy-MM-dd HH:mm:ss.SSSSSS").alias(
                    "EventTimestampText"
                ),
                F.date_format("EventTimestampNtz", "yyyy-MM-dd HH:mm:ss.SSSSSS").alias(
                    "EventTimestampNtzText"
                ),
            ).collect()
        )

        assert rows[0].EventDate.isoformat() == "2026-09-30"
        # Date intentionally rejects offsets so its calendar day never changes with session zone.
        assert rows[1].EventDate is None
        assert rows[2].EventDate is None
        assert rows[0].EventTimestampText == "2026-09-30 12:34:56.123456"
        assert rows[1].EventTimestampText == "2026-09-30 12:34:56.000000"
        assert rows[2].EventTimestampText == "2026-09-30 17:34:56.000000"
        assert rows[0].EventTimestampNtzText == "2026-09-30 12:34:56.123456"
        assert rows[1].EventTimestampNtzText is None
        assert rows[2].EventTimestampNtzText is None
    finally:
        spark.conf.set("spark.sql.session.timeZone", previous_timezone)
        spark.conf.set("spark.sql.legacy.timeParserPolicy", previous_time_parser_policy)


def test_top_level_and_nested_double_share_one_numeric_token_contract(
    spark: SparkSession,
) -> None:
    """Ensure a numeric token cannot succeed or fail solely because it appears inside an array."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: numeric_parity
parser_config_name: Numeric Parity
version: "1"
columns:
  - source_column_name: scalar
    target_column_name: Scalar
    expected_data_type: double
    parser: {type: double, on_parse_error: null}
  - source_column_name: nested
    target_column_name: Nested
    expected_data_type: array<double>
    parser:
      type: array
      element_parser: double
      on_element_error: null
"""
    )
    tokens = ["1d", "1f", "0x1p3", "1e5", ".5", "+1.5", "007"]
    bronze_df = spark.range(len(tokens)).select(
        F.col("id").cast("integer").alias("row_id"),
        F.element_at(
            F.array(*(F.lit(token) for token in tokens)),
            (F.col("id") + 1).cast("integer"),
        ).alias("scalar"),
        F.element_at(
            F.array(*(F.lit(f'["{token}"]') for token in tokens)),
            (F.col("id") + 1).cast("integer"),
        ).alias("nested"),
    )

    rows = (
        SparkDataFrameParser()
        .parse_dataframe(bronze_df, config, key_columns=["row_id"])
        .parsed_df.orderBy("row_id")
        .collect()
    )

    assert [row.Scalar for row in rows] == [None, None, None, 100000.0, 0.5, 1.5, 7.0]
    assert [row.Nested[0] for row in rows] == [
        None,
        None,
        None,
        100000.0,
        0.5,
        1.5,
        7.0,
    ]


def test_ansi_and_non_ansi_modes_produce_identical_handled_outputs(
    spark: SparkSession,
) -> None:
    """Keep parser policy—not Spark's global cast mode—in control of representative bad values."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: ansi_parity
parser_config_name: ANSI Parity
version: "1"
columns:
  - source_column_name: integer_value
    target_column_name: IntegerValue
    expected_data_type: integer
    parser: {type: integer, on_parse_error: null, audit: true}
  - source_column_name: double_value
    target_column_name: DoubleValue
    expected_data_type: double
    parser: {type: double, on_parse_error: null, audit: true}
  - source_column_name: state_value
    target_column_name: StateValue
    expected_data_type: string
    parser: {type: string, format: state_us, on_parse_error: null, audit: true}
  - source_column_name: date_value
    target_column_name: DateValue
    expected_data_type: date
    parser: {type: date, on_parse_error: null, audit: true}
  - source_column_name: array_value
    target_column_name: ArrayValue
    expected_data_type: array<double>
    parser:
      type: array
      element_parser: double
      on_element_error: null
      audit: true
"""
    )
    bronze_df = spark.sql(
        """
SELECT
  '1.9' AS integer_value,
  '1d' AS double_value,
  'Ill.' AS state_value,
  '09/30/2026 12:00 AM' AS date_value,
  '["1d","2.5"]' AS array_value
"""
    )
    previous_ansi = spark.conf.get("spark.sql.ansi.enabled")
    snapshots: dict[str, tuple[dict, list[dict]]] = {}
    try:
        for ansi_value in ("true", "false"):
            spark.conf.set("spark.sql.ansi.enabled", ansi_value)
            parsing = SparkDataFrameParser().parse_dataframe(
                bronze_df,
                config,
                key_columns=["integer_value"],
            )
            target = parsing.parsed_df.first().asDict(recursive=True)
            audit = [
                item.asDict(recursive=True)
                for item in parsing.results_df.first().spark_parser_parse_results
            ]
            snapshots[ansi_value] = target, audit
    finally:
        spark.conf.set("spark.sql.ansi.enabled", previous_ansi)

    assert snapshots["true"] == snapshots["false"]
    target = snapshots["true"][0]
    assert target["IntegerValue"] is None
    assert target["DoubleValue"] is None
    assert target["StateValue"] == "IL"
    assert target["DateValue"].isoformat() == "2026-09-30"
    assert target["ArrayValue"] == [None, 2.5]


def test_nested_defaults_and_zero_invalidation_are_visible_in_audit(
    spark: SparkSession,
) -> None:
    """Record every nested fabricated value so target data remains explainable downstream."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: nested_default_audit
parser_config_name: Nested Default Audit
version: "1"
columns:
  - source_column_name: values
    target_column_name: Values
    expected_data_type: array<integer>
    parser:
      type: array
      element_parser:
        type: integer
        zero_is_valid: false
        is_nullable: false
        default_on_null: -1
      on_element_error: null
      audit: true
"""
    )
    parsing = SparkDataFrameParser().parse_dataframe(
        spark.range(1).select(
            F.lit(1).cast("integer").alias("row_id"),
            F.lit('[1,"bad",0,null]').alias("values"),
        ),
        config,
        key_columns=["row_id"],
    )

    assert parsing.parsed_df.first().Values == [1, -1, -1, -1]
    audit = parsing.results_df.first().spark_parser_parse_results[0]
    assert audit.nested_error_paths == ["$[1]"]
    assert audit.nested_default_on_null_paths == ["$[1]", "$[2]", "$[3]"]
    assert audit.nested_zero_invalidated_paths == ["$[2]"]
    assert audit.actions_applied == [
        "nested_parse_errors_resolved",
        "nested_zero_invalidated",
        "nested_default_on_null_applied",
    ]
    assert audit.changed is True


def test_map_error_paths_escape_apostrophes_in_keys(spark: SparkSession) -> None:
    """Keep a diagnostic map key unambiguous when the source key itself contains an apostrophe."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: escaped_map_path
parser_config_name: Escaped Map Path
version: "1"
columns:
  - source_column_name: values
    target_column_name: Values
    expected_data_type: map<string,integer>
    parser:
      type: map
      value_parser: integer
      on_value_error: null
      audit: true
"""
    )
    parsing = SparkDataFrameParser().parse_dataframe(
        spark.range(1).select(
            F.lit(1).cast("integer").alias("row_id"),
            F.lit("{\"a'b\":\"bad\"}").alias("values"),
        ),
        config,
        key_columns=["row_id"],
    )

    audit = parsing.results_df.first().spark_parser_parse_results[0]
    assert audit.nested_error_paths == ["$['a\\'b']"]


def test_audit_schema_is_consistent_with_or_without_audited_columns(spark: SparkSession) -> None:
    template = """
parser_config_id: audit_schema
parser_config_name: Audit Schema
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: string
    parser:
      type: string
      audit: {audit}
"""
    df = spark.sql("SELECT 'x' AS value")
    compiler = YamlParserConfigCompiler()
    audited = SparkDataFrameParser().parse_dataframe(
        df,
        compiler.compile_text(template.format(audit="true")),
        key_columns=["value"],
    )
    empty = SparkDataFrameParser().parse_dataframe(
        df,
        compiler.compile_text(template.format(audit="false")),
        key_columns=["value"],
    )

    audited_type = audited.results_df.schema["spark_parser_parse_results"].dataType
    empty_type = empty.results_df.schema["spark_parser_parse_results"].dataType
    assert audited_type == empty_type
    assert audited_type.elementType.fieldNames() == [
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
    assert audited.results_df.first().spark_parser_parse_results
    assert empty.results_df.first().spark_parser_parse_results == []


def test_duplicate_input_names_reject_ambiguous_explicit_keys(
    spark: SparkSession,
) -> None:
    """Reject an explicitly selected key whose input name occurs more than once."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: duplicate_default_keys
parser_config_name: Duplicate Default Keys
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: string
    parser: string
"""
    )
    df = spark.sql("SELECT 'ok' AS value, 1 AS duplicate, 2 AS duplicate")

    with pytest.raises(TypeError, match="key_columns"):
        SparkDataFrameParser().parse_dataframe(df, config)
    with pytest.raises(SchemaValidationError, match="key_columns are ambiguous"):
        SparkDataFrameParser().parse_dataframe(df, config, key_columns=["duplicate"])


def test_fail_default_boolean_trim_and_decimal_runtime_contracts(
    spark: SparkSession,
) -> None:
    compiler = YamlParserConfigCompiler()
    fail_config = compiler.compile_text(
        """
parser_config_id: fail
parser_config_name: Fail
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: integer
    parser: integer
"""
    )
    failing = SparkDataFrameParser().parse_dataframe(
        spark.sql("SELECT 'abc' AS value"),
        fail_config,
        key_columns=["value"],
    )
    assert failing.parsed_df.columns == ["Value"]
    with pytest.raises((Py4JJavaError, PySparkException)):
        failing.parsed_df.select("Value").collect()

    behavior_config = compiler.compile_text(
        """
parser_config_id: behavior
parser_config_name: Behavior
version: "1"
globals:
  true_values: ["true", Y]
  false_values: ["false", N]
  boolean_case_sensitive: false
columns:
  - source_column_name: bad_integer
    target_column_name: DefaultedInteger
    expected_data_type: integer
    parser:
      type: integer
      on_parse_error: default
      default_on_error: -1
      audit: true
  - source_column_name: boolean_value
    target_column_name: BooleanValue
    expected_data_type: boolean
    parser:
      type: boolean
      on_parse_error: null
  - source_column_name: trim_value
    target_column_name: TrimValue
    expected_data_type: string
    parser:
      type: string
      collapse_whitespace: false
      trim_whitespace: true
  - source_column_name: nbsp_value
    target_column_name: NbspValue
    expected_data_type: string
    parser: string
  - source_column_name: decimal_value
    target_column_name: DecimalValue
    expected_data_type: decimal(18,2)
    parser: decimal
"""
    )
    df = spark.sql(
        """
SELECT
  'abc' AS bad_integer,
  'y' AS boolean_value,
  concat(char(9), ' x ', char(10)) AS trim_value,
  concat(char(160), 'x', char(160)) AS nbsp_value,
  '1.239' AS decimal_value
"""
    )
    parsing = SparkDataFrameParser().parse_dataframe(
        df,
        behavior_config,
        key_columns=["bad_integer"],
    )
    row = parsing.parsed_df.first()
    assert row.DefaultedInteger == -1
    assert row.BooleanValue is True
    assert row.TrimValue == "x"
    assert row.NbspValue == "x"
    assert str(row.DecimalValue) == "1.24"
    audit = parsing.results_df.first().spark_parser_parse_results[0]
    assert audit.actions_applied == ["parse_error_default_applied"]
    assert audit.options["type"] == "integer"


def test_wide_config_uses_constant_depth_projection_stages(
    spark: SparkSession,
    capsys: pytest.CaptureFixture[str],
) -> None:
    column_count = 40
    df = spark.sql("SELECT " + ", ".join(f"'x' AS c{index}" for index in range(column_count)))
    columns = "\n".join(
        f"  - source_column_name: c{index}\n"
        f"    target_column_name: C{index}\n"
        "    expected_data_type: string\n"
        "    parser: string"
        for index in range(column_count)
    )
    config = YamlParserConfigCompiler().compile_text(
        f"parser_config_id: wide\nparser_config_name: Wide\nversion: '1'\ncolumns:\n{columns}\n"
    )
    parsing = SparkDataFrameParser().parse_dataframe(
        df,
        config,
        key_columns=["c0"],
    )

    parsing.parsed_df.explain(mode="extended")
    explain_output = capsys.readouterr().out
    analyzed_marker = "== Analyzed Logical Plan =="
    assert analyzed_marker in explain_output
    analyzed = explain_output.split(analyzed_marker, maxsplit=1)[1].split("\n== ", maxsplit=1)[0]
    assert analyzed.count("Project") <= 10


def test_timestamp_options_use_iso_text() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: timestamp_option
parser_config_name: Timestamp Option
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: timestamp
    parser:
      type: timestamp
      is_nullable: false
      default_on_null: "1970-01-01T00:00:00"
"""
    )
    value = config.columns[0].parser.default_on_null
    assert SparkDataFrameParser._option_text(value) == "1970-01-01T00:00:00"


def test_complex_types_parse_recursively_with_paths_and_canonical_audit(
    spark: SparkSession,
) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: complex_runtime
parser_config_name: Complex Runtime
version: "1"
globals:
  true_values: [Y]
  false_values: [N]
  boolean_case_sensitive: false
columns:
  - source_column_name: names
    target_column_name: Names
    expected_data_type: array<string>
    parser:
      type: array
      element_parser: {type: string, format: upper}
      on_element_error: drop
      drop_null_elements: true
      distinct: true
      audit: true
  - source_column_name: object
    target_column_name: Object
    expected_data_type: struct<street:string,zip:string,scores:array<integer>,approved:boolean,due:date>
    parser:
      type: struct
      fields:
        - source_field_name: address
          target_field_name: street
          parser: {type: string, format: address_us_v1}
        - source_field_name: postal
          target_field_name: zip
          parser: {type: string, format: zip, on_parse_error: null}
        - source_field_name: values
          target_field_name: scores
          parser: {type: array, element_parser: integer, on_element_error: null}
        - {source_field_name: is_approved, target_field_name: approved, parser: boolean}
        - source_field_name: due_date
          target_field_name: due
          parser: date
      audit: true
  - source_column_name: attributes
    target_column_name: Attributes
    expected_data_type: map<string,decimal(8,2)>
    parser: {type: map, value_parser: decimal, on_value_error: drop, audit: true}
"""
    )
    df = spark.sql(
        """
SELECT
  1 AS id,
  '[" alice ","BOB",null,"alice"]' AS names,
  '{"address":"123 mccormick st. apt #4b","postal":"1234","values":[1,"bad",3],"is_approved":"y","due_date":"08/27/2026 12:00 AM"}' AS object,
  '{"a":12.345,"bad":"x","empty":null}' AS attributes
"""
    )

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["id"])
    row = parsing.parsed_df.first()
    assert row.Names == ["ALICE", "BOB"]
    assert row.Object.street == "123 McCormick St Apt #4B"
    assert row.Object.zip == "01234"
    assert row.Object.scores == [1, None, 3]
    assert row.Object.approved is True
    assert str(row.Object.due) == "2026-08-27"
    assert str(row.Attributes["a"]) == "12.35"
    assert "bad" not in row.Attributes
    assert row.Attributes["empty"] is None

    audits = {
        item.target_column_name: item
        for item in parsing.results_df.first().spark_parser_parse_results
    }
    assert audits["Names"].parsed_value == '["ALICE","BOB"]'
    assert audits["Object"].nested_error_paths == ["$.scores[1]"]
    assert audits["Attributes"].nested_error_paths == ["$['bad']"]
    assert "nested_parse_errors_resolved" in audits["Object"].actions_applied
    assert '"type":"string"' in audits["Names"].options["element_parser"]


def test_delimited_arrays_complex_defaults_and_malformed_json(
    spark: SparkSession,
) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: complex_edges
parser_config_name: Complex Edges
version: "1"
columns:
  - source_column_name: numbers
    target_column_name: Numbers
    expected_data_type: array<integer>
    parser:
      type: array
      input_format: delimited
      delimiter: "|"
      element_parser: integer
      on_element_error: drop
      distinct: true
      audit: true
  - source_column_name: object
    target_column_name: Object
    expected_data_type: struct<a:integer>
    parser:
      type: struct
      fields:
        - {source_field_name: a, target_field_name: a, parser: integer}
      on_parse_error: default
      default_on_error: {a: -1}
      audit: true
"""
    )
    df = spark.sql("SELECT 1 AS id, '001|2|bad|2' AS numbers, 'not json' AS object")

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["id"])
    row = parsing.parsed_df.first()
    assert row.Numbers == [1, 2]
    assert row.Object.a == -1
    audits = {
        item.target_column_name: item
        for item in parsing.results_df.first().spark_parser_parse_results
    }
    assert audits["Numbers"].nested_error_paths == ["$[2]"]
    assert audits["Object"].nested_error_paths == ["$"]
    assert audits["Object"].actions_applied == ["parse_error_default_applied"]


def test_new_scalar_types_are_first_class(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: new_scalars
parser_config_name: New Scalars
version: "1"
columns:
  - {source_column_name: byte_value, target_column_name: ByteValue, expected_data_type: byte, parser: byte}
  - {source_column_name: short_value, target_column_name: ShortValue, expected_data_type: short, parser: short}
  - {source_column_name: float_value, target_column_name: FloatValue, expected_data_type: float, parser: float}
  - source_column_name: local_time
    target_column_name: LocalTime
    expected_data_type: timestamp_ntz
    parser: {type: timestamp_ntz, formats: ["MM/dd/yyyy h:mm a"]}
  - source_column_name: hex_value
    target_column_name: HexValue
    expected_data_type: binary
    parser: {type: binary, encoding: hex, audit: true}
  - source_column_name: base64_value
    target_column_name: Base64Value
    expected_data_type: binary
    parser: binary
  - source_column_name: utf8_value
    target_column_name: Utf8Value
    expected_data_type: binary
    parser: {type: binary, encoding: utf8}
"""
    )
    df = spark.sql(
        "SELECT '127' byte_value, '32767' short_value, '1.25' float_value, "
        "'08/27/2026 02:30 PM' local_time, '4869' hex_value, "
        "'SGk=' base64_value, 'Hi' utf8_value"
    )

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["byte_value"])
    row = parsing.parsed_df.first()
    assert row.ByteValue == 127
    assert row.ShortValue == 32767
    assert row.FloatValue == pytest.approx(1.25)
    assert str(row.LocalTime) == "2026-08-27 14:30:00"
    assert bytes(row.HexValue) == b"Hi"
    assert bytes(row.Base64Value) == b"Hi"
    assert bytes(row.Utf8Value) == b"Hi"
    audit = parsing.results_df.first().spark_parser_parse_results[0]
    assert audit.parsed_value == "SGk="


def test_integral_boundaries_reject_byte_wraparound(spark: SparkSession) -> None:
    """Keep signed byte bounds identical at top level and inside arrays."""
    compiler = YamlParserConfigCompiler()
    config = compiler.compile_text(
        """
parser_config_id: integral_boundaries
parser_config_name: Integral Boundaries
version: "1"
columns:
  - source_column_name: scalar
    target_column_name: ByteValue
    expected_data_type: byte
    parser: {type: byte, on_parse_error: null}
  - source_column_name: scalar
    target_column_name: ShortValue
    expected_data_type: short
    parser: {type: short, on_parse_error: null}
  - source_column_name: nested
    target_column_name: NestedByte
    expected_data_type: array<byte>
    parser: {type: array, element_parser: byte, on_element_error: null}
"""
    )
    tokens = ["127", "128", "200", "255", "256", "-128", "-129"]
    rows = (
        SparkDataFrameParser()
        .parse_dataframe(
            spark.range(len(tokens)).select(
                F.col("id").cast("integer").alias("row_id"),
                F.element_at(
                    F.array(*(F.lit(token) for token in tokens)),
                    (F.col("id") + 1).cast("integer"),
                ).alias("scalar"),
                F.element_at(
                    F.array(*(F.lit(f'["{token}"]') for token in tokens)),
                    (F.col("id") + 1).cast("integer"),
                ).alias("nested"),
            ),
            config,
            key_columns=["row_id"],
        )
        .parsed_df.orderBy("row_id")
        .collect()
    )
    assert [row.ByteValue for row in rows] == [127, None, None, None, None, -128, None]
    assert [row.NestedByte[0] for row in rows] == [
        127,
        None,
        None,
        None,
        None,
        -128,
        None,
    ]
    assert [row.ShortValue for row in rows] == [127, 128, 200, 255, 256, -128, -129]

    failing = SparkDataFrameParser().parse_dataframe(
        spark.sql("SELECT '128' AS value"),
        compiler.compile_text(
            """
parser_config_id: byte_fail
parser_config_name: Byte Fail
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: byte
    parser: byte
"""
        ),
        key_columns=["value"],
    )
    with pytest.raises((Py4JJavaError, PySparkException)):
        failing.parsed_df.collect()


def test_nested_fail_policy_raises_when_materialized(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: nested_fail
parser_config_name: Nested Fail
version: "1"
columns:
  - source_column_name: values
    target_column_name: Values
    expected_data_type: array<integer>
    parser: {type: array, element_parser: integer, on_element_error: fail}
"""
    )
    parsing = SparkDataFrameParser().parse_dataframe(
        spark.sql("SELECT '[1,\"bad\"]' AS values"),
        config,
        key_columns=["values"],
    )

    with pytest.raises((Py4JJavaError, PySparkException)) as exc_info:
        parsing.parsed_df.collect()
    message = str(exc_info.value)
    assert "source 'values'" in message
    assert "target column 'Values'" in message
    assert "$[1]" in message


def test_arbitrarily_nested_containers_fail_at_the_exact_child_path(
    spark: SparkSession,
) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: recursive
parser_config_name: Recursive
version: "1"
columns:
  - source_column_name: payload
    target_column_name: Payload
    expected_data_type: array<struct<name:string,scores:array<integer>>>
    parser:
      type: array
      on_element_error: drop
      element_parser:
        type: struct
        fields:
          - {source_field_name: raw_name, target_field_name: name, parser: {type: string, format: upper}}
          - source_field_name: raw_scores
            target_field_name: scores
            parser: {type: array, element_parser: integer, on_element_error: null}
      audit: true
  - source_column_name: nested_map
    target_column_name: NestedMap
    expected_data_type: map<string,array<integer>>
    parser:
      type: map
      value_parser:
        type: array
        element_parser: integer
        on_element_error: null
      on_value_error: fail
      audit: true
"""
    )
    df = spark.sql(
        'SELECT \'[{"raw_name":"alice","raw_scores":[1,"bad"]},"wrong shape"]\' AS payload, '
        '\'{"x":[1,"bad"]}\' AS nested_map'
    )

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["payload"])
    row = parsing.parsed_df.first().Payload
    assert len(row) == 1
    assert row[0].name == "ALICE"
    assert row[0].scores == [1, None]
    assert parsing.parsed_df.first().NestedMap["x"] == [1, None]
    audits = parsing.results_df.first().spark_parser_parse_results
    assert audits[0].nested_error_paths == ["$[0].scores[1]", "$[1]"]
    assert audits[1].nested_error_paths == ["$['x'][1]"]


def test_json_null_is_a_successful_complex_null(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: json_null
parser_config_name: JSON Null
version: "1"
columns:
  - source_column_name: payload
    target_column_name: Payload
    expected_data_type: array<string>
    parser:
      type: array
      element_parser: string
      is_nullable: false
      default_on_null: []
      audit: true
"""
    )
    parsing = SparkDataFrameParser().parse_dataframe(
        spark.sql("SELECT 'null' AS payload"),
        config,
        key_columns=["payload"],
    )

    assert parsing.parsed_df.first().Payload == []
    audit = parsing.results_df.first().spark_parser_parse_results[0]
    assert audit.actions_applied == ["json_null_to_null", "default_on_null_applied"]
    assert audit.error is None


def test_timestamp_ntz_honors_error_policies_under_ansi(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: timestamp_ntz_errors
parser_config_name: Timestamp NTZ Errors
version: "1"
columns:
  - source_column_name: local_time
    target_column_name: LocalTime
    expected_data_type: timestamp_ntz
    parser:
      type: timestamp_ntz
      formats: [MM/dd/yyyy HH:mm]
      on_parse_error: null
      audit: true
  - source_column_name: local_times
    target_column_name: LocalTimes
    expected_data_type: array<timestamp_ntz>
    parser:
      type: array
      element_parser: {type: timestamp_ntz, formats: [MM/dd/yyyy HH:mm]}
      on_element_error: null
      audit: true
"""
    )
    df = spark.range(1).select(
        F.lit("not-a-time").alias("local_time"),
        F.lit('["08/27/2026 14:30","bad"]').alias("local_times"),
    )

    previous_policy = spark.conf.get("spark.sql.legacy.timeParserPolicy")
    try:
        spark.conf.set("spark.sql.legacy.timeParserPolicy", "EXCEPTION")
        with pytest.raises(SchemaValidationError, match="require.*CORRECTED"):
            SparkDataFrameParser().parse_dataframe(
                df,
                config,
                key_columns=["local_time"],
            )

        spark.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        parsing = SparkDataFrameParser().parse_dataframe(
            df,
            config,
            key_columns=["local_time"],
        )
        row = parsing.parsed_df.first()
        assert row.LocalTime is None
        assert str(row.LocalTimes[0]) == "2026-08-27 14:30:00"
        assert row.LocalTimes[1] is None
        audits = parsing.results_df.first().spark_parser_parse_results
        assert audits[0].actions_applied == ["parse_error_to_null"]
        assert audits[1].nested_error_paths == ["$[1]"]
    finally:
        spark.conf.set("spark.sql.legacy.timeParserPolicy", previous_policy)


def test_nested_numeric_parser_rejects_json_wrapper_injection(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: nested_numeric_guard
parser_config_name: Nested Numeric Guard
version: "1"
columns:
  - source_column_name: numbers
    target_column_name: Numbers
    expected_data_type: array<integer>
    parser:
      type: array
      element_parser: integer
      on_element_error: null
      audit: true
  - source_column_name: rates
    target_column_name: Rates
    expected_data_type: array<double>
    parser:
      type: array
      input_format: delimited
      delimiter: ";"
      element_parser: double
      on_element_error: drop
"""
    )
    df = spark.range(1).select(
        F.lit('["1,\\"value\\":2"]').alias("numbers"),
        F.lit("NaN;Infinity;-Infinity;1.5").alias("rates"),
    )

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["numbers"])
    row = parsing.parsed_df.first()
    assert row.Numbers == [None]
    assert row.Rates == [1.5]
    audit = parsing.results_df.first().spark_parser_parse_results[0]
    assert audit.nested_error_paths == ["$[0]"]


def test_strict_json_validation_and_duplicate_map_keys_follow_error_policy(
    spark: SparkSession,
) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: strict_json
parser_config_name: Strict JSON
version: "1"
columns:
  - source_column_name: object
    target_column_name: Object
    expected_data_type: struct<a:integer>
    parser:
      type: struct
      fields:
        - {source_field_name: a, target_field_name: a, parser: integer}
      on_parse_error: null
      audit: true
  - source_column_name: empty_array
    target_column_name: EmptyArray
    expected_data_type: array<string>
    parser: {type: array, element_parser: string}
  - source_column_name: empty_map
    target_column_name: EmptyMap
    expected_data_type: map<string,string>
    parser: {type: map, value_parser: string}
  - source_column_name: duplicate_map
    target_column_name: DuplicateMap
    expected_data_type: map<string,integer>
    parser:
      type: map
      value_parser: integer
      on_parse_error: null
      audit: true
  - source_column_name: duplicate_map
    target_column_name: DefaultedDuplicateMap
    expected_data_type: map<string,integer>
    parser:
      type: map
      value_parser: integer
      on_parse_error: default
      default_on_error: {}
  - source_column_name: nested_duplicate_map
    target_column_name: NestedDuplicateMap
    expected_data_type: array<map<string,integer>>
    parser:
      type: array
      element_parser: {type: map, value_parser: integer}
      on_element_error: null
      audit: true
"""
    )
    df = spark.range(1).select(
        F.lit("{'a':1}").alias("object"),
        F.lit("[]").alias("empty_array"),
        F.lit("{}").alias("empty_map"),
        F.lit('{"a":1,"a":2}').alias("duplicate_map"),
        F.lit('[{"a":1,"a":2}]').alias("nested_duplicate_map"),
    )

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["object"])
    row = parsing.parsed_df.first()
    assert row.Object is None
    assert row.EmptyArray == []
    assert row.EmptyMap == {}
    assert row.DuplicateMap is None
    assert row.DefaultedDuplicateMap == {}
    assert row.NestedDuplicateMap == [None]
    audits = {
        item.target_column_name: item
        for item in parsing.results_df.first().spark_parser_parse_results
    }
    assert audits["Object"].nested_error_paths == ["$"]
    assert audits["Object"].actions_applied == ["parse_error_to_null"]
    assert audits["DuplicateMap"].nested_error_paths == ["$"]
    assert audits["DuplicateMap"].actions_applied == ["parse_error_to_null"]
    assert audits["NestedDuplicateMap"].nested_error_paths == ["$[0]"]


def test_nested_delimited_arrays_honor_their_input_format(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: nested_delimited
parser_config_name: Nested Delimited
version: "1"
columns:
  - source_column_name: payload
    target_column_name: Payload
    expected_data_type: struct<tags:array<string>>
    parser:
      type: struct
      fields:
        - source_field_name: raw_tags
          target_field_name: tags
          parser:
            type: array
            input_format: delimited
            delimiter: "|"
            element_parser: {type: string, format: upper}
      audit: true
"""
    )
    df = spark.range(1).select(F.lit('{"raw_tags":"a|b|c"}').alias("payload"))

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["payload"])
    assert parsing.parsed_df.first().Payload.tags == ["A", "B", "C"]
    audit = parsing.results_df.first().spark_parser_parse_results[0]
    assert audit.nested_error_paths == []


def test_only_lowercase_json_null_is_the_json_literal(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: json_null_case
parser_config_name: JSON Null Case
version: "1"
columns:
  - source_column_name: payload
    target_column_name: Payload
    expected_data_type: array<string>
    parser:
      type: array
      element_parser: string
      is_nullable: false
      default_on_null: [NULL_DEFAULT]
      on_parse_error: default
      default_on_error: [ERROR_DEFAULT]
      audit: true
"""
    )
    df = spark.range(1).select(
        F.lit(1).alias("id"),
        F.lit("null").alias("payload"),
    ).union(
        spark.range(1).select(
            F.lit(2).alias("id"),
            F.lit("NULL").alias("payload"),
        )
    )

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["id"])
    assert [row.Payload for row in parsing.parsed_df.collect()] == [
        ["NULL_DEFAULT"],
        ["ERROR_DEFAULT"],
    ]
    audits = {
        row.id: row.spark_parser_parse_results[0]
        for row in parsing.results_df.collect()
    }
    assert audits[1].actions_applied == ["json_null_to_null", "default_on_null_applied"]
    assert audits[2].actions_applied == ["parse_error_default_applied"]
