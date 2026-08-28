"""Small native-Spark smoke test for the first-round runtime contract."""

import os
import shutil

import pytest
from py4j.protocol import Py4JJavaError

from spark_parser import SparkDataFrameParser, YamlParserConfigCompiler

pyspark = pytest.importorskip("pyspark")
from pyspark.errors import PySparkException  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    """Use an active Databricks session or a local session when Java exists."""
    active = SparkSession.getActiveSession()
    if active is not None:
        yield active
        return
    if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
        pytest.skip("A Java runtime is required for the Spark execution smoke test.")
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
  '123456' AS zip_code
"""
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


def test_address_county_and_zip_edge_cases_under_ansi(spark: SparkSession) -> None:
    assert spark.conf.get("spark.sql.ansi.enabled") == "true"
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: location_edges
parser_config_name: Location Edges
version: "1"
columns:
  - source_column_name: address
    silver_column_name: Address
    expected_data_type: string
    parser:
      type: string
      format: address_us_v1
      audit: true
  - source_column_name: address
    silver_column_name: AddressRequired
    expected_data_type: string
    parser:
      type: string
      format: address_us_v1
      is_nullable: false
      default_on_null: UNKNOWN
  - source_column_name: county
    silver_column_name: County
    expected_data_type: string
    parser:
      type: string
      format: county
      on_parse_error: null
  - source_column_name: zip_code
    silver_column_name: ZipCode
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
    silver_column_name: DisplayLabel
    expected_data_type: string
    parser:
      type: string
      format: title
  - source_column_name: state
    silver_column_name: StateCode
    expected_data_type: string
    parser:
      type: string
      format: state_us
      on_parse_error: null
      audit: true
  - source_column_name: state
    silver_column_name: RequiredStateCode
    expected_data_type: string
    parser:
      type: string
      format: state_us
      on_parse_error: default
      default_on_error: UNKNOWN
      audit: true
  - source_column_name: state
    silver_column_name: PreservedStateCode
    expected_data_type: string
    parser:
      type: string
      format: state_us
      on_parse_error: preserve
      audit: true
"""
    )
    bronze_df = spark.createDataFrame(
        [
            (1, "  LOAN   status ", "Illinois"),
            (2, "mixed CASE", "  new   york "),
            (3, "account owner", "il"),
            (4, "district record", "District of Columbia"),
            (5, "invalid state", "  Mul  "),
        ],
        "row_id integer, label string, state string",
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
        "Invalid State",
    ]
    assert [row.StateCode for row in rows] == ["IL", "NY", "IL", "DC", None]
    assert [row.RequiredStateCode for row in rows] == [
        "IL",
        "NY",
        "IL",
        "DC",
        "UNKNOWN",
    ]
    assert [row.PreservedStateCode for row in rows] == [
        "IL",
        "NY",
        "IL",
        "DC",
        "  Mul  ",
    ]

    invalid_audit = parsing.results_df.collect()[-1].spark_parser_parse_results
    assert invalid_audit[0].actions_applied == ["parse_error_to_null"]
    assert invalid_audit[1].actions_applied == ["parse_error_default_applied"]
    assert invalid_audit[2].actions_applied == ["parse_error_preserved"]
    assert invalid_audit[2].original_value == "  Mul  "
    assert invalid_audit[2].parsed_value == "  Mul  "
    assert invalid_audit[2].changed is True


def test_nested_string_error_policies_can_preserve_raw_tokens(spark: SparkSession) -> None:
    """Preserve invalid formatted strings inside structs, arrays, and maps without losing paths."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: nested_preserve
parser_config_name: Nested Preserve
version: "1"
columns:
  - source_column_name: profile
    silver_column_name: Profile
    expected_data_type: struct<state:string>
    parser:
      type: struct
      fields:
        - source_field_name: state
          silver_field_name: state
          parser: {type: string, format: state_us, on_parse_error: preserve}
      audit: true
  - source_column_name: states
    silver_column_name: States
    expected_data_type: array<string>
    parser:
      type: array
      element_parser: {type: string, format: state_us}
      on_element_error: preserve
      audit: true
  - source_column_name: state_map
    silver_column_name: StateMap
    expected_data_type: map<string,string>
    parser:
      type: map
      value_parser: {type: string, format: state_us}
      on_value_error: preserve
      audit: true
"""
    )
    bronze_df = spark.createDataFrame(
        [
            (
                1,
                '{"state":"Mul"}',
                '["Illinois","Mul","ny"]',
                '{"home":"Illinois","other":"Mul"}',
            )
        ],
        "row_id integer, profile string, states string, state_map string",
    )

    parsing = SparkDataFrameParser().parse_dataframe(bronze_df, config, key_columns=["row_id"])
    row = parsing.parsed_df.first()
    assert row.Profile.state == "Mul"
    assert row.States == ["IL", "Mul", "NY"]
    assert row.StateMap == {"home": "IL", "other": "Mul"}

    audits = {
        item.silver_column_name: item
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
    silver_column_name: EventDate
    expected_data_type: date
    parser:
      type: date
      on_parse_error: null
  - source_column_name: event_timestamp
    silver_column_name: EventTimestamp
    expected_data_type: timestamp
    parser:
      type: timestamp
      on_parse_error: null
  - source_column_name: event_timestamp
    silver_column_name: EventTimestampNtz
    expected_data_type: timestamp_ntz
    parser:
      type: timestamp_ntz
      on_parse_error: null
"""
    )
    bronze_df = spark.createDataFrame(
        [
            (1, "2026-09-29", "2026-09-29 01:02:03"),
            (2, "09/30/2026 12:00 AM", "09/30/2026 12:00 AM"),
            (3, "09/30/2026 12:00:00 AM", "09/30/2026 12:00:00 AM"),
            # A bare slash date remains invalid. Accepting it would silently guess whether the
            # source uses month/day or day/month ordering.
            (4, "09/10/2026", "09/10/2026"),
        ],
        "row_id integer, event_date string, event_timestamp string",
    )

    rows = (
        SparkDataFrameParser()
        .parse_dataframe(bronze_df, config)
        .parsed_df.orderBy("row_id")
        .collect()
    )
    assert rows[0].EventDate.isoformat() == "2026-09-29"
    assert rows[1].EventDate.isoformat() == "2026-09-30"
    assert rows[2].EventDate.isoformat() == "2026-09-30"
    assert rows[3].EventDate is None
    assert str(rows[0].EventTimestamp) == "2026-09-29 01:02:03"
    assert str(rows[1].EventTimestamp) == "2026-09-30 00:00:00"
    assert str(rows[2].EventTimestamp) == "2026-09-30 00:00:00"
    assert rows[3].EventTimestamp is None
    assert str(rows[0].EventTimestampNtz) == "2026-09-29 01:02:03"
    assert str(rows[1].EventTimestampNtz) == "2026-09-30 00:00:00"
    assert str(rows[2].EventTimestampNtz) == "2026-09-30 00:00:00"
    assert rows[3].EventTimestampNtz is None


def test_audit_schema_is_stable_with_or_without_audited_columns(spark: SparkSession) -> None:
    template = """
parser_config_id: audit_schema
parser_config_name: Audit Schema
version: "1"
columns:
  - source_column_name: value
    silver_column_name: Value
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
    assert audited.results_df.first().spark_parser_parse_results
    assert empty.results_df.first().spark_parser_parse_results == []


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
    silver_column_name: Value
    expected_data_type: integer
    parser: integer
"""
    )
    failing = SparkDataFrameParser().parse_dataframe(
        spark.sql("SELECT 'abc' AS value"),
        fail_config,
        key_columns=["value"],
    )
    assert failing.parsed_df.count() == 1
    with pytest.raises((Py4JJavaError, PySparkException)):
        failing.parsed_df.collect()

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
    silver_column_name: DefaultedInteger
    expected_data_type: integer
    parser:
      type: integer
      on_parse_error: default
      default_on_error: -1
      audit: true
  - source_column_name: boolean_value
    silver_column_name: BooleanValue
    expected_data_type: boolean
    parser:
      type: boolean
      on_parse_error: null
  - source_column_name: trim_value
    silver_column_name: TrimValue
    expected_data_type: string
    parser:
      type: string
      collapse_whitespace: false
      trim_whitespace: true
  - source_column_name: nbsp_value
    silver_column_name: NbspValue
    expected_data_type: string
    parser: string
  - source_column_name: decimal_value
    silver_column_name: DecimalValue
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


def test_wide_config_uses_constant_depth_projection_stages(spark: SparkSession) -> None:
    column_count = 40
    df = spark.sql("SELECT " + ", ".join(f"'x' AS c{index}" for index in range(column_count)))
    columns = "\n".join(
        f"  - source_column_name: c{index}\n"
        f"    silver_column_name: C{index}\n"
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

    analyzed = parsing.parsed_df._jdf.queryExecution().analyzed().treeString()
    assert analyzed.count("Project") <= 10


def test_timestamp_options_use_iso_text() -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: timestamp_option
parser_config_name: Timestamp Option
version: "1"
columns:
  - source_column_name: value
    silver_column_name: Value
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
    silver_column_name: Names
    expected_data_type: array<string>
    parser:
      type: array
      element_parser: {type: string, format: upper}
      on_element_error: drop
      drop_null_elements: true
      distinct: true
      audit: true
  - source_column_name: object
    silver_column_name: Object
    expected_data_type: struct<street:string,zip:string,scores:array<integer>,approved:boolean,due:date>
    parser:
      type: struct
      fields:
        - source_field_name: address
          silver_field_name: street
          parser: {type: string, format: address_us_v1}
        - source_field_name: postal
          silver_field_name: zip
          parser: {type: string, format: zip, on_parse_error: null}
        - source_field_name: values
          silver_field_name: scores
          parser: {type: array, element_parser: integer, on_element_error: null}
        - {source_field_name: is_approved, silver_field_name: approved, parser: boolean}
        - source_field_name: due_date
          silver_field_name: due
          parser: {type: date, formats: [MM/dd/yyyy]}
      audit: true
  - source_column_name: attributes
    silver_column_name: Attributes
    expected_data_type: map<string,decimal(8,2)>
    parser: {type: map, value_parser: decimal, on_value_error: drop, audit: true}
"""
    )
    df = spark.sql(
        """
SELECT
  1 AS id,
  '[" alice ","BOB",null,"alice"]' AS names,
  '{"address":"123 mccormick st. apt #4b","postal":"1234","values":[1,"bad",3],"is_approved":"y","due_date":"08/27/2026"}' AS object,
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
        item.silver_column_name: item
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
    silver_column_name: Numbers
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
    silver_column_name: Object
    expected_data_type: struct<a:integer>
    parser:
      type: struct
      fields:
        - {source_field_name: a, silver_field_name: a, parser: integer}
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
        item.silver_column_name: item
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
  - {source_column_name: byte_value, silver_column_name: ByteValue, expected_data_type: byte, parser: byte}
  - {source_column_name: short_value, silver_column_name: ShortValue, expected_data_type: short, parser: short}
  - {source_column_name: float_value, silver_column_name: FloatValue, expected_data_type: float, parser: float}
  - source_column_name: local_time
    silver_column_name: LocalTime
    expected_data_type: timestamp_ntz
    parser: {type: timestamp_ntz, formats: [MM/dd/yyyy HH:mm]}
  - source_column_name: hex_value
    silver_column_name: HexValue
    expected_data_type: binary
    parser: {type: binary, encoding: hex, audit: true}
  - source_column_name: base64_value
    silver_column_name: Base64Value
    expected_data_type: binary
    parser: binary
  - source_column_name: utf8_value
    silver_column_name: Utf8Value
    expected_data_type: binary
    parser: {type: binary, encoding: utf8}
"""
    )
    df = spark.sql(
        "SELECT '127' byte_value, '32767' short_value, '1.25' float_value, "
        "'08/27/2026 14:30' local_time, '4869' hex_value, "
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


def test_nested_fail_policy_raises_when_materialized(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: nested_fail
parser_config_name: Nested Fail
version: "1"
columns:
  - source_column_name: values
    silver_column_name: Values
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
    assert "silver column 'Values'" in message
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
    silver_column_name: Payload
    expected_data_type: array<struct<name:string,scores:array<integer>>>
    parser:
      type: array
      on_element_error: drop
      element_parser:
        type: struct
        fields:
          - {source_field_name: raw_name, silver_field_name: name, parser: {type: string, format: upper}}
          - source_field_name: raw_scores
            silver_field_name: scores
            parser: {type: array, element_parser: integer, on_element_error: null}
      audit: true
  - source_column_name: nested_map
    silver_column_name: NestedMap
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
    silver_column_name: Payload
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
    silver_column_name: LocalTime
    expected_data_type: timestamp_ntz
    parser:
      type: timestamp_ntz
      formats: [MM/dd/yyyy HH:mm]
      on_parse_error: null
      audit: true
  - source_column_name: local_times
    silver_column_name: LocalTimes
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

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["local_time"])
    row = parsing.parsed_df.first()
    assert row.LocalTime is None
    assert str(row.LocalTimes[0]) == "2026-08-27 14:30:00"
    assert row.LocalTimes[1] is None
    audits = parsing.results_df.first().spark_parser_parse_results
    assert audits[0].actions_applied == ["parse_error_to_null"]
    assert audits[1].nested_error_paths == ["$[1]"]


def test_nested_numeric_parser_rejects_json_wrapper_injection(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: nested_numeric_guard
parser_config_name: Nested Numeric Guard
version: "1"
columns:
  - source_column_name: numbers
    silver_column_name: Numbers
    expected_data_type: array<integer>
    parser:
      type: array
      element_parser: integer
      on_element_error: null
      audit: true
  - source_column_name: rates
    silver_column_name: Rates
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
    silver_column_name: Object
    expected_data_type: struct<a:integer>
    parser:
      type: struct
      fields:
        - {source_field_name: a, silver_field_name: a, parser: integer}
      on_parse_error: null
      audit: true
  - source_column_name: empty_array
    silver_column_name: EmptyArray
    expected_data_type: array<string>
    parser: {type: array, element_parser: string}
  - source_column_name: empty_map
    silver_column_name: EmptyMap
    expected_data_type: map<string,string>
    parser: {type: map, value_parser: string}
  - source_column_name: duplicate_map
    silver_column_name: DuplicateMap
    expected_data_type: map<string,integer>
    parser:
      type: map
      value_parser: integer
      on_parse_error: null
      audit: true
  - source_column_name: duplicate_map
    silver_column_name: DefaultedDuplicateMap
    expected_data_type: map<string,integer>
    parser:
      type: map
      value_parser: integer
      on_parse_error: default
      default_on_error: {}
  - source_column_name: nested_duplicate_map
    silver_column_name: NestedDuplicateMap
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
        item.silver_column_name: item
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
    silver_column_name: Payload
    expected_data_type: struct<tags:array<string>>
    parser:
      type: struct
      fields:
        - source_field_name: raw_tags
          silver_field_name: tags
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
    silver_column_name: Payload
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
