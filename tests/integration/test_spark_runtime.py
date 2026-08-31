"""Native-Spark behavioral tests for the complete runtime contract."""

import importlib.util
import json
import os
import shutil
import sys
from unittest.mock import Mock

import pytest

from spark_parser import (
    DataFrameParsing,
    SchemaValidationError,
    SparkDataFrameParser,
    YamlParserConfigCompiler,
    parser,
)

if importlib.util.find_spec("pyspark") is None and os.environ.get("SPARK_PARSER_REQUIRE_JAVA") == "1":
    pytest.fail(
        "SPARK_PARSER_REQUIRE_JAVA=1, but PySpark is not installed",
        pytrace=False,
    )
pyspark = pytest.importorskip("pyspark")
from py4j.protocol import Py4JJavaError  # noqa: E402
from pyspark import StorageLevel  # noqa: E402
from pyspark.errors import PySparkException, PySparkNotImplementedError  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.utils import is_remote  # noqa: E402

pytestmark = pytest.mark.spark

_PORTABLE_TEST_SESSION_SETTINGS = {
    "spark.sql.ansi.enabled": "true",
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.legacy.timeParserPolicy": "CORRECTED",
}


def _connect_mode_requested() -> bool:
    """Recognize configured Connect clients even before they register an active session."""
    return (
        is_remote()
        or "SPARK_REMOTE" in os.environ
        or os.environ.get("SPARK_API_MODE", "").lower() == "connect"
        or os.environ.get("MASTER", "").startswith("sc://")
    )


@pytest.fixture(scope="module")
def spark():
    """Use an active Databricks session or a local session when Java exists."""
    active = SparkSession.getActiveSession()
    if active is not None:
        previous_settings = {
            key: active.conf.get(key) for key in _PORTABLE_TEST_SESSION_SETTINGS
        }
        applied_settings: list[str] = []
        try:
            for key, value in _PORTABLE_TEST_SESSION_SETTINGS.items():
                active.conf.set(key, value)
                applied_settings.append(key)
            yield active
        finally:
            for key in reversed(applied_settings):
                active.conf.set(key, previous_settings[key])
        return
    if _connect_mode_requested():
        pytest.fail(
            "Spark Connect/serverless integration tests require an ambient active SparkSession",
            pytrace=False,
        )
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
    builder = SparkSession.builder.master("local[1]").appName("spark-parser-test")
    for key, value in _PORTABLE_TEST_SESSION_SETTINGS.items():
        builder = builder.config(key, value)
    session = builder.getOrCreate()
    try:
        yield session
    finally:
        session.stop()


def _is_connect_session(session: SparkSession) -> bool:
    """Identify Connect without reading a runtime configuration key."""
    return _connect_mode_requested() or type(session).__module__.startswith(
        "pyspark.sql.connect."
    )


@pytest.fixture
def classic_spark(spark: SparkSession) -> SparkSession:
    """Provide a session only for tests that deliberately require classic driver internals."""
    if _is_connect_session(spark):
        pytest.skip("requires classic Spark driver configuration or SparkContext access")
    return spark


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


def test_public_parser_facade_builds_shared_lazy_projections(spark: SparkSession) -> None:
    """Exercise the documented high-level entry point without optional cache APIs."""
    df = spark.range(1).select(
        F.col("id").alias("row_id"),
        F.lit("42").alias("value"),
    )
    parsing = parser.parse_dataframe(
        df,
        {
            "parser_config_id": "public_facade",
            "parser_config_name": "Public Facade",
            "version": "1",
            "columns": [
                {
                    "source_column_name": "value",
                    "target_column_name": "Value",
                    "expected_data_type": "integer",
                    "parser": "integer",
                }
            ],
        },
        key_columns=["row_id"],
    )

    assert isinstance(parsing, DataFrameParsing)
    assert parsing.parsed_df.first().Value == 42
    assert parsing.results_df.first().row_id == 0


def test_dataframe_parsing_persistence_delegates_without_requiring_cache_support() -> None:
    """Verify the adapter contract without invoking cache APIs forbidden by serverless Spark."""
    evaluated = Mock()
    parsing = DataFrameParsing(
        evaluated,
        parsed_columns=(),
        key_columns=(),
        result_columns=(),
    )

    assert parsing.persist() is parsing
    evaluated.persist.assert_called_once_with(StorageLevel.MEMORY_AND_DISK_DESER)
    evaluated.persist.reset_mock()
    assert parsing.persist(StorageLevel.DISK_ONLY) is parsing
    evaluated.persist.assert_called_once_with(StorageLevel.DISK_ONLY)

    assert parsing.unpersist() is parsing
    evaluated.unpersist.assert_called_once_with(blocking=False)
    evaluated.unpersist.reset_mock()
    assert parsing.unpersist(blocking=True) is parsing
    evaluated.unpersist.assert_called_once_with(blocking=True)

    unavailable = RuntimeError("DataFrame cache APIs are unavailable")
    evaluated.persist.side_effect = unavailable
    with pytest.raises(RuntimeError) as captured:
        parsing.persist()
    assert captured.value is unavailable

    release_unavailable = RuntimeError("DataFrame cache release is unavailable")
    evaluated.unpersist.side_effect = release_unavailable
    with pytest.raises(RuntimeError) as captured:
        parsing.unpersist()
    assert captured.value is release_unavailable


@pytest.mark.classic_spark
def test_classic_persistence_caches_the_complete_shared_plan(
    classic_spark: SparkSession,
) -> None:
    """Lock cache state and full-plan evaluation semantics on cache-capable Spark."""
    spark = classic_spark
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: shared_plan_cache
parser_config_name: Shared Plan Cache
version: "1"
columns:
  - source_column_name: safe
    target_column_name: Safe
    expected_data_type: string
    parser: string
  - source_column_name: bad
    target_column_name: Bad
    expected_data_type: integer
    parser: integer
"""
    )
    parsing = SparkDataFrameParser().parse_dataframe(
        spark.sql("SELECT 'ok' AS safe, 'not-an-integer' AS bad"),
        config,
        key_columns=["safe"],
    )

    assert parsing._evaluated.storageLevel == StorageLevel.NONE
    assert parsing.parsed_df.select("Safe").first().Safe == "ok"
    assert parsing.persist() is parsing
    try:
        assert parsing._evaluated.storageLevel == StorageLevel.MEMORY_AND_DISK_DESER
        with pytest.raises((Py4JJavaError, PySparkException)) as exc_info:
            parsing.parsed_df.select("Safe").collect()
        message = str(exc_info.value)
        assert "source 'bad'" in message
        assert "target column 'Bad'" in message
    finally:
        parsing.unpersist(blocking=True)
    assert parsing._evaluated.storageLevel == StorageLevel.NONE

    assert parsing.persist(StorageLevel.DISK_ONLY) is parsing
    try:
        assert parsing._evaluated.storageLevel == StorageLevel.DISK_ONLY
    finally:
        parsing.unpersist(blocking=True)
    assert parsing._evaluated.storageLevel == StorageLevel.NONE


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


def test_interest_rate_index_v1_canonicalizes_the_approved_corpus(
    spark: SparkSession,
) -> None:
    """Canonicalize approved index families, tenors, abbreviations, and exact source codes."""
    cases = [
        ("SOFR Term - Overnight", "SOFR Term Overnight"),
        ("SOFR Term - 1 Month", "SOFR Term 1-Month"),
        ("SOFR Term - 3 Month", "SOFR Term 3-Month"),
        ("SOFR Term - 6 Month", "SOFR Term 6-Month"),
        ("SOFR Term - 12 Month", "SOFR Term 12-Month"),
        ("SOFR Term 12 Month", "SOFR Term 12-Month"),
        ("1 Year Constant Maturity Treasury (CMT)", "1-Year Constant Maturity Treasury (CMT)"),
        ("2 Year Constant Maturity Treasury (CMT)", "2-Year Constant Maturity Treasury (CMT)"),
        ("3 Year Constant Maturity Treasury (CMT)", "3-Year Constant Maturity Treasury (CMT)"),
        ("5 Year Constant Maturity Treasury (CMT)", "5-Year Constant Maturity Treasury (CMT)"),
        ("7 Year Constant Maturity Treasury (CMT)", "7-Year Constant Maturity Treasury (CMT)"),
        ("10 Year Constant Maturity Treasury (CMT)", "10-Year Constant Maturity Treasury (CMT)"),
        ("30 Year Constant Maturity Treasury (CMT)", "30-Year Constant Maturity Treasury (CMT)"),
        ("BSBY - Overnight", "BSBY Overnight"),
        ("BSBY - 1 Month", "BSBY 1-Month"),
        ("BSBY - 3 Month", "BSBY 3-Month"),
        ("BSBY - 6 Month", "BSBY 6-Month"),
        ("BSBY - 12 Month", "BSBY 12-Month"),
        ("Ameribor - Overnight", "Ameribor Overnight"),
        ("Ameribor - 1 Week", "Ameribor 1-Week"),
        ("Ameribor - 1 Month", "Ameribor 1-Month"),
        ("Ameribor - 3 Month", "Ameribor 3-Month"),
        ("Ameribor - 6 Month", "Ameribor 6-Month"),
        ("Ameribor - 1 Year", "Ameribor 1-Year"),
        ("Ameribor - 2 Year", "Ameribor 2-Year"),
        ("Ameribor - 30 Day Average", "Ameribor 30-Day Average"),
        ("Ameribor - 90 Day Average", "Ameribor 90-Day Average"),
        ("Ameribor - Derived 30T", "Ameribor Derived 30T"),
        ("Ameribor - Derived 90T", "Ameribor Derived 90T"),
        ("Prime", "Prime"),
        ("1 Month LIBOR", "1-Month LIBOR"),
        ("2 Month LIBOR", "2-Month LIBOR"),
        ("3 Month LIBOR", "3-Month LIBOR"),
        ("6 Month LIBOR", "6-Month LIBOR"),
        ("9 Month LIBOR", "9-Month LIBOR"),
        ("12 Month LIBOR", "12-Month LIBOR"),
        ("Treasury - 3 mos", "Treasury 3-Month"),
        ("Treasury - 6 mos", "Treasury 6-Month"),
        ("Treasury - 1 yr", "Treasury 1-Year"),
        ("Treasury - 2 yr", "Treasury 2-Year"),
        ("Treasury - 3 yr", "Treasury 3-Year"),
        ("Treasury - 5 yr", "Treasury 5-Year"),
        ("Treasury - 7 yr", "Treasury 7-Year"),
        ("Treasury - 10 yr", "Treasury 10-Year"),
        ("Treasury - 30 yr", "Treasury 30-Year"),
        ("Treasury 1Y", "Treasury 1-Year"),
        ("Treasury Avg - 12 mos", "Treasury Avg 12-Month"),
        ("USD Swap 1 Year", "USD Swap 1-Year"),
        ("USD Swap 5 Year", "USD Swap 5-Year"),
        ("USD Swap 10 Year", "USD Swap 10-Year"),
        ("Freddie Mac - 1 mo", "Freddie Mac 1-Month"),
        ("Freddie Mac - 3 mos", "Freddie Mac 3-Month"),
        ("Freddie Mac - 6 mos", "Freddie Mac 6-Month"),
        ("Freddie Mac - 12 mos", "Freddie Mac 12-Month"),
        ("FHLB - 1 yr", "FHLB 1-Year"),
        ("FHLB - 2 yr", "FHLB 2-Year"),
        ("FHLB - 3 yr", "FHLB 3-Year"),
        ("FHLB - 5 yr", "FHLB 5-Year"),
        ("FHLB - 7 yr", "FHLB 7-Year"),
        ("FHLB - 10 yr", "FHLB 10-Year"),
        ("SOFR 12-Month", "SOFR 12-Month"),
        ("SOFR 12-month", "SOFR 12-Month"),
        ("SOFR 30 day average", "SOFR 30-Day Average"),
        ("TSFR6M", "SOFR Term 6-Month"),
        ("SOFR 12 Month", "SOFR 12-Month"),
        ("SOFR180A", "SOFR 180-Day Average"),
        ("SOFR", "SOFR"),
        ("SOFR 1-Month", "SOFR 1-Month"),
        ("SOFR 1 month", "SOFR 1-Month"),
        ("SOFR 1 MONTH", "SOFR 1-Month"),
        ("SOFR 30 Day Average", "SOFR 30-Day Average"),
        ("SOFR30", "SOFR 30-Day"),
        ("SOFR30A", "SOFR 30-Day Average"),
        ("RCF6M", "RCF 6-Month"),
        ("SOFR - 30 Day", "SOFR 30-Day"),
        ("SOFR 30 Day", "SOFR 30-Day"),
        ("RCF12M", "RCF 12-Month"),
        ("10 yr CMT", "10-Year Constant Maturity Treasury (CMT)"),
        ("LIBOR - 1 Year (Daily)", "LIBOR 1-Year (Daily)"),
        ("SOFR Avg - 30 days", "SOFR 30-Day Average"),
        ("SOFR Term - 12M", "SOFR Term 12-Month"),
        ("SOFR 1M", "SOFR 1-Month"),
        ("BSBY - 12M", "BSBY 12-Month"),
        ("Ameribor - 2Yr", "Ameribor 2-Year"),
        ("1M LIBOR", "1-Month LIBOR"),
        ("Treasury - 1Y", "Treasury 1-Year"),
        ("USD Swap 5yr", "USD Swap 5-Year"),
        ("Freddie Mac - 12M", "Freddie Mac 12-Month"),
        ("FHLB - 10Yr", "FHLB 10-Year"),
        ("10Y CMT", "10-Year Constant Maturity Treasury (CMT)"),
        ("RCF 6M", "RCF 6-Month"),
    ]
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: interest_rate_indexes
parser_config_name: Interest Rate Indexes
version: "1"
columns:
  - source_column_name: financial_index
    target_column_name: FinancialIndex
    expected_data_type: string
    parser:
      type: string
      format: interest_rate_index_v1
"""
    )

    source_df = spark.range(len(cases)).select(
        (F.col("id") + 1).alias("row_id"),
        F.element_at(
            F.array(*(F.lit(source) for source, _ in cases)),
            (F.col("id") + 1).cast("integer"),
        ).alias("financial_index"),
    )
    parsed = SparkDataFrameParser().parse_dataframe(
        source_df,
        config,
        key_columns=["row_id"],
    )
    assert [row.FinancialIndex for row in parsed.parsed_df.orderBy("row_id").collect()] == [
        expected for _, expected in cases
    ]
    assert all(" - " not in expected for _, expected in cases)

    # Every emitted label is itself a canonical input.
    canonical_values = sorted({expected for _, expected in cases})
    canonical_df = spark.range(len(canonical_values)).select(
        (F.col("id") + 1).alias("row_id"),
        F.element_at(
            F.array(*(F.lit(value) for value in canonical_values)),
            (F.col("id") + 1).cast("integer"),
        ).alias("financial_index"),
    )
    reparsed = SparkDataFrameParser().parse_dataframe(
        canonical_df,
        config,
        key_columns=["row_id"],
    )
    assert [row.FinancialIndex for row in reparsed.parsed_df.orderBy("row_id").collect()] == [
        *canonical_values
    ]


def test_title_business_v1_adds_only_bounded_business_title_rules(
    spark: SparkSession,
) -> None:
    """Retain strict title behavior separately from frozen acronyms and numeric hyphens."""
    cases = [
        ("fhlb   advance", "Fhlb   Advance", "FHLB   Advance"),
        ("p&i payment", "P&i Payment", "P&I Payment"),
        ("ust rcf cmt", "Ust Rcf Cmt", "UST RCF CMT"),
        ("12-month rate", "12-month Rate", "12-Month Rate"),
        ("term 30-day", "Term 30-day", "Term 30-Day"),
        ("2-factor authentication", "2-factor Authentication", "2-Factor Authentication"),
        ("3-month/6-month rates", "3-month/6-month Rates", "3-Month/6-Month Rates"),
        ("12-month-1-year", "12-month-1-year", "12-Month-1-Year"),
        ("state-of-the-art", "State-of-the-art", "State-of-the-art"),
        (
            "12- month 12--month 12/month",
            "12- Month 12--month 12/month",
            "12- Month 12--month 12/month",
        ),
        ("12–month section_12-month", "12–month Section_12-month", "12–month Section_12-month"),
        ("trust account", "Trust Account", "Trust Account"),
        ("rcf6m cmt2 p&ix fhlb_code", "Rcf6m Cmt2 P&ix Fhlb_code", "Rcf6m Cmt2 P&ix Fhlb_code"),
        (
            "abc12-month 12.5-month 1,200-month",
            "Abc12-month 12.5-month 1,200-month",
            "Abc12-month 12.5-month 1,200-month",
        ),
        ("cmt-backed fhlb's", "Cmt-backed Fhlb's", "CMT-backed FHLB's"),
        (
            "(fhlb), p&i; ust/rcf cmt.",
            "(fhlb), P&i; Ust/rcf Cmt.",
            "(FHLB), P&I; UST/RCF CMT.",
        ),
        ("FHLB 12-Month RCF", "Fhlb 12-month Rcf", "FHLB 12-Month RCF"),
        ("sofr libor usd", "Sofr Libor Usd", "Sofr Libor Usd"),
        (None, None, None),
    ]
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: business_titles
parser_config_name: Business Titles
version: "1"
columns:
  - source_column_name: label
    target_column_name: StrictTitle
    expected_data_type: string
    parser:
      type: string
      format: title
      collapse_whitespace: false
  - source_column_name: label
    target_column_name: BusinessTitle
    expected_data_type: string
    parser:
      type: string
      format: title_business_v1
      collapse_whitespace: false
"""
    )
    source_df = spark.range(len(cases)).select(
        (F.col("id") + 1).alias("row_id"),
        F.element_at(
            F.array(*(F.lit(source).cast("string") for source, _, _ in cases)),
            (F.col("id") + 1).cast("integer"),
        ).alias("label"),
    )
    rows = (
        SparkDataFrameParser()
        .parse_dataframe(source_df, config, key_columns=["row_id"])
        .parsed_df.orderBy("row_id")
        .collect()
    )
    assert [row.StrictTitle for row in rows] == [strict for _, strict, _ in cases]
    assert [row.BusinessTitle for row in rows] == [business for _, _, business in cases]


def test_interest_rate_index_v1_fails_closed_and_honors_null_markers(
    spark: SparkSession,
) -> None:
    """Reject partial/unknown lookalikes while preserving raw values or replacing configured nulls."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: interest_rate_index_boundaries
parser_config_name: Interest Rate Index Boundaries
version: "1"
columns:
  - source_column_name: financial_index
    target_column_name: PreservedIndex
    expected_data_type: string
    parser:
      type: string
      format: interest_rate_index_v1
      on_parse_error: preserve
      audit: true
  - source_column_name: financial_index
    target_column_name: NullAwareIndex
    expected_data_type: string
    parser:
      type: string
      format: interest_rate_index_v1
      null_markers: [NAP]
      null_marker_case_sensitive: false
      replace_null_markers: true
      on_parse_error: preserve
      audit: true
"""
    )
    values = [
        "NAP",
        "nap",
        "null",
        "Unknown Index 12M",
        "XSOFR30",
        "SOFR30AX",
        "RCF6MM",
        "SOFR Term - 2 Month",
        "SOFR 01M",
        "SOFR 0M",
        "SOFR 1.5M",
        "SOFR 12MM",
        "$12M",
        None,
        "  sofr   term - 3 MONTH  ",
    ]
    source_df = spark.range(len(values)).select(
        (F.col("id") + 1).alias("row_id"),
        F.element_at(
            F.array(*(F.lit(value).cast("string") for value in values)),
            (F.col("id") + 1).cast("integer"),
        ).alias("financial_index"),
    )
    parsed = SparkDataFrameParser().parse_dataframe(
        source_df,
        config,
        key_columns=["row_id"],
    )
    rows = parsed.parsed_df.orderBy("row_id").collect()
    assert [row.PreservedIndex for row in rows] == [
        *values[:-1],
        "SOFR Term 3-Month",
    ]
    assert [row.NullAwareIndex for row in rows] == [
        None,
        None,
        *values[2:-1],
        "SOFR Term 3-Month",
    ]

    audits = parsed.results_df.orderBy("row_id").collect()
    assert audits[0].spark_parser_parse_results[0].actions_applied == [
        "parse_error_preserved"
    ]
    assert audits[0].spark_parser_parse_results[1].actions_applied == [
        "null_marker_replaced"
    ]
    assert audits[3].spark_parser_parse_results[0].parsed_value == "Unknown Index 12M"
    assert audits[-2].spark_parser_parse_results[0].actions_applied == []
    assert audits[-1].spark_parser_parse_results[0].actions_applied == []

    fail_config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: interest_rate_index_fail
parser_config_name: Interest Rate Index Fail
version: "1"
columns:
  - source_column_name: financial_index
    target_column_name: FinancialIndex
    expected_data_type: string
    parser:
      type: string
      format: interest_rate_index_v1
"""
    )
    failing = SparkDataFrameParser().parse_dataframe(
        spark.sql("SELECT 'NAP' AS financial_index"),
        fail_config,
        key_columns=["financial_index"],
    )
    with pytest.raises((Py4JJavaError, PySparkException)):
        failing.parsed_df.collect()


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


def test_location_formats_recognize_the_complete_unicode_whitespace_set(
    spark: SparkSession,
) -> None:
    """Treat non-ASCII White_Space consistently in scalar and list-oriented profiles."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: unicode_location_whitespace
parser_config_name: Unicode Location Whitespace
version: "1"
columns:
  - source_column_name: address
    target_column_name: Address
    expected_data_type: string
    parser:
      type: string
      format: address_us_v1
      collapse_whitespace: false
      trim_whitespace: false
  - source_column_name: county
    target_column_name: County
    expected_data_type: string
    parser:
      type: string
      format: county
      collapse_whitespace: false
      trim_whitespace: false
  - source_column_name: state
    target_column_name: State
    expected_data_type: string
    parser:
      type: string
      format: state_us
      collapse_whitespace: false
      trim_whitespace: false
  - source_column_name: zip_code
    target_column_name: ZipCode
    expected_data_type: string
    parser:
      type: string
      format: zip
      collapse_whitespace: false
      trim_whitespace: false
"""
    )
    df = spark.range(1).select(
        F.lit("\u2003123\u2003main\u202fstreet\u3000").alias("address"),
        F.lit("\u2003o'brien\u202fcounty\u3000").alias("county"),
        F.lit("illinois\u202f,\u3000new\u2003york").alias("state"),
        F.lit("123\u202f,\u300012345\u2003-\u20036789").alias("zip_code"),
    )

    row = (
        SparkDataFrameParser()
        .parse_dataframe(
            df,
            config,
            key_columns=["address"],
        )
        .parsed_df.first()
    )
    assert row.asDict() == {
        "Address": "123 Main St",
        "County": "O'Brien County",
        "State": "IL, NY",
        "ZipCode": "00123, 12345-6789",
    }


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
    assert all(item.actions_applied == ["nested_parse_errors_resolved"] for item in audits.values())


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

    parsed = SparkDataFrameParser().parse_dataframe(
        bronze_df,
        config,
        key_columns=["row_id"],
    ).parsed_df
    rows = (
        parsed.orderBy("row_id")
        .select(
            "EventDate",
            F.date_format("EventTimestamp", "yyyy-MM-dd HH:mm:ss").alias(
                "EventTimestampText"
            ),
            F.date_format("EventTimestampNtz", "yyyy-MM-dd HH:mm:ss").alias(
                "EventTimestampNtzText"
            ),
        )
        .collect()
    )
    assert rows[0].EventDate.isoformat() == "2026-09-29"
    assert rows[1].EventDate.isoformat() == "2026-09-30"
    assert rows[2].EventDate.isoformat() == "2026-09-30"
    assert rows[3].EventDate.isoformat() == "2026-06-11"
    assert rows[4].EventDate.isoformat() == "2026-06-11"
    assert rows[5].EventDate is None
    assert rows[0].EventTimestampText == "2026-09-29 01:02:03"
    assert rows[1].EventTimestampText == "2026-09-30 08:08:00"
    assert rows[2].EventTimestampText == "2026-09-30 08:08:00"
    assert rows[3].EventTimestampText == "2026-06-11 20:08:17"
    assert rows[4].EventTimestampText == "2026-06-11 20:08:17"
    assert rows[5].EventTimestampText is None
    assert rows[0].EventTimestampNtzText == "2026-09-29 01:02:03"
    assert rows[1].EventTimestampNtzText == "2026-09-30 08:08:00"
    assert rows[2].EventTimestampNtzText == "2026-09-30 08:08:00"
    assert rows[3].EventTimestampNtzText == "2026-06-11 20:08:17"
    assert rows[4].EventTimestampNtzText == "2026-06-11 20:08:17"
    assert rows[5].EventTimestampNtzText is None


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

        parsed_df = (
            SparkDataFrameParser()
            .parse_dataframe(
                bronze_df,
                config,
                key_columns=["row_id"],
            )
            .parsed_df
        )
        # Render timestamps inside Spark. Collecting TimestampType as a Python datetime uses the
        # host process timezone, which is separate from spark.sql.session.timeZone and would make
        # this assertion test the workstation rather than the parser expression.
        rows = (
            parsed_df.orderBy("row_id")
            .select(
                "EventDate",
                F.date_format("EventTimestamp", "yyyy-MM-dd HH:mm:ss.SSSSSS").alias(
                    "EventTimestampText"
                ),
                F.date_format("EventTimestampNtz", "yyyy-MM-dd HH:mm:ss.SSSSSS").alias(
                    "EventTimestampNtzText"
                ),
            )
            .collect()
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


def test_custom_offset_date_formats_preserve_authored_day_across_session_timezones(
    spark: SparkSession,
) -> None:
    """Treat date as authored calendar fields, not a timezone projection of an instant."""
    previous_timezone = spark.conf.get("spark.sql.session.timeZone")
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: offset_date
parser_config_name: Offset Date
version: "1"
columns:
  - source_column_name: value
    target_column_name: EventDate
    expected_data_type: date
    parser:
      type: date
      formats: ["yyyy-MM-dd'T'HH:mm:ss[.SSSSSS]XXX"]
      on_parse_error: null
"""
    )
    values = [
        "2024-03-05T23:30:00Z",
        "2024-03-05T23:30:00+18:00",
        "2024-02-30T12:00:00Z",
        "2024-03-05T12:00:00+18:01",
        "not-a-date",
    ]
    df = spark.range(len(values)).select(
        F.col("id").alias("row_id"),
        F.element_at(
            F.array(*(F.lit(value) for value in values)),
            (F.col("id") + 1).cast("integer"),
        ).alias("value"),
    )
    try:
        for timezone_name in ("UTC", "Asia/Tokyo", "America/Los_Angeles"):
            spark.conf.set("spark.sql.session.timeZone", timezone_name)
            rows = (
                SparkDataFrameParser()
                .parse_dataframe(df, config, key_columns=["row_id"])
                .parsed_df.orderBy("row_id")
                .collect()
            )
            assert [None if row.EventDate is None else row.EventDate.isoformat() for row in rows] == [
                "2024-03-05",
                "2024-03-05",
                None,
                None,
                None,
            ]
    finally:
        spark.conf.set("spark.sql.session.timeZone", previous_timezone)


def test_builtin_datetime_guards_reject_final_line_terminators_without_raising(
    spark: SparkSession,
) -> None:
    """Keep Spark 3.5's EXCEPTION policy behind the parser's true full-token guard."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: datetime_line_terminators
parser_config_name: Datetime Line Terminators
version: "1"
columns:
  - source_column_name: valid_date
    target_column_name: ValidDate
    expected_data_type: date
    parser:
      type: date
      collapse_whitespace: false
      trim_whitespace: false
      on_parse_error: null
  - source_column_name: bad_date
    target_column_name: BadDate
    expected_data_type: date
    parser:
      type: date
      collapse_whitespace: false
      trim_whitespace: false
      on_parse_error: null
  - source_column_name: bad_timestamp
    target_column_name: BadTimestamp
    expected_data_type: timestamp
    parser:
      type: timestamp
      collapse_whitespace: false
      trim_whitespace: false
      on_parse_error: null
"""
    )
    df = spark.range(1).select(
        F.lit("2024-01-02").alias("valid_date"),
        F.lit("2024-01-02\n").alias("bad_date"),
        F.lit("2024-01-02T03:04:05\r\n").alias("bad_timestamp"),
    )

    row = SparkDataFrameParser().parse_dataframe(
        df,
        config,
        key_columns=["valid_date"],
    ).parsed_df.first()
    assert row.ValidDate.isoformat() == "2024-01-02"
    assert row.BadDate is None
    assert row.BadTimestamp is None


def test_temporal_default_literals_are_spark_timezone_owned_and_boundary_safe(
    spark: SparkSession,
) -> None:
    """Avoid driver-local datetime conversion for naive, offset-aware, and year-one defaults."""
    previous_timezone = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    try:
        config = YamlParserConfigCompiler().compile_text(
            """
parser_config_id: temporal_default_literals
parser_config_name: Temporal Default Literals
version: "1"
columns:
  - source_column_name: date_boundary
    target_column_name: DateBoundary
    expected_data_type: date
    parser:
      type: date
      is_nullable: false
      default_on_null: "0001-01-01"
  - source_column_name: naive_timestamp
    target_column_name: NaiveTimestamp
    expected_data_type: timestamp
    parser:
      type: timestamp
      is_nullable: false
      default_on_null: "1970-01-01T00:00:00"
  - source_column_name: aware_timestamp
    target_column_name: AwareTimestamp
    expected_data_type: timestamp
    parser:
      type: timestamp
      is_nullable: false
      default_on_null: "1970-01-01T00:00:00+05:00"
  - source_column_name: local_timestamp
    target_column_name: LocalTimestamp
    expected_data_type: timestamp_ntz
    parser:
      type: timestamp_ntz
      is_nullable: false
      default_on_null: "1970-01-01T00:00:00"
  - source_column_name: timestamp_boundary
    target_column_name: TimestampBoundary
    expected_data_type: timestamp
    parser:
      type: timestamp
      is_nullable: false
      default_on_null: "0001-01-01T00:00:00"
"""
        )
        df = spark.range(1).select(
            F.col("id").alias("row_id"),
            *(
                F.lit(None).cast("string").alias(name)
                for name in (
                    "date_boundary",
                    "naive_timestamp",
                    "aware_timestamp",
                    "local_timestamp",
                    "timestamp_boundary",
                )
            ),
        )

        parsed = (
            SparkDataFrameParser()
            .parse_dataframe(
                df,
                config,
                key_columns=["row_id"],
            )
            .parsed_df
        )
        row = parsed.select(
            F.date_format("DateBoundary", "yyyy-MM-dd").alias("DateBoundary"),
            F.date_format("NaiveTimestamp", "yyyy-MM-dd HH:mm:ss").alias("NaiveTimestamp"),
            F.date_format("AwareTimestamp", "yyyy-MM-dd HH:mm:ss").alias("AwareTimestamp"),
            F.date_format("LocalTimestamp", "yyyy-MM-dd HH:mm:ss").alias("LocalTimestamp"),
            F.date_format("TimestampBoundary", "yyyy-MM-dd HH:mm:ss").alias("TimestampBoundary"),
        ).first()
        assert row.asDict() == {
            "DateBoundary": "0001-01-01",
            "NaiveTimestamp": "1970-01-01 00:00:00",
            "AwareTimestamp": "1969-12-31 19:00:00",
            "LocalTimestamp": "1970-01-01 00:00:00",
            "TimestampBoundary": "0001-01-01 00:00:00",
        }
    finally:
        spark.conf.set("spark.sql.session.timeZone", previous_timezone)


def test_complex_audit_preserves_timestamp_microseconds(spark: SparkSession) -> None:
    """Serialize nested timestamp and timestamp_ntz values without millisecond truncation."""
    previous_timezone = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    try:
        config = YamlParserConfigCompiler().compile_text(
            """
parser_config_id: timestamp_audit_precision
parser_config_name: Timestamp Audit Precision
version: "1"
columns:
  - source_column_name: payload
    target_column_name: Payload
    expected_data_type: struct<event:timestamp,local:timestamp_ntz>
    parser:
      type: struct
      fields:
        - {source_field_name: event, target_field_name: event, parser: timestamp}
        - {source_field_name: local, target_field_name: local, parser: timestamp_ntz}
      audit: true
"""
        )
        df = spark.range(1).select(
            F.lit(
                '{"event":"2026-08-30T01:02:03.123456Z","local":"2026-08-30T01:02:03.654321"}'
            ).alias("payload")
        )

        audit = (
            SparkDataFrameParser()
            .parse_dataframe(
                df,
                config,
                key_columns=["payload"],
            )
            .results_df.first()
            .spark_parser_parse_results[0]
        )
        assert audit.parsed_value == (
            '{"event":"2026-08-30T01:02:03.123456Z","local":"2026-08-30T01:02:03.654321"}'
        )
    finally:
        spark.conf.set("spark.sql.session.timeZone", previous_timezone)


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
    tokens = [
        "1d",
        "1f",
        "0x1p3",
        ".",
        "+.",
        "-.",
        ".e2",
        "+.e2",
        "-.e2",
        "1e5",
        ".5",
        "-.5",
        "+1.5",
        "1.",
        "1.e2",
        "007",
    ]
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

    expected = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        100000.0,
        0.5,
        -0.5,
        1.5,
        1.0,
        100.0,
        7.0,
    ]
    assert [row.Scalar for row in rows] == expected
    assert [row.Nested[0] for row in rows] == expected


def test_numeric_full_token_guards_reject_final_line_terminators(spark: SparkSession) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: numeric_line_terminators
parser_config_name: Numeric Line Terminators
version: "1"
columns:
  - source_column_name: scalar
    target_column_name: Scalar
    expected_data_type: integer
    parser:
      type: integer
      collapse_whitespace: false
      trim_whitespace: false
      on_parse_error: null
  - source_column_name: nested
    target_column_name: Nested
    expected_data_type: array<integer>
    parser:
      type: array
      element_parser:
        type: integer
        collapse_whitespace: false
        trim_whitespace: false
      on_element_error: null
"""
    )
    tokens = ["123", "123\n", "123\r", "123\r\n", "123\u2028"]
    df = spark.range(len(tokens)).select(
        F.col("id").alias("row_id"),
        F.element_at(
            F.array(*(F.lit(token) for token in tokens)),
            (F.col("id") + 1).cast("integer"),
        ).alias("scalar"),
        F.element_at(
            F.array(*(F.lit(json.dumps([token])) for token in tokens)),
            (F.col("id") + 1).cast("integer"),
        ).alias("nested"),
    )

    rows = (
        SparkDataFrameParser()
        .parse_dataframe(df, config, key_columns=["row_id"])
        .parsed_df.orderBy("row_id")
        .collect()
    )
    assert [row.Scalar for row in rows] == [123, None, None, None, None]
    assert [row.Nested for row in rows] == [[123], [None], [None], [None], [None]]


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
            F.lit('{"a\'b":"bad"}').alias("values"),
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


def test_schema_preflight_does_not_read_restricted_case_setting(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use Catalyst's active resolver when a managed runtime withholds its configuration."""
    lower_probe = "__spark_parser_test_case_probe"
    upper_probe = lower_probe.upper()
    probe = spark.range(0).select(F.lit(None).cast("string").alias(lower_probe))
    case_sensitive = bool(probe.drop(upper_probe).schema.fields)
    source_name = "value" if case_sensitive else "VALUE"
    key_name = "row_id" if case_sensitive else "ROW_ID"
    config = YamlParserConfigCompiler().compile_mapping(
        {
            "parser_config_id": "restricted_resolver_setting",
            "parser_config_name": "Restricted Resolver Setting",
            "version": "1",
            "columns": [
                {
                    "source_column_name": source_name,
                    "target_column_name": "Value",
                    "expected_data_type": "string",
                    "parser": "string",
                }
            ],
        }
    )
    runtime_config_type = type(spark.conf)
    original_get = runtime_config_type.get

    def restricted_get(runtime_config, key, *args, **kwargs):
        if key == "spark.sql.caseSensitive":
            raise RuntimeError("configuration spark.sql.caseSensitive is not available")
        return original_get(runtime_config, key, *args, **kwargs)

    monkeypatch.setattr(runtime_config_type, "get", restricted_get)
    parsing = SparkDataFrameParser().parse_dataframe(
        spark.sql("SELECT 'ok' AS value, 'key' AS row_id"),
        config,
        key_columns=[key_name],
    )
    assert parsing.key_columns == (key_name,)
    assert parsing.results_df.columns[0] == key_name
    assert parsing.parsed_df.first().Value == "ok"


def test_schema_preflight_rejects_malformed_unicode_names(spark: SparkSession) -> None:
    """Reject malformed identifiers without consulting mutable resolver configuration."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: malformed_runtime_names
parser_config_name: Malformed Runtime Names
version: "1"
columns:
  - source_column_name: VALUE
    target_column_name: Parsed
    expected_data_type: string
    parser: string
"""
    )
    safe_df = spark.sql("SELECT 'ok' AS VALUE")
    with pytest.raises(ValueError, match="column_prefix.*well-formed Unicode"):
        SparkDataFrameParser().parse_dataframe(
            safe_df,
            config,
            key_columns=["VALUE"],
            column_prefix="bad\ud800prefix",
        )
    with pytest.raises(SchemaValidationError, match="key_columns.*well-formed Unicode"):
        SparkDataFrameParser().parse_dataframe(
            safe_df,
            config,
            key_columns=["bad\ud800key"],
        )


@pytest.mark.classic_spark
def test_schema_preflight_matches_both_classic_case_resolver_modes(
    classic_spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise both configurable Catalyst resolver modes on classic Spark."""
    spark = classic_spark
    compiler = YamlParserConfigCompiler()
    base_config = compiler.compile_text(
        """
parser_config_id: resolver_base
parser_config_name: Resolver Base
version: "1"
columns:
  - source_column_name: VALUE
    target_column_name: Parsed
    expected_data_type: string
    parser: string
"""
    )
    target_collision_config = compiler.compile_text(
        """
parser_config_id: resolver_targets
parser_config_name: Resolver Targets
version: "1"
columns:
  - source_column_name: Value
    target_column_name: Output
    expected_data_type: string
    parser: string
  - source_column_name: Value
    target_column_name: output
    expected_data_type: string
    parser: string
"""
    )
    unicode_distinct_config = compiler.compile_text(
        """
parser_config_id: resolver_unicode_distinct
parser_config_name: Resolver Unicode Distinct
version: "1"
columns:
  - source_column_name: ß
    target_column_name: ß
    expected_data_type: string
    parser: string
  - source_column_name: ss
    target_column_name: ss
    expected_data_type: string
    parser: string
"""
    )
    unicode_target_collision_config = compiler.compile_text(
        """
parser_config_id: resolver_unicode_targets
parser_config_name: Resolver Unicode Targets
version: "1"
columns:
  - source_column_name: Value
    target_column_name: K
    expected_data_type: string
    parser: string
  - source_column_name: Value
    target_column_name: k
    expected_data_type: string
    parser: string
"""
    )
    nested_source_collision_config = compiler.compile_text(
        """
parser_config_id: resolver_nested_sources
parser_config_name: Resolver Nested Sources
version: "1"
columns:
  - source_column_name: payload
    target_column_name: Payload
    expected_data_type: struct<x:string,y:string>
    parser:
      type: struct
      fields:
        - {source_field_name: A, target_field_name: x, parser: string}
        - {source_field_name: a, target_field_name: y, parser: string}
"""
    )
    nested_target_collision_config = compiler.compile_text(
        """
parser_config_id: resolver_nested_targets
parser_config_name: Resolver Nested Targets
version: "1"
columns:
  - source_column_name: payload
    target_column_name: Payload
    expected_data_type: struct<`K`:string,k:string>
    parser:
      type: struct
      fields:
        - {source_field_name: first, target_field_name: K, parser: string}
        - {source_field_name: second, target_field_name: k, parser: string}
"""
    )
    previous_case_sensitive = spark.conf.get("spark.sql.caseSensitive")
    try:
        spark.conf.set("spark.sql.caseSensitive", "false")
        runtime_config_type = type(spark.conf)
        original_get = runtime_config_type.get

        def restricted_get(runtime_config, key, *args, **kwargs):
            if key == "spark.sql.caseSensitive":
                raise RuntimeError("configuration spark.sql.caseSensitive is not available")
            return original_get(runtime_config, key, *args, **kwargs)

        monkeypatch.setattr(runtime_config_type, "get", restricted_get)
        case_mismatch = spark.sql("SELECT 'ok' AS value, 'key' AS row_id")
        parsing = SparkDataFrameParser().parse_dataframe(
            case_mismatch,
            base_config,
            key_columns=["ROW_ID"],
        )
        assert parsing.key_columns == ("ROW_ID",)
        assert parsing.results_df.columns[0] == "ROW_ID"
        assert parsing.parsed_df.first().Parsed == "ok"

        ambiguous_source = spark.sql("SELECT 'one' AS Value, 'two' AS value")
        with pytest.raises(
            SchemaValidationError,
            match="Configured input columns are ambiguous",
        ):
            SparkDataFrameParser().parse_dataframe(
                ambiguous_source,
                base_config,
                key_columns=["Value"],
            )

        ambiguous_key = spark.sql("SELECT 'payload' AS VALUE, 'one' AS Key, 'two' AS key")
        with pytest.raises(SchemaValidationError, match="key_columns are ambiguous"):
            SparkDataFrameParser().parse_dataframe(
                ambiguous_key,
                base_config,
                key_columns=["KEY"],
            )

        reserved_collision = spark.sql("SELECT 'ok' AS VALUE, 'occupied' AS SPARK_PARSER_CONFIG")
        with pytest.raises(
            SchemaValidationError,
            match="reserved parser output columns",
        ):
            SparkDataFrameParser().parse_dataframe(
                reserved_collision,
                base_config,
                key_columns=["VALUE"],
            )

        with pytest.raises(SchemaValidationError, match="target columns collide"):
            SparkDataFrameParser().parse_dataframe(
                spark.sql("SELECT 'one' AS Value"),
                target_collision_config,
                key_columns=["Value"],
            )

        unicode_distinct = SparkDataFrameParser().parse_dataframe(
            spark.range(1).select(
                F.lit("eszett").alias("ß"),
                F.lit("double-s").alias("ss"),
            ),
            unicode_distinct_config,
            key_columns=["ß"],
        )
        assert unicode_distinct.parsed_df.first().asDict() == {
            "ß": "eszett",
            "ss": "double-s",
        }

        with pytest.raises(SchemaValidationError, match="target columns collide"):
            SparkDataFrameParser().parse_dataframe(
                spark.sql("SELECT 'one' AS Value"),
                unicode_target_collision_config,
                key_columns=["Value"],
            )

        nested_df = spark.range(1).select(
            F.lit("key").alias("row_id"),
            F.lit('{"A":"one","a":"two"}').alias("payload"),
        )
        with pytest.raises(SchemaValidationError, match="struct source fields collide"):
            SparkDataFrameParser().parse_dataframe(
                nested_df,
                nested_source_collision_config,
                key_columns=["row_id"],
            )
        with pytest.raises(SchemaValidationError, match="struct target fields collide"):
            SparkDataFrameParser().parse_dataframe(
                nested_df,
                nested_target_collision_config,
                key_columns=["row_id"],
            )

        spark.conf.set("spark.sql.caseSensitive", "true")
        exact = SparkDataFrameParser().parse_dataframe(
            ambiguous_source,
            target_collision_config,
            key_columns=["value"],
        )
        assert exact.parsed_df.columns == ["Output", "output"]
        assert exact.parsed_df.first().asDict() == {"Output": "one", "output": "one"}
    finally:
        spark.conf.set("spark.sql.caseSensitive", previous_case_sensitive)


def test_case_insensitive_unicode_vocabularies_use_sparks_runtime_case_table(
    spark: SparkSession,
) -> None:
    """Keep Python's Unicode version out of Spark-side Boolean and null-marker matching."""
    marker = "\u2c2f"
    config = YamlParserConfigCompiler().compile_text(
        f"""
parser_config_id: unicode_vocabularies
parser_config_name: Unicode Vocabularies
version: "1"
globals:
  true_values: ["{marker}"]
  false_values: [N]
  boolean_case_sensitive: false
  null_markers: ["{marker}"]
  null_marker_case_sensitive: false
columns:
  - source_column_name: boolean_value
    target_column_name: BooleanValue
    expected_data_type: boolean
    parser: {{type: boolean, on_parse_error: null}}
  - source_column_name: null_value
    target_column_name: NullValue
    expected_data_type: string
    parser: {{type: string, replace_null_markers: true}}
"""
    )
    df = spark.range(1).select(
        F.lit(marker).alias("boolean_value"),
        F.lit(marker).alias("null_value"),
    )

    row = SparkDataFrameParser().parse_dataframe(
        df,
        config,
        key_columns=["boolean_value"],
    ).parsed_df.first()
    assert row.BooleanValue is True
    assert row.NullValue is None

    # A legitimate non-ASCII vocabulary remains usable after Spark-owned overlap validation.
    distinct_config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: unicode_distinct_vocabularies
parser_config_name: Unicode Distinct Vocabularies
version: "1"
globals:
  true_values: ["Ä"]
  false_values: ["Ö"]
  boolean_case_sensitive: false
columns:
  - source_column_name: boolean_value
    target_column_name: BooleanValue
    expected_data_type: boolean
    parser: boolean
"""
    )
    distinct_rows = (
        SparkDataFrameParser()
        .parse_dataframe(
            spark.range(2).select(
                F.col("id").alias("row_id"),
                F.when(F.col("id") == 0, F.lit("ä"))
                .otherwise(F.lit("ö"))
                .alias("boolean_value"),
            ),
            distinct_config,
            key_columns=["row_id"],
        )
        .parsed_df.orderBy("BooleanValue", ascending=False)
        .collect()
    )
    assert [row.BooleanValue for row in distinct_rows] == [True, False]

    # A familiar non-ASCII case pair is allowed through Spark-free compilation, then rejected at
    # DataFrame binding by the same Spark lowercasing implementation used for row values. Binding
    # must not depend on rows existing or on a nested Boolean element being evaluated.
    overlap_config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: unicode_overlapping_vocabularies
parser_config_name: Unicode Overlapping Vocabularies
version: "1"
globals:
  true_values: ["Ä"]
  false_values: ["ä"]
  boolean_case_sensitive: false
columns:
  - source_column_name: boolean_value
    target_column_name: BooleanValue
    expected_data_type: boolean
    parser: {type: boolean, on_parse_error: null}
"""
    )
    nested_overlap_config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: nested_unicode_overlapping_vocabularies
parser_config_name: Nested Unicode Overlapping Vocabularies
version: "1"
globals:
  true_values: ["Ä"]
  false_values: ["ä"]
  boolean_case_sensitive: false
columns:
  - source_column_name: boolean_values
    target_column_name: BooleanValues
    expected_data_type: array<boolean>
    parser: {type: array, element_parser: boolean}
"""
    )
    SparkDataFrameParser().parse_dataframe(
        spark.range(0).select(F.lit("ä").alias("boolean_value")),
        distinct_config,
        key_columns=["boolean_value"],
    )
    with pytest.raises(SchemaValidationError, match="overlap.*BooleanValue"):
        SparkDataFrameParser().parse_dataframe(
            spark.range(0).select(F.lit("Ä").alias("boolean_value")),
            overlap_config,
            key_columns=["boolean_value"],
        )
    with pytest.raises(SchemaValidationError, match="overlap.*BooleanValues"):
        SparkDataFrameParser().parse_dataframe(
            spark.range(0).select(F.lit("[]").alias("boolean_values")),
            nested_overlap_config,
            key_columns=["boolean_values"],
        )


@pytest.mark.classic_spark
def test_boolean_overlap_binding_is_metadata_only_under_classic_optimizer_changes(
    classic_spark: SparkSession,
) -> None:
    """Prove classic optimizer overrides neither start jobs nor alter overlap validation."""
    spark = classic_spark
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: classic_unicode_overlap_probe
parser_config_name: Classic Unicode Overlap Probe
version: "1"
globals:
  true_values: ["Ä"]
  false_values: ["ä"]
  boolean_case_sensitive: false
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: boolean
    parser: boolean
"""
    )
    job_group = "spark-parser-unicode-overlap-binding"
    excluded_rules_setting = "spark.sql.optimizer.excludedRules"
    previous_excluded_rules = spark.conf.get(excluded_rules_setting, "")
    local_property_names = (
        "spark.job.description",
        "spark.jobGroup.id",
        "spark.job.interruptOnCancel",
    )
    previous_local_properties = {
        name: spark.sparkContext.getLocalProperty(name) for name in local_property_names
    }
    spark.sparkContext.setJobGroup(job_group, job_group)
    try:
        spark.conf.set(
            excluded_rules_setting,
            "org.apache.spark.sql.catalyst.optimizer.ConstantFolding",
        )
        with pytest.raises(SchemaValidationError, match="overlap.*Value"):
            SparkDataFrameParser().parse_dataframe(
                spark.range(0).select(F.lit("Ä").alias("value")),
                config,
                key_columns=["value"],
            )
        assert spark.sparkContext.statusTracker().getJobIdsForGroup(job_group) == []
        assert spark.conf.get(excluded_rules_setting) == (
            "org.apache.spark.sql.catalyst.optimizer.ConstantFolding"
        )
    finally:
        spark.conf.set(excluded_rules_setting, previous_excluded_rules)
        for name, value in previous_local_properties.items():
            spark.sparkContext.setLocalProperty(name, value)


def test_boolean_overlap_validation_fails_closed_when_from_json_is_shadowed(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: shadowed_boolean_overlap_probe
parser_config_name: Shadowed Boolean Overlap Probe
version: "1"
globals:
  true_values: ["Ä"]
  false_values: ["ä"]
  boolean_case_sensitive: false
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: boolean
    parser: boolean
"""
    )
    monkeypatch.setattr(
        "spark_parser.spark_runtime.F.from_json",
        lambda _value, _schema: F.lit("shadowed"),
    )
    with pytest.raises(SchemaValidationError, match="built-in from_json.*not shadowed"):
        SparkDataFrameParser().parse_dataframe(
            spark.range(0).select(F.lit("Ä").alias("value")),
            config,
            key_columns=["value"],
        )


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
  - source_column_name: unicode_space_value
    target_column_name: UnicodeSpaceValue
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
    ).withColumn("unicode_space_value", F.lit("\u2003A\u202fB\u3000"))
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
    assert row.UnicodeSpaceValue == "A B"
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


def test_audited_nested_plan_carriers_keep_logical_and_physical_plans_linear(
    spark: SparkSession,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Budget expression size, not only Project depth, for recursively audited containers."""

    def depth_config(depth: int):
        data_type = "integer"
        parser_options: dict[str, object] = {"type": "integer"}
        for _ in range(depth):
            data_type = f"array<{data_type}>"
            parser_options = {
                "type": "array",
                "element_parser": parser_options,
                "on_element_error": "null",
            }
        parser_options["audit"] = True
        return YamlParserConfigCompiler().compile_mapping(
            {
                "parser_config_id": f"audited_depth_{depth}",
                "parser_config_name": "Audited Depth",
                "version": "1",
                "columns": [
                    {
                        "source_column_name": "src",
                        "target_column_name": "Value",
                        "expected_data_type": data_type,
                        "parser": parser_options,
                    }
                ],
            }
        )

    analyzed_sizes: list[int] = []
    deepest = None
    deepest_explain = ""
    for depth in range(1, 5):
        payload: object = "bad"
        for _ in range(depth):
            payload = [payload]
        parsing = SparkDataFrameParser().parse_dataframe(
            spark.range(1).select(
                F.lit("key").alias("key"),
                F.lit(json.dumps(payload)).alias("src"),
            ),
            depth_config(depth),
            key_columns=["key"],
        )
        parsing.parsed_df.explain(mode="extended")
        explain_output = capsys.readouterr().out
        analyzed = explain_output.split("== Analyzed Logical Plan ==", maxsplit=1)[1].split(
            "== Optimized Logical Plan ==",
            maxsplit=1,
        )[0]
        analyzed_sizes.append(len(analyzed))
        deepest = parsing
        deepest_explain = explain_output

    assert deepest is not None
    # Before the carrier binding, depth four was about 7.4 MB and 148 times depth one. Keep a
    # generous cross-version budget while making exponential re-embedding unambiguously fail.
    assert analyzed_sizes[-1] < 500_000
    assert analyzed_sizes[-1] < analyzed_sizes[0] * 6
    optimized = deepest_explain.split("== Optimized Logical Plan ==", maxsplit=1)[1].split(
        "== Physical Plan ==",
        maxsplit=1,
    )[0]
    physical = deepest_explain.split("== Physical Plan ==", maxsplit=1)[1]
    assert len(optimized) < 500_000
    assert len(physical) < 500_000

    audit = deepest.results_df.first().spark_parser_parse_results[0]
    assert audit.nested_error_paths == ["$[0][0][0][0]"]


@pytest.mark.classic_spark
def test_analyzer_iteration_exhaustion_is_a_clear_metadata_only_binding_error(
    classic_spark: SparkSession,
) -> None:
    spark = classic_spark
    previous_iterations = spark.conf.get("spark.sql.analyzer.maxIterations")
    job_group = "spark-parser-analyzer-exhaustion"
    local_property_names = (
        "spark.job.description",
        "spark.jobGroup.id",
        "spark.job.interruptOnCancel",
    )
    previous_local_properties = {
        name: spark.sparkContext.getLocalProperty(name) for name in local_property_names
    }
    spark.sparkContext.setJobGroup(job_group, job_group)
    config = YamlParserConfigCompiler().compile_mapping(
        {
            "parser_config_id": "analyzer_exhaustion",
            "parser_config_name": "Analyzer Exhaustion",
            "version": "1",
            "columns": [
                {
                    "source_column_name": "src",
                    "target_column_name": "Value",
                    "expected_data_type": "array<array<array<array<integer>>>>",
                    "parser": {
                        "type": "array",
                        "audit": True,
                        "on_element_error": "null",
                        "element_parser": {
                            "type": "array",
                            "on_element_error": "null",
                            "element_parser": {
                                "type": "array",
                                "on_element_error": "null",
                                "element_parser": {
                                    "type": "array",
                                    "on_element_error": "null",
                                    "element_parser": "integer",
                                },
                            },
                        },
                    },
                }
            ],
        }
    )
    try:
        spark.conf.set("spark.sql.analyzer.maxIterations", "2")
        with pytest.raises(
            SchemaValidationError,
            match=(
                r"spark\.sql\.analyzer\.maxIterations=2.*"
                r"maximum configured complex nesting depth is 4"
            ),
        ):
            SparkDataFrameParser().parse_dataframe(
                spark.range(1).select(
                    F.lit("key").alias("key"),
                    F.lit('[[[["bad"]]]]').alias("src"),
                ),
                config,
                key_columns=["key"],
            )

        class StackExhaustedPlan:
            """Minimal plan object that reproduces the JVM error text without a long depth-64 run."""

            def withColumns(self, _columns):
                raise RuntimeError(
                    "java.lang.StackOverflowError at "
                    "ColumnNodeToExpressionConverter.apply"
                )

        with pytest.raises(
            SchemaValidationError,
            match=r"driver JVM thread stack.*maximum configured complex nesting depth is 4",
        ):
            SparkDataFrameParser._with_columns_checked(StackExhaustedPlan(), {}, config)
        assert spark.sparkContext.statusTracker().getJobIdsForGroup(job_group) == []
    finally:
        spark.conf.set("spark.sql.analyzer.maxIterations", previous_iterations)
        for name, value in previous_local_properties.items():
            spark.sparkContext.setLocalProperty(name, value)


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


def test_documented_option_branches_materialize_through_native_expressions(
    spark: SparkSession,
) -> None:
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: option_branches
parser_config_name: Option Branches
version: "1"
columns:
  - source_column_name: display
    target_column_name: LowerValue
    expected_data_type: string
    parser: {type: string, format: lower}
  - source_column_name: display
    target_column_name: PascalValue
    expected_data_type: string
    parser: {type: string, format: pascal}
  - source_column_name: map_value
    target_column_name: MapValue
    expected_data_type: map<string,string>
    parser:
      type: map
      value_parser: string
      drop_null_values: true
  - source_column_name: struct_value
    target_column_name: StructValue
    expected_data_type: struct<count:integer>
    parser:
      type: struct
      audit: true
      fields:
        - source_field_name: count
          target_field_name: count
          parser:
            type: integer
            on_parse_error: default
            default_on_error: -1
  - source_column_name: binary_value
    target_column_name: BinaryValue
    expected_data_type: binary
    parser:
      type: binary
      encoding: base64
      is_nullable: false
      default_on_null: SGk=
"""
    )
    df = spark.range(1).select(
        F.lit(" hELLo wORLD ").alias("display"),
        F.lit('{"a":"","b":"x"}').alias("map_value"),
        F.lit('{"count":"bad"}').alias("struct_value"),
        F.lit(None).cast("string").alias("binary_value"),
    )

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["display"])
    row = parsing.parsed_df.first()
    assert row.LowerValue == "hello world"
    assert row.PascalValue == "HelloWorld"
    assert row.MapValue == {"b": "x"}
    assert row.StructValue["count"] == -1
    assert bytes(row.BinaryValue) == b"Hi"
    audit = parsing.results_df.first().spark_parser_parse_results[0]
    assert audit.nested_error_paths == ["$.count"]
    assert audit.actions_applied == ["nested_parse_errors_resolved"]


def test_base64_runtime_uses_the_compilers_strict_padded_alphabet(
    spark: SparkSession,
) -> None:
    """Reject Spark decoder extensions that are invalid for compiled Base64 defaults."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: strict_base64
parser_config_name: Strict Base64
version: "1"
columns:
  - source_column_name: value
    target_column_name: Value
    expected_data_type: binary
    parser:
      type: binary
      encoding: base64
      collapse_whitespace: false
      trim_whitespace: false
      empty_is_null: false
      on_parse_error: null
"""
    )
    tokens = [
        "",
        "SGk=",
        "AAAA",
        "SGk",
        "S Gk=",
        "SG\tk=",
        "SGk!",
        "AAAA=",
        "SGk=\n",
        "SGk=\r\n",
        "SGk=\u2028",
    ]
    rows = (
        SparkDataFrameParser()
        .parse_dataframe(
            spark.range(len(tokens)).select(
                F.col("id").alias("row_id"),
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

    assert [None if row.Value is None else bytes(row.Value) for row in rows] == [
        b"",
        b"Hi",
        b"\x00\x00\x00",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]


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


def test_timestamp_ntz_honors_error_policies_under_ansi(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

        class ConnectNewSessionUnsupportedError(PySparkNotImplementedError):
            def __init__(self) -> None:
                Exception.__init__(self, "SparkSession.newSession is unavailable")

            def getCondition(self) -> str:
                return "NOT_IMPLEMENTED"

            def getErrorClass(self) -> str:
                return "NOT_IMPLEMENTED"

        def connect_new_session(_session):
            raise ConnectNewSessionUnsupportedError

        # Some Connect/serverless session classes omit this method entirely. Leave those classes
        # untouched so the runtime exercises its real missing-capability fallback.
        if callable(getattr(type(spark), "newSession", None)):
            monkeypatch.setattr(type(spark), "newSession", connect_new_session)
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


def _invalid_custom_datetime_config():
    """Compile the invalid-pattern fixture shared by portable and classic metadata probes."""
    return YamlParserConfigCompiler().compile_text(
        """
parser_config_id: invalid_custom_datetime
parser_config_name: Invalid Custom Datetime
version: "1"
columns:
  - source_column_name: occurred_at
    target_column_name: OccurredAt
    expected_data_type: timestamp
    parser:
      type: timestamp
      formats: ['invalid[']
      on_parse_error: null
"""
    )


def test_invalid_custom_datetime_pattern_fails_during_dataframe_binding(
    spark: SparkSession,
) -> None:
    config = _invalid_custom_datetime_config()
    previous_policy = spark.conf.get("spark.sql.legacy.timeParserPolicy")
    try:
        spark.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        with pytest.raises(
            SchemaValidationError,
            match=r"Custom datetime format 'invalid\[' is invalid",
        ):
            SparkDataFrameParser().parse_dataframe(
                spark.sql("SELECT 'anything' AS occurred_at"),
                config,
                key_columns=["occurred_at"],
            )
    finally:
        spark.conf.set("spark.sql.legacy.timeParserPolicy", previous_policy)


@pytest.mark.classic_spark
def test_invalid_datetime_probe_preserves_classic_optimizer_configuration(
    classic_spark: SparkSession,
) -> None:
    """Keep invalid-pattern analysis independent of caller optimizer exclusions."""
    spark = classic_spark
    config = _invalid_custom_datetime_config()
    policy_setting = "spark.sql.legacy.timeParserPolicy"
    excluded_rules_setting = "spark.sql.optimizer.excludedRules"
    previous_policy = spark.conf.get(policy_setting)
    previous_excluded_rules = spark.conf.get(excluded_rules_setting, "")
    try:
        spark.conf.set(policy_setting, "CORRECTED")
        spark.conf.set(
            excluded_rules_setting,
            "org.apache.spark.sql.catalyst.optimizer.ConstantFolding",
        )
        with pytest.raises(
            SchemaValidationError,
            match=r"Custom datetime format 'invalid\[' is invalid",
        ):
            SparkDataFrameParser().parse_dataframe(
                spark.sql("SELECT 'anything' AS occurred_at"),
                config,
                key_columns=["occurred_at"],
            )
        assert spark.conf.get(excluded_rules_setting) == (
            "org.apache.spark.sql.catalyst.optimizer.ConstantFolding"
        )
    finally:
        spark.conf.set(policy_setting, previous_policy)
        spark.conf.set(excluded_rules_setting, previous_excluded_rules)


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
  - source_column_name: duplicate_object
    target_column_name: DuplicateObject
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
      value_parser:
        type: integer
        is_nullable: false
        default_on_null: -1
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
      element_parser:
        type: map
        value_parser:
          type: integer
          is_nullable: false
          default_on_null: -1
      on_element_error: null
      audit: true
"""
    )
    df = spark.range(1).select(
        F.lit("{'a':1}").alias("object"),
        F.lit('{"a":1,"unused":1,"unused":2}').alias("duplicate_object"),
        F.lit("[]").alias("empty_array"),
        F.lit("{}").alias("empty_map"),
        F.lit('{"a":null,"a":2}').alias("duplicate_map"),
        F.lit('[{"a":null,"a":2}]').alias("nested_duplicate_map"),
    )

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["object"])
    row = parsing.parsed_df.first()
    assert row.Object is None
    assert row.DuplicateObject is None
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
    assert audits["DuplicateObject"].nested_error_paths == ["$"]
    assert audits["DuplicateObject"].actions_applied == ["parse_error_to_null"]
    assert audits["DuplicateMap"].nested_error_paths == ["$"]
    assert audits["DuplicateMap"].nested_default_on_null_paths == []
    assert audits["DuplicateMap"].actions_applied == ["parse_error_to_null"]
    assert audits["NestedDuplicateMap"].nested_error_paths == ["$[0]"]
    assert audits["NestedDuplicateMap"].nested_default_on_null_paths == []
    assert audits["NestedDuplicateMap"].actions_applied == ["nested_parse_errors_resolved"]


def test_json_requires_one_complete_standard_container_token(spark: SparkSession) -> None:
    """Reject valid prefixes, appended documents, and non-standard numeric tokens."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: strict_complete_json
parser_config_name: Strict Complete JSON
version: "1"
columns:
  - source_column_name: array_trailing
    target_column_name: ArrayTrailing
    expected_data_type: array<integer>
    parser: {type: array, element_parser: integer, on_parse_error: null, audit: true}
  - source_column_name: struct_trailing
    target_column_name: StructTrailing
    expected_data_type: struct<a:integer>
    parser:
      type: struct
      fields:
        - {source_field_name: a, target_field_name: a, parser: integer}
      on_parse_error: null
  - source_column_name: map_trailing
    target_column_name: MapTrailing
    expected_data_type: map<string,integer>
    parser: {type: map, value_parser: integer, on_parse_error: null}
  - source_column_name: array_early_close
    target_column_name: ArrayEarlyClose
    expected_data_type: array<integer>
    parser: {type: array, element_parser: integer, on_parse_error: null, audit: true}
  - source_column_name: struct_member_injection
    target_column_name: StructMemberInjection
    expected_data_type: struct<a:integer>
    parser:
      type: struct
      fields:
        - {source_field_name: a, target_field_name: a, parser: integer}
      on_parse_error: null
      audit: true
  - source_column_name: map_member_injection
    target_column_name: MapMemberInjection
    expected_data_type: map<string,integer>
    parser: {type: map, value_parser: integer, on_parse_error: null, audit: true}
  - source_column_name: escaped_array
    target_column_name: EscapedArray
    expected_data_type: array<string>
    parser: {type: array, element_parser: string, on_parse_error: null}
  - source_column_name: nonfinite_array
    target_column_name: NonfiniteArray
    expected_data_type: array<string>
    parser: {type: array, element_parser: string, on_parse_error: null}
  - source_column_name: nonfinite_map
    target_column_name: NonfiniteMap
    expected_data_type: map<string,string>
    parser: {type: map, value_parser: string, on_parse_error: null}
  - source_column_name: nested_nonfinite
    target_column_name: NestedNonfinite
    expected_data_type: struct<payload:array<string>>
    parser:
      type: struct
      fields:
        - source_field_name: payload
          target_field_name: payload
          parser: {type: array, element_parser: string, on_parse_error: null}
      audit: true
"""
    )
    df = spark.range(1).select(
        F.lit("[1] trailing").alias("array_trailing"),
        F.lit('{"a":1}{"a":2}').alias("struct_trailing"),
        F.lit('{"a":1} trailing').alias("map_trailing"),
        F.lit("[]} ignored []").alias("array_early_close"),
        F.lit('{"a":1}, "junk":1').alias("struct_member_injection"),
        F.lit('{"a":1}, "junk":1').alias("map_member_injection"),
        F.lit(json.dumps(["]", "{", 'quote " and slash \\'])).alias("escaped_array"),
        F.lit("[NaN]").alias("nonfinite_array"),
        F.lit('{"a":Infinity}').alias("nonfinite_map"),
        F.lit('{"payload":"[NaN]"}').alias("nested_nonfinite"),
    )

    parsing = SparkDataFrameParser().parse_dataframe(
        df,
        config,
        key_columns=["array_trailing"],
    )
    row = parsing.parsed_df.first()
    assert row.asDict(recursive=True) == {
        "ArrayTrailing": None,
        "StructTrailing": None,
        "MapTrailing": None,
        "ArrayEarlyClose": None,
        "StructMemberInjection": None,
        "MapMemberInjection": None,
        "EscapedArray": ["]", "{", 'quote " and slash \\'],
        "NonfiniteArray": None,
        "NonfiniteMap": None,
        "NestedNonfinite": {"payload": None},
    }
    audits = {
        item.target_column_name: item
        for item in parsing.results_df.first().spark_parser_parse_results
    }
    assert audits["ArrayTrailing"].nested_error_paths == ["$"]
    assert audits["ArrayTrailing"].actions_applied == ["parse_error_to_null"]
    for target in (
        "ArrayEarlyClose",
        "StructMemberInjection",
        "MapMemberInjection",
    ):
        assert audits[target].nested_error_paths == ["$"]
        assert audits[target].actions_applied == ["parse_error_to_null"]
    assert audits["NestedNonfinite"].nested_error_paths == ["$.payload"]
    assert audits["NestedNonfinite"].actions_applied == ["nested_parse_errors_resolved"]


def test_complex_json_handles_large_strings_and_escape_runs_without_jvm_stack_growth(
    spark: SparkSession,
) -> None:
    """Keep ordinary large bronze text inside JSON on a constant-stack regex path."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: large_json_strings
parser_config_name: Large JSON Strings
version: "1"
columns:
  - source_column_name: array_value
    target_column_name: ArrayValue
    expected_data_type: array<string>
    parser: {type: array, element_parser: string, on_parse_error: null}
  - source_column_name: struct_value
    target_column_name: StructValue
    expected_data_type: struct<note:string>
    parser:
      type: struct
      fields:
        - {source_field_name: note, target_field_name: note, parser: string}
      on_parse_error: null
  - source_column_name: map_value
    target_column_name: MapValue
    expected_data_type: map<string,string>
    parser: {type: map, value_parser: string, on_parse_error: null}
  - source_column_name: nested_value
    target_column_name: NestedValue
    expected_data_type: array<struct<note:string>>
    parser:
      type: array
      element_parser:
        type: struct
        fields:
          - {source_field_name: note, target_field_name: note, parser: string}
      on_parse_error: null
  - source_column_name: escaped_value
    target_column_name: EscapedValue
    expected_data_type: array<string>
    parser: {type: array, element_parser: string, on_parse_error: null}
  - source_column_name: malformed_value
    target_column_name: MalformedValue
    expected_data_type: array<string>
    parser: {type: array, element_parser: string, on_parse_error: null}
"""
    )
    long_text = "a" * 100_000
    escape_text = '"' * 50_000
    df = spark.range(1).select(
        F.lit(json.dumps([long_text])).alias("array_value"),
        F.lit(json.dumps({"note": long_text})).alias("struct_value"),
        F.lit(json.dumps({"note": long_text})).alias("map_value"),
        F.lit(json.dumps([{"note": long_text}])).alias("nested_value"),
        F.lit(json.dumps([escape_text])).alias("escaped_value"),
        F.lit('["' + long_text).alias("malformed_value"),
    )

    row = SparkDataFrameParser().parse_dataframe(
        df,
        config,
        key_columns=["array_value"],
    ).parsed_df.first()
    assert row.ArrayValue == [long_text]
    assert row.StructValue.note == long_text
    assert row.MapValue == {"note": long_text}
    assert row.NestedValue[0].note == long_text
    assert row.EscapedValue == [escape_text]
    assert row.MalformedValue is None


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
    df = (
        spark.range(1)
        .select(
            F.lit(1).alias("id"),
            F.lit("null").alias("payload"),
        )
        .union(
            spark.range(1).select(
                F.lit(2).alias("id"),
                F.lit("NULL").alias("payload"),
            )
        )
    )

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["id"])
    assert [row.Payload for row in parsing.parsed_df.collect()] == [
        ["NULL_DEFAULT"],
        ["ERROR_DEFAULT"],
    ]
    audits = {row.id: row.spark_parser_parse_results[0] for row in parsing.results_df.collect()}
    assert audits[1].actions_applied == ["json_null_to_null", "default_on_null_applied"]
    assert audits[2].actions_applied == ["parse_error_default_applied"]


def test_complex_defaults_apply_recursive_child_final_value_contracts(
    spark: SparkSession,
) -> None:
    """Apply child null defaults and zero invalidation inside parent complex defaults."""
    config = YamlParserConfigCompiler().compile_text(
        """
parser_config_id: recursive_complex_defaults
parser_config_name: Recursive Complex Defaults
version: "1"
columns:
  - source_column_name: array_value
    target_column_name: ArrayValue
    expected_data_type: array<integer>
    parser:
      type: array
      element_parser:
        type: integer
        zero_is_valid: false
        is_nullable: false
        default_on_null: -1
      is_nullable: false
      default_on_null: [null, 0, 2]
  - source_column_name: struct_value
    target_column_name: StructValue
    expected_data_type: struct<a:integer,values:array<integer>>
    parser:
      type: struct
      fields:
        - source_field_name: a
          target_field_name: a
          parser: {type: integer, is_nullable: false, default_on_null: 5}
        - source_field_name: values
          target_field_name: values
          parser:
            type: array
            element_parser:
              type: integer
              zero_is_valid: false
              is_nullable: false
              default_on_null: 6
      is_nullable: false
      default_on_null: {a: null, values: [0, null]}
  - source_column_name: map_value
    target_column_name: MapValue
    expected_data_type: map<string,integer>
    parser:
      type: map
      value_parser:
        type: integer
        zero_is_valid: false
        is_nullable: false
        default_on_null: -1
      is_nullable: false
      default_on_null: {z: 0, a: null, m: 2}
      audit: true
"""
    )
    df = spark.range(1).select(
        F.col("id").alias("row_id"),
        F.lit(None).cast("string").alias("array_value"),
        F.lit(None).cast("string").alias("struct_value"),
        F.lit(None).cast("string").alias("map_value"),
    )

    parsing = SparkDataFrameParser().parse_dataframe(df, config, key_columns=["row_id"])
    row = parsing.parsed_df.first()
    assert row.ArrayValue == [-1, -1, 2]
    assert row.StructValue.asDict(recursive=True) == {"a": 5, "values": [6, 6]}
    assert row.MapValue == {"a": -1, "m": 2, "z": -1}
    audit = parsing.results_df.first().spark_parser_parse_results[0]
    assert audit.parsed_value == '{"a":-1,"m":2,"z":-1}'


def test_nested_paths_and_map_results_are_unambiguous_and_deterministic(
    spark: SparkSession,
) -> None:
    """Sort map keys and escape every unsafe dynamic or configured path segment."""
    config = YamlParserConfigCompiler().compile_mapping(
        {
            "parser_config_id": "deterministic_paths",
            "parser_config_name": "Deterministic Paths",
            "version": "1",
            "columns": [
                {
                    "source_column_name": "map_value",
                    "target_column_name": "MapValue",
                    "expected_data_type": "map<string,integer>",
                    "parser": {
                        "type": "map",
                        "value_parser": "integer",
                        "on_value_error": "null",
                        "audit": True,
                    },
                },
                {
                    "source_column_name": "struct_value",
                    "target_column_name": "StructValue",
                    "expected_data_type": "struct<`a.b`:integer>",
                    "parser": {
                        "type": "struct",
                        "fields": [
                            {
                                "source_field_name": "raw",
                                "target_field_name": "a.b",
                                "parser": {"type": "integer", "on_parse_error": "null"},
                            }
                        ],
                        "audit": True,
                    },
                },
            ],
        }
    )
    map_value = {
        "z": "bad",
        "a.b": "bad",
        "O'Brien": "bad",
        "back\\slash": "bad",
        "line\nbreak": "bad",
        "雪": "bad",
        "m": "1",
    }
    df = spark.range(1).select(
        F.lit(json.dumps(map_value, ensure_ascii=False)).alias("map_value"),
        F.lit('{"raw":"bad"}').alias("struct_value"),
    )

    parsing = SparkDataFrameParser().parse_dataframe(
        df,
        config,
        key_columns=["map_value"],
    )
    audits = {
        item.target_column_name: item
        for item in parsing.results_df.first().spark_parser_parse_results
    }
    assert audits["MapValue"].nested_error_paths == [
        "$['O\\'Brien']",
        "$['a.b']",
        "$['back\\\\slash']",
        "$['line\\nbreak']",
        "$['z']",
        "$['雪']",
    ]
    assert audits["StructValue"].nested_error_paths == ["$['a.b']"]
