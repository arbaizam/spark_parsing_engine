"""Datetime contracts across session defaults and legacy calendar policies."""

from contextlib import contextmanager

import pytest
from test_spark_runtime import classic_spark as _classic_spark_fixture
from test_spark_runtime import spark as _spark_fixture

# Keep optional-runtime collection guards before direct Spark imports.
# isort: split
from pyspark.errors import PySparkException
from pyspark.sql import functions as F
from pyspark.sql import types as T

from spark_parser import parser

pytestmark = [pytest.mark.spark, pytest.mark.classic_spark]
classic_spark = _classic_spark_fixture
spark = _spark_fixture


@contextmanager
def _settings(session, **settings):
    previous = {name: session.conf.get(name) for name in settings}
    try:
        for name, value in settings.items():
            session.conf.set(name, value)
        yield
    finally:
        for name, value in previous.items():
            session.conf.set(name, value)


def _input(session, values):
    return session.range(len(values)).select(
        F.col("id").cast("string").alias("row_id"),
        F.element_at(
            F.array(*(F.lit(value).cast("string") for value in values)),
            (F.col("id") + 1).cast("integer"),
        ).alias("value"),
    )


def _binding(target, data_type, **options):
    return {
        "source_column_name": "value",
        "target_column_name": target,
        "expected_data_type": data_type,
        "parser": {"type": data_type, "on_parse_error": "null", **options},
    }


def _config(*columns):
    return parser.compile_mapping(
        {
            "parser_config_id": "datetime_regression",
            "parser_config_name": "Datetime Regression",
            "version": "1",
            "columns": [
                {
                    "source_column_name": "row_id",
                    "target_column_name": "RowId",
                    "expected_data_type": "integer",
                    "parser": "integer",
                },
                *columns,
            ],
        }
    )


@pytest.mark.parametrize("timestamp_type", ["TIMESTAMP_LTZ", "TIMESTAMP_NTZ"])
@pytest.mark.parametrize("policy", ["fail", "null", "default"])
def test_timestamp_values_and_defaults_keep_the_declared_type(
    classic_spark, timestamp_type, policy
):
    null_default = "2000-01-02T03:04:05+02:00"
    options = {
        "is_nullable": False,
        "default_on_null": null_default,
        "on_parse_error": policy,
        "audit": True,
    }
    if policy == "default":
        options["default_on_error"] = "2001-01-02T03:04:05+02:00"
    config = _config(_binding("EventTime", "timestamp", **options))
    values = [
        "2026-09-30T12:34:56-05:00",
        "2026-09-30T12:34:56.123456",
        None,
        "bad",
        "2026-09-30T12:34:56+18:01",
    ]
    with _settings(classic_spark, **{"spark.sql.timestampType": timestamp_type}):
        parsing = parser.parse_dataframe(
            _input(classic_spark, values), config, key_columns=["row_id"], error_mode="collect"
        )
        assert parsing.parsed_df.schema["EventTime"].dataType == T.TimestampType()
        rows = (
            parsing.parsed_df.orderBy("RowId")
            .select(F.col("EventTime").cast("string").alias("text"), "spark_parser_parse_errors")
            .collect()
        )
        fallback = "2001-01-02 01:04:05" if policy == "default" else "2000-01-02 01:04:05"
        assert [row.text for row in rows] == [
            "2026-09-30 17:34:56",
            "2026-09-30 12:34:56.123456",
            "2000-01-02 01:04:05",
            fallback,
            fallback,
        ]
        assert [len(row.spark_parser_parse_errors) for row in rows] == [0, 0, 0, 1, 1]
        resolution = "default_on_error" if policy == "default" else "default_on_null"
        assert rows[3].spark_parser_parse_errors[0].resolution == resolution
        audits = parsing.results_df.orderBy("RowId").collect()
        assert [row.spark_parser_parse_results[0].parsed_value for row in audits] == [
            row.text for row in rows
        ]


@pytest.mark.parametrize("time_policy", ["CORRECTED", "EXCEPTION", "LEGACY"])
@pytest.mark.parametrize("ansi", ["true", "false"])
@pytest.mark.parametrize("error_mode", ["configured", "collect"])
def test_builtin_dates_use_the_gregorian_calendar_without_escaping_error_policy(
    classic_spark, time_policy, ansi, error_mode
):
    values = ["1500-02-29", "1400-02-29", "1500-02-28", "1600-02-29", "1582-10-10", None]
    config = _config(
        _binding("CalendarDate", "date", audit=True),
        _binding("EventTime", "timestamp", formats=["yyyy-MM-dd"]),
        _binding("LocalTime", "timestamp_ntz", formats=["yyyy-MM-dd"]),
    )
    with _settings(
        classic_spark,
        **{
            "spark.sql.legacy.timeParserPolicy": time_policy,
            "spark.sql.ansi.enabled": ansi,
        },
    ):
        parsing = parser.parse_dataframe(
            _input(classic_spark, values), config, key_columns=["row_id"], error_mode=error_mode
        )
        rows = (
            parsing.parsed_df.orderBy("RowId")
            .select(
                "CalendarDate",
                *(F.col(name).cast("string").alias(name) for name in ("EventTime", "LocalTime")),
            )
            .collect()
        )
        expected = [None, None, "1500-02-28", "1600-02-29", "1582-10-10", None]
        # Spark's LEGACY date-to-string formatter itself shifts dates inside the 1582 cutover;
        # inspect the actual DateType value, whose epoch-day representation is Gregorian.
        assert [
            None if row.CalendarDate is None else row.CalendarDate.isoformat() for row in rows
        ] == expected
        for name in ("EventTime", "LocalTime"):
            assert [row[name] for row in rows] == [
                None if value is None else value + " 00:00:00" for value in expected
            ]
        audits = parsing.results_df.orderBy("RowId").collect()
        assert audits[0].spark_parser_parse_results[0].actions_applied == ["parse_error_to_null"]
        assert audits[4].spark_parser_parse_results[0].parsed_value == "1582-10-10"
        if error_mode == "collect":
            assert [len(row.spark_parser_parse_errors) for row in audits] == [3, 3, 0, 0, 0, 0]


@pytest.mark.parametrize("time_policy", ["CORRECTED", "EXCEPTION", "LEGACY"])
def test_us_12_hour_formats_validate_clock_fields_and_preserve_noon_and_midnight(
    classic_spark, time_policy
):
    values = [
        "09/30/2026 12:00 AM",
        "09/30/2026 12:00 pm",
        "09/30/2026 1:02:03 PM",
        "09/30/2026 0:00 AM",
        "09/30/2026 13:00 PM",
        "09/30/2026 1:60 AM",
        "09/30/2026 1:02:60 PM",
        "02/29/1500 1:00 AM",
        "02/29/1600 1:00 AM",
    ]
    config = _config(_binding("EventTime", "timestamp"), _binding("LocalTime", "timestamp_ntz"))
    with _settings(classic_spark, **{"spark.sql.legacy.timeParserPolicy": time_policy}):
        rows = (
            parser.parse_dataframe(_input(classic_spark, values), config, key_columns=["row_id"])
            .parsed_df.orderBy("RowId")
            .select(
                F.col("EventTime").cast("string").alias("event"),
                F.col("LocalTime").cast("string").alias("local"),
            )
            .collect()
        )
        expected = [
            "2026-09-30 00:00:00",
            "2026-09-30 12:00:00",
            "2026-09-30 13:02:03",
            None,
            None,
            None,
            None,
            None,
            "1600-02-29 01:00:00",
        ]
        assert [row.event for row in rows] == expected
        assert [row.local for row in rows] == expected


@pytest.mark.parametrize("timestamp_type", ["TIMESTAMP_LTZ", "TIMESTAMP_NTZ"])
def test_custom_offsets_keep_explicit_timestamp_type_and_authored_date(
    classic_spark, timestamp_type
):
    values = [
        "30/09/2026 23:34:56 -05:00",
        "30/09/2026 23:34:56 +18:01",
        "30/09/2026 23:34:56 -05:00JUNK",
        '"},"value":"valid',
    ]
    pattern = "dd/MM/yyyy HH:mm:ss XXX"
    config = _config(
        _binding("EventTime", "timestamp", formats=[pattern]),
        _binding("CalendarDate", "date", formats=[pattern]),
    )
    with _settings(classic_spark, **{"spark.sql.timestampType": timestamp_type}):
        parsed = parser.parse_dataframe(
            _input(classic_spark, values), config, key_columns=["row_id"]
        ).parsed_df
        assert parsed.schema["EventTime"].dataType == T.TimestampType()
        rows = (
            parsed.orderBy("RowId")
            .select(
                F.col("EventTime").cast("string").alias("event"),
                F.col("CalendarDate").cast("string").alias("date"),
            )
            .collect()
        )
        assert [row.event for row in rows] == ["2026-10-01 04:34:56", None, None, None]
        assert [row.date for row in rows] == ["2026-09-30", None, None, None]


def test_configured_timestamp_failure_retains_explicit_type_and_raises(classic_spark):
    config = _config(_binding("EventTime", "timestamp", on_parse_error="fail"))
    with _settings(classic_spark, **{"spark.sql.timestampType": "TIMESTAMP_NTZ"}):
        parsed = parser.parse_dataframe(
            _input(classic_spark, ["bad"]), config, key_columns=["row_id"]
        ).parsed_df
        assert parsed.schema["EventTime"].dataType == T.TimestampType()
        # Spark 3.5 wraps executor errors in Py4JJavaError; newer releases expose PySparkException.
        from py4j.protocol import Py4JJavaError

        with pytest.raises((Py4JJavaError, PySparkException), match="Spark Parser could not parse"):
            parsed.collect()
