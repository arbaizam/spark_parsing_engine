"""Behavioral regressions for domain formatter boundaries and canonical output."""

from collections.abc import Callable

import pytest
from test_spark_runtime import spark as _spark_fixture

# Keep the shared fixture's optional-PySpark skip ahead of direct Spark imports.
# isort: split
from pyspark.sql import Column, SparkSession
from pyspark.sql import functions as F

from spark_parser.address_formats import format_address_us_v1, format_county, format_state_us
from spark_parser.property_type_formats import format_property_type_v1

pytestmark = pytest.mark.spark
spark = _spark_fixture


def _assert_canonical_cases(
    spark: SparkSession,
    formatter: Callable[[Column], Column],
    cases: list[tuple[str | None, str | None]],
) -> None:
    source = spark.range(len(cases)).select(
        "id",
        F.element_at(
            F.array(*(F.lit(value).cast("string") for value, _ in cases)),
            (F.col("id") + 1).cast("integer"),
        ).alias("source"),
    )
    formatted = source.select("id", formatter(F.col("source")).alias("once"))
    rows = (
        formatted.select("id", "once", formatter(F.col("once")).alias("twice"))
        .orderBy("id")
        .collect()
    )
    expected = [value for _, value in cases]
    assert [row.once for row in rows] == expected
    assert [row.twice for row in rows] == expected


def test_state_lists_reject_empty_or_invalid_components(spark: SparkSession) -> None:
    _assert_canonical_cases(
        spark,
        format_state_us,
        [
            ("IL,", None),
            (",IL", None),
            (",IL,", None),
            ("Illinois, \u2003", None),
            ("IL,,TX", None),
            ("IL, ,TX", None),
            ("I,L", None),
            ("Washington, D.C.,", None),
            (",Washington, D.C.", None),
            ("Washington, D.C.,,Illinois", None),
            ("Washington, D.C.", "DC"),
            ("Washington, D.C., Illinois", "DC, IL"),
            ("Ill., \u2003TX", "IL, TX"),
            ("Puerto Rico", "PR"),
            (None, None),
        ],
    )


def test_address_unit_context_survives_punctuation_cleanup(spark: SparkSession) -> None:
    _assert_canonical_cases(
        spark,
        format_address_us_v1,
        [
            ("123 Main Street Apt , 4b", "123 Main St Apt 4B"),
            ("123 Main Street Apt . , 4b", "123 Main St Apt 4B"),
            ("123 Main Street Apt 4b", "123 Main St Apt 4B"),
            ("123 Main Street Suite , b12", "123 Main St Ste B12"),
            ("123 Center Street , Apt #4b", "123 Center St Apt #4B"),
            (", .", ""),
            (None, None),
        ],
    )


def test_address_and_county_do_not_truncate_long_tokens(spark: SparkSession) -> None:
    name = "mc" + "a" * 1_200
    canonical_name = "McA" + "a" * 1_199
    unit = "#4" + "b" * 1_200
    _assert_canonical_cases(
        spark,
        format_address_us_v1,
        [(f"{name} street Apt {unit}", f"{canonical_name} St Apt {unit.upper()}")],
    )
    _assert_canonical_cases(spark, format_county, [(name, f"{canonical_name} County")])


def test_mixed_use_parentheses_are_balanced_and_idempotent(spark: SparkSession) -> None:
    _assert_canonical_cases(
        spark,
        format_property_type_v1,
        [
            ("Mixed Use - ( OFFICE )", "Mixed Use - Office"),
            ("Mixed Use - ((OFFICE))", "Mixed Use - Office"),
            ("Mixed Use - ( ( OFFICE ) )", "Mixed Use - Office"),
            ("Mixed Use - (OFFICE) & (RETAIL)", "Mixed Use - (Office) & (retail)"),
            ("Mixed Use - ((OFFICE) (RETAIL))", "Mixed Use - (Office) (retail)"),
            ("Mixed Use - (OFFICE (MEDICAL))", "Mixed Use - Office (medical)"),
            ("Mixed Use - (OFFICE))", "Mixed Use - (Office))"),
            ("Mixed Use - ((OFFICE)", "Mixed Use - ((Office)"),
            ("Mixed Use - () - OFFICE", "Mixed Use - Office"),
            ("Mixed Use - (( ))", "Mixed Use"),
            ("Mixed Use - ((\U0001f3e2 OFFICE))", "Mixed Use - \U0001f3e2 Office"),
            ("Mixed Use - ((\u00c9COLE))", "Mixed Use - \u00c9cole"),
            ("Mixed Use - Condo (STORIES UNKNOWN)", "Mixed Use - Condo (stories unknown)"),
            (None, None),
        ],
    )
