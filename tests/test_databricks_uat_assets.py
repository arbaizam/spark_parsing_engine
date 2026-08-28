"""Guard the checked-in Databricks UAT configuration and notebook contract."""

from pathlib import Path

from spark_parser import (
    ChildErrorMode,
    ParseErrorMode,
    ParserType,
    StringFormat,
    YamlParserConfigCompiler,
)

ROOT = Path(__file__).resolve().parents[1]
UAT_ROOT = ROOT / "databricks" / "uat"


def test_databricks_uat_config_compiles_with_required_coverage() -> None:
    """Keep the checked-in UAT fixture aligned with the parser behaviors it promises to validate."""
    config = YamlParserConfigCompiler().compile_path(UAT_ROOT / "spark_parser_uat.yaml")
    options = {column.silver_column_name: column.parser for column in config.columns}

    assert config.owner == "Data Engineering"
    assert {option.parser_type for option in options.values()} >= {
        ParserType.ARRAY,
        ParserType.STRUCT,
        ParserType.MAP,
    }
    assert options["Amount"].on_parse_error is ParseErrorMode.NULL
    assert options["Quantity"].on_parse_error is ParseErrorMode.DEFAULT
    assert options["LoanStatus"].string_format is StringFormat.TITLE
    assert options["StateCode"].string_format is StringFormat.STATE_US
    assert options["StateCode"].on_parse_error is ParseErrorMode.PRESERVE
    assert options["EventDate"].formats == (
        "yyyy-MM-dd",
        "MM/dd/yyyy hh:mm a",
        "MM/dd/yyyy hh:mm:ss a",
    )
    assert options["EventTimestamp"].formats == (
        "yyyy-MM-dd HH:mm:ss",
        "MM/dd/yyyy hh:mm a",
        "MM/dd/yyyy hh:mm:ss a",
    )
    assert options["EventTimestampNtz"].formats == options["EventTimestamp"].formats
    assert options["Aliases"].on_element_error is ChildErrorMode.DROP
    assert options["Profile"].field_parsers[1].parser.on_element_error is ChildErrorMode.NULL
    assert options["Attributes"].on_value_error is ChildErrorMode.DROP
    assert all(options[name].audit for name in options if name != "RecordId")


def test_databricks_uat_notebook_has_source_format_and_handoff_gates() -> None:
    """Catch accidental loss of Databricks source format or critical integration gates."""
    notebook = (UAT_ROOT / "spark_parser_uat.py").read_text(encoding="utf-8")

    # ``compile`` is a lightweight syntax check because Databricks-provided globals are unavailable
    # in local unit tests. String sentinels ensure review-critical operations remain present.
    assert notebook.startswith("# Databricks notebook source\n")
    compile(notebook, str(UAT_ROOT / "spark_parser_uat.py"), "exec")
    for required_contract in (
        "%pip install --no-deps --force-reinstall",
        'spark.conf.set("spark.sql.ansi.enabled", "true")',
        '.format("delta")',
        'mode("errorifexists")',
        "rules_engine_input_df",
        "rules_engine_parser_results_df",
    ):
        assert required_contract in notebook
