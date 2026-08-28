"""Public package surface for Spark Parser."""

from spark_parser.compiler_yaml import YamlParserConfigCompiler
from spark_parser.defaults import PARSER_DEFAULTS
from spark_parser.enums import (
    BooleanValuesMode,
    NullMarkersMode,
    ParseErrorMode,
    ParserType,
    StringFormat,
)
from spark_parser.exceptions import (
    CompilationError,
    SchemaValidationError,
    SchemaWarning,
    SparkParserError,
)
from spark_parser.models import ColumnParser, ParserConfig, ParserGlobals, ParserOptions
from spark_parser.serializer import ParserConfigSerializer
from spark_parser.service import SparkParserService, UatReviewReport, parser
from spark_parser.version import __version__

__all__ = [
    "ColumnParser",
    "CompilationError",
    "BooleanValuesMode",
    "NullMarkersMode",
    "PARSER_DEFAULTS",
    "ParseErrorMode",
    "ParserConfig",
    "ParserConfigSerializer",
    "ParserGlobals",
    "ParserOptions",
    "ParserType",
    "SchemaValidationError",
    "SchemaWarning",
    "SparkDataFrameParser",
    "SparkParserError",
    "SparkParserService",
    "StringFormat",
    "DataFrameParsing",
    "YamlParserConfigCompiler",
    "UatReviewReport",
    "__version__",
    "parser",
]


def __getattr__(name: str):
    """Load Spark-backed objects only when requested."""
    if name == "DataFrameParsing":
        from spark_parser.dataframe_parsing import DataFrameParsing

        return DataFrameParsing
    if name == "SparkDataFrameParser":
        from spark_parser.spark_runtime import SparkDataFrameParser

        return SparkDataFrameParser
    raise AttributeError(name)
