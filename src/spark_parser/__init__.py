"""Public package surface for Spark Parser."""

from spark_parser.compiler_yaml import YamlParserConfigCompiler
from spark_parser.defaults import PARSER_DEFAULTS
from spark_parser.enums import NullMarkersMode, ParseErrorMode, ParserType, StringFormat
from spark_parser.exceptions import CompilationError, SchemaValidationError, SparkParserError
from spark_parser.models import ColumnParser, ParserConfig, ParserGlobals, ParserOptions
from spark_parser.serializer import ParserConfigSerializer
from spark_parser.version import __version__

__all__ = [
    "ColumnParser",
    "CompilationError",
    "NullMarkersMode",
    "PARSER_DEFAULTS",
    "ParseErrorMode",
    "ParserConfig",
    "ParserConfigSerializer",
    "ParserGlobals",
    "ParserOptions",
    "ParserType",
    "SchemaValidationError",
    "SparkDataFrameParser",
    "SparkParserError",
    "StringFormat",
    "DataFrameParsing",
    "YamlParserConfigCompiler",
    "__version__",
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
