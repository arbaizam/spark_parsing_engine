"""Public package surface for Spark Parser."""

from spark_parser.compiler_yaml import YamlParserConfigCompiler
from spark_parser.data_types import SparkDataType, SparkStructField, parse_spark_data_type
from spark_parser.defaults import PARSER_DEFAULTS
from spark_parser.enums import (
    BinaryEncoding,
    BooleanValuesMode,
    ChildErrorMode,
    ComplexInputFormat,
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
from spark_parser.models import (
    ColumnParser,
    NestedValueParser,
    ParserConfig,
    ParserGlobals,
    ParserOptions,
    StructFieldParser,
)
from spark_parser.serializer import ParserConfigSerializer
from spark_parser.service import SparkParserService, UatReviewReport, parser
from spark_parser.version import __version__

__all__ = [
    "ColumnParser",
    "CompilationError",
    "BinaryEncoding",
    "BooleanValuesMode",
    "ChildErrorMode",
    "ComplexInputFormat",
    "NullMarkersMode",
    "PARSER_DEFAULTS",
    "ParseErrorMode",
    "ParserConfig",
    "ParserConfigSerializer",
    "ParserGlobals",
    "ParserOptions",
    "ParserType",
    "NestedValueParser",
    "SchemaValidationError",
    "SchemaWarning",
    "SparkDataFrameParser",
    "SparkParserError",
    "SparkParserService",
    "SparkDataType",
    "SparkStructField",
    "StructFieldParser",
    "StringFormat",
    "DataFrameParsing",
    "YamlParserConfigCompiler",
    "UatReviewReport",
    "__version__",
    "parser",
    "parse_spark_data_type",
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
