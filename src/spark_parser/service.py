"""High-level API for compilation, parsing, discovery, and configuration review.

The service keeps callers away from internal compiler/runtime wiring and provides one consistent
entry point for notebooks, jobs, authoring tools, and tests. Report generation is intentionally
Spark-free: a team can review resolved behavior before binding a configuration to live data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from spark_parser.compiler_yaml import YamlParserConfigCompiler
from spark_parser.data_types import parse_spark_data_type
from spark_parser.defaults import PARSER_DEFAULTS
from spark_parser.enums import ParserType
from spark_parser.exceptions import CompilationError
from spark_parser.metadata import config_description, parser_description
from spark_parser.models import ColumnParser, ParserConfig, ParserOptions
from spark_parser.serializer import ParserConfigSerializer
from spark_parser.version import __version__


def _markdown_text(value: Any) -> str:
    """Render one arbitrary report value safely inside a Markdown table cell."""
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple)):
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    else:
        rendered = str(value)
    # Pipes would create extra columns and line breaks would break the row. Escape/flatten only the
    # structural Markdown characters; otherwise preserve the human-readable representation.
    return rendered.replace("|", "\\|").replace("\n", " ")


def _walk_parser_options(options: ParserOptions):
    """Yield a parser node followed by every recursive child in deterministic order."""
    yield options
    if options.element_parser is not None:
        yield from _walk_parser_options(options.element_parser.parser)
    for field in options.field_parsers:
        yield from _walk_parser_options(field.parser)
    if options.value_parser is not None:
        yield from _walk_parser_options(options.value_parser.parser)


def _schema_tree(column: ColumnParser) -> str:
    """Render one column's recursive source-to-target parser tree as plain text."""
    lines = [
        f"{column.target_column_name}: {column.expected_data_type} "
        f"[{column.parser.parser_type.value}] <- {column.source_column_name}"
    ]

    def append(options: ParserOptions, prefix: str) -> None:
        """Append child nodes using indentation while preserving compiled schema order."""
        if options.element_parser is not None:
            element = options.element_parser
            lines.append(
                f"{prefix}[]: {element.expected_data_type} "
                f"[{element.parser.parser_type.value}; on error={options.on_element_error.value}]"
            )
            append(element.parser, prefix + "  ")
        for field in options.field_parsers:
            lines.append(
                f"{prefix}.{field.target_field_name}: {field.expected_data_type} "
                f"[{field.parser.parser_type.value}] <- {field.source_field_name}"
            )
            append(field.parser, prefix + "  ")
        if options.value_parser is not None:
            value = options.value_parser
            lines.append(
                f"{prefix}{{value}}: {value.expected_data_type} "
                f"[{value.parser.parser_type.value}; on error={options.on_value_error.value}]"
            )
            append(value.parser, prefix + "  ")

    append(column.parser, "  ")
    return "\n".join(lines)


@dataclass(frozen=True)
class UatReviewReport:
    """Immutable structured result of reviewing one parser configuration.

    Invalid reports carry authoring errors without raising, which makes them suitable for review
    UIs and CI artifacts. Valid reports include the complete resolved configuration so reviewers do
    not have to infer inherited defaults from source shorthand.
    """

    is_valid: bool
    source: str | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]
    validation_checks: tuple[dict[str, str], ...]
    column_reviews: tuple[dict[str, Any], ...]
    resolved_config: dict[str, Any] | None

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached JSON-compatible report mapping safe for caller mutation."""
        # deepcopy protects this frozen report's nested dictionaries/lists from being modified via
        # a value returned to an authoring UI.
        return deepcopy(
            {
                "report_type": "spark_parser_uat_config_review",
                "engine_version": __version__,
                "is_valid": self.is_valid,
                "source": self.source,
                "errors": list(self.errors),
                "warnings": list(self.warnings),
                "summary": self.summary,
                "validation_checks": list(self.validation_checks),
                "column_reviews": list(self.column_reviews),
                "resolved_config": self.resolved_config,
            }
        )

    def to_json(self, *, indent: int = 2) -> str:
        """Render UTF-8-friendly JSON for durable review artifacts and automation."""
        return json.dumps(self.to_mapping(), indent=indent, ensure_ascii=False, default=str)

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the structured report as newline-terminated UTF-8 JSON and return its path."""
        target = Path(path)
        target.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return target

    def to_markdown(self) -> str:
        """Render a standalone human-readable Markdown review document.

        Invalid reports stop after errors/warnings because there is no trustworthy resolved
        configuration to describe. Valid reports include summaries, evidence-based checks, parser
        trees, effective options, behavioral guidance, and copy-ready canonical YAML.
        """
        status = "PASS" if self.is_valid else "FAIL"
        lines = [
            "# Spark Parser UAT Configuration Review",
            "",
            f"**Validation status:** {status}",
            f"**Source:** {_markdown_text(self.source)}",
            f"**Engine version:** {__version__}",
            "",
        ]
        if self.errors:
            lines.extend(["## Errors", ""])
            lines.extend(f"- {_markdown_text(error)}" for error in self.errors)
            lines.append("")
        if self.warnings:
            lines.extend(["## Warnings", ""])
            lines.extend(f"- {_markdown_text(warning)}" for warning in self.warnings)
            lines.append("")
        if not self.is_valid:
            # Never attempt to render tables from absent/partial config state. The invalid report is
            # still a useful artifact containing source identity and actionable compiler errors.
            return "\n".join(lines)

        lines.extend(
            [
                "## Configuration summary",
                "",
                "| Property | Value |",
                "| --- | --- |",
                *(
                    f"| {_markdown_text(key)} | {_markdown_text(value)} |"
                    for key, value in self.summary.items()
                ),
                "",
                "## Resolved globals",
                "",
                "| Setting | Effective value |",
                "| --- | --- |",
                *(
                    f"| {_markdown_text(key)} | {_markdown_text(value)} |"
                    for key, value in (self.resolved_config or {}).get("globals", {}).items()
                ),
                "",
                "## Validation checks",
                "",
                "| Check | Status | Detail |",
                "| --- | --- | --- |",
                *(
                    "| "
                    f"{_markdown_text(check['check'])} | {_markdown_text(check['status'])} | "
                    f"{_markdown_text(check['detail'])} |"
                    for check in self.validation_checks
                ),
                "",
                "## Column review",
                "",
                "| Source | Target | Expected type | Parser | Format | Nullable | Error mode | Audit |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                *(
                    "| "
                    f"{_markdown_text(column['source_column_name'])} | "
                    f"{_markdown_text(column['target_column_name'])} | "
                    f"{_markdown_text(column['expected_data_type'])} | "
                    f"{_markdown_text(column['parser_type'])} | "
                    f"{_markdown_text(column['format_or_formats'])} | "
                    f"{_markdown_text(column['is_nullable'])} | "
                    f"{_markdown_text(column['on_parse_error'])} | "
                    f"{_markdown_text(column['audit'])} |"
                    for column in self.column_reviews
                ),
                "",
                "## Resolved schema and parser tree",
                "",
                *(
                    f"### {_markdown_text(column['target_column_name'])}\n\n"
                    f"```text\n{column['schema_tree']}\n```\n"
                    for column in self.column_reviews
                ),
                "## Resolved parser options",
                "",
            ]
        )
        # Options differ by parser type, so one small table per column is clearer and more consistent
        # than a very wide union of every possible parser argument.
        for column in self.column_reviews:
            lines.extend(
                [
                    f"### {_markdown_text(column['target_column_name'])}",
                    "",
                    "| Option | Effective value |",
                    "| --- | --- |",
                    *(
                        f"| {_markdown_text(key)} | {_markdown_text(value)} |"
                        for key, value in column["resolved_parser_options"].items()
                    ),
                    "",
                    "Key behaviors:",
                    "",
                    *(f"- {behavior}" for behavior in column["key_behaviors"]),
                    "",
                ]
            )
            if column["gotchas"]:
                lines.extend(
                    [
                        "Gotchas:",
                        "",
                        *(f"- {gotcha}" for gotcha in column["gotchas"]),
                        "",
                    ]
                )
        # YAML is generated from already-resolved JSON-compatible data. It is documentation, not a
        # round trip through untrusted Python object constructors.
        resolved_yaml = yaml.safe_dump(
            self.resolved_config,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).rstrip()
        lines.extend(
            [
                "## Canonical resolved configuration",
                "",
                "This copy-ready YAML includes inherited values and every effective default.",
                "",
                "```yaml",
                resolved_yaml,
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    def write_markdown(self, path: str | Path) -> Path:
        """Write newline-terminated UTF-8 Markdown and return the resulting path."""
        target = Path(path)
        target.write_text(self.to_markdown() + "\n", encoding="utf-8")
        return target


class _ParserMetadataAccessor:
    """Bind a discoverable service attribute to one canonical parser type."""

    def __init__(self, parser_type: ParserType) -> None:
        """Store the parser type described by this lightweight accessor."""
        self._parser_type = parser_type

    def describe(self) -> dict[str, Any]:
        """Return arguments, defaults, behavior, and gotchas for this parser."""
        return parser_description(self._parser_type)


class _ConfigMetadataAccessor:
    """Expose configuration authoring metadata through ``parser.config``."""

    @staticmethod
    def describe() -> dict[str, Any]:
        """Return top-level, global, and column configuration metadata."""
        return config_description()


class SparkParserService:
    """Facade for the package's compiler, serializer, runtime, and review metadata.

    Parser accessor attributes are stateless and shared. Compiler and serializer instances are
    created per service so callers may instantiate isolated facades for dependency injection while
    most users rely on the module-level :data:`parser` singleton.
    """

    # Attribute access such as ``parser.decimal.describe()`` is intentionally discoverable in
    # notebooks and IDE completion; users need not memorize a separate metadata registry API.
    config = _ConfigMetadataAccessor()
    string = _ParserMetadataAccessor(ParserType.STRING)
    byte = _ParserMetadataAccessor(ParserType.BYTE)
    short = _ParserMetadataAccessor(ParserType.SHORT)
    integer = _ParserMetadataAccessor(ParserType.INTEGER)
    long = _ParserMetadataAccessor(ParserType.LONG)
    float = _ParserMetadataAccessor(ParserType.FLOAT)
    decimal = _ParserMetadataAccessor(ParserType.DECIMAL)
    double = _ParserMetadataAccessor(ParserType.DOUBLE)
    binary = _ParserMetadataAccessor(ParserType.BINARY)
    boolean = _ParserMetadataAccessor(ParserType.BOOLEAN)
    date = _ParserMetadataAccessor(ParserType.DATE)
    timestamp = _ParserMetadataAccessor(ParserType.TIMESTAMP)
    timestamp_ntz = _ParserMetadataAccessor(ParserType.TIMESTAMP_NTZ)
    array = _ParserMetadataAccessor(ParserType.ARRAY)
    struct = _ParserMetadataAccessor(ParserType.STRUCT)
    map = _ParserMetadataAccessor(ParserType.MAP)

    def __init__(self) -> None:
        """Create the stateless compiler and serializer collaborators used by this facade."""
        self._compiler = YamlParserConfigCompiler()
        self._serializer = ParserConfigSerializer()

    def describe(self, parser_type: str | ParserType | None = None) -> dict[str, Any]:
        """Describe one parser, or return a fresh complete parser catalog mapping."""
        if parser_type is None:
            return {member.value: parser_description(member) for member in ParserType}
        try:
            normalized = (
                parser_type if isinstance(parser_type, ParserType) else ParserType(parser_type)
            )
        except ValueError as exc:
            allowed = ", ".join(member.value for member in ParserType)
            raise CompilationError(
                f"Unknown parser type {parser_type!r}; expected one of: {allowed}."
            ) from exc
        return parser_description(normalized)

    @staticmethod
    def defaults() -> dict[str, Any]:
        """Return a detached view of all compiler defaults safe for caller mutation."""
        return deepcopy(PARSER_DEFAULTS)

    @staticmethod
    def normalize_data_type(value: str) -> str:
        """Validate supported Spark DDL and return its canonical representation without Spark."""
        return parse_spark_data_type(value).canonical

    def compile_text(self, text: str) -> ParserConfig:
        """Compile YAML text into an immutable, fully resolved configuration."""
        return self._compiler.compile_text(text)

    def compile_path(self, path: str | Path) -> ParserConfig:
        """Compile a UTF-8 YAML file."""
        return self._compiler.compile_path(path)

    def compile_mapping(self, payload: Mapping[str, Any]) -> ParserConfig:
        """Compile an already-loaded YAML-compatible mapping."""
        return self._compiler.compile_mapping(payload)

    def compile_yaml(self, source: str | Path | Mapping[str, Any]) -> ParserConfig:
        """Compile YAML text, an existing YAML path, or a YAML-compatible mapping."""
        config, _ = self._compile_source(source)
        return config

    def to_mapping(self, config: ParserConfig) -> dict[str, Any]:
        """Serialize all resolved config values to a JSON-compatible mapping."""
        return self._serializer.to_mapping(config)

    def canonical_json(self, config: ParserConfig) -> str:
        """Return deterministic canonical JSON for a compiled config."""
        return self._serializer.canonical_json(config)

    def content_hash(self, config: ParserConfig) -> str:
        """Return the SHA-256 identity of fully resolved config content."""
        return self._serializer.content_hash(config)

    def parse_dataframe(
        self,
        df: Any,
        config: ParserConfig | str | Path | Mapping[str, Any],
        *,
        key_columns: Sequence[str] | None = None,
        column_prefix: str = "spark_parser",
    ):
        """Compile when necessary and build lazy parsed/audit Spark projections.

        Importing the runtime inside this method preserves the package's Spark-free compiler and
        metadata use cases. Passing an already compiled config avoids redundant authoring work.
        """
        resolved = config if isinstance(config, ParserConfig) else self.compile_yaml(config)
        from spark_parser.spark_runtime import SparkDataFrameParser

        return SparkDataFrameParser().parse_dataframe(
            df,
            resolved,
            key_columns=key_columns,
            column_prefix=column_prefix,
        )

    def review_yaml(
        self,
        source: str | Path | Mapping[str, Any],
    ) -> UatReviewReport:
        """Validate authoring input and return a detailed report instead of raising.

        Only expected input/compilation failures are converted into an invalid report. Programming
        errors outside that boundary should still surface normally rather than being hidden.
        """
        source_label: str | None = None
        try:
            config, source_label = self._compile_source(source)
        except (CompilationError, TypeError, ValueError, OSError) as exc:
            return UatReviewReport(
                is_valid=False,
                source=source_label or self._source_label(source),
                errors=(str(exc),),
                warnings=(),
                summary={},
                validation_checks=(),
                column_reviews=(),
                resolved_config=None,
            )

        resolved = self._serializer.to_mapping(config)
        report_warnings: list[str] = []
        # Ownership metadata is not required for execution, but missing values reduce the usefulness
        # of a UAT artifact and therefore deserve explicit review warnings.
        if config.description is None:
            report_warnings.append("description is not set; UAT scope may be less clear.")
        if config.owner is None:
            report_warnings.append("owner is not set.")
        if config.owner_department is None:
            report_warnings.append("owner_department is not set.")
        audited_count = sum(column.parser.audit for column in config.columns)
        if audited_count == 0:
            report_warnings.append(
                "No columns have audit enabled; results_df will contain an empty parse-results array."
            )

        # Reusing one bronze source for multiple target interpretations is allowed. Report it so a
        # reviewer can distinguish an intentional fan-out from accidental duplicate authoring.
        repeated_sources = sorted(
            {
                column.source_column_name
                for column in config.columns
                if sum(
                    candidate.source_column_name == column.source_column_name
                    for candidate in config.columns
                )
                > 1
            }
        )
        column_reviews: list[dict[str, Any]] = []
        resolved_columns = resolved["columns"]
        for column, resolved_column in zip(config.columns, resolved_columns, strict=True):
            # ``strict=True`` asserts the serializer preserved one-to-one column correspondence.
            description = parser_description(column.parser.parser_type)
            parser_options = deepcopy(resolved_column["parser"])
            parser_options.setdefault("default_on_null", None)
            parser_options.setdefault("default_on_error", None)
            format_or_formats = parser_options.get("format", parser_options.get("formats"))
            column_reviews.append(
                {
                    "source_column_name": column.source_column_name,
                    "target_column_name": column.target_column_name,
                    "expected_data_type": column.expected_data_type,
                    "parser_type": column.parser.parser_type.value,
                    "format_or_formats": format_or_formats,
                    "is_nullable": column.parser.is_nullable,
                    "on_parse_error": column.parser.on_parse_error.value,
                    "audit": column.parser.audit,
                    "resolved_parser_options": parser_options,
                    "schema_tree": _schema_tree(column),
                    "key_behaviors": description["key_behaviors"],
                    "gotchas": description["gotchas"],
                }
            )

        # Flatten the recursive parser forest once so validation evidence includes nested parser
        # nodes rather than reporting only top-level columns.
        all_options = [
            options for column in config.columns for options in _walk_parser_options(column.parser)
        ]
        boolean_columns = [
            options for options in all_options if options.parser_type is ParserType.BOOLEAN
        ]
        nonnullable_count = sum(not options.is_nullable for options in all_options)
        error_default_count = sum(
            options.on_parse_error.value == "default" for options in all_options
        )
        type_pairs = sorted(
            {
                f"{column.parser.parser_type.value} -> {column.expected_data_type}"
                for column in config.columns
            }
        )
        # Checks describe concrete compiler evidence. They are not a second validation engine; a
        # PASS means the strict compiler already proved the stated invariant.
        checks = (
            {
                "check": "YAML and metadata contract",
                "status": "PASS",
                "detail": (
                    f"Compiled config {config.parser_config_id!r} version {config.version!r} "
                    f"with {len(config.columns)} column mapping(s); no duplicate or unknown "
                    "keys were accepted."
                ),
            },
            {
                "check": "Target column uniqueness",
                "status": "PASS",
                "detail": (
                    f"Validated {len(config.columns)} non-empty target name(s); all are unique."
                ),
            },
            {
                "check": "Parser/type compatibility",
                "status": "PASS",
                "detail": "Validated effective parser/type pairs: " + ", ".join(type_pairs) + ".",
            },
            {
                "check": "Defaults and conditional options",
                "status": "PASS",
                "detail": (
                    "Resolved every optional value; "
                    f"{nonnullable_count} non-nullable parser node(s) have typed null defaults and "
                    f"{error_default_count} parser node(s) use typed parse-error defaults."
                ),
            },
            {
                "check": "Boolean vocabularies",
                "status": "PASS" if boolean_columns else "N/A",
                "detail": (
                    f"Validated non-empty, non-overlapping effective token sets for "
                    f"{len(boolean_columns)} Boolean parser node(s)."
                    if boolean_columns
                    else "No Boolean parser nodes are configured."
                ),
            },
        )
        summary = {
            "parser_config_id": config.parser_config_id,
            "parser_config_name": config.parser_config_name,
            "version": config.version,
            "content_hash": self._serializer.content_hash(config),
            "minimum_spark_version": "3.5",
            "column_count": len(config.columns),
            "parser_node_count": len(all_options),
            "audited_column_count": audited_count,
            "repeated_source_columns": repeated_sources,
            "description": config.description,
            "owner": config.owner,
            "owner_department": config.owner_department,
        }
        return UatReviewReport(
            is_valid=True,
            source=source_label,
            errors=(),
            warnings=tuple(report_warnings),
            summary=summary,
            validation_checks=checks,
            column_reviews=tuple(column_reviews),
            resolved_config=resolved,
        )

    def _compile_source(
        self,
        source: str | Path | Mapping[str, Any],
    ) -> tuple[ParserConfig, str]:
        """Disambiguate mapping, path, and inline-text inputs and return a source label."""
        if isinstance(source, Mapping):
            return self.compile_mapping(source), "mapping"
        if isinstance(source, Path):
            return self.compile_path(source), str(source)
        if not isinstance(source, str):
            raise TypeError("YAML source must be text, a path, or a mapping.")
        if "\n" not in source and "\r" not in source:
            # A single-line string may be either YAML text or a path. Existing files win; a missing
            # string ending in .yaml/.yml is treated as a path typo instead of confusing YAML.
            try:
                path = Path(source)
                if path.is_file():
                    return self.compile_path(path), str(path)
            except OSError:
                pass
            if Path(source).suffix.lower() in {".yaml", ".yml"}:
                raise CompilationError(f"YAML file does not exist: {source}")
        return self.compile_text(source), "inline YAML"

    @staticmethod
    def _source_label(source: Any) -> str | None:
        """Return a best-effort report label when compilation failed before source resolution."""
        if isinstance(source, Mapping):
            return "mapping"
        if isinstance(source, Path):
            return str(source)
        if isinstance(source, str):
            return "inline YAML" if "\n" in source or "\r" in source else source
        return None


# Most consumers need no mutable service state, so a package-level singleton is the ergonomic
# default. Tests and dependency-injected applications may still instantiate SparkParserService.
parser = SparkParserService()
"""Convenience singleton for compilation, parsing, discovery, and UAT reporting."""
