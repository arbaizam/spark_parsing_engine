"""High-level public service API and UAT configuration reporting."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spark_parser.compiler_yaml import YamlParserConfigCompiler
from spark_parser.defaults import PARSER_DEFAULTS
from spark_parser.enums import ParserType
from spark_parser.exceptions import CompilationError
from spark_parser.metadata import config_description, parser_description
from spark_parser.models import ParserConfig
from spark_parser.serializer import ParserConfigSerializer
from spark_parser.version import __version__


def _markdown_text(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple)):
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    else:
        rendered = str(value)
    return rendered.replace("|", "\\|").replace("\n", " ")


@dataclass(frozen=True)
class UatReviewReport:
    """Structured validation and resolved-behavior report for UAT review."""

    is_valid: bool
    source: str | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]
    validation_checks: tuple[dict[str, str], ...]
    column_reviews: tuple[dict[str, Any], ...]
    resolved_config: dict[str, Any] | None

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached JSON-compatible report mapping."""
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
        """Render the report as JSON for storage or downstream review tooling."""
        return json.dumps(self.to_mapping(), indent=indent, ensure_ascii=False, default=str)

    def to_markdown(self) -> str:
        """Render a human-readable Markdown UAT document."""
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
                "| Source | Silver | Expected type | Parser | Format | Nullable | Error mode | Audit |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                *(
                    "| "
                    f"{_markdown_text(column['source_column_name'])} | "
                    f"{_markdown_text(column['silver_column_name'])} | "
                    f"{_markdown_text(column['expected_data_type'])} | "
                    f"{_markdown_text(column['parser_type'])} | "
                    f"{_markdown_text(column['format_or_formats'])} | "
                    f"{_markdown_text(column['is_nullable'])} | "
                    f"{_markdown_text(column['on_parse_error'])} | "
                    f"{_markdown_text(column['audit'])} |"
                    for column in self.column_reviews
                ),
                "",
                "## Resolved parser options",
                "",
            ]
        )
        for column in self.column_reviews:
            lines.extend(
                [
                    f"### {_markdown_text(column['silver_column_name'])}",
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
        return "\n".join(lines)


class _ParserMetadataAccessor:
    def __init__(self, parser_type: ParserType) -> None:
        self._parser_type = parser_type

    def describe(self) -> dict[str, Any]:
        """Return arguments, defaults, behavior, and gotchas for this parser."""
        return parser_description(self._parser_type)


class _ConfigMetadataAccessor:
    @staticmethod
    def describe() -> dict[str, Any]:
        """Return top-level, global, and column configuration metadata."""
        return config_description()


class SparkParserService:
    """One entry point for compilation, parsing, metadata, and UAT review."""

    config = _ConfigMetadataAccessor()
    string = _ParserMetadataAccessor(ParserType.STRING)
    integer = _ParserMetadataAccessor(ParserType.INTEGER)
    long = _ParserMetadataAccessor(ParserType.LONG)
    decimal = _ParserMetadataAccessor(ParserType.DECIMAL)
    double = _ParserMetadataAccessor(ParserType.DOUBLE)
    boolean = _ParserMetadataAccessor(ParserType.BOOLEAN)
    date = _ParserMetadataAccessor(ParserType.DATE)
    timestamp = _ParserMetadataAccessor(ParserType.TIMESTAMP)

    def __init__(self) -> None:
        self._compiler = YamlParserConfigCompiler()
        self._serializer = ParserConfigSerializer()

    def describe(self, parser_type: str | ParserType | None = None) -> dict[str, Any]:
        """Describe one parser, or return the complete parser catalog."""
        if parser_type is None:
            return {member.value: parser_description(member) for member in ParserType}
        normalized = parser_type if isinstance(parser_type, ParserType) else ParserType(parser_type)
        return parser_description(normalized)

    @staticmethod
    def defaults() -> dict[str, Any]:
        """Return a detached view of all compiler defaults."""
        return deepcopy(PARSER_DEFAULTS)

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
        """Compile YAML text, a YAML path, or a YAML-compatible mapping."""
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
        """Compile when necessary and build parsed/audit Spark projections."""
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
        """Validate YAML and return a detailed structured UAT review report."""
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
            description = parser_description(column.parser.parser_type)
            resolved_column["parser"].setdefault("default_on_null", None)
            resolved_column["parser"].setdefault("default_on_error", None)
            parser_options = deepcopy(resolved_column["parser"])
            format_or_formats = parser_options.get("format", parser_options.get("formats"))
            column_reviews.append(
                {
                    "source_column_name": column.source_column_name,
                    "silver_column_name": column.silver_column_name,
                    "expected_data_type": column.expected_data_type,
                    "parser_type": column.parser.parser_type.value,
                    "format_or_formats": format_or_formats,
                    "is_nullable": column.parser.is_nullable,
                    "on_parse_error": column.parser.on_parse_error.value,
                    "audit": column.parser.audit,
                    "resolved_parser_options": parser_options,
                    "key_behaviors": description["key_behaviors"],
                    "gotchas": description["gotchas"],
                }
            )

        boolean_columns = [
            column for column in config.columns if column.parser.parser_type is ParserType.BOOLEAN
        ]
        nonnullable_count = sum(not column.parser.is_nullable for column in config.columns)
        error_default_count = sum(
            column.parser.on_parse_error.value == "default" for column in config.columns
        )
        type_pairs = sorted(
            {
                f"{column.parser.parser_type.value} -> {column.expected_data_type}"
                for column in config.columns
            }
        )
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
                "check": "Silver column uniqueness",
                "status": "PASS",
                "detail": (
                    f"Validated {len(config.columns)} non-empty silver name(s); all are unique."
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
                    f"{nonnullable_count} non-nullable mapping(s) have typed null defaults and "
                    f"{error_default_count} mapping(s) use typed parse-error defaults."
                ),
            },
            {
                "check": "Boolean vocabularies",
                "status": "PASS" if boolean_columns else "N/A",
                "detail": (
                    f"Validated non-empty, non-overlapping effective token sets for "
                    f"{len(boolean_columns)} Boolean mapping(s)."
                    if boolean_columns
                    else "No Boolean mappings are configured."
                ),
            },
        )
        summary = {
            "parser_config_id": config.parser_config_id,
            "parser_config_name": config.parser_config_name,
            "version": config.version,
            "content_hash": self._serializer.content_hash(config),
            "column_count": len(config.columns),
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
        if isinstance(source, Mapping):
            return self.compile_mapping(source), "mapping"
        if isinstance(source, Path):
            return self.compile_path(source), str(source)
        if not isinstance(source, str):
            raise TypeError("YAML source must be text, a path, or a mapping.")
        if "\n" not in source and "\r" not in source:
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
        if isinstance(source, Mapping):
            return "mapping"
        if isinstance(source, Path):
            return str(source)
        if isinstance(source, str):
            return "inline YAML" if "\n" in source or "\r" in source else source
        return None


parser = SparkParserService()
"""Convenience singleton for compilation, parsing, discovery, and UAT reporting."""
