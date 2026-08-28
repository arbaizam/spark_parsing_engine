"""Deterministic parser configuration serialization and hashing."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from spark_parser.enums import NUMERIC_PARSER_TYPES, ParserType
from spark_parser.models import ParserConfig


class ParserConfigSerializer:
    """Produce explicit, JSON-compatible canonical parser metadata."""

    def to_mapping(self, config: ParserConfig) -> dict[str, Any]:
        """Return a fully resolved mapping suitable for reporting and hashing."""
        columns: list[dict[str, Any]] = []
        for column in config.columns:
            options = column.parser
            parser_payload: dict[str, Any] = {
                "type": options.parser_type.value,
                "trim_whitespace": options.trim_whitespace,
                "collapse_whitespace": options.collapse_whitespace,
                "empty_is_null": options.empty_is_null,
                "replace_null_markers": options.replace_null_markers,
                "null_markers": list(options.null_markers),
                "null_markers_mode": options.null_markers_mode.value,
                "null_marker_case_sensitive": options.null_marker_case_sensitive,
                "is_nullable": options.is_nullable,
                "on_parse_error": options.on_parse_error.value,
                "audit": options.audit,
            }
            if options.default_on_null is not None:
                parser_payload["default_on_null"] = self._json_value(options.default_on_null)
            if options.default_on_error is not None:
                parser_payload["default_on_error"] = self._json_value(options.default_on_error)
            if options.parser_type in NUMERIC_PARSER_TYPES:
                parser_payload["zero_is_valid"] = options.zero_is_valid
            if options.parser_type is ParserType.STRING:
                parser_payload["format"] = (
                    options.string_format.value if options.string_format is not None else None
                )
            if options.parser_type in {ParserType.DATE, ParserType.TIMESTAMP}:
                parser_payload["formats"] = list(options.formats)
            if options.parser_type is ParserType.BOOLEAN:
                parser_payload.update(
                    true_values=list(options.true_values),
                    false_values=list(options.false_values),
                    boolean_case_sensitive=options.boolean_case_sensitive,
                    boolean_values_mode=options.boolean_values_mode.value,
                )
            columns.append(
                {
                    "source_column_name": column.source_column_name,
                    "silver_column_name": column.silver_column_name,
                    "expected_data_type": column.expected_data_type,
                    "parser": parser_payload,
                }
            )
        return {
            "parser_config_id": config.parser_config_id,
            "parser_config_name": config.parser_config_name,
            "version": config.version,
            "description": config.description,
            "owner": config.owner,
            "owner_department": config.owner_department,
            "globals": {
                "null_markers": list(config.globals.null_markers),
                "null_marker_case_sensitive": config.globals.null_marker_case_sensitive,
                "true_values": list(config.globals.true_values),
                "false_values": list(config.globals.false_values),
                "boolean_case_sensitive": config.globals.boolean_case_sensitive,
            },
            "columns": columns,
        }

    def canonical_json(self, config: ParserConfig) -> str:
        """Return stable canonical JSON for one config version."""
        return json.dumps(
            self.to_mapping(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def content_hash(self, config: ParserConfig) -> str:
        """Return the SHA-256 hash of canonical config content."""
        return hashlib.sha256(self.canonical_json(config).encode("utf-8")).hexdigest()

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return value
