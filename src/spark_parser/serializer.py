"""Deterministic serialization and hashing for compiled parser configurations.

Serialization deliberately emits effective values, including inherited globals and defaults.
That makes reports self-contained and ensures a content hash identifies complete resolved
configuration content rather than the incidental shorthand used in the source YAML.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from spark_parser.enums import NUMERIC_PARSER_TYPES, ParserType
from spark_parser.models import ParserConfig, ParserOptions


class ParserConfigSerializer:
    """Produce explicit, JSON-compatible canonical parser metadata.

    Returned structures are newly allocated and safe for a caller to modify. Enum members,
    decimals, dates, timestamps, and tuples are converted into deterministic public representations
    before JSON encoding.
    """

    def to_mapping(self, config: ParserConfig) -> dict[str, Any]:
        """Return a fully resolved mapping suitable for reporting and hashing.

        The original column order is retained for human review even though canonical JSON later
        sorts mapping keys. Reordering configured columns remains a meaningful behavior change.
        """
        columns = [
            {
                "source_column_name": column.source_column_name,
                "target_column_name": column.target_column_name,
                "expected_data_type": column.expected_data_type,
                "parser": self.parser_mapping(column.parser),
            }
            for column in config.columns
        ]
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

    def parser_mapping(self, options: ParserOptions) -> dict[str, Any]:
        """Serialize one scalar parser's fully resolved behavior."""
        payload: dict[str, Any] = {
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
            payload["default_on_null"] = self._json_value(options.default_on_null)
        if options.default_on_error is not None:
            payload["default_on_error"] = self._json_value(options.default_on_error)
        if options.parser_type in NUMERIC_PARSER_TYPES:
            payload["zero_is_valid"] = options.zero_is_valid
        if options.parser_type is ParserType.STRING:
            payload["format"] = (
                options.string_format.value if options.string_format is not None else None
            )
        if options.parser_type in {
            ParserType.DATE,
            ParserType.TIMESTAMP,
            ParserType.TIMESTAMP_NTZ,
        }:
            payload["formats"] = list(options.formats)
        if options.parser_type is ParserType.BINARY:
            payload["encoding"] = options.binary_encoding.value
        if options.parser_type is ParserType.BOOLEAN:
            payload.update(
                true_values=list(options.true_values),
                false_values=list(options.false_values),
                boolean_case_sensitive=options.boolean_case_sensitive,
                boolean_values_mode=options.boolean_values_mode.value,
            )
        return payload

    def canonical_json(self, config: ParserConfig) -> str:
        """Return whitespace-free, key-sorted JSON for deterministic identity checks."""
        return json.dumps(
            self.to_mapping(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def content_hash(self, config: ParserConfig) -> str:
        """Return the SHA-256 identity of the complete resolved configuration content."""
        return hashlib.sha256(self.canonical_json(config).encode("utf-8")).hexdigest()

    @staticmethod
    def _json_value(value: Any) -> Any:
        """Convert a typed scalar default into a lossless JSON-compatible value."""
        # Decimal is rendered as text so binary floating-point conversion cannot change an exact
        # default before it contributes to a content hash.
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return value
