"""Deterministic serialization and hashing for compiled parser configurations.

Serialization deliberately emits effective values, including inherited globals and defaults.
That makes reports self-contained and ensures a content hash identifies runtime behavior rather
than the incidental shorthand used in the source YAML.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from spark_parser.enums import NUMERIC_PARSER_TYPES, ParserType
from spark_parser.models import ParserConfig, ParserOptions


class ParserConfigSerializer:
    """Produce explicit, JSON-compatible canonical parser metadata.

    Returned structures are newly allocated and safe for a caller to modify. Enum members,
    decimals, dates, timestamps, tuples, and nested parser nodes are converted into stable public
    representations before JSON encoding.
    """

    def to_mapping(self, config: ParserConfig) -> dict[str, Any]:
        """Return a fully resolved mapping suitable for reporting and hashing.

        The original column order is retained for human review even though canonical JSON later
        sorts mapping keys. Reordering configured columns remains a meaningful behavior change.
        """
        columns = [
            {
                "source_column_name": column.source_column_name,
                "silver_column_name": column.silver_column_name,
                "expected_data_type": column.expected_data_type,
                "parser": self.parser_mapping(
                    column.parser,
                    include_audit=True,
                    include_error_mode=True,
                ),
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

    def parser_mapping(
        self,
        options: ParserOptions,
        *,
        include_audit: bool,
        include_error_mode: bool,
    ) -> dict[str, Any]:
        """Serialize one parser node, recursively including applicable child parsers.

        Nested array/map children delegate immediate failures to the parent container policy, so
        callers can omit child ``on_parse_error`` fields. Struct fields own their error policy and
        therefore include it. ``include_audit`` follows the same top-level-only ownership rule.
        """
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
        }
        if include_error_mode:
            payload["on_parse_error"] = options.on_parse_error.value
        if include_audit:
            payload["audit"] = options.audit
        if options.default_on_null is not None:
            payload["default_on_null"] = self._json_value(options.default_on_null)
        if include_error_mode and options.default_on_error is not None:
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
        if options.parser_type is ParserType.ARRAY:
            assert options.element_parser is not None
            # Audit and direct parse-error policy belong to the array container. The element node
            # still carries normalization, nullability, formatting, and typed-default behavior.
            payload.update(
                input_format=options.input_format.value,
                element_parser=self.parser_mapping(
                    options.element_parser.parser,
                    include_audit=False,
                    include_error_mode=False,
                ),
                on_element_error=options.on_element_error.value,
                drop_null_elements=options.drop_null_elements,
                distinct=options.distinct,
            )
            if options.delimiter is not None:
                payload["delimiter"] = options.delimiter
        if options.parser_type is ParserType.STRUCT:
            # Preserve compiled schema order rather than authoring-list order. The compiler has
            # already aligned every configured field to its expected Spark struct field.
            payload.update(
                input_format=options.input_format.value,
                fields=[
                    {
                        "source_field_name": field.source_field_name,
                        "silver_field_name": field.silver_field_name,
                        "parser": self.parser_mapping(
                            field.parser,
                            include_audit=False,
                            include_error_mode=True,
                        ),
                    }
                    for field in options.field_parsers
                ],
            )
        if options.parser_type is ParserType.MAP:
            assert options.value_parser is not None
            payload.update(
                input_format=options.input_format.value,
                value_parser=self.parser_mapping(
                    options.value_parser.parser,
                    include_audit=False,
                    include_error_mode=False,
                ),
                on_value_error=options.on_value_error.value,
                drop_null_values=options.drop_null_values,
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
        """Return the SHA-256 identity of the configuration's resolved behavior."""
        return hashlib.sha256(self.canonical_json(config).encode("utf-8")).hexdigest()

    @staticmethod
    def _json_value(value: Any) -> Any:
        """Recursively convert typed defaults into lossless JSON-compatible values."""
        # Decimal is rendered as text so binary floating-point conversion cannot change an exact
        # default before it contributes to a content hash.
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, Mapping):
            # Mapping order is retained here for readable reports; canonical_json sorts keys when
            # byte-for-byte stability matters.
            return {key: ParserConfigSerializer._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ParserConfigSerializer._json_value(item) for item in value]
        return value
