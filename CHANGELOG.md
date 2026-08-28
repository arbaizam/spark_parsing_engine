# Changelog

## 0.4.0

- Add `title` string formatting for display text that retains normalized spaces.
- Add `state_us` string formatting for the 50 US states and Washington, DC.
- Accept US month-first 12-hour timestamp strings such as `09/30/2026 12:00 AM` and
  `09/30/2026 12:00:00 AM` as built-in fallbacks for date, timestamp, and
  timestamp-without-timezone parsing after their ISO formats.
- Make malformed `timestamp_ntz` values honor configured error policies under ANSI mode.
- Reject incomplete nested numeric tokens and non-finite nested floating-point values.
- Treat strict-JSON failures and duplicate map keys as container parse errors instead of allowing
  silent null structs or Spark `DUPLICATED_MAP_KEY` failures.
- Honor `input_format: delimited` for recursively nested arrays.
- Resolve outer complex-parser `collapse_whitespace` to its effective value of `false` in
  serialization, UAT reports, parser metadata, and row-level audit options.
- Include `nested_error_paths` in the stable parser audit struct.
- Include top-level source and silver column identity in nested fail-mode error messages.
- Treat only lowercase `null` as the JSON null literal; other spellings follow normal null-marker
  or parse-error behavior.
- Add recursive first-class `array`, `struct`, and string-keyed `map` parsers.
- Support JSON complex values plus literal-delimited scalar arrays without Python UDFs.
- Add `fail`, `null`, and `drop` child-error policies with nested audit paths.
- Add byte/tinyint, short/smallint, float/real, binary, and timestamp-without-timezone parsers.
- Parse and canonicalize nested Spark DDL types without requiring a running Spark session.
- Support recursive formatting, typed defaults, source-to-silver struct field renaming, array
  deduplication/null removal, and map null-value removal.
- Render complex audit values as canonical JSON and binary audit values as base64.
- Expand Markdown UAT output with resolved globals, recursive schema trees, configuration metadata,
  and a copy-ready canonical YAML appendix.
- Preserve nulls through `address_us_v1`, including missing-source and non-null-default paths.
- Stabilize the nested audit-array schema for audited and non-audited configurations.
- Normalize address punctuation and contextual suffix/unit casing, including `#4B`.
- Normalize leading/trailing tabs, line breaks, and non-breaking spaces.
- Reject zero parse-error defaults when `zero_is_valid` is false.
- Align Boolean overlap validation with Spark runtime lowercase matching.
- Make canonical serialized mappings recompilable.
- Replace per-column projection chains with constant-depth, width-scalable Spark stages.
- Make UAT validation summaries evidence-based and improve compiler/path errors.
- Add Spark 3.5 and ANSI-mode regression coverage.
- Add global Boolean vocabularies with column `replace` and `extend` modes.
- Add `address_us_v1`, `county`, and `zip` string formats.
- Add the `SparkParserService`, discoverable parser metadata, and UAT reports.
