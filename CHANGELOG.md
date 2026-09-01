# Changelog

## 0.5.0

- Narrow the package contract to top-level bronze strings and scalar Spark targets. Configured
  `array`, `struct`, and `map` parsing is intentionally removed; decode or flatten complex input
  upstream, parse its scalar leaves, and reconstruct complex output downstream when needed.
- Remove recursive datatype models, parser options, child-error policies, container defaults,
  serialization branches, metadata accessors, audit-path fields, and runtime expression carriers.
- Replace the recursive Spark DDL parser with a small scalar grammar that retains every scalar
  alias plus bounded `decimal(p,s)` validation.
- Simplify the runtime and audit contract without changing scalar normalization, display formats,
  typed defaults, error policies, schema checks, or deterministic configuration identity.
- Update the reference YAML, Databricks guide, system tests, and package documentation around the
  explicit scalar processing boundary.

## 0.4.0

- Remove cosmetic family/value separator hyphens from canonical interest-rate labels while
  retaining every former `Family - Value` form as a backward-compatible input alias.
- Let datetime metadata probes use the active Spark Connect/serverless session when its
  `SparkSession` does not expose `newSession`, including in the portable timestamp NTZ test.
- Add `title_business_v1` as ordinary title formatting plus exact `FHLB`, `P&I`, `UST`, `RCF`,
  and `CMT` exceptions and bounded numeric-hyphen capitalization; keep `title` unchanged.
- Add the fail-closed `interest_rate_index_v1` string profile for approved benchmark labels,
  compact month/year tenors, and exact vendor/source aliases while preserving `title` behavior.
- Keep the portable Spark integration suite independent of optional DataFrame cache APIs, which
  Databricks serverless compute rejects, while retaining and separately testing classic-Spark
  `DataFrameParsing.persist()` and `unpersist()` delegation; align the default storage level with
  PySpark's `MEMORY_AND_DISK_DESER` contract.
- Remove the recursive Java regular expression that overflowed executor stacks on ordinary long
  strings inside JSON arrays, structs, and maps; add 100,000-character and escape-heavy coverage.
- Carry each nested value, failure state, and diagnostic-path set through one bound Catalyst struct,
  converting audited and non-audited plan growth from exponential to linear while preserving exact
  array, struct, and map error/default/zero paths.
- Make `PARSER_DEFAULTS` deeply immutable with mapping proxies and tuple values; use
  `parser.defaults()` for the detached JSON-shaped dictionaries and lists.
- Report Spark analyzer-iteration or driver-stack exhaustion as a metadata-only schema-binding error
  with the active setting/depth and tuning guidance instead of leaking an opaque JVM failure.
- Preserve authored calendar fields for offset-bearing `date` formats independently of the Spark
  session timezone, while retaining offset support for explicitly configured date patterns.
- Make fractional timestamp defaults identical on Python 3.10 and newer, use true Java end-of-input
  anchors for numeric, Base64, and built-in datetime guards, and validate non-ASCII
  case-insensitive vocabularies during metadata-only Spark binding, including empty inputs.
- Share `dec` and `numeric` aliases across datatype and parser names, and reject underscores,
  surrounding whitespace, and non-ASCII digits in string-authored decimal defaults without removing
  supported scientific notation.
- Preserve source/target column and struct-field names verbatim, make `compile_yaml`/`review_yaml`
  file selection explicit through `pathlib.Path`, and expose configuration reviews as honest mutable
  data-transfer objects with detached `to_mapping()` copies.
- Add source-scoped branch-coverage enforcement, all Python 3.10/3.12 × PySpark 3.5/4.1 runtime
  boundary pairings, Python 3.13/PySpark 4.1 and Python 3.11 unit lanes, public-facade coverage,
  plan-size budgets, and strict sdist file selection.
- Organize test assets into explicit unit, integration, system, and fixture directories, and make
  the wide-plan regression test compatible with Spark Connect.
- Add a comprehensive Databricks user-guide notebook covering discovery, YAML authoring,
  configuration review, compilation, scalar and recursive parsing, audit output, errors, schema
  validation, configuration identity, and pipeline integration boundaries.
- Add `title` string formatting for display text that retains normalized spaces.
- Add `state_us` string formatting for the 50 US states and Washington, DC, including postal codes,
  punctuation-tolerant names, and an explicit set of conventional abbreviations.
- Add string-only `preserve` error handling so invalid formatted values can retain their exact raw
  tokens at top level and inside string struct fields, arrays, and map values.
- Accept US month-first 12-hour timestamp strings such as `09/30/2026 12:00 AM` and
  `09/30/2026 12:00:00 AM` as built-in fallbacks for date, timestamp, and
  timestamp-without-timezone parsing after their ISO formats.
- Accept local/offset-bearing ISO timestamps and optional microseconds while keeping
  timestamp-without-timezone input and defaults strictly timezone-free.
- Make malformed `timestamp_ntz` values honor configured error policies under ANSI mode.
- Reject incomplete nested numeric tokens and non-finite nested floating-point values.
- Treat strict-JSON failures and duplicate map keys as container parse errors instead of allowing
  silent null structs or Spark `DUPLICATED_MAP_KEY` failures.
- Honor `input_format: delimited` for recursively nested arrays.
- Resolve outer complex-parser `collapse_whitespace` to its effective value of `false` in
  serialization, configuration reviews, parser metadata, and row-level audit options.
- Include error, final-null default, and invalid-zero path arrays in the parser audit struct.
- Include top-level source and target column identity in nested fail-mode error messages.
- Treat only lowercase `null` as the JSON null literal; other spellings follow normal null-marker
  or parse-error behavior.
- Add recursive first-class `array`, `struct`, and string-keyed `map` parsers.
- Support JSON complex values plus literal-delimited scalar arrays without Python UDFs.
- Add `fail`, `null`, and `drop` child-error policies with nested audit paths.
- Add byte/tinyint, short/smallint, float/real, binary, and timestamp-without-timezone parsers.
- Parse and canonicalize nested Spark DDL types without requiring a running Spark session.
- Support recursive formatting, typed defaults, source-to-target struct field renaming, array
  deduplication/null removal, and map null-value removal.
- Render complex audit values as canonical JSON and binary audit values as base64.
- Expand Markdown configuration-review output with resolved globals, recursive schema trees,
  configuration metadata, and a copy-ready canonical YAML appendix.
- Preserve nulls through `address_us_v1`, including missing-source and non-null-default paths.
- Stabilize the nested audit-array schema for audited and non-audited configurations.
- Normalize address punctuation and contextual suffix/unit casing, including `#4B`.
- Normalize leading/trailing tabs, line breaks, and non-breaking spaces.
- Reject zero parse-error defaults when `zero_is_valid` is false.
- Align Boolean overlap validation with Spark runtime lowercase matching.
- Make canonical serialized mappings recompilable.
- Replace per-column projection chains with constant-depth, width-scalable Spark stages.
- Make configuration-review validation summaries evidence-based and improve compiler/path errors.
- Add Spark 3.5 and ANSI-mode regression coverage.
- Add a Java-required CI matrix across the Spark 3.5 floor and a newer Spark line so skipped runtime
  tests cannot produce a release green.
- Add a source-based Databricks system-test notebook covering real scalar and recursive Spark
  execution, error policies, ANSI parity, audit output, and input-schema safety.
- Add separate system-test and unit-test summaries with their execution contracts.
- Add global Boolean vocabularies with column `replace` and `extend` modes.
- Add `address_us_v1`, `county`, and `zip` string formats.
- Add the `SparkParserService`, discoverable parser metadata, and configuration-review reports.
