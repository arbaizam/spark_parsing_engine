# Spark Parser

Spark Parser converts load-specific bronze string columns into explicitly typed silver
columns using strict YAML configuration. All row transformations use native Spark SQL
expressions—there are no Python or pandas UDFs.

The package provides:

- strict YAML compilation with resolved, code-owned defaults;
- source-to-silver column mapping and exact Spark datatype validation;
- string, integer, long, decimal, double, Boolean, date, and timestamp parsers;
- native address, county, and ZIP normalization profiles;
- optional row-level audit structs aligned to rules-engine-style output;
- a discoverable service API (`parser.string.describe()`); and
- structured JSON or Markdown UAT configuration reports.

## Quick start

```yaml
parser_config_id: bronze_customer_load
parser_config_name: Bronze Customer Load
version: 1.0.0
description: Parse one bronze customer delivery for silver.
owner: Data Engineering
owner_department: Enterprise Data

globals:
  null_markers: [NA, "Null", N/A]
  null_marker_case_sensitive: false
  true_values: ["true", Y, "yes"]
  false_values: ["false", N, "no"]
  boolean_case_sensitive: false

columns:
  - source_column_name: customer_name
    silver_column_name: CustomerName
    expected_data_type: string
    parser:
      type: string
      audit: true

  - source_column_name: mailing_address
    silver_column_name: MailingAddress
    expected_data_type: string
    parser:
      type: string
      format: address_us_v1
      audit: true

  - source_column_name: account_balance
    silver_column_name: AccountBalance
    expected_data_type: decimal(18,2)
    parser:
      type: decimal
      replace_null_markers: true
      zero_is_valid: false
      on_parse_error: null
      audit: true
```

```python
from spark_parser import parser

config = parser.compile_path("customer_parser.yaml")
parsing = parser.parse_dataframe(
    bronze_df,
    config,
    key_columns=["load_id", "row_id"],
)

silver_df = parsing.parsed_df
audit_df = parsing.results_df
schema_warnings = parsing.warnings
```

`parser` is the package-level `SparkParserService` singleton. You may instantiate
`SparkParserService()` when dependency injection or isolated service objects are preferable.

## Service API

The service is the recommended entry point for common package operations.

| Method/property | Result and behavior |
| --- | --- |
| `parser.compile_text(text)` | Compile YAML text to an immutable, fully resolved `ParserConfig`; raises `CompilationError` on invalid authoring metadata. |
| `parser.compile_path(path)` | Read and compile a UTF-8 YAML file. |
| `parser.compile_mapping(mapping)` | Compile an already-loaded YAML-compatible mapping. |
| `parser.compile_yaml(source)` | Convenience dispatcher accepting YAML text, a path, or a mapping. |
| `parser.parse_dataframe(df, config, ...)` | Parse a DataFrame. `config` may already be compiled or may be any input accepted by `compile_yaml`. |
| `parser.to_mapping(config)` | Return a JSON-compatible mapping containing every resolved option. |
| `parser.canonical_json(config)` | Return deterministic canonical JSON. |
| `parser.content_hash(config)` | Return the SHA-256 identity of the resolved configuration. |
| `parser.defaults()` | Return a detached mapping of all code-owned defaults. |
| `parser.describe()` | Return metadata for every parser type. |
| `parser.describe("date")` | Return metadata for one parser by name. |
| `parser.string.describe()` | Return the string parser's arguments, defaults, behavior, and gotchas. The same accessor exists for every parser type. |
| `parser.config.describe()` | Return top-level, global, and column metadata definitions. |
| `parser.review_yaml(source)` | Validate YAML and return a `UatReviewReport` instead of raising for authoring errors. |

Parser metadata accessors are `string`, `integer`, `long`, `decimal`, `double`, `boolean`,
`date`, and `timestamp`. Descriptions are machine-readable dictionaries so notebooks,
catalogs, and authoring UIs can render the same defaults used by the compiler.

```python
string_help = parser.string.describe()
for argument in string_help["arguments"]:
    print(argument["name"], argument["required"], argument["default"])

all_defaults = parser.defaults()
config_help = parser.config.describe()
```

## UAT configuration report

`review_yaml()` accepts YAML text, a `Path`/path string, or a mapping. A valid report contains:

- config identity, owner metadata, version, and canonical content hash;
- an evidence-based summary of the compiler validations the config passed, with `N/A` for
  checks that do not apply;
- source-to-silver mappings and exact expected datatypes;
- every fully resolved parser option, including inherited globals and defaults;
- effective error, nullability, formatting, and audit behavior;
- parser-specific key behaviors and gotchas; and
- review warnings such as missing ownership metadata or no audited columns.

It does not inspect a DataFrame, so input-schema warnings such as a missing source column are
reported later by `parse_dataframe()`.

```python
report = parser.review_yaml("customer_parser.yaml")

if not report.is_valid:
    raise ValueError(report.errors)

display(report.to_markdown())
report_json = report.to_json()
report_payload = report.to_mapping()
```

`UatReviewReport` properties are `is_valid`, `source`, `errors`, `warnings`, `summary`,
`validation_checks`, `column_reviews`, and `resolved_config`. `to_mapping()` returns a detached
JSON-compatible structure; `to_json(indent=2)` and `to_markdown()` provide storage and human
review formats. An invalid report contains its compiler errors and renders a `FAIL` status
without raising the compilation exception.

## Configuration metadata

### Top-level arguments

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `parser_config_id` | Yes | — | Stable, non-empty ID for the parsing configuration. |
| `parser_config_name` | Yes | — | Non-empty human-readable name. |
| `version` | Yes | — | Non-empty version string. Versions remain important; lifecycle status is intentionally absent. |
| `columns` | Yes | — | Non-empty ordered list of source-to-silver parser mappings. |
| `description` | No | `null` | Purpose and UAT scope. |
| `owner` | No | `null` | Accountable person or team. |
| `owner_department` | No | `null` | Accountable department. |
| `globals` | No | `{}` | Global null and Boolean vocabularies inherited by columns. |

Duplicate YAML keys, non-string metadata keys, and unknown arguments are rejected. Quote
numeric-looking versions such as `"1"` so YAML retains them as strings.
YAML merge keys (`<<`) are intentionally unsupported because they can hide inherited keys from
a strict review; ordinary anchors and aliases remain usable.

### Global arguments

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `null_markers` | No | `[]` | Ordered global null-token strings. Duplicates are removed. Tokens are inert until a column sets `replace_null_markers: true`. |
| `null_marker_case_sensitive` | No | `true` | Default exact-case null-token matching. |
| `true_values` | No | `["true"]` | Non-empty global tokens mapped to Boolean true. |
| `false_values` | No | `["false"]` | Non-empty global tokens mapped to Boolean false. |
| `boolean_case_sensitive` | No | `false` | Whether global Boolean-token matching requires exact case. The default is case-insensitive. |

When null-marker case sensitivity is `true`, `NA` matches only `NA`. When it is `false`, the
normalized input and all markers are lowercased before comparison, so `NA`, `na`, and `Na`
match the same marker. Matching occurs after whitespace collapse and trim. Each column may
override this global setting.

Quote Boolean-looking YAML tokens—especially `"true"`, `"false"`, `"yes"`, `"no"`, `"on"`,
and `"off"`. Without quotes, YAML can load them as actual booleans rather than bronze string
tokens, and strict compilation rejects them.

### Column arguments

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `source_column_name` | Yes | — | Exact top-level bronze name. It may be reused by multiple mappings. A missing source warns rather than failing. |
| `silver_column_name` | Yes | — | Non-empty, unique name emitted by `parsed_df`. Duplicate silver names fail compilation. |
| `expected_data_type` | Yes | — | Exact target Spark type: `string`, `integer`, `long`, `decimal(p,s)`, `double`, `boolean`, `date`, or `timestamp`. Aliases `int` and `bigint` compile to `integer` and `long`. |
| `parser` | Yes | — | Matching scalar parser name or a mapping containing `type` and options. |

`expected_data_type` is intentionally separate from `parser.type`: the parser selects the
conversion implementation, while the expected type fixes the silver schema—including integer
width and decimal precision/scale. The compiler requires them to agree.

Scalar form such as `parser: date` applies every safe default. Mapping form is required when
setting any option:

```yaml
- source_column_name: arm_next_rate_change_date
  silver_column_name: ArmNextRateChangeDate
  expected_data_type: date
  parser:
    type: date
    audit: true
```

If `arm_next_rate_change_date` is absent from the DataFrame, parsing emits a `SchemaWarning`,
adds a message to `DataFrameParsing.warnings`, and produces a typed null. For a mapping with
`is_nullable: false`, its required `default_on_null` is assigned instead. An audited mapping
records `source_column_missing`, sets `effective` to false, and supplies a missing-source error.

## Common parser arguments

These options apply to every parser type.

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `type` | Mapping form only | — | Parser implementation matching `expected_data_type`. |
| `collapse_whitespace` | No | `true` | Replace every run of whitespace—left, right, and internal—with one ordinary space. |
| `trim_whitespace` | No | `true` | Remove leading and trailing spaces, tabs, line breaks, and non-breaking spaces after collapse. Both defaults together normalize all surrounding and internal whitespace. |
| `empty_is_null` | No | `true` | Convert an empty normalized string to null. |
| `replace_null_markers` | No | `false` | Convert effective null-token matches to null. |
| `null_markers` | No | Inherited globals | Column null-token list. Supplying it does not by itself enable replacement. |
| `null_markers_mode` | No | `replace` | With column `null_markers`, `replace` uses only that list and `extend` appends it to globals. It cannot be supplied without column markers. |
| `null_marker_case_sensitive` | No | Inherited global | Override null-token case sensitivity for this column. |
| `is_nullable` | No | `true` | Allow the final silver value to remain null. |
| `default_on_null` | Conditional | No default | Required only when `is_nullable: false`; must be non-null and exactly valid for the expected type. |
| `on_parse_error` | No | `fail` | `fail`, `null`, or `default`; see the error-mode section below. |
| `default_on_error` | Conditional | No default | Required only with `on_parse_error: default`; must fit the expected type. |
| `audit` | No | `false` | Add a row-level audit struct for this column. |

Compilation rejects misplaced parser-specific arguments, invalid option types, unknown keys,
incompatible typed defaults, empty required lists, and contradictory conditional arguments.

## Parser-specific reference

### String

String supports the common arguments plus `format`.

| `format` | Result | Example |
| --- | --- | --- |
| `null` or `none` | Preserve case after whitespace normalization. | `"  Acme   LLC "` → `"Acme LLC"` |
| `lower` | Lowercase the normalized value. | `"Acme LLC"` → `"acme llc"` |
| `upper` | Uppercase the normalized value. | `"Acme LLC"` → `"ACME LLC"` |
| `pascal` | Lowercase, title-case, and remove spaces. Intended for identifiers rather than human names. | `"account status"` → `"AccountStatus"` |
| `address_us_v1` | Apply deterministic US address display normalization. | `"123 mccormick st. apt 4b"` → `"123 McCormick St Apt 4B"` |
| `county` | Smart-case a county name and ensure exactly one trailing `County`. | `"mclean county"` → `"McLean County"` |
| `zip` | Return a canonical ZIP5 or ZIP+4 string. | `"1234"` → `"01234"`; `"123456"` → `"00012-3456"` |

#### Address profile

`address_us_v1` is a versioned, native-Spark display formatter. It:

- collapses whitespace and removes commas/periods from tokens;
- recognizes common USPS suffix names and aliases (`Street`/`St.` → `St`, `Avenue` → `Ave`);
- canonicalizes directionals (`northwest` → `NW`);
- canonicalizes common secondary-unit designators (`apartment` → `Apt`, `suite` → `Ste`);
- smart-cases `Mc` names (`mccormick` → `McCormick`), common exceptions such as `McLean`,
  apostrophe names, and hyphenated names; and
- uppercases alphanumeric values following a unit designator and hash-prefixed unit values
  (`Apt 4b` → `Apt 4B`, `Apt #4b` → `Apt #4B`).

Only the final suffix-like token is treated as the street suffix, preventing names such as
`123 Center Street` from becoming `123 Ctr St`. Empty punctuation-only tokens are removed before
joining, so commas do not create doubled spaces. Null input remains null.

Suffix output is deliberately punctuation-free (`St`, not `St.`), consistent with canonical
USPS abbreviations. The formatter follows [USPS Publication 28 suffix conventions](https://pe.usps.com/text/pub28/28apc_002.htm),
but it is not an address parser, deliverability check, geocoder, or authoritative postal
validator. Ambiguous human names cannot be perfectly inferred from casing alone. The package
therefore keeps this deterministic profile lightweight and native to Spark rather than adding
a Python UDF or a large external postal model.

#### County profile

`county` removes one existing case-insensitive `County` suffix, smart-cases the remaining name,
and appends exactly one ` County`. A value containing only `County` is a parse error. It does not
convert or infer Parish, Borough, Municipality, or Census Area; select this profile only when the
target domain truly requires counties.

#### ZIP profile

ZIP values remain strings so leading zeroes are preserved.

| Normalized input | Output |
| --- | --- |
| 1–5 digits | Left-pad to five digits (`1234` → `01234`). |
| 6–9 digits | Treat the last four digits as the extension and left-pad the remaining base to five (`123456` → `00012-3456`). |
| `1–5 digits` + hyphen + `1–4 digits` | Pad each component independently (`123-45` → `00123-0045`). |
| Non-digits, malformed hyphens, empty components, or more than nine digits | Parse error handled by `on_parse_error`. |

Audited padding records `zip_padded`; inserting or changing ZIP+4 formatting records
`zip_plus4_formatted`.

### Integer and long

`integer` targets a signed 32-bit Spark integer. `long` targets a signed 64-bit Spark long.
Their aliases are `int` and `bigint`. Overflow and non-integral input are parse errors.

Both add `zero_is_valid` (default `true`). When false, a successfully parsed zero becomes null
before final null handling. If the column is non-nullable, `default_on_null` is then assigned.
A zero `default_on_null` or `default_on_error` is rejected when zero is invalid because it could
not survive the subsequent zero-invalidating step.

### Decimal

`expected_data_type` must be `decimal(p,s)`, with precision from 1 through 38 and scale from zero
through precision. Typed defaults are checked against that exact precision and scale. Decimal is
recommended over double for currency and other exact base-10 values. It supports
`zero_is_valid` with the same behavior as integer and long. Spark rounds source values with excess
scale to the configured scale (for example, `1.239` parsed as `decimal(18,2)` becomes `1.24`);
typed defaults with excess scale remain compile-time errors.

### Double

Double accepts finite numeric typed defaults and uses Spark's double conversion. It supports
`zero_is_valid`. Use decimal when exact base-10 representation matters.

### Boolean

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `true_values` | No | Inherited global `["true"]` | Non-empty string tokens mapped to true. |
| `false_values` | No | Inherited global `["false"]` | Non-empty string tokens mapped to false. |
| `boolean_values_mode` | No | `replace` | When column token lists are supplied, `replace` replaces each supplied side and `extend` appends it to globals. An omitted side continues to inherit its global list. |
| `boolean_case_sensitive` | No | Inherited global `false` | Exact-case matching when true; lowercase comparison when false. |

Effective true and false sets must not overlap under the effective case rule. An unknown non-null
token is a parse error—it is never silently treated as false.

```yaml
globals:
  true_values: ["true", Y]
  false_values: ["false", N]

columns:
  - source_column_name: approval_status
    silver_column_name: IsApproved
    expected_data_type: boolean
    parser:
      type: boolean
      true_values: [approved]
      false_values: [rejected]
      boolean_values_mode: extend
```

### Date and timestamp

| Parser | Argument | Default | Behavior |
| --- | --- | --- | --- |
| `date` | `formats` | `[yyyy-MM-dd]` | Non-empty ordered Spark datetime patterns; first successful parse wins, then casts to date. |
| `timestamp` | `formats` | `[yyyy-MM-dd HH:mm:ss]` | Non-empty ordered Spark datetime patterns; first successful parse wins. |

Formats cascade in declared order. Automatic inference is not performed because ambiguous forms
such as `MM/dd/yyyy` and `dd/MM/yyyy` can both parse valid but different dates. Patterns are Spark
datetime patterns, not Python `strptime` patterns. Timestamp interpretation follows the active
Spark SQL session timezone.

## Parsing order and error modes

Every column follows this order:

1. Collapse whitespace and trim when enabled.
2. Convert the normalized empty string to null when enabled.
3. Match and optionally replace effective null markers.
4. Apply parser-specific conversion or string formatting.
5. Resolve a non-null value that failed conversion.
6. Convert numeric zero to null when `zero_is_valid: false`.
7. Apply `default_on_null` for a non-nullable target.

`on_parse_error` controls step 5:

- `fail` (default) constructs a lazy failure and raises when an action materializes that silver
  expression;
- `null` returns a typed null; or
- `default` assigns the required `default_on_error`.

There is no `ignore` mode because an invalid bronze string cannot be retained in a typed silver
column. Quarantine routing is not included yet; it needs an explicit `quarantine_df` contract
rather than overloading null/error handling.

## Compile-time and DataFrame validation

Strict compilation validates, without starting Spark:

- YAML syntax and duplicate/unsupported keys;
- required IDs, version, columns, and parser types;
- unique non-empty `silver_column_name` values;
- supported expected types, decimal precision/scale, and parser/type compatibility;
- option placement and primitive types;
- conditional `default_on_null`/`default_on_error` rules and typed values;
- global/column null-marker modes; and
- non-empty, non-overlapping effective Boolean vocabularies.

DataFrame binding validates lazily available schema metadata:

- duplicate configured source names in the input schema are ambiguous and fail;
- present configured sources must have Spark `string` type or fail;
- missing configured sources warn and bind as typed null/default;
- reserved parser output names may not already exist;
- keys must be existing, unique, unambiguous non-empty names; and
- `column_prefix` must be non-empty.

Actual bad values in `on_parse_error: fail` columns raise only when Spark materializes the failing
silver expression. Projection-pruning actions such as `parsed_df.count()` may not evaluate that
expression; a full silver write, `collect()`, or a select that consumes the column will.

## DataFrameParsing outputs

`parse_dataframe()` returns one `DataFrameParsing` object around a shared lazy Spark plan.

| Property/method | Behavior |
| --- | --- |
| `parsed_df` | Only silver columns, aliased to `silver_column_name` and ordered like the config. |
| `results_df` | Selected row keys followed by nested parser audit/identity metadata. |
| `warnings` | Tuple of recoverable schema warnings, currently missing configured sources. |
| `key_columns` | Effective ordered result-key names. |
| `result_columns` | Names of the three parser result columns. |
| `persist(storage_level=MEMORY_AND_DISK)` | Persist the shared plan and return the same object. Useful before materializing both projections. |
| `unpersist(blocking=False)` | Release the shared plan and return the same object. |

`parse_dataframe()` parameters:

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `df` | Yes | — | Bronze Spark DataFrame. Configured present sources must be top-level strings. |
| `config` | Yes | — | With the service API: compiled config, YAML text/path, or mapping. The lower-level runtime requires `ParserConfig`. |
| `key_columns` | No | All input columns | Ordered non-empty row identity used by `results_df`; declaring keys starts no action. |
| `column_prefix` | No | `spark_parser` | Prefix for reserved result fields. |

`results_df` fields are:

| Column | Definition |
| --- | --- |
| `spark_parser_parse_results` | Canonically typed array containing one struct per column with `audit: true`; an empty array with the identical nested schema when no columns are audited. |
| `spark_parser_config` | Struct with configuration `id`, `version`, and resolved canonical `content_hash`. |
| `spark_parser_engine_version` | Installed package version. |

Each parse-result struct contains:

| Nested field | Type | Definition |
| --- | --- | --- |
| `source_column_name` | String | Configured bronze source. |
| `silver_column_name` | String | Configured silver output. |
| `parser_type` | String | Canonical parser applied. |
| `expected_data_type` | String | Canonical target Spark datatype. |
| `original_value` | Nullable string | Unmodified bronze value; null for a missing source. |
| `parsed_value` | Nullable string | Final typed value rendered as text for uniform audit storage. |
| `changed` | Boolean | Whether a material null/default/error/missing-source/ZIP action occurred. |
| `effective` | Boolean | False for a missing source; true when the configured source was available. |
| `actions_applied` | Array of strings | Ordered material actions applied to this row and column. |
| `options` | Map of string to string | Every fully resolved effective option for this parser. |
| `error` | Nullable string | Handled parse or missing-source description; fail mode raises for bad values. |

Possible actions are `source_column_missing`, `empty_string_to_null`, `null_marker_replaced`,
`parse_error_to_null`, `parse_error_default_applied`, `zero_invalidated`,
`default_on_null_applied`, `zip_padded`, and `zip_plus4_formatted`.

Routine whitespace normalization, normal successful datatype conversion, and routine case/address/
county formatting are visible through `original_value`, `parsed_value`, and the resolved `options`
map but do not add noisy action entries or independently set `changed`.

## Migrating from 0.2.x

Version 0.3.0 introduced an intentional breaking column-metadata change:

| 0.2.x | 0.3.x |
| --- | --- |
| `column_name` | `source_column_name` plus `silver_column_name` |
| `data_type` | `expected_data_type` |

The compiler detects legacy column keys and returns a targeted migration error. Canonical payload
shape also changed, so `content_hash` values from 0.2.x are not comparable with 0.3.x hashes even
when the logical parsing behavior is equivalent. Fully resolved mappings from
`ParserConfigSerializer.to_mapping()` are recompilable in 0.3.1. See [CHANGELOG.md](CHANGELOG.md)
for release details.

## Defaults and exhaustive YAML reference

Defaults have one code-owned source of truth in
[`spark_parser.defaults`](src/spark_parser/defaults.py). `PARSER_DEFAULTS` is the live public
module mapping and should be treated as read-only; `parser.defaults()` returns a detached copy for
safe manipulation. The compiler resolves inherited and omitted values, and serialization/UAT
reports materialize those effective values.

[`examples/all_parsers.yaml`](examples/all_parsers.yaml) shows every top-level, global, column,
common, and parser-specific argument with required/default indicators. The repository
[`test_config.yaml`](test_config.yaml) is a smaller load-specific example.

Lower-level classes remain public for targeted use:

```python
from spark_parser import (
    ParserConfigSerializer,
    SparkDataFrameParser,
    YamlParserConfigCompiler,
)
```

The high-level service composes these APIs and is preferred for ordinary application code.
