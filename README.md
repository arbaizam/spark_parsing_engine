# Spark Parser

Spark Parser turns top-level bronze string columns into typed scalar Spark columns from a strict
YAML contract. It is designed for ingestion pipelines that need predictable cleanup, explicit
error behavior, deterministic configuration identity, and optional row-level audit evidence.

Every row transformation uses native Spark SQL expressions. The runtime does not collect source
rows and does not use Python or pandas UDFs.

Spark Parser 0.5.0 deliberately supports scalar targets only:

- `string`, `byte`, `short`, `integer`, `long`, `float`, `decimal(p,s)`, and `double`;
- `binary` and `boolean`; and
- `date`, `timestamp`, and `timestamp_ntz`.

Configured `array`, `struct`, and `map` parsers are not supported. Decode or flatten complex source
data upstream into top-level strings, parse those scalar leaves here, and reconstruct complex
outputs downstream when the target model needs them. Keeping this boundary explicit makes the
compiler, runtime plan, audit contract, and failure model substantially easier to understand.

## Quick start

For local development, install the Spark and test extras:

```text
python -m pip install -e ".[spark,test]"
```

Do not install the Spark extra on Databricks. The Databricks runtime already owns Spark and Py4J.
Install the package wheel with `--no-deps` on a runtime that provides Spark 3.5 or newer and PyYAML
6 or newer.

```yaml
parser_config_id: bronze_customer_load
parser_config_name: Bronze Customer Load
version: "1.0.0"
description: Parse flattened customer values for the target layer.
owner: Data Engineering
owner_department: Enterprise Data

globals:
  null_markers: [NA, "Null", N/A]
  null_marker_case_sensitive: false

columns:
  - source_column_name: record_id
    target_column_name: RecordId
    expected_data_type: string
    parser: string

  - source_column_name: customer_name
    target_column_name: CustomerName
    expected_data_type: string
    parser:
      type: string
      format: title_business_v1
      audit: true

  - source_column_name: rate_index
    target_column_name: RateIndex
    expected_data_type: string
    parser:
      type: string
      format: interest_rate_index_v1
      on_parse_error: preserve
      audit: true

  - source_column_name: account_balance
    target_column_name: AccountBalance
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

target_df = parsing.parsed_df
audit_df = parsing.results_df
schema_warnings = parsing.warnings
```

For most callers, the package-level `parser` service is the only object needed. See the
[Spark Parser user guide](examples/spark_parser_user_guide.py) for an executable Databricks
walkthrough and [`examples/all_parsers.yaml`](examples/all_parsers.yaml) for every authoring
option.

## Complex source-data boundary

The parser requires every present configured source to be a top-level Spark `string` column. A
complex source should be made explicit before it crosses that boundary:

1. Decode JSON, arrays, maps, or structs with the ingestion layer or Spark readers.
2. Select, explode, or flatten the scalar leaves that need normalization.
3. Cast those leaves to top-level strings and run Spark Parser.
4. Optionally reconstruct arrays, structs, or maps in the downstream transformation layer.

For a JSON object, an upstream projection can look like this:

```python
from pyspark.sql import functions as F
from pyspark.sql import types as T

payload_schema = T.StructType(
    [
        T.StructField("customer_name", T.StringType()),
        T.StructField("account_balance", T.StringType()),
        T.StructField("rate_index", T.StringType()),
    ]
)

flat_df = (
    bronze_df
    .withColumn("decoded", F.from_json("payload_json", payload_schema))
    .select(
        "record_id",
        F.col("decoded.customer_name").alias("customer_name"),
        F.col("decoded.account_balance").alias("account_balance"),
        F.col("decoded.rate_index").alias("rate_index"),
    )
)

parsed = parser.parse_dataframe(flat_df, config, key_columns=["record_id"])
reconstructed_df = parsed.parsed_df.select(
    "RecordId",
    F.struct("AccountBalance", "RateIndex").alias("LoanTerms"),
)
```

Array workflows can use `explode` or `posexplode` upstream and regroup downstream when order must
be retained. CSV quoting, JSON shape validation, duplicate object keys, array deduplication, and
container-level error policy belong to those surrounding stages rather than this package.

## Package design

The normal flow is:

`YAML or mapping` → `YamlParserConfigCompiler` → `ParserConfig` → `SparkDataFrameParser` →
`DataFrameParsing`

- [`service.py`](src/spark_parser/service.py) exposes the shared `parser` facade for compilation,
  metadata, review, serialization, hashing, and DataFrame parsing.
- [`compiler_yaml.py`](src/spark_parser/compiler_yaml.py) is the Spark-free trust boundary. It
  rejects duplicate or unknown keys, resolves inherited defaults, validates scalar parser/type
  compatibility and typed defaults, and returns immutable models.
- [`data_types.py`](src/spark_parser/data_types.py) canonicalizes scalar aliases and validates
  `decimal(p,s)` without importing PySpark.
- [`models.py`](src/spark_parser/models.py) contains the frozen scalar configuration dataclasses.
- [`defaults.py`](src/spark_parser/defaults.py), [`enums.py`](src/spark_parser/enums.py), and
  [`metadata.py`](src/spark_parser/metadata.py) define one closed public vocabulary.
- [`serializer.py`](src/spark_parser/serializer.py) produces resolved mappings, canonical JSON, and
  deterministic SHA-256 configuration hashes.
- [`spark_runtime.py`](src/spark_parser/spark_runtime.py) validates input schemas and builds the
  native scalar Spark expressions and audit records.
- [`address_formats.py`](src/spark_parser/address_formats.py),
  [`interest_rate_formats.py`](src/spark_parser/interest_rate_formats.py), and
  [`title_formats.py`](src/spark_parser/title_formats.py) keep domain formatting rules outside the
  runtime orchestration code.
- [`dataframe_parsing.py`](src/spark_parser/dataframe_parsing.py) exposes the lazy `parsed_df` and
  `results_df` projections.

PySpark is loaded only when the DataFrame runtime is requested, so config authoring, review,
datatype validation, serialization, and hashing work in a lightweight Python environment.

## Service API

| Method/property | Result and behavior |
| --- | --- |
| `parser.compile_text(text)` | Compile YAML text into an immutable `ParserConfig`. |
| `parser.compile_path(path)` | Read and compile a UTF-8 YAML file. |
| `parser.compile_mapping(mapping)` | Compile an already-loaded YAML-compatible mapping. |
| `parser.compile_yaml(source)` | Dispatch YAML text, a `pathlib.Path`, or a mapping by type. |
| `parser.parse_dataframe(df, config, ...)` | Bind a compiled or authorable config to a DataFrame. |
| `parser.to_mapping(config)` | Return every resolved option as a JSON-compatible mapping. |
| `parser.canonical_json(config)` | Return deterministic canonical JSON. |
| `parser.content_hash(config)` | Return the resolved configuration's SHA-256 identity. |
| `parser.defaults()` | Return a detached JSON-shaped copy of code-owned defaults. |
| `parser.normalize_data_type(value)` | Validate and canonicalize a supported scalar datatype. |
| `parser.describe()` | Return metadata for every scalar parser. |
| `parser.<type>.describe()` | Return arguments, defaults, behavior, and gotchas for one parser. |
| `parser.config.describe()` | Return top-level, global, and column authoring metadata. |
| `parser.review_yaml(source)` | Return a review report instead of raising for authoring errors. |

Metadata accessors exist for `string`, `byte`, `short`, `integer`, `long`, `float`, `decimal`,
`double`, `binary`, `boolean`, `date`, `timestamp`, and `timestamp_ntz`.

## Configuration review

`review_yaml()` accepts YAML text, a `pathlib.Path`, or an already loaded mapping. String inputs are
always YAML text. A valid `ConfigReviewReport` includes:

- identity, ownership, version, and canonical content hash;
- compiler validation evidence;
- every scalar source-to-target binding and resolved option;
- effective error, nullability, formatting, and audit behavior;
- warnings such as missing ownership or inert global null markers; and
- copy-ready canonical YAML with inherited defaults filled in.

```python
from pathlib import Path

report = parser.review_yaml(Path("customer_parser.yaml"))
if not report.is_valid:
    raise ValueError(report.errors)

report.write_markdown("customer_parser_review.md")
report.write_json("customer_parser_review.json")
```

Configuration review does not inspect source data. Missing or incompatible DataFrame columns are
reported when `parse_dataframe()` binds the compiled contract.

## Configuration metadata

Top-level arguments are `parser_config_id`, `parser_config_name`, `version`, `columns`, and the
optional `description`, `owner`, `owner_department`, and `globals`. `columns` is a non-empty ordered
list of these bindings:

| Column argument | Required | Behavior |
| --- | ---: | --- |
| `source_column_name` | Yes | Top-level bronze string source; multiple targets may reuse it. |
| `target_column_name` | Yes | Unique, non-blank target name preserved verbatim. |
| `expected_data_type` | Yes | Exact supported scalar target datatype. |
| `parser` | Yes | Matching scalar parser name or options mapping. |

Accepted aliases are:

| Alias | Canonical type |
| --- | --- |
| `tinyint` | `byte` |
| `smallint` | `short` |
| `int` | `integer` |
| `bigint` | `long` |
| `real` | `float` |
| `dec`, `numeric` | `decimal` |
| `bool` | `boolean` |
| `timestamp_ltz` | `timestamp` |

Decimal targets must declare precision and scale, such as `decimal(18,2)`. Spark precision is
limited to 38 and scale must be between zero and precision.

Global arguments define the inherited `null_markers`, `null_marker_case_sensitive`, `true_values`,
`false_values`, and `boolean_case_sensitive`. Quote YAML Boolean-like strings such as `"true"`,
`"false"`, `"yes"`, and `"no"`.

## Common parser arguments

| Argument | Default | Behavior |
| --- | --- | --- |
| `type` | Required in mapping form | Parser implementation; must match the target datatype. |
| `collapse_whitespace` | `true` | Replace each whitespace run with one ordinary space. |
| `trim_whitespace` | `true` | Remove Unicode edge whitespace. |
| `empty_is_null` | `true` | Convert an empty normalized string to null. |
| `replace_null_markers` | `false` | Enable effective null-marker replacement. |
| `null_markers` | Inherited | Column null markers. |
| `null_markers_mode` | `replace` | Replace or extend global markers. |
| `null_marker_case_sensitive` | Inherited | Exact-case null comparison when true. |
| `is_nullable` | `true` | Allow the final target value to remain null. |
| `default_on_null` | — | Required when `is_nullable: false`. |
| `on_parse_error` | `fail` | `fail`, `null`, or `default`; strings also allow `preserve`. |
| `default_on_error` | — | Required when `on_parse_error: default`. |
| `audit` | `false` | Emit one row-level audit record for the column. |

Every column follows the same order:

1. Collapse and trim whitespace when enabled.
2. Convert an empty value and enabled null markers to null.
3. Parse or format the scalar value.
4. Resolve a non-null parse failure.
5. Optionally invalidate numeric zero.
6. Apply `default_on_null` when the result may not remain null.

`preserve` returns the exact pre-normalization source token and is accepted only for a string
target. A raw invalid string cannot inhabit a numeric, date, Boolean, timestamp, or binary column.

## Scalar parser reference

| Parser | Target and specific options |
| --- | --- |
| `string` | `string`; optional `format`. |
| `byte`, `short`, `integer`, `long` | Matching signed Spark integer; optional `zero_is_valid`. |
| `float`, `double` | Matching Spark floating type; optional `zero_is_valid`. |
| `decimal` | `decimal(p,s)`; optional `zero_is_valid`. |
| `binary` | `binary`; `encoding` is `base64` (default), `hex`, or `utf8`. |
| `boolean` | `boolean`; configurable true/false tokens, case sensitivity, and replace/extend mode. |
| `date` | `date`; ordered Spark datetime `formats`. |
| `timestamp` | `timestamp`; ordered formats, including offset-aware patterns. |
| `timestamp_ntz` | `timestamp_ntz`; ordered timezone-free formats. |

Numeric text is validated before conversion so malformed or non-finite tokens follow the configured
error policy under both ANSI and permissive Spark modes. Use `decimal(p,s)` when exact base-10
representation matters. Binary audit output is canonical base64 regardless of source encoding.

Datetime formats are Spark patterns, not Python `strptime` patterns. Formats cascade in list order.
The built-in defaults cover ISO and the known US month-first 12-hour export; `timestamp_ntz`
rejects offset-bearing inputs rather than silently discarding timezone meaning.

### String formats

| Format | Behavior | Example |
| --- | --- | --- |
| `null` / omitted | Keep the normalized string. | `"  loan   status "` → `"loan status"` |
| `lower` | Lowercase. | `"ACTIVE"` → `"active"` |
| `upper` | Uppercase. | `"active"` → `"ACTIVE"` |
| `title` | Spark title casing with normalized spaces. | `"loan STATUS"` → `"Loan Status"` |
| `title_business_v1` | Title casing plus exact business exceptions and bounded numeric-hyphen casing. | `"fhlb 12-month advance"` → `"FHLB 12-Month Advance"` |
| `pascal` | Init-capitalize and remove spaces for identifiers. | `"loan status"` → `"LoanStatus"` |
| `address_us_v1` | Deterministic US address display normalization. | `"123 n main st apt 4b"` → `"123 N Main St Apt 4B"` |
| `county` | Smart-case and ensure one trailing `County`. | `"mcclain COUNTY"` → `"McClain County"` |
| `state_us` | Canonicalize supported US state/territory values to postal codes. | `"Illinois"` → `"IL"` |
| `zip` | Validate/pad ZIP5 and format ZIP+4. | `"1234"` → `"01234"` |
| `interest_rate_index_v1` | Canonicalize the approved interest-rate catalog. | `"SOFR Term - 12M"` → `"SOFR Term 12-Month"` |

`title` remains the strict general formatter. `title_business_v1` starts with the same title
pipeline, then restores only `FHLB`, `P&I`, `UST`, `RCF`, and `CMT` as complete tokens and changes a
bounded integer-hyphen suffix such as `12-month` to `12-Month`. It does not infer unknown acronyms.

`interest_rate_index_v1` is fail-closed. It accepts only its canonical catalog, safe tenor
normalizations, and explicit source aliases. Generic SOFR labels stay generic; cosmetic
family/value separator hyphens are removed while tenor hyphens remain. For example,
`SOFR - 30-Day` becomes `SOFR 30-Day` and `Treasury - 1 yr` becomes `Treasury 1-Year`. Configure
`on_parse_error: preserve` when an unknown value such as `NAP` should remain unchanged.

Address, county, state, ZIP, business-title, and interest-rate profiles are deterministic display
normalizers, not external validation services. State and ZIP also accept multiple comma-separated
property values and fail the full value if one component is invalid.

## Compile-time and DataFrame validation

Compilation checks YAML shape, required metadata, unique targets, supported scalar datatypes,
parser/type compatibility, option placement, typed defaults, null-marker modes, and effective
Boolean vocabularies. Duplicate and unknown keys fail rather than being ignored.

When binding a DataFrame, Spark Parser also checks that:

- every present configured source is a top-level string;
- source and target names are unambiguous under Spark's active case-sensitivity resolver;
- missing sources fail unless `on_missing_source="warn"` explicitly allows substitution;
- result-column names do not collide with sources, targets, or row keys; and
- row keys are existing, unique, unambiguous names.

`on_parse_error: fail` is lazy: it raises only when Spark evaluates the failing target expression.
An optimizer-pruned `count()` may not materialize every target. A full target write or explicit
projection does.

## DataFrameParsing outputs and audit contract

`parse_dataframe()` returns a `DataFrameParsing` wrapper:

| Property/method | Behavior |
| --- | --- |
| `parsed_df` | Target scalar columns, in configuration order. |
| `results_df` | Explicit row keys followed by audit and configuration identity columns. |
| `warnings` | Recoverable schema warnings such as an explicitly allowed missing source. |
| `persist()` / `unpersist()` | Optional shared-plan caching on runtimes that support it. |

Databricks serverless may reject DataFrame cache APIs; omit persistence there. Row keys belong to
`results_df` and are not copied into `parsed_df` unless configured as target columns too.

The result columns are:

| Column | Definition |
| --- | --- |
| `<prefix>_parse_results` | Array containing one audit struct per `audit: true` scalar column. |
| `<prefix>_config` | Struct containing configuration `id`, `version`, and content hash. |
| `<prefix>_engine_version` | Installed Spark Parser version. |

Each audit struct has `source_column_name`, `target_column_name`, `parser_type`,
`expected_data_type`, `original_value`, `parsed_value`, `changed`, `effective`, `actions_applied`,
`options`, and `error`. `parsed_value` is text, except binary values are represented as canonical
base64. The audit schema intentionally has no nested child-path fields.

Possible actions are `source_column_missing`, `empty_string_to_null`, `null_marker_replaced`,
`parse_error_to_null`, `parse_error_default_applied`, `parse_error_preserved`, `zero_invalidated`,
`default_on_null_applied`, `zip_padded`, and `zip_plus4_formatted`. Routine normalization and
successful formatting are visible by comparing original and parsed values but do not add actions.

## Testing

```text
python -m pytest tests/unit -q
SPARK_PARSER_REQUIRE_JAVA=1 python -m pytest tests/integration -q
```

The Databricks system notebook is [`tests/system/spark_parser_system_tests.py`](tests/system/spark_parser_system_tests.py).
See the [unit-test summary](docs/spark_parser_unit_test_summary.md) and
[system-test summary](docs/spark_parser_system_test_summary.md) for their current contracts.

## Defaults and exhaustive YAML reference

All defaults come from [`spark_parser.defaults`](src/spark_parser/defaults.py). `PARSER_DEFAULTS`
is deeply immutable; `parser.defaults()` returns a detached JSON-shaped copy. Compilation,
serialization, metadata, and review reports all use the same values.

[`examples/all_parsers.yaml`](examples/all_parsers.yaml) documents every scalar parser and option,
including every string-format profile. Lower-level public classes remain available for focused
tooling, but normal application code should use the shared `parser` service.
