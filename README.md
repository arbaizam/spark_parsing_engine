# Spark Parser

Spark Parser converts load-specific bronze string columns into explicitly
typed silver columns from strict YAML configuration. It uses native Spark SQL
expressions only; it does not use Python or pandas UDFs.

## Configuration

Each configuration has versioned identity metadata aligned with the rules
engine and exactly one parser for every column promoted to silver:

```yaml
parser_config_id: bronze_customer_load
parser_config_name: Bronze Customer Load
version: 1.0.0
owner: Data Engineering
owner_department: Enterprise Data
description: Parses one bronze customer load.

globals:
  null_markers: [NA, "Null"]
  null_marker_case_sensitive: false

columns:
  - column_name: customer_name
    data_type: string
    parser:
      type: string
      format: pascal
      replace_null_markers: true
      audit: true

  - column_name: account_balance
    data_type: decimal(18,2)
    parser:
      type: decimal
      zero_is_valid: false
      on_parse_error: null
      audit: true
```

`parser` may be a scalar such as `parser: long` when every option should use
its safe default. See [examples/all_parsers.yaml](examples/all_parsers.yaml)
for all supported types.

Supported parsers are `string`, `integer`, `long`, `decimal`, `double`,
`boolean`, `date`, and `timestamp`. Integer data types may be authored as
`int`, and long types as `bigint`; compiled metadata uses canonical names.
Decimal precision is limited to 38 and scale cannot exceed precision.

## Defaults

Defaults have one code-owned source of truth in
[`spark_parser.defaults`](src/spark_parser/defaults.py). The YAML compiler and
canonical dataclass models both consume those constants. Authoring tools can
read the JSON-compatible public mapping directly:

```python
from spark_parser import PARSER_DEFAULTS
```

The compiler resolves every inherited or omitted value, and
`ParserConfigSerializer.to_mapping()` materializes those effective values for
reporting and hashing. The commented
[`examples/all_parsers.yaml`](examples/all_parsers.yaml) file also shows every
argument, requirement, allowed value, and default in YAML form.

## YAML argument reference

### Top-level configuration

| Argument | Required | Default | Allowed value and behavior |
|---|---:|---|---|
| `parser_config_id` | Yes | — | Non-empty stable identifier for this parsing configuration. |
| `parser_config_name` | Yes | — | Non-empty human-readable name. |
| `version` | Yes | — | Non-empty string identifying the immutable configuration version. Quote numeric-looking versions such as `"1"`. |
| `columns` | Yes | — | Non-empty ordered list of column parser mappings. Every configured column is promoted to `parsed_df`. |
| `description` | No | `null` | Human-readable purpose. |
| `owner` | No | `null` | Owning person or team. |
| `owner_department` | No | `null` | Owning department. |
| `globals` | No | `{}` | Global null-marker defaults inherited by column parsers. |

Lifecycle status is intentionally absent. Version and canonical content hash
provide configuration identity.

### Global arguments

| Argument | Required | Default | Allowed value and behavior |
|---|---:|---|---|
| `null_markers` | No | `[]` | List of exact strings available to every column. Duplicates are removed while preserving order. Markers are inactive unless a column enables `replace_null_markers`. |
| `null_marker_case_sensitive` | No | `true` | Whether inherited marker matching distinguishes case. A column may override it. |

### Column arguments

| Argument | Required | Default | Allowed value and behavior |
|---|---:|---|---|
| `column_name` | Yes | — | Exact, unique top-level bronze column name. The silver output keeps this name. |
| `data_type` | Yes | — | `string`, `integer`, `long`, `decimal(p,s)`, `double`, `boolean`, `date`, or `timestamp`. `int` and `bigint` normalize to `integer` and `long`. |
| `parser` | Yes | — | Matching parser name as a scalar, or a mapping containing `type` and options. `parser: long` is shorthand for a long parser with defaults. |

The parser type must match the target datatype. A decimal parser matches any
valid `decimal(p,s)` target.

### Common parser arguments

These arguments apply to every parser type.

| Argument | Required | Default | Allowed value and behavior |
|---|---:|---|---|
| `type` | Yes in mapping form | — | One canonical parser type matching `data_type`. Scalar parser form supplies this value directly. |
| `collapse_whitespace` | No | `true` | Replaces each run of whitespace—including spaces, tabs, and line breaks—with one ordinary space. |
| `trim_whitespace` | No | `true` | Removes whitespace from both ends after collapsing. |
| `empty_is_null` | No | `true` | Converts an empty string remaining after whitespace normalization to null. |
| `replace_null_markers` | No | `false` | Converts effective null-marker values to null. |
| `null_markers` | No | Inherited global list | Column list used for marker matching. Supplying it activates `null_markers_mode`; it does not by itself enable replacement. |
| `null_markers_mode` | No | `replace` | `replace` uses only column markers; `extend` appends them to globals. May be explicitly supplied only with column `null_markers`. |
| `null_marker_case_sensitive` | No | Inherited global value | Overrides marker case sensitivity for this column. |
| `is_nullable` | No | `true` | Whether the final parsed value may remain null. |
| `default_on_null` | Conditional | No default | Required when `is_nullable: false`; forbidden otherwise. Must be a non-null value that exactly fits the target datatype. |
| `on_parse_error` | No | `fail` | `fail` raises during the first Spark action; `null` produces typed null; `default` uses `default_on_error`. YAML `null` may be quoted or unquoted. |
| `default_on_error` | Conditional | No default | Required only with `on_parse_error: default`; forbidden for other modes. Must exactly fit the target datatype. |
| `audit` | No | `false` | Includes this column in each row's `spark_parser_parse_results` array. |

There is no `ignore` mode because an invalid raw string cannot be retained in
a typed target column. Row quarantine is deferred until the package has an
explicit `quarantine_df` routing contract.

### String parser arguments

| Argument | Required | Default | Allowed value and behavior |
|---|---:|---|---|
| `format` | No | `null` | `null`/`none` leaves case unchanged; `lower`, `upper`, or `pascal` applies case formatting after null handling. Pascal lowercases, title-cases, and removes whitespace between words. |

### Numeric parser arguments

Applies to `integer`, `long`, `decimal`, and `double`.

| Argument | Required | Default | Allowed value and behavior |
|---|---:|---|---|
| `zero_is_valid` | No | `true` | When `false`, any successfully parsed numeric value equal to zero becomes null. If `is_nullable: false`, `default_on_null` is then applied. A zero null-default is rejected. |

### Date and timestamp parser arguments

| Parser | Argument | Required | Default | Allowed value and behavior |
|---|---|---:|---|---|
| `date` | `formats` | No | `[yyyy-MM-dd]` | Non-empty ordered list of Spark datetime patterns. The first successful parse wins. |
| `timestamp` | `formats` | No | `[yyyy-MM-dd HH:mm:ss]` | Non-empty ordered list of Spark datetime patterns. The first successful parse wins. |

Automatic format inference is not implemented in this release. Ambiguous
formats such as `MM/dd/yyyy` and `dd/MM/yyyy` must remain explicit.

### Boolean parser arguments

| Argument | Required | Default | Allowed value and behavior |
|---|---:|---|---|
| `true_values` | No | `["true"]` | Non-empty list of source strings parsed as true. |
| `false_values` | No | `["false"]` | Non-empty list of source strings parsed as false. |
| `boolean_case_sensitive` | No | `false` | Controls case sensitivity for true/false matching. The two effective lists must not overlap. |

Boolean tokens are strings because every bronze source value is a string.
Quote YAML words such as `"true"` and `"false"` to prevent YAML from loading
them as Boolean values.

## Parsing behavior

Defaults are intentionally conservative: normalization is predictable,
datatype failures stop execution, and no business value is treated as a null
marker unless explicitly enabled.

Column `null_markers` use the global list when omitted. When supplied,
`null_markers_mode: replace` replaces the globals and `extend` appends to
them. Marker matching happens after whitespace normalization and empty-string
handling. It is case-sensitive unless the global or column setting explicitly
disables case sensitivity.

Parsing occurs in this order:

1. Collapse internal whitespace and trim both ends when enabled.
2. Convert an empty normalized string to null when enabled.
3. Replace configured null markers.
4. Parse and resolve parsing failures.
5. Convert numeric zero to null when zero is invalid.
6. Apply an explicit non-null default when required.

A zero default is rejected when `zero_is_valid: false`.

## Spark API

```python
from spark_parser import SparkDataFrameParser, YamlParserConfigCompiler

config = YamlParserConfigCompiler().compile_path("customer_parser.yaml")
parsing = SparkDataFrameParser().parse_dataframe(
    bronze_df,
    config,
    key_columns=["load_id", "row_id"],
)

silver_df = parsing.parsed_df
audit_df = parsing.results_df
```

`parse_dataframe()` arguments:

| Argument | Required | Default | Behavior |
|---|---:|---|---|
| `df` | Yes | — | Input Spark DataFrame. Every configured column must exist exactly once and have Spark `string` type. |
| `config` | Yes | — | Compiled `ParserConfig`. |
| `key_columns` | No | All input columns | Ordered, non-empty sequence of existing unique column names included in `results_df`. This declares row identity without starting a validation action. |
| `column_prefix` | No | `spark_parser` | Non-empty prefix for parser result columns. Input columns may not conflict with the resulting reserved names. |

`parsed_df` contains only configured columns, in configuration order.
Unconfigured bronze columns are not promoted. Missing configured columns,
non-string configured inputs, ambiguous names, and invalid key metadata fail
before a Spark action.

`results_df` contains the selected row keys followed by:

| Column | Definition |
|---|---|
| `spark_parser_parse_results` | One nested audit struct per column with `audit: true`. |
| `spark_parser_config` | Immutable `id`, `version`, and canonical `content_hash`. |
| `spark_parser_engine_version` | Installed package version. |

Each parse-result struct uses column name, parser type, and target datatype as
its identity:

| Nested field | Type | Definition |
|---|---|---|
| `column_name` | String | Configured bronze and silver column name. |
| `parser_type` | String | Canonical parser type applied. |
| `data_type` | String | Canonical target Spark datatype. |
| `original_value` | Nullable string | Unmodified bronze value. |
| `parsed_value` | Nullable string | Final typed value rendered as text for uniform audit storage. |
| `changed` | Boolean | Whether a material null/default/error action occurred. |
| `effective` | Boolean | True when this configured parse result was assigned. |
| `actions_applied` | Array of strings | Ordered material actions such as `empty_string_to_null`, `null_marker_replaced`, `parse_error_to_null`, `parse_error_default_applied`, `zero_invalidated`, and `default_on_null_applied`. |
| `options` | Map of string to string | Every fully resolved effective parser option. |
| `error` | Nullable string | Handled parse-error description; fail mode raises instead of returning a row. |

Routine whitespace collapsing/trimming, case formatting, and datatype
conversion do not set `changed`. Empty-to-null conversion, null-marker
replacement, handled parse errors, zero invalidation, and explicit default
substitution do.

Both projections share one lazy Spark plan. Call `parsing.persist()` before
materializing both when reuse justifies caching, then `parsing.unpersist()`.
When `key_columns` is omitted, all original input columns are used as result
keys, matching the rules-engine convention.

`persist(storage_level=StorageLevel.MEMORY_AND_DISK)` returns the same
`DataFrameParsing` object. `unpersist(blocking=False)` releases that shared
plan and also returns the same object.
