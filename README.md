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

## Safe defaults

| Option | Default | Behavior |
|---|---|---|
| `trim_whitespace` | `true` | Trims source text before marker matching and parsing. |
| `format` | `null` | Performs no string case formatting. |
| `replace_null_markers` | `false` | Does not reinterpret business values as null. |
| `is_nullable` | `true` | Allows a final null. |
| `on_parse_error` | `fail` | Raises during the first materializing Spark action. |
| `zero_is_valid` | `true` | Preserves numeric zero. |
| `audit` | `false` | Omits the column from row-level audit structs. |

When `is_nullable: false`, `default_on_null` is required. When
`on_parse_error: default`, `default_on_error` is required. Defaults are
validated against the target type during YAML compilation.

Column `null_markers` use the global list when omitted. When supplied,
`null_markers_mode: replace` replaces the globals and `extend` appends to
them. Marker matching happens after trimming. It is case-sensitive unless the
global or column setting explicitly disables case sensitivity.

Parsing occurs in this order:

1. Trim source text when enabled.
2. Replace configured null markers.
3. Parse and resolve parsing failures.
4. Convert numeric zero to null when zero is invalid.
5. Apply an explicit non-null default when required.

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
its identity. It includes original and parsed values, material-change state, applied
actions, every effective option, and a handled parsing error when applicable.
Routine whitespace trimming, case formatting, and datatype conversion do not
set `changed`; null-marker replacement, handled parse errors, zero
invalidation, and explicit default substitution do.

Both projections share one lazy Spark plan. Call `parsing.persist()` before
materializing both when reuse justifies caching, then `parsing.unpersist()`.
When `key_columns` is omitted, all original input columns are used as result
keys, matching the rules-engine convention.
