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

For a source-linked step-through of compilation, schema binding, lazy expression stages, policy
order, and audit output, download the
[Spark Parser execution explorer](docs/spark_parser_execution_explorer.html) and open it in a
browser. It is pinned to code snapshot `7e5b110` and uses scenarios asserted by the test suite.

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
  [`interest_rate_formats.py`](src/spark_parser/interest_rate_formats.py),
  [`property_type_formats.py`](src/spark_parser/property_type_formats.py), and
  [`title_formats.py`](src/spark_parser/title_formats.py) keep domain formatting rules outside
  the runtime orchestration code.
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
| `parser.parse_dataframe(df, config, ...)` | Bind a compiled or authorable config to a DataFrame; optional `error_mode="collect"` records conversion failures inline. |
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
always YAML text. Semantic validation visits every independently checkable field and column before
returning, so an invalid report's ordered `errors` tuple contains all actionable findings with
paths such as `columns[2].parser.default_on_error`. Malformed YAML, duplicate YAML keys, unreadable
files, and invalid outer mapping shapes remain single errors because no trustworthy object tree is
available for continued validation. A valid `ConfigReviewReport` includes:

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

Numeric text is validated before conversion so malformed or non-finite tokens, and nonzero
floating-point tokens that underflow to zero, follow the configured error policy under both ANSI
and permissive Spark modes. Use `decimal(p,s)` when exact base-10 representation matters. Binary
audit output is canonical base64 regardless of source encoding.

Datetime formats are Spark patterns, not Python `strptime` patterns. Formats cascade in list order.
The built-in defaults cover ISO and the known US month-first 12-hour export; `timestamp_ntz`
rejects offset-bearing inputs rather than silently discarding timezone meaning.

### String formats

| Format | Behavior | Example |
| --- | --- | --- |
| YAML `null`, a case-insensitive `"null"` / `"none"`, or omitted | Keep the normalized string. | `"  loan   status "` → `"loan status"` |
| `lower` | Lowercase. | `"ACTIVE"` → `"active"` |
| `upper` | Uppercase. | `"active"` → `"ACTIVE"` |
| `title` | Spark title casing with normalized spaces. | `"loan STATUS"` → `"Loan Status"` |
| `title_business_v1` | Title casing plus exact business exceptions and business hyphen rules. | `"fhlb semi-annual 2 yrs"` → `"FHLB Semi-Annual 2-Years"` |
| `pascal` | Init-capitalize and remove spaces for identifiers. | `"loan status"` → `"LoanStatus"` |
| `address_us_v1` | Deterministic US address display normalization. | `"123 n main st apt 4b"` → `"123 N Main St Apt 4B"` |
| `county` | Smart-case and ensure one trailing `County`. | `"mcclain COUNTY"` → `"McClain County"` |
| `state_us` | Canonicalize supported US state/territory values to postal codes. | `"Illinois"` → `"IL"` |
| `zip` | Validate/pad ZIP5 and format ZIP+4. | `"1234"` → `"01234"` |
| `interest_rate_index_v1` | Canonicalize the approved interest-rate catalog. | `"SOFR Term - 12M"` → `"SOFR Term 12-Month"` |
| `property_type_v1` | Canonicalize approved loan property types and restructure mixed-use values. | `"Warehouse-Mixed use"` → `"Mixed Use - Warehouse"` |

#### Business title profile

`title` remains the strict general formatter. `title_business_v1` starts with the same title
pipeline and adds five versioned rules:

- restore the complete, case-insensitive tokens `FHLB`, `P&I`, `UST`, `RCF`, and `CMT`;
- expand the complete, case-insensitive token `Yrs` to `Years`;
- canonicalize the complete frequency aliases `semiannual`, `semiannually`, `semi-annually`,
  `biannual`, `biannually`, `bi-annually`, `biweekly`, and `bimonthly` to `Semi-Annual`,
  `Bi-Annual`, `Bi-Weekly`, or `Bi-Monthly` as appropriate;
- capitalize the first lowercase letter in every non-empty component after an ASCII hyphen
  (`semi-annual` → `Semi-Annual`, `bi-weekly` → `Bi-Weekly`, and `12-month` → `12-Month`);
- replace one or more ASCII spaces between a bounded ASCII integer and the complete plural word
  `Years` or `Months` with an ASCII hyphen (`2 years` → `2-Years`, `18 months` → `18-Months`).

Letters, numbers, and underscores count as part of an identifier for exception matching. Thus
`cmt-backed` becomes `CMT-Backed`, while `RCF6M` and `fhlb_code` receive only ordinary title casing.
The hyphen rule applies to ordinary compounds (`state-of-the-art` → `State-Of-The-Art`) but does
not treat non-ASCII dashes or a hyphen followed by whitespace as component boundaries. This profile
does not hyphenate singular `Year`/`Month`, embedded integers, decimals, or comma-formatted numbers.
Aliases and acronym exceptions require complete tokens, so identifier text such as
`biannually_code` or `yrs_code` remains ordinary title casing. The profile does not otherwise infer
acronyms, repair spelling, or validate business terminology.

#### Interest-rate index profile

`interest_rate_index_v1` is a closed, case-insensitive canonicalization profile. It normalizes only
an approved complete value; an unsupported tenor, unknown family, or partial opaque-code match is
a parse error. Canonical labels omit a cosmetic family/value separator hyphen while retaining the
hyphen inside a tenor:

| Canonical family | Approved canonical values |
| --- | --- |
| `SOFR Term` and `BSBY` | `Overnight`; 1-, 3-, 6-, and 12-Month |
| `Ameribor` | `Overnight`; 1-Week; 1-, 3-, and 6-Month; 1- and 2-Year; 30- and 90-Day Average; `Derived 30T`; `Derived 90T` |
| Constant Maturity Treasury | `N-Year Constant Maturity Treasury (CMT)`, where N is 1, 2, 3, 5, 7, 10, or 30 |
| `LIBOR` | `N-Month LIBOR`, where N is 1, 2, 3, 6, 9, or 12; `LIBOR 1-Year (Daily)` |
| `Treasury` and `Treasury Avg` | `Treasury N-Month` for 3 or 6; `Treasury N-Year` for 1, 2, 3, 5, 7, 10, or 30; `Treasury Avg 12-Month` |
| `USD Swap` | 1-, 5-, and 10-Year |
| `Freddie Mac` | 1-, 3-, 6-, and 12-Month |
| `FHLB` | 1-, 2-, 3-, 5-, 7-, and 10-Year |
| Generic `SOFR` | `SOFR`; 1- and 12-Month; 30-Day; 30- and 180-Day Average |
| `RCF` | 6- and 12-Month |
| Other | `Prime` |

Month aliases are `M`, `mo`, `mos`, `month`, and `months`; year aliases are `Y`, `Yr`, `yrs`,
`year`, and `years`. Spelled `day`/`days` and `week`/`weeks` are normalized only where the catalog
allows them. Units are always singular in canonical output, and equivalent durations are not
converted: `12-Month` stays `12-Month`, not `1-Year`. Former `Family - Value` spellings remain
accepted inputs, so `SOFR - 30-Day` becomes `SOFR 30-Day` and `Treasury - 1 yr` becomes
`Treasury 1-Year`.

The catalog also recognizes these complete-value source/vendor aliases case-insensitively:

| Source alias | Canonical output |
| --- | --- |
| `TSFR6M` | `SOFR Term 6-Month` |
| `SOFR30` | `SOFR 30-Day` |
| `SOFR30A` or `SOFR Avg - 30 days` | `SOFR 30-Day Average` |
| `SOFR180A` | `SOFR 180-Day Average` |
| `RCF6M` | `RCF 6-Month` |
| `RCF12M` | `RCF 12-Month` |

Approved CMT tenors also accept shorthand such as `10Y CMT`. Generic SOFR labels stay generic;
`Term` is not inferred. Opaque aliases must match the complete normalized value, so `XSOFR30` does
not partially rewrite. `NAP` and the text `null` are ordinary unknown values unless configured as
null markers with `replace_null_markers: true`. Use `on_parse_error: preserve` when an unknown
value should survive exactly as received.

#### Property-type profile

`property_type_v1` is a case-insensitive, versioned loan-property profile. A supported non-mixed
value must match a complete canonical value or approved alias. `Condominium` is the canonical
spelling; the source misspelling `Condominimum` remains accepted. `Four-Unit` remains distinct from
`Duplex`, `Two-Unit`, `Three-Unit`, and `Multifamily`. LIHTC is business-significant, so supported
spellings produce `Multifamily - LIHTC`, not plain `Multifamily`.

Mixed-use values follow a structural rule instead of being collapsed into another property type.
The complete phrase `mixed use` or `mixed-use` is moved to the front as `Mixed Use`, and every
remaining hyphen component is retained behind it with canonical ` - ` separators. Known tokens
such as `Multifamily` and `LIHTC` use their catalog spelling; any unmatched component is trimmed,
lowercased, and given an uppercase first letter. Parentheses wrapping an entire component are
cosmetic and removed, while parentheses inside a longer component remain. For example:

- `Warehouse-Mixed use` becomes `Mixed Use - Warehouse`;
- `Residential-Mixed Use` becomes `Mixed Use - Residential`; and
- `Mixed Use-MULTIFAMILY-Over 20` becomes `Mixed Use - Multifamily - Over 20`.

The fixed catalog covers the supplied residential, multifamily, agricultural, equipment,
industrial, office, retail, and specialty categories. An unknown non-mixed value is a parse error:
the profile never partially matches it or treats `Other` as a fallback. Only explicitly approved
full-value aliases may collapse a source-specific qualifier. Existing `on_parse_error` behavior
applies; use `preserve` to retain an unresolved source token exactly for downstream rules. Ranges
such as `1-4 family`, combined values such as `RE/Equipment`, and the distinct construction category
`Modular Housing` therefore remain unresolved.

#### US address and county profiles

`address_us_v1` is deterministic display normalization built from Spark expressions. It collapses
whitespace, removes commas and periods from tokens, canonicalizes common USPS suffixes,
directionals, and secondary-unit designators, and smart-cases `Mc`, apostrophe, and hyphen names.
Alphanumeric values after a unit designator, including hash-prefixed values, are uppercased
(`Apt #4b` → `Apt #4B`). Only the last suffix-like token is treated as a street suffix, so
`123 Center Street` becomes `123 Center St`, not `123 Ctr St`. The output uses punctuation-free
suffixes such as `St`, not `St.`. This is display cleanup, not geocoding, postal validation, or a
deliverability check.

`county` removes one existing trailing `County`, smart-cases the remaining name, and adds one
canonical ` County` suffix. A value containing only `County` is a parse error. The profile does not
infer that a parish, borough, municipality, or census area is a county.

#### US state and ZIP profiles

`state_us` recognizes the 50 states, Washington DC, and USPS territories `AS`, `GU`, `MP`, `PR`,
and `VI` by full name or two-letter code. It also accepts an explicit set of conventional state
abbreviations such as `Ill.`, `Calif.`, and `Wash.`. Matching is case-insensitive after whitespace
normalization, and periods and commas are ignored during scalar lookup. Output is always the
uppercase two-letter code; arbitrary abbreviation-like values are not inferred.

ZIP values remain strings so leading zeroes survive:

| Normalized ZIP input | Output |
| --- | --- |
| 1–5 digits | Left-pad to five digits (`1234` → `01234`). |
| 9 digits | Insert the ZIP+4 separator (`123456789` → `12345-6789`). |
| 1–5 digits, hyphen, then 1–4 digits | Pad each component (`123-45` → `00123-0045`). |
| 6–8 compact digits, non-digits, malformed hyphens, or an empty component | Parse error handled by `on_parse_error`. |

State and ZIP profiles also accept comma-separated property values. Components are normalized
independently, kept in source order, and rejoined with canonical `, ` spacing; for example,
`Illinois, tx` becomes `IL, TX` and `1234, 67890` becomes `01234, 67890`. `Washington, D.C.` is
recognized as one DC value rather than split. If any component is empty or invalid, the entire
field follows `on_parse_error`; partial lists are never returned. ZIP padding records
`zip_padded`, while inserting or changing ZIP+4 formatting records `zip_plus4_formatted`.

## Compile-time and DataFrame validation

Compilation checks YAML shape, required metadata, unique targets, supported scalar datatypes,
parser/type compatibility, option placement, typed defaults, null-marker modes, and effective
Boolean vocabularies. Duplicate and unknown keys fail rather than being ignored. Compilation raises
one `CompilationError` after collecting independent semantic findings; its ordered `.errors` tuple
contains the individual messages. A single finding retains its prior exception text.

When binding a DataFrame, Spark Parser also checks that:

- every present configured source is a top-level string;
- source and target names are unambiguous under Spark's active case-sensitivity resolver;
- missing sources fail unless `on_missing_source="warn"` explicitly allows substitution;
- generated result-column names do not collide with existing input fields, including present
  sources and row keys; and
- row keys are existing, unique, unambiguous input names whose projected result names do not
  collide; and
- pass-through row keys do not share names with configured targets, since their original values
  could differ from the parsed values exposed under those target names.

Independent metadata findings are raised together through `SchemaValidationError.errors`. Checks
whose prerequisites are invalid are skipped to avoid cascade messages, and recoverable
missing-source warnings are emitted only if no fatal schema issue exists.

With the default `error_mode="configured"`, `on_parse_error: fail` is lazy: it raises only when
Spark evaluates the failing target expression. An optimizer-pruned `count()` may not materialize
every target. A full target write or explicit projection does.

## DataFrameParsing outputs and audit contract

`parse_dataframe()` returns a `DataFrameParsing` wrapper:

| Property/method | Behavior |
| --- | --- |
| `parsed_df` | Target scalar columns, in configuration order; collection mode appends an error array. |
| `results_df` | Target-mapped row keys followed by audit and configuration identity columns, plus collection metadata when enabled. |
| `error_mode` | Execution mode: `"configured"` (default) or `"collect"`. |
| `warnings` | Recoverable schema warnings such as an explicitly allowed missing source. |
| `persist()` / `unpersist()` | Optional shared-plan caching on runtimes that support it. |

Databricks serverless may reject DataFrame cache APIs; omit persistence there. A row key that is a
configured source appears in `results_df` under its target name with its parsed target value and
type, allowing direct joins to `parsed_df`. Unconfigured row keys pass through unchanged. A source
that feeds multiple targets also remains a pass-through key because it has no unique target mapping.
Row keys belong to `results_df` and are not copied into `parsed_df` unless configured as target
columns too. Materializing a mapped key applies that target's null, default, and fail policies.

The result columns are:

| Column | Definition |
| --- | --- |
| `<prefix>_parse_results` | Array containing one audit struct per `audit: true` scalar column. |
| `<prefix>_config` | Struct containing configuration `id`, `version`, and content hash. |
| `<prefix>_engine_version` | Installed Spark Parser version. |
| `<prefix>_parse_errors` | Collection mode only: ordered conversion-error array, also appended to `parsed_df`. |
| `<prefix>_error_mode` | Collection mode only: the string `"collect"`. |

See [Error modes by example](#error-modes-by-example) for the same input parsed in default and
collection modes, including output values, individual errors, and row filtering.

Collection mode replaces a failed `on_parse_error: fail` conversion with a typed null without
calling Spark's `raise_error`. Configured `null`, `default`, and `preserve` policies retain their
behavior. Numeric zero invalidation and the final `default_on_null` stage still apply, so a failed
non-nullable target receives its configured null default. All conversion failures appear in the
error array, including columns with `audit: false`; successful rows receive `[]`. Errors stay in
configuration order and retain the source value before normalization.

Each error struct contains `source_column_name`, `target_column_name`, `original_value`,
`expected_data_type`, `error_code`, `message`, and `resolution`. The code is `PARSE_ERROR` and the
message is `Value could not be parsed as TYPE.`, using the expected datatype for `TYPE`. String
format failures also identify the rejecting profile, for example,
`Value could not be parsed as string with format 'state_us'.`
`resolution` describes the final outcome: `null`, `default_on_null`, `default_on_error`, or
`preserve`, including any effect of zero invalidation and the final null default. The default
`column_prefix="spark_parser"` controls the generated names shown above; collection metadata names
must not collide with input, target, or key columns.

Null source values, empty strings or null markers normalized to null, zero invalidation alone,
and allowed missing-source warnings are not conversion errors. Configuration, schema, and unrelated
Spark failures still raise. Collection stays lazy and uses native Spark expressions; materializing
the error array evaluates every configured conversion it describes. Mapped parsed keys can become
null or defaults on failure, so the inline error array preserves each error's association with its
parsed row without requiring a join. The execution mode is exposed separately from the authored
configuration: it is not a YAML option and does not change the configuration content hash.

Collection retains the full original source string in every error entry, with no truncation.
When one source feeds several failing targets, each target's error repeats that string; for
example, a 200,000-character token feeding five failing targets creates five copies in the error
array. This increases row width and the memory, transfer, and storage required to materialize
diagnostics. Writing both `parsed_df` and `results_df` stores the error array in both outputs.
For wide or large-token loads, retain diagnostics in the output that needs them and project out
unneeded diagnostic fields before writing other outputs.

Each audit struct has `source_column_name`, `target_column_name`, `parser_type`,
`expected_data_type`, `original_value`, `parsed_value`, `changed`, `effective`, `actions_applied`,
`options`, and `error`. `parsed_value` is text, except binary values are represented as canonical
base64. `changed` is true when a material null, default, error, missing-source, zero, or ZIP action
occurs, or when a string parser's final text differs from its source. The audit schema intentionally
has no nested child-path fields.

Possible actions are `source_column_missing`, `empty_string_to_null`, `null_marker_replaced`,
`parse_error_collected`, `parse_error_to_null`, `parse_error_default_applied`, `parse_error_preserved`,
`zero_invalidated`, `default_on_null_applied`, `zip_padded`, and `zip_plus4_formatted`. Routine string normalization and
successful formatting do not add action entries, but they set `changed` when the final text differs;
ZIP formatting follows the same comparison while retaining its specific ZIP actions. Successful
non-string conversion does not set `changed` on its own. For an audited, suppressed `fail` error,
collection mode records `parse_error_collected` then `parse_error_to_null`, followed by
`default_on_null_applied` when applicable, and marks `changed` as true. Here
`parse_error_collected` specifically identifies suppression of the configured `fail` policy.
Use `<prefix>_parse_errors` to enumerate or count all collected failures across every policy.

## Error modes by example

These examples use an existing SparkSession named `spark`. The input has one valid row, one row
with two malformed values, and one row with null/empty values. `record_id` is explicitly configured
so `RecordId` appears in both output DataFrames. Auditing stays at its default of `false`.

```python
from pyspark.sql import functions as F
from spark_parser import parser

bronze_df = spark.sql("""
SELECT * FROM VALUES
  ('r1', '12',   '2026-09-04'),
  ('r2', 'oops', 'not-a-date'),
  ('r3', NULL,   '')
AS source(record_id, units, opened_date)
""")

config = parser.compile_yaml("""
parser_config_id: error_mode_example
parser_config_name: Error Mode Example
version: "1"
columns:
  - source_column_name: record_id
    target_column_name: RecordId
    expected_data_type: string
    parser: string
  - source_column_name: units
    target_column_name: Units
    expected_data_type: integer
    parser: integer
  - source_column_name: opened_date
    target_column_name: OpenedDate
    expected_data_type: date
    parser: date
""")
```

### Default mode: raise when a failing target is evaluated

Omitting `error_mode` is equivalent to `error_mode="configured"`. This configuration uses the
default `on_parse_error: fail`, so materializing `Units` raises on `r2`:

```python
strict = parser.parse_dataframe(bronze_df, config, key_columns=["record_id"])

# Building the parsing wrapper does not scan the rows. This action evaluates Units and raises.
strict.parsed_df.select("Units").collect()
```

The Spark exception includes this message; the enclosing exception class varies by runtime:

```text
Spark Parser could not parse source 'units' into target column 'Units' as integer: oops
```

The action fails rather than returning a partial set of valid rows. The default `parsed_df` has
only `RecordId`, `Units`, and `OpenedDate`; it has no `spark_parser_parse_errors` column. Use an
action that evaluates the target to check failure behavior: `count()` can prune target expressions.

### Collection mode: return parsed rows with all conversion errors

Use the same input and compiled configuration, changing only the execution mode:

```python
collected = parser.parse_dataframe(
    bronze_df, config, key_columns=["record_id"], error_mode="collect"
)

collected.parsed_df.select(
    "RecordId", "Units", "OpenedDate",
    F.size("spark_parser_parse_errors").alias("ErrorCount"),
).orderBy("RecordId").show()
```

The result is:

| RecordId | Units | OpenedDate | ErrorCount |
| --- | --- | --- | --- |
| r1 | 12 | 2026-09-04 | 0 |
| r2 | NULL | NULL | 2 |
| r3 | NULL | NULL | 0 |

Both malformed values in `r2` are recorded, even though neither column enables auditing.
The error array is `[]` for `r1` and `r3`: a null source and an empty string normalized to null
are not conversion failures.

Separate the rows and inspect each error without joining to `results_df`:

```python
bad_rows = collected.parsed_df.filter(F.size("spark_parser_parse_errors") > 0)
good_rows = collected.parsed_df.filter(F.size("spark_parser_parse_errors") == 0)

bad_rows.select(
    "RecordId", F.explode("spark_parser_parse_errors").alias("error")
).select(
    "RecordId", "error.target_column_name", "error.original_value", "error.resolution"
).show(truncate=False)
```

`bad_rows` contains `r2`; `good_rows` contains `r1` and `r3`. The exploded error details are:

| RecordId | target_column_name | original_value | resolution |
| --- | --- | --- | --- |
| r2 | Units | oops | null |
| r2 | OpenedDate | not-a-date | null |

Each entry also has `error_code="PARSE_ERROR"`, its source column and expected datatype, and a
message such as `Value could not be parsed as integer.` The same error array appears in
`collected.results_df`, alongside `spark_parser_error_mode="collect"` and the unchanged
configuration identity. The optional `spark_parser_parse_results` audit array remains empty here.

### Collection mode with a required-value default

To see how final null handling interacts with collection, make `Units` non-nullable with a
fallback of `0`. `to_mapping()` returns a detached configuration, leaving `config` unchanged:

```python
required_config = parser.to_mapping(config)
required_config["columns"][1]["parser"].update(is_nullable=False, default_on_null=0)

with_defaults = parser.parse_dataframe(
    bronze_df, required_config, key_columns=["record_id"], error_mode="collect"
)
with_defaults.parsed_df.select(
    "RecordId", "Units",
    F.size("spark_parser_parse_errors").alias("ErrorCount"),
).orderBy("RecordId").show()
```

| RecordId | Units | ErrorCount |
| --- | --- | --- |
| r1 | 12 | 0 |
| r2 | 0 | 2 |
| r3 | 0 | 0 |

The `Units` error for `r2` now has `resolution="default_on_null"`; assigning a default does not
erase the error. `r3` gets the same default without a conversion error. In default execution mode,
the malformed `r2` value would still raise: `on_parse_error: fail` runs before `default_on_null`.

The execution mode and each column's `on_parse_error` policy are separate controls. For malformed,
non-null input, the policies compare as follows (targets are nullable unless noted):

| Column policy | Default execution (`configured`) | Collection execution (`collect`) | Collected resolution |
| --- | --- | --- | --- |
| `fail` (default) | Raises when evaluated. | Returns NULL and records an error. | `null` |
| `fail`, `is_nullable: false`, `default_on_null: 0` | Raises before the null default. | Returns 0 and records an error. | `default_on_null` |
| `null` | Returns NULL. | Returns NULL and records an error. | `null` |
| `default`, `default_on_error: 7` | Returns 7. | Returns 7 and records an error. | `default_on_error` |
| `preserve` on a string formatter | Returns the exact original token. | Preserves the token and records an error. | `preserve` |

For example, a `state_us` string formatter with `on_parse_error: preserve` returns an unrecognized
token such as `"Atlantis"` in either execution mode. Collection additionally records the failure
with `message="Value could not be parsed as string with format 'state_us'."` and
`resolution="preserve"`.
Default execution can record handled errors in `results_df` when `audit: true`; it never adds the
collection error array. Neither execution mode converts invalid configuration or incompatible
source schemas into row-level conversion errors.

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
