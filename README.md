# Spark Parser

Spark Parser turns raw bronze strings into typed Spark columns from a strict YAML contract. It is
built for ingestion pipelines where source files arrive as strings but the next layer needs a
predictable schema, consistent cleanup rules, and a record of what changed.

Every row transformation is a native Spark SQL expression. There are no Python or pandas UDFs, so
Spark can optimize the generated plan and run it the same way in local tests and Databricks jobs.

Use the package to:

- map source columns to target names and Spark datatypes;
- parse scalar values such as integers, decimals, Booleans, dates, timestamps, and binary data;
- parse nested arrays, structs, and string-keyed maps recursively;
- normalize common US address, county, state, and ZIP values;
- decide what happens to nulls, invalid values, and bad nested elements;
- produce optional row-level audit details alongside the parsed data; and
- review, serialize, and hash the fully resolved configuration before a load runs.

## Quick start

For local development, install the Spark and test extras:

```text
python -m pip install -e ".[spark,test]"
```

Do not install the Spark extra on Databricks. The Databricks runtime already owns Spark and Py4J;
installing another PySpark distribution beside it can break that connection. Install the package
wheel with `--no-deps` on a runtime that provides Spark 3.5 or newer and PyYAML 6 or newer.

```yaml
parser_config_id: bronze_customer_load
parser_config_name: Bronze Customer Load
version: 1.0.0
description: Parse one bronze customer delivery for target.
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
    target_column_name: CustomerName
    expected_data_type: string
    parser:
      type: string
      audit: true

  - source_column_name: mailing_address
    target_column_name: MailingAddress
    expected_data_type: string
    parser:
      type: string
      format: address_us_v1
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

  - source_column_name: borrower_names_json
    target_column_name: BorrowerNames
    expected_data_type: array<string>
    parser:
      type: array
      element_parser:
        type: string
        format: pascal
      on_element_error: null
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

For most callers, `parser` is the only object they need. It is a package-level
`SparkParserService` instance that handles compilation, review, serialization, and DataFrame
parsing. Create a separate `SparkParserService()` only when an application needs its own service
instance for dependency injection or test isolation.

For an executable end-to-end walkthrough, open the
[Spark Parser user guide](examples/spark_parser_user_guide.py) as a Databricks source notebook.

## How the package fits together

The main path through the package is:

`YAML or mapping` → `YamlParserConfigCompiler` → `ParserConfig` → `SparkDataFrameParser` →
`DataFrameParsing`

`SparkParserService` sits in front of that path and wires the pieces together. The compiler side
does not import PySpark, which means config authoring, review, datatype validation, serialization,
and hashing can run in a lightweight Python environment. PySpark is loaded only when code asks for
the DataFrame runtime.

### Public entry point and orchestration

- [`__init__.py`](src/spark_parser/__init__.py) defines the supported public import surface. It
  exports the compiler, models, enums, service, and errors directly, while loading
  `SparkDataFrameParser` and `DataFrameParsing` only when requested. That lazy boundary keeps
  compiler-only tools from needing PySpark.

- [`service.py`](src/spark_parser/service.py) is the package facade. It powers the shared `parser`
  object, delegates compilation and serialization, loads the Spark runtime only for DataFrame
  work, exposes parser/config metadata, and builds `ConfigReviewReport` objects in JSON or
  Markdown form. Application code should normally start here.

### Configuration contract

- [`compiler_yaml.py`](src/spark_parser/compiler_yaml.py) is the trust boundary for authored
  configuration. It reads YAML or mappings, rejects duplicate and unknown keys, resolves inherited
  defaults, checks parser/type compatibility, validates recursive parser trees and typed defaults,
  and returns an immutable `ParserConfig`. None of that work requires a Spark session.

- [`models.py`](src/spark_parser/models.py) contains the frozen dataclasses shared by the compiler,
  serializer, service, and runtime. These objects represent global settings, parser options,
  nested fields/elements/values, column mappings, and the complete compiled configuration. Once a
  config is compiled, these models are the contract passed through the rest of the package.

- [`data_types.py`](src/spark_parser/data_types.py) parses the supported Spark DDL grammar without
  importing PySpark. It canonicalizes aliases, decimal precision and scale, and recursively nested
  arrays, structs, and maps into `SparkDataType` objects. The compiler uses that model to catch
  schema mistakes before a cluster is involved.

- [`defaults.py`](src/spark_parser/defaults.py) is the single source of truth for every omitted
  parser option, including null behavior, Boolean vocabularies, datetime formats, binary encoding,
  and complex-parser settings. The compiler, machine-readable metadata API, serializer, and audit
  output read from these values so runtime defaults have one definition; tests keep the published
  contract vocabulary aligned.

- [`enums.py`](src/spark_parser/enums.py) defines the accepted vocabulary for parser types,
  formatting modes, error policies, input formats, and encodings. It also groups numeric and
  complex parser families for the validation and runtime branches that apply to them.

### Discovery, serialization, and identity

- [`metadata.py`](src/spark_parser/metadata.py) builds the dictionaries returned by methods such as
  `parser.string.describe()` and `parser.config.describe()`. It turns the package's enums and
  defaults into machine-readable help for notebooks, config editors, catalogs, and review tools.

- [`serializer.py`](src/spark_parser/serializer.py) converts a compiled config into a fully
  resolved, JSON-compatible mapping. It also produces deterministic canonical JSON and the SHA-256
  content hash, so identity covers the complete resolved configuration rather than the shorthand
  used in the source YAML.

### Spark execution

- [`spark_runtime.py`](src/spark_parser/spark_runtime.py) is the execution engine. It checks the
  incoming DataFrame schema, builds native Spark expressions for scalar and recursive parsing,
  applies null/default/error policies, and constructs audit metadata. It returns a lazy plan and
  never collects source rows or calls a Python UDF.

- [`address_formats.py`](src/spark_parser/address_formats.py) holds the Spark-expression helpers
  for `address_us_v1`, `county`, `state_us`, and `zip`. Keeping these domain-specific rules outside
  the main runtime makes the lookup tables and formatting decisions easier to review and test.

- [`dataframe_parsing.py`](src/spark_parser/dataframe_parsing.py) wraps the shared lazy plan produced
  by the runtime. Its `parsed_df` and `results_df` properties expose the target and audit
  projections, while `persist()` and `unpersist()` let runtimes with DataFrame-cache support avoid
  recomputing the plan when materializing both.

### Package-wide support

- [`exceptions.py`](src/spark_parser/exceptions.py) defines the public error hierarchy:
  `CompilationError` for bad configuration, `SchemaValidationError` for an incompatible DataFrame,
  and `SchemaWarning` for recoverable binding issues such as a missing source column. All package
  errors inherit from `SparkParserError`.

- [`version.py`](src/spark_parser/version.py) owns the package version. That value appears in wheel
  metadata, configuration review reports, and the parser identity columns written to
  `results_df`.

## Service API

The shared `parser` service covers the normal workflow:

| Method/property | Result and behavior |
| --- | --- |
| `parser.compile_text(text)` | Compile YAML text to an immutable, fully resolved `ParserConfig`; raises `CompilationError` on invalid authoring metadata. |
| `parser.compile_path(path)` | Read and compile a UTF-8 YAML file. |
| `parser.compile_mapping(mapping)` | Compile an already-loaded YAML-compatible mapping. |
| `parser.compile_yaml(source)` | Type-driven dispatcher accepting YAML text, a `pathlib.Path`, or a mapping. A string is always text; use `Path(...)` or `compile_path()` for a file. |
| `parser.parse_dataframe(df, config, ...)` | Parse a DataFrame. `config` may already be compiled or may be any input accepted by `compile_yaml`. |
| `parser.to_mapping(config)` | Return a JSON-compatible mapping containing every resolved option. |
| `parser.canonical_json(config)` | Return deterministic canonical JSON. |
| `parser.content_hash(config)` | Return the SHA-256 identity of the resolved configuration. |
| `parser.defaults()` | Return a detached mapping of all code-owned defaults. |
| `parser.normalize_data_type(value)` | Validate supported scalar/complex Spark DDL and return its canonical representation without starting Spark. |
| `parser.describe()` | Return metadata for every parser type. |
| `parser.describe("date")` | Return metadata for one parser by name. |
| `parser.string.describe()` | Return the string parser's arguments, defaults, behavior, and gotchas. The same accessor exists for every parser type. |
| `parser.config.describe()` | Return top-level, global, and column metadata definitions. |
| `parser.review_yaml(source)` | Validate YAML and return a `ConfigReviewReport` instead of raising for authoring errors. |

Metadata accessors are available for `string`, `byte`, `short`, `integer`, `long`, `float`,
`decimal`, `double`, `binary`, `boolean`, `date`, `timestamp`, `timestamp_ntz`, `array`, `struct`,
and `map`. Each one returns a plain dictionary, which makes it easy for notebooks and config tools
to show the same arguments and defaults the compiler uses.

```python
string_help = parser.string.describe()
for argument in string_help["arguments"]:
    print(argument["name"], argument["required"], argument["default"])

all_defaults = parser.defaults()
config_help = parser.config.describe()
```

## Configuration review report

Use `review_yaml()` when you want to inspect a config without running a DataFrame. It accepts YAML
text, a `pathlib.Path`, or an already loaded mapping. String inputs are always treated as YAML text,
so a same-named local file cannot silently change how a configuration is interpreted. A valid
report shows:

- identity, ownership, version, and the canonical content hash;
- which compiler checks passed and which ones did not apply;
- every source-to-target mapping and target datatype;
- the resolved parser tree for nested arrays, structs, and maps;
- global settings and inherited column options;
- effective error, nullability, formatting, and audit behavior;
- warnings such as missing ownership metadata or a config with no audited columns; and
- canonical YAML with inherited values and defaults filled in.

The report only reviews configuration. Data-dependent issues, such as a configured source column
that is missing from an input DataFrame, are reported later by `parse_dataframe()`.

```python
from pathlib import Path

report = parser.review_yaml(Path("customer_parser.yaml"))

if not report.is_valid:
    raise ValueError(report.errors)

display(report.to_markdown())
report.write_markdown("customer_parser_review.md")
report.write_json("customer_parser_review.json")
report_json = report.to_json()
report_payload = report.to_mapping()
```

`ConfigReviewReport` exposes `is_valid`, `source`, `errors`, `warnings`, `summary`,
`validation_checks`, `column_reviews`, and `resolved_config`. Use `to_mapping()` or `to_json()` for
automation and `to_markdown()` for a review document. `write_json()` and `write_markdown()` save
UTF-8 files and return the destination `Path`. Invalid configs still produce a report: errors are
included in the result and the Markdown status is `FAIL`. Reports are transparent mutable
data-transfer objects containing JSON-shaped dictionaries. Treat their fields as report-owned, or
use `to_mapping()` when an independently mutable copy is needed.

## Testing

Run the Spark-independent unit suite with:

```text
python -m pytest tests/unit -q
```

Run the real-Spark integration suite with a compatible Spark and Java runtime using:

```text
SPARK_PARSER_REQUIRE_JAVA=1 python -m pytest tests/integration -q
```

To run both pytest suites in one command:

```text
SPARK_PARSER_REQUIRE_JAVA=1 python -m pytest tests/unit tests/integration -q
```

On PowerShell:

```powershell
$env:SPARK_PARSER_REQUIRE_JAVA = "1"
python -m pytest tests/unit tests/integration -q
```

CI runs every Python 3.10/3.12 and PySpark 3.5/4.1 boundary pairing plus Python 3.13 with PySpark
4.1, requires the Spark tier, and enforces at least 90% combined statement-and-branch coverage for
the package. A separate Spark-independent lane covers Python 3.11, so every advertised Python minor
is exercised while Python 3.13 runtime coverage stays on the current Spark line.

When executed against an ambient Spark Connect or Databricks serverless session, five tests marked
`classic_spark` skip before touching unsupported APIs. Those proofs deliberately inspect
`SparkContext`, mutate internal resolver or optimizer settings, or exercise live cache state. All
portable parser behavior continues to run, including the restricted-configuration and
cache-delegation contracts.

The Databricks system notebook lives under `tests/system`. It runs from a repository checkout and
does not need a wheel, Volume, release metadata, or table writes. See the
[unit-test summary](docs/spark_parser_unit_test_summary.md) and
[system-test summary](docs/spark_parser_system_test_summary.md) for their inventories and execution
contracts.

## Configuration metadata

### Top-level arguments

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `parser_config_id` | Yes | — | Stable, non-empty ID for the parsing configuration. |
| `parser_config_name` | Yes | — | Non-empty human-readable name. |
| `version` | Yes | — | Non-empty version string identifying this config contract. Lifecycle status is outside the parser config. |
| `columns` | Yes | — | Non-empty ordered list of source-to-target parser mappings. |
| `description` | No | `null` | Purpose and configuration scope. |
| `owner` | No | `null` | Accountable person or team. |
| `owner_department` | No | `null` | Accountable department. |
| `globals` | No | `{}` | Global null and Boolean vocabularies inherited by columns. |

The compiler rejects duplicate YAML keys, non-string metadata keys, and unknown arguments. Quote a
numeric-looking version such as `"1"`; otherwise YAML may load it as a number. Merge keys (`<<`)
are not supported because they can hide values during review, though ordinary anchors and aliases
still work.

### Global arguments

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `null_markers` | No | `[]` | Ordered global null-token strings. Duplicates are removed. Tokens are inert until a column sets `replace_null_markers: true`. |
| `null_marker_case_sensitive` | No | `true` | Default exact-case null-token matching. |
| `true_values` | No | `["true"]` | Non-empty global tokens mapped to Boolean true. |
| `false_values` | No | `["false"]` | Non-empty global tokens mapped to Boolean false. |
| `boolean_case_sensitive` | No | `false` | Whether global Boolean-token matching requires exact case. The default is case-insensitive. |

With case-sensitive matching, `NA` matches only `NA`. With case-insensitive matching, `NA`, `na`,
and `Na` all match. Null markers are checked after whitespace collapse and trimming, and a column
can override the global setting.

Quote tokens such as `"true"`, `"false"`, `"yes"`, `"no"`, `"on"`, and `"off"`. YAML may otherwise
load them as Boolean values instead of the source strings the parser expects.

### Column arguments

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `source_column_name` | Yes | — | Top-level bronze name preserved verbatim and resolved with Spark's active identifier resolver (exact when `spark.sql.caseSensitive=true`). It may be reused by multiple mappings. A missing source fails DataFrame binding unless the caller explicitly selects `on_missing_source="warn"`. |
| `target_column_name` | Yes | — | Non-blank authored name preserved verbatim and emitted by `parsed_df`. Exact duplicates fail compilation; resolver-sensitive collisions fail DataFrame binding. |
| `expected_data_type` | Yes | — | Exact target Spark DDL type. Scalars and recursively nested `array<T>`, `struct<field:T,...>`, and `map<string,T>` are supported. |
| `parser` | Yes | — | Matching scalar or complex parser name, or a mapping containing `type` and options. |

`parser.type` chooses the conversion, while `expected_data_type` defines the target schema. Keeping
them separate lets the compiler verify details such as integer width, decimal precision and scale,
and every field inside a nested type. The parser tree must match that schema all the way down.
Accepted scalar aliases are `tinyint` → `byte`, `smallint` → `short`, `int` → `integer`, `bigint`
→ `long`, `real` → `float`, `bool` → `boolean`, `dec`/`numeric` → `decimal`, and
`timestamp_ltz` → `timestamp`.
The same aliases are accepted by `parser.describe(alias)`; metadata catalogs and serialized
mappings use canonical names.

Use the short form, such as `parser: date`, when the defaults are right for the column. Use mapping
form when you need to set an option:

```yaml
- source_column_name: arm_next_rate_change_date
  target_column_name: ArmNextRateChangeDate
  expected_data_type: date
  parser:
    type: date
    audit: true
```

If `arm_next_rate_change_date` is missing from the DataFrame, binding fails by default. A caller may
set `on_missing_source="warn"` only when substituting a typed null—or the required
`default_on_null` for a non-nullable mapping—is intentional. That mode emits a `SchemaWarning`,
adds the message to `DataFrameParsing.warnings`, and records `source_column_missing` in an enabled
audit entry.

## Common parser arguments

These options apply to every parser type.

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `type` | Mapping form only | — | Parser implementation matching `expected_data_type`. |
| `collapse_whitespace` | No | `true` for scalar/leaf parsers; resolved `false` for outer complex containers | Replace every run of whitespace—left, right, and internal—with one ordinary space. |
| `trim_whitespace` | No | `true` | Remove leading and trailing spaces, tabs, line breaks, and non-breaking spaces after collapse. Both defaults together normalize all surrounding and internal whitespace. |
| `empty_is_null` | No | `true` | Convert an empty normalized string to null. |
| `replace_null_markers` | No | `false` | Convert effective null-token matches to null. |
| `null_markers` | No | Inherited globals | Column null-token list. Supplying it does not by itself enable replacement. |
| `null_markers_mode` | No | `replace` | With column `null_markers`, `replace` uses only that list and `extend` appends it to globals. It cannot be supplied without column markers. |
| `null_marker_case_sensitive` | No | Inherited global | Override null-token case sensitivity for this column. |
| `is_nullable` | No | `true` | Allow the final target value to remain null. |
| `default_on_null` | Conditional | No default | Required only when `is_nullable: false`; must be non-null and exactly valid for the expected type. |
| `on_parse_error` | No | `fail` | `fail`, `null`, `default`, or string-only `preserve`; see the error-mode section below. |
| `default_on_error` | Conditional | No default | Required only with `on_parse_error: default`; must fit the expected type. |
| `audit` | No | `false` | Add a row-level audit struct for this column. |

`review_yaml()` warns when global null markers are configured but no parser node enables
`replace_null_markers`; in that configuration the markers are intentionally inert.

The compiler rejects options on the wrong parser, invalid value types, unknown keys, incompatible
defaults, empty required lists, and contradictory settings.

For a complex JSON value, the container parser applies trimming, empty/null-marker handling,
nullability, defaults, and `on_parse_error` to the whole source string. It does not collapse
whitespace inside raw JSON because that could change a quoted value. As a result,
`collapse_whitespace` resolves to `false` for outer `array`, `struct`, and `map` parsers everywhere
that effective options are reported.

After decoding, each leaf parser applies its own whitespace rules. Nested parsers report their
options and error paths through the top-level audit entry instead of creating separate entries.
The exact lowercase JSON literal `null` is a successful typed null and records
`json_null_to_null` when audit is enabled.

## Parser-specific reference

### String

String supports the common arguments plus `format`.

| `format` | Result | Example |
| --- | --- | --- |
| `null` or `none` | Preserve case after whitespace normalization. | `"  Acme   LLC "` → `"Acme LLC"` |
| `lower` | Lowercase the normalized value. | `"Acme LLC"` → `"acme llc"` |
| `upper` | Uppercase the normalized value. | `"Acme LLC"` → `"ACME LLC"` |
| `title` | Lowercase and capitalize words while retaining normalized spaces. | `"  LOAN   status "` → `"Loan Status"` |
| `pascal` | Lowercase, title-case, and remove spaces. Intended for identifiers rather than human names. | `"account status"` → `"AccountStatus"` |
| `address_us_v1` | Apply deterministic US address display normalization. | `"123 mccormick st. apt 4b"` → `"123 McCormick St Apt 4B"` |
| `county` | Smart-case a county name and ensure exactly one trailing `County`. | `"mclean county"` → `"McLean County"` |
| `state_us` | Return one or more uppercase two-letter abbreviations for US states, territories, or Washington, DC. | `"Illinois"` → `"IL"`; `"Puerto Rico, tx"` → `"PR, TX"` |
| `zip` | Return one or more canonical ZIP5 or ZIP+4 values as a string. | `"1234"` → `"01234"`; `"1234, 67890"` → `"01234, 67890"` |

`title` uses Spark's `initcap` behavior. It works well for display labels that should keep their
spaces, but it does not know the `Mc`, apostrophe, hyphen, street-suffix, or unit rules used by
`address_us_v1` and `county`.

#### Address profile

`address_us_v1` is a versioned display formatter built from Spark expressions. It:

- collapses whitespace and removes commas/periods from tokens;
- recognizes common USPS suffix names and aliases (`Street`/`St.` → `St`, `Avenue` → `Ave`);
- canonicalizes directionals (`northwest` → `NW`);
- canonicalizes common secondary-unit designators (`apartment` → `Apt`, `suite` → `Ste`);
- smart-cases `Mc` names (`mccormick` → `McCormick`), common exceptions such as `McLean`,
  apostrophe names, and hyphenated names; and
- uppercases alphanumeric values following a unit designator and hash-prefixed unit values
  (`Apt 4b` → `Apt 4B`, `Apt #4b` → `Apt #4B`).

Only the last suffix-like token is treated as a street suffix. That keeps an address such as
`123 Center Street` from turning into `123 Ctr St`. Tokens made empty by comma/period cleanup are
removed before the address is joined back together, and null stays null.

Suffixes are punctuation-free (`St`, not `St.`), following
[USPS Publication 28 conventions](https://pe.usps.com/text/pub28/28apc_002.htm). This formatter is
not a geocoder, deliverability check, or full postal validator, and casing alone cannot resolve
every human-name ambiguity. Its job is narrower: provide predictable display cleanup without a
Python UDF or an external postal model.

#### County profile

`county` removes an existing `County` suffix, smart-cases the name, and adds one canonical
` County` suffix. A value containing only `County` is a parse error. It does not turn parishes,
boroughs, municipalities, or census areas into counties, so use it only for a true county field.

#### State profile

`state_us` recognizes the 50 US state names, the USPS territories `AS`, `GU`, `MP`, `PR`, and `VI`,
their full names, their two-letter postal abbreviations, conventional state abbreviations such as
`Ill.`, `Calif.`, and `Wash.`, plus `District of Columbia`, `Washington DC`, `Washington, D.C.`,
and `DC`. Matching is case-insensitive after whitespace normalization; periods and commas are
ignored during lookup. The allow-list is fixed, so stripping punctuation does not make an
arbitrary three-letter value valid. Output is always the uppercase two-letter abbreviation.

Comma-separated property values are parsed independently, kept in source order, and joined with
canonical `, ` spacing. `Washington, D.C.` remains one recognized value rather than being split.
If any component is empty or invalid, the entire field follows `on_parse_error`; the formatter does
not return partially parsed lists.

An unknown non-null value follows `on_parse_error`: use `preserve` when a nonstandard source value
such as `Mul` must survive, or use `null` or an explicit default when invalid state text should not
reach the target.

```yaml
- source_column_name: state
  target_column_name: StateCode
  expected_data_type: string
  parser:
    type: string
    format: state_us
    on_parse_error: preserve
    audit: true
```

#### ZIP profile

ZIP values remain strings so leading zeroes are preserved.

| Normalized input | Output |
| --- | --- |
| 1–5 digits | Left-pad to five digits (`1234` → `01234`). |
| 9 digits | Treat the last four digits as the extension (`123456789` → `12345-6789`). |
| `1–5 digits` + hyphen + `1–4 digits` | Pad each component independently (`123-45` → `00123-0045`). |
| 6–8 compact digits, non-digits, malformed hyphens, or empty components | Parse error handled by `on_parse_error`. |

Audited padding records `zip_padded`; inserting or changing ZIP+4 formatting records
`zip_plus4_formatted`.

#### Multiple-property state and ZIP fields

When one loan field contains a comma-separated state or ZIP for each property, keep the target as
`string` and use the ordinary `state_us` or `zip` profile. A single value follows the original
scalar behavior; multiple values are parsed independently and rejoined with canonical spacing.

```yaml
- source_column_name: property_states
  target_column_name: PropertyStates
  expected_data_type: string
  parser:
    type: string
    format: state_us
    on_parse_error: fail

- source_column_name: property_zips
  target_column_name: PropertyZips
  expected_data_type: string
  parser:
    type: string
    format: zip
    on_parse_error: fail
```

For example, `Illinois` becomes `IL`, `Illinois, tx` becomes `IL, TX`, `1234` becomes `01234`,
and `1234, 67890` becomes `01234, 67890`.

### Byte, short, integer, and long

`byte`, `short`, `integer`, and `long` map to Spark's signed 8-bit, 16-bit, 32-bit, and 64-bit
integers. Their aliases are `tinyint`, `smallint`, `int`, and `bigint`. Overflow and fractional
input are parse errors.

Numeric parsing accepts safe alternate spellings such as a leading plus sign, leading zeroes,
`.5`, and `5.` by normalizing them before strict JSON conversion. Scientific notation is accepted
for decimal/float/double values; integral parsers reject exponent notation such as `1e5` rather
than silently coercing it.

Each adds `zero_is_valid` (default `true`). When false, a successfully parsed zero becomes null
before final null handling. If the column is non-nullable, `default_on_null` is then assigned.
A zero `default_on_null` or `default_on_error` is rejected when zero is invalid because it could
not survive the subsequent zero-invalidating step.

### Decimal

`expected_data_type` must be `decimal(p,s)`, with precision from 1 through 38 and scale from zero
through precision. Defaults are checked against that exact shape. Use decimal instead of double
for currency and other exact base-10 values. It supports `zero_is_valid` with the same behavior as
the integer parsers. Spark rounds source values to the configured scale (`1.239` becomes `1.24` in
`decimal(18,2)`), while an over-scale default fails at compile time. String-authored decimal
defaults use the same ASCII decimal/scientific grammar as normalized bronze numeric input;
underscores and surrounding whitespace are rejected during authoring. Resolved mappings serialize
exact decimals as strings, which recompiles losslessly without a binary floating-point detour.

### Float and double

Float and double use Spark's single- and double-precision representations. Defaults must be finite,
and a float default must fit Spark's binary32 range without becoming infinity or underflowing to
zero. Both support `zero_is_valid`. Use decimal when exact base-10 representation matters.

### Binary

Binary adds `encoding`, with `base64` as the default. `hex` and `utf8` are also supported.
Base64 uses the standard alphabet and requires correct padding; embedded whitespace, missing or
excess padding, and non-alphabet characters are invalid. Hexadecimal accepts an empty or odd-length
digit sequence, matching Spark, but rejects whitespace and non-ASCII digits. Invalid base64 or
hexadecimal input follows `on_parse_error`; UTF-8 accepts every normalized string. The typed target
value is Spark `binary`, while audit `parsed_value` is always canonical base64 so audit storage
remains printable and encoding-independent.

### Boolean

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `true_values` | No | Inherited global `["true"]` | Non-empty string tokens mapped to true. |
| `false_values` | No | Inherited global `["false"]` | Non-empty string tokens mapped to false. |
| `boolean_values_mode` | No | `replace` | When column token lists are supplied, `replace` replaces each supplied side and `extend` appends it to globals. An omitted side continues to inherit its global list. |
| `boolean_case_sensitive` | No | Inherited global `false` | Exact-case matching when true; lowercase comparison when false. |

The resolved true and false sets cannot overlap under the active case rule. Exact and ASCII-only
overlap fails Spark-free compilation. When a case-insensitive vocabulary contains non-ASCII text,
the review check is `DEFERRED`; `parse_dataframe()` lowers and compares both sets with the active
Spark runtime's Unicode tables during metadata-only binding, including for empty DataFrames and
nested empty containers. This avoids both ambiguous output and host-dependent rejection when
Python and Spark implement different Unicode versions.
Any other non-null token is a parse error, not an implicit false.

```yaml
globals:
  true_values: ["true", Y]
  false_values: ["false", N]

columns:
  - source_column_name: approval_status
    target_column_name: IsApproved
    expected_data_type: boolean
    parser:
      type: boolean
      true_values: [approved]
      false_values: [rejected]
      boolean_values_mode: extend
```

### Date, timestamp, and timestamp_ntz

| Parser | Argument | Default | Behavior |
| --- | --- | --- | --- |
| `date` | `formats` | `[yyyy-MM-dd, yyyy-MM-dd'T'HH:mm:ss[.SSSSSS], yyyy-MM-dd HH:mm:ss[.SSSSSS], MM/dd/yyyy h:mm a, MM/dd/yyyy h:mm:ss a]` | Non-empty ordered Spark datetime patterns; first successful parse preserves the authored calendar fields, then casts to date without session-timezone conversion. Offset-bearing input requires an explicitly configured offset-bearing format. |
| `timestamp` | `formats` | `[yyyy-MM-dd'T'HH:mm:ss[.SSSSSS]XXX, yyyy-MM-dd'T'HH:mm:ss[.SSSSSS], yyyy-MM-dd HH:mm:ss[.SSSSSS], MM/dd/yyyy h:mm a, MM/dd/yyyy h:mm:ss a]` | First successful parse wins. ISO offsets, local ISO timestamps, optional microseconds, and the two known US exports are built in. |
| `timestamp_ntz` | `formats` | `[yyyy-MM-dd'T'HH:mm:ss[.SSSSSS], yyyy-MM-dd HH:mm:ss[.SSSSSS], MM/dd/yyyy h:mm a, MM/dd/yyyy h:mm:ss a]` | First successful parse wins without applying a session timezone. Offset-bearing input is rejected. |

Formats are tried in order. The parser does not guess because forms such as `MM/dd/yyyy` and
`dd/MM/yyyy` can both be valid while meaning different dates. These are Spark datetime patterns,
not Python `strptime` patterns. `timestamp` represents an instant and follows the active Spark SQL
session timezone. `timestamp_ntz` represents local wall-clock time, so it does no timezone
conversion and rejects offset-bearing defaults at compile time. Timestamp defaults use strict ISO
text; offset-aware defaults are canonicalized to the equivalent UTC instant, while naive timestamp
and timestamp-NTZ defaults retain their authored local wall-clock value.

The built-in formats cover ISO text, optional microseconds, `Z` or numeric offsets for
`timestamp`, SQL-style local timestamps, and known US exports such as `09/30/2026 8:08 AM`.
Slash-based values are always month/day/year and use a 12-hour clock from `1` through `12`, with or
without a leading zero; there is no locale guessing. The `date` parser drops the time after a
successful parse, while `timestamp` and `timestamp_ntz` keep it. When an explicitly configured
date format contains an offset, the parser validates that offset but preserves the calendar date
written in the source rather than projecting the represented instant through the session timezone.

Set `formats` when a source uses a different contract. A bare value such as `09/30/2026` is not a
default because its locale is ambiguous. Offset-bearing values are also excluded from the default
`date` formats because reducing an instant-bearing value to a date discards its time and offset
semantics; configure such a format explicitly when retaining the authored calendar day is intended.

Before Spark parses a built-in datetime format, the package checks that the whole token has the
right shape. This avoids a Spark 3.5 `spark.sql.legacy.timeParserPolicy=EXCEPTION` failure when the
parser is simply moving to the next format. Patterns outside the package's built-in guarded pattern
set require `spark.sql.legacy.timeParserPolicy=CORRECTED`; DataFrame binding fails with an
actionable error under `EXCEPTION` or `LEGACY`. This prevents a malformed row from bypassing
`on_parse_error` and aborting the job.

### Array

Array parses each value with a child parser. The element type comes from `array<T>` and must match
`element_parser`.

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `input_format` | No | `json` | `json` for any recursive element type, or `delimited` for scalar elements. |
| `delimiter` | Conditional | — | Required for `delimited`; treated literally rather than as a regular expression. |
| `element_parser` | Yes | — | Recursive parser matching `T`. The array owns element-level audit and error policy, so `audit` and `on_parse_error` are not accepted here. |
| `on_element_error` | No | `fail` | `fail` raises lazily, `null` sends a typed null through final-null handling, `drop` removes the bad element, and string-only `preserve` retains its raw token. |
| `drop_null_elements` | No | `false` | Remove both source nulls and values resolved to null. |
| `distinct` | No | `false` | Remove duplicate parsed elements, preserving first occurrence order; rejected when the element tree contains a non-comparable map. |

```yaml
- source_column_name: borrower_scores
  target_column_name: BorrowerScores
  expected_data_type: array<decimal(8,2)>
  parser:
    type: array
    input_format: json
    element_parser:
      type: decimal
      zero_is_valid: false
    on_element_error: null
    drop_null_elements: false
    distinct: false
    audit: true
```

Delimited input is a literal split; it does not implement CSV quoting or escaping. Use Spark's CSV
reader for CSV-like data, or normalize the source to a JSON array first. JSON arrays keep their
order. Error paths use zero-based indexes such as `$[2]` or `$[1].birth_date`.

### Struct

Struct parses a JSON object into the schema declared by `expected_data_type`. Its `fields` list
must configure every target field once, though source and target field names may differ.

| Field argument | Required | Behavior |
| --- | ---: | --- |
| `source_field_name` | Yes | Non-blank JSON field name preserved verbatim. |
| `target_field_name` | Yes | Non-blank name preserved verbatim; must exactly match one field in the parent `struct<...>` datatype. Use a backtick-quoted DDL field when the name contains spaces or punctuation. |
| `parser` | Yes | Recursive parser inferred against that target field's datatype. |

```yaml
- source_column_name: property_json
  target_column_name: Property
  expected_data_type: struct<street:string,zip:string,scores:array<integer>>
  parser:
    type: struct
    input_format: json
    fields:
      - source_field_name: address
        target_field_name: street
        parser:
          type: string
          format: address_us_v1
      - source_field_name: postal_code
        target_field_name: zip
        parser:
          type: string
          format: zip
          on_parse_error: null
      - source_field_name: raw_scores
        target_field_name: scores
        parser:
          type: array
          element_parser: integer
          on_element_error: drop
```

Unknown JSON fields are ignored. A configured field that is missing or null follows its child
parser's `is_nullable` and `default_on_null` settings. A bad field follows that child's
`on_parse_error` policy and produces a path such as `$.zip`. The container shape is checked before
its fields are parsed, so a JSON array or malformed object cannot slip through as an all-null
struct.

### Map

Map parses a JSON object into `map<string,T>`. Keys are Spark strings, while values may use any
supported scalar or nested datatype.

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `input_format` | No | `json` | JSON object input; no inference. |
| `value_parser` | Yes | — | Recursive parser matching `T`. Its immediate error outcome is owned by `on_value_error`. |
| `on_value_error` | No | `fail` | `fail` raises, `null` sends a typed null through final-null handling, `drop` removes the entry, and string-only `preserve` retains its raw value. |
| `drop_null_values` | No | `false` | Remove entries whose source or resolved value is null. |

```yaml
- source_column_name: balances_json
  target_column_name: Balances
  expected_data_type: map<string,decimal(18,2)>
  parser:
    type: map
    value_parser: decimal
    on_value_error: drop
    drop_null_values: false
    audit: true
```

Map error paths include the source key, for example `$['principal']`. Duplicate keys make the whole
JSON map or struct invalid and follow `on_parse_error`, or the parent's child-error policy when
nested. Struct detection covers the complete source object, including fields the configured schema
does not select. Handling duplicates at this boundary avoids Spark's `DUPLICATED_MAP_KEY` failure
and runtime-dependent struct winners.

### Recursive nesting and complex defaults

Complex parsers are recursive. YAML composition depth is capped at 256 syntax nodes, while Spark
datatype nesting is independently capped at 64 complex containers. The larger YAML budget accounts
for the sequence and mapping nodes needed to author one logical nested parser. Both compiler-side
bounds make their respective authoring stage fail deterministically on hostile input. An
`array<struct<...>>`, a `map<string,array<decimal(18,2)>>`, and a struct containing other complex
types all go through the same recursive compiler and runtime. Struct fields follow the order in
`expected_data_type`; arrays keep source order unless values are dropped or deduplicated.

The 64-container bound is a compiler safety ceiling, not a promise that every such plan is cheap on
an untuned Spark session. Higher-order expression resolution is iterative, and the practical depth
depends on container shape, Spark version, audit diagnostics, and
`spark.sql.analyzer.maxIterations`. If the active analyzer budget is exhausted,
`parse_dataframe` raises a metadata-only `SchemaValidationError` that reports the setting and
configured depth before any data action starts. A driver JVM thread-stack exhaustion during the same
metadata-only planning step is translated similarly, with guidance to adjust the startup stack size
or nesting. Intentionally deep schemas can raise the analyzer setting without changing their parser
config, but should be load-tested at their real depth; even a linear 64-level audited plan is
operationally expensive.

Complex defaults use YAML lists for arrays and mappings for structs or maps. A struct default must
contain its full target field set. The compiler checks every nested value and range before the
runtime turns the default into a typed Spark literal. Each authored default is capped at 10,000
expanded nodes, counting containers, elements, scalar or null values, and map keys. Shared YAML
aliases count at every emitted position and cyclic containers are rejected, preventing a compact
authoring graph from expanding into an unsafe Spark literal.

Supported datatypes follow Spark's
[SQL datatype model](https://spark.apache.org/docs/3.5.7/sql-ref-datatypes.html). Complex decoding
uses native [`from_json`](https://spark.apache.org/docs/3.5.7/api/python/reference/pyspark.sql/api/pyspark.sql.functions.from_json.html)
and higher-order Spark expressions, never a Python or pandas UDF.

JSON decoding rejects comments, single-quoted strings, unquoted field names, and numeric literals
with leading zeroes. Put a leading-zero value in quotes when it is valid source data—for example,
`"001"` inside an integer array parses to `1`. A malformed top-level container follows
`on_parse_error`; a malformed nested container follows its field, element, or value policy.

Spark `variant` is not supported because the package still targets Spark 3.5, while Variant starts
in Spark 4.x. Interval types need a defined bronze encoding before they can be added. Spark
`char`/`varchar` are table-schema types rather than general DataFrame expression types. UUIDs,
enums, email addresses, and similar domain values stay strings here; validate those in the
downstream rules engine.

## Parsing order and error modes

Every column follows this order:

1. Collapse whitespace and trim when enabled.
2. Convert the normalized empty string to null when enabled.
3. Match and optionally replace effective null markers.
4. Apply parser-specific conversion or string formatting.
5. Resolve a non-null value that failed conversion.
6. Convert numeric zero to null when `zero_is_valid: false`.
7. Apply `default_on_null` for a non-nullable target.

`on_parse_error` decides what happens at step 5:

- `fail` (default) raises when Spark materializes the bad target expression;
- `null` returns a typed null;
- `default` assigns the required `default_on_error`; and
- `preserve` returns the exact, pre-normalization bronze token and is accepted only for a string
  parser.

For a complex column, those modes handle malformed top-level JSON. Struct fields use their own
`on_parse_error`; arrays use `on_element_error`; maps use `on_value_error`. Child policies support
`fail`, `null`, `drop`, or `preserve` when the child is a string.

A container policy does not cascade through a successfully decoded complex child. For example,
`on_element_error: drop` can drop a malformed struct element, but a valid struct whose field cannot
be parsed still follows that field parser's own `on_parse_error` policy. Set the field policy
explicitly when the whole ingestion path must remain fail-open.

Handled child errors still appear in the top-level audit's `nested_error_paths`, even when the bad
element or entry was dropped or preserved. Nested defaults and zero invalidation have their own
path arrays. A nested `fail` message names the top-level source and target columns, the expected
child type, and the failing path.

Diagnostic paths are JSONPath-like rather than an external JSONPath standard. Unsafe field and map
key text uses JSON escaping inside single-quoted bracket segments, so a key containing `"` is
rendered unambiguously as `$['e\"f']`.

`preserve` is not available for non-string targets because a raw invalid string cannot live in an
integer, date, binary, or complex Spark column. Quarantine routing is also outside this API; it
needs its own `quarantine_df` contract instead of another parse-error mode.

## Compile-time and DataFrame validation

Before Spark starts, compilation checks:

- YAML syntax and duplicate/unsupported keys;
- required IDs, version, columns, and parser types;
- unique non-empty `target_column_name` values;
- supported recursive Spark DDL types, decimal precision/scale, and parser/type compatibility;
- complete struct field coverage, unique source/target nested names, recursive child parsers,
  string map keys, and complex input-format constraints;
- option placement and primitive types;
- conditional `default_on_null`/`default_on_error` rules and typed values;
- global/column null-marker modes; and
- non-empty effective Boolean vocabularies, plus exact and ASCII-only overlap.

When a config is bound to a DataFrame, names follow Spark's active identifier resolver: matching is
exact when `spark.sql.caseSensitive=true` and case-insensitive otherwise. The runtime checks:

- ambiguous configured sources, including source-schema collisions, fail;
- present configured sources must have Spark `string` type or fail;
- missing configured sources fail unless `on_missing_source="warn"` explicitly permits typed
  null/default substitution;
- configured targets and nested struct source/target fields may not collide under the resolver;
- reserved parser output names may not already exist or collide with keys;
- keys must be existing, unique, unambiguous non-empty names under the resolver;
- `column_prefix` must be non-empty; and
- non-ASCII case-insensitive Boolean vocabularies are lowered and checked for overlap with Spark's
  runtime Unicode tables during metadata-only binding, before any data action starts.

A bad value under `on_parse_error: fail` raises only when Spark evaluates that target expression.
An optimizer-pruned action such as `parsed_df.count()` may never touch it. Collect the target
column directly, or perform a full target write, when the job must materialize every parsed value.

## DataFrameParsing outputs

`parse_dataframe()` returns a `DataFrameParsing` wrapper around one shared lazy Spark plan.

| Property/method | Behavior |
| --- | --- |
| `parsed_df` | Only target columns, aliased to `target_column_name` and ordered like the config. |
| `results_df` | Selected row keys followed by nested parser audit/identity metadata. |
| `warnings` | Tuple of recoverable schema warnings produced by explicit policies such as `on_missing_source="warn"`. |
| `key_columns` | Effective ordered result-key names. |
| `result_columns` | Names of the three parser result columns. |
| `persist(storage_level=MEMORY_AND_DISK_DESER)` | Persist the shared plan and return the same object. Useful before materializing both projections. |
| `unpersist(blocking=False)` | Release the shared plan and return the same object. |

Persistence does not change parser rules or successful values, but it changes evaluation scope and
frequency. Databricks serverless compute rejects all DataFrame and SQL cache APIs, including
`persist()` and `unpersist()`; omit these calls there. Both projections remain valid lazy plans but
are separate evaluations when independently materialized. The wrapper deliberately propagates
Spark's platform error rather than pretending that a requested cache operation succeeded. On
supported runtimes, persisting the shared evaluated plan can surface a configured `fail` policy
from any target when the first projection is materialized, because Spark populates the cache for
the complete shared plan.

Row keys belong to `results_df`; they are not copied into `parsed_df` automatically. If a
downstream join needs the same key in both outputs, configure it as a target column too (the
Databricks system tests do this with `RecordId`) or add it before the handoff.

`parse_dataframe()` parameters:

| Argument | Required | Default | Behavior |
| --- | ---: | --- | --- |
| `df` | Yes | — | Bronze Spark DataFrame. Configured present sources must be top-level strings. |
| `config` | Yes | — | With the service API: compiled config, YAML text/path, or mapping. The lower-level runtime requires `ParserConfig`. |
| `key_columns` | Yes | — | Explicit ordered non-empty row identity used by `results_df`; declaring keys starts no action. |
| `on_missing_source` | No | `fail` | `fail` rejects a missing configured source during binding; `warn` emits `SchemaWarning` and substitutes a typed null/default. |
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
| `target_column_name` | String | Configured target output. |
| `parser_type` | String | Canonical parser applied. |
| `expected_data_type` | String | Canonical target Spark datatype. |
| `original_value` | Nullable string | Unmodified bronze value; null for a missing source. |
| `parsed_value` | Nullable string | Final scalar rendered as text; complex values rendered as canonical JSON; binary rendered as canonical base64. |
| `changed` | Boolean | Whether a material null/default/error/missing-source/ZIP action occurred, including nested defaults and zero invalidation. |
| `effective` | Boolean | False for a missing source; true when the configured source was available. |
| `actions_applied` | Array of strings | Ordered material actions applied to this row and column. |
| `options` | Map of string to string | Every fully resolved effective option for this parser. |
| `error` | Nullable string | Handled parse or missing-source description; fail mode raises for bad values. |
| `nested_error_paths` | Array of strings | JSONPath-like locations of handled child failures; empty for none. |
| `nested_default_on_null_paths` | Array of strings | Locations where a nested non-nullable parser supplied `default_on_null`; empty for none. |
| `nested_zero_invalidated_paths` | Array of strings | Locations where a nested numeric zero was invalidated; empty for none. |

Possible actions are `source_column_missing`, `empty_string_to_null`, `null_marker_replaced`,
`parse_error_to_null`, `parse_error_default_applied`, `parse_error_preserved`, `zero_invalidated`,
`default_on_null_applied`, `json_null_to_null`, `nested_parse_errors_resolved`,
`nested_zero_invalidated`, `nested_default_on_null_applied`, `zip_padded`, and
`zip_plus4_formatted`.

Normal whitespace cleanup, successful type conversion, and routine string formatting are visible
by comparing `original_value` with `parsed_value` and reviewing `options`. They do not add action
entries or set `changed` on their own.

## Defaults and exhaustive YAML reference

All defaults come from [`spark_parser.defaults`](src/spark_parser/defaults.py). `PARSER_DEFAULTS`
is a deeply immutable mapping whose sequence values are tuples. `parser.defaults()` returns the
detached JSON-shaped dictionaries and lists to edit or pass directly to `json.dumps`. Compilation
fills in omitted and inherited values, and both
serialization and review reports show the result.

[`examples/all_parsers.yaml`](examples/all_parsers.yaml) shows every top-level, global, column,
common, and parser-specific argument with required/default indicators.

Lower-level classes remain public for targeted use:

```python
from spark_parser import (
    ParserConfigSerializer,
    SparkDataFrameParser,
    YamlParserConfigCompiler,
)
```

These lower-level classes are useful for focused tooling and tests. Most application code should
use the `parser` service shown above.
