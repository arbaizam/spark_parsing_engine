# Databricks UAT workflow

This workflow validates the built `spark-parser` wheel on Databricks without publishing or
deploying it. It is intentionally small: two representative bronze rows exercise the integration
surface while the repository test suite remains the exhaustive behavioral gate.

## Existing coverage reviewed

The repository already has strong local coverage for strict YAML compilation and canonical
serialization (`test_compiler_yaml.py`, `test_serializer.py`), service metadata and configuration
review reports (`test_service.py`), and native Spark execution (`test_spark_runtime.py`). The Spark
suite runs with ANSI mode enabled and covers scalar parsing, all error policies, stable audit
schema, nested array/struct/map combinations, strict JSON, duplicate map keys, deep nested paths,
and lazy fail-mode materialization.

Those tests do not prove that a built wheel is the code imported by Databricks, or that outputs
survive the target workspace's Delta and rules-engine boundary. This notebook is the integration
gate for those remaining concerns; it deliberately does not repeat every parser edge case.

## Before Databricks

From a clean checkout of the release commit:

```text
python -m pytest
python -m build --wheel
```

Record the commit, wheel filename, version, and SHA-256. Place the wheel and
`spark_parser_uat.yaml` in an approved Unity Catalog volume or Workspace Files location. Do not
rename the wheel. The UAT operator needs `USE CATALOG`, `USE SCHEMA`, `CREATE TABLE`, and read
permissions on an existing isolated UAT catalog and schema.

The notebook installs the exact wheel with notebook-scoped `%pip`, then restarts Python. The
install uses `--no-deps` so the wheel cannot replace the Databricks Runtime's Spark libraries.
Use compute that already provides Python 3.10 or newer, PySpark 3.5 or newer, and PyYAML 6 or
newer, which are the package's declared runtime requirements.

## Run the notebook

Import or open `spark_parser_uat.py` as a Databricks source notebook and supply these parameters:

| Parameter | Required | Example | Purpose |
| --- | ---: | --- | --- |
| `wheel_path` | Yes | `/Volumes/uat/libs/wheels/spark_parser-0.4.0-py3-none-any.whl` | Exact wheel to install. `/Volumes` and `/Workspace` paths are accepted. |
| `expected_version` | Yes | `0.4.0` | Version that both package metadata and `spark_parser.__version__` must report. |
| `expected_wheel_sha256` | Recommended | `64 lowercase hex characters` | Fails the run if the staged artifact differs from the approved wheel. |
| `config_path` | Yes | `/Workspace/Shared/spark_parser_uat/spark_parser_uat.yaml` | Absolute path to the supplied UAT config. |
| `target_catalog` | Yes | `uat` | Existing catalog for disposable UAT output. |
| `target_schema` | Yes | `spark_parser` | Existing schema for disposable UAT output. |
| `table_prefix` | No | `spark_parser_uat` | Prefix for both managed Delta tables and temporary handoff views. |
| `run_id` | No | `release_040_001` | Unique suffix. A UTC timestamp is generated when omitted. |

Use a unique `run_id`. Writes use Delta `errorifexists`; the notebook never overwrites or drops a
table and does not create the target schema. It leaves evidence tables in place for review:

- `<catalog>.<schema>.<prefix>_<run_id>_silver`
- `<catalog>.<schema>.<prefix>_<run_id>_audit`

The UAT owner may drop these exact tables after sign-off under the team's normal retention process.

## What a passing run proves

| Gate | Validation |
| --- | --- |
| Wheel | The requested wheel is installed in a clean Python process; package metadata, module version, and optional SHA-256 match. |
| Bronze to silver | Whitespace/case normalization, decimal rounding, dates, null markers, and typed output match expected values. |
| Error policies | `null` and `default` produce asserted values and audit actions; a separate materialized `fail` parse raises as expected. |
| Nested data | Array normalization/deduplication, struct field mapping, nested arrays, map value dropping, and JSONPath-like nested error paths are asserted. |
| ANSI | `spark.sql.ansi.enabled` is set to and asserted as `true` before parsing. |
| Audit | Row keys, parser identity, configuration hash, action names, errors, and nested paths are materialized. |
| Delta | Silver and audit DataFrames are written as Delta, read back, and compared for exact schema and row equality. |
| Rules engine | The Delta-read silver data and flattened parser results pass non-null/unique-key and configuration-hash checks, then become temporary views. |

The final cell exits with a JSON `PASS` summary containing artifact provenance, configuration
identity, Delta table names, and these session-scoped handoff views:

- `<prefix>_<run_id>_rules_input`: typed silver rows, keyed by `RecordId`;
- `<prefix>_<run_id>_parser_results`: one parser result per audited silver column, also keyed by
  `RecordId` and carrying parser config/version/hash metadata.

The repository does not declare a concrete rules-engine dependency or API. Run the project-specific
rules-engine adapter in the same job session against `rules_engine_input_df` and
`rules_engine_parser_results_df`, or have the next job task read the reported silver and audit
Delta tables. This keeps parser UAT independent of one rules-engine implementation while making the
handoff schema explicit and testable.

## Sign-off evidence

Retain the notebook run URL and final JSON summary with the release record. Confirm that:

1. repository tests and wheel build passed at the same commit;
2. `expected_wheel_sha256` was supplied and matched;
3. the notebook exited with `"status": "PASS"`;
4. the silver and audit tables were inspected by the UAT owner; and
5. the downstream rules-engine validation consumed the reported handoff tables or DataFrames.
