# Spark Parser Unit and Integration Test Summary

Sources: `tests/unit/` and `tests/integration/`. Shared test data lives under `tests/fixtures/`.

The suite contains **62 explicit test functions** and collects **85 pytest cases** after parameter
expansion. Of those cases, **56** are Spark-independent compiler, serializer, and service unit
tests. The remaining **29** are Spark integration tests; 28 materialize a real Spark session and
one verifies serialized timestamp options without starting Spark.

The Databricks system-test notebook is separate from these counts and is documented in
`spark_parser_system_test_summary.md`.

## Coverage Matrix

| Area | Explicit Tests | Collected Cases | Current Contract Covered |
| --- | ---: | ---: | --- |
| YAML compilation and datatype grammar | 24 | 47 | Strict YAML shapes and scalar types, recursive Spark DDL, all scalar and complex parser contracts, typed defaults, Boolean vocabulary inheritance, duplicate-key rejection, source fan-out, target uniqueness, metadata normalization, and canonical parser/type aliases. |
| Serialization | 2 | 2 | Deterministic mappings, canonical JSON, semantic/order-sensitive content hashing, caller detachment, and recompilation of resolved configuration. |
| Service and configuration review | 7 | 7 | Discoverable parser/config metadata, public error behavior, valid and invalid review reports, inert-null-marker warnings, compiler/metadata invariants, datetime-guard invariants, resolved options, Markdown/JSON artifacts, paths, and evidence-based validation results. |
| Native Spark runtime | 29 | 29 | Scalar and recursive complex parsing, territory and multi-property string formats, strict ZIP shapes, signed byte boundaries, date/time fallbacks and custom-format policy, ANSI parity, all error policies, strict JSON, nested audit paths, fail-closed schema guards, wide configurations, custom output prefixes, and lazy fail-mode materialization. |
| **Total** | **62** | **85** | |

The previous Databricks notebook/config layout checks were removed with the release-test workflow.
Notebook packaging and workspace layout are not unit behavior; the replacement Databricks system
notebook asserts its own current runtime contract.

## Execution

Run the Spark-independent unit suite locally or from a Databricks Git-folder notebook shell cell:

```bash
python -m pytest tests/unit -q
```

From a Databricks notebook whose current directory is not the repository root:

```bash
%sh
cd /Workspace/path/to/spark_parser
python -m pytest tests/unit -q
```

Run the Spark integration suite in local development or CI with compatible Spark and Java
installed:

```bash
SPARK_PARSER_REQUIRE_JAVA=1 python -m pytest tests/integration -q
```

On PowerShell:

```powershell
$env:SPARK_PARSER_REQUIRE_JAVA = "1"
python -m pytest tests/integration -q
```

`SPARK_PARSER_REQUIRE_JAVA=1` converts a missing Java runtime from a skip into a failure so a full
runtime test cannot report success without executing its Spark cases.

Run both pytest tiers together with:

```bash
python -m pytest tests/unit tests/integration -q
```
