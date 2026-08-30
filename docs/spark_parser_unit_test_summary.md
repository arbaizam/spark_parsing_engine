# Spark Parser Unit and Integration Test Summary

Sources: `tests/unit/` and `tests/integration/`. Shared test data lives under `tests/fixtures/`.

The suite contains **99 explicit test functions** and collects **144 pytest cases** after parameter
expansion. Of those cases, **106** are Spark-independent compiler, serializer, and service unit
tests. The remaining **38** are Spark integration tests; 37 materialize a real Spark session and
one verifies serialized timestamp options without starting Spark.

The Databricks system-test notebook is separate from these counts and is documented in
`spark_parser_system_test_summary.md`.

## Coverage Matrix

| Area | Explicit Tests | Collected Cases | Current Contract Covered |
| --- | ---: | ---: | --- |
| YAML compilation and datatype grammar | 50 | 95 | Strict YAML shapes and scalar types, source-located duplicate-key errors, well-formed Unicode, bounded YAML/DDL recursion, complex-default cycle and expansion limits, recursive Spark DDL, all scalar and complex parser contracts, canonical typed defaults, Boolean vocabulary inheritance, source fan-out, target uniqueness, metadata normalization, and canonical parser/type aliases. |
| Serialization | 2 | 2 | Deterministic mappings, canonical JSON, semantic/order-sensitive content hashing, caller detachment, and recompilation of resolved configuration. |
| Service and configuration review | 9 | 9 | Discoverable parser/config metadata, immutable process-wide defaults and detached copies, public error behavior, valid and invalid review reports, inert-null-marker warnings, compiler/metadata invariants, resolved options, injection-safe and round-trip-safe Markdown/YAML/JSON artifacts, paths, and evidence-based validation results. |
| Native Spark runtime | 38 | 38 | Scalar and recursive complex parsing, full Unicode whitespace normalization, strict numeric/JSON/Base64 tokens, deterministic map output and nested audit paths, recursive default/null/zero behavior, Spark-exact identifier resolution, literal hostile column names, date/time defaults and custom-format policy, ANSI parity, fail-closed schema guards, wide configurations, custom output prefixes, and lazy fail-mode materialization. |
| **Total** | **99** | **144** | |

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
SPARK_PARSER_REQUIRE_JAVA=1 python -m pytest tests/unit tests/integration -q
```

On PowerShell:

```powershell
$env:SPARK_PARSER_REQUIRE_JAVA = "1"
python -m pytest tests/unit tests/integration -q
```
