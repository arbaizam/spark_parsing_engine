# Spark Parser Unit and Integration Test Summary

Sources: `tests/unit/` and `tests/integration/`. Shared test data lives under `tests/fixtures/`.

The suite contains **120 explicit test functions** and collects **180 pytest cases** after parameter
expansion. Of those cases, **126** are Spark-independent compiler, serializer, and service unit
tests. The remaining **54** are Spark integration tests; 52 materialize a real Spark session and
two verify Spark-facing contracts without starting Spark or invoking platform-optional cache APIs.

The Databricks system-test notebook is separate from these counts and is documented in
`spark_parser_system_test_summary.md`.

## Coverage Matrix

| Area | Explicit Tests | Collected Cases | Current Contract Covered |
| --- | ---: | ---: | --- |
| YAML compilation and datatype grammar | 54 | 114 | Strict YAML shapes and scalar types, source-located duplicate-key errors, well-formed Unicode, bounded YAML/DDL recursion, complex-default cycle and expansion limits, recursive Spark DDL, all scalar and complex parser contracts, cross-Python timestamp defaults, strict decimal text, Boolean vocabulary inheritance, source fan-out, target uniqueness, metadata normalization, and canonical parser/type aliases. |
| Serialization | 2 | 2 | Deterministic mappings, canonical JSON, semantic/order-sensitive content hashing, caller detachment, and recompilation of resolved configuration. |
| Service and configuration review | 10 | 10 | Discoverable parser/config metadata and aliases, deeply immutable public defaults with detached JSON copies, public compilation/serialization facades, public error behavior, mutable review data-transfer objects with detached mappings, type-driven YAML text/`Path`/mapping dispatch, evidence-based deferred Unicode checks, inert-null-marker warnings, compiler/metadata invariants, resolved options, injection-safe and round-trip-safe Markdown/YAML/JSON artifacts, and evidence-based validation results. |
| Native Spark runtime | 54 | 54 | Scalar and recursive complex parsing, 100,000-character JSON fields, linear audited-plan budgets, analyzer-exhaustion reporting, full Unicode whitespace/case normalization and metadata-only Spark-owned Boolean-overlap validation on empty inputs, strict numeric/JSON/Base64/datetime tokens, deterministic map output and nested audit paths, recursive default/null/zero behavior, Spark-exact identifier resolution, public facade projections and cache-adapter delegation, classic shared-cache semantics, date/timezone stability, ANSI parity, fail-closed schema guards, wide configurations, custom output prefixes, and lazy fail-mode materialization. |
| **Total** | **120** | **180** | |

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

`SPARK_PARSER_REQUIRE_JAVA=1` converts a missing PySpark or Java runtime from a skip into a failure
so a full runtime test cannot report success without executing its Spark cases.

Portable integration coverage never invokes `persist()` or `unpersist()` on a live DataFrame.
Databricks serverless compute explicitly rejects those optional cache APIs, so exact delegation is
verified with an in-memory test double. One `classic_spark` test locks supported cache state and
full-plan evaluation semantics; Connect/serverless skips it before touching a cache API.

Five tests carry the `classic_spark` marker because they deliberately require `SparkContext` or
mutable internal SQL settings. Local/CI Spark executes all five. Spark Connect and Databricks
serverless skip only those proofs while continuing to run the portable parser behavior.

Run both pytest tiers together with:

```bash
SPARK_PARSER_REQUIRE_JAVA=1 python -m pytest tests/unit tests/integration -q
```

On PowerShell:

```powershell
$env:SPARK_PARSER_REQUIRE_JAVA = "1"
python -m pytest tests/unit tests/integration -q
```
