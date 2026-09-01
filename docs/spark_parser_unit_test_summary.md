# Spark Parser Unit and Integration Test Summary

Sources: `tests/unit/` and `tests/integration/`. Shared test data lives under `tests/fixtures/`.

The 0.5.0 release suite covers the scalar-only package contract and collects **148 pytest cases**:
**109 Spark-independent unit cases** and **39 Spark integration cases**. The Databricks system
notebook is separate from those counts and is documented in
`spark_parser_system_test_summary.md`.

## Coverage matrix

| Area | Current contract covered |
| --- | --- |
| YAML compilation and datatype grammar | Strict YAML shapes, source-located duplicate-key errors, well-formed Unicode, bounded YAML composition, all scalar parser contracts, scalar aliases, `decimal(p,s)` bounds, typed defaults, cross-Python timestamp defaults, Boolean vocabulary inheritance, source fan-out, target uniqueness, and clear rejection of array/struct/map datatypes and parser names. |
| Serialization | Deterministic resolved mappings, canonical JSON, semantic/order-sensitive content hashing, caller detachment, and recompilation of scalar configurations. |
| Service and configuration review | Discoverable scalar parser/config metadata, immutable defaults with detached JSON copies, compilation/serialization facades, public errors, review reports, type-driven YAML text/`Path`/mapping dispatch, deferred Unicode checks, warnings, Markdown/YAML/JSON safety, and scalar source-to-target schema reporting. |
| Native Spark runtime | Every scalar datatype, string display profiles including `title_business_v1` and `interest_rate_index_v1`, Unicode normalization, strict numeric/Base64/datetime tokens, Boolean overlap checks, defaults and error policies, scalar audit records, Spark identifier resolution, public facade projections, cache delegation, date/timezone stability, ANSI parity, schema guards, wide configurations, output prefixes, and lazy fail-mode materialization. |

Configured arrays, structs, maps, recursive defaults, child-error policies, and nested audit paths are
not part of the 0.5.0 contract. Complex source data is decoded or flattened upstream and complex
target data may be reconstructed downstream.

## Execution

Run the Spark-independent unit suite:

```bash
python -m pytest tests/unit -q
```

Run the Spark integration suite with compatible Spark and Java installed:

```bash
SPARK_PARSER_REQUIRE_JAVA=1 python -m pytest tests/integration -q
```

On PowerShell:

```powershell
$env:SPARK_PARSER_REQUIRE_JAVA = "1"
python -m pytest tests/integration -q
```

`SPARK_PARSER_REQUIRE_JAVA=1` turns a missing PySpark or Java runtime into a failure so runtime
validation cannot report success without executing its Spark cases.

Portable integration coverage avoids live cache APIs because Databricks serverless rejects them.
Cache delegation is verified with an in-memory test double; tests marked `classic_spark` exercise
the additional contracts only on a classic local/CI Spark session.

Run both tiers together with:

```bash
SPARK_PARSER_REQUIRE_JAVA=1 python -m pytest tests/unit tests/integration -q
```
