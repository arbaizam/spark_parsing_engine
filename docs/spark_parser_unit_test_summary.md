# Spark Parser Unit and Integration Test Summary

Sources: `tests/unit/` and `tests/integration/`. Shared test data lives under `tests/fixtures/`.

The 0.5.0 release suite covers the scalar-only package contract and collects **302 pytest cases**:
**189 Spark-independent unit cases** and **113 Spark integration cases**. The Databricks system
notebook is separate from those counts and is documented in
`spark_parser_system_test_summary.md`.

## Coverage matrix

| Area | Current contract covered |
| --- | --- |
| YAML compilation and datatype grammar | Strict YAML shapes, source-located duplicate-key errors, well-formed Unicode, bounded YAML composition, dependency-aware aggregate findings with stable field paths, all scalar parser contracts, scalar aliases, `decimal(p,s)` bounds, typed defaults, cross-Python timestamp and Base64 default validation, Boolean vocabulary inheritance, source fan-out, target uniqueness, and clear rejection of array/struct/map datatypes and parser names. |
| Serialization | Deterministic resolved mappings, canonical JSON, semantic/order-sensitive content hashing, caller detachment, and recompilation of scalar configurations. |
| Public exceptions | Singleton and aggregate validation messages, ordinary Exception constructor compatibility, and preservation of messages, arguments, and individual errors across pickle round trips. |
| Service and configuration review | Discoverable scalar parser/config metadata, immutable defaults with detached JSON copies, compilation/serialization facades, public errors, review reports, type-driven YAML text/`Path`/mapping dispatch, deferred Unicode checks, warnings, Markdown/YAML/JSON safety, and scalar source-to-target schema reporting. |
| Native Spark runtime | Every scalar datatype, string display profiles including `title_business_v1`, `interest_rate_index_v1`, and `property_type_v1`, value-aware string/ZIP audit flags, target-mapped and pass-through result keys, Unicode normalization, strict numeric/Base64/datetime tokens, floating-width underflow rejection, Boolean overlap checks, defaults and error policies, aggregate metadata-only schema findings with guarded dependencies, Spark identifier resolution, public facade projections, cache delegation, date/timezone stability, ANSI parity, schema guards, wide configurations, output prefixes, and lazy fail-mode materialization. Regression cases cover empty state-list components, punctuation-only address tokens, balanced mixed-use wrappers and repeated formatting, explicit timestamp target types, and invalid historical dates under legacy time-parser policies. |
| Error collection and result identity | Every failed conversion appears in both projections regardless of auditing, with stable schemas/order, original values, format-aware messages, final resolutions, and unchanged configuration identity. Coverage includes valid/empty/null/missing inputs, mixed policies, failed parsed keys, persistence, generated-name collisions, and rejection of pass-through keys that conflict with parsed targets under the active resolver. |

Configured arrays, structs, maps, recursive defaults, child-error policies, and nested audit paths are
not part of the 0.5.0 contract. Complex source data is decoded or flattened upstream and complex
target data may be reconstructed downstream.

## Execution

Refresh the counts above after adding or changing parametrized cases:

```bash
python -m pytest tests/unit --collect-only -q
python -m pytest tests/integration --collect-only -q
```

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

Without PySpark installed, full-suite collection skips the integration module and still runs the
unit suite. CI exercises this optional-dependency boundary with `python -m pytest -q`. It also
installs the built wheel into a clean environment outside the checkout and verifies compilation,
metadata, reports, serialization, and hashing without PySpark.

Portable integration coverage avoids live cache APIs because Databricks serverless rejects them.
Cache delegation is verified with an in-memory test double; tests marked `classic_spark` exercise
the additional contracts only on a classic local/CI Spark session.

Run both tiers together with:

```bash
SPARK_PARSER_REQUIRE_JAVA=1 python -m pytest tests/unit tests/integration -q
```
