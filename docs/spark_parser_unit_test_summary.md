# Spark Parser Unit Test Summary

Source: the pytest suite under `tests/`.

The suite contains **56 explicit test functions** and collects **79 pytest cases** after parameter
expansion. Of those cases, **53** are Spark-independent compiler, serializer, and service tests.
The remaining **26** are marked `spark`; 25 materialize a real Spark session and one verifies
serialized timestamp options without starting Spark.

The Databricks system-test notebook is separate from these counts and is documented in
`spark_parser_system_test_summary.md`.

## Coverage Matrix

| Area | Explicit Tests | Collected Cases | Current Contract Covered |
| --- | ---: | ---: | --- |
| YAML compilation and datatype grammar | 24 | 47 | Strict YAML shapes and scalar types, recursive Spark DDL, all scalar and complex parser contracts, typed defaults, Boolean vocabulary inheritance, duplicate-key rejection, source fan-out, target uniqueness, metadata normalization, and canonical parser/type aliases. |
| Serialization | 1 | 1 | Deterministic mappings, canonical JSON, content hashing, caller detachment, and recompilation of resolved configuration. |
| Service and configuration review | 5 | 5 | Discoverable parser/config metadata, public error behavior, valid and invalid review reports, resolved options, Markdown/JSON artifacts, paths, and evidence-based validation results. |
| Native Spark runtime | 26 | 26 | Scalar and recursive complex parsing, string formats, date/time fallbacks, ANSI parity, all error policies, strict JSON, nested audit paths, schema warnings and guards, wide configurations, custom output prefixes, and lazy fail-mode materialization. |
| **Total** | **56** | **79** | |

The previous Databricks notebook/config layout checks were removed with the release-test workflow.
Notebook packaging and workspace layout are not unit behavior; the replacement Databricks system
notebook asserts its own current runtime contract.

## Execution

Run the Spark-independent unit suite locally or from a Databricks Git-folder notebook shell cell:

```bash
python -m pytest tests -q -m "not spark"
```

From a Databricks notebook whose current directory is not the repository root:

```bash
%sh
cd /Workspace/path/to/spark_parser
python -m pytest tests -q -m "not spark"
```

Run the complete pytest suite in local development or CI with compatible Spark and Java installed:

```bash
SPARK_PARSER_REQUIRE_JAVA=1 python -m pytest tests -q
```

On PowerShell:

```powershell
$env:SPARK_PARSER_REQUIRE_JAVA = "1"
python -m pytest tests -q
```

`SPARK_PARSER_REQUIRE_JAVA=1` converts a missing Java runtime from a skip into a failure so a full
runtime test cannot report success without executing its Spark cases.
