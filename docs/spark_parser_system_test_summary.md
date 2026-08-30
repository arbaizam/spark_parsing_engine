# Spark Parser System Test Summary

Source: `tests/system/spark_parser_system_tests.py`.

The Databricks notebook covers behavior that requires a real Spark session. It imports the package
directly from the repository checkout and does not build, install, hash, publish, or verify a wheel.
Compiler-only permutations remain in the pytest suite. The notebook contains eight focused system
tests and does not run pytest itself.

## Test Inventory

| Test ID | Boundary | What we prove |
| --- | --- | --- |
| ST-001 | Repository configuration and compiler | The complete parser reference and representative system configuration compile, every public parser type is represented, and resolved configuration hashing is deterministic. |
| ST-002 | Native scalar Spark expressions | String formats, state normalization, decimal rounding, integers, dates, timestamps, and timestamp-without-timezone values materialize with their expected Databricks values and types. |
| ST-003 | Recursive Spark expressions | Arrays, structs, nested arrays, maps, child defaults, dropped children, invalid-zero handling, and JSONPath-like audit paths materialize correctly. |
| ST-004 | Error policies and lazy execution | Null, default, preserve, and nested child-error policies produce their configured values and actions, while `fail` raises only when the parsed value is materialized. |
| ST-005 | Spark SQL mode | Fully materialized handled target and audit outputs are identical with ANSI mode enabled and disabled. |
| ST-006 | DataFrame output contract | Target order, result keys, prefixed result columns, parser identity, content hash, engine version, and ordered audit fields match the public contract. |
| ST-007 | Recoverable input-schema drift | A missing configured source produces a typed null plus both a DataFrame warning and row-level audit evidence. |
| ST-008 | Input-schema safety | Non-string configured sources, reserved result-column collisions, and ambiguous default row keys fail before parser expressions are constructed. |

## Execution Contract

Run the notebook from a Databricks Git checkout containing both `pyproject.toml` and
`src/spark_parser`. The setup cell locates the repository root and adds `<repository>/src` to
`sys.path`.

Requirements:

- a Databricks runtime providing Spark 3.5 or newer;
- permission to attach the notebook to compute; and
- the complete repository checkout, including `examples/all_parsers.yaml`.

No widgets or other parameters are required. The notebook creates two fixed in-memory source rows,
writes no files or tables, and does not modify the workspace. It temporarily exercises ANSI mode
and restores the original Spark SQL settings after a successful run.

## Execution

Open `tests/system/spark_parser_system_tests.py` in the Databricks Git folder, attach compute, and
select **Run all**.

The run succeeds only when execution reaches:

```text
PASS: All 8 current-contract Spark Parser system tests completed.
```

Any failed assertion or unexpected exception stops the notebook at the responsible test ID.
