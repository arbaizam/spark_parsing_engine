# Spark Parser System Test Summary

Source: `tests/system/spark_parser_system_tests.py`.

The Databricks notebook covers behavior that requires a real Spark session. It imports the package
directly from the repository checkout and does not build, install, hash, publish, or verify a wheel.
Compiler-only permutations remain in the pytest suite. The notebook contains nine focused system
tests and does not run pytest itself.

The notebook exercises the 0.5.0 scalar-only boundary. Complex source data must be decoded or
flattened into top-level strings before parsing and may be reconstructed downstream.

## Test Inventory

| Test ID | Boundary | What we prove |
| --- | --- | --- |
| ST-001 | Repository configuration and compiler | The notebook proves it imported `spark_parser` from the current repository checkout, rejects Spark older than 3.5, compiles the complete parser reference and representative system configuration, represents every public parser type, and hashes resolved configuration deterministically. |
| ST-002 | Native scalar Spark expressions | String formats, state normalization, decimal rounding, integers, dates, timestamps, and timestamp-without-timezone values materialize with their expected Databricks values and exact Spark types. Date/time expectations are rendered by Spark before collection so Python's process timezone cannot change the assertion. |
| ST-003 | Domain display profiles | Bounded `title_business_v1` rules, approved interest-rate canonicalization, and `property_type_v1` aliases materialize correctly. Property coverage includes corrected `Condominium`, `Four-Unit`, retained `LIHTC`, mixed-use restructuring with retained descriptors, action-free canonical changes, and explicit preservation of unknown non-mixed values. |
| ST-004 | Error policies and lazy execution | Null, default, and string-only preserve policies produce their configured scalar values and actions, while `fail` raises a supported Spark exception containing the exact source, target, and datatype context only when the parsed value is materialized. |
| ST-005 | Spark SQL mode | Fully materialized handled target and audit outputs are identical with ANSI mode enabled and disabled. |
| ST-006 | DataFrame output contract | Scalar target order, target-mapped result keys, prefixed result columns, parser identity, content hash, engine version, and the fixed scalar audit fields match the public contract. |
| ST-007 | Explicitly recoverable input-schema drift | With `on_missing_source="warn"`, a missing configured source produces an explicitly asserted nullable Spark `string` field containing null, plus both a DataFrame warning and row-level audit evidence. |
| ST-008 | Input-schema safety | Non-string configured sources, reserved result-column collisions, and omitted explicit row keys fail before parser expressions are constructed. |
| ST-009 | Error collection and configured-mode compatibility | Both projections fully materialize multiple ordered conversion errors, including a column with auditing disabled. Required null defaults, explicit null/default/preserve policies, original values, string-format diagnostics, and audit actions match the resulting values. Valid and null-input rows have empty error arrays. A custom prefix, mapped keys, diagnostic schema, execution metadata, and unchanged configuration hash retain their contracts, while default configured mode still raises when an invalid target is explicitly materialized. |

## Execution Contract

Run the notebook from a Databricks Git checkout containing both `pyproject.toml` and
`src/spark_parser`. The setup cell locates the repository root, puts `<repository>/src` first on
`sys.path`, clears any `spark_parser` modules cached by an earlier notebook run, and ST-001 verifies
the imported package file is under that exact source directory.

Requirements:

- a Databricks runtime providing Spark 3.5 or newer;
- permission to attach the notebook to compute; and
- the complete repository checkout, including `examples/all_parsers.yaml`.

No widgets or other parameters are required. The notebook creates fixed in-memory source rows,
writes no files or tables, and does not modify the workspace. Each test that changes Spark SQL
settings applies its own temporary values and restores the originals in a `finally` path, including
when that test fails.

## Execution

Open `tests/system/spark_parser_system_tests.py` in the Databricks Git folder, attach compute, and
select **Run all**.

Every test records its ID in an ordered pass registry. The run succeeds only when all nine IDs
have been recorded exactly once, in order, and execution reaches:

```text
PASS: All 9 current-contract Spark Parser system tests completed.
```

Any failed assertion or unexpected exception stops the notebook at the responsible test ID.
