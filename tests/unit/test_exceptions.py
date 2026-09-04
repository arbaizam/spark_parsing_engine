"""Public exception construction and serialization contracts without a Spark runtime."""

import pickle

import pytest

from spark_parser import CompilationError, SchemaValidationError, SparkParserError

EXCEPTION_TYPES = (CompilationError, SchemaValidationError)


@pytest.mark.parametrize("exception_type", EXCEPTION_TYPES)
def test_exception_without_arguments(exception_type: type[SparkParserError]) -> None:
    error = exception_type()

    assert error.args == ()
    assert str(error) == ""
    assert error.errors == ()


@pytest.mark.parametrize("exception_type", EXCEPTION_TYPES)
@pytest.mark.parametrize(
    ("messages", "expected_errors", "expected_message"),
    [
        ("first error", ("first error",), "first error"),
        (("first error",), ("first error",), "first error"),
        (
            ["first error", "second error"],
            ("first error", "second error"),
            "Validation failed with 2 errors:\n- first error\n- second error",
        ),
        (
            ("first error", "second error"),
            ("first error", "second error"),
            "Validation failed with 2 errors:\n- first error\n- second error",
        ),
    ],
)
def test_validation_messages(
    exception_type: type[SparkParserError],
    messages: object,
    expected_errors: tuple[str, ...],
    expected_message: str,
) -> None:
    error = exception_type(messages)

    assert error.errors == expected_errors
    assert error.args == (expected_message,)
    assert str(error) == expected_message


@pytest.mark.parametrize("exception_type", EXCEPTION_TYPES)
@pytest.mark.parametrize(
    "arguments",
    [("",), (None,), ("first", 42), (["first", 42],), ([],), (["first", ""],)],
)
def test_legacy_constructor_preserves_exception_behavior(
    exception_type: type[SparkParserError], arguments: tuple[object, ...]
) -> None:
    error = exception_type(*arguments)
    ordinary_error = Exception(*arguments)

    assert error.args == ordinary_error.args
    assert str(error) == str(ordinary_error)
    assert error.errors == ((str(ordinary_error),) if str(ordinary_error) else ())


@pytest.mark.parametrize("exception_type", EXCEPTION_TYPES)
@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_pickle_preserves_aggregate_errors_and_legacy_arguments(
    exception_type: type[SparkParserError], protocol: int
) -> None:
    for arguments in (
        (["first error", "second error"],),
        (["single error"],),
        (),
        ("first", 42),
        (["first", 42],),
        ("",),
    ):
        original = exception_type(*arguments)

        restored = pickle.loads(pickle.dumps(original, protocol=protocol))

        assert type(restored) is exception_type
        assert restored.args == original.args
        assert str(restored) == str(original)
        assert restored.errors == original.errors
