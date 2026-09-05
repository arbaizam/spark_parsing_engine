"""Shared, PySpark-free token grammar for encoded binary values."""

# Keep the body unanchored: Python uses fullmatch, while Spark uses Java's \A and \z anchors.
# Python's Base64 decoder has changed its padding checks across supported interpreter versions;
# validating this grammar explicitly keeps compiled defaults and source tokens on one contract.
BASE64_TOKEN_PATTERN = r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?"
