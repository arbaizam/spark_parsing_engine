"""Deterministic native-Spark formatting profiles for US location strings.

Every helper returns a Spark :class:`~pyspark.sql.Column`; no value is collected to the driver and
no Python UDF crosses the JVM boundary. The dictionaries below are intentionally explicit so a
maintainer can review each business rule and understand exactly which transformations are allowed.
"""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

# Directionals are recognized case-insensitively after punctuation cleanup.
_DIRECTIONALS = {
    "n": "N",
    "north": "N",
    "s": "S",
    "south": "S",
    "e": "E",
    "east": "E",
    "w": "W",
    "west": "W",
    "ne": "NE",
    "northeast": "NE",
    "nw": "NW",
    "northwest": "NW",
    "se": "SE",
    "southeast": "SE",
    "sw": "SW",
    "southwest": "SW",
}

# Common Publication 28 suffix names and aliases. Changes to this table alter canonical silver
# output, so additions should be accompanied by focused runtime examples and UAT review.
_SUFFIXES = {
    "alley": "Aly",
    "aly": "Aly",
    "avenue": "Ave",
    "av": "Ave",
    "ave": "Ave",
    "boulevard": "Blvd",
    "blvd": "Blvd",
    "center": "Ctr",
    "centre": "Ctr",
    "ctr": "Ctr",
    "circle": "Cir",
    "cir": "Cir",
    "court": "Ct",
    "ct": "Ct",
    "cove": "Cv",
    "cv": "Cv",
    "crossing": "Xing",
    "xing": "Xing",
    "drive": "Dr",
    "dr": "Dr",
    "expressway": "Expy",
    "expy": "Expy",
    "freeway": "Fwy",
    "fwy": "Fwy",
    "gardens": "Gdns",
    "gdns": "Gdns",
    "highway": "Hwy",
    "hwy": "Hwy",
    "junction": "Jct",
    "jct": "Jct",
    "lane": "Ln",
    "ln": "Ln",
    "loop": "Loop",
    "park": "Park",
    "parkway": "Pkwy",
    "pkwy": "Pkwy",
    "place": "Pl",
    "pl": "Pl",
    "plaza": "Plz",
    "plz": "Plz",
    "road": "Rd",
    "rd": "Rd",
    "route": "Rte",
    "rte": "Rte",
    "square": "Sq",
    "sq": "Sq",
    "street": "St",
    "str": "St",
    "st": "St",
    "terrace": "Ter",
    "ter": "Ter",
    "trail": "Trl",
    "trl": "Trl",
    "turnpike": "Tpke",
    "tpke": "Tpke",
    "way": "Way",
}

# Secondary-unit designators are also used to recognize the following alphanumeric unit value.
_UNIT_DESIGNATORS = {
    "apartment": "Apt",
    "apt": "Apt",
    "building": "Bldg",
    "bldg": "Bldg",
    "box": "Box",
    "department": "Dept",
    "dept": "Dept",
    "floor": "Fl",
    "fl": "Fl",
    "hangar": "Hngr",
    "hngr": "Hngr",
    "lot": "Lot",
    "po": "PO",
    "room": "Rm",
    "rm": "Rm",
    "rr": "RR",
    "suite": "Ste",
    "ste": "Ste",
    "trailer": "Trlr",
    "trlr": "Trlr",
    "unit": "Unit",
    "us": "US",
}

# ``initcap`` cannot infer internal capitalization. Keep the small, intentional exception list
# visible rather than embedding opaque regular-expression replacements.
_NAME_EXCEPTIONS = {
    "dekalb": "DeKalb",
    "delacruz": "DeLaCruz",
    "desoto": "DeSoto",
    "dupage": "DuPage",
    "lasalle": "LaSalle",
    "mchenry": "McHenry",
    "mclean": "McLean",
}

# Canonical USPS abbreviations for the 50 states. Washington, DC is included because it appears in
# the same address field in real US source data, even though it is not a state. Territories are
# intentionally excluded: adding them should be an explicit domain decision rather than an
# accidental expansion of a field documented as a state code.
_US_STATE_NAMES = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "district of columbia": "DC",
    "washington dc": "DC",
    "washington d.c.": "DC",
}

# Full names and already-abbreviated inputs share one lookup. Lowercase keys match the normalized
# comparison expression; values always use the canonical uppercase two-letter representation.
_US_STATE_TOKENS = {
    **_US_STATE_NAMES,
    **{abbreviation.lower(): abbreviation for abbreviation in _US_STATE_NAMES.values()},
}


def _map_lookup(mapping: dict[str, str], key: Column) -> Column:
    """Build a native Spark map lookup from a small code-owned Python dictionary."""
    pairs: list[Column] = []
    for source, target in mapping.items():
        pairs.extend((F.lit(source), F.lit(target)))
    # ``element_at`` returns null when a token is not present. Later coalesce/when clauses use that
    # null to continue through the formatting precedence rules.
    return F.element_at(F.create_map(*pairs), key)


def _clean_token(token: Column) -> Column:
    """Lowercase one token and remove punctuation ignored by these display profiles."""
    return F.regexp_replace(F.lower(token), r"[,.]", "")


def _smart_token(
    token: Column,
    *,
    address: bool,
    suffix_allowed: Column | None = None,
    previous_token: Column | None = None,
) -> Column:
    """Format one token according to explicit, ordered address/name rules.

    Rule order matters: named exceptions win first, then address vocabulary, then Mc casing,
    ordinals, unit identifiers, and finally general apostrophe/hyphen-aware title casing.
    """
    cleaned = _clean_token(token)
    exception = _map_lookup(_NAME_EXCEPTIONS, cleaned)
    mapped = F.lit(None).cast("string")
    if address:
        suffix = _map_lookup(_SUFFIXES, cleaned)
        if suffix_allowed is not None:
            # Street-like words can occur inside a proper name. Only the last suffix-looking token
            # is treated as the street suffix, preventing ``Center Street`` from becoming two
            # abbreviations.
            suffix = F.when(suffix_allowed, suffix)
        mapped = F.coalesce(
            _map_lookup(_DIRECTIONALS, cleaned),
            suffix,
            _map_lookup(_UNIT_DESIGNATORS, cleaned),
        )
    mc_cased = F.concat(
        F.lit("Mc"),
        F.upper(F.substring(cleaned, 3, 1)),
        F.substring(cleaned, 4, 1000),
    )
    title_cased = F.concat_ws(
        "-",
        # Split twice so both ``Smith-Jones`` and ``O'Brien`` retain their separators while each
        # component receives predictable title casing.
        F.transform(
            F.split(cleaned, "-"),
            lambda hyphen_part: F.concat_ws(
                "'",
                F.transform(F.split(hyphen_part, "'"), lambda part: F.initcap(part)),
            ),
        ),
    )
    previous_cleaned = _clean_token(previous_token) if previous_token is not None else F.lit("")
    alphanumeric = cleaned.rlike(r"^(?=.*\d)(?=.*[a-z])[a-z0-9-]+$")
    unit_value = previous_cleaned.isin(*_UNIT_DESIGNATORS)
    hash_unit_value = cleaned.rlike(r"^#(?=.*\d)(?=.*[a-z])[a-z0-9-]+$")
    # The chained conditions below are a precedence table. Reordering them is a behavioral change.
    return (
        F.when(exception.isNotNull(), exception)
        .when(mapped.isNotNull(), mapped)
        .when(cleaned.rlike(r"^mc[a-z].*"), mc_cased)
        .when(cleaned.rlike(r"^(?:\d+(?:st|nd|rd|th))$"), cleaned)
        .when(hash_unit_value, F.concat(F.lit("#"), F.upper(F.substring(cleaned, 2, 1000))))
        .when(unit_value & alphanumeric, F.upper(cleaned))
        .otherwise(title_cased)
    )


def format_address_us_v1(value: Column) -> Column:
    """Return a USPS-oriented display representation using native Spark expressions.

    The input is expected to have already passed common whitespace normalization. Nulls remain
    null, and empty punctuation-only tokens are removed before joining the final value.
    """
    tokens = F.split(value, " ")
    # ``array_position(reverse(...))`` identifies the final suffix candidate without collecting
    # the token array. Spark array positions are one-based, while transform indices are zero-based.
    suffix_flags = F.transform(tokens, lambda token: _clean_token(token).isin(*_SUFFIXES))
    last_suffix_index = F.size(tokens) - F.array_position(F.reverse(suffix_flags), True)
    # Prepending an empty token lets every transformed token read its predecessor safely.
    padded_tokens = F.concat(F.array(F.lit("")), tokens)
    formatted_tokens = F.transform(
        tokens,
        lambda token, index: _smart_token(
            token,
            address=True,
            suffix_allowed=index == last_suffix_index,
            previous_token=F.element_at(padded_tokens, index + F.lit(1)),
        ),
    )
    formatted = F.concat_ws(" ", F.filter(formatted_tokens, lambda token: token != ""))
    return F.when(value.isNotNull(), formatted).otherwise(F.lit(None).cast("string"))


def format_county(value: Column) -> Column:
    """Return a smart-cased county name ending in exactly one ``County``.

    A value containing only the suffix has no meaningful county name and therefore returns null;
    the caller's configured parse-error policy decides whether that null fails, remains null, or
    becomes a default.
    """
    # Remove at most the semantic trailing suffix before rebuilding one canonical suffix.
    core = F.trim(F.regexp_replace(value, r"(?i)(?:^|\s+)county\.?$", ""))
    formatted = F.concat_ws(
        " ",
        F.filter(
            F.transform(F.split(core, " "), lambda token: _smart_token(token, address=False)),
            lambda token: token != "",
        ),
    )
    return F.when(
        (core != "") & (formatted != ""), F.concat(formatted, F.lit(" County"))
    ).otherwise(F.lit(None).cast("string"))


def format_state_us(value: Column) -> Column:
    """Return a canonical two-letter US state/DC abbreviation.

    ``value`` is expected to have passed common whitespace normalization. Matching is
    case-insensitive for full names and abbreviations. Unknown non-null values return null so the
    ordinary string parser error policy can fail, preserve null, or assign a configured default.
    """
    return _map_lookup(_US_STATE_TOKENS, F.lower(value))


def format_zip(value: Column) -> Column:
    """Return ZIP5 or ZIP+4, padding short components with leading zeroes.

    The return type is string so leading zeroes survive. Malformed input becomes null and is later
    resolved by the configured parse-error policy.
    """
    compact = F.regexp_replace(value, r"\s+", "")
    # Classify first, then construct output only from a matching shape. This avoids permissive
    # substring extraction accidentally accepting letters or multiple hyphens.
    hyphenated = compact.rlike(r"^\d{1,5}-\d{1,4}$")
    digits_five_or_less = compact.rlike(r"^\d{1,5}$")
    digits_plus_four = compact.rlike(r"^\d{6,9}$")
    base = F.substring_index(compact, "-", 1)
    extension = F.substring_index(compact, "-", -1)
    digits_base = F.regexp_extract(compact, r"^(\d{1,5})(\d{4})$", 1)
    digits_extension = F.regexp_extract(compact, r"^(\d{1,5})(\d{4})$", 2)
    return (
        F.when(
            hyphenated,
            F.concat(F.lpad(base, 5, "0"), F.lit("-"), F.lpad(extension, 4, "0")),
        )
        .when(digits_five_or_less, F.lpad(compact, 5, "0"))
        .when(
            digits_plus_four,
            F.concat(
                F.lpad(digits_base, 5, "0"),
                F.lit("-"),
                digits_extension,
            ),
        )
        .otherwise(F.lit(None).cast("string"))
    )
