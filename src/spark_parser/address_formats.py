"""Deterministic native-Spark formatting profiles for US location strings."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

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

# Common Publication 28 suffix names and aliases. The versioned profile can be
# expanded without silently changing previously versioned formatting behavior.
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

_NAME_EXCEPTIONS = {
    "dekalb": "DeKalb",
    "delacruz": "DeLaCruz",
    "desoto": "DeSoto",
    "dupage": "DuPage",
    "lasalle": "LaSalle",
    "mchenry": "McHenry",
    "mclean": "McLean",
}


def _map_lookup(mapping: dict[str, str], key: Column) -> Column:
    pairs: list[Column] = []
    for source, target in mapping.items():
        pairs.extend((F.lit(source), F.lit(target)))
    return F.element_at(F.create_map(*pairs), key)


def _clean_token(token: Column) -> Column:
    return F.regexp_replace(F.lower(token), r"[,.]", "")


def _smart_token(
    token: Column,
    *,
    address: bool,
    suffix_allowed: Column | None = None,
    previous_token: Column | None = None,
) -> Column:
    cleaned = _clean_token(token)
    exception = _map_lookup(_NAME_EXCEPTIONS, cleaned)
    mapped = F.lit(None).cast("string")
    if address:
        suffix = _map_lookup(_SUFFIXES, cleaned)
        if suffix_allowed is not None:
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
    """Return a USPS-oriented display representation without a Python UDF."""
    tokens = F.split(value, " ")
    suffix_flags = F.transform(tokens, lambda token: _clean_token(token).isin(*_SUFFIXES))
    last_suffix_index = F.size(tokens) - F.array_position(F.reverse(suffix_flags), True)
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
    """Return a smart-cased county name ending in exactly one ``County``."""
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


def format_zip(value: Column) -> Column:
    """Return ZIP5 or ZIP+4, padding short components with leading zeroes."""
    compact = F.regexp_replace(value, r"\s+", "")
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
