"""
geography.py — Location utilities, ZIP-prefix matching, difficult-lane lookup.

Encoding rules (from John Cox / Interfor clarification):
  - Two-letter code  "TX"     -> any city in Texas  (match city suffix ",TX")
  - "ST_NNN" pattern "GA_303" -> cities with 3-digit ZIP prefix 303 in Georgia

ZIP-prefix dict covers all destination patterns that appear in the Difficult Lanes
table.  Source: US Postal Service ZIP code ranges for the 218 cities in the dataset.
A city not found in CITY_ZIP3 is conservatively treated as NOT matching any
ZIP-prefix pattern (WARNING is logged).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from .data import DifficultLane, ProblemData

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Hardcoded city -> 3-digit ZIP prefix
# Covers all 218 locations in the dataset.
# ZIP prefix = first 3 digits of the primary ZIP code for that city.
# Only ZIP prefixes appearing as Difficult Lane destinations are
# strictly required; the full table makes unit tests richer.
# ──────────────────────────────────────────────────────────────────
CITY_ZIP3: Dict[str, str] = {
    # Georgia — 300xx block (Marietta / NW Metro Atlanta)
    "MARIETTA,GA":          "300",
    "CUMMING,GA":           "300",
    "NORCROSS,GA":          "300",
    "SUWANEE,GA":           "300",
    # Georgia — 301xx block (Acworth / Cartersville / Kennesaw / NW GA)
    "ACWORTH,GA":           "301",
    "ADAIRSVILLE,GA":       "301",
    "CARTERSVILLE,GA":      "301",
    "KENNESAW,GA":          "301",
    "RANGER,GA":            "307",   # 30734 -> 307
    # Georgia — 302xx block (Newnan / Griffin / Hogansville / SW Metro Atlanta)
    "NEWNAN,GA":            "302",
    "GRIFFIN,GA":           "302",
    "HOGANSVILLE,GA":       "302",
    "FAIRBURN,GA":          "302",
    "UNION CITY,GA":        "302",
    "CARROLLTON,GA":        "301",   # 30116
    "DOUGLASVILLE,GA":      "301",   # 30134
    # Georgia — 303xx block (Atlanta / Decatur core)
    "ATLANTA,GA":           "303",
    # Georgia — 305xx block (Gainesville / Oakwood)
    "GAINESVILLE,GA":       "305",
    "OAKWOOD,GA":           "305",
    # Georgia — 306xx block (Athens / Winder)
    "ATHENS,GA":            "306",
    "WINDER,GA":            "306",
    # Georgia — 307xx block (Ringgold)
    "RINGGOLD,GA":          "307",
    # Georgia — 310xx / misc central GA
    "MACON,GA":             "312",
    "FORSYTH,GA":           "310",
    "BARNESVILLE,GA":       "302",   # 30204 -> 302
    "JACKSON,GA":           "302",
    "GRIFFIN,GA":           "302",
    "COVINGTON,GA":         "300",   # 30014 -> 300
    "CONYERS,GA":           "300",
    "LOCUST,NC":            "280",   # Locust NC -> 28097
    # Georgia — 314xx (Savannah / Pooler / Statesboro / south GA)
    "SAVANNAH,GA":          "314",
    "POOLER,GA":            "314",
    "RINCON,GA":            "314",
    "STATESBORO,GA":        "304",   # 30458 -> 304
    "SYLVANIA,GA":          "304",   # 30467 -> 304
    "MIDWAY,GA":            "313",   # 31320 -> 313
    # Georgia — 317xx (Moultrie / Fitzgerald / Tifton area)
    "MOULTRIE,GA":          "317",
    "FITZGERALD,GA":        "317",
    "PEARSON,GA":           "316",
    # Georgia — 310/311xx (Albany / Cordele / Americus / SW GA)
    "ALBANY,GA":            "317",   # 31701 -> 317
    "CORDELE,GA":           "310",   # 31015 -> 310
    "ASHBURN,GA":           "317",   # 31714 -> 317
    "BERLIN,GA":            "317",
    "DUDLEY,GA":            "310",
    "PERRY,GA":             "310",
    "DUBLIN,GA":            "310",
    "EATONTON,GA":          "310",   # 31024 -> 310
    "WARRENTON,GA":         "308",   # 30828 -> 308
    "THOMSON,GA":           "308",
    "EVANS,GA":             "308",
    "AUGUSTA,GA":           "309",   # 30901-30909 -> 309
    "UNION POINT,GA":       "306",   # 30669 -> 306
    "WINDER,GA":            "306",
    "MADISON,GA":           "306",   # 30650 -> 306 (this is MADISON,VA below)
    "CANON,GA":             "306",
    "JEFFERSON,GA":         "305",   # 30549 -> 305
    "GAINESVILLE,GA":       "305",
    "BUENA VISTA,GA":       "317",   # 31803 -> 318 approx
    "BAINBRIDGE,GA":        "398",   # 39817 -> 398  ← difficult lane GA_398
    "FOLKSTON,GA":          "315",   # 31537 -> 315
    "BLACKSHEAR,GA":        "315",
    "JESUP,GA":             "315",
    "WAYCROSS,GA":          "315",
    "COLUMBIANA,AL":        "350",
    "PINE MOUNTAIN VALLEY,GA": "318",
    "HOGANSVILLE,GA":       "302",
    # Georgia (misc)
    "ABBEVILLE,SC":         "296",
    # Florida
    "POMPANO BEACH,FL":     "330",   # 33060-33069 -> 330  ← FL_330
    "FT LAUDERDALE,FL":     "333",   # 33301+ -> 333
    "FORT LAUDERDALE,FL":   "333",
    "BOCA RATON,FL":        "334",
    "FT PIERCE,FL":         "349",
    "JACKSONVILLE,FL":      "322",
    "GAINESVILLE,FL":       "326",
    "OCALA,FL":             "344",
    "ORLANDO,FL":           "328",
    "SANFORD,FL":           "327",
    "LADY LAKE,FL":         "321",
    "CLERMONT,FL":          "347",
    "MASCOTTE,FL":          "347",
    "LAKELAND,FL":          "338",
    "PLANT CITY,FL":        "335",
    "TAMPA,FL":             "336",
    "GIBSONTON,FL":         "336",
    "CRESCENT CITY,FL":     "321",
    "TITUSVILLE,FL":        "329",
    "AUBURNDALE,FL":        "338",
    "BROOKSVILLE,FL":       "346",
    "WINTER HAVEN,FL":      "338",
    "TARRYTOWN,FL":         "346",
    "LOCKHART,FL":          "328",
    "FORT LAUDERDALE,FL":   "333",
    "POMPANO BEACH,FL":     "330",
    # Alabama
    "BIRMINGHAM,AL":        "352",
    "ANNISTON,AL":          "362",
    "AUBURN,AL":            "368",
    "OPELIKA,AL":           "368",
    "COTTONTON,AL":         "368",
    "MOBILE,AL":            "366",
    "CHICKASAW,AL":         "366",
    "SPANISH FORT,AL":      "365",
    "SHEFFIELD,AL":         "356",
    "HALEYVILLE,AL":        "355",
    "HARTSELLE,AL":         "356",
    "SCOTTSBORO,AL":        "358",
    "MILLPORT,AL":          "355",
    "MOUNDVILLE,AL":        "354",
    "OAKMAN,AL":            "355",
    "TUSCALOOSA,AL":        "354",
    "COLUMBIANA,AL":        "350",
    "VANCE,AL":             "351",
    "ASHFORD,AL":           "363",
    "SHELBY,AL":            "350",
    # South Carolina
    "COLUMBIA,SC":          "292",
    "CHARLESTON,SC":        "294",
    "NORTH CHARLESTON,SC":  "294",
    "CONWAY,SC":            "295",
    "EFFINGHAM,SC":         "295",
    "FLORENCE,SC":          "295",
    "MARION,SC":            "295",
    "PAWLEYS ISLAND,SC":    "295",
    "LITTLE RIVER,SC":      "295",
    "MYRTLE BEACH,SC":      "295",
    "LORIS,SC":             "295",
    "ALLENDALE,SC":         "298",
    "RIDGELAND,SC":         "299",
    "SUMMERVILLE,SC":       "294",
    "LEXINGTON,SC":         "291",
    "INMAN,SC":             "293",
    "SPARTANBURG,SC":       "293",
    "ROCK HILL,SC":         "297",
    "LANCASTER,SC":         "297",
    "COWPENS,SC":           "293",
    "LUGOFF,SC":            "290",
    "SUMTER,SC":            "291",
    "FAIR PLAY,SC":         "296",
    "TOWNVILLE,SC":         "296",
    "JOHNS ISLAND,SC":      "294",
    "PAGELAND,SC":          "297",
    # North Carolina
    "FAYETTEVILLE,NC":      "283",
    "CHARLOTTE,NC":         "282",
    "RALEIGH,NC":           "276",
    "GREENVILLE,NC":        "278",
    "WINSTON-SALEM,NC":     "271",
    "APEX,NC":              "275",
    "MORRISVILLE,NC":       "275",
    "CHADBOURN,NC":         "284",
    "CLINTON,NC":           "283",
    "DUNN,NC":              "283",
    "ELIZABETH CITY,NC":    "279",
    "HARMONY,NC":           "286",
    "HARRISBURG,NC":        "280",
    "LOCUST,NC":            "280",
    "MIDDLESEX,NC":         "275",
    "MONROE,NC":            "281",
    "MORGANTON,NC":         "286",
    "NEW LONDON,NC":        "280",
    "RICHFIELD,NC":         "280",
    "RUTHERFORDTON,NC":     "281",
    "SALISBURY,NC":         "281",
    "ALBEMARLE,NC":         "280",
    "GASTONIA,NC":          "280",
    # Virginia
    "CHESAPEAKE,VA":        "233",
    "CREWE,VA":             "239",
    "CULPEPER,VA":          "227",
    "ELKWOOD,VA":           "227",
    "FREDERICKSBURG,VA":    "224",
    "MADISON,VA":           "227",
    "MILFORD,VA":           "224",
    "NORTH CHESTERFIELD,VA":"232",
    "ORANGE,VA":            "227",
    "WARSAW,VA":            "225",
    "WEST POINT,VA":        "230",
    "WINCHESTER,VA":        "226",
    # Tennessee
    "KNOXVILLE,TN":         "379",   # 37901-37999 -> 379  ← TN_379
    "MORRISTOWN,TN":        "378",
    "PINEY FLATS,TN":       "376",
    "MOSHEIM,TN":           "378",
    "ASHLAND CITY,TN":      "370",
    "LEBANON,TN":           "370",
    "MURFREESBORO,TN":      "371",
    "MEMPHIS,TN":           "381",
    "MORRISTOWN,TN":        "378",
    # Kentucky
    "LOUISVILLE,KY":        "402",   # 40202-40218 -> 402  ← KY_402
    "MILLWOOD,KY":          "427",
    # Mississippi
    "HATTIESBURG,MS":       "394",
    "NEWTON,MS":            "393",
    "MOUNT OLIVE,MS":       "394",
    "WAYNESBORO,MS":        "394",
    "PHILADELPHIA,MS":      "393",
    # Louisiana
    "BELLE CHASSE,LA":      "700",
    "THIBODAUX,LA":         "703",
    "DERIDDER,LA":          "706",
    "OPELOUSAS,LA":         "705",
    "WEST MONROE,LA":       "712",
    # Texas
    "AUSTIN,TX":            "787",
    "CARROLLTON,TX":        "750",
    "CONROE,TX":            "773",
    "FORT WORTH,TX":        "761",
    "FRISCO,TX":            "750",
    "GRANDVIEW,TX":         "760",
    "HENDERSON,TX":         "756",
    "HOUSTON,TX":           "770",
    "IRVING,TX":            "750",
    "MANSFIELD,TX":         "760",
    "MOSCOW,TX":            "759",
    "NEW WAVERLY,TX":       "773",
    "PINELAND,TX":          "759",
    "ROSENBERG,TX":         "774",
    "SAGINAW,TX":           "761",
    "TEXAS CITY,TX":        "775",
    # Oklahoma
    "STILWELL,OK":          "744",
    # Arkansas
    "FAYETTEVILLE,AR":      "727",
    # West Virginia
    "Clay,WV":              "251",
    # Maryland
    "MT. AIRY,MD":          "217",
    "BALTIMORE,MD":         "212",
    # New Jersey
    "BERLIN,NJ":            "080",
    # Pennsylvania
    "MILLERSBURG,PA":       "170",
    # Ohio
    "COLUMBUS,OH":          "432",
    # Indiana
    "GRABILL,IN":           "467",
    "HUNTINGBURG,IN":       "476",
    "INDIANAPOLIS,IN":      "462",
    "MILROY,IN":            "462",
    "SHELBYVILLE,IN":       "461",
    # Illinois
    "HILLSBORO,IL":         "620",
    "SPRINGFIELD,IL":       "626",
    # Missouri
    "JEFFERSON CITY,MO":    "651",
    "ST LOUIS,MO":          "631",
    "ST PETERS,MO":         "633",
    # Kansas
    "KANSAS CITY,KS":       "661",
    # Wisconsin
    "PRENTICE,WI":          "544",
    # Michigan
    "EDWARDSBURG,MI":       "490",
    # Rhode Island
    "NORTH KINGSTOWN,RI":   "028",
}


def get_state(city_st: str) -> str:
    """Extract two-letter state from 'CITY,ST' format. Returns '' if not parseable."""
    if "," in city_st:
        return city_st.rsplit(",", 1)[1].strip()
    return ""


def get_zip3(city_st: str) -> Optional[str]:
    """Return 3-digit ZIP prefix for a city, or None if unknown."""
    z = CITY_ZIP3.get(city_st)
    if z is None:
        logger.debug("ZIP3 unknown for location: %s", city_st)
    return z


def matches_destination_key(city_st: str, dest_key: str) -> bool:
    """
    Return True if city_st matches the difficult-lane destination_key.

    dest_key formats:
      "TX"     -> two-letter state code: match if city state == "TX"
      "GA_303" -> ZIP prefix: match if city in GA AND ZIP3 == "303"
    """
    if "_" not in dest_key:
        # State-code match
        return get_state(city_st) == dest_key
    # ZIP-prefix match: "GA_303" -> state="GA", prefix="303"
    parts = dest_key.split("_", 1)
    state_code, prefix = parts[0], parts[1]
    if get_state(city_st) != state_code:
        return False
    z3 = get_zip3(city_st)
    if z3 is None:
        logger.warning(
            "Cannot determine ZIP prefix for '%s'; conservatively NOT matching '%s'",
            city_st, dest_key,
        )
        return False
    return z3 == prefix


def find_difficult_lane_adder(
    origin: str,
    dest_city: str,
    difficult_lanes: List[DifficultLane],
) -> float:
    """
    Return the largest applicable difficult-lane adder ($/mile) for an order,
    or 0.0 if no difficult lane applies.

    An adder applies when:
      dl.origin == origin  AND  dest_city matches dl.destination_key
    """
    best = 0.0
    for dl in difficult_lanes:
        if dl.origin == origin and matches_destination_key(dest_city, dl.destination_key):
            if dl.adder_per_mile > best:
                best = dl.adder_per_mile
    return best


# ──────────────────────────────────────────────────────────────────
# Distance utilities
# ──────────────────────────────────────────────────────────────────

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two (lat, lon) points."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def city_distance_miles(
    city_a: str,
    city_b: str,
    problem: ProblemData,
) -> float:
    """
    Return lane mileage if available, else Haversine fallback.
    Logs a warning if fallback is used.
    """
    lane_key = (city_a, city_b)
    if lane_key in problem.lane_mileage:
        return problem.lane_mileage[lane_key]
    # Haversine fallback
    if city_a in problem.locations and city_b in problem.locations:
        lat1, lon1 = problem.locations[city_a]
        lat2, lon2 = problem.locations[city_b]
        d = haversine_miles(lat1, lon1, lat2, lon2)
        logger.warning("Using Haversine fallback for (%s -> %s): %.1f mi", city_a, city_b, d)
        return d
    logger.error("Cannot compute distance (%s -> %s): not in lanes or locations", city_a, city_b)
    return float("inf")
