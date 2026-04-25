"""Tests for geography.py — location parsing, ZIP matching, difficult-lane lookup."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from solver.geography import (
    get_state,
    get_zip3,
    matches_destination_key,
    find_difficult_lane_adder,
    haversine_miles,
    CITY_ZIP3,
)
from solver.data import DifficultLane


# ── get_state ─────────────────────────────────────────────────────

def test_get_state_normal():
    assert get_state("ALBANY,GA") == "GA"
    assert get_state("HOUSTON,TX") == "TX"
    assert get_state("LITTLE RIVER,SC") == "SC"


def test_get_state_no_comma():
    assert get_state("TX") == ""
    assert get_state("GA_303") == ""


def test_get_state_multi_comma():
    # Edge case: city name contains extra comma — use last split
    assert get_state("ST. LOUIS,MO") == "MO"


# ── get_zip3 ──────────────────────────────────────────────────────

def test_get_zip3_known_cities():
    assert get_zip3("ATLANTA,GA") == "303"
    assert get_zip3("BAINBRIDGE,GA") == "398"
    assert get_zip3("LOUISVILLE,KY") == "402"
    assert get_zip3("KNOXVILLE,TN") == "379"
    assert get_zip3("POMPANO BEACH,FL") == "330"


def test_get_zip3_unknown_returns_none():
    assert get_zip3("UNKNOWN CITY,XX") is None


def test_get_zip3_all_218_cities_have_entry():
    """
    Every city in CITY_ZIP3 should have a 3-digit string value.
    This catches typos in the static dict.
    """
    for city, prefix in CITY_ZIP3.items():
        assert isinstance(prefix, str), f"{city} has non-string prefix"
        assert len(prefix) == 3, f"{city} prefix '{prefix}' is not 3 digits"
        assert prefix.isdigit(), f"{city} prefix '{prefix}' is not numeric"


# ── matches_destination_key ───────────────────────────────────────

class TestMatchesDestinationKey:
    def test_state_match_tx(self):
        assert matches_destination_key("HOUSTON,TX", "TX")
        assert matches_destination_key("AUSTIN,TX", "TX")
        assert matches_destination_key("CONROE,TX", "TX")

    def test_state_no_match(self):
        assert not matches_destination_key("ALBANY,GA", "TX")
        assert not matches_destination_key("HOUSTON,TX", "GA")

    def test_zip_prefix_match_ga303(self):
        # ATLANTA,GA -> ZIP3="303" -> matches "GA_303"
        assert matches_destination_key("ATLANTA,GA", "GA_303")

    def test_zip_prefix_no_match_wrong_prefix(self):
        assert not matches_destination_key("ATLANTA,GA", "GA_302")
        assert not matches_destination_key("MARIETTA,GA", "GA_303")

    def test_zip_prefix_match_bainbridge_ga398(self):
        assert matches_destination_key("BAINBRIDGE,GA", "GA_398")

    def test_zip_prefix_match_louisville_ky402(self):
        assert matches_destination_key("LOUISVILLE,KY", "KY_402")

    def test_zip_prefix_match_knoxville_tn379(self):
        assert matches_destination_key("KNOXVILLE,TN", "TN_379")

    def test_zip_prefix_match_pompano_fl330(self):
        assert matches_destination_key("POMPANO BEACH,FL", "FL_330")

    def test_state_mismatch_in_zip_key(self):
        # GA_303 should not match a TX city even if it somehow had ZIP prefix 303
        assert not matches_destination_key("HOUSTON,TX", "GA_303")

    def test_wv_state_match(self):
        assert matches_destination_key("Clay,WV", "WV")

    def test_va_state_match(self):
        assert matches_destination_key("CHESAPEAKE,VA", "VA")


# ── find_difficult_lane_adder ─────────────────────────────────────

SAMPLE_DIFFICULT_LANES = [
    DifficultLane("ALBANY,GA", "TX", 1.75),
    DifficultLane("ALBANY,GA", "WV", 2.00),
    DifficultLane("ALBANY,GA", "GA_303", 2.00),
    DifficultLane("CONWAY,SC", "TX", 1.15),
    DifficultLane("CONWAY,SC", "WV", 1.00),
]


class TestFindDifficultLaneAdder:
    def test_exact_match_state(self):
        adder = find_difficult_lane_adder("ALBANY,GA", "HOUSTON,TX", SAMPLE_DIFFICULT_LANES)
        assert adder == pytest.approx(1.75)

    def test_exact_match_wv(self):
        adder = find_difficult_lane_adder("ALBANY,GA", "Clay,WV", SAMPLE_DIFFICULT_LANES)
        assert adder == pytest.approx(2.00)

    def test_zip_prefix_match(self):
        adder = find_difficult_lane_adder("ALBANY,GA", "ATLANTA,GA", SAMPLE_DIFFICULT_LANES)
        assert adder == pytest.approx(2.00)

    def test_different_origin_no_match(self):
        # BARNESVILLE,GA is not in SAMPLE_DIFFICULT_LANES
        adder = find_difficult_lane_adder("BARNESVILLE,GA", "HOUSTON,TX", SAMPLE_DIFFICULT_LANES)
        assert adder == pytest.approx(0.0)

    def test_no_difficult_lane_same_state(self):
        # ALBANY,GA -> ALBANY,GA: no TX/WV/GA_303 match
        adder = find_difficult_lane_adder("ALBANY,GA", "ALBANY,GA", SAMPLE_DIFFICULT_LANES)
        assert adder == pytest.approx(0.0)

    def test_conway_tx(self):
        adder = find_difficult_lane_adder("CONWAY,SC", "CONROE,TX", SAMPLE_DIFFICULT_LANES)
        assert adder == pytest.approx(1.15)

    def test_returns_largest_adder(self):
        # If multiple rules match, returns max (currently data has no overlap,
        # but function should handle it)
        dls = [
            DifficultLane("X,GA", "TX", 1.00),
            DifficultLane("X,GA", "TX", 2.00),
        ]
        assert find_difficult_lane_adder("X,GA", "HOUSTON,TX", dls) == pytest.approx(2.00)


# ── haversine_miles ───────────────────────────────────────────────

def test_haversine_same_point():
    assert haversine_miles(33.0, -84.0, 33.0, -84.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_atlanta_to_savannah():
    # Atlanta (33.75, -84.39) to Savannah (32.08, -81.09): ~246 road miles
    # Haversine (straight-line) ~ 210-215 miles
    d = haversine_miles(33.754466, -84.389815, 32.079007, -81.092134)
    assert 200 < d < 230, f"Expected ~215 mi, got {d:.1f}"


def test_haversine_symmetry():
    d1 = haversine_miles(30.0, -90.0, 35.0, -80.0)
    d2 = haversine_miles(35.0, -80.0, 30.0, -90.0)
    assert d1 == pytest.approx(d2, abs=1e-6)
