#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for validate_config.py — covering type validation, boundary checks,
allowed values, special validators, and cross-parameter checks."""

import pytest
from dotmap import DotMap

from cutana import get_default_config, validate_config
from cutana.validate_config import (
    _validate_flux_conversion_config,
    validate_channel_order_consistency,
    validate_config_for_processing,
)


def _make_valid_config(**overrides):
    """Build a valid config, optionally overriding fields."""
    cfg = get_default_config()
    cfg.source_catalogue = "dummy.csv"
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = obj[p]
        obj[parts[-1]] = val
    return cfg


# ────────────────────────────────────────────────────────────────────
# Parametrized type / boundary / allowed-value tests
# ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "override_key, bad_value, match_pattern",
    [
        ("name", 42, "must be a string"),
        ("output_dir", 123, "must be a string/directory path"),
        ("source_catalogue", 999, "must be a string/file path"),
        ("max_workers", "four", "must be an integer"),
        ("max_workers", True, "must be an integer"),
        ("padding_factor", "big", "must be a number"),
        ("padding_factor", True, "must be a number"),
        ("write_to_disk", 1, "must be a boolean"),
        ("available_extensions", "not_a_list", "must be a list"),
        ("loadbalancer", {"memory_safety_margin": 0.1}, "must be a DotMap"),
    ],
    ids=[
        "str",
        "directory",
        "file",
        "int",
        "bool_as_int",
        "float",
        "bool_as_float",
        "bool",
        "list",
        "dotmap",
    ],
)
def test_wrong_type_rejected(override_key, bad_value, match_pattern):
    cfg = _make_valid_config(**{override_key: bad_value})
    with pytest.raises(ValueError, match=match_pattern):
        validate_config(cfg, check_paths=False)


@pytest.mark.parametrize(
    "override_key, bad_value",
    [
        ("max_workers", 0),
        ("padding_factor", 0.01),
    ],
    ids=["int_below_min", "float_below_min"],
)
def test_below_min_rejected(override_key, bad_value):
    cfg = _make_valid_config(**{override_key: bad_value})
    with pytest.raises(ValueError, match="must be >="):
        validate_config(cfg, check_paths=False)


@pytest.mark.parametrize(
    "override_key, bad_value",
    [
        ("max_workers", 9999),
        ("padding_factor", 99.9),
    ],
    ids=["int_above_max", "float_above_max"],
)
def test_above_max_rejected(override_key, bad_value):
    cfg = _make_valid_config(**{override_key: bad_value})
    with pytest.raises(ValueError, match="must be <="):
        validate_config(cfg, check_paths=False)


@pytest.mark.parametrize(
    "override_key, bad_value",
    [
        ("log_level", "VERBOSE"),
        ("output_format", "png"),
        ("normalisation_method", "sqrt"),
    ],
    ids=["log_level", "output_format", "normalisation_method"],
)
def test_disallowed_value_rejected(override_key, bad_value):
    cfg = _make_valid_config(**{override_key: bad_value})
    with pytest.raises(ValueError, match="must be one of"):
        validate_config(cfg, check_paths=False)


def test_file_rejects_nonexistent_path():
    cfg = _make_valid_config(source_catalogue="/no/such/file.csv")
    with pytest.raises(ValueError, match="file does not exist"):
        validate_config(cfg, check_paths=True)


# ────────────────────────────────────────────────────────────────────
# Special validators — valid inputs accepted
# ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"fits_extensions": ["PRIMARY", 0, "SCI", 1]},
        {"selected_extensions": [{"name": "VIS", "ext": "IMAGE"}]},
        {"channel_weights": {"VIS": [1.0, 0.0], "NIR": [0.0, 1.0]}},
        {"channel_weights": None},
        {"user_flux_conversion_function": lambda x, y: x},
    ],
    ids=[
        "fits_extensions",
        "selected_extensions_dict",
        "channel_weights_dict",
        "channel_weights_none",
        "callable_function",
    ],
)
def test_special_validators_accept_valid(overrides):
    cfg = _make_valid_config(**overrides)
    validate_config(cfg, check_paths=False)


def test_flux_keywords_accepts_dict():
    cfg = _make_valid_config()
    cfg.flux_conversion_keywords = {"AB_zeropoint": "MAGZERO"}
    validate_config(cfg, check_paths=False)


# ────────────────────────────────────────────────────────────────────
# Special validators — invalid inputs rejected
# ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "override_key, bad_value, match_pattern",
    [
        ("fits_extensions", "PRIMARY", "must be a list"),
        ("fits_extensions", ["PRIMARY", 3.14], "must be str or int"),
        ("fits_extensions", ["PRIMARY", None], "must be str or int"),
        ("selected_extensions", "VIS", "must be a list"),
        ("selected_extensions", [{"ext": "IMAGE"}], "must have 'name' key"),
        ("selected_extensions", [3.14], "must be str, int, or dict"),
        ("channel_weights", [1, 2, 3], "must be a dict"),
        ("channel_weights", {123: [1.0]}, "channel names must be strings"),
        ("channel_weights", {"VIS": 1.0}, "weights must be lists"),
        ("channel_weights", {"VIS": [1.0, "bad"]}, "weights must contain only numbers"),
        ("user_flux_conversion_function", "not_callable", "must be callable or None"),
    ],
    ids=[
        "fits_ext_non_list",
        "fits_ext_float_elem",
        "fits_ext_none_elem",
        "sel_ext_non_list",
        "sel_ext_no_name",
        "sel_ext_float_elem",
        "weights_non_dict",
        "weights_non_str_keys",
        "weights_non_list_vals",
        "weights_non_numeric",
        "function_not_callable",
    ],
)
def test_special_validators_reject_invalid(override_key, bad_value, match_pattern):
    cfg = _make_valid_config(**{override_key: bad_value})
    with pytest.raises(ValueError, match=match_pattern):
        validate_config(cfg, check_paths=False)


def test_flux_keywords_rejects_non_dict():
    cfg = _make_valid_config()
    cfg.flux_conversion_keywords = "not_a_dict"
    with pytest.raises(ValueError, match="must be a dict or DotMap"):
        validate_config(cfg, check_paths=False)


# ────────────────────────────────────────────────────────────────────
# Cross-parameter checks
# ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "extensions",
    [[], None],
    ids=["empty_list", "none"],
)
def test_processing_requires_selected_extensions(extensions):
    cfg = _make_valid_config(selected_extensions=extensions)
    with pytest.raises(ValueError, match="required for processing"):
        validate_config_for_processing(cfg, check_paths=False)


@pytest.mark.parametrize(
    "func, should_raise, match_pattern",
    [
        ("not_callable", True, "must be callable"),
        (lambda x: x, True, "must take exactly 2 arguments"),
        (lambda img, hdr: img, False, None),
        (None, False, None),
    ],
    ids=["non_callable", "wrong_arity", "valid_2arg", "none"],
)
def test_flux_conversion_cross_validation(func, should_raise, match_pattern):
    cfg = DotMap(user_flux_conversion_function=func)
    if should_raise:
        with pytest.raises(ValueError, match=match_pattern):
            _validate_flux_conversion_config(cfg)
    else:
        _validate_flux_conversion_config(cfg)


@pytest.mark.parametrize(
    "output_format, should_raise",
    [
        ("zarr", True),
        ("fits", False),
    ],
    ids=["zarr_rejected", "fits_accepted"],
)
def test_cutout_extraction_requires_fits(output_format, should_raise):
    cfg = _make_valid_config(do_only_cutout_extraction=True, output_format=output_format)
    if should_raise:
        with pytest.raises(ValueError, match="output_format must be 'fits'"):
            validate_config(cfg, check_paths=False)
    else:
        validate_config(cfg, check_paths=False)


def test_unexpected_keys_does_not_raise():
    cfg = _make_valid_config()
    d = cfg.toDict()
    d["totally_unknown_param"] = 42
    cfg2 = DotMap(d, _dynamic=False)
    cfg2.source_catalogue = "dummy.csv"
    validate_config(cfg2, check_paths=False)


# ────────────────────────────────────────────────────────────────────
# Strict channel order validation (weak_check=False)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "names, weights, should_raise, match_pattern",
    [
        (["VIS", "NIR", "MIR"], {"VIS": [1.0], "NIR": [0.5], "MIR": [0.3]}, False, None),
        (["PRIMARY"], {"PRIMARY": [1.0]}, False, None),
        (["VIS", "NIR"], {"VIS": [1.0], "MIR": [0.3]}, True, "Channel mismatch"),
        (["NIR", "VIS"], {"VIS": [1.0], "NIR": [0.5]}, True, "Channel order mismatch"),
    ],
    ids=["matching_order", "single_channel", "mismatched_sets", "wrong_order"],
)
def test_strict_channel_order_validation(names, weights, should_raise, match_pattern):
    if should_raise:
        with pytest.raises(AssertionError, match=match_pattern):
            validate_channel_order_consistency(names, weights, weak_check=False)
    else:
        validate_channel_order_consistency(names, weights, weak_check=False)
