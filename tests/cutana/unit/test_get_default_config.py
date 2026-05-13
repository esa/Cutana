#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for get_default_config.py — covering create_config_from_dict
round-trips, enum serialization, numpy dtype serialization, and TOML I/O."""

from enum import Enum

import numpy as np
import pytest
from dotmap import DotMap

from cutana import (
    create_config_from_dict,
    get_default_config,
    load_config_toml,
    save_config_toml,
    validate_config,
)

# ────────────────────────────────────────────────────────────────────
# _clean_dict enum serialization (lines 270-275)
# ────────────────────────────────────────────────────────────────────


class _DummyEnum(Enum):
    """Minimal enum for testing serialization round-trips."""

    VALUE_A = 0
    VALUE_B = 1


class TestEnumSerialization:
    """Tests for enum serialization via save_config_toml / _clean_dict."""

    def test_enum_serialized_to_marker_string(self, tmp_path):
        cfg = get_default_config()
        cfg.source_catalogue = "test.csv"
        # Inject an enum value into the config
        cfg.test_enum = _DummyEnum.VALUE_B

        toml_path = tmp_path / "cfg.toml"
        save_config_toml(cfg, str(toml_path))

        raw = toml_path.read_text()
        assert "__enum__" in raw
        assert "__1" in raw  # enum value

    def test_enum_round_trip_via_toml(self, tmp_path):
        """save -> load should restore the enum value."""
        cfg = get_default_config()
        cfg.source_catalogue = "test.csv"
        cfg.test_enum = _DummyEnum.VALUE_A

        toml_path = tmp_path / "cfg.toml"
        save_config_toml(cfg, str(toml_path))

        restored = load_config_toml(str(toml_path))
        assert restored.test_enum == _DummyEnum.VALUE_A
        assert isinstance(restored.test_enum, _DummyEnum)


# ────────────────────────────────────────────────────────────────────
# Numpy dtype serialization (lines 266-269)
# ────────────────────────────────────────────────────────────────────


class TestNumpyDtypeSerialization:
    """Tests for numpy dtype class serialization in _clean_dict."""

    def test_numpy_dtype_serialized_to_marker(self, tmp_path):
        cfg = get_default_config()
        cfg.source_catalogue = "test.csv"
        cfg.test_dtype = np.float32

        toml_path = tmp_path / "cfg.toml"
        save_config_toml(cfg, str(toml_path))

        raw = toml_path.read_text()
        assert "__numpy_dtype__float32" in raw

    def test_numpy_dtype_round_trip(self, tmp_path):
        cfg = get_default_config()
        cfg.source_catalogue = "test.csv"
        cfg.test_dtype = np.uint8

        toml_path = tmp_path / "cfg.toml"
        save_config_toml(cfg, str(toml_path))

        restored = load_config_toml(str(toml_path))
        assert restored.test_dtype is np.uint8


# ────────────────────────────────────────────────────────────────────
# create_config_from_dict round-trip (lines 155-229)
# ────────────────────────────────────────────────────────────────────


class TestCreateConfigFromDict:
    """Tests for create_config_from_dict merging and type restoration."""

    def test_overrides_are_applied(self):
        cfg = create_config_from_dict({"max_workers": 7, "target_resolution": 128})
        assert cfg.max_workers == 7
        assert cfg.target_resolution == 128

    def test_defaults_are_preserved_for_unset_keys(self):
        cfg = create_config_from_dict({"max_workers": 3})
        assert cfg.target_resolution == 256  # default value
        assert cfg.output_format == "zarr"

    def test_nested_override_merges_correctly(self):
        cfg = create_config_from_dict({"normalisation": {"percentile": 95.0}})
        assert cfg.normalisation.percentile == 95.0
        # Other nested defaults should remain
        assert cfg.normalisation.contrast == 0.25

    def test_numpy_dtype_restored_from_string(self):
        cfg = create_config_from_dict({"test_dtype": "__numpy_dtype__float64"})
        assert cfg.test_dtype is np.float64

    def test_enum_restored_from_string(self):
        # Use our test enum
        marker = f"__enum__{_DummyEnum.__module__}.{_DummyEnum.__qualname__}__1"
        cfg = create_config_from_dict({"test_enum": marker})
        assert cfg.test_enum == _DummyEnum.VALUE_B

    def test_full_round_trip_save_load(self, tmp_path):
        """save_config_toml -> load_config_toml produces equivalent config."""
        original = get_default_config()
        original.source_catalogue = "test.csv"
        original.max_workers = 4
        original.normalisation.percentile = 98.5

        toml_path = tmp_path / "round_trip.toml"
        save_config_toml(original, str(toml_path))
        restored = load_config_toml(str(toml_path))

        assert restored.max_workers == 4
        assert restored.normalisation.percentile == 98.5
        assert restored.output_format == "zarr"
        # Restored config should pass validation
        validate_config(restored, check_paths=False)


# ────────────────────────────────────────────────────────────────────
# _clean_dict: callable filtering & None removal (lines 256-281)
# ────────────────────────────────────────────────────────────────────


class TestCleanDict:
    """Tests for _clean_dict behavior (tested indirectly via save_config_toml)."""

    def test_none_values_omitted(self, tmp_path):
        cfg = get_default_config()
        cfg.source_catalogue = "test.csv"
        # external_fitsbolt_cfg defaults to None
        assert cfg.external_fitsbolt_cfg is None

        toml_path = tmp_path / "cfg.toml"
        save_config_toml(cfg, str(toml_path))

        raw = toml_path.read_text()
        assert "external_fitsbolt_cfg" not in raw

    def test_callable_values_omitted(self, tmp_path):
        cfg = get_default_config()
        cfg.source_catalogue = "test.csv"
        cfg.user_flux_conversion_function = lambda img, hdr: img

        toml_path = tmp_path / "cfg.toml"
        save_config_toml(cfg, str(toml_path))

        raw = toml_path.read_text()
        assert "user_flux_conversion_function" not in raw

    def test_empty_sub_dicts_omitted(self, tmp_path):
        cfg = get_default_config()
        cfg.source_catalogue = "test.csv"
        # Create a config with a sub-dict that will become empty after cleaning
        d = cfg.toDict()
        d["empty_section"] = {"only_none": None}
        cfg2 = DotMap(d, _dynamic=False)

        toml_path = tmp_path / "cfg.toml"
        save_config_toml(cfg2, str(toml_path))

        raw = toml_path.read_text()
        assert "empty_section" not in raw


# ────────────────────────────────────────────────────────────────────
# load_config_toml (line 309 — file I/O path)
# ────────────────────────────────────────────────────────────────────


class TestLoadConfigToml:
    """Tests for load_config_toml."""

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config_toml("/nonexistent/config.toml")

    def test_load_produces_valid_config(self, tmp_path):
        cfg = get_default_config()
        cfg.source_catalogue = "test.csv"
        toml_path = tmp_path / "cfg.toml"
        save_config_toml(cfg, str(toml_path))

        loaded = load_config_toml(str(toml_path))
        assert isinstance(loaded, DotMap)
        validate_config(loaded, check_paths=False)
