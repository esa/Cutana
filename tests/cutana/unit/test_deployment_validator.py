#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for deployment_validator.py using real data and environment."""

import pytest

from cutana.deployment_validator import DeploymentValidator


@pytest.fixture
def validator():
    """Create a DeploymentValidator instance and clean up after test."""
    v = DeploymentValidator(verbose=False)
    yield v
    v.cleanup()


def test_all_dependencies_importable(validator):
    """All dependencies listed in pyproject.toml should be importable."""
    assert validator.validate_dependencies() is True


def test_run_all_validations_comprehensive(validator):
    """Run full validation suite once and assert all properties."""
    results = validator.run_all_validations()
    assert isinstance(results, dict)
    expected_keys = {
        "python_environment",
        "dependencies",
        "configuration",
        "end_to_end",
        "git_access",
    }
    assert set(results.keys()) == expected_keys
    for key, value in results.items():
        assert isinstance(value, bool), f"Result for '{key}' is {type(value)}, expected bool"
    assert results["dependencies"] is True
    assert results["configuration"] is True
    assert results["end_to_end"] is True
