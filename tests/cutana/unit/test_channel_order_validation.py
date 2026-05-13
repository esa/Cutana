#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for channel order validation functionality.

Tests the validate_channel_order_consistency function to ensure it properly
catches channel order mismatches between data tensor and channel weights.
"""

import pytest

from cutana.validate_config import validate_channel_order_consistency


class TestChannelOrderValidation:
    """Unit tests for channel order validation."""

    def test_matching_channel_order_passes(self):
        """Test that matching channel orders pass validation."""
        tensor_channel_names = ["channel_a", "channel_b", "channel_c"]
        channel_weights = {
            "channel_a": [1.0, 0.0],
            "channel_b": [0.0, 1.0],
            "channel_c": [0.5, 0.5],
        }

        # This should not raise any exception
        validate_channel_order_consistency(tensor_channel_names, channel_weights)

    def test_different_channel_sets_fails(self):
        """Test that different channel sets fail validation."""
        tensor_channel_names = ["channel_a", "channel_b", "channel_c"]
        channel_weights = {
            "channel_a": [1.0, 0.0],
            "channel_b": [0.0, 1.0],
            "channel_d": [0.5, 0.5],  # Different channel name
        }

        with pytest.raises(AssertionError) as exc_info:
            validate_channel_order_consistency(tensor_channel_names, channel_weights)

        assert "Channel mapping incomplete." in str(exc_info.value)

    def test_missing_channels_in_weights_fails(self):
        """Tensor extensions absent from channel_weights are rejected (#315):
        ``combine_channels`` would zero them out positionally."""
        tensor_channel_names = ["channel_a", "channel_b", "channel_c"]
        channel_weights = {
            "channel_a": [1.0, 0.0],
            "channel_b": [0.0, 1.0],
            # Missing channel_c
        }
        with pytest.raises(AssertionError, match="silently drop tensor extensions"):
            validate_channel_order_consistency(tensor_channel_names, channel_weights)

    def test_extra_channels_in_weights_fails(self):
        """Test that extra channels in weights fail validation."""
        tensor_channel_names = ["channel_a", "channel_b"]
        channel_weights = {
            "channel_a": [1.0, 0.0],
            "channel_b": [0.0, 1.0],
            "channel_c": [0.5, 0.5],  # Extra channel
        }

        with pytest.raises(AssertionError) as exc_info:
            validate_channel_order_consistency(tensor_channel_names, channel_weights)

        error_message = str(exc_info.value)
        assert "Channel mapping incomplete" in error_message
        assert "Missing:" in error_message

    def test_wrong_channel_order_fails(self):
        """Test that wrong channel order fails validation."""
        tensor_channel_names = ["channel_a", "channel_b", "channel_c"]
        channel_weights = {
            "channel_c": [1.0, 0.0],  # Wrong order
            "channel_b": [0.0, 1.0],
            "channel_a": [0.5, 0.5],
        }

        with pytest.raises(AssertionError) as exc_info:
            validate_channel_order_consistency(tensor_channel_names, channel_weights)

        error_message = str(exc_info.value)
        assert "Channel order mismatch!" in error_message
        assert "Data tensor maps to channels in order:" in error_message
        assert "channel_weights expects:" in error_message

    def test_single_channel_passes(self):
        """Test that single channel validation passes."""
        tensor_channel_names = ["single_channel"]
        channel_weights = {"single_channel": [1.0]}

        # This should not raise any exception
        validate_channel_order_consistency(tensor_channel_names, channel_weights)

    def test_empty_channels_fails(self):
        """Test that empty channel lists fail validation."""
        tensor_channel_names = []
        channel_weights = {}

        # This should pass since both are empty (though probably not a real use case)
        with pytest.raises(AssertionError) as exc_info:
            validate_channel_order_consistency(tensor_channel_names, channel_weights)

        assert "do not contain any config channel name." in str(exc_info.value)

    def test_alphabetical_vs_insertion_order(self):
        """Test the specific case that caused the original issue."""
        # This simulates the case where FITS files are processed alphabetically
        # but channel weights are defined in a different order
        tensor_channel_names = ["AAA_extension", "BBB_extension", "CCC_extension"]  # Alphabetical
        channel_weights = {
            "CCC_extension": [1.0, 0.0],  # Different order
            "BBB_extension": [0.0, 1.0],
            "AAA_extension": [0.5, 0.5],
        }

        with pytest.raises(AssertionError) as exc_info:
            validate_channel_order_consistency(tensor_channel_names, channel_weights)

        error_message = str(exc_info.value)
        assert "Channel order mismatch!" in error_message

    def test_extra_tensor_channels_silent_drop_fails(self):
        """Regression test for issue #315 (scenario B).

        ``combine_channels`` applies ``channel_weights`` positionally to tensor
        extensions. With one weight and a multi-channel tensor whose first
        extension is unrelated to the weight key, the weight binds to extension
        0 and the named extension's data is dropped — silent corruption. The
        validator must reject this configuration.
        """
        tensor_channel_names = ["tile_VIS", "tile_NIR-H"]
        channel_weights = {"NIR-H": [1.0]}

        with pytest.raises(AssertionError):
            validate_channel_order_consistency(tensor_channel_names, channel_weights)

    def test_corrected_alphabetical_order_passes(self):
        """Test that corrected alphabetical order passes."""
        # This simulates the corrected case where channel weights match
        # the alphabetical processing order
        tensor_channel_names = ["AAA_extension", "BBB_extension", "CCC_extension"]  # Alphabetical
        channel_weights = {
            "AAA_extension": [1.0, 0.0],  # Matching order
            "BBB_extension": [0.0, 1.0],
            "CCC_extension": [0.5, 0.5],
        }

        # This should not raise any exception
        validate_channel_order_consistency(tensor_channel_names, channel_weights)
