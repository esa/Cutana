#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for StreamingOrchestrator.
"""

import pytest

from cutana import StreamingOrchestrator, get_default_config


@pytest.fixture
def base_config(tmp_path):
    config = get_default_config()
    config.output_format = "fits"
    config.target_resolution = 32
    config.selected_extensions = ["VIS"]
    config.channel_weights = {"VIS": [1.0]}
    config.skip_memory_calibration_wait = True
    dummy_catalogue = tmp_path / "dummy.csv"
    dummy_catalogue.touch()
    config.source_catalogue = str(dummy_catalogue)
    return config


class TestInitStreamingValidation:
    """Tests for init_streaming parameter validation."""

    def test_do_only_cutout_extraction_with_write_to_disk_false_raises(self, base_config):
        """do_only_cutout_extraction=True is only meaningful when writing to disk."""
        base_config.do_only_cutout_extraction = True

        orchestrator = StreamingOrchestrator(base_config)
        try:
            with pytest.raises(ValueError, match="do_only_cutout_extraction"):
                orchestrator.init_streaming(batch_size=10, write_to_disk=False)
        finally:
            orchestrator.cleanup()

    def test_do_only_cutout_extraction_with_write_to_disk_true_does_not_raise_early(
        self, base_config, tmp_path
    ):
        """do_only_cutout_extraction=True + write_to_disk=True must not raise the guard."""
        base_config.do_only_cutout_extraction = True
        base_config.output_dir = str(tmp_path)

        orchestrator = StreamingOrchestrator(base_config)
        try:
            # The guard must not fire; subsequent I/O will raise (empty catalogue),
            # but we only care that the ValueError guard is not triggered.
            with pytest.raises(Exception) as exc_info:
                orchestrator.init_streaming(batch_size=10, write_to_disk=True)
            assert "do_only_cutout_extraction" not in str(exc_info.value)
        finally:
            orchestrator.cleanup()
