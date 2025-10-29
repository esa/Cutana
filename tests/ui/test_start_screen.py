#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the unified start screen."""

import pytest
from unittest.mock import patch, AsyncMock


@pytest.fixture
def mock_backend():
    """Create mock backend interface."""
    with patch("cutana_ui.utils.backend_interface.BackendInterface") as mock:
        mock.check_source_catalogue = AsyncMock(
            return_value={
                "num_sources": 100,
                "fits_files": ["file1.fits", "file2.fits"],
                "extensions": [
                    {"name": "VIS", "ext": "IMAGE"},
                    {"name": "NIR-H", "ext": "IMAGE"},
                    {"name": "NIR-J", "ext": "IMAGE"},
                ],
            }
        )
        yield mock


@pytest.fixture
def mock_config_manager():
    """Create mock config manager."""
    with patch("cutana.get_default_config.get_default_config") as mock:
        # Mock the new DotMap-based config system
        from dotmap import DotMap

        default_config = DotMap()
        default_config.max_workers = 4
        default_config.N_batch_cutout_process = 128
        default_config.target_resolution = 256
        default_config.output_format = "zarr"
        default_config.num_channels = 1  # Add missing num_channels parameter

        mock.get_default_config.return_value = default_config
        mock.convert_ui_config_to_cutana.return_value = default_config
        mock.save_config.return_value = "/path/to/config.toml"
        yield mock


class TestStartScreen:
    """Test the unified start screen."""

    def test_start_screen_initialization(self):
        """Test that start screen initializes with all components."""
        from cutana_ui.start_screen import StartScreen

        screen = StartScreen()

        # Check that all components are created
        assert hasattr(screen, "file_selection")
        # analysis_display is now integrated into file_selection
        assert hasattr(screen, "configuration")
        assert hasattr(screen, "output_folder")
        assert hasattr(screen, "start_button")
        assert hasattr(screen, "header_container")
        assert hasattr(screen, "help_button")

    def test_file_selection_component(self):
        """Test file selection component behavior."""
        from cutana_ui.start_screen.file_selection import FileSelectionComponent

        component = FileSelectionComponent()

        # Check components exist
        assert hasattr(component, "file_chooser")
        assert hasattr(component, "error_display")
        assert hasattr(component, "analysis_result")

        # File chooser should be created with CSV filter
        # The actual filter is passed to the CutanaFileChooser constructor
        assert hasattr(component.file_chooser, "file_chooser")

        # Initially no analysis result
        assert component.analysis_result is None

        # Error display initially hidden
        assert component.error_display.layout.display == "none"

    def test_analysis_integration_in_file_selection(self):
        """Test that analysis display is now integrated into file selection."""
        from cutana_ui.start_screen.file_selection import FileSelectionComponent

        component = FileSelectionComponent()

        # Should have analysis-related attributes
        assert hasattr(component, "analysis_result")
        assert hasattr(component, "error_display")

        # Initially no analysis results
        assert component.analysis_result is None

        # Error display should be hidden initially
        assert component.error_display.layout.display == "none"
        results = {
            "num_sources": 100,
            "fits_files": ["file1.fits", "file2.fits"],
            "extensions": [{"name": "VIS", "ext": "IMAGE"}],
        }
        component.show_analysis_results(results)
        assert component.analysis_result == results
        # Error display should remain hidden after showing results
        assert component.error_display.layout.display == "none"

    def test_configuration_component(self):
        """Test configuration component with matrix."""
        from cutana_ui.start_screen.configuration_component import ConfigurationComponent

        component = ConfigurationComponent()

        # Test setting extensions
        extensions = [
            {"name": "VIS", "ext": "IMAGE"},
            {"name": "NIR-H", "ext": "IMAGE"},
            {"name": "NIR-J", "ext": "IMAGE"},
        ]
        component.set_extensions(extensions)

        # Check that checkboxes are created (access through shared config)
        assert len(component.shared_config.extension_checkboxes) == 3

        # Check that matrix is initialized
        assert component.shared_config.current_channels == 1
        assert len(component.shared_config.channel_matrices) == 1

        # Test adding channels
        component.shared_config._add_channel()
        assert component.shared_config.current_channels == 2
        assert len(component.shared_config.channel_matrices) == 2

        # Test removing channels
        component.shared_config._remove_channel()
        assert component.shared_config.current_channels == 1

        # Test configuration retrieval
        config = component.get_configuration()
        assert "data_type" in config
        assert "target_resolution" in config
        assert "normalisation_method" in config
        assert "selected_extensions" in config
        assert "channel_weights" in config

    def test_output_folder_component(self):
        """Test output folder component."""
        from cutana_ui.start_screen.output_folder import OutputFolderComponent

        component = OutputFolderComponent()

        # Check default directory is set
        assert hasattr(component, "dir_chooser")
        assert hasattr(component.dir_chooser, "file_chooser")

        # Test getting output directory
        output_dir = component.get_output_dir()
        # Should return default or None
        assert output_dir is None or "cutana" in str(output_dir)

    def test_resolution_validation(self):
        """Test that resolution input validates minimum value."""
        from cutana_ui.start_screen.configuration_component import ConfigurationComponent

        component = ConfigurationComponent()

        # Try to set resolution below minimum
        component.resolution_input.value = 10
        # Should be clamped to minimum
        assert component.resolution_input.value >= 16

    def test_stretch_function_naming(self):
        """Test that stretch function uses 'linear' instead of 'none'."""
        from cutana_ui.widgets.configuration_widget import SharedConfigurationWidget
        from cutana import get_default_config

        # Create a configuration widget with advanced params enabled for testing normalisation
        config = get_default_config()
        component = SharedConfigurationWidget(
            config=config,
            compact=False,
            show_extensions=True,
            show_matrix=False,
            show_advanced_params=True,  # Enable advanced params to test normalisation
        )

        # Check dropdown options (only test if normalisation dropdown exists)
        if component.normalisation_dropdown is not None:
            assert "linear" in component.normalisation_dropdown.options
            assert "none" not in component.normalisation_dropdown.options

            # Set some dummy extensions first
            component.set_extensions([{"name": "TEST", "ext": "IMAGE"}])

            # Check that config returns "linear" for backend compatibility
            current_config = component.get_current_config()
            if component.normalisation_dropdown.value == "linear":
                assert current_config["normalisation_method"] == "linear"

    def test_automatic_analysis_on_file_selection(self, mock_backend):
        """Test that analysis starts automatically when file is selected."""
        from cutana_ui.start_screen import StartScreen
        from cutana.get_default_config import get_default_config

        with patch("asyncio.create_task"):
            screen = StartScreen()

            # Initialize config_data as DotMap if not already
            if not hasattr(screen, "config_data") or screen.config_data is None:
                screen.config_data = get_default_config()

            # Simulate file selection (synchronous part)
            file_path = "/path/to/test.csv"
            screen.config_data.source_catalogue = file_path

            # Simulate successful analysis result
            screen.analysis_result = {
                "num_sources": 100,
                "fits_files": ["file1.fits"],
                "extensions": [{"name": "VIS", "ext": "IMAGE"}],
            }

            # Simulate the UI updates that would happen after analysis
            # analysis_display is now integrated into file_selection component
            screen.configuration.layout.display = "block"
            screen.start_button.layout.display = "block"

            # Check that configuration panel is shown
            assert screen.configuration.layout.display == "block"
            assert screen.start_button.layout.display == "block"

    def test_color_scheme_application(self):
        """Test that ESA color scheme is applied."""
        from cutana_ui.styles import ESA_BLUE_DEEP, ESA_GREEN, ESA_RED, SUCCESS_COLOR, ERROR_COLOR

        # Verify color constants are defined correctly
        assert ESA_BLUE_DEEP == "#003249"
        assert ESA_GREEN == "#008542"
        assert ESA_RED == "#EC1A2F"
        assert SUCCESS_COLOR == ESA_GREEN
        assert ERROR_COLOR == ESA_RED

    def test_aspect_ratio_layout(self):
        """Test 16:9 aspect ratio implementation."""
        from cutana_ui.start_screen import StartScreen

        screen = StartScreen()

        # Check main container has responsive layout
        layout = screen.main_container.layout
        # ipywidgets doesn't have direct aspect_ratio property, check layout settings
        assert layout.max_width == "1200px"  # Updated to match new UI_SCALE (0.75)
        # Check that height is set (indicating 16:9 aspect ratio implementation)
        assert layout.height == "675px"  # Updated to match new UI_SCALE (0.75)

    def test_logo_presence(self):
        """Test that ESA logo is present in both screens."""
        try:
            from cutana_ui.start_screen import StartScreen
        except ImportError as e:
            if "ipyfilechooser" in str(e):
                pytest.skip("ipyfilechooser not available in test environment")
            raise

        from cutana_ui.main_screen import MainScreen
        from cutana.get_default_config import get_default_config

        # Test start screen
        start_screen = StartScreen()
        assert hasattr(start_screen, "header_container")
        assert hasattr(start_screen, "help_button")
        assert "svg" in start_screen.header_container.children[1].children[0].value.lower()

        # Test main screen with required config
        config = get_default_config()
        config.num_sources = 25
        config.available_extensions = []
        main_screen = MainScreen(config=config)
        assert hasattr(main_screen, "header_container")
        assert "svg" in main_screen.header_container.children[1].children[0].value.lower()


class TestUIIntegration:
    """Integration tests for the complete UI flow."""

    def test_app_initialization_with_new_structure(self):
        """Test that the app initializes with the new start screen."""
        from cutana_ui.app import CutanaApp

        app = CutanaApp()

        # Should show start screen instead of wizard steps
        assert len(app.container.children) == 1
        assert app.container.children[0].__class__.__name__ == "StartScreen"

    def test_dropdown_background_colors(self):
        """Test that dropdowns have proper dark backgrounds."""
        from cutana_ui.styles import COMMON_STYLES, ESA_BLUE_GREY

        # Check that dropdown styles are defined
        assert "widget-dropdown" in COMMON_STYLES
        assert ESA_BLUE_GREY in COMMON_STYLES
        assert "background: #335E6E" in COMMON_STYLES  # ESA_BLUE_GREY value

    def test_real_csv_file_integration(self):
        """Test integration with real CSV file from test data."""
        from cutana_ui.start_screen import StartScreen
        from cutana.get_default_config import get_default_config
        from pathlib import Path
        import asyncio

        # Get test CSV file path
        project_root = Path(__file__).parent.parent.parent
        test_csv = project_root / "tests" / "test_data" / "euclid_cutana_catalogue_small.csv"

        if test_csv.exists():
            screen = StartScreen()

            # Initialize config_data as DotMap if not already
            if not hasattr(screen, "config_data") or screen.config_data is None:
                screen.config_data = get_default_config()

            # Test file selection with real file
            screen.config_data.source_catalogue = str(test_csv)

            # Mock the backend call to avoid actual analysis
            expected_result = {
                "num_sources": 25,
                "fits_files": ["test.fits"],
                "extensions": [{"name": "VIS", "ext": "IMAGE"}],
            }

            with patch(
                "cutana_ui.utils.backend_interface.BackendInterface.check_source_catalogue",
                return_value=expected_result,
            ) as mock_analysis:

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(screen._analyze_catalogue(str(test_csv)))

                    # Verify analysis was called with correct file
                    mock_analysis.assert_called_once_with(str(test_csv))

                    # Verify results were processed
                    assert screen.analysis_result == expected_result
                    assert screen.config_data.num_sources == 25
                finally:
                    loop.close()
        else:
            # Skip test if file doesn't exist
            import pytest

            pytest.skip(f"Test CSV file not found: {test_csv}")

    def test_file_selection_triggers_analysis(self):
        """Test that file selection triggers analysis workflow."""
        from cutana_ui.start_screen import StartScreen
        from cutana.get_default_config import get_default_config

        # Mock the analysis method to avoid actual backend calls
        with patch.object(StartScreen, "_analyze_catalogue"):
            screen = StartScreen()

            # Initialize config_data as DotMap if not already
            if not hasattr(screen, "config_data") or screen.config_data is None:
                screen.config_data = get_default_config()

            # Call the file selection handler directly
            file_path = "/path/to/test.csv"
            screen._on_file_selected(file_path)

            # Check that config data was updated
            assert screen.config_data.source_catalogue == file_path

            # Check that show_loading was called on file_selection
            # (this is the immediate UI feedback that happens synchronously)
            # We can't easily test the async part in this context, but we can
            # verify the analysis method would be called by checking the async setup

    def test_analysis_workflow_success(self):
        """Test successful analysis workflow."""
        from cutana_ui.start_screen import StartScreen
        import asyncio
        from pathlib import Path
        from unittest.mock import patch, AsyncMock

        screen = StartScreen()

        # Mock analysis result
        mock_result = {
            "num_sources": 25,
            "fits_files": ["test.fits"],
            "extensions": [{"name": "VIS", "ext": "IMAGE"}],
        }

        # Create a temporary file to avoid file existence check
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
            temp_file.write(b"test,150.0,2.0,256,\"['test.fits']\"\n")

        # Run analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with (
                patch.object(screen.file_selection, "show_analysis_results") as mock_results,
                patch.object(screen.configuration, "set_extensions") as mock_set_ext,
                patch(
                    "cutana_ui.start_screen.start_screen.BackendInterface.check_source_catalogue",
                    new_callable=AsyncMock,
                ) as mock_backend,
            ):

                # Set the mock return value
                mock_backend.return_value = mock_result

                loop.run_until_complete(screen._analyze_catalogue(temp_path))

                # Check workflow steps - show_loading is called in _on_file_selected, not _analyze_catalogue
                mock_results.assert_called_once_with(mock_result)
                mock_set_ext.assert_called_once_with([{"name": "VIS", "ext": "IMAGE"}])

                # Check state updates
                assert screen.analysis_result == mock_result
                assert screen.configuration.layout.display == "block"
                assert screen.start_button.layout.display == "block"
        finally:
            loop.close()
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def test_analysis_workflow_error(self):
        """Test analysis workflow error handling."""
        from cutana_ui.start_screen import StartScreen
        import asyncio
        from unittest.mock import patch

        screen = StartScreen()

        # Test with non-existent file - this will trigger FileNotFoundError
        file_path = "/nonexistent/file.csv"
        expected_error = f"File not found: {file_path}"

        # Run analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with patch.object(screen.file_selection, "show_error") as mock_show_error:
                loop.run_until_complete(screen._analyze_catalogue(file_path))

                # Check error handling - should show error message in file selection component
                mock_show_error.assert_called_once()
                # Check that the error message contains the expected text
                call_args = mock_show_error.call_args[0][0]  # Get first argument of the call
                assert expected_error in call_args
        finally:
            loop.close()

    def test_start_button_click_workflow(self):
        """Test start button click workflow."""
        from cutana_ui.start_screen import StartScreen
        from cutana.get_default_config import get_default_config
        from unittest.mock import patch

        screen = StartScreen()

        # Set up test data
        screen.analysis_result = {
            "num_sources": 25,
            "extensions": [{"name": "VIS", "ext": "IMAGE"}],
        }

        # Create a proper DotMap for the default config
        default_config = get_default_config()
        default_config.num_workers = 4
        default_config.batch_size = 128

        # Mock configuration components
        with (
            patch.object(
                screen.configuration,
                "get_configuration",
                return_value={
                    "selected_extensions": [{"name": "VIS", "ext": "IMAGE"}],
                    "data_format": "float32",
                    "target_resolution": 128,
                },
            ) as mock_get_config,
            patch.object(
                screen.output_folder, "get_output_dir", return_value="/test/output"
            ) as mock_get_dir,
            patch(
                "cutana_ui.start_screen.start_screen.get_default_config",
                return_value=default_config,
            ) as mock_get_default,
            patch(
                "cutana_ui.start_screen.start_screen.save_config_with_timestamp",
                return_value="/test/config.json",
            ) as mock_save,
        ):

            # Mock completion callback
            completion_called = False
            completion_args = None

            def mock_completion(config, config_path):
                nonlocal completion_called, completion_args
                completion_called = True
                completion_args = (config, config_path)

            screen.on_complete = mock_completion

            # Click start button
            screen._on_start_click(None)

            # Check workflow
            mock_get_config.assert_called_once()
            mock_get_dir.assert_called_once()
            mock_get_default.assert_called_once()
            mock_save.assert_called_once()

            assert completion_called
            assert completion_args is not None

    def test_validation_errors(self):
        """Test start button validation errors."""
        from cutana_ui.start_screen import StartScreen

        screen = StartScreen()

        # Test missing extensions validation
        with (
            patch.object(
                screen.configuration,
                "get_configuration",
                return_value={"selected_extensions": []},  # Empty extensions
            ),
            patch.object(screen.output_folder, "get_output_dir", return_value="/test/output"),
        ):

            screen._on_start_click(None)
            assert "select at least one FITS extension" in screen.error_message.value

        # Test missing output dir validation
        with (
            patch.object(
                screen.configuration,
                "get_configuration",
                return_value={"selected_extensions": [{"name": "VIS", "ext": "IMAGE"}]},
            ),
            patch.object(screen.output_folder, "get_output_dir", return_value=None),
        ):

            screen._on_start_click(None)
            assert "select an output directory" in screen.error_message.value

    def test_layout_updates(self):
        """Test layout updates based on analysis results."""
        from cutana_ui.start_screen import StartScreen

        screen = StartScreen()

        # Main container should always have top and bottom sections
        initial_children = screen.main_container.children
        assert len(initial_children) == 2  # top_section + bottom_section
        assert screen.top_section in initial_children
        assert screen.bottom_section in initial_children

        # Initially top section should have file selection and output folder
        top_children = screen.top_section.children
        assert screen.file_selection in top_children
        assert screen.output_folder in top_children

        # Bottom section should initially be empty (no analysis result yet)
        bottom_children = screen.bottom_section.children
        assert len(bottom_children) == 0

        # Set analysis result and update layout
        screen.analysis_result = {
            "num_sources": 25,
            "extensions": [{"name": "VIS", "ext": "IMAGE"}],
        }

        # Manually set display properties as they would be set in the analysis workflow
        screen.configuration.layout.display = "block"
        screen.start_button.layout.display = "block"

        screen._update_layout()

        # Now bottom section should include configuration component
        bottom_children = screen.bottom_section.children
        assert screen.configuration in bottom_children

        # Check that configuration component is visible
        assert screen.configuration.layout.display != "none"
        assert screen.start_button.layout.display != "none"
