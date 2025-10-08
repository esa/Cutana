#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Zarr cutout writer module for Cutana - handles zarr archive output using images-to-zarr.

This module provides static functions for:
- Direct memory-to-zarr conversion using images-to-zarr convert function
- Incremental sub-batch writing with append mode
- Smart chunking to stay below 2GB limit
- Compression and chunking strategies
- No temporary files - writes directly from memory to zarr
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from images_to_zarr import convert
from dotmap import DotMap


def generate_process_subfolder(process_id: str) -> str:
    """
    Generate a unique subfolder name for a specific process.

    Args:
        process_id: Unique process identifier (e.g., "cutout_process_000_unique_id")

    Returns:
        Subfolder name (e.g., "batch_cutout_process_000_unique_id")
    """
    return f"batch_{process_id}"


def calculate_optimal_chunk_shape(
    n_sources: int,
    height: int,
    width: int,
    n_channels: int,
    dtype: np.dtype,
    max_chunk_size_gb: float = 1.8,
) -> Tuple[int, int, int, int]:
    """
    Calculate optimal chunk shape to stay below 2GB limit.

    Args:
        n_sources: Number of sources in the batch
        height: Image height
        width: Image width
        n_channels: Number of channels
        dtype: Data type
        max_chunk_size_gb: Maximum chunk size in GB (default 1.8 to stay safely below 2GB)

    Returns:
        Tuple of (chunk_n_sources, chunk_height, chunk_width, chunk_channels)
    """
    # Calculate bytes per element
    bytes_per_element = dtype.itemsize

    # Calculate total size per image
    bytes_per_image = height * width * n_channels * bytes_per_element

    # Convert max size to bytes
    max_chunk_size_bytes = max_chunk_size_gb * 1024 * 1024 * 1024

    # Calculate maximum number of images per chunk
    max_images_per_chunk = int(max_chunk_size_bytes / bytes_per_image)

    # Ensure at least 1 image per chunk
    chunk_n_sources = min(max(1, max_images_per_chunk), n_sources)

    # If a single image is too large, we need to chunk spatially
    if chunk_n_sources == 1 and bytes_per_image > max_chunk_size_bytes:
        # Calculate how much we need to reduce spatial dimensions
        reduction_factor = np.sqrt(bytes_per_image / max_chunk_size_bytes)

        # Apply reduction to both dimensions equally
        chunk_height = max(1, int(height / reduction_factor))
        chunk_width = max(1, int(width / reduction_factor))
    else:
        # Use full spatial dimensions
        chunk_height = height
        chunk_width = width

    logger.debug(
        f"Calculated chunk shape: ({chunk_n_sources}, {chunk_height}, {chunk_width}, {n_channels}) "
        f"for images of shape ({n_sources}, {height}, {width}, {n_channels}) "
        f"with dtype {dtype}"
    )

    return (chunk_n_sources, chunk_height, chunk_width, n_channels)


def prepare_cutouts_for_zarr(
    batch_data: Dict[str, Any],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Prepare cutout data for zarr conversion by organizing into 4D NCHW format.
    Handles single batch result from sub-batch processing.

    Args:
        batch_data: Single batch result containing cutouts tensor and metadata

    Returns:
        Tuple of (images_array, metadata_list)
        - images_array: 4D numpy array in NCHW format (N sources, C channels, H height, W width)
    """
    try:
        if not batch_data:
            logger.warning("No batch data provided")
            return np.array([]), []

        cutouts_tensor = batch_data.get("cutouts")  # Shape: (N, H, W, C)
        metadata = batch_data.get("metadata", [])

        if cutouts_tensor is None or cutouts_tensor.size == 0:
            logger.warning("No valid cutouts in batch result")
            return np.array([]), []

        # Convert from NHWC to NCHW format for zarr
        if len(cutouts_tensor.shape) == 4:  # (N, H, W, C)
            images_array = np.transpose(cutouts_tensor, (0, 3, 1, 2))  # (N, C, H, W)
        else:
            logger.error(f"Unexpected cutouts tensor shape: {cutouts_tensor.shape}")
            return np.array([]), []

        logger.info(
            f"Prepared {images_array.shape[0]} sources for zarr conversion (shape: {images_array.shape})"
        )
        return images_array, metadata

    except Exception as e:
        logger.error(f"Failed to prepare cutouts for zarr: {e}")
        raise


def create_zarr_from_memory(
    images: np.ndarray,
    metadata: List[Dict[str, Any]],
    output_path: str,
    config: DotMap,
    append: bool = False,
) -> str:
    """
    Create or append to zarr archive directly from images in memory using images_to_zarr.

    Args:
        images: 4D numpy array in NCHW format (samples, channels, height, width)
        metadata: List of original metadata for each image
        output_path: Full path to zarr archive (including images.zarr)
        config: Configuration DotMap
        append: If True, append to existing zarr archive

    Returns:
        Path to created/updated zarr archive
    """
    try:
        # Ensure parent directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Validate image array format
        if len(images.shape) != 4:
            raise ValueError(f"Images must be 4D NCHW format, got shape: {images.shape}")

        # Convert NCHW to NHWC format expected by images_to_zarr
        images_formatted = np.transpose(images, (0, 2, 3, 1))  # (N, H, W, C)

        # Get data type for chunk calculation
        dtype = images_formatted.dtype

        # Calculate optimal chunk shape to stay below 2GB
        n_sources, height, width, n_channels = images_formatted.shape
        chunk_n_sources, chunk_height, chunk_width, chunk_channels = calculate_optimal_chunk_shape(
            n_sources, height, width, n_channels, dtype
        )

        # Use lz4 compression as it's more compatible with large chunks
        # Use images_to_zarr convert function with original metadata
        zarr_path = convert(
            output_dir=str(output_path_obj.parent),  # Parent directory
            images=images_formatted,
            image_metadata=metadata,  # Pass metadata list directly
            chunk_shape=(chunk_n_sources, chunk_height, chunk_width, chunk_channels),
            compressor="lz4",  # LZ4 is more compatible with larger chunks
            overwrite=not append,  # Don't overwrite if appending
            append=append,  # Enable append mode
            num_parallel_workers=1,  # Parallelism happens on cutout_process level
        )

        action = "Appended to" if append else "Created"
        logger.info(f"{action} zarr archive: {zarr_path} (shape: {images.shape})")
        return str(zarr_path)

    except Exception as e:
        logger.error(f"Failed to create/append zarr from memory: {e}")
        raise


def create_process_zarr_archive_initial(
    batch_data: Dict[str, Any],
    output_path: str,
    config: DotMap,
) -> Optional[str]:
    """
    Create initial zarr archive for the first sub-batch.

    Args:
        batch_data: First sub-batch cutout data dictionary
        output_path: Full path to zarr archive
        config: Configuration DotMap

    Returns:
        Path to created zarr archive, or None if failed
    """
    try:
        if not batch_data:
            logger.warning("No batch data provided for initial zarr creation")
            return None

        # Prepare cutouts as 4D NCHW array
        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        if images_array.size == 0:
            logger.warning("No valid cutouts to process for initial zarr")
            return None

        # Create initial zarr archive
        zarr_path = create_zarr_from_memory(
            images=images_array,
            metadata=metadata_list,
            output_path=output_path,
            config=config,
            append=False,  # Create new archive
        )

        logger.info(f"Created initial zarr archive: {zarr_path}")
        return zarr_path

    except Exception as e:
        logger.error(f"Failed to create initial zarr archive: {e}")
        return None


def append_to_zarr_archive(
    batch_data: Dict[str, Any],
    output_path: str,
    config: DotMap,
) -> Optional[str]:
    """
    Append a sub-batch to an existing zarr archive.

    Args:
        batch_data: Sub-batch cutout data dictionary
        output_path: Full path to existing zarr archive
        config: Configuration DotMap

    Returns:
        Path to updated zarr archive, or None if failed
    """
    try:
        if not batch_data:
            logger.warning("No batch data provided for zarr append")
            return None

        # Prepare cutouts as 4D NCHW array
        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        if images_array.size == 0:
            logger.warning("No valid cutouts to append to zarr")
            return None

        # Append to existing zarr archive
        zarr_path = create_zarr_from_memory(
            images=images_array,
            metadata=metadata_list,
            output_path=output_path,
            config=config,
            append=True,  # Append to existing archive
        )

        logger.info(f"Appended to zarr archive: {zarr_path}")
        return zarr_path

    except Exception as e:
        logger.error(f"Failed to append to zarr archive: {e}")
        return None
