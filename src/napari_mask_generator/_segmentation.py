import numpy as np
from skimage import filters, morphology, measure, segmentation, feature
from scipy import ndimage as ndi
from typing import List


def segment_cells(
        image: np.ndarray,
        roi_mask: np.ndarray,
        threshold_method: str = "otsu",
        min_cell_area: int = 200,
        max_cell_area: int = 10000,
) -> np.ndarray:
    """
    Segment cells within an ROI mask using thresholding + watershed.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed grayscale image.
    roi_mask : np.ndarray
        Binary mask defining the region of interest.
    threshold_method : str
        Either 'otsu' or 'adaptive'.
    min_cell_area : int
        Minimum cell area in pixels.
    max_cell_area : int
        Maximum cell area in pixels.

    Returns
    -------
    np.ndarray
        Labeled array where each cell has a unique integer ID.
    """
    # Apply ROI mask
    masked_image = image * roi_mask

    # --- Thresholding ---
    if threshold_method == "otsu":
        thresh = filters.threshold_otsu(masked_image[roi_mask > 0])
        binary = masked_image > thresh
    elif threshold_method == "adaptive":
        thresh = filters.threshold_local(masked_image, block_size=51)
        binary = masked_image > thresh
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")

    binary = binary & roi_mask.astype(bool)

    # --- Morphological cleanup ---
    binary = morphology.remove_small_objects(binary, min_size=min_cell_area)
    binary = morphology.remove_small_holes(binary, area_threshold=500)
    binary = morphology.binary_closing(binary, morphology.disk(3))

    # --- Watershed to separate touching cells ---
    distance = ndi.distance_transform_edt(binary)
    distance_smooth = filters.gaussian(distance, sigma=2)

    local_max_coords = feature.peak_local_max(
        distance_smooth,
        min_distance=15,
        labels=binary
    )
    local_max_mask = np.zeros_like(binary, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)

    labels = segmentation.watershed(-distance_smooth, markers, mask=binary)

    # --- Filter by area ---
    filtered_labels = np.zeros_like(labels)
    for region in measure.regionprops(labels):
        if min_cell_area <= region.area <= max_cell_area:
            filtered_labels[labels == region.label] = region.label

    return filtered_labels


def labels_to_polygons(
        labels: np.ndarray,
        tolerance: float = 1.0
) -> List[np.ndarray]:
    """
    Convert labeled regions to polygon contours for napari Shapes layer.

    Parameters
    ----------
    labels : np.ndarray
        Labeled array from segment_cells.
    tolerance : float
        Polygon approximation tolerance. Higher = fewer vertices.

    Returns
    -------
    List[np.ndarray]
        List of (N, 2) arrays in (row, col) format.
    """
    from skimage.measure import approximate_polygon

    polygons = []
    for region in measure.regionprops(labels):
        single_mask = labels == region.label
        contours = measure.find_contours(single_mask, level=0.5)
        if contours:
            contour = max(contours, key=len)
            approx = approximate_polygon(contour, tolerance=tolerance)
            polygons.append(approx)
    return polygons