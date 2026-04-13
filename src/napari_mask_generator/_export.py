import json
import csv
import numpy as np
import tifffile
from datetime import datetime
from typing import Dict
from pathlib import Path


# ------------------------------------------------------------------
# Directory Structure
# ------------------------------------------------------------------

def get_output_dirs(base_dir: Path) -> Dict[str, Path]:
    """
    Create and return all output subdirectories.
    All named independently of the chosen filename
    so all images share the same folder structure.
    """
    dirs = {
        "instance":         base_dir / "instance_masks",
        "semantic":         base_dir / "semantic_masks",
        "boundaries":       base_dir / "boundary_masks",
        "overlay":          base_dir / "boundary_overlays",
        "cellpose":         base_dir / "cellpose",
        "coco":             base_dir / "coco_annotations",
        "statistics":       base_dir / "statistics",
        "session":          base_dir / "sessions",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ------------------------------------------------------------------
# Mask Exports
# ------------------------------------------------------------------

def export_instance_tiff(
        instance_mask: np.ndarray,
        path: str
):
    """Each cell = unique integer ID. Background = 0."""
    path = str(Path(path).with_suffix(".tiff"))
    tifffile.imwrite(path, instance_mask.astype(np.uint16))
    print(f"Saved instance mask:    {path}")


def export_semantic_tiff(
        instance_mask: np.ndarray,
        path: str
):
    """Binary mask: cell=1, background=0."""
    path     = str(Path(path).with_suffix(".tiff"))
    semantic = (instance_mask > 0).astype(np.uint8)
    tifffile.imwrite(path, semantic)
    print(f"Saved semantic mask:    {path}")


def export_boundary_tiff(
        instance_mask: np.ndarray,
        path: str
):
    """Boundary pixels = 255, everything else = 0."""
    from skimage.segmentation import find_boundaries
    path     = str(Path(path).with_suffix(".tiff"))
    boundary = (
        find_boundaries(instance_mask, mode="outer")
        .astype(np.uint8) * 255
    )
    tifffile.imwrite(path, boundary)
    print(f"Saved boundary mask:    {path}")


def export_boundary_overlay_tiff(
        instance_mask: np.ndarray,
        image: np.ndarray = None,
        path: str = "boundary_overlay.tiff"
):
    """Original image with cyan boundary overlay."""
    from skimage.segmentation import find_boundaries

    path = str(Path(path).with_suffix(".tiff"))

    if image is None:
        print("No image for overlay - skipping")
        return

    # Handle (C, H, W) multichannel
    if image.ndim == 3 and image.shape[0] in (1, 2, 3, 4):
        display = image.max(axis=0).astype(float)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        from skimage.color import rgb2gray
        display = rgb2gray(image).astype(float)
    else:
        display = image.squeeze().astype(float)

    # Normalise to 0-255
    dmin, dmax = display.min(), display.max()
    if dmax > dmin:
        display = ((display - dmin) / (dmax - dmin) * 255)
    display = display.astype(np.uint8)

    # Build RGB
    rgb            = np.stack([display, display, display], axis=-1)
    boundaries     = find_boundaries(instance_mask, mode="outer")
    rgb[boundaries, 0] = 0
    rgb[boundaries, 1] = 255
    rgb[boundaries, 2] = 255

    tifffile.imwrite(path, rgb)
    print(f"Saved boundary overlay: {path}")


# ------------------------------------------------------------------
# DL Training Formats
# ------------------------------------------------------------------

def export_coco_json(
        instance_mask: np.ndarray,
        accepted_masks: Dict[int, np.ndarray],
        image_shape: tuple,
        image_filename: str = "image.tiff",
        path: str = "annotations.json"
):
    """COCO instance segmentation format."""
    path = str(Path(path).with_suffix(".json"))
    H, W = image_shape[:2]

    coco = {
        "info": {
            "description": "Cell mask annotations",
            "date_created": datetime.now().isoformat(),
            "version": "1.0"
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "cell", "supercategory": "cell"}
        ],
        "images": [
            {
                "id": 1,
                "file_name": image_filename,
                "height": H,
                "width": W
            }
        ],
        "annotations": []
    }

    for cell_id, polygon_data in accepted_masks.items():
        coords       = polygon_data[:, :2]
        segmentation = [
            float(v) for pt in coords
            for v in (pt[1], pt[0])
        ]

        cell_mask = instance_mask == cell_id
        if not np.any(cell_mask):
            continue

        rows = np.any(cell_mask, axis=1)
        cols = np.any(cell_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        coco["annotations"].append({
            "id":           int(cell_id),
            "image_id":     1,
            "category_id":  1,
            "segmentation": [segmentation],
            "area":         float(np.sum(cell_mask)),
            "bbox": [
                float(cmin), float(rmin),
                float(cmax - cmin), float(rmax - rmin)
            ],
            "iscrowd": 0
        })

    with open(path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved COCO JSON:        {path} "
          f"({len(accepted_masks)} cells)")


def export_cellpose_npy(
        instance_mask: np.ndarray,
        image: np.ndarray = None,
        path: str = "cellpose_masks.npy"
):
    """Cellpose training format."""
    from skimage.segmentation import find_boundaries
    path     = str(Path(path).with_suffix(".npy"))
    outlines = find_boundaries(instance_mask, mode="outer")
    data     = {
        "masks":    instance_mask.astype(np.uint16),
        "outlines": outlines,
    }
    if image is not None:
        data["img"] = image
    np.save(path, data, allow_pickle=True)
    print(f"Saved Cellpose .npy:    {path}")


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def export_cell_statistics_csv(
        instance_mask: np.ndarray,
        accepted_masks: Dict[int, np.ndarray],
        path: str
):
    """
    Per-cell statistics CSV.
    One row per cell with all measurements.
    """
    from skimage.measure import regionprops

    path  = str(Path(path).with_suffix(".csv"))
    props = {
        r.label: r
        for r in regionprops(instance_mask.astype(np.int32))
    }

    rows = []
    for cell_id, polygon_data in sorted(accepted_masks.items()):
        row = {"cell_id": cell_id}

        if cell_id in props:
            p    = props[cell_id]
            area = int(p.area)

            row["pixel_area"]          = area
            row["centroid_row"]        = round(p.centroid[0], 2)
            row["centroid_col"]        = round(p.centroid[1], 2)
            row["bbox_row_min"]        = int(p.bbox[0])
            row["bbox_col_min"]        = int(p.bbox[1])
            row["bbox_row_max"]        = int(p.bbox[2])
            row["bbox_col_max"]        = int(p.bbox[3])
            row["bbox_height_px"]      = int(p.bbox[2] - p.bbox[0])
            row["bbox_width_px"]       = int(p.bbox[3] - p.bbox[1])
            row["equivalent_diameter"] = round(
                p.equivalent_diameter_area, 2
            )
            row["eccentricity"]        = round(p.eccentricity, 4)
            row["major_axis_length"]   = round(
                p.major_axis_length, 2
            )
            row["minor_axis_length"]   = round(
                p.minor_axis_length, 2
            )
            row["aspect_ratio"]        = round(
                p.major_axis_length / p.minor_axis_length, 4
            ) if p.minor_axis_length > 0 else None

            h_bb = p.bbox[2] - p.bbox[0]
            w_bb = p.bbox[3] - p.bbox[1]
            row["bbox_aspect_ratio"]   = round(
                w_bb / h_bb, 4
            ) if h_bb > 0 else None

        else:
            for key in [
                "pixel_area", "centroid_row", "centroid_col",
                "bbox_row_min", "bbox_col_min",
                "bbox_row_max", "bbox_col_max",
                "bbox_height_px", "bbox_width_px",
                "equivalent_diameter", "eccentricity",
                "major_axis_length", "minor_axis_length",
                "aspect_ratio", "bbox_aspect_ratio"
            ]:
                row[key] = None

        # Polygon info
        row["n_polygon_vertices"] = len(polygon_data)
        coords    = polygon_data[:, :2]
        diffs     = np.diff(coords, axis=0)
        perimeter = float(np.sum(np.sqrt((diffs**2).sum(axis=1))))
        close     = coords[-1] - coords[0]
        perimeter += float(np.sqrt((close**2).sum()))
        row["polygon_perimeter_px"] = round(perimeter, 2)

        rows.append(row)

    if not rows:
        print("No cells to export.")
        return []

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved statistics CSV:   {path}")
    return rows


def export_summary_json(
        instance_mask: np.ndarray,
        accepted_masks: Dict[int, np.ndarray],
        stats_rows: list,
        path: str
):
    """
    Overall summary statistics as JSON.
    Contains dataset-level aggregates not per-cell data.
    """
    path = str(Path(path).with_suffix(".json"))

    areas = [
        r["pixel_area"] for r in stats_rows
        if r.get("pixel_area") is not None
    ]
    eccentricities = [
        r["eccentricity"] for r in stats_rows
        if r.get("eccentricity") is not None
    ]
    major_axes = [
        r["major_axis_length"] for r in stats_rows
        if r.get("major_axis_length") is not None
    ]
    minor_axes = [
        r["minor_axis_length"] for r in stats_rows
        if r.get("minor_axis_length") is not None
    ]

    summary = {
        "annotation_info": {
            "date_created":       datetime.now().isoformat(),
            "image_shape":        list(instance_mask.shape),
            "total_cells":        len(accepted_masks),
            "total_annotated_px": int(np.sum(instance_mask > 0)),
            "background_px":      int(np.sum(instance_mask == 0)),
            "coverage_fraction":  round(
                float(np.sum(instance_mask > 0)) /
                float(instance_mask.size), 4
            ),
        },
        "area_statistics_px": {
            "min":    int(min(areas))           if areas else None,
            "max":    int(max(areas))           if areas else None,
            "mean":   round(np.mean(areas), 2)  if areas else None,
            "median": round(np.median(areas), 2) if areas else None,
            "std":    round(np.std(areas), 2)   if areas else None,
            "total":  int(sum(areas))           if areas else None,
        },
        "shape_statistics": {
            "mean_eccentricity":     round(
                np.mean(eccentricities), 4
            ) if eccentricities else None,
            "mean_major_axis_px":    round(
                np.mean(major_axes), 2
            ) if major_axes else None,
            "mean_minor_axis_px":    round(
                np.mean(minor_axes), 2
            ) if minor_axes else None,
            "mean_aspect_ratio":     round(
                np.mean(major_axes) / np.mean(minor_axes), 4
            ) if (major_axes and minor_axes and
                  np.mean(minor_axes) > 0) else None,
        },
        "cell_ids": sorted(list(accepted_masks.keys())),
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary JSON:     {path}")
    return summary