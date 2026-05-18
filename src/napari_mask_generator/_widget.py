import numpy as np
from pathlib import Path
from napari import Viewer
from napari.layers import Image, Shapes, Labels, Points
from magicgui.widgets import (
    Container, PushButton, Label,
    LineEdit, CheckBox
)
from qtpy.QtWidgets import QSizePolicy


class MaskGenWidget(Container):
    def __init__(
            self,
            viewer: Viewer,
            default_filename: str = "cell_masks",
            default_save_folder: str = str(Path.home())
    ):
        super().__init__()
        self._viewer          = viewer
        self._accepted_masks  = {}
        self._uncertain_masks = {}
        self._boundary_masks  = {}
        self._next_cell_id    = 1
        self._image_shape     = None
        self._ignore_shape_events = False

        self.native.setMaximumWidth(280)
        self.native.setMinimumWidth(220)
        self.native.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Preferred
        )

        self._status_label = Label(
            value="Open image then click Start Drawing"
        )
        self._status_label.native.setWordWrap(True)
        self._status_label.native.setMaximumWidth(260)

        self._counter_label = Label(
            value="Label ID: 1  |  Accepted: 0"
        )
        self._counter_label.native.setMaximumWidth(260)

        self._filename_input = LineEdit(
            label="Filename",
            value=default_filename
        )
        self._filename_input.native.setMaximumWidth(260)

        self._save_dir_input = LineEdit(
            label="Save folder",
            value=default_save_folder
        )
        self._save_dir_input.native.setMaximumWidth(260)

        self._zoom_checkbox = CheckBox(
            label="Zoom to cell on accept",
            value=False
        )
        self._zoom_checkbox.native.setMaximumWidth(260)

        self._labels_checkbox = CheckBox(
            label="Show cell name labels",
            value=False
        )
        self._labels_checkbox.native.setMaximumWidth(260)

        self._start_button     = PushButton(label="Start Drawing")
        self._accept_button    = PushButton(label="Accept Polygon")
        self._reject_button    = PushButton(label="Reject Polygon")
        self._uncertain_button = PushButton(
            label="Uncertain (flag cell)"
        )
        self._boundary_button  = PushButton(
            label="Boundary Cell (exclude)"
        )
        self._undo_button      = PushButton(label="Undo Last")
        self._save_button      = PushButton(label="Save Masks")
        self._load_button      = PushButton(label="Load Session")

        for btn in [
            self._start_button,     self._accept_button,
            self._reject_button,    self._uncertain_button,
            self._boundary_button,  self._undo_button,
            self._save_button,      self._load_button,
        ]:
            btn.native.setMaximumWidth(260)

        self._accept_button.enabled    = False
        self._reject_button.enabled    = False
        self._uncertain_button.enabled = False
        self._boundary_button.enabled  = False
        self._undo_button.enabled      = False
        self._save_button.enabled      = False

        self.extend([
            self._status_label,
            self._counter_label,
            self._filename_input,
            self._save_dir_input,
            self._zoom_checkbox,
            self._labels_checkbox,
            self._start_button,
            self._accept_button,
            self._reject_button,
            self._uncertain_button,
            self._boundary_button,
            self._undo_button,
            self._save_button,
            self._load_button,
        ])

        for btn, callback in [
            (self._start_button,     self._start_drawing),
            (self._accept_button,    self._accept_polygon),
            (self._reject_button,    self._reject_polygon),
            (self._uncertain_button, self._mark_uncertain),
            (self._boundary_button,  self._mark_boundary),
            (self._undo_button,      self._undo_last),
            (self._save_button,      self._save_masks),
            (self._load_button,      self._load_session),
        ]:
            try:
                btn.clicked.disconnect()
            except Exception:
                pass
            btn.clicked.connect(callback)

        try:
            self._labels_checkbox.changed.disconnect()
        except Exception:
            pass
        self._labels_checkbox.changed.connect(
            self._update_cell_labels
        )

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------
    @property
    def _save_path(self) -> Path:
        folder = Path(self._save_dir_input.value.strip())
        name   = self._filename_input.value.strip() or "cell_masks"
        return folder / name

    def _update_counter(self):
        self._counter_label.value = (
            f"Label ID: {self._next_cell_id}  |  "
            f"Accepted: {len(self._accepted_masks)}"
        )

    def _get_image_shape(self, image_layer) -> tuple:
        shape = image_layer.data.shape
        ndim  = len(shape)
        print(f"Raw image shape: {shape}")
        if ndim == 2:
            h, w = shape
        elif ndim == 3:
            if shape[2] in (3, 4):
                h, w = shape[0], shape[1]
            else:
                h, w = shape[1], shape[2]
        else:
            h, w = shape[-2], shape[-1]
        print(f"2D mask shape: ({h}, {w})")
        return (h, w)

    # ------------------------------------------------------------------
    # Mask building
    # ------------------------------------------------------------------
    def _build_mask(self) -> np.ndarray:
        from skimage.draw import polygon as skimage_polygon
        h, w = self._image_shape
        mask = np.zeros((h, w), dtype=np.int32)
        for cell_id, poly in self._accepted_masks.items():
            coords      = poly[:, :2].astype(int)
            coords[:,0] = np.clip(coords[:,0], 0, h - 1)
            coords[:,1] = np.clip(coords[:,1], 0, w - 1)
            try:
                rr, cc = skimage_polygon(
                    coords[:,0], coords[:,1], (h, w)
                )
                if len(rr) > 0:
                    mask[rr, cc] = cell_id
            except Exception as e:
                print(f"Rasterise error cell {cell_id}: {e}")
        return mask

    def _refresh_labels_data(self):
        """
        Update the labels layer data IN PLACE.
        Does NOT remove/re-add the layer - preserves
        layer order and Shapes layer active state.
        """
        if "Instance Masks" not in self._viewer.layers:
            return
        self._viewer.layers["Instance Masks"].data = (
            self._build_mask()
        )

    def _update_labels_layer(self):
        """
        Full remove and re-add of labels layer.
        Only used during _start_drawing and _load_session.
        During annotation use _refresh_labels_data() instead.
        """
        new_mask = self._build_mask()
        if "Instance Masks" in self._viewer.layers:
            self._viewer.layers.remove("Instance Masks")
        self._viewer.add_labels(
            new_mask,
            name="Instance Masks",
            opacity=0.7
        )
        if ("Draw Polygons Here" in self._viewer.layers and
                "Instance Masks" in self._viewer.layers):
            li = self._viewer.layers.index("Instance Masks")
            di = self._viewer.layers.index("Draw Polygons Here")
            if li > di:
                self._viewer.layers.move(li, di)
        print(f"Labels layer updated: "
              f"unique={np.unique(new_mask)}")

    # ------------------------------------------------------------------
    # Cell label points layer
    # ------------------------------------------------------------------
    def _update_cell_labels(self):
        from skimage.measure import regionprops
        if "Cell Labels" in self._viewer.layers:
            self._viewer.layers.remove("Cell Labels")
        if (not self._labels_checkbox.value or
                not self._accepted_masks):
            return
        mask  = self._build_mask()
        props = {
            r.label: r
            for r in regionprops(mask.astype(np.int32))
        }
        if not props:
            return
        centroids   = []
        text_labels = []
        text_colors = []
        for cell_id in sorted(self._accepted_masks.keys()):
            if cell_id in props:
                p = props[cell_id]
                centroids.append([p.centroid[0], p.centroid[1]])
                text_labels.append(f"Cell {cell_id}")
                if cell_id in self._boundary_masks:
                    text_colors.append("pink")
                elif cell_id in self._uncertain_masks:
                    text_colors.append("orange")
                else:
                    text_colors.append("white")
        if not centroids:
            return
        self._viewer.add_points(
            np.array(centroids),
            name="Cell Labels",
            text={
                "string": text_labels,
                "size":   12,
                "color":  text_colors,
                "anchor": "center",
            },
            size=0,
            opacity=1.0,
        )
        n  = len(self._viewer.layers)
        li = self._viewer.layers.index("Cell Labels")
        if li < n - 1:
            self._viewer.layers.move(li, n - 1)
        # Re-activate shapes layer after labels update
        if "Draw Polygons Here" in self._viewer.layers:
            self._viewer.layers.selection.active = (
                self._viewer.layers["Draw Polygons Here"]
            )

    # ------------------------------------------------------------------
    # Zoom to last accepted cell
    # ------------------------------------------------------------------
    def _zoom_to_last_accepted(self):
        if not self._zoom_checkbox.value or \
                not self._accepted_masks:
            return
        last_id      = max(self._accepted_masks.keys())
        poly         = self._accepted_masks[last_id]
        coords       = poly[:, :2]
        r_min, r_max = coords[:,0].min(), coords[:,0].max()
        c_min, c_max = coords[:,1].min(), coords[:,1].max()
        centre_r     = (r_min + r_max) / 2
        centre_c     = (c_min + c_max) / 2
        h_poly       = (r_max - r_min) * 1.5
        w_poly       = (c_max - c_min) * 1.5
        try:
            h_cv = self._viewer.window.qt_viewer.canvas.size[1]
            w_cv = self._viewer.window.qt_viewer.canvas.size[0]
            zoom = (
                min(h_cv / h_poly, w_cv / w_poly)
                if h_poly > 0 and w_poly > 0 else 5.0
            )
        except Exception:
            zoom = 5.0
        self._viewer.camera.center = (centre_r, centre_c)
        self._viewer.camera.zoom   = zoom

    # ------------------------------------------------------------------
    # Shape recolouring
    # ------------------------------------------------------------------
    def _recolour_shape(
        self,
        drawing,
        index: int,
        edge_rgba: list,
        face_rgba: list,
        edge_width: float = 2.0
    ):
        """
        Recolour a single accepted shape using selection-based
        setters, then reset current_* back to drawing defaults
        so the next polygon drawn is invisible with no fill.
        """
        try:
            if index < 0 or index >= len(drawing.data):
                return
            prev_selected = set(drawing.selected_data)
            # Select only this shape and recolour it
            drawing.selected_data      = {index}
            drawing.current_edge_color = edge_rgba
            drawing.current_face_color = face_rgba
            drawing.current_edge_width = edge_width
            # Deselect
            drawing.selected_data = prev_selected
        except Exception as e:
            print(f"[RECOLOUR] index={index}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # ALWAYS reset current_* back to drawing defaults
            # so the next polygon the user draws is invisible
            # with no fill until it is accepted
            try:
                drawing.current_edge_color = [0.0, 0.75, 1.0, 1.0]
                drawing.current_face_color = [0.0, 0.0,  0.0, 0.0]
                drawing.current_edge_width = 0
            except Exception as e:
                print(f"[RECOLOUR] Reset defaults failed: {e}")

    def _recolour_all_shapes(self, drawing):
        """Restore colours on all shapes after undo."""
        for i, cid in enumerate(
            sorted(self._accepted_masks.keys())
        ):
            if cid in self._boundary_masks:
                ec = [1.0, 0.4, 0.7, 1.0]
                fc = [1.0, 0.4, 0.7, 0.15]
            elif cid in self._uncertain_masks:
                ec = [1.0, 0.5, 0.0, 1.0]
                fc = [1.0, 0.5, 0.0, 0.15]
            else:
                ec = [0.0, 1.0, 0.0, 1.0]
                fc = [0.0, 1.0, 0.0, 0.15]
            self._recolour_shape(drawing, i, ec, fc, 2.0)
        # Reset drawing defaults after all recolouring
        # _recolour_shape already does this in its finally
        # block but call explicitly here for safety
        try:
            drawing.current_edge_color = [0.0, 0.75, 1.0, 1.0]
            drawing.current_face_color = [0.0, 0.0,  0.0, 0.0]
            drawing.current_edge_width = 0
        except Exception as e:
            print(f"[RECOLOUR_ALL] Reset defaults failed: {e}")

    # ------------------------------------------------------------------
    # Shape deletion listener
    # ------------------------------------------------------------------
    def _connect_shape_deletion(self):
        if "Draw Polygons Here" not in self._viewer.layers:
            return
        drawing = self._viewer.layers["Draw Polygons Here"]
        try:
            drawing.events.data.disconnect(
                self._on_shapes_data_changed
            )
        except Exception:
            pass
        drawing.events.data.connect(self._on_shapes_data_changed)
        print("[CONNECT] Shape deletion listener connected.")

    def _on_shapes_data_changed(self, event):
        if self._ignore_shape_events:
            return
        if "Draw Polygons Here" not in self._viewer.layers:
            return
        drawing      = self._viewer.layers["Draw Polygons Here"]
        n_shapes_now = len(drawing.data)
        n_accepted   = len(self._accepted_masks)
        if n_shapes_now < n_accepted:
            deleted_ids = self._find_deleted_cell_ids(drawing)
            for cid in deleted_ids:
                print(f"[DELETE] Removing cell {cid}")
                self._accepted_masks.pop(cid, None)
                self._uncertain_masks.pop(cid, None)
                self._boundary_masks.pop(cid, None)
            if deleted_ids:
                self._next_cell_id = (
                    max(self._accepted_masks.keys()) + 1
                    if self._accepted_masks else 1
                )
                self._ignore_shape_events = True
                try:
                    self._refresh_labels_data()
                    if self._labels_checkbox.value:
                        self._update_cell_labels()
                finally:
                    self._ignore_shape_events = False
                self._update_counter()
                self._status_label.value = (
                    f"Cell(s) {deleted_ids} deleted."
                )

    def _find_deleted_cell_ids(self, drawing) -> list:
        """
        Find cells whose polygons are no longer in the layer.
        Uses (n_vertices, row0, col0) fingerprint.
        Safely reads shapes one at a time to avoid the
        inhomogeneous array crash from drawing.data stacking.
        """
        current_fps = set()
        try:
            # Read shapes individually via index to avoid
            # the full array stack that crashes on mixed sizes
            for i in range(len(drawing.data)):
                try:
                    shape = drawing.data[i]
                    if shape is not None and len(shape) >= 1:
                        current_fps.add((
                            len(shape),
                            round(float(shape[0, 0]), 2),
                            round(float(shape[0, 1]), 2),
                        ))
                except Exception:
                    continue
        except Exception as e:
            print(f"[FIND_DELETED] {e}")
            return []
        return [
            cid for cid, poly in self._accepted_masks.items()
            if (
                len(poly),
                round(float(poly[0, 0]), 2),
                round(float(poly[0, 1]), 2),
            ) not in current_fps
        ]

    # ------------------------------------------------------------------
    # Start Drawing
    # ------------------------------------------------------------------
    def _start_drawing(self):
        image_layers = [
            l for l in self._viewer.layers
            if isinstance(l, Image)
        ]
        if not image_layers:
            self._status_label.value = "Open an image first!"
            return

        self._image_shape = self._get_image_shape(image_layers[0])
        h, w              = self._image_shape

        for name in [
            "Draw Polygons Here", "Instance Masks", "Cell Labels"
        ]:
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)

        self._viewer.add_labels(
            np.zeros((h, w), dtype=np.int32),
            name="Instance Masks",
            opacity=0.7
        )
        drawing = self._viewer.add_shapes(
            name="Draw Polygons Here",
            # These become the defaults for newly drawn shapes
            edge_color=[0.0, 0.75, 1.0, 1.0],  # napari blue
            face_color=[0.0, 0.0,  0.0, 0.0],  # transparent
            edge_width=0,                        # invisible until accepted
        )
        # Explicitly set current_* so every new polygon
        # inherits these until we reset them after accept
        drawing.current_edge_color = [0.0, 0.75, 1.0, 1.0]
        drawing.current_face_color = [0.0, 0.0,  0.0, 0.0]
        drawing.current_edge_width = 0

        self._viewer.layers.selection.active = drawing

        self._accepted_masks  = {}
        self._uncertain_masks = {}
        self._boundary_masks  = {}
        self._next_cell_id    = 1
        self._update_counter()

        self._start_button.enabled     = False
        self._accept_button.enabled    = True
        self._reject_button.enabled    = True
        self._uncertain_button.enabled = True
        self._boundary_button.enabled  = True
        self._save_button.enabled      = True
        self._undo_button.enabled      = False

        self._status_label.value = (
            "Select the polygon tool (P) in the toolbar.\n"
            "Draw polygon, double-click to finish.\n"
            "Then click Accept, Reject, Uncertain or Boundary."
        )
        self._connect_shape_deletion()

    # ------------------------------------------------------------------
    # Shared polygon processing
    # ------------------------------------------------------------------
    def _process_polygon(self, category: str):
        """
        Shared logic for Accept / Uncertain / Boundary.
        Reads the last drawn polygon, stores it, recolours it.
        Uses _refresh_labels_data() to update the mask in place
        so the Shapes layer is never disturbed.
        """
        from skimage.draw import polygon as skimage_polygon

        if "Draw Polygons Here" not in self._viewer.layers:
            self._status_label.value = "Click Start Drawing first!"
            return
        drawing = self._viewer.layers["Draw Polygons Here"]
        n       = len(drawing.data)

        if n == 0:
            self._status_label.value = "Draw a polygon first!"
            return

        # Read last shape safely - individual index access
        # avoids the full array stack that crashes on mixed sizes
        try:
            poly_raw = drawing.data[n - 1]
        except Exception as e:
            print(f"[PROCESS] Cannot read last shape: {e}")
            self._status_label.value = (
                "Cannot read polygon - finish drawing first\n"
                "(double-click to close)."
            )
            return

        # Validate
        try:
            poly = np.array(poly_raw, dtype=np.float64)
            if poly.ndim != 2 or poly.shape[0] < 3:
                self._status_label.value = (
                    "Finish the polygon first\n"
                    "(double-click to close)."
                )
                return
        except (ValueError, TypeError):
            self._status_label.value = (
                "Finish the polygon first\n"
                "(double-click to close)."
            )
            return

        h, w   = self._image_shape
        coords = poly[:, :2].astype(int)
        coords[:,0] = np.clip(coords[:,0], 0, h - 1)
        coords[:,1] = np.clip(coords[:,1], 0, w - 1)
        rr, cc = skimage_polygon(coords[:,0], coords[:,1], (h, w))
        if len(rr) == 0:
            self._status_label.value = (
                "Polygon has no area — draw larger."
            )
            return

        cell_id                       = self._next_cell_id
        self._accepted_masks[cell_id] = poly

        if category == "uncertain":
            self._uncertain_masks[cell_id] = poly
            ec  = [1.0, 0.5, 0.0, 1.0]
            fc  = [1.0, 0.5, 0.0, 0.15]
            tag = "uncertain (orange)"
        elif category == "boundary":
            self._boundary_masks[cell_id] = poly
            ec  = [1.0, 0.4, 0.7, 1.0]
            fc  = [1.0, 0.4, 0.7, 0.15]
            tag = "boundary (pink)"
        else:
            ec  = [0.0, 1.0, 0.0, 1.0]
            fc  = [0.0, 1.0, 0.0, 0.15]
            tag = "accepted (green)"

        # Update labels in place - does NOT touch Shapes layer
        self._ignore_shape_events = True
        try:
            self._refresh_labels_data()
            if self._labels_checkbox.value:
                self._update_cell_labels()
        finally:
            self._ignore_shape_events = False

        self._zoom_to_last_accepted()

        # Recolour using selection-based setters
        self._recolour_shape(drawing, n - 1, ec, fc, 2.0)

        # Restore Shapes layer as active
        self._viewer.layers.selection.active = drawing

        self._next_cell_id       += 1
        self._undo_button.enabled = True
        self._update_counter()
        self._connect_shape_deletion()

        self._status_label.value = (
            f"Cell {cell_id} {tag}.\n"
            f"Draw next polygon then accept/reject/flag."
        )
        print(f"[{category.upper()}] cell={cell_id} px={len(rr)}")

    # ------------------------------------------------------------------
    # Accept / Uncertain / Boundary
    # ------------------------------------------------------------------
    def _accept_polygon(self):
        self._process_polygon("certain")

    def _mark_uncertain(self):
        self._process_polygon("uncertain")

    def _mark_boundary(self):
        self._process_polygon("boundary")

    # ------------------------------------------------------------------
    # Reject
    # ------------------------------------------------------------------
    def _reject_polygon(self):
        if "Draw Polygons Here" not in self._viewer.layers:
            self._status_label.value = "Click Start Drawing first!"
            return
        drawing = self._viewer.layers["Draw Polygons Here"]
        n       = len(drawing.data)
        if n == 0:
            self._status_label.value = "No polygon to reject."
            return

        # Select last shape and remove it using the
        # public API - this is the correct napari way
        # to remove a specific shape without corrupting
        # the layer's internal state
        self._ignore_shape_events = True
        try:
            drawing.selected_data = {n - 1}
            drawing.remove_selected()
        except Exception as e:
            print(f"[REJECT] {e}")
        finally:
            self._ignore_shape_events = False

        self._viewer.layers.selection.active = drawing
        self._status_label.value = "Rejected. Draw next polygon."

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------
    def _undo_last(self):
        if not self._accepted_masks:
            self._status_label.value = "Nothing to undo."
            return

        last_id = max(self._accepted_masks.keys())
        self._accepted_masks.pop(last_id, None)
        self._uncertain_masks.pop(last_id, None)
        self._boundary_masks.pop(last_id, None)
        self._next_cell_id = last_id

        if "Draw Polygons Here" in self._viewer.layers:
            drawing = self._viewer.layers["Draw Polygons Here"]
            n       = len(drawing.data)
            if n > 0:
                self._ignore_shape_events = True
                try:
                    drawing.selected_data = {n - 1}
                    drawing.remove_selected()
                except Exception as e:
                    print(f"[UNDO] {e}")
                finally:
                    self._ignore_shape_events = False
            self._recolour_all_shapes(drawing)
            self._viewer.layers.selection.active = drawing

        self._ignore_shape_events = True
        try:
            self._refresh_labels_data()
            if self._labels_checkbox.value:
                self._update_cell_labels()
        finally:
            self._ignore_shape_events = False

        self._undo_button.enabled = len(self._accepted_masks) > 0
        self._update_counter()
        self._connect_shape_deletion()
        self._status_label.value = (
            f"Undone cell {last_id}. Draw next polygon."
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _save_masks(self):
        from ._export import (
            get_output_dirs,
            export_instance_tiff,
            export_semantic_tiff,
            export_boundary_tiff,
            export_boundary_overlay_tiff,
            export_coco_json,
            export_cellpose_npy,
            export_cell_statistics_csv,
            export_summary_json,
        )
        if not self._accepted_masks:
            self._status_label.value = "No masks to save."
            return

        instance_mask = self._build_mask().astype(np.uint16)
        image_layers  = [
            l for l in self._viewer.layers
            if isinstance(l, Image)
        ]
        image    = image_layers[0].data if image_layers else None
        base_dir = Path(self._save_dir_input.value.strip())
        filename = (
            self._filename_input.value.strip() or "cell_masks"
        )
        print(f"\n[SAVE] {filename} → {base_dir}")

        if not base_dir.exists():
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self._status_label.value = (
                    f"Cannot create folder:\n{e}"
                )
                return

        try:
            dirs = get_output_dirs(base_dir)
        except Exception as e:
            self._status_label.value = f"Dir error:\n{e}"
            return

        saved         = []
        failed        = []
        uncertain_ids = set(self._uncertain_masks.keys())
        boundary_ids  = set(self._boundary_masks.keys())

        for label, fn, args, kwargs in [
            (
                "instance_mask", export_instance_tiff,
                [instance_mask],
                {"path": str(
                    dirs["instance"] /
                    f"{filename}_instance_mask.tiff"
                )}
            ),
            (
                "semantic_mask", export_semantic_tiff,
                [instance_mask],
                {"path": str(
                    dirs["semantic"] /
                    f"{filename}_semantic_mask.tiff"
                )}
            ),
            (
                "boundary_mask", export_boundary_tiff,
                [instance_mask],
                {"path": str(
                    dirs["boundaries"] /
                    f"{filename}_boundary_mask.tiff"
                )}
            ),
            (
                "boundary_overlay", export_boundary_overlay_tiff,
                [instance_mask, image],
                {"path": str(
                    dirs["overlay"] /
                    f"{filename}_boundary_overlay.tiff"
                )}
            ),
            (
                "coco_json", export_coco_json,
                [instance_mask, self._accepted_masks],
                {
                    "image_shape":   instance_mask.shape,
                    "path":          str(
                        dirs["coco"] /
                        f"{filename}_coco_annotation.json"
                    ),
                    "uncertain_ids": uncertain_ids,
                    "boundary_ids":  boundary_ids,
                }
            ),
            (
                "cellpose", export_cellpose_npy,
                [instance_mask, image],
                {"path": str(
                    dirs["cellpose"] /
                    f"{filename}_cellpose.npy"
                )}
            ),
        ]:
            try:
                print(f"[SAVE] {label}...")
                fn(*args, **kwargs)
                saved.append(label)
            except Exception as e:
                failed.append(f"{label}: {e}")
                print(f"[SAVE ERROR] {label}: {e}")
                import traceback
                traceback.print_exc()

        stats_rows = []
        try:
            stats_rows = export_cell_statistics_csv(
                instance_mask, self._accepted_masks,
                path=str(
                    dirs["statistics"] /
                    f"{filename}_statistics.csv"
                ),
                uncertain_ids=uncertain_ids,
                boundary_ids=boundary_ids,
            )
            saved.append("statistics_csv")
        except Exception as e:
            failed.append(f"statistics_csv: {e}")
            import traceback
            traceback.print_exc()

        try:
            export_summary_json(
                instance_mask, self._accepted_masks,
                stats_rows or [],
                path=str(
                    dirs["statistics"] /
                    f"{filename}_summary.json"
                ),
                uncertain_ids=uncertain_ids,
                boundary_ids=boundary_ids,
            )
            saved.append("summary_json")
        except Exception as e:
            failed.append(f"summary_json: {e}")
            import traceback
            traceback.print_exc()

        try:
            self._save_session(str(dirs["session"] / filename))
            saved.append("session")
        except Exception as e:
            failed.append(f"session: {e}")
            import traceback
            traceback.print_exc()

        print(f"[SAVE] OK={saved}  FAIL={failed}")
        n = len(self._accepted_masks)
        if failed:
            self._status_label.value = (
                f"Partial save {len(saved)}/"
                f"{len(saved)+len(failed)} — check terminal."
            )
        else:
            self._status_label.value = (
                f"Saved {n} masks! "
                f"Certain: "
                f"{n-len(uncertain_ids)-len(boundary_ids)} | "
                f"Uncertain: {len(uncertain_ids)} | "
                f"Boundary: {len(boundary_ids)}"
            )

    # ------------------------------------------------------------------
    # Save session
    # ------------------------------------------------------------------
    def _save_session(self, save_stem: str):
        cell_ids      = np.array(
            list(self._accepted_masks.keys()), dtype=np.int32
        )
        uncertain_ids = np.array(
            list(self._uncertain_masks.keys()), dtype=np.int32
        )
        boundary_ids  = np.array(
            list(self._boundary_masks.keys()), dtype=np.int32
        )
        np.savez(
            f"{save_stem}_session.npz",
            instance_mask = self._build_mask(),
            cell_ids      = cell_ids,
            uncertain_ids = uncertain_ids,
            boundary_ids  = boundary_ids,
            next_cell_id  = np.array([self._next_cell_id]),
        )
        np.savez(
            f"{save_stem}_polygons.npz",
            **{
                f"cell_{int(cid)}": np.array(
                    self._accepted_masks[cid], dtype=np.float64
                )
                for cid in cell_ids
            }
        )
        print(f"Session saved: {save_stem} | "
              f"{len(cell_ids)} cells")

    # ------------------------------------------------------------------
    # Load session
    # ------------------------------------------------------------------
    def _load_session(self):
        image_layers = [
            l for l in self._viewer.layers
            if isinstance(l, Image)
        ]
        if not image_layers:
            self._status_label.value = "Open image first!"
            return

        base_dir          = Path(self._save_dir_input.value.strip())
        filename          = (
            self._filename_input.value.strip() or "cell_masks"
        )
        session_dir       = base_dir / "sessions"
        session_path      = session_dir / f"{filename}_session.npz"
        polygons_path_npz = session_dir / f"{filename}_polygons.npz"
        polygons_path_npy = session_dir / f"{filename}_polygons.npy"

        if not session_path.exists():
            self._status_label.value = (
                "No session found.\nCheck filename and folder."
            )
            return

        try:
            session            = np.load(
                session_path, allow_pickle=True
            )
            cell_ids           = session["cell_ids"]
            self._next_cell_id = int(session["next_cell_id"][0])
            uncertain_ids      = (
                session["uncertain_ids"]
                if "uncertain_ids" in session
                else np.array([], dtype=np.int32)
            )
            boundary_ids = (
                session["boundary_ids"]
                if "boundary_ids" in session
                else np.array([], dtype=np.int32)
            )

            if polygons_path_npz.exists():
                poly_data = np.load(
                    polygons_path_npz, allow_pickle=True
                )
                self._accepted_masks = {
                    int(cid): np.array(
                        poly_data[f"cell_{int(cid)}"],
                        dtype=np.float64
                    )
                    for cid in cell_ids
                    if f"cell_{int(cid)}" in poly_data
                }
            elif polygons_path_npy.exists():
                polygons = np.load(
                    polygons_path_npy, allow_pickle=True
                )
                self._accepted_masks = {
                    int(cid): np.array(
                        polygons[i], dtype=np.float64
                    )
                    for i, cid in enumerate(cell_ids)
                    if i < len(polygons)
                }
            else:
                self._status_label.value = (
                    "No polygon file found.\n"
                    "Check sessions folder."
                )
                return

            self._uncertain_masks = {
                int(cid): self._accepted_masks[int(cid)]
                for cid in uncertain_ids
                if int(cid) in self._accepted_masks
            }
            self._boundary_masks = {
                int(cid): self._accepted_masks[int(cid)]
                for cid in boundary_ids
                if int(cid) in self._accepted_masks
            }
            self._image_shape = self._get_image_shape(
                image_layers[0]
            )

            for name in [
                "Instance Masks",
                "Draw Polygons Here",
                "Cell Labels",
            ]:
                if name in self._viewer.layers:
                    self._viewer.layers.remove(name)

            self._viewer.add_labels(
                self._build_mask(),
                name="Instance Masks",
                opacity=0.7
            )
            drawing = self._viewer.add_shapes(
                name="Draw Polygons Here",
                edge_color=[0.0, 0.75, 1.0, 1.0],
                face_color=[0.0, 0.0,  0.0, 0.0],
                edge_width=0,
            )
            # Set drawing defaults for next polygon
            drawing.current_edge_color = [0.0, 0.75, 1.0, 1.0]
            drawing.current_face_color = [0.0, 0.0,  0.0, 0.0]
            drawing.current_edge_width = 0

            for cid in sorted(self._accepted_masks.keys()):
                poly = self._accepted_masks.get(cid)
                if poly is None:
                    continue
                try:
                    p = np.array(poly, dtype=np.float64)
                    if p.ndim != 2 or p.shape[0] < 3:
                        continue
                except Exception:
                    continue
                if cid in self._boundary_masks:
                    ec = [1.0, 0.4, 0.7, 1.0]
                    fc = [1.0, 0.4, 0.7, 0.15]
                elif cid in self._uncertain_masks:
                    ec = [1.0, 0.5, 0.0, 1.0]
                    fc = [1.0, 0.5, 0.0, 0.15]
                else:
                    ec = [0.0, 1.0, 0.0, 1.0]
                    fc = [0.0, 1.0, 0.0, 0.15]
                try:
                    drawing.add_polygons(
                        [p],
                        edge_color=ec,
                        face_color=fc,
                        edge_width=2,
                    )
                except Exception as e:
                    print(f"[LOAD] poly {cid}: {e}")

            if ("Instance Masks" in self._viewer.layers and
                    "Draw Polygons Here" in self._viewer.layers):
                li = self._viewer.layers.index("Instance Masks")
                di = self._viewer.layers.index("Draw Polygons Here")
                if li > di:
                    self._viewer.layers.move(li, di)

            self._viewer.layers.selection.active = drawing
            self._connect_shape_deletion()

            if self._labels_checkbox.value:
                self._update_cell_labels()

            self._start_button.enabled     = False
            self._accept_button.enabled    = True
            self._reject_button.enabled    = True
            self._uncertain_button.enabled = True
            self._boundary_button.enabled  = True
            self._undo_button.enabled      = True
            self._save_button.enabled      = True
            self._update_counter()

            n = len(self._accepted_masks)
            self._status_label.value = (
                f"Loaded {n} masks "
                f"({len(self._uncertain_masks)} uncertain, "
                f"{len(self._boundary_masks)} boundary).\n"
                f"Select polygon tool (P) and draw."
            )
            print(f"[LOAD] {n} cells loaded")

        except Exception as e:
            self._status_label.value = f"Load error: {e}"
            print(f"[LOAD ERROR] {e}")
            import traceback
            traceback.print_exc()