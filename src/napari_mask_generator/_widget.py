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
        self._next_cell_id    = 1
        self._image_shape     = None

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

        # --- Buttons ---
        self._start_button     = PushButton(label="Start Drawing")
        self._accept_button    = PushButton(label="Accept Polygon")
        self._reject_button    = PushButton(label="Reject Polygon")
        self._uncertain_button = PushButton(
            label="Uncertain (flag cell)"
        )
        self._undo_button      = PushButton(label="Undo Last")
        self._save_button      = PushButton(label="Save Masks")
        self._load_button      = PushButton(label="Load Session")

        for btn in [
            self._start_button,     self._accept_button,
            self._reject_button,    self._uncertain_button,
            self._undo_button,      self._save_button,
            self._load_button,
        ]:
            btn.native.setMaximumWidth(260)

        # Disable until started
        self._accept_button.enabled    = False
        self._reject_button.enabled    = False
        self._uncertain_button.enabled = False
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
            self._undo_button,
            self._save_button,
            self._load_button,
        ])

        # Connect buttons - disconnect first to prevent
        # double connections if widget is recreated
        for btn, callback in [
            (self._start_button,     self._start_drawing),
            (self._accept_button,    self._accept_polygon),
            (self._reject_button,    self._reject_polygon),
            (self._uncertain_button, self._mark_uncertain),
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
    # Properties
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
        """Correctly extract 2D (H, W) from any image type."""
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

    def _build_mask(self) -> np.ndarray:
        """Build int32 Labels mask from all accepted polygons."""
        from skimage.draw import polygon as skimage_polygon

        h, w = self._image_shape
        mask = np.zeros((h, w), dtype=np.int32)
        for cell_id, poly in self._accepted_masks.items():
            coords      = poly[:, :2].astype(int)
            coords[:,0] = np.clip(coords[:,0], 0, h-1)
            coords[:,1] = np.clip(coords[:,1], 0, w-1)
            try:
                rr, cc = skimage_polygon(
                    coords[:,0], coords[:,1], (h, w)
                )
                if len(rr) > 0:
                    mask[rr, cc] = cell_id
            except Exception as e:
                print(f"Rasterise error cell {cell_id}: {e}")
        return mask

    def _update_labels_layer(self):
        """Remove and re-add the Instance Masks Labels layer."""
        new_mask = self._build_mask()

        if "Instance Masks" in self._viewer.layers:
            self._viewer.layers.remove("Instance Masks")

        self._viewer.add_labels(
            new_mask,
            name="Instance Masks",
            opacity=0.7
        )

        # Move Labels below Shapes
        if ("Draw Polygons Here" in self._viewer.layers and
                "Instance Masks" in self._viewer.layers):
            li = self._viewer.layers.index("Instance Masks")
            di = self._viewer.layers.index("Draw Polygons Here")
            if li > di:
                self._viewer.layers.move(li, di)

        print(f"Labels layer updated: unique={np.unique(new_mask)}")

    # ------------------------------------------------------------------
    # Cell Label Points Layer
    # ------------------------------------------------------------------

    def _update_cell_labels(self):
        """
        Show or hide cell name labels as a Points layer.
        Certain cells labelled white, uncertain cells orange.
        """
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
                # Orange for uncertain, white for certain
                text_colors.append(
                    "orange"
                    if cell_id in self._uncertain_masks
                    else "white"
                )

        if not centroids:
            return

        centroids = np.array(centroids)

        self._viewer.add_points(
            centroids,
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

        # Move to top
        if "Cell Labels" in self._viewer.layers:
            n  = len(self._viewer.layers)
            li = self._viewer.layers.index("Cell Labels")
            if li < n - 1:
                self._viewer.layers.move(li, n - 1)

    # ------------------------------------------------------------------
    # Zoom to Cell
    # ------------------------------------------------------------------

    def _zoom_to_last_accepted(self):
        """Zoom to last accepted polygon if checkbox is on."""
        if not self._zoom_checkbox.value:
            return
        if not self._accepted_masks:
            return

        last_id = max(self._accepted_masks.keys())
        poly    = self._accepted_masks[last_id]
        coords  = poly[:, :2]

        r_min    = coords[:,0].min()
        r_max    = coords[:,0].max()
        c_min    = coords[:,1].min()
        c_max    = coords[:,1].max()
        centre_r = (r_min + r_max) / 2
        centre_c = (c_min + c_max) / 2
        h_poly   = (r_max - r_min) * 1.5
        w_poly   = (c_max - c_min) * 1.5

        try:
            h_canvas = (
                self._viewer.window.qt_viewer.canvas.size[1]
            )
            w_canvas = (
                self._viewer.window.qt_viewer.canvas.size[0]
            )
            if h_poly > 0 and w_poly > 0:
                zoom = min(
                    h_canvas / h_poly,
                    w_canvas / w_poly
                )
            else:
                zoom = 5.0
        except Exception:
            zoom = 5.0

        self._viewer.camera.center = (centre_r, centre_c)
        self._viewer.camera.zoom   = zoom

    # ------------------------------------------------------------------
    # Start Drawing
    # ------------------------------------------------------------------

    def _start_drawing(self):
        """Initialise layers and reset state."""
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
            edge_color="cyan",
            face_color=[0, 1, 1, 0.15],
            edge_width=2,
        )
        self._viewer.layers.selection.active = drawing

        self._accepted_masks  = {}
        self._uncertain_masks = {}
        self._next_cell_id    = 1
        self._update_counter()

        # Disable start, enable drawing buttons
        self._start_button.enabled     = False
        self._accept_button.enabled    = True
        self._reject_button.enabled    = True
        self._uncertain_button.enabled = True
        self._save_button.enabled      = True
        self._undo_button.enabled      = False

        self._status_label.value = (
            "Select polygon tool in toolbar.\n"
            "Draw polygon, double-click to finish.\n"
            "Then click Accept, Reject or Uncertain."
        )

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _accept_polygon(self):
        """Accept last drawn polygon as a certain cell."""
        from skimage.draw import polygon as skimage_polygon

        if "Draw Polygons Here" not in self._viewer.layers:
            self._status_label.value = "Click Start Drawing first!"
            return

        drawing = self._viewer.layers["Draw Polygons Here"]
        n       = len(drawing.data)
        print(f"[ACCEPT] {n} shapes, cell_id={self._next_cell_id}")

        if n == 0:
            self._status_label.value = "Draw a polygon first!"
            return

        poly   = drawing.data[-1].copy()
        coords = poly[:, :2].astype(int)
        h, w   = self._image_shape

        coords[:,0] = np.clip(coords[:,0], 0, h-1)
        coords[:,1] = np.clip(coords[:,1], 0, w-1)

        rr, cc = skimage_polygon(coords[:,0], coords[:,1], (h,w))
        print(f"[ACCEPT] pixels={len(rr)}")

        if len(rr) == 0:
            self._status_label.value = (
                "Polygon has no area! Draw larger."
            )
            return

        cell_id                       = self._next_cell_id
        self._accepted_masks[cell_id] = poly
        # NOT added to _uncertain_masks - this is a certain cell

        self._update_labels_layer()

        if self._labels_checkbox.value:
            self._update_cell_labels()

        self._zoom_to_last_accepted()

        # Turn polygon GREEN
        try:
            idx     = n - 1
            ec      = np.array(drawing.edge_color)
            fc      = np.array(drawing.face_color)
            ec[idx] = [0, 1, 0, 1]
            fc[idx] = [0, 1, 0, 0.15]
            drawing.edge_color = ec
            drawing.face_color = fc
        except Exception as e:
            print(f"Recolour error: {e}")

        self._viewer.layers.selection.active = drawing
        self._next_cell_id += 1
        self._undo_button.enabled = True
        self._update_counter()
        self._status_label.value = (
            f"Cell {cell_id} accepted!\n"
            f"Next label: {self._next_cell_id}"
        )

    # ------------------------------------------------------------------
    # Mark Uncertain
    # ------------------------------------------------------------------

    def _mark_uncertain(self):
        """
        Mark last drawn polygon as uncertain.
        Stored in both accepted_masks and uncertain_masks.
        Displayed orange. Flagged as iscrowd=1 in COCO export.
        DL frameworks ignore iscrowd=1 during training.
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

        poly   = drawing.data[-1].copy()
        coords = poly[:, :2].astype(int)
        h, w   = self._image_shape

        coords[:,0] = np.clip(coords[:,0], 0, h-1)
        coords[:,1] = np.clip(coords[:,1], 0, w-1)

        rr, cc = skimage_polygon(coords[:,0], coords[:,1], (h,w))

        if len(rr) == 0:
            self._status_label.value = (
                "Polygon too small! Draw larger."
            )
            return

        cell_id = self._next_cell_id

        # Store in BOTH dicts
        self._accepted_masks[cell_id]  = poly
        self._uncertain_masks[cell_id] = poly

        self._update_labels_layer()

        if self._labels_checkbox.value:
            self._update_cell_labels()

        self._zoom_to_last_accepted()

        # Turn polygon ORANGE
        try:
            idx     = n - 1
            ec      = np.array(drawing.edge_color)
            fc      = np.array(drawing.face_color)
            ec[idx] = [1.0, 0.5, 0.0, 1.0]
            fc[idx] = [1.0, 0.5, 0.0, 0.15]
            drawing.edge_color = ec
            drawing.face_color = fc
        except Exception as e:
            print(f"Recolour error: {e}")

        self._viewer.layers.selection.active = drawing
        self._next_cell_id   += 1
        self._undo_button.enabled = True
        self._update_counter()
        self._status_label.value = (
            f"Cell {cell_id} flagged uncertain (orange).\n"
            f"Will be ignored in DL training.\n"
            f"Next label: {self._next_cell_id}"
        )
        print(f"[UNCERTAIN] cell_id={cell_id}")

    # ------------------------------------------------------------------
    # Reject
    # ------------------------------------------------------------------

    def _reject_polygon(self):
        """Remove last drawn polygon without saving."""
        if "Draw Polygons Here" not in self._viewer.layers:
            self._status_label.value = "Click Start Drawing first!"
            return
        drawing = self._viewer.layers["Draw Polygons Here"]
        if len(drawing.data) == 0:
            self._status_label.value = "No polygon to reject."
            return
        drawing.data = list(drawing.data)[:-1]
        self._status_label.value = "Rejected. Draw next."

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def _undo_last(self):
        """Remove last accepted mask (certain or uncertain)."""
        if not self._accepted_masks:
            self._status_label.value = "Nothing to undo."
            return

        last_id = max(self._accepted_masks.keys())
        del self._accepted_masks[last_id]

        # Also remove from uncertain if it was flagged
        if last_id in self._uncertain_masks:
            del self._uncertain_masks[last_id]

        self._next_cell_id = last_id

        if "Draw Polygons Here" in self._viewer.layers:
            dl   = self._viewer.layers["Draw Polygons Here"]
            data = list(dl.data)
            if data:
                dl.data = data[:-1]

        self._update_labels_layer()
        self._update_cell_labels()

        if "Draw Polygons Here" in self._viewer.layers:
            self._viewer.layers.selection.active = (
                self._viewer.layers["Draw Polygons Here"]
            )

        self._undo_button.enabled = len(self._accepted_masks) > 0
        self._update_counter()
        self._status_label.value = (
            f"Undone label {last_id}. Draw again."
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

        print(f"\n[SAVE] filename={filename}")
        print(f"[SAVE] base_dir={base_dir}")
        print(f"[SAVE] accepted={len(self._accepted_masks)}")
        print(f"[SAVE] uncertain={len(self._uncertain_masks)}")

        # Check base dir accessible
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
            self._status_label.value = (
                f"Failed to create dirs:\n{e}"
            )
            return

        saved  = []
        failed = []

        # Uncertain IDs for flagging
        uncertain_ids = set(self._uncertain_masks.keys())

        for label, fn, args, kwargs in [
            (
                "instance_mask",
                export_instance_tiff,
                [instance_mask],
                {"path": str(
                    dirs["instance"] /
                    f"{filename}_instance_mask.tiff"
                )}
            ),
            (
                "semantic_mask",
                export_semantic_tiff,
                [instance_mask],
                {"path": str(
                    dirs["semantic"] /
                    f"{filename}_semantic_mask.tiff"
                )}
            ),
            (
                "boundary_mask",
                export_boundary_tiff,
                [instance_mask],
                {"path": str(
                    dirs["boundaries"] /
                    f"{filename}_boundary_mask.tiff"
                )}
            ),
            (
                "boundary_overlay",
                export_boundary_overlay_tiff,
                [instance_mask, image],
                {"path": str(
                    dirs["overlay"] /
                    f"{filename}_boundary_overlay.tiff"
                )}
            ),
            (
                "coco_json",
                export_coco_json,
                [instance_mask, self._accepted_masks],
                {
                    "image_shape":    instance_mask.shape,
                    "path":           str(
                        dirs["coco"] /
                        f"{filename}_coco_annotation.json"
                    ),
                    "uncertain_ids":  uncertain_ids
                }
            ),
            (
                "cellpose",
                export_cellpose_npy,
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

        # Statistics CSV
        stats_rows = []
        try:
            print(f"[SAVE] statistics_csv...")
            stats_rows = export_cell_statistics_csv(
                instance_mask,
                self._accepted_masks,
                path=str(
                    dirs["statistics"] /
                    f"{filename}_statistics.csv"
                ),
                uncertain_ids=uncertain_ids
            )
            saved.append("statistics_csv")
        except Exception as e:
            failed.append(f"statistics_csv: {e}")
            print(f"[SAVE ERROR] statistics_csv: {e}")
            import traceback
            traceback.print_exc()

        # Summary JSON
        try:
            print(f"[SAVE] summary_json...")
            export_summary_json(
                instance_mask,
                self._accepted_masks,
                stats_rows or [],
                path=str(
                    dirs["statistics"] /
                    f"{filename}_summary.json"
                ),
                uncertain_ids=uncertain_ids
            )
            saved.append("summary_json")
        except Exception as e:
            failed.append(f"summary_json: {e}")
            print(f"[SAVE ERROR] summary_json: {e}")
            import traceback
            traceback.print_exc()

        # Session
        try:
            print(f"[SAVE] session...")
            self._save_session(
                str(dirs["session"] / filename)
            )
            saved.append("session")
        except Exception as e:
            failed.append(f"session: {e}")
            print(f"[SAVE ERROR] session: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n[SAVE] Saved: {saved}")
        print(f"[SAVE] Failed: {failed}")

        n         = len(self._accepted_masks)
        n_certain = n - len(self._uncertain_masks)
        n_uncert  = len(self._uncertain_masks)

        if failed:
            self._status_label.value = (
                f"Partial save: {len(saved)}/9\n"
                f"Failed: {len(failed)}\n"
                f"Check terminal."
            )
        else:
            self._status_label.value = (
                f"Saved {n} masks!\n"
                f"Certain: {n_certain} | "
                f"Uncertain: {n_uncert}"
            )

    def _save_session(self, save_stem: str):
        """Save session including uncertain mask IDs."""
        cell_ids = np.array(
            list(self._accepted_masks.keys()), dtype=np.int32
        )
        uncertain_ids = np.array(
            list(self._uncertain_masks.keys()), dtype=np.int32
        )
        polygons = np.empty(len(cell_ids), dtype=object)
        for i, cid in enumerate(cell_ids):
            polygons[i] = self._accepted_masks[cid]

        np.savez(
            f"{save_stem}_session.npz",
            instance_mask=self._build_mask(),
            cell_ids=cell_ids,
            uncertain_ids=uncertain_ids,
            next_cell_id=np.array([self._next_cell_id]),
        )
        np.save(
            f"{save_stem}_polygons.npy",
            polygons, allow_pickle=True
        )
        print(f"Session saved: {save_stem}")

    # ------------------------------------------------------------------
    # Load Session
    # ------------------------------------------------------------------

    def _load_session(self):
        """Load previous session including uncertain flags."""
        image_layers = [
            l for l in self._viewer.layers
            if isinstance(l, Image)
        ]
        if not image_layers:
            self._status_label.value = "Open image first!"
            return

        base_dir      = Path(self._save_dir_input.value.strip())
        filename      = (
            self._filename_input.value.strip() or "cell_masks"
        )
        session_dir   = base_dir / "sessions"
        session_path  = session_dir / f"{filename}_session.npz"
        polygons_path = session_dir / f"{filename}_polygons.npy"

        if not session_path.exists():
            self._status_label.value = (
                "No session found.\n"
                "Check filename and folder."
            )
            return

        try:
            session            = np.load(
                session_path, allow_pickle=True
            )
            cell_ids           = session["cell_ids"]
            self._next_cell_id = int(session["next_cell_id"][0])
            polygons           = np.load(
                polygons_path, allow_pickle=True
            )

            self._accepted_masks = {
                int(cid): polygons[i]
                for i, cid in enumerate(cell_ids)
            }

            # Restore uncertain masks
            uncertain_ids = session.get("uncertain_ids", [])
            self._uncertain_masks = {
                int(cid): self._accepted_masks[int(cid)]
                for cid in uncertain_ids
                if int(cid) in self._accepted_masks
            }

            self._image_shape = self._get_image_shape(
                image_layers[0]
            )

            for name in [
                "Instance Masks",
                "Draw Polygons Here",
                "Cell Labels"
            ]:
                if name in self._viewer.layers:
                    self._viewer.layers.remove(name)

            # Rebuild Labels layer
            self._viewer.add_labels(
                self._build_mask(),
                name="Instance Masks",
                opacity=0.7
            )

            # Rebuild drawing layer
            # Certain = green, uncertain = orange
            drawing = self._viewer.add_shapes(
                name="Draw Polygons Here",
                edge_color="cyan",
                face_color=[0, 1, 1, 0.15],
                edge_width=2,
            )
            for cid, poly in self._accepted_masks.items():
                if cid in self._uncertain_masks:
                    drawing.add_polygons(
                        [poly],
                        edge_color=[1.0, 0.5, 0.0, 1.0],
                        face_color=[1.0, 0.5, 0.0, 0.15],
                    )
                else:
                    drawing.add_polygons(
                        [poly],
                        edge_color="green",
                        face_color=[0, 1, 0, 0.1],
                    )

            # Move Labels below Shapes
            li = self._viewer.layers.index("Instance Masks")
            di = self._viewer.layers.index("Draw Polygons Here")
            if li > di:
                self._viewer.layers.move(li, di)

            self._viewer.layers.selection.active = drawing

            if self._labels_checkbox.value:
                self._update_cell_labels()

            # Keep start disabled - mid session
            self._start_button.enabled     = False
            self._accept_button.enabled    = True
            self._reject_button.enabled    = True
            self._uncertain_button.enabled = True
            self._undo_button.enabled      = True
            self._save_button.enabled      = True

            self._update_counter()
            n       = len(self._accepted_masks)
            n_uncert = len(self._uncertain_masks)
            self._status_label.value = (
                f"Loaded {n} masks "
                f"({n_uncert} uncertain).\n"
                f"Continue drawing."
            )

        except Exception as e:
            self._status_label.value = f"Load error: {e}"
            print(f"Load error: {e}")
            import traceback
            traceback.print_exc()