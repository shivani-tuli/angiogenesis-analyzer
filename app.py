"""
Angiogenesis Analyzer — Desktop Application
============================================
Standalone Tkinter GUI that wraps the analysis engine.
Double-click to launch (or run: python app.py).
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

import analyzer as engine

# ---------------------------------------------------------------------------
# Colour palette (for the legend)
# ---------------------------------------------------------------------------
LEGEND = [
    ("Segments",          "#FF00FF"),
    ("Branches",          "#00FF00"),
    ("Twigs",             "#00FFFF"),
    ("Master segments",   "#FFA500"),
    ("Junctions",         "#FF0000"),
    ("Master junctions",  "#0000FF"),
    ("Meshes",            "#87CEEB"),
    ("Extremities",       "#0064FF"),
    ("Isolated",          "#B4B4B4"),
]


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class AngiogenesisApp(tk.Tk):
    """Desktop Angiogenesis Analyzer."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Angiogenesis Analyzer")
        self.geometry("1400x900")
        self.minsize(1100, 700)
        self.configure(bg="#1a1a2e")

        # State
        self._original_image: Optional[np.ndarray] = None
        self._current_result: Optional[engine.AnalysisResult] = None
        self._all_results: List[engine.AnalysisResult] = []
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._overlay_visible = True
        self._current_file_path: Optional[str] = None

        # Style
        self._setup_styles()
        self._build_menu()
        self._build_ui()

        # Key bindings
        self.bind("<h>", lambda e: self._hide_overlay())
        self.bind("<s>", lambda e: self._show_overlay())
        self.bind("<b>", lambda e: self._blink_overlay())

    # ----- Styles -----
    def _setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # Colours
        bg = "#1a1a2e"
        fg = "#e0e0e0"
        accent = "#0f3460"
        highlight = "#e94560"
        card = "#16213e"

        self.style.configure(".", background=bg, foreground=fg, font=("Helvetica", 11))
        self.style.configure("TFrame", background=bg)
        self.style.configure("Card.TFrame", background=card)
        self.style.configure("TLabel", background=bg, foreground=fg, font=("Helvetica", 11))
        self.style.configure("CardLabel.TLabel", background=card, foreground=fg)
        self.style.configure("Header.TLabel", background=bg, foreground="#ffffff",
                             font=("Helvetica", 16, "bold"))
        self.style.configure("SubHeader.TLabel", background=card, foreground="#ffffff",
                             font=("Helvetica", 13, "bold"))
        self.style.configure("Accent.TButton", background=highlight, foreground="#ffffff",
                             font=("Helvetica", 12, "bold"), padding=(16, 8))
        self.style.map("Accent.TButton",
                       background=[("active", "#c23152"), ("disabled", "#555")])
        self.style.configure("TButton", background=accent, foreground="#ffffff",
                             font=("Helvetica", 11), padding=(12, 6))
        self.style.map("TButton",
                       background=[("active", "#1a4a80")])
        self.style.configure("Treeview", background=card, foreground=fg,
                             fieldbackground=card, font=("Helvetica", 10),
                             rowheight=24)
        self.style.configure("Treeview.Heading", background=accent, foreground="#ffffff",
                             font=("Helvetica", 10, "bold"))

        # Combobox
        self.style.configure("TCombobox", fieldbackground=card, background=card,
                             foreground=fg, selectbackground=accent)

        # LabelFrame
        self.style.configure("TLabelframe", background=card, foreground=fg)
        self.style.configure("TLabelframe.Label", background=card, foreground="#ffffff",
                             font=("Helvetica", 11, "bold"))

        # Scale
        self.style.configure("TScale", background=bg, troughcolor=accent)

    # ----- Menu bar -----
    def _build_menu(self):
        menubar = tk.Menu(self, bg="#16213e", fg="#e0e0e0", activebackground="#0f3460",
                          activeforeground="#fff", font=("Helvetica", 11))

        file_menu = tk.Menu(menubar, tearoff=0, bg="#16213e", fg="#e0e0e0",
                            activebackground="#0f3460", activeforeground="#fff")
        file_menu.add_command(label="Open Image…", accelerator="⌘O",
                              command=self._open_image)
        file_menu.add_command(label="Batch Process Folder…",
                              command=self._batch_process)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results as CSV…",
                              command=lambda: self._export("csv"))
        file_menu.add_command(label="Export Results as Excel…",
                              command=lambda: self._export("xlsx"))
        file_menu.add_separator()
        file_menu.add_command(label="Save Overlay Image…",
                              command=self._save_overlay)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", accelerator="⌘Q",
                              command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0, bg="#16213e", fg="#e0e0e0",
                            activebackground="#0f3460", activeforeground="#fff")
        view_menu.add_command(label="Show Overlay (s)", command=self._show_overlay)
        view_menu.add_command(label="Hide Overlay (h)", command=self._hide_overlay)
        view_menu.add_command(label="Blink Overlay (b)", command=self._blink_overlay)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0, bg="#16213e", fg="#e0e0e0",
                            activebackground="#0f3460", activeforeground="#fff")
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    # ----- Main UI layout -----
    def _build_ui(self):
        # --- Top: Title bar ---
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", padx=20, pady=(15, 5))
        ttk.Label(top_bar, text="🔬 Angiogenesis Analyzer", style="Header.TLabel").pack(side="left")
        self._status_label = ttk.Label(top_bar, text="Ready — Open an image to begin",
                                       style="TLabel")
        self._status_label.pack(side="right")

        # --- Main content: left (image) + right (controls + results) ---
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=20, pady=10)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # --- Left: Image canvas ---
        img_frame = ttk.Frame(main, style="Card.TFrame")
        img_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self._canvas = tk.Canvas(img_frame, bg="#0d0d1a", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=2, pady=2)

        # --- Right panel ---
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)

        # --- Controls card ---
        ctrl_frame = ttk.LabelFrame(right, text="  ⚙  Analysis Settings  ", padding=10)
        ctrl_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        # Image type
        ttk.Label(ctrl_frame, text="Image type:", style="CardLabel.TLabel").grid(
            row=0, column=0, sticky="w", pady=3)
        self._image_type_var = tk.StringVar(value="Phase Contrast")
        cb = ttk.Combobox(ctrl_frame, textvariable=self._image_type_var,
                          values=["Phase Contrast", "Fluorescence"], state="readonly", width=18)
        cb.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3)

        # Min object size
        ttk.Label(ctrl_frame, text="Min object size:", style="CardLabel.TLabel").grid(
            row=1, column=0, sticky="w", pady=3)
        self._min_size_var = tk.IntVar(value=150)
        ttk.Entry(ctrl_frame, textvariable=self._min_size_var, width=8).grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=3)

        # Max loop area
        ttk.Label(ctrl_frame, text="Max loop area:", style="CardLabel.TLabel").grid(
            row=2, column=0, sticky="w", pady=3)
        self._max_loop_var = tk.IntVar(value=100)
        ttk.Entry(ctrl_frame, textvariable=self._max_loop_var, width=8).grid(
            row=2, column=1, sticky="w", padx=(8, 0), pady=3)

        # Twig threshold
        ttk.Label(ctrl_frame, text="Twig threshold:", style="CardLabel.TLabel").grid(
            row=3, column=0, sticky="w", pady=3)
        self._twig_var = tk.IntVar(value=25)
        ttk.Entry(ctrl_frame, textvariable=self._twig_var, width=8).grid(
            row=3, column=1, sticky="w", padx=(8, 0), pady=3)

        # Gaussian sigma
        ttk.Label(ctrl_frame, text="Gaussian σ:", style="CardLabel.TLabel").grid(
            row=4, column=0, sticky="w", pady=3)
        self._sigma_var = tk.DoubleVar(value=1.0)
        ttk.Entry(ctrl_frame, textvariable=self._sigma_var, width=8).grid(
            row=4, column=1, sticky="w", padx=(8, 0), pady=3)

        # --- Crop margins (to exclude scale bars / legends) ---
        sep = ttk.Separator(ctrl_frame, orient="horizontal")
        sep.grid(row=5, column=0, columnspan=2, sticky="ew", pady=6)

        ttk.Label(ctrl_frame, text="Crop margins (px):", style="CardLabel.TLabel",
                  font=("Helvetica", 10, "italic")).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(0, 2))

        crop_grid = ttk.Frame(ctrl_frame)
        crop_grid.grid(row=7, column=0, columnspan=2, sticky="ew")

        ttk.Label(crop_grid, text="Top:", style="CardLabel.TLabel").grid(row=0, column=0, sticky="w")
        self._crop_top_var = tk.IntVar(value=0)
        ttk.Entry(crop_grid, textvariable=self._crop_top_var, width=5).grid(row=0, column=1, padx=(2, 8))

        ttk.Label(crop_grid, text="Bottom:", style="CardLabel.TLabel").grid(row=0, column=2, sticky="w")
        self._crop_bottom_var = tk.IntVar(value=60)
        ttk.Entry(crop_grid, textvariable=self._crop_bottom_var, width=5).grid(row=0, column=3, padx=(2, 0))

        ttk.Label(crop_grid, text="Left:", style="CardLabel.TLabel").grid(row=1, column=0, sticky="w", pady=2)
        self._crop_left_var = tk.IntVar(value=0)
        ttk.Entry(crop_grid, textvariable=self._crop_left_var, width=5).grid(row=1, column=1, padx=(2, 8), pady=2)

        ttk.Label(crop_grid, text="Right:", style="CardLabel.TLabel").grid(row=1, column=2, sticky="w", pady=2)
        self._crop_right_var = tk.IntVar(value=0)
        ttk.Entry(crop_grid, textvariable=self._crop_right_var, width=5).grid(row=1, column=3, padx=(2, 0), pady=2)

        ctrl_frame.columnconfigure(1, weight=1)

        # --- Action buttons ---
        btn_frame = ttk.Frame(right)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        self._analyze_btn = ttk.Button(btn_frame, text="▶  Analyze Image",
                                       style="Accent.TButton",
                                       command=self._run_analysis)
        self._analyze_btn.pack(fill="x", pady=(0, 5))

        self._batch_btn = ttk.Button(btn_frame, text="📂  Batch Process Folder → Excel",
                                     style="TButton",
                                     command=self._batch_process)
        self._batch_btn.pack(fill="x")

        # --- Results table ---
        results_frame = ttk.LabelFrame(right, text="  📊  Results  ", padding=5)
        results_frame.grid(row=2, column=0, sticky="nsew")
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(results_frame, columns=("value",), show="tree headings",
                                  selectmode="none")
        self._tree.heading("#0", text="Parameter")
        self._tree.heading("value", text="Value")
        self._tree.column("#0", width=180, stretch=True)
        self._tree.column("value", width=100, anchor="e")
        self._tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scrollbar.set)

        # --- Legend ---
        legend_frame = ttk.LabelFrame(right, text="  🎨  Legend  ", padding=5)
        legend_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for i, (name, colour) in enumerate(LEGEND):
            row_f = ttk.Frame(legend_frame)
            row_f.pack(fill="x", padx=4, pady=1)
            swatch = tk.Canvas(row_f, width=14, height=14, bg=colour,
                               highlightthickness=1, highlightbackground="#444")
            swatch.pack(side="left", padx=(0, 6))
            ttk.Label(row_f, text=name, style="CardLabel.TLabel",
                      font=("Helvetica", 9)).pack(side="left")

    # =====================================================================
    # Actions
    # =====================================================================

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self._load_image(path)

    def _load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Error", f"Could not read image:\n{path}")
            return
        self._original_image = img
        self._current_result = None
        self._current_file_path = path
        self._display_image(img)
        name = os.path.basename(path)
        self._status_label.config(text=f"Loaded: {name}")

    def _display_image(self, img: np.ndarray):
        """Fit image into canvas and display."""
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.update_idletasks()
        cw = max(self._canvas.winfo_width(), 400)
        ch = max(self._canvas.winfo_height(), 400)

        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor="center", image=self._photo)

    def _run_analysis(self):
        if self._original_image is None:
            messagebox.showinfo("Info", "Please open an image first.")
            return
        self._analyze_btn.config(state="disabled")
        self._status_label.config(text="⏳ Analyzing…")
        self.update_idletasks()

        # Run in background thread
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _crop_image(self, img: np.ndarray) -> np.ndarray:
        """Apply crop margins to exclude scale bars / legends."""
        t = max(0, self._crop_top_var.get())
        b = max(0, self._crop_bottom_var.get())
        l = max(0, self._crop_left_var.get())
        r = max(0, self._crop_right_var.get())
        h, w = img.shape[:2]
        y1 = min(t, h - 1)
        y2 = max(h - b, y1 + 1)
        x1 = min(l, w - 1)
        x2 = max(w - r, x1 + 1)
        return img[y1:y2, x1:x2]

    def _analysis_worker(self):
        try:
            img = self._crop_image(self._original_image)
            img_type = "phase_contrast" if self._image_type_var.get() == "Phase Contrast" \
                else "fluorescence"
            name = os.path.basename(self._current_file_path) if self._current_file_path else "image"

            result = engine.analyze(
                img,
                image_name=name,
                image_type=img_type,
                sigma=self._sigma_var.get(),
                min_object_size=self._min_size_var.get(),
                max_loop_area=self._max_loop_var.get(),
                twig_threshold=self._twig_var.get(),
            )
            self._current_result = result
            self._all_results.append(result)

            # Schedule UI update on main thread
            self.after(0, self._on_analysis_done)
        except Exception as exc:
            self.after(0, lambda: self._on_analysis_error(str(exc)))

    def _on_analysis_done(self):
        result = self._current_result
        self._overlay_visible = True
        cropped = self._crop_image(self._original_image)
        overlay = engine.render_overlay(cropped, result)
        self._display_image(overlay)
        self._populate_results(result)
        self._analyze_btn.config(state="normal")
        self._status_label.config(text="✅ Analysis complete")

    def _on_analysis_error(self, msg: str):
        self._analyze_btn.config(state="normal")
        self._status_label.config(text="❌ Error")
        messagebox.showerror("Analysis Error", msg)

    def _populate_results(self, result: engine.AnalysisResult):
        self._tree.delete(*self._tree.get_children())
        for key, val in result.as_dict().items():
            self._tree.insert("", "end", text=key, values=(val,))

    # ----- Overlay controls -----
    def _show_overlay(self):
        if self._current_result and self._original_image is not None:
            cropped = self._crop_image(self._original_image)
            overlay = engine.render_overlay(cropped, self._current_result)
            self._display_image(overlay)
            self._overlay_visible = True

    def _hide_overlay(self):
        if self._original_image is not None:
            self._display_image(self._crop_image(self._original_image))
            self._overlay_visible = False

    def _blink_overlay(self):
        if not self._current_result:
            return
        if self._overlay_visible:
            self._hide_overlay()
        else:
            self._show_overlay()

    # ----- Export -----
    def _export(self, fmt: str):
        if not self._all_results:
            messagebox.showinfo("Info", "No results to export. Run an analysis first.")
            return
        if fmt == "csv":
            path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV", "*.csv")])
            if path:
                engine.export_csv(self._all_results, path)
                messagebox.showinfo("Exported", f"CSV saved to:\n{path}")
        else:
            path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel", "*.xlsx")])
            if path:
                engine.export_excel(self._all_results, path)
                messagebox.showinfo("Exported", f"Excel saved to:\n{path}")

    def _save_overlay(self):
        if self._current_result is None or self._original_image is None:
            messagebox.showinfo("Info", "Run an analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif")],
        )
        if not path:
            return
        overlay = engine.render_overlay(self._original_image, self._current_result)
        cv2.imwrite(path, overlay)
        messagebox.showinfo("Saved", f"Overlay image saved to:\n{path}")

    # ----- Batch process -----
    def _batch_process(self):
        folder = filedialog.askdirectory(title="Select folder of images")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        )
        if not files:
            messagebox.showinfo("Info", "No image files found in selected folder.")
            return

        self._analyze_btn.config(state="disabled")
        self._batch_btn.config(state="disabled")
        self._status_label.config(text=f"⏳ Batch: 0/{len(files)}")
        self.update_idletasks()

        threading.Thread(target=self._batch_worker, args=(folder, files), daemon=True).start()

    def _batch_worker(self, folder: str, files: list):
        img_type = "phase_contrast" if self._image_type_var.get() == "Phase Contrast" \
            else "fluorescence"
        batch_results: List[engine.AnalysisResult] = []
        failed: List[str] = []
        for i, fname in enumerate(files):
            self.after(0, lambda i=i: self._status_label.config(
                text=f"⏳ Batch: {i + 1}/{len(files)} — {files[i]}"))
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                failed.append(fname)
                continue
            try:
                img = self._crop_image(img)
                result = engine.analyze(
                    img, image_name=fname, image_type=img_type,
                    sigma=self._sigma_var.get(),
                    min_object_size=self._min_size_var.get(),
                    max_loop_area=self._max_loop_var.get(),
                    twig_threshold=self._twig_var.get(),
                )
                batch_results.append(result)
                self._all_results.append(result)
            except Exception as exc:
                failed.append(f"{fname} ({exc})")

        # Auto-export batch results as BOTH CSV and Excel to same folder
        if batch_results:
            xlsx_path = os.path.join(folder, "angiogenesis_results.xlsx")
            csv_path = os.path.join(folder, "angiogenesis_results.csv")
            engine.export_excel(batch_results, xlsx_path)
            engine.export_csv(batch_results, csv_path)
            self.after(0, lambda: self._on_batch_done(
                len(batch_results), xlsx_path, len(failed)))
        else:
            self.after(0, lambda: self._on_batch_done(0, "", len(failed)))

    def _on_batch_done(self, count: int, path: str, n_failed: int = 0):
        self._analyze_btn.config(state="normal")
        self._batch_btn.config(state="normal")
        if count > 0:
            self._status_label.config(text=f"✅ Batch complete — {count} images")
            msg = f"Processed {count} images."
            if n_failed > 0:
                msg += f"\n({n_failed} images failed)"
            msg += f"\n\nExcel saved to:\n{path}"
            messagebox.showinfo("Batch Complete", msg)
        else:
            self._status_label.config(text="⚠ Batch — no images processed")
            messagebox.showwarning("Batch", "No images could be processed.")

    # ----- About -----
    def _show_about(self):
        messagebox.showinfo(
            "About Angiogenesis Analyzer",
            "Angiogenesis Analyzer v1.0\n\n"
            "A standalone desktop application replicating the\n"
            "ImageJ Angiogenesis Analyzer by Gilles Carpentier.\n\n"
            "Analyzes Endothelial Tube Formation Assay (ETFA)\n"
            "images and quantifies network morphology.\n\n"
            "Built with Python, scikit-image, OpenCV & Tkinter.",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = AngiogenesisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
