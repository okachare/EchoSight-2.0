"""Tabbed desktop workspace for EchoSight 2.0."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

from PIL import Image, ImageEnhance, ImageFilter, ImageTk

from . import __version__
from .exporting import ExportReport, export_run
from .frames import SUPPORTED_IMAGES, LoadedFrame, load_frames
from .inference import InferenceEngine, InferenceResult, ModelInfo, ModelLoader
from .rendering import RenderOptions, render_result
from .runtime import log_directory
from .training import export_training_frames

LOGGER = logging.getLogger(__name__)


class FitImageCanvas(tk.Canvas):
    """Image canvas with fit-to-view, wheel zoom, and drag panning."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, background="#17191d", borderwidth=0, highlightthickness=0, cursor="crosshair")
        self.source_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_origin: tuple[int, int] | None = None
        self.bind("<Configure>", lambda _: self._draw())
        self.bind("<MouseWheel>", self._zoom_image)
        self.bind("<Button-4>", lambda event: self._zoom_image(event, 1))
        self.bind("<Button-5>", lambda event: self._zoom_image(event, -1))
        self.bind("<ButtonPress-1>", self._start_pan)
        self.bind("<B1-Motion>", self._pan_image)
        self.bind("<ButtonRelease-1>", lambda _: self.configure(cursor="crosshair"))
        self.bind("<Double-Button-1>", lambda _: self.reset_view())

    def set_image(self, image: Image.Image | None) -> None:
        self.source_image = image
        self._draw()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._draw()

    def _zoom_image(self, event: tk.Event, direction: int | None = None) -> str:
        if self.source_image is None:
            return "break"
        step = direction if direction is not None else (1 if event.delta > 0 else -1)
        previous = self.zoom
        self.zoom = min(8.0, max(0.25, self.zoom * (1.15 if step > 0 else 1 / 1.15)))
        ratio = self.zoom / previous
        center_x = self.winfo_width() / 2
        center_y = self.winfo_height() / 2
        self.pan_x = event.x - center_x - (event.x - center_x - self.pan_x) * ratio
        self.pan_y = event.y - center_y - (event.y - center_y - self.pan_y) * ratio
        self._draw()
        return "break"

    def _start_pan(self, event: tk.Event) -> None:
        self.drag_origin = (event.x, event.y)
        self.configure(cursor="fleur")

    def _pan_image(self, event: tk.Event) -> None:
        if self.drag_origin is None or self.source_image is None:
            return
        self.pan_x += event.x - self.drag_origin[0]
        self.pan_y += event.y - self.drag_origin[1]
        self.drag_origin = (event.x, event.y)
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        if self.source_image is None:
            self.create_text(
                max(1, self.winfo_width()) // 2,
                max(1, self.winfo_height()) // 2,
                text="Open an image to begin",
                fill="#65717c",
                font=("Segoe UI", 12),
            )
            return
        width = max(1, self.winfo_width() - 32)
        height = max(1, self.winfo_height() - 32)
        fit_scale = min(width / self.source_image.width, height / self.source_image.height)
        scale = fit_scale * self.zoom
        preview = self.source_image.resize(
            (max(1, round(self.source_image.width * scale)), max(1, round(self.source_image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(preview)
        self.create_image(self.winfo_width() / 2 + self.pan_x, self.winfo_height() / 2 + self.pan_y, image=self.photo)


class ActivityIndicator(tk.Canvas):
    """Small animated ring shown while background work is active."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, width=28, height=28, background="#171717", borderwidth=0, highlightthickness=0)
        self.angle = 0
        self.running = False
        self.job: str | None = None
        self._draw()

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._animate()

    def stop(self) -> None:
        self.running = False
        if self.job is not None:
            self.after_cancel(self.job)
            self.job = None
        self._draw()

    def _animate(self) -> None:
        if not self.running:
            return
        self.angle = (self.angle + 18) % 360
        self._draw()
        self.job = self.after(45, self._animate)

    def _draw(self) -> None:
        self.delete("all")
        color = "#42b879" if self.running else "#656b70"
        self.create_oval(5, 5, 23, 23, outline="#36393d", width=3)
        self.create_arc(5, 5, 23, 23, start=self.angle, extent=105, style="arc", outline=color, width=3)


class EchoSightApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"EchoSight 2.0 | v{__version__}")
        self.geometry("1420x880")
        self.minsize(1040, 680)
        self.configure(background="#171717")

        self.model_path: Path | None = None
        self.model_parent: Path | None = None
        self.model_info: ModelInfo | None = None
        self.inference_engine: InferenceEngine | None = None
        self.frames: list[LoadedFrame] = []
        self.results: dict[int, InferenceResult] = {}
        self.inference_failures: dict[int, str] = {}
        self.current_analysis_index: int | None = None
        self.current_result_index: int | None = None
        self.hidden_annotations: dict[int, set[int]] = {}
        self.preprocess_profiles: dict[int, tuple[float, float, float, float]] = {}
        self.annotation_profiles: dict[int, RenderOptions] = {}
        self.annotation_variables: list[tk.BooleanVar] = []
        self.result_sort_column = "frame"
        self.result_sort_descending = False
        self.preprocess_popup: tk.Toplevel | None = None
        self.annotation_popup: tk.Toplevel | None = None
        self.annotation_color: tuple[int, int, int] | None = None
        self.choosing_annotation_color = False
        self.inference_running = False
        self.export_running = False
        self.cancel_inference = threading.Event()
        self.resume_activity = threading.Event()
        self.resume_activity.set()
        self.model_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.image_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.inference_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.export_events: queue.Queue[tuple[str, object]] = queue.Queue()

        self.show_boxes = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_masks = tk.BooleanVar(value=True)
        self.show_heatmap = tk.BooleanVar(value=True)
        self.annotation_font_size = tk.DoubleVar(value=10.0)
        self.annotation_thickness = tk.DoubleVar(value=3.0)
        self.annotation_transparency = tk.DoubleVar(value=0.0)
        self.label_transparency = tk.DoubleVar(value=0.0)
        self.brightness = tk.DoubleVar(value=1.0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.sharpness = tk.DoubleVar(value=1.0)
        self.denoise_strength = tk.DoubleVar(value=0.0)

        self._configure_styles()
        self._build_layout()
        self._log("EchoSight 2.0 session started")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind_all("<Button-1>", self._dismiss_popups, add="+")
        self.after_idle(self._maximize_window)

    def _maximize_window(self) -> None:
        try:
            self.state("zoomed")
        except tk.TclError:
            self.attributes("-fullscreen", True)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        font = "Segoe UI Variable Text"
        style.configure("App.TFrame", background="#171717")
        style.configure("Panel.TFrame", background="#222222")
        style.configure("Header.TLabel", background="#171717", foreground="#f2f3f4", font=("Segoe UI Variable Display Semibold", 18))
        style.configure("PanelTitle.TLabel", background="#222222", foreground="#eceeef", font=(font, 9, "bold"))
        style.configure("PanelText.TLabel", background="#222222", foreground="#b4b9be", font=(font, 9))
        style.configure("Accent.TButton", background="#36a269", foreground="#ffffff", borderwidth=0, padding=(14, 9), font=(font, 9, "bold"))
        style.map("Accent.TButton", background=[("active", "#42b879"), ("disabled", "#2d4738")], foreground=[("disabled", "#82978a")])
        style.configure("Load.TButton", background="#447fbd", foreground="#ffffff", borderwidth=0, padding=(14, 9), font=(font, 9, "bold"))
        style.map("Load.TButton", background=[("active", "#5591cf"), ("disabled", "#304155")], foreground=[("disabled", "#8391a0")])
        style.configure("Tool.TButton", background="#333333", foreground="#e2e5e7", borderwidth=0, padding=(11, 8), font=(font, 9))
        style.map("Tool.TButton", background=[("active", "#414141"), ("disabled", "#292929")], foreground=[("disabled", "#6f7479")])
        style.configure("Danger.TButton", background="#4b2d2d", foreground="#f0b2ae", borderwidth=0, padding=(11, 8), font=(font, 9))
        style.map("Danger.TButton", background=[("active", "#613535"), ("disabled", "#2a2525")], foreground=[("disabled", "#785e5c")])
        style.configure("Panel.TCheckbutton", background="#222222", foreground="#b4b9be", font=(font, 9))
        style.map("Panel.TCheckbutton", background=[("active", "#222222")], foreground=[("active", "#f0f2f3")])
        style.configure("TNotebook", background="#171717", borderwidth=0)
        style.configure("TNotebook.Tab", background="#242424", foreground="#8f969c", padding=(20, 9), font=(font, 9))
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#303030"), ("active", "#2a2a2a")],
            foreground=[("selected", "#f4f5f6"), ("active", "#d9dcdf")],
            padding=[("selected", (28, 12)), ("!selected", (20, 9))],
            font=[("selected", (font, 10, "bold")), ("!selected", (font, 9))],
        )
        style.configure("Status.TLabel", background="#111111", foreground="#939aa1", padding=(12, 7), font=(font, 9))
        style.configure("Results.Treeview", background="#1b1b1b", fieldbackground="#1b1b1b", foreground="#d8dcdf", rowheight=30, borderwidth=0, font=(font, 9))
        style.configure("Results.Treeview.Heading", background="#303030", foreground="#e4e7e9", relief="flat", padding=(8, 8), font=(font, 9, "bold"))
        style.map("Results.Treeview", background=[("selected", "#315342")], foreground=[("selected", "#ffffff")])
        style.configure("TPanedwindow", background="#171717")
        style.configure("TScrollbar", background="#373737", troughcolor="#1d1d1d", bordercolor="#1d1d1d", arrowcolor="#aeb4b9")
        style.configure("TScale", background="#222222", troughcolor="#3a3a3a")
        style.configure("TProgressbar", background="#36a269", troughcolor="#2a2a2a", borderwidth=0)

    def _build_layout(self) -> None:
        header = ttk.Frame(self, style="App.TFrame", padding=(20, 13, 20, 10))
        header.pack(fill="x")
        ttk.Label(header, text="EchoSight 2.0", style="Header.TLabel").pack(side="left")
        self.activity = ActivityIndicator(header)
        self.activity.pack(side="right", padx=(10, 0))
        self.activity_text = tk.StringVar(value="Ready")
        ttk.Label(header, textvariable=self.activity_text, style="Header.TLabel", font=("Segoe UI Variable Text", 9)).pack(side="right")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        self.analysis_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.results_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.analysis_tab, text="Analysis")
        self.notebook.add(self.results_tab, text="Results")
        self._build_analysis_tab()
        self._build_results_tab()

        self.status_text = tk.StringVar(value="Ready | Load a model and images")
        ttk.Label(self, textvariable=self.status_text, style="Status.TLabel", anchor="w").pack(fill="x", side="bottom")

    def _build_analysis_tab(self) -> None:
        panes = ttk.Panedwindow(self.analysis_tab, orient="horizontal")
        panes.pack(fill="both", expand=True)

        sources = ttk.Frame(panes, style="Panel.TFrame", padding=12, width=250)
        viewer = ttk.Frame(panes, style="Panel.TFrame")
        side = ttk.Panedwindow(panes, orient="vertical", width=340)
        panes.add(sources, weight=1)
        panes.add(viewer, weight=4)
        panes.add(side, weight=2)

        ttk.Label(sources, text="SOURCES", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))
        actions = ttk.Frame(sources, style="Panel.TFrame")
        actions.pack(fill="x", pady=(0, 10))
        self.load_model_button = ttk.Button(actions, text="Load Model", style="Load.TButton", command=self._select_model)
        self.load_model_button.pack(fill="x")
        self.open_images_button = ttk.Button(actions, text="Open Images", style="Tool.TButton", command=self._select_images)
        self.open_images_button.pack(fill="x", pady=(6, 0))
        self.model_name = tk.StringVar(value="No model loaded")
        ttk.Label(sources, textvariable=self.model_name, style="PanelText.TLabel", wraplength=220).pack(anchor="w", pady=(0, 8))
        self.image_list = self._new_listbox(sources)
        self.image_list.pack(fill="both", expand=True)
        self.image_list.bind("<<ListboxSelect>>", self._show_analysis_selection)

        self.analysis_canvas = FitImageCanvas(viewer)
        self.analysis_canvas.pack(fill="both", expand=True)
        self.preprocess_button = ttk.Button(viewer, text="\u2699", width=3, style="Tool.TButton", command=self._toggle_preprocess_popup)
        self.preprocess_button.place(relx=1.0, rely=1.0, x=-18, y=-18, anchor="se")

        model_panel = ttk.Frame(side, style="Panel.TFrame", padding=12, height=340)
        terminal_panel = ttk.Frame(side, style="Panel.TFrame", padding=12, height=260)
        side.add(model_panel, weight=3)
        side.add(terminal_panel, weight=2)
        ttk.Label(model_panel, text="MODEL INFORMATION", style="PanelTitle.TLabel").pack(anchor="w")
        model_info_area = ttk.Frame(model_panel, style="Panel.TFrame")
        model_info_area.pack(fill="both", expand=True, pady=(8, 12))
        model_scrollbar = ttk.Scrollbar(model_info_area, orient="vertical")
        self.model_detail = tk.Text(
            model_info_area,
            background="#191a1c",
            foreground="#c1c7cc",
            borderwidth=0,
            wrap="word",
            state="disabled",
            height=8,
            font=("Consolas", 8),
            yscrollcommand=model_scrollbar.set,
        )
        model_scrollbar.configure(command=self.model_detail.yview)
        self.model_detail.pack(side="left", fill="both", expand=True)
        model_scrollbar.pack(side="right", fill="y")
        self._set_model_detail("Select the trained model's parent folder. EchoSight will locate the deployable OpenVINO IR or ONNX artifact.")
        self.run_all_button = ttk.Button(model_panel, text="Run All", style="Accent.TButton", state="disabled", command=self._run_all)
        self.run_all_button.pack(fill="x", pady=(12, 0))
        run_row = ttk.Frame(model_panel, style="Panel.TFrame")
        run_row.pack(fill="x", pady=(6, 0))
        run_row.columnconfigure((0, 1), weight=1, uniform="run-actions")
        self.run_selected_button = ttk.Button(run_row, text="Run Selected", style="Tool.TButton", state="disabled", command=self._run_selected)
        self.run_selected_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.run_button = ttk.Button(run_row, text="Run Current", style="Tool.TButton", state="disabled", command=self._run_current)
        self.run_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        activity_row = ttk.Frame(model_panel, style="Panel.TFrame")
        activity_row.pack(fill="x", pady=(6, 0))
        activity_row.columnconfigure((0, 1), weight=1, uniform="activity-actions")
        self.pause_button = ttk.Button(activity_row, text="Pause", style="Tool.TButton", state="disabled", command=self._toggle_pause)
        self.pause_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.cancel_button = ttk.Button(activity_row, text="Cancel", style="Danger.TButton", state="disabled", command=self._cancel_run)
        self.cancel_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.progress = ttk.Progressbar(model_panel, mode="determinate", maximum=1)
        self.progress.pack(fill="x", pady=(10, 0))

        ttk.Label(terminal_panel, text="TERMINAL", style="PanelTitle.TLabel").pack(anchor="w")
        self.terminal = tk.Text(terminal_panel, background="#0b0e12", foreground="#9fb0bd", insertbackground="#ffffff", borderwidth=0, wrap="word", state="disabled", font=("Consolas", 8))
        self.terminal.pack(fill="both", expand=True, pady=(8, 0))

    def _build_preprocess_controls(self, parent: ttk.Frame) -> None:
        for label, variable, start, end in (
            ("Brightness", self.brightness, 0.5, 1.5),
            ("Contrast", self.contrast, 0.5, 1.5),
            ("Sharpness", self.sharpness, 0.5, 2.0),
            ("Denoise strength", self.denoise_strength, 0.0, 5.0),
        ):
            row = ttk.Frame(parent, style="Panel.TFrame")
            row.pack(fill="x")
            ttk.Label(row, text=label, style="PanelText.TLabel", width=17).pack(side="left")
            value = ttk.Label(row, style="PanelText.TLabel", width=5, anchor="e")
            value.pack(side="right")

            def update(_: str | None = None, current=variable, output=value) -> None:
                output.configure(text=f"{current.get():.1f}")
                self._refresh_analysis()

            ttk.Scale(row, variable=variable, from_=start, to=end, command=update).pack(side="left", fill="x", expand=True, padx=(4, 8))
            update()

    def _toggle_preprocess_popup(self) -> None:
        if self.preprocess_popup is not None and self.preprocess_popup.winfo_exists():
            self._close_preprocess_popup()
            return
        self._load_preprocess_profile(self.current_analysis_index)
        popup = tk.Toplevel(self)
        self.preprocess_popup = popup
        popup.overrideredirect(True)
        popup.configure(background="#45484c")
        panel = ttk.Frame(popup, style="Panel.TFrame", padding=14)
        panel.pack(fill="both", expand=True, padx=1, pady=1)
        header = ttk.Frame(panel, style="Panel.TFrame")
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="IMAGE ADJUSTMENTS", style="PanelTitle.TLabel").pack(side="left")
        self._build_preprocess_controls(panel)
        ttk.Button(panel, text="Apply to current frame", style="Accent.TButton", command=self._apply_preprocessing_current).pack(fill="x", pady=(10, 0))
        ttk.Button(panel, text="Apply to all frames", style="Tool.TButton", command=self._apply_preprocessing_all).pack(fill="x", pady=(6, 0))
        ttk.Button(panel, text="Reset", style="Tool.TButton", command=self._reset_preprocessing).pack(fill="x", pady=(10, 0))
        popup.update_idletasks()
        x = self.preprocess_button.winfo_rootx() + self.preprocess_button.winfo_width() - popup.winfo_reqwidth()
        y = self.preprocess_button.winfo_rooty() - popup.winfo_reqheight() - 8
        popup.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _close_preprocess_popup(self) -> None:
        if self.preprocess_popup is not None and self.preprocess_popup.winfo_exists():
            self.preprocess_popup.destroy()
        self.preprocess_popup = None
        self._load_preprocess_profile(self.current_analysis_index)
        self._refresh_analysis()

    def _apply_preprocessing_current(self) -> None:
        if self.current_analysis_index is None:
            return
        self.preprocess_profiles[self.current_analysis_index] = self._preprocess_draft()
        self._refresh_analysis()
        self._refresh_result()

    def _apply_preprocessing_all(self) -> None:
        settings = self._preprocess_draft()
        self.preprocess_profiles = {index: settings for index in range(len(self.frames))}
        self._refresh_analysis()
        self._refresh_result()

    def _reset_preprocessing(self) -> None:
        self.brightness.set(1.0)
        self.contrast.set(1.0)
        self.sharpness.set(1.0)
        self.denoise_strength.set(0.0)
        self._refresh_analysis()

    def _build_results_tab(self) -> None:
        panes = ttk.Panedwindow(self.results_tab, orient="horizontal")
        panes.pack(fill="both", expand=True)
        result_list_panel = ttk.Frame(panes, style="Panel.TFrame", padding=12)
        result_viewer = ttk.Frame(panes, style="Panel.TFrame")
        detail_panel = ttk.Frame(panes, style="Panel.TFrame", padding=12)
        panes.add(result_list_panel, weight=0)
        panes.add(result_viewer, weight=1)
        panes.add(detail_panel, weight=0)

        ttk.Label(result_list_panel, text="RESULT FRAMES", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))
        columns = ("frame", "type", "annotations", "confidence")
        self.result_list = ttk.Treeview(result_list_panel, columns=columns, show="headings", style="Results.Treeview", selectmode="extended")
        self.result_list.column("frame", width=135, minwidth=90, anchor="w", stretch=True)
        self.result_list.column("type", width=65, minwidth=55, anchor="w", stretch=False)
        self.result_list.column("annotations", width=82, minwidth=70, anchor="center", stretch=False)
        self.result_list.column("confidence", width=120, minwidth=100, anchor="e", stretch=False)
        for column, label in (("frame", "Frame"), ("type", "Type"), ("annotations", "Annotations"), ("confidence", "Highest confidence")):
            self.result_list.heading(column, text=label, command=lambda key=column: self._sort_results(key))
        result_vertical = ttk.Scrollbar(result_list_panel, orient="vertical", command=self.result_list.yview)
        result_horizontal = ttk.Scrollbar(result_list_panel, orient="horizontal", command=self.result_list.xview)
        self.result_list.configure(yscrollcommand=result_vertical.set, xscrollcommand=result_horizontal.set)
        self.result_list.pack(side="left", fill="both", expand=True)
        result_vertical.pack(side="right", fill="y")
        result_horizontal.pack(side="bottom", fill="x")
        self.result_list.bind("<<TreeviewSelect>>", self._show_result_selection)
        self.results_canvas = FitImageCanvas(result_viewer)
        self.results_canvas.pack(fill="both", expand=True)
        self.annotation_style_button = ttk.Button(
            result_viewer,
            text="\U0001f3a8",
            width=3,
            style="Tool.TButton",
            command=self._toggle_annotation_popup,
        )
        self.annotation_style_button.place(relx=1.0, rely=1.0, x=-18, y=-18, anchor="se")

        ttk.Label(detail_panel, text="RESULT DETAILS", style="PanelTitle.TLabel").pack(anchor="w")
        self.result_text = tk.StringVar(value="Run inference in Analysis to populate results.")
        ttk.Label(detail_panel, textvariable=self.result_text, style="PanelText.TLabel", wraplength=295, justify="left").pack(anchor="w", pady=(8, 12))
        ttk.Label(detail_panel, text="OVERLAYS", style="PanelTitle.TLabel").pack(anchor="w")
        for text, variable in (("Bounding boxes", self.show_boxes), ("Labels", self.show_labels), ("Instance masks", self.show_masks), ("Anomaly heatmap", self.show_heatmap)):
            ttk.Checkbutton(detail_panel, text=text, variable=variable, command=self._refresh_annotation_views, style="Panel.TCheckbutton").pack(anchor="w")
        self.export_button = ttk.Button(
            detail_panel,
            text="Save All",
            style="Accent.TButton",
            state="disabled",
            command=lambda: self._select_export_destination(current_only=False),
        )
        self.export_button.pack(fill="x", pady=(14, 0))
        self.export_current_button = ttk.Button(
            detail_panel,
            text="Save Current",
            style="Tool.TButton",
            state="disabled",
            command=lambda: self._select_export_destination(current_only=True),
        )
        self.export_current_button.pack(fill="x", pady=(6, 0))
        ttk.Label(detail_panel, text="MODEL RETRAINING", style="PanelTitle.TLabel").pack(anchor="w", pady=(14, 5))
        self.mark_false_hits_button = ttk.Button(
            detail_panel,
            text="Mark for Training (False Hits)",
            style="Danger.TButton",
            state="disabled",
            command=lambda: self._mark_for_training("False_Hits"),
        )
        self.mark_false_hits_button.pack(fill="x")
        self.mark_misses_button = ttk.Button(
            detail_panel,
            text="Mark for Training (Misses)",
            style="Tool.TButton",
            state="disabled",
            command=lambda: self._mark_for_training("Misses"),
        )
        self.mark_misses_button.pack(fill="x", pady=(6, 0))
        ttk.Label(detail_panel, text="ANNOTATIONS", style="PanelTitle.TLabel").pack(anchor="w", pady=(14, 5))
        annotation_area = ttk.Frame(detail_panel, style="Panel.TFrame")
        annotation_area.pack(fill="both", expand=True)
        self.annotation_canvas = tk.Canvas(annotation_area, background="#222222", borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(annotation_area, orient="vertical", command=self.annotation_canvas.yview)
        self.annotation_container = ttk.Frame(self.annotation_canvas, style="Panel.TFrame")
        self.annotation_window = self.annotation_canvas.create_window((0, 0), window=self.annotation_container, anchor="nw")
        self.annotation_canvas.configure(yscrollcommand=scrollbar.set)
        self.annotation_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.annotation_container.bind("<Configure>", lambda _: self.annotation_canvas.configure(scrollregion=self.annotation_canvas.bbox("all")))
        self.annotation_canvas.bind("<Configure>", lambda event: self.annotation_canvas.itemconfigure(self.annotation_window, width=event.width))

    def _toggle_annotation_popup(self) -> None:
        if self.annotation_popup is not None and self.annotation_popup.winfo_exists():
            self._close_annotation_popup()
            return
        self._load_annotation_profile(self.current_result_index)
        popup = tk.Toplevel(self)
        self.annotation_popup = popup
        popup.overrideredirect(True)
        popup.configure(background="#45484c")
        panel = ttk.Frame(popup, style="Panel.TFrame", padding=14)
        panel.pack(fill="both", expand=True, padx=1, pady=1)
        header = ttk.Frame(panel, style="Panel.TFrame")
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="ANNOTATION CONTROL", style="PanelTitle.TLabel").pack(side="left")

        self._build_annotation_scale(panel, "Font size", self.annotation_font_size, 6, 32, "pt")
        self._build_annotation_scale(panel, "Line thickness", self.annotation_thickness, 1, 12, "px")
        self._build_annotation_scale(panel, "Annotation transparency", self.annotation_transparency, 0, 1, "%")
        self._build_annotation_scale(panel, "Label transparency", self.label_transparency, 0, 1, "%")

        color_row = ttk.Frame(panel, style="Panel.TFrame")
        color_row.pack(fill="x", pady=(7, 0))
        ttk.Label(color_row, text="Annotation color", style="PanelText.TLabel", width=21).pack(side="left")
        color = self._annotation_color_hex()
        self.annotation_color_button = tk.Button(
            color_row,
            text=color or "Current palette",
            command=self._choose_annotation_color,
            background=color or "#333333",
            foreground="#ffffff",
            activebackground=color or "#414141",
            activeforeground="#ffffff",
            borderwidth=0,
            padx=8,
            pady=4,
        )
        self.annotation_color_button.pack(side="right")
        ttk.Button(panel, text="Apply to current frame", style="Accent.TButton", command=self._apply_annotation_current).pack(fill="x", pady=(10, 0))
        ttk.Button(panel, text="Apply to all frames", style="Tool.TButton", command=self._apply_annotation_all).pack(fill="x", pady=(6, 0))
        ttk.Button(panel, text="Reset", style="Tool.TButton", command=self._reset_annotation_style).pack(fill="x", pady=(10, 0))
        popup.update_idletasks()
        x = self.annotation_style_button.winfo_rootx() + self.annotation_style_button.winfo_width() - popup.winfo_reqwidth()
        y = self.annotation_style_button.winfo_rooty() - popup.winfo_reqheight() - 8
        popup.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _build_annotation_scale(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        start: float,
        end: float,
        unit: str,
    ) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="PanelText.TLabel", width=21).pack(side="left")
        value = ttk.Label(row, style="PanelText.TLabel", width=6, anchor="e")
        value.pack(side="right")

        def update(_: str | None = None) -> None:
            current = variable.get()
            value.configure(text=f"{round(current * 100)}%" if unit == "%" else f"{round(current)} {unit}")
            self._refresh_result()

        ttk.Scale(row, variable=variable, from_=start, to=end, command=update).pack(side="left", fill="x", expand=True, padx=(4, 8))
        update()

    def _choose_annotation_color(self) -> None:
        initial = self._annotation_color_hex() or "#25b9a7"
        self.choosing_annotation_color = True
        try:
            rgb, hex_color = colorchooser.askcolor(color=initial, parent=self.annotation_popup or self, title="Choose annotation color")
        finally:
            self.choosing_annotation_color = False
        if rgb is None or hex_color is None:
            return
        self.annotation_color = tuple(round(channel) for channel in rgb)
        self.annotation_color_button.configure(background=hex_color, activebackground=hex_color, text=hex_color)
        self._refresh_result()

    def _annotation_color_hex(self) -> str | None:
        if self.annotation_color is None:
            return None
        return "#{:02x}{:02x}{:02x}".format(*self.annotation_color)

    def _close_annotation_popup(self) -> None:
        if self.annotation_popup is not None and self.annotation_popup.winfo_exists():
            self.annotation_popup.destroy()
        self.annotation_popup = None
        self._load_annotation_profile(self.current_result_index)
        self._refresh_result()

    def _apply_annotation_current(self) -> None:
        if self.current_result_index is None:
            return
        self.annotation_profiles[self.current_result_index] = self._annotation_draft()
        self._refresh_result()

    def _apply_annotation_all(self) -> None:
        options = self._annotation_draft()
        self.annotation_profiles = {index: options for index in range(len(self.frames))}
        self._refresh_result()

    def _reset_annotation_style(self) -> None:
        self.annotation_font_size.set(10.0)
        self.annotation_thickness.set(3.0)
        self.annotation_transparency.set(0.0)
        self.label_transparency.set(0.0)
        self.annotation_color = None
        self._refresh_result()

    def _dismiss_popups(self, event: tk.Event) -> None:
        if self.choosing_annotation_color:
            return
        for popup, close in (
            (self.preprocess_popup, self._close_preprocess_popup),
            (self.annotation_popup, self._close_annotation_popup),
        ):
            if popup is not None and popup.winfo_exists() and not self._is_descendant(event.widget, popup):
                close()

    @staticmethod
    def _is_descendant(widget: tk.Misc, parent: tk.Misc) -> bool:
        current: tk.Misc | None = widget
        while current is not None:
            if current == parent:
                return True
            current = current.master
        return False

    @staticmethod
    def _new_listbox(parent: tk.Misc) -> tk.Listbox:
        return tk.Listbox(parent, background="#1b1b1b", foreground="#d5d9dc", selectbackground="#315342", selectforeground="#ffffff", borderwidth=0, highlightthickness=0, activestyle="none", font=("Segoe UI Variable Text", 9), exportselection=False, selectmode="extended")

    def _select_model(self) -> None:
        selected = filedialog.askdirectory(title="Select trained model parent folder", mustexist=True)
        if not selected:
            return
        parent = Path(selected)
        try:
            models = ModelLoader.discover(parent)
        except OSError as error:
            self._log(f"ERROR | Model folder could not be searched: {error}")
            messagebox.showerror("Model folder error", str(error))
            return
        if not models:
            message = "No supported model was found below this folder. Expected an OpenVINO .xml/.bin pair or an ONNX file."
            self._log(f"ERROR | {message} | {parent}")
            messagebox.showerror("No model found", message)
            return
        if len(models) > 1:
            preview = "\n".join(f"- {path.relative_to(parent.resolve())}" for path in models[:6])
            message = f"This folder contains {len(models)} distinct models. Select the direct parent folder for one trained model.\n\n{preview}"
            self._log(f"ERROR | Multiple models found below {parent}: {len(models)}")
            messagebox.showerror("Multiple models found", message)
            return
        path = models[0]
        self.model_path = None
        self.model_parent = parent
        self.model_info = None
        self.inference_engine = None
        self.results.clear()
        self.inference_failures.clear()
        self._clear_results_ui()
        self.model_name.set(parent.name)
        self._set_model_detail("Reading model metadata and preparing CPU compilation...")
        self._set_activity(f"Reading model | {path.name}", True)
        self.load_model_button.configure(state="disabled")
        self._log(f"Model folder: {parent}")
        self._log(f"Discovered model: {path.relative_to(parent.resolve())}")
        threading.Thread(target=self._model_worker, args=(path, parent), daemon=True).start()
        self.after(100, self._poll_model)

    def _model_worker(self, path: Path, parent: Path) -> None:
        try:
            loader = ModelLoader()
            self.model_events.put(("progress", f"Inspecting metadata | {path.name}"))
            info = loader.inspect(path, parent)
            self.model_events.put(("progress", f"Compiling {info.model_type or info.task_type.value} for CPU"))
            model = loader.core.read_model(info.path)
            compiled = loader.core.compile_model(model, loader.device)
            self.model_events.put(("loaded", (info, compiled)))
        except Exception as error:
            LOGGER.exception("Model loading failed")
            self.model_events.put(("error", error))

    def _poll_model(self) -> None:
        try:
            event, payload = self.model_events.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_model)
            return
        if event == "progress":
            self._set_activity(str(payload), True)
            self._log(str(payload))
            self.after(50, self._poll_model)
            return
        self.load_model_button.configure(state="normal")
        self._set_activity("Ready", False)
        if event == "error":
            self.model_name.set("No model loaded")
            self._set_model_detail("Model loading failed. See Terminal and application log.")
            self._log(f"ERROR | Model loading failed: {payload}")
            messagebox.showerror("Model loading error", str(payload))
            return
        self.model_info, compiled = payload
        self.model_path = self.model_info.path
        self.inference_engine = InferenceEngine(self.model_info, compiled)
        size = self.model_info.input_size
        input_description = f"{size[1]} x {size[0]}" if size else "Dynamic"
        outputs = "\n".join(f"  {item.name}: {item.shape}" for item in self.model_info.outputs)
        labels = ", ".join(self.model_info.labels) if self.model_info.labels else "Embedded or unavailable"
        metadata = dict(self.model_info.metadata)
        metrics = "\n".join(f"  {name}: {value:.2%}" for name, value in self.model_info.metrics) or "  Not available with this export"
        collaterals = ", ".join(self.model_info.collaterals) or "Embedded model metadata only"
        self._set_model_detail(
            f"Model: {metadata.get('model_name', self.model_info.path.stem)}\n"
            f"Model type: {self.model_info.model_type or 'Not embedded'}\n"
            f"Model version: {metadata.get('model_version', 'Not embedded')}\n"
            f"GetiTune version: {metadata.get('getitune_version', 'Not embedded')}\n"
            f"Task: {self.model_info.task_type.value.replace('_', ' ').title()}\n"
            f"Generated output: {self.model_info.output_mode}\n"
            f"Format / device: {self.model_info.format} / {self.model_info.device}\n"
            f"Training completed: {self.model_info.training_date or 'Not available'}\n"
            f"Exported / modified: {self.model_info.artifact_date or 'Not available'}\n"
            f"Package folder: {self.model_parent.name if self.model_parent else 'Not available'}\n"
            f"Package collateral: {collaterals}\n"
            f"Artifact: {self.model_info.path.name} ({self.model_info.path.stat().st_size / 1_048_576:.1f} MB)\n\n"
            f"Labels ({len(self.model_info.labels)}): {labels}\n"
            f"Confidence threshold: {self.model_info.confidence_threshold:.1%}\n"
            f"IoU threshold: {self._metadata_percent(metadata, 'iou_threshold')}\n\n"
            f"Input: {input_description} | {self.model_info.inputs[0].element_type}\n"
            f"Resize: {metadata.get('resize_type', 'Not embedded')}\n"
            f"Intensity: {metadata.get('intensity_mode', 'Not embedded')}\n"
            f"Mean: {metadata.get('mean_values', 'Not embedded')}\n"
            f"Scale: {metadata.get('scale_values', 'Not embedded')}\n"
            f"Reverse channels: {metadata.get('reverse_input_channels', 'Not embedded')}\n"
            f"Outputs:\n{outputs}\n\nEvaluation scores:\n{metrics}"
        )
        self._log(
            f"Model ready | {metadata.get('model_name', self.model_info.task_type.value)} | "
            f"threshold {self.model_info.confidence_threshold:.1%} | {len(self.model_info.metrics)} metric(s)"
        )
        self._update_run_state()

    def _select_images(self) -> None:
        selected = filedialog.askopenfilenames(title="Select images or TIFF files", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
        if not selected:
            return
        paths = [Path(item) for item in selected if Path(item).suffix.lower() in SUPPORTED_IMAGES]
        self.open_images_button.configure(state="disabled")
        self._set_activity("Loading images", True)
        self._log(f"Loading {len(paths)} image file(s)")
        threading.Thread(target=self._image_worker, args=(paths,), daemon=True).start()
        self.after(100, self._poll_images)

    def _image_worker(self, paths: list[Path]) -> None:
        frames: list[LoadedFrame] = []
        errors: list[str] = []
        for file_position, path in enumerate(paths, start=1):
            self._wait_if_paused()
            try:
                def report(
                    frame_number: int,
                    frame_count: int,
                    current_file: int = file_position,
                    current_name: str = path.name,
                ) -> None:
                    self._wait_if_paused()
                    self.image_events.put(("progress", (current_file, len(paths), current_name, frame_number, frame_count)))

                frames.extend(load_frames(path, report))
            except (OSError, ValueError) as error:
                errors.append(f"{path.name}: {error}")
        self.image_events.put(("loaded", (frames, errors, len(paths))))

    def _poll_images(self) -> None:
        try:
            event, payload = self.image_events.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_images)
            return
        if event == "progress":
            file_position, file_count, name, frame_number, frame_count = payload
            detail = f"Loading {name} | Frame {frame_number}/{frame_count} | File {file_position}/{file_count}"
            self._set_activity(detail, True)
            self._log(detail)
            self.after(10, self._poll_images)
            return
        self.frames, errors, count = payload
        self.results.clear()
        self.inference_failures.clear()
        self.hidden_annotations.clear()
        self.preprocess_profiles.clear()
        self.annotation_profiles.clear()
        self.image_list.delete(0, tk.END)
        self._clear_results_ui()
        for frame in self.frames:
            self.image_list.insert(tk.END, frame.display_name)
        if self.frames:
            self.image_list.selection_set(0)
            self._select_analysis_frame(0)
        self.open_images_button.configure(state="normal")
        self._set_activity("Ready", False)
        self._log(f"Loaded {len(self.frames)} frame(s) from {count} file(s)")
        for error in errors:
            self._log(f"ERROR | {error}")
        self._update_run_state()

    def _show_analysis_selection(self, _event: object) -> None:
        selection = self.image_list.curselection()
        if selection:
            active = self.image_list.index(tk.ACTIVE)
            self._select_analysis_frame(active if active in selection else selection[0])

    def _select_analysis_frame(self, index: int) -> None:
        if index != self.current_analysis_index:
            self.analysis_canvas.reset_view()
        self.current_analysis_index = index
        if self.preprocess_popup is None:
            self._load_preprocess_profile(index)
        self._refresh_analysis()
        frame = self.frames[index]
        self.status_text.set(f"{frame.display_name} | {frame.image.width} x {frame.image.height}")

    @staticmethod
    def _process_image(image: Image.Image, settings: tuple[float, float, float, float]) -> Image.Image:
        brightness, contrast, sharpness, denoise_strength = settings
        processed = ImageEnhance.Brightness(image).enhance(brightness)
        processed = ImageEnhance.Contrast(processed).enhance(contrast)
        processed = ImageEnhance.Sharpness(processed).enhance(sharpness)
        return processed.filter(ImageFilter.GaussianBlur(denoise_strength)) if denoise_strength > 0 else processed

    @staticmethod
    def _default_preprocess_settings() -> tuple[float, float, float, float]:
        return 1.0, 1.0, 1.0, 0.0

    def _preprocess_draft(self) -> tuple[float, float, float, float]:
        return self.brightness.get(), self.contrast.get(), self.sharpness.get(), self.denoise_strength.get()

    def _preprocess_settings(self, index: int | None = None, live: bool = False) -> tuple[float, float, float, float]:
        if live and index == self.current_analysis_index and self.preprocess_popup is not None:
            return self._preprocess_draft()
        if index is None:
            index = self.current_analysis_index
        return self.preprocess_profiles.get(index, self._default_preprocess_settings())

    def _load_preprocess_profile(self, index: int | None) -> None:
        settings = self.preprocess_profiles.get(index, self._default_preprocess_settings())
        self.brightness.set(settings[0])
        self.contrast.set(settings[1])
        self.sharpness.set(settings[2])
        self.denoise_strength.set(settings[3])

    def _processed_image(self, image: Image.Image, index: int | None = None, live: bool = False) -> Image.Image:
        return self._process_image(image, self._preprocess_settings(index, live))

    def _refresh_analysis(self) -> None:
        if self.current_analysis_index is not None and self.current_analysis_index < len(self.frames):
            index = self.current_analysis_index
            image = self._processed_image(self.frames[index].image, index, live=True)
            self.analysis_canvas.set_image(image)

    def _run_current(self) -> None:
        if self.current_analysis_index is not None:
            index = self.current_analysis_index
            self._start_inference([(index, self.frames[index])])

    def _run_selected(self) -> None:
        indexes = self.image_list.curselection()
        self._start_inference([(index, self.frames[index]) for index in indexes])

    def _run_all(self) -> None:
        self._start_inference(list(enumerate(self.frames)))

    def _start_inference(self, frames: list[tuple[int, LoadedFrame]]) -> None:
        if self.inference_engine is None or self.inference_running or not frames:
            return
        self.inference_running = True
        self.cancel_inference.clear()
        self.resume_activity.set()
        self.pause_button.configure(text="Pause", state="normal")
        self.progress.configure(maximum=len(frames), value=0)
        self.load_model_button.configure(state="disabled")
        self.open_images_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self._update_run_state()
        self._set_activity("Running inference", True)
        self._log(f"Inference started | {len(frames)} frame(s)")
        settings = {index: self._preprocess_settings(index) for index, _frame in frames}
        for index, _frame in frames:
            self.inference_failures.pop(index, None)
        threading.Thread(target=self._inference_worker, args=(frames, settings), daemon=True).start()
        self.after(100, self._poll_inference)

    def _inference_worker(
        self,
        frames: list[tuple[int, LoadedFrame]],
        settings: dict[int, tuple[float, float, float, float]],
    ) -> None:
        completed = 0
        failed = 0
        try:
            for position, (index, frame) in enumerate(frames, start=1):
                if not self._wait_if_paused():
                    self.inference_events.put(("cancelled", completed))
                    return
                self.inference_events.put(("progress", (position, len(frames), frame.display_name)))
                try:
                    image = self._process_image(frame.image, settings[index])
                    result = self.inference_engine.infer(image, frame.source)
                    completed += 1
                    self.inference_events.put(("result", (position, len(frames), index, result)))
                except Exception as error:
                    failed += 1
                    LOGGER.exception("Inference failed for %s", frame.display_name)
                    self.inference_events.put(("failure", (position, len(frames), index, frame.display_name, str(error))))
            self.inference_events.put(("complete", (completed, failed)))
        except Exception as error:
            LOGGER.exception("Inference failed")
            self.inference_events.put(("error", error))

    def _poll_inference(self) -> None:
        try:
            event, payload = self.inference_events.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_inference)
            return
        if event == "progress":
            position, total, name = payload
            detail = f"Inferencing {name} | Frame {position}/{total}"
            if self.resume_activity.is_set():
                self._set_activity(detail, True)
            self._log(detail)
            self.after(10, self._poll_inference)
            return
        if event == "result":
            position, total, index, result = payload
            first_result = not self.results
            self.results[index] = result
            self.inference_failures.pop(index, None)
            self.progress.configure(value=position)
            self._refresh_result_list()
            self._log(f"Frame {index + 1}/{len(self.frames)} | {result.summary} | {result.duration_ms:.1f} ms")
            if first_result:
                self._select_result_frame(index)
            self.status_text.set(f"Processed {position} of {total}" if self.resume_activity.is_set() else "Paused")
            self.after(10, self._poll_inference)
            return
        if event == "failure":
            position, total, index, name, message = payload
            self.results.pop(index, None)
            self.inference_failures[index] = message
            self.hidden_annotations.pop(index, None)
            self.progress.configure(value=position)
            self._refresh_result_list()
            if self.current_result_index == index:
                self.current_result_index = None
                self.results_canvas.set_image(None)
                self.result_text.set(f"Inference failed for {name}\n\n{message}")
            self._log(f"ERROR | {name} | {message}")
            self.status_text.set(f"Processed {position} of {total} | 1 failure" if self.resume_activity.is_set() else "Paused")
            self.after(10, self._poll_inference)
            return
        if event == "error":
            self._log(f"ERROR | Inference failed: {payload}")
            self._finish_inference("Inference failed")
            messagebox.showerror("Inference error", str(payload))
        elif event == "cancelled":
            self._finish_inference(f"Cancelled after {payload} frame(s)")
        else:
            completed, failed = payload
            self._finish_inference(f"Inference complete: {completed} succeeded, {failed} failed")

    def _finish_inference(self, status: str) -> None:
        self.inference_running = False
        self.load_model_button.configure(state="normal")
        self.open_images_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.pause_button.configure(text="Pause", state="disabled")
        self.resume_activity.set()
        self._set_activity("Ready", False)
        self.status_text.set(status)
        self._log(status)
        self._update_run_state()

    def _cancel_run(self) -> None:
        self.cancel_inference.set()
        self.resume_activity.set()
        self.cancel_button.configure(state="disabled")
        self._log("Cancellation requested")

    def _toggle_pause(self) -> None:
        if not self.inference_running and not self.export_running:
            return
        if self.resume_activity.is_set():
            self.resume_activity.clear()
            self.pause_button.configure(text="Resume")
            self._set_activity("Paused", False)
            self._log("Activity paused")
        else:
            self.resume_activity.set()
            self.pause_button.configure(text="Pause")
            self._set_activity("Resuming", True)
            self._log("Activity resumed")

    def _wait_if_paused(self) -> bool:
        while not self.resume_activity.wait(0.1):
            if self.cancel_inference.is_set():
                return False
        return not self.cancel_inference.is_set()

    def _refresh_result_list(self) -> None:
        selected = set(self.result_list.selection())
        ordered = self._ordered_result_indexes()
        self.result_list.delete(*self.result_list.get_children())
        for index in ordered:
            result = self.results[index]
            annotations, confidence = self._result_metrics(result)
            self.result_list.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    self.frames[index].display_name,
                    result.task_type.value.replace("_", " ").title(),
                    annotations,
                    f"{confidence:.1%}" if confidence is not None else "--",
                ),
            )
        self._update_result_headings()
        restored = [item for item in selected if int(item) in self.results]
        if restored:
            self.result_list.selection_set(restored)
        elif self.current_result_index in self.results:
            self.result_list.selection_set(str(self.current_result_index))
        self._update_training_button_state()

    def _show_result_selection(self, _event: object) -> None:
        selection = self.result_list.selection()
        if selection:
            focused = self.result_list.focus()
            index = int(focused) if focused in selection else int(selection[0])
            if index != self.current_result_index:
                self._select_result_frame(index)
        self._update_training_button_state()

    @staticmethod
    def _result_metrics(result: InferenceResult) -> tuple[int, float | None]:
        confidences = [item.confidence for item in result.detections]
        confidences.extend(item.confidence for item in result.classifications)
        if result.anomaly_score is not None:
            confidences.append(result.anomaly_score)
        return len(result.detections) + len(result.classifications) + int(result.anomaly_score is not None), max(confidences, default=None)

    def _ordered_result_indexes(self) -> list[int]:
        def sort_key(index: int) -> object:
            result = self.results[index]
            annotations, confidence = self._result_metrics(result)
            keys = {
                "frame": (self.frames[index].source.name.casefold(), self.frames[index].frame_number),
                "type": result.task_type.value,
                "annotations": annotations,
                "confidence": confidence if confidence is not None else -1.0,
            }
            return keys[self.result_sort_column]

        return sorted(self.results, key=sort_key, reverse=self.result_sort_descending)

    def _sort_results(self, column: str) -> None:
        if self.result_sort_column == column:
            self.result_sort_descending = not self.result_sort_descending
        else:
            self.result_sort_column = column
            self.result_sort_descending = False
        self._refresh_result_list()

    def _update_result_headings(self) -> None:
        labels = {"frame": "Frame", "type": "Type", "annotations": "Annotations", "confidence": "Highest confidence"}
        for column, label in labels.items():
            indicator = ""
            if column == self.result_sort_column:
                indicator = " \u25bc" if self.result_sort_descending else " \u25b2"
            self.result_list.heading(column, text=label + indicator, command=lambda key=column: self._sort_results(key))

    def _select_result_frame(self, index: int) -> None:
        if index != self.current_result_index:
            self.results_canvas.reset_view()
        self.current_result_index = index
        if self.annotation_popup is None:
            self._load_annotation_profile(index)
        if index < len(self.frames):
            self.image_list.selection_clear(0, tk.END)
            self.image_list.selection_set(index)
            self.image_list.activate(index)
            self._select_analysis_frame(index)
        if index in self.results and str(index) not in self.result_list.selection():
            self.result_list.selection_set(str(index))
        if index in self.results:
            self.result_list.see(str(index))
        self._refresh_result()
        self._show_result_details(self.results[index])

    def _refresh_result(self) -> None:
        if self.current_result_index is None or self.current_result_index not in self.results:
            return
        index = self.current_result_index
        rendered = render_result(
            self._processed_image(self.frames[index].image, index),
            self.results[index],
            self._render_options(index, live=True),
            self.hidden_annotations.get(index, set()),
        )
        self.results_canvas.set_image(rendered)

    def _refresh_annotation_views(self) -> None:
        self._refresh_result()

    def _annotation_draft(self) -> RenderOptions:
        return RenderOptions(
            show_boxes=self.show_boxes.get(),
            show_labels=self.show_labels.get(),
            show_masks=self.show_masks.get(),
            show_heatmap=self.show_heatmap.get(),
            font_family="Calibri",
            font_size=round(self.annotation_font_size.get()),
            annotation_color=self.annotation_color,
            annotation_thickness=round(self.annotation_thickness.get()),
            annotation_opacity=1.0 - self.annotation_transparency.get(),
            label_opacity=1.0 - self.label_transparency.get(),
        )

    def _render_options(self, index: int | None = None, live: bool = False) -> RenderOptions:
        if index is None:
            index = self.current_result_index
        if live and self.annotation_popup is not None and index == self.current_result_index:
            return self._annotation_draft()
        applied = self.annotation_profiles.get(index, RenderOptions())
        return replace(
            applied,
            show_boxes=self.show_boxes.get(),
            show_labels=self.show_labels.get(),
            show_masks=self.show_masks.get(),
            show_heatmap=self.show_heatmap.get(),
        )

    def _load_annotation_profile(self, index: int | None) -> None:
        options = self.annotation_profiles.get(index, RenderOptions())
        self.annotation_font_size.set(options.font_size)
        self.annotation_thickness.set(options.annotation_thickness)
        self.annotation_transparency.set(1.0 - options.annotation_opacity)
        self.label_transparency.set(1.0 - options.label_opacity)
        self.annotation_color = options.annotation_color

    def _select_export_destination(self, current_only: bool = False) -> None:
        if self.model_info is None or (not self.results and not self.inference_failures):
            return
        if current_only and self.current_result_index not in self.results:
            return
        selected = filedialog.askdirectory(title="Select folder for exported run", mustexist=True)
        if not selected:
            return
        current_index = self.current_result_index if current_only else self.current_analysis_index
        settings = self._preprocess_settings(current_index)
        settings_by_frame = {index: self._preprocess_settings(index) for index in range(len(self.frames))}
        preprocessing = {
            "brightness": settings[0],
            "contrast": settings[1],
            "sharpness": settings[2],
            "denoise_strength": settings[3],
            "by_frame": {
                str(index): {
                    "brightness": values[0],
                    "contrast": values[1],
                    "sharpness": values[2],
                    "denoise_strength": values[3],
                }
                for index, values in settings_by_frame.items()
            },
        }
        self.export_running = True
        self.resume_activity.set()
        self.pause_button.configure(text="Pause", state="normal")
        self.load_model_button.configure(state="disabled")
        self.open_images_button.configure(state="disabled")
        self._update_run_state()
        self._set_activity("Exporting results", True)
        selected_results, selected_failures = self._export_payload(current_only)
        self._log(f"Export started | {len(selected_results)} result(s), {len(selected_failures)} failure(s)")
        threading.Thread(
            target=self._export_worker,
            args=(
                Path(selected),
                list(self.frames),
                selected_results,
                self.model_info,
                self.model_parent,
                preprocessing,
                settings_by_frame,
                {index: self._render_options(index) for index in range(len(self.frames))},
                {index: set(hidden) for index, hidden in self.hidden_annotations.items()},
                selected_failures,
            ),
            daemon=True,
        ).start()
        self.after(100, self._poll_export)

    def _export_payload(self, current_only: bool) -> tuple[dict[int, InferenceResult], dict[int, str]]:
        if current_only and self.current_result_index is not None:
            return {self.current_result_index: self.results[self.current_result_index]}, {}
        return dict(self.results), dict(self.inference_failures)

    def _mark_for_training(self, category: str) -> None:
        indexes = self._selected_result_indexes()
        if not indexes:
            messagebox.showinfo("Mark for Training", "Select one or more result frames first.")
            return
        selected = filedialog.askdirectory(title="Select folder for training images", mustexist=True)
        if not selected:
            return
        if self.model_parent is not None:
            model_folder_name = self.model_parent.name
        elif self.model_info is not None:
            model_folder_name = self.model_info.path.parent.name
        else:
            model_folder_name = "Model"
        try:
            report = export_training_frames(
                Path(selected),
                [(index, self.frames[index]) for index in indexes],
                category,
                model_folder_name,
            )
        except (OSError, ValueError) as error:
            self._log(f"ERROR | Training image export failed: {error}")
            messagebox.showerror("Mark for Training", str(error))
            return
        label = "False Hits" if category == "False_Hits" else "Misses"
        self.status_text.set(f"Marked {len(report.images)} frame(s) for training: {label}")
        self._log(f"Marked for training | {label} | {len(report.images)} frame(s) | {report.directory}")
        messagebox.showinfo(
            "Training images saved",
            f"Saved {len(report.images)} selected frame(s).\n\n{report.directory}",
        )

    def _selected_result_indexes(self) -> list[int]:
        return sorted(int(item) for item in self.result_list.selection() if int(item) in self.results)

    def _update_training_button_state(self) -> None:
        enabled = bool(self.result_list.selection()) and not self.inference_running and not self.export_running
        state = "normal" if enabled else "disabled"
        self.mark_false_hits_button.configure(state=state)
        self.mark_misses_button.configure(state=state)

    def _export_worker(
        self,
        destination: Path,
        frames: list[LoadedFrame],
        results: dict[int, InferenceResult],
        model_info: ModelInfo,
        model_parent: Path | None,
        preprocessing: dict[str, object],
        settings: dict[int, tuple[float, float, float, float]],
        render_options: dict[int, RenderOptions],
        hidden_annotations: dict[int, set[int]],
        failures: dict[int, str],
    ) -> None:
        try:
            def report_progress(position: int, total: int, name: str) -> None:
                self._wait_if_paused()
                self.export_events.put(("progress", (position, total, name)))

            report = export_run(
                destination=destination,
                frames=frames,
                results=results,
                model_info=model_info,
                model_parent=model_parent,
                preprocessing=preprocessing,
                render_options_by_frame=render_options,
                hidden_annotations=hidden_annotations,
                failures=failures,
                frame_image_transform=lambda image, index: self._process_image(image, settings[index]),
                progress=report_progress,
            )
            self.export_events.put(("complete", report))
        except Exception as error:
            LOGGER.exception("Result export failed")
            self.export_events.put(("error", error))

    def _poll_export(self) -> None:
        try:
            event, payload = self.export_events.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_export)
            return
        if event == "progress":
            position, total, name = payload
            detail = f"Exporting {name} | Frame {position}/{total}"
            self._set_activity(detail, True)
            self._log(detail)
            self.after(10, self._poll_export)
            return
        self.export_running = False
        self.resume_activity.set()
        self.pause_button.configure(text="Pause", state="disabled")
        self.load_model_button.configure(state="normal")
        self.open_images_button.configure(state="normal")
        self._set_activity("Ready", False)
        self._update_run_state()
        if event == "error":
            self._log(f"ERROR | Export failed: {payload}")
            messagebox.showerror("Export error", str(payload))
            return
        report: ExportReport = payload
        self.status_text.set(f"Export complete: {report.directory}")
        self._log(f"Export complete | {report.directory}")
        messagebox.showinfo(
            "Export complete",
            f"Exported {report.exported_frames} result(s) and {report.failed_frames} failure(s).\n\n{report.directory}",
        )

    def _show_result_details(self, result: InferenceResult) -> None:
        lines = [result.summary, f"Inference: {result.duration_ms:.1f} ms", f"Input: {result.input_size[0]} x {result.input_size[1]}"]
        lines.extend(f"{item.label}: {item.confidence:.1%}" for item in result.detections)
        lines.extend(f"{item.label}: {item.confidence:.1%}" for item in result.classifications)
        self.result_text.set("\n".join(lines))
        self._build_annotation_controls(result)

    def _build_annotation_controls(self, result: InferenceResult) -> None:
        for child in self.annotation_container.winfo_children():
            child.destroy()
        self.annotation_variables.clear()
        entries: list[str] = []
        if result.anomaly_score is not None:
            entries.append(f"Anomaly heatmap | {result.anomaly_score:.1%}")
        entries.extend(f"{item.label} | {item.confidence:.1%}" for item in result.detections)
        entries.extend(f"{item.label} | {item.confidence:.1%}" for item in result.classifications)
        if not entries:
            ttk.Label(self.annotation_container, text="No annotations", style="PanelText.TLabel").pack(anchor="w")
            return
        hidden = self.hidden_annotations.setdefault(self.current_result_index, set())
        for index, text in enumerate(entries):
            variable = tk.BooleanVar(value=index not in hidden)
            self.annotation_variables.append(variable)
            ttk.Checkbutton(
                self.annotation_container,
                text=text,
                variable=variable,
                command=lambda annotation=index, state=variable: self._toggle_annotation(annotation, state),
                style="Panel.TCheckbutton",
            ).pack(anchor="w", pady=2)

    def _toggle_annotation(self, index: int, state: tk.BooleanVar) -> None:
        if self.current_result_index is None:
            return
        hidden = self.hidden_annotations.setdefault(self.current_result_index, set())
        hidden.discard(index) if state.get() else hidden.add(index)
        self._refresh_result()

    def _clear_results_ui(self) -> None:
        self.result_list.delete(*self.result_list.get_children())
        self.results_canvas.set_image(None)
        self.result_text.set("Run inference in Analysis to populate results.")
        for child in self.annotation_container.winfo_children():
            child.destroy()
        self.current_result_index = None
        self._update_training_button_state()

    def _update_run_state(self) -> None:
        state = "normal" if self.inference_engine and self.frames and not self.inference_running and not self.export_running else "disabled"
        self.run_button.configure(state=state)
        self.run_all_button.configure(state=state)
        self.run_selected_button.configure(state=state)
        export_state = "normal" if self.model_info and (self.results or self.inference_failures) and not self.inference_running and not self.export_running else "disabled"
        self.export_button.configure(state=export_state)
        current_export_state = "normal" if self.current_result_index in self.results and export_state == "normal" else "disabled"
        self.export_current_button.configure(state=current_export_state)
        self._update_training_button_state()

    def _set_activity(self, text: str, active: bool) -> None:
        self.activity_text.set(text)
        self.status_text.set(text)
        self.activity.start() if active else self.activity.stop()

    def _set_model_detail(self, text: str) -> None:
        self.model_detail.configure(state="normal")
        self.model_detail.delete("1.0", tk.END)
        self.model_detail.insert("1.0", text)
        self.model_detail.configure(state="disabled")

    @staticmethod
    def _metadata_percent(metadata: dict[str, str], key: str) -> str:
        try:
            return f"{float(metadata[key]):.1%}"
        except (KeyError, ValueError):
            return "Not embedded"

    def _log(self, message: str) -> None:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        line = f"{timestamp} | {message}"
        LOGGER.info(message)
        self.terminal.configure(state="normal")
        self.terminal.insert(tk.END, line + "\n")
        self.terminal.see(tk.END)
        self.terminal.configure(state="disabled")

    def _close(self) -> None:
        self.cancel_inference.set()
        self.resume_activity.set()
        self._close_preprocess_popup()
        self._close_annotation_popup()
        self.destroy()


def configure_logging() -> Path:
    directory = log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"app-{datetime.now().astimezone():%Y%m%d}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", handlers=[logging.FileHandler(log_path, encoding="utf-8")])
    return log_path


def main() -> int:
    log_path = configure_logging()
    try:
        app = EchoSightApp()
        app.mainloop()
        return 0
    except Exception as error:
        LOGGER.exception("EchoSight 2.0 failed to start")
        try:
            messagebox.showerror("EchoSight 2.0 startup error", f"{error}\n\nLog: {log_path}")
        except tk.TclError:
            pass
        return 1
