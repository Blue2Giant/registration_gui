import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import queue
import cv2
from PIL import Image, ImageTk

from ..core.config import load_config, save_config, AppConfig
from ..core.folder_pairs import find_pairs, parse_pairs_txt, ImagePair
from ..core.pipeline import RegistrationPipeline, TaskInputs, TaskOutputs

# --- Styles & Colors ---
BG_COLOR = "#F7F7F8"
ACCENT_COLOR = "#1C4F9C"  # Fudan Blue
TEXT_COLOR = "#1F2937"
HEADER_FONT = ("Segoe UI", 12, "bold")
NORMAL_FONT = ("Segoe UI", 10)
MONO_FONT = ("Consolas", 10)

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Registration Tool (Tkinter Fallback Mode)")
        self.geometry("1280x850")
        self.configure(bg=BG_COLOR)
        
        self.config = load_config()
        self.pairs: list[ImagePair] = []
        self.current_pipeline: RegistrationPipeline | None = None
        self.last_outputs: TaskOutputs | None = None
        self.cancel_flag = False

        self.compare_zoom: float | None = None
        self.compare_offset_x: float = 0.0
        self.compare_offset_y: float = 0.0
        self.compare_image_cache: dict[str, tuple[str, Image.Image]] = {}
        self.compare_canvas_image_id: int | None = None
        self.compare_canvas_photo: ImageTk.PhotoImage | None = None
        self.compare_pan_start: tuple[int, int, float, float] | None = None
        self.left_scroll_active: bool = False

        self.batch_active: bool = False
        self.batch_pairs: list[ImagePair] = []
        self.batch_index: int = 0
        self.batch_mode: str = ""
        self.batch_algo_entry = None
        self.batch_out_root: str = ""
        
        self.queue = queue.Queue()
        
        self._setup_styles()
        self._setup_ui()
        self._load_state()
        self.fixed_path_var.trace_add("write", lambda *args: self._update_thumbnail(self.fixed_path_var.get(), self.lbl_fixed_thumb))
        self.moving_path_var.trace_add("write", lambda *args: self._update_thumbnail(self.moving_path_var.get(), self.lbl_moving_thumb))
        
        self.after(100, self._process_queue)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=NORMAL_FONT)
        style.configure("TLabelframe", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TLabelframe.Label", background=BG_COLOR, foreground=TEXT_COLOR, font=HEADER_FONT)
        style.configure("TButton", font=NORMAL_FONT, padding=5)
        
        style.configure("Primary.TButton", background=ACCENT_COLOR, foreground="white")
        style.map("Primary.TButton", background=[("active", "#153E7E")]) # Darker Fudan Blue
        
        style.configure("Toggle.TButton", font=("Segoe UI", 10, "bold"), padding=8)

    def _setup_ui(self):
        # Header / Logo Area
        # Using Fudan Blue for the header background
        header_frame = tk.Frame(self, bg=ACCENT_COLOR, height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        
        # Container to center content or keep left aligned
        h_container = tk.Frame(header_frame, bg=ACCENT_COLOR)
        h_container.pack(side=tk.LEFT, padx=20, pady=15)

        # Title text first (White text on Blue bg)
        lbl_title = tk.Label(h_container, text="Image Registration Tool", font=("Segoe UI", 22, "bold"), bg=ACCENT_COLOR, fg="white")
        lbl_title.pack(side=tk.LEFT)

        # Logo after text
        # Ensure we look for logo in the correct absolute path or relative to this file
        # This file is in app/ui_tk/, so assets is in app/assets/
        logo_path = (Path(__file__).parent.parent / "assets" / "logo.png").resolve()
        
        self.logo_img = None
        if logo_path.exists():
            try:
                pil_img = Image.open(logo_path)
                h = 50
                w = int(pil_img.width * (h / pil_img.height))
                pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(pil_img)
                # Logo on Blue bg
                lbl_logo = tk.Label(h_container, image=self.logo_img, bg=ACCENT_COLOR)
                lbl_logo.pack(side=tk.LEFT, padx=(20, 0))
            except Exception as e:
                print(f"Failed to load logo: {e}")
        else:
            print(f"Logo not found at: {logo_path}")

        # Main Content
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.left_frame = ttk.Frame(self.paned, width=320)
        self.paned.add(self.left_frame, weight=1)
        
        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=4)
        
        self._setup_left_panel()
        self._setup_right_panel()

    def _setup_left_panel(self):
        scroll_host = ttk.Frame(self.left_frame)
        scroll_host.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.left_canvas = tk.Canvas(scroll_host, bg=BG_COLOR, highlightthickness=0)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=self.left_canvas.yview)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)

        container = ttk.Frame(self.left_canvas)
        left_window_id = self.left_canvas.create_window((0, 0), window=container, anchor="nw")

        def _sync_left_scroll_region(_event=None):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

        def _sync_left_width(event):
            self.left_canvas.itemconfigure(left_window_id, width=event.width)

        container.bind("<Configure>", _sync_left_scroll_region)
        self.left_canvas.bind("<Configure>", _sync_left_width)

        def _on_left_enter(_event):
            self.left_scroll_active = True

        def _on_left_leave(_event):
            self.left_scroll_active = False

        def _on_left_wheel(event):
            if not self.left_scroll_active:
                return
            if not self.left_canvas.winfo_viewable():
                return
            delta = event.delta
            if not delta:
                return
            steps = int(delta / 120)
            if steps == 0:
                steps = 1 if delta > 0 else -1
            self.left_canvas.yview_scroll(-steps, "units")

        scroll_host.bind("<Enter>", _on_left_enter)
        scroll_host.bind("<Leave>", _on_left_leave)
        self.bind_all("<MouseWheel>", _on_left_wheel, add="+")

        # Algorithm Selection
        exe_group = ttk.LabelFrame(container, text="Registration Algorithm")
        exe_group.pack(fill=tk.X, pady=5)
        
        self.algo_combo = ttk.Combobox(exe_group, state="readonly", font=NORMAL_FONT)
        self.algo_combo.pack(fill=tk.X, padx=10, pady=8)
        self.algo_combo.bind("<<ComboboxSelected>>", lambda e: self._update_algo_hint())

        hint_frame = tk.Frame(exe_group, bg="#EEF2FF")
        hint_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.lbl_algo_hint_title = tk.Label(
            hint_frame,
            text="Environment / Notes",
            bg="#EEF2FF",
            fg=ACCENT_COLOR,
            font=("Segoe UI", 9, "bold"),
        )
        self.lbl_algo_hint_title.pack(anchor="w", padx=8, pady=(6, 0))

        self.txt_algo_hint = tk.Text(
            hint_frame,
            height=4,
            wrap="word",
            font=("Segoe UI", 9),
            bg="#EEF2FF",
            fg="#111827",
            relief="flat",
            borderwidth=0,
            padx=8,
            pady=6,
        )
        self.txt_algo_hint.pack(fill=tk.X)
        self.txt_algo_hint.config(state=tk.DISABLED)

        # Input Mode
        input_group = ttk.LabelFrame(container, text="Input Data")
        input_group.pack(fill=tk.X, pady=10)
        
        self.mode_var = tk.StringVar(value="folder")
        radio_frame = ttk.Frame(input_group)
        radio_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Radiobutton(radio_frame, text="Folder Mode", variable=self.mode_var, value="folder", command=self._on_mode_change).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(radio_frame, text="TXT Pairs", variable=self.mode_var, value="txt", command=self._on_mode_change).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(radio_frame, text="Single Pair", variable=self.mode_var, value="pair", command=self._on_mode_change).pack(side=tk.LEFT)
        
        self.folder_frame = ttk.Frame(input_group)
        self.folder_path_var = tk.StringVar()
        
        f_entry_frame = ttk.Frame(self.folder_frame)
        f_entry_frame.pack(fill=tk.X)
        ttk.Entry(f_entry_frame, textvariable=self.folder_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(f_entry_frame, text="...", width=3, command=self._browse_folder).pack(side=tk.LEFT, padx=(5,0))

        ttk.Label(
            self.folder_frame,
            text="要求：文件名严格为 <key>_1 和 <key>_2（例如 pair1_1.jpg / pair1_2.jpg）",
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(5, 0))
        
        ttk.Label(self.folder_frame, text="Available Pairs:", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(5,0))
        
        list_frame = ttk.Frame(self.folder_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        self.pair_list = tk.Listbox(list_frame, height=8, font=NORMAL_FONT, bg="white", relief="flat", borderwidth=1)
        self.pair_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.pair_list.bind("<<ListboxSelect>>", self._on_pair_select)
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.pair_list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.pair_list.config(yscrollcommand=sb.set)

        thumbs_frame = ttk.Frame(self.folder_frame)
        thumbs_frame.pack(fill=tk.X, pady=(6, 0))
        self.lbl_folder_fixed_thumb = ttk.Label(thumbs_frame)
        self.lbl_folder_fixed_thumb.pack(side=tk.LEFT, padx=(0, 8))
        self.lbl_folder_moving_thumb = ttk.Label(thumbs_frame)
        self.lbl_folder_moving_thumb.pack(side=tk.LEFT)

        self.txt_frame = ttk.Frame(input_group)
        self.pairs_txt_path_var = tk.StringVar()

        t_entry_frame = ttk.Frame(self.txt_frame)
        t_entry_frame.pack(fill=tk.X)
        ttk.Entry(t_entry_frame, textvariable=self.pairs_txt_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(t_entry_frame, text="...", width=3, command=self._browse_pairs_txt).pack(side=tk.LEFT, padx=(5,0))

        ttk.Label(
            self.txt_frame,
            text="TXT 每行一对：fixed绝对路径,moving绝对路径（逗号分隔）",
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(5, 0))

        ttk.Label(self.txt_frame, text="Available Pairs:", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(5,0))

        t_list_frame = ttk.Frame(self.txt_frame)
        t_list_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        self.txt_pair_list = tk.Listbox(t_list_frame, height=8, font=NORMAL_FONT, bg="white", relief="flat", borderwidth=1)
        self.txt_pair_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.txt_pair_list.bind("<<ListboxSelect>>", self._on_txt_pair_select)
        t_sb = ttk.Scrollbar(t_list_frame, orient=tk.VERTICAL, command=self.txt_pair_list.yview)
        t_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_pair_list.config(yscrollcommand=t_sb.set)

        t_thumbs_frame = ttk.Frame(self.txt_frame)
        t_thumbs_frame.pack(fill=tk.X, pady=(6, 0))
        self.lbl_txt_fixed_thumb = ttk.Label(t_thumbs_frame)
        self.lbl_txt_fixed_thumb.pack(side=tk.LEFT, padx=(0, 8))
        self.lbl_txt_moving_thumb = ttk.Label(t_thumbs_frame)
        self.lbl_txt_moving_thumb.pack(side=tk.LEFT)
        
        self.pair_frame = ttk.Frame(input_group)
        self.fixed_path_var = tk.StringVar()
        self.moving_path_var = tk.StringVar()
        
        ttk.Label(self.pair_frame, text="Fixed (Target):").pack(anchor=tk.W)
        p1_frame = ttk.Frame(self.pair_frame)
        p1_frame.pack(fill=tk.X, pady=(0,2))
        ttk.Entry(p1_frame, textvariable=self.fixed_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(p1_frame, text="...", width=3, command=lambda: self._browse_file(self.fixed_path_var)).pack(side=tk.LEFT, padx=(5,0))
        
        self.lbl_fixed_thumb = ttk.Label(self.pair_frame)
        self.lbl_fixed_thumb.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(self.pair_frame, text="Moving (Source):").pack(anchor=tk.W)
        p2_frame = ttk.Frame(self.pair_frame)
        p2_frame.pack(fill=tk.X, pady=(0,2))
        ttk.Entry(p2_frame, textvariable=self.moving_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(p2_frame, text="...", width=3, command=lambda: self._browse_file(self.moving_path_var)).pack(side=tk.LEFT, padx=(5,0))

        self.lbl_moving_thumb = ttk.Label(self.pair_frame)
        self.lbl_moving_thumb.pack(anchor=tk.W, pady=(0, 5))

        # Output Dir
        out_group = ttk.LabelFrame(container, text="Output")
        out_group.pack(fill=tk.X, pady=5)
        self.out_path_var = tk.StringVar()
        o_frame = ttk.Frame(out_group)
        o_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Entry(o_frame, textvariable=self.out_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(o_frame, text="...", width=3, command=self._browse_output).pack(side=tk.LEFT, padx=(5,0))

        # Controls
        ctrl_group = ttk.Frame(container)
        ctrl_group.pack(fill=tk.X, pady=20)
        
        self.run_btn = ttk.Button(ctrl_group, text="▶ Run Registration", style="Primary.TButton", command=self._run_task)
        self.run_btn.pack(fill=tk.X, pady=2, ipady=5)

        self.batch_run_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ctrl_group,
            text="批量顺序运行全部 pairs（Folder/TXT）",
            variable=self.batch_run_all_var,
        ).pack(anchor=tk.W, pady=(6, 2))
        
        self.cancel_btn = ttk.Button(ctrl_group, text="Stop", state=tk.DISABLED, command=self._cancel_task)
        self.cancel_btn.pack(fill=tk.X, pady=2)

    def _setup_right_panel(self):
        self.right_paned = ttk.PanedWindow(self.right_frame, orient=tk.VERTICAL)
        self.right_paned.pack(fill=tk.BOTH, expand=True)
        
        # --- Custom Tabs Area ---
        tabs_container = ttk.Frame(self.right_paned)
        self.right_paned.add(tabs_container, weight=4)
        
        # Toggle Buttons Frame
        btn_bar = tk.Frame(tabs_container, bg=BG_COLOR)
        btn_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Distinct colors for the sections
        self.btn_matches = tk.Button(btn_bar, text="Matches Visualization", font=("Segoe UI", 10, "bold"),
                                     bg="#10B981", fg="white", relief="flat", padx=15, pady=5,
                                     command=lambda: self._switch_tab("matches"))
        self.btn_matches.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_fusion = tk.Button(btn_bar, text="Checkerboard Fusion", font=("Segoe UI", 10, "bold"),
                                    bg="#8B5CF6", fg="white", relief="flat", padx=15, pady=5,
                                    command=lambda: self._switch_tab("fusion"))
        self.btn_fusion.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_compare = tk.Button(btn_bar, text="Compare", font=("Segoe UI", 10, "bold"),
                                     bg="#F59E0B", fg="white", relief="flat", padx=15, pady=5,
                                     command=lambda: self._switch_tab("compare"))
        self.btn_compare.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_matrix = tk.Button(btn_bar, text="Transform Matrix", font=("Segoe UI", 10, "bold"),
                                    bg="#3B82F6", fg="white", relief="flat", padx=15, pady=5,
                                    command=lambda: self._switch_tab("matrix"))
        self.btn_matrix.pack(side=tk.LEFT)

        # Content Area (Stacked Frames)
        self.content_area = ttk.Frame(tabs_container)
        self.content_area.pack(fill=tk.BOTH, expand=True)
        
        self.frame_matches = ttk.Frame(self.content_area)
        self.lbl_matches_img = ttk.Label(self.frame_matches, text="No matches generated yet.", anchor="center")
        self.lbl_matches_img.pack(fill=tk.BOTH, expand=True)
        
        self.frame_fusion = ttk.Frame(self.content_area)
        self.lbl_checker_img = ttk.Label(self.frame_fusion, text="No fusion result generated yet.", anchor="center")
        self.lbl_checker_img.pack(fill=tk.BOTH, expand=True)

        self.frame_compare = ttk.Frame(self.content_area)
        self.compare_ctrl = tk.Frame(self.frame_compare, bg="#374151", padx=10, pady=5)
        self.compare_ctrl.place(relx=0.98, rely=0.02, anchor="ne")
        tk.Label(self.compare_ctrl, text="Compare (Up/Down to switch):", bg="#374151", fg="white", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.compare_layer_var = tk.StringVar(value="fixed")
        for layer_id, label in [("fixed", "Reference"), ("warped", "Warped")]:
            rb = tk.Radiobutton(
                self.compare_ctrl,
                text=label,
                variable=self.compare_layer_var,
                value=layer_id,
                bg="#374151",
                fg="white",
                selectcolor="#1C4F9C",
                activebackground="#374151",
                activeforeground="white",
                command=self._on_compare_layer_change,
            )
            rb.pack(side=tk.LEFT, padx=5)

        self.compare_canvas = tk.Canvas(self.frame_compare, bg="#0B1220", highlightthickness=0)
        self.compare_canvas.pack(fill=tk.BOTH, expand=True)
        self.compare_canvas.bind("<Configure>", self._on_compare_canvas_configure)
        self.compare_canvas.bind("<Enter>", lambda e: self.compare_canvas.focus_set())
        self.compare_canvas.bind("<MouseWheel>", self._on_compare_wheel)
        self.compare_canvas.bind("<ButtonPress-1>", self._on_compare_pan_start)
        self.compare_canvas.bind("<B1-Motion>", self._on_compare_pan_move)

        self.bind("<Up>", lambda e: self._cycle_compare_layer(-1))
        self.bind("<Down>", lambda e: self._cycle_compare_layer(1))

        self.frame_matrix = ttk.Frame(self.content_area)
        mat_container = ttk.Frame(self.frame_matrix)
        mat_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.txt_matrix = tk.Text(mat_container, font=MONO_FONT, bg="white", relief="flat", borderwidth=1, padx=10, pady=10)
        self.txt_matrix.pack(fill=tk.BOTH, expand=True)

        # Show default tab
        self._switch_tab("matches")

        # Lower Logs Panel
        log_frame = ttk.LabelFrame(self.right_paned, text="Execution Logs")
        self.right_paned.add(log_frame, weight=1)
        
        self.txt_log = tk.Text(log_frame, height=8, font=MONO_FONT, bg="#1E1E1E", fg="#D4D4D4", insertbackground="white", state=tk.DISABLED)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def _update_thumbnail(self, path, label):
        if not path or not Path(path).exists():
            label.config(image="", text="")
            return
        try:
            pil_img = Image.open(path)
            pil_img.thumbnail((120, 80))
            tk_img = ImageTk.PhotoImage(pil_img)
            label.config(image=tk_img, text="")
            label.image = tk_img
        except Exception:
            label.config(image="", text="Invalid image")

    def _get_compare_image(self, layer: str) -> Image.Image | None:
        if not self.last_outputs:
            return None

        if layer == "fixed":
            path = self.fixed_path_var.get()
        else:
            path = self.last_outputs.warped_path

        if not path or not Path(path).exists():
            return None

        cached = self.compare_image_cache.get(layer)
        if cached and cached[0] == path:
            return cached[1]

        try:
            img = Image.open(path)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            if img.mode == "RGBA":
                img = img.convert("RGB")
            self.compare_image_cache[layer] = (path, img)
            return img
        except Exception:
            return None

    def _fit_compare_view(self, img: Image.Image) -> None:
        cw = max(int(self.compare_canvas.winfo_width()), 1)
        ch = max(int(self.compare_canvas.winfo_height()), 1)
        iw, ih = img.size
        if iw <= 0 or ih <= 0:
            self.compare_zoom = 1.0
            self.compare_offset_x = 0.0
            self.compare_offset_y = 0.0
            return

        zoom = min(cw / iw, ch / ih)
        zoom = max(min(zoom, 8.0), 0.05)
        self.compare_zoom = float(zoom)
        self.compare_offset_x = (cw - iw * zoom) * 0.5
        self.compare_offset_y = (ch - ih * zoom) * 0.5

    def _render_compare_view(self) -> None:
        if not self.last_outputs:
            self.compare_canvas.delete("all")
            self.compare_canvas.create_text(
                self.compare_canvas.winfo_width() // 2,
                self.compare_canvas.winfo_height() // 2,
                text="No compare result generated yet.",
                fill="white",
                font=("Segoe UI", 12, "bold"),
            )
            return

        layer = self.compare_layer_var.get()
        img = self._get_compare_image(layer)
        if img is None:
            self.compare_canvas.delete("all")
            self.compare_canvas.create_text(
                self.compare_canvas.winfo_width() // 2,
                self.compare_canvas.winfo_height() // 2,
                text="Failed to load compare image.",
                fill="white",
                font=("Segoe UI", 12, "bold"),
            )
            return

        cw = int(self.compare_canvas.winfo_width())
        ch = int(self.compare_canvas.winfo_height())
        if cw < 20 or ch < 20:
            self.after(50, self._render_compare_view)
            return

        if self.compare_zoom is None:
            self._fit_compare_view(img)

        zoom = float(self.compare_zoom or 1.0)
        zoom = max(min(zoom, 8.0), 0.05)
        self.compare_zoom = zoom

        iw, ih = img.size
        ox = float(self.compare_offset_x)
        oy = float(self.compare_offset_y)

        ix0 = (0 - ox) / zoom
        iy0 = (0 - oy) / zoom
        ix1 = (cw - ox) / zoom
        iy1 = (ch - oy) / zoom

        ix0_i = int(max(0, min(iw, ix0)))
        iy0_i = int(max(0, min(ih, iy0)))
        ix1_i = int(max(0, min(iw, ix1)))
        iy1_i = int(max(0, min(ih, iy1)))

        self.compare_canvas.delete("all")
        if ix1_i <= ix0_i or iy1_i <= iy0_i:
            return

        crop = img.crop((ix0_i, iy0_i, ix1_i, iy1_i))
        target_w = max(1, int(round((ix1_i - ix0_i) * zoom)))
        target_h = max(1, int(round((iy1_i - iy0_i) * zoom)))
        if crop.size[0] != target_w or crop.size[1] != target_h:
            crop = crop.resize((target_w, target_h), Image.Resampling.LANCZOS)

        self.compare_canvas_photo = ImageTk.PhotoImage(crop)
        draw_x = ox + ix0_i * zoom
        draw_y = oy + iy0_i * zoom
        self.compare_canvas_image_id = self.compare_canvas.create_image(draw_x, draw_y, anchor="nw", image=self.compare_canvas_photo)

    def _on_compare_layer_change(self) -> None:
        if self.frame_compare.winfo_viewable():
            self._render_compare_view()

    def _on_compare_canvas_configure(self, event) -> None:
        if self.frame_compare.winfo_viewable():
            self._render_compare_view()

    def _on_compare_wheel(self, event) -> None:
        if not self.frame_compare.winfo_viewable():
            return
        img = self._get_compare_image(self.compare_layer_var.get())
        if img is None:
            return
        if self.compare_zoom is None:
            self._fit_compare_view(img)

        steps = int(event.delta / 120) if event.delta else 0
        if steps == 0:
            return
        factor = 1.15 ** steps
        old_zoom = float(self.compare_zoom or 1.0)
        new_zoom = max(min(old_zoom * factor, 8.0), 0.05)
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        mx = float(event.x)
        my = float(event.y)
        ix = (mx - self.compare_offset_x) / old_zoom
        iy = (my - self.compare_offset_y) / old_zoom
        self.compare_zoom = new_zoom
        self.compare_offset_x = mx - ix * new_zoom
        self.compare_offset_y = my - iy * new_zoom
        self._render_compare_view()

    def _on_compare_pan_start(self, event) -> None:
        self.compare_pan_start = (int(event.x), int(event.y), float(self.compare_offset_x), float(self.compare_offset_y))

    def _on_compare_pan_move(self, event) -> None:
        if self.compare_pan_start is None:
            return
        sx, sy, ox, oy = self.compare_pan_start
        self.compare_offset_x = ox + (float(event.x) - sx)
        self.compare_offset_y = oy + (float(event.y) - sy)
        self._render_compare_view()

    def _cycle_compare_layer(self, delta: int) -> None:
        if not self.frame_compare.winfo_viewable():
            return
        layers = ["fixed", "warped"]
        curr = self.compare_layer_var.get()
        idx = layers.index(curr) if curr in layers else 0
        new_idx = (idx + delta) % len(layers)
        self.compare_layer_var.set(layers[new_idx])
        self._render_compare_view()

    def _switch_tab(self, name):
        # Hide all
        self.frame_matches.pack_forget()
        self.frame_fusion.pack_forget()
        self.frame_compare.pack_forget()
        self.frame_matrix.pack_forget()
        
        # Reset button styles (optional, keeping simple flat colors)
        # Show selected
        if name == "matches":
            self.frame_matches.pack(fill=tk.BOTH, expand=True)
        elif name == "fusion":
            self.frame_fusion.pack(fill=tk.BOTH, expand=True)
        elif name == "compare":
            self.frame_compare.pack(fill=tk.BOTH, expand=True)
            self._render_compare_view()
        elif name == "matrix":
            self.frame_matrix.pack(fill=tk.BOTH, expand=True)

    def _load_state(self):
        values = [a.name for a in self.config.algorithms]
        self.algo_combo['values'] = values
        if values:
            self.algo_combo.current(0)
        self._update_algo_hint()
            
        self.mode_var.set(self.config.last_input_mode)
        self._on_mode_change()
        
        self.folder_path_var.set(self.config.last_folder)
        self.pairs_txt_path_var.set(self.config.last_pairs_txt)
        self.fixed_path_var.set(self.config.last_fixed)
        self.moving_path_var.set(self.config.last_moving)
        self.out_path_var.set(self.config.last_output_root)
        
        if self.config.last_input_mode == "folder" and self.config.last_folder:
            self._scan_folder()
        if self.config.last_input_mode == "txt" and self.config.last_pairs_txt:
            self._scan_pairs_txt()

    def _update_algo_hint(self) -> None:
        idx = self.algo_combo.current()
        hint = ""
        if idx >= 0 and idx < len(self.config.algorithms):
            hint = (self.config.algorithms[idx].env_hint or "").strip()
        if not hint:
            hint = "（未提供说明）"
        self.txt_algo_hint.config(state=tk.NORMAL)
        self.txt_algo_hint.delete("1.0", tk.END)
        self.txt_algo_hint.insert(tk.END, hint)
        self.txt_algo_hint.config(state=tk.DISABLED)

    def _on_mode_change(self):
        mode = self.mode_var.get()
        self.folder_frame.pack_forget()
        self.txt_frame.pack_forget()
        self.pair_frame.pack_forget()
        if mode == "folder":
            self.folder_frame.pack(fill=tk.X, padx=10, pady=5)
        elif mode == "txt":
            self.txt_frame.pack(fill=tk.X, padx=10, pady=5)
        else:
            self.pair_frame.pack(fill=tk.X, padx=10, pady=5)

    def _browse_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.folder_path_var.set(d)
            self.config.last_folder = d
            self._save_config()
            self._scan_folder()

    def _scan_folder(self):
        folder = self.folder_path_var.get()
        self.pairs = find_pairs(folder)
        self.pair_list.delete(0, tk.END)
        for p in self.pairs:
            self.pair_list.insert(tk.END, p.key)
        if self.pairs:
            self.pair_list.selection_clear(0, tk.END)
            self.pair_list.selection_set(0)
            self.pair_list.activate(0)
            self._on_pair_select(None)

    def _on_pair_select(self, event):
        sel = self.pair_list.curselection()
        if sel:
            pair = self.pairs[sel[0]]
            self._apply_selected_pair(pair, self.lbl_folder_fixed_thumb, self.lbl_folder_moving_thumb)

    def _browse_pairs_txt(self):
        f = filedialog.askopenfilename(title="Select pairs TXT", filetypes=[("Text", "*.txt"), ("All Files", "*.*")])
        if f:
            self.pairs_txt_path_var.set(f)
            self.config.last_pairs_txt = f
            self._save_config()
            self._scan_pairs_txt()

    def _scan_pairs_txt(self):
        txt_path = self.pairs_txt_path_var.get()
        self.pairs = parse_pairs_txt(txt_path)
        self.txt_pair_list.delete(0, tk.END)
        for p in self.pairs:
            self.txt_pair_list.insert(tk.END, p.key)
        if self.pairs:
            self.txt_pair_list.selection_clear(0, tk.END)
            self.txt_pair_list.selection_set(0)
            self.txt_pair_list.activate(0)
            self._on_txt_pair_select(None)

    def _on_txt_pair_select(self, event):
        sel = self.txt_pair_list.curselection()
        if sel:
            pair = self.pairs[sel[0]]
            self._apply_selected_pair(pair, self.lbl_txt_fixed_thumb, self.lbl_txt_moving_thumb)

    def _apply_selected_pair(self, pair: ImagePair, fixed_thumb: ttk.Label, moving_thumb: ttk.Label) -> None:
        self.fixed_path_var.set(pair.fixed_path)
        self.moving_path_var.set(pair.moving_path)
        self._update_thumbnail(pair.fixed_path, fixed_thumb)
        self._update_thumbnail(pair.moving_path, moving_thumb)

    def _browse_file(self, var):
        f = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.bmp;*.tif;*.tiff"), ("All Files", "*.*")])
        if f:
            var.set(f)
            self._save_config()

    def _browse_output(self):
        d = filedialog.askdirectory()
        if d:
            self.out_path_var.set(d)
            self.config.last_output_root = d
            self._save_config()

    def _save_config(self):
        self.config.last_input_mode = self.mode_var.get()
        self.config.last_folder = self.folder_path_var.get()
        self.config.last_pairs_txt = self.pairs_txt_path_var.get()
        self.config.last_fixed = self.fixed_path_var.get()
        self.config.last_moving = self.moving_path_var.get()
        save_config(self.config)

    def _log(self, msg):
        self.queue.put(("log", msg))

    def _process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                kind, data = msg
                if kind == "log":
                    self.txt_log.config(state=tk.NORMAL)
                    self.txt_log.insert(tk.END, data + "\n")
                    self.txt_log.see(tk.END)
                    self.txt_log.config(state=tk.DISABLED)
                elif kind == "success":
                    self._on_success(data)
                elif kind == "error":
                    self._on_error(data)
        except queue.Empty:
            pass
        self.after(100, self._process_queue)

    def _run_task(self):
        algo_idx = self.algo_combo.current()
        if algo_idx < 0:
            messagebox.showerror("Error", "No algorithm selected")
            return
        
        algo_entry = self.config.algorithms[algo_idx]
        out_root = self.out_path_var.get()
        
        mode = self.mode_var.get()
        if mode in ("folder", "txt") and bool(self.batch_run_all_var.get()):
            self._start_batch_run(mode, algo_entry, out_root)
            return

        fixed, moving, key = "", "", ""
        if mode == "folder":
            pair = self._get_selected_pair_or_error("folder")
            if pair is None:
                return
            fixed, moving, key = pair.fixed_path, pair.moving_path, pair.key
        elif mode == "txt":
            pair = self._get_selected_pair_or_error("txt")
            if pair is None:
                return
            fixed, moving, key = pair.fixed_path, pair.moving_path, pair.key
        else:
            fixed = self.fixed_path_var.get()
            moving = self.moving_path_var.get()
            key = "custom_pair"
            if not fixed or not moving:
                messagebox.showerror("Error", "Please select both images")
                return

        self._start_single_run(algo_entry, out_root, fixed, moving, key)

    def _start_single_run(self, algo_entry, out_root: str, fixed: str, moving: str, key: str) -> None:
        self.batch_active = False
        out_dir = str((Path(out_root) / algo_entry.name / key).resolve())
        inputs = TaskInputs(
            algo_name=algo_entry.name,
            command=algo_entry.command,
            command_cwd=algo_entry.cwd,
            algorithms_root=self.config.algorithms_root,
            fixed_path=fixed,
            moving_path=moving,
            output_dir=out_dir,
            repo_root=str(Path.cwd()),
            ransac_thresh_px=self.config.ransac_thresh_px,
            checker_tile_px=self.config.checker_tile_px,
            generate_matches_if_missing=self.config.generate_matches_if_missing,
        )
        self._start_pipeline(inputs, clear_log=True, header=f"Running: {key}")

    def _start_pipeline(self, inputs: TaskInputs, clear_log: bool, header: str | None) -> None:
        self.cancel_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.txt_log.config(state=tk.NORMAL)
        if clear_log:
            self.txt_log.delete(1.0, tk.END)
        if header:
            self.txt_log.insert(tk.END, header + "\n")
            self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

        def on_log(s): self._log(s)
        def on_success(o): self.queue.put(("success", o))
        def on_error(e): self.queue.put(("error", e))
        def is_cancelled(): return self.cancel_flag

        self.current_pipeline = RegistrationPipeline(inputs, on_log, on_success, on_error, is_cancelled)
        threading.Thread(target=self.current_pipeline.run, daemon=True).start()

    def _get_selected_pair_or_error(self, mode: str) -> ImagePair | None:
        if mode == "folder":
            sel = self.pair_list.curselection()
        else:
            sel = self.txt_pair_list.curselection()
        if not sel:
            messagebox.showerror("Error", "No pair selected")
            return None
        idx = sel[0]
        if idx < 0 or idx >= len(self.pairs):
            messagebox.showerror("Error", "Invalid selection")
            return None
        return self.pairs[idx]

    def _start_batch_run(self, mode: str, algo_entry, out_root: str) -> None:
        if not self.pairs:
            messagebox.showerror("Error", "No pairs loaded")
            return
        self.batch_active = True
        self.batch_mode = mode
        self.batch_pairs = list(self.pairs)
        self.batch_index = 0
        self.batch_algo_entry = algo_entry
        self.batch_out_root = out_root
        self._run_next_batch()

    def _run_next_batch(self) -> None:
        if not self.batch_active:
            self.run_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            return
        if self.cancel_flag:
            self.batch_active = False
            self.run_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            return
        if self.batch_index >= len(self.batch_pairs):
            self.batch_active = False
            self.run_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            self._log("Batch finished.")
            return

        pair = self.batch_pairs[self.batch_index]
        self._highlight_batch_selection(pair)

        out_dir = str((Path(self.batch_out_root) / self.batch_algo_entry.name / pair.key).resolve())
        inputs = TaskInputs(
            algo_name=self.batch_algo_entry.name,
            command=self.batch_algo_entry.command,
            command_cwd=self.batch_algo_entry.cwd,
            algorithms_root=self.config.algorithms_root,
            fixed_path=pair.fixed_path,
            moving_path=pair.moving_path,
            output_dir=out_dir,
            repo_root=str(Path.cwd()),
            ransac_thresh_px=self.config.ransac_thresh_px,
            checker_tile_px=self.config.checker_tile_px,
            generate_matches_if_missing=self.config.generate_matches_if_missing,
        )
        header = f"Batch {self.batch_index + 1}/{len(self.batch_pairs)}: {pair.key}"
        self._start_pipeline(inputs, clear_log=True, header=header)

    def _highlight_batch_selection(self, pair: ImagePair) -> None:
        if self.batch_mode == "folder":
            widget = self.pair_list
            fixed_thumb = self.lbl_folder_fixed_thumb
            moving_thumb = self.lbl_folder_moving_thumb
        else:
            widget = self.txt_pair_list
            fixed_thumb = self.lbl_txt_fixed_thumb
            moving_thumb = self.lbl_txt_moving_thumb
        for i in range(len(self.batch_pairs)):
            if self.batch_pairs[i].key == pair.key:
                widget.selection_clear(0, tk.END)
                widget.selection_set(i)
                widget.activate(i)
                widget.see(i)
                break
        self._apply_selected_pair(pair, fixed_thumb, moving_thumb)

    def _cancel_task(self):
        self.cancel_flag = True
        self.batch_active = False
        self._log("Cancelling...")

    def _on_success(self, out: TaskOutputs):
        self.last_outputs = out
        self.compare_image_cache = {}
        self.compare_zoom = None
        self.compare_offset_x = 0.0
        self.compare_offset_y = 0.0
        
        self._show_image(self.lbl_matches_img, out.matches_vis_path)
        self._show_image(self.lbl_checker_img, out.checkerboard_path)

        self.compare_layer_var.set("fixed")
        if self.frame_compare.winfo_viewable():
            self._render_compare_view()
        
        self.txt_matrix.delete(1.0, tk.END)
        self.txt_matrix.insert(tk.END, "RMSE: {:.4f}\nInliers: {}\n\n".format(out.rmse, out.inliers_count))
        self.txt_matrix.insert(tk.END, "Homography (Affine 3x3):\n")
        for row in out.H_3x3:
            self.txt_matrix.insert(tk.END, "[ {:.6f}, {:.6f}, {:.6f} ]\n".format(*row))

        if self.batch_active and not self.cancel_flag:
            self.batch_index += 1
            self.after(10, self._run_next_batch)
        else:
            self.run_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)

    def _on_error(self, err):
        self.batch_active = False
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        messagebox.showerror("Task Failed", err)

    def _show_image(self, label, path):
        try:
            cv_img = cv2.imread(path)
            if cv_img is None:
                raise ValueError("Could not read image")
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_img)
            
            # Smart resize
            w, h = pil_img.size
            
            # Get current available size in tab, fallback to defaults
            disp_w = self.content_area.winfo_width()
            disp_h = self.content_area.winfo_height()
            if disp_w < 100: disp_w = 800
            if disp_h < 100: disp_h = 600
            
            scale = min(disp_w/w, disp_h/h, 1.0)
            if scale < 1.0:
                pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
            
            tk_img = ImageTk.PhotoImage(pil_img)
            label.config(image=tk_img, text="")
            label.image = tk_img # Keep ref
        except Exception as e:
            label.config(text=f"Failed to load image: {e}", image="")
