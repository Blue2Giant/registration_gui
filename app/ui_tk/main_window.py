import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import threading
from pathlib import Path
import queue
import cv2
from PIL import Image, ImageTk
import numpy as np

from ..core.config import load_config, save_config, AppConfig
from ..core.folder_pairs import find_pairs, parse_pairs_txt, ImagePair
from ..core.manual_pipeline import ManualInputs, ManualRegistrationPipeline
from ..core.pipeline import RegistrationPipeline, TaskInputs, TaskOutputs

# --- Styles & Colors ---
BG_COLOR = "#F7F7F8"
ACCENT_COLOR = "#1C4F9C"  # Fudan Blue
TEXT_COLOR = "#1F2937"
HEADER_FONT = ("Segoe UI", 12, "bold")
NORMAL_FONT = ("Segoe UI", 10)
MONO_FONT = ("Consolas", 10)

class MainWindow(tk.Tk):
    def __init__(self, show_logo: bool = True):
        super().__init__()
        self.show_logo = show_logo
        self.title("Registration Tool (Tkinter Fallback Mode)")
        self.geometry("1280x850")
        self.configure(bg=BG_COLOR)
        
        self.config = load_config()
        self.pairs: list[ImagePair] = []
        self.current_pipeline: RegistrationPipeline | None = None
        self.last_outputs: TaskOutputs | None = None
        self.cancel_flag = False
        self._ransac_syncing = False

        self.compare_zoom: float | None = None
        self.compare_offset_x: float = 0.0
        self.compare_offset_y: float = 0.0
        self.compare_image_cache: dict[str, tuple[str, Image.Image]] = {}
        self.compare_canvas_image_id: int | None = None
        self.compare_canvas_photo: ImageTk.PhotoImage | None = None
        self.compare_pan_start: tuple[int, int, float, float] | None = None
        self.current_tab: str = "matches"
        self.left_scroll_active: bool = False
        
        # Font Scaling
        self.font_map: dict[tuple, tkfont.Font] = {}
        self.font_scale: float = 1.0
        self._ui_scale_min: float = 0.6
        self._ui_scale_max: float = 2.5

        self.batch_active: bool = False
        self.batch_pairs: list[ImagePair] = []
        self.batch_index: int = 0
        self.batch_mode: str = ""
        self.batch_algo_entry = None
        self.batch_out_root: str = ""

        self.manual_fixed_img: Image.Image | None = None
        self.manual_moving_img: Image.Image | None = None
        self.manual_fixed_photo: ImageTk.PhotoImage | None = None
        self.manual_moving_photo: ImageTk.PhotoImage | None = None
        self.manual_fixed_zoom: float | None = None
        self.manual_fixed_offset_x: float = 0.0
        self.manual_fixed_offset_y: float = 0.0
        self.manual_moving_zoom: float | None = None
        self.manual_moving_offset_x: float = 0.0
        self.manual_moving_offset_y: float = 0.0
        self.manual_fixed_pan_start: tuple[int, int, float, float] | None = None
        self.manual_moving_pan_start: tuple[int, int, float, float] | None = None
        self.manual_points_fixed: list[tuple[float, float]] = []
        self.manual_points_moving: list[tuple[float, float]] = []
        self.manual_pending_fixed: tuple[float, float] | None = None
        self.manual_pending_moving: tuple[float, float] | None = None
        
        self.queue = queue.Queue()
        
        self._setup_styles()
        self._setup_ui()
        self._load_state()
        self.bind_all("<Control-0>", lambda _e: self._reset_ui_scale(), add="+")
        self.fixed_path_var.trace_add("write", lambda *args: self._update_thumbnail(self.fixed_path_var.get(), self.lbl_fixed_thumb))
        self.moving_path_var.trace_add("write", lambda *args: self._update_thumbnail(self.moving_path_var.get(), self.lbl_moving_thumb))
        
        self.after(100, self._process_queue)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=self._get_font("Segoe UI", 10))
        style.configure("TLabelframe", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TLabelframe.Label", background=BG_COLOR, foreground=TEXT_COLOR, font=self._get_font("Segoe UI", 12, "bold"))
        style.configure("TButton", font=self._get_font("Segoe UI", 10), padding=5)
        
        style.configure("Primary.TButton", background=ACCENT_COLOR, foreground="white")
        style.map("Primary.TButton", background=[("active", "#153E7E")]) # Darker Fudan Blue
        
        style.configure("Toggle.TButton", font=self._get_font("Segoe UI", 10, "bold"), padding=8)
        style.configure("RansacParam.TLabel", background=BG_COLOR, foreground=ACCENT_COLOR, font=self._get_font("Segoe UI", 10, "bold"))
        style.configure("RansacHelp.TLabel", background=BG_COLOR, foreground="#6B7280", font=self._get_font("Segoe UI", 8))
        style.configure(
            "Ransac.TSpinbox",
            foreground=ACCENT_COLOR,
            fieldbackground="white",
            font=self._get_font("Consolas", 10, "bold"),
            padding=2,
        )

    def _is_ctrl_pressed(self, event) -> bool:
        return bool(int(getattr(event, "state", 0)) & 0x0004)

    def _get_font(self, family: str, size: int, weight: str = "normal") -> tkfont.Font:
        key = (family, size, weight)
        if key not in self.font_map:
            current_size = int(size * self.font_scale)
            if current_size < 1: current_size = 1
            # Note: size in points (positive)
            f = tkfont.Font(family=family, size=current_size, weight=weight)
            self.font_map[key] = f
        return self.font_map[key]

    def _ui_zoom_steps(self, steps: int) -> None:
        if steps == 0:
            return
        
        factor = 1.10 ** int(steps)
        new_scale = float(self.font_scale * factor)
        new_scale = max(self._ui_scale_min, min(self._ui_scale_max, new_scale))
        
        if abs(new_scale - self.font_scale) < 1e-6:
            return
            
        self.font_scale = new_scale
        for (family, base_size, weight), font_obj in self.font_map.items():
            s = int(base_size * self.font_scale)
            if s < 1: s = 1
            font_obj.configure(size=s)
            
        self.after_idle(self._refresh_after_ui_scale)

    def _reset_ui_scale(self) -> None:
        self.font_scale = 1.0
        for (family, base_size, weight), font_obj in self.font_map.items():
            font_obj.configure(size=base_size)
        self.after_idle(self._refresh_after_ui_scale)

    def _refresh_after_ui_scale(self) -> None:
        try:
            self._render_compare_view()
        except Exception:
            pass
        try:
            self._manual_render()
        except Exception:
            pass

    def _setup_ui(self):
        # Header / Logo Area
        # Using Fudan Blue for the header background
        header_frame = tk.Frame(self, bg=ACCENT_COLOR, height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        
        # Container to center content or keep left aligned
        h_container = tk.Frame(header_frame, bg=ACCENT_COLOR)
        h_container.pack(side=tk.LEFT, padx=20, pady=15)

        # Title text first (White text on Blue bg)
        lbl_title = tk.Label(h_container, text="Keypoint Based Image Registration Tool", font=self._get_font("Segoe UI", 22, "bold"), bg=ACCENT_COLOR, fg="white")
        lbl_title.pack(side=tk.LEFT)

        # Logo after text
        # Ensure we look for logo in the correct absolute path or relative to this file
        # This file is in app/ui_tk/, so assets is in app/assets/
        if self.show_logo:
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
            delta = event.delta
            if not delta:
                return
            steps = int(delta / 120)
            if steps == 0:
                steps = 1 if delta > 0 else -1
            if self._is_ctrl_pressed(event):
                self._ui_zoom_steps(steps)
                return "break"
            if not self.left_scroll_active:
                return
            if not self.left_canvas.winfo_viewable():
                return
            self.left_canvas.yview_scroll(-steps, "units")

        scroll_host.bind("<Enter>", _on_left_enter)
        scroll_host.bind("<Leave>", _on_left_leave)
        self.bind_all("<MouseWheel>", _on_left_wheel, add="+")

        # Algorithm Selection
        exe_group = ttk.LabelFrame(container, text="Registration Algorithm")
        exe_group.pack(fill=tk.X, pady=5)
        
        self.algo_combo = ttk.Combobox(exe_group, state="readonly", font=self._get_font("Segoe UI", 10))
        self.algo_combo.pack(fill=tk.X, padx=10, pady=8)
        self.algo_combo.bind("<<ComboboxSelected>>", self._on_algorithm_changed)

        hint_frame = tk.Frame(exe_group, bg="#EEF2FF")
        hint_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.lbl_algo_hint_title = tk.Label(
            hint_frame,
            text="Environment / Notes",
            bg="#EEF2FF",
            fg=ACCENT_COLOR,
            font=self._get_font("Segoe UI", 9, "bold"),
        )
        self.lbl_algo_hint_title.pack(anchor="w", padx=8, pady=(6, 0))

        self.txt_algo_hint = tk.Text(
            hint_frame,
            height=4,
            wrap="word",
            font=self._get_font("Segoe UI", 9),
            bg="#EEF2FF",
            fg="#111827",
            relief="flat",
            borderwidth=0,
            padx=8,
            pady=6,
        )
        self.txt_algo_hint.pack(fill=tk.X)
        self.txt_algo_hint.config(state=tk.DISABLED)

        transform_group = ttk.LabelFrame(container, text="Transform Model")
        transform_group.pack(fill=tk.X, pady=5)
        self.transform_model_var = tk.StringVar(value="affine")
        self.transform_combo = ttk.Combobox(
            transform_group,
            state="readonly",
            font=self._get_font("Segoe UI", 10),
            values=["affine", "homography", "fsc-affine", "fsc-perspective"],
            textvariable=self.transform_model_var,
        )
        self.transform_combo.pack(fill=tk.X, padx=10, pady=8)
        self.transform_combo.bind("<<ComboboxSelected>>", lambda e: (self._update_ransac_controls_state(), self._save_config()))
        self.transform_model_var.trace_add("write", lambda *args: self._update_ransac_controls_state())

        ransac_group = ttk.LabelFrame(container, text="RANSAC (OpenCV)")
        ransac_group.pack(fill=tk.X, pady=5)

        self.ransac_thresh_var = tk.StringVar(value=f"{float(self.config.ransac_thresh_px):g}")
        self.ransac_max_iters_var = tk.StringVar(value=str(int(self.config.ransac_max_iters)))
        self.ransac_confidence_var = tk.StringVar(value=f"{float(self.config.ransac_confidence):.4f}")
        self.ransac_refine_iters_var = tk.StringVar(value=str(int(self.config.ransac_refine_iters)))

        r1 = ttk.Frame(ransac_group)
        r1.pack(fill=tk.X, padx=10, pady=(8, 0))
        ttk.Label(r1, text="阈值 thresh (px)", style="RansacParam.TLabel").pack(side=tk.LEFT)
        self.spin_ransac_thresh = ttk.Spinbox(
            r1,
            from_=0.1,
            to=9999.0,
            increment=0.5,
            textvariable=self.ransac_thresh_var,
            width=12,
            command=self._on_ransac_params_commit,
            style="Ransac.TSpinbox",
        )
        self.spin_ransac_thresh.pack(side=tk.RIGHT)
        self.spin_ransac_thresh.bind("<Return>", lambda _e: self._on_ransac_params_commit())
        self.spin_ransac_thresh.bind("<FocusOut>", lambda _e: self._on_ransac_params_commit())
        ttk.Label(
            ransac_group,
            text="决定“内点”的最大重投影误差阈值（像素）。增大→内点更多但可能更不准；减小→更严格但可能失败。",
            style="RansacHelp.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=12, pady=(0, 6))

        r2 = ttk.Frame(ransac_group)
        r2.pack(fill=tk.X, padx=10, pady=(2, 0))
        ttk.Label(r2, text="maxIters", style="RansacParam.TLabel").pack(side=tk.LEFT)
        self.spin_ransac_max_iters = ttk.Spinbox(
            r2,
            from_=1,
            to=200000,
            increment=100,
            textvariable=self.ransac_max_iters_var,
            width=12,
            command=self._on_ransac_params_commit,
            style="Ransac.TSpinbox",
        )
        self.spin_ransac_max_iters.pack(side=tk.RIGHT)
        self.spin_ransac_max_iters.bind("<Return>", lambda _e: self._on_ransac_params_commit())
        self.spin_ransac_max_iters.bind("<FocusOut>", lambda _e: self._on_ransac_params_commit())
        ttk.Label(
            ransac_group,
            text="RANSAC 最大迭代次数。增大→更稳但更慢；减小→更快但在外点多时更易失败。",
            style="RansacHelp.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=12, pady=(0, 6))

        r3 = ttk.Frame(ransac_group)
        r3.pack(fill=tk.X, padx=10, pady=(2, 0))
        ttk.Label(r3, text="confidence", style="RansacParam.TLabel").pack(side=tk.LEFT)
        self.spin_ransac_confidence = ttk.Spinbox(
            r3,
            from_=0.50,
            to=0.9999,
            increment=0.001,
            format="%.4f",
            textvariable=self.ransac_confidence_var,
            width=12,
            command=self._on_ransac_params_commit,
            style="Ransac.TSpinbox",
        )
        self.spin_ransac_confidence.pack(side=tk.RIGHT)
        self.spin_ransac_confidence.bind("<Return>", lambda _e: self._on_ransac_params_commit())
        self.spin_ransac_confidence.bind("<FocusOut>", lambda _e: self._on_ransac_params_commit())
        ttk.Label(
            ransac_group,
            text="成功概率期望。更接近 1 通常更稳但更慢；过低可能更快但更容易不准/失败。",
            style="RansacHelp.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=12, pady=(0, 6))

        r4 = ttk.Frame(ransac_group)
        r4.pack(fill=tk.X, padx=10, pady=(2, 0))
        ttk.Label(r4, text="refineIters (affine)", style="RansacParam.TLabel").pack(side=tk.LEFT)
        self.spin_ransac_refine_iters = ttk.Spinbox(
            r4,
            from_=0,
            to=1000,
            increment=1,
            textvariable=self.ransac_refine_iters_var,
            width=12,
            command=self._on_ransac_params_commit,
            style="Ransac.TSpinbox",
        )
        self.spin_ransac_refine_iters.pack(side=tk.RIGHT)
        self.spin_ransac_refine_iters.bind("<Return>", lambda _e: self._on_ransac_params_commit())
        self.spin_ransac_refine_iters.bind("<FocusOut>", lambda _e: self._on_ransac_params_commit())
        ttk.Label(
            ransac_group,
            text="仅 affine 使用：在内点上做迭代优化。增大→可能更准但更慢。",
            style="RansacHelp.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=12, pady=(0, 6))

        btn_row = ttk.Frame(ransac_group)
        btn_row.pack(fill=tk.X, padx=10, pady=(2, 8))
        ttk.Button(btn_row, text="Reset Defaults", command=self._reset_ransac_defaults).pack(side=tk.RIGHT)
        self._update_ransac_controls_state()

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
            font=self._get_font("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(5, 0))
        
        ttk.Label(self.folder_frame, text="Available Pairs:", font=self._get_font("Segoe UI", 9)).pack(anchor=tk.W, pady=(5,0))
        
        list_frame = ttk.Frame(self.folder_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        self.pair_list = tk.Listbox(list_frame, height=8, font=self._get_font("Segoe UI", 10), bg="white", relief="flat", borderwidth=1)
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
            font=self._get_font("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(5, 0))

        ttk.Label(self.txt_frame, text="Available Pairs:", font=self._get_font("Segoe UI", 9)).pack(anchor=tk.W, pady=(5,0))

        t_list_frame = ttk.Frame(self.txt_frame)
        t_list_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        self.txt_pair_list = tk.Listbox(t_list_frame, height=8, font=self._get_font("Segoe UI", 10), bg="white", relief="flat", borderwidth=1)
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

        manual_group = ttk.LabelFrame(container, text="Manual Matches")
        manual_group.pack(fill=tk.BOTH, expand=False, pady=5)
        self.txt_manual_points = tk.Text(
            manual_group,
            height=8,
            font=self._get_font("Consolas", 10),
            bg="white",
            relief="flat",
            borderwidth=1,
            padx=8,
            pady=6,
        )
        self.txt_manual_points.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        self.txt_manual_points.config(state=tk.DISABLED)
        self._manual_update_points_text()

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
        self.btn_matches = tk.Button(btn_bar, text="Matches Visualization", font=self._get_font("Segoe UI", 10, "bold"),
                                     bg="#10B981", fg="white", relief="flat", padx=15, pady=5,
                                     command=lambda: self._switch_tab("matches"))
        self.btn_matches.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_fusion = tk.Button(btn_bar, text="Checkerboard Fusion", font=self._get_font("Segoe UI", 10, "bold"),
                                    bg="#8B5CF6", fg="white", relief="flat", padx=15, pady=5,
                                    command=lambda: self._switch_tab("fusion"))
        self.btn_fusion.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_compare = tk.Button(btn_bar, text="Compare", font=self._get_font("Segoe UI", 10, "bold"),
                                     bg="#F59E0B", fg="white", relief="flat", padx=15, pady=5,
                                     command=lambda: self._switch_tab("compare"))
        self.btn_compare.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_matrix = tk.Button(btn_bar, text="Transform Matrix", font=self._get_font("Segoe UI", 10, "bold"),
                                    bg="#3B82F6", fg="white", relief="flat", padx=15, pady=5,
                                    command=lambda: self._switch_tab("matrix"))
        self.btn_matrix.pack(side=tk.LEFT)

        self.btn_manual = tk.Button(btn_bar, text="Manual", font=self._get_font("Segoe UI", 10, "bold"),
                                    bg="#EF4444", fg="white", relief="flat", padx=15, pady=5,
                                    command=lambda: self._switch_tab("manual"))
        self.btn_manual.pack(side=tk.LEFT, padx=(5, 0))

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
        tk.Label(self.compare_ctrl, text="Compare (Up/Down to switch):", bg="#374151", fg="white", font=self._get_font("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))

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
        self.compare_warped_radio = rb

        self.frame_matrix = ttk.Frame(self.content_area)
        mat_container = ttk.Frame(self.frame_matrix)
        mat_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.txt_matrix = tk.Text(mat_container, font=self._get_font("Consolas", 10), bg="white", relief="flat", borderwidth=1, padx=10, pady=10)
        self.txt_matrix.pack(fill=tk.BOTH, expand=True)

        self.frame_manual = ttk.Frame(self.content_area)
        manual_bar = tk.Frame(self.frame_manual, bg="#111827", padx=10, pady=6)
        manual_bar.pack(fill=tk.X)
        tk.Label(
            manual_bar,
            text="手动配准：先点左图一个点，再点右图一个点生成一对匹配（双击点可删除）。",
            bg="#111827",
            fg="white",
            font=self._get_font("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Button(manual_bar, text="Reload", command=self._manual_reload_pair).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(manual_bar, text="Clear", command=self._manual_clear_points).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(manual_bar, text="Apply", command=self._manual_apply).pack(side=tk.RIGHT)

        manual_paned = ttk.PanedWindow(self.frame_manual, orient=tk.HORIZONTAL)
        manual_paned.pack(fill=tk.BOTH, expand=True)
        self.manual_canvas_fixed = tk.Canvas(manual_paned, bg="black", highlightthickness=0)
        self.manual_canvas_moving = tk.Canvas(manual_paned, bg="black", highlightthickness=0)
        manual_paned.add(self.manual_canvas_fixed, weight=1)
        manual_paned.add(self.manual_canvas_moving, weight=1)

        self.manual_canvas_fixed.bind("<Configure>", lambda e: self._manual_render())
        self.manual_canvas_moving.bind("<Configure>", lambda e: self._manual_render())
        self.manual_canvas_fixed.bind("<Button-1>", lambda e: self._manual_on_click("fixed", e.x, e.y))
        self.manual_canvas_moving.bind("<Button-1>", lambda e: self._manual_on_click("moving", e.x, e.y))
        self.manual_canvas_fixed.bind("<Double-Button-1>", lambda e: self._manual_on_double_click("fixed", e.x, e.y))
        self.manual_canvas_moving.bind("<Double-Button-1>", lambda e: self._manual_on_double_click("moving", e.x, e.y))
        self.manual_canvas_fixed.bind("<MouseWheel>", lambda e: self._manual_on_wheel("fixed", e))
        self.manual_canvas_moving.bind("<MouseWheel>", lambda e: self._manual_on_wheel("moving", e))
        self.manual_canvas_fixed.bind("<ButtonPress-3>", lambda e: self._manual_on_pan_start("fixed", e))
        self.manual_canvas_moving.bind("<ButtonPress-3>", lambda e: self._manual_on_pan_start("moving", e))
        self.manual_canvas_fixed.bind("<B3-Motion>", lambda e: self._manual_on_pan_move("fixed", e))
        self.manual_canvas_moving.bind("<B3-Motion>", lambda e: self._manual_on_pan_move("moving", e))

        # Show default tab
        self._switch_tab("matches")

        # Lower Logs Panel
        log_frame = ttk.LabelFrame(self.right_paned, text="Execution Logs")
        self.right_paned.add(log_frame, weight=1)
        
        self.txt_log = tk.Text(log_frame, height=8, font=self._get_font("Consolas", 10), bg="#1E1E1E", fg="#D4D4D4", insertbackground="white", state=tk.DISABLED)
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
            path = getattr(self.last_outputs, "compare_path", "") or self.last_outputs.warped_path

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

    def _update_compare_controls(self) -> None:
        label = "Warped"
        if self.last_outputs and getattr(self.last_outputs, "result_source", "") == "native_wssf":
            compare_path = getattr(self.last_outputs, "compare_path", "") or ""
            warped_path = getattr(self.last_outputs, "warped_path", "") or ""
            if compare_path and compare_path != warped_path:
                label = "Fusion"
        self.compare_warped_radio.config(text=label)

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
                font=self._get_font("Segoe UI", 12, "bold"),
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
                font=self._get_font("Segoe UI", 12, "bold"),
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
        if self._is_ctrl_pressed(event):
            return
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

    def _manual_reload_pair(self) -> None:
        fixed = self.fixed_path_var.get()
        moving = self.moving_path_var.get()
        if not fixed or not Path(fixed).exists() or not moving or not Path(moving).exists():
            self.manual_fixed_img = None
            self.manual_moving_img = None
            self.manual_fixed_zoom = None
            self.manual_fixed_offset_x = 0.0
            self.manual_fixed_offset_y = 0.0
            self.manual_moving_zoom = None
            self.manual_moving_offset_x = 0.0
            self.manual_moving_offset_y = 0.0
            self._manual_clear_points()
            self._manual_render()
            return
        try:
            self.manual_fixed_img = Image.open(fixed)
            self.manual_moving_img = Image.open(moving)
            if self.manual_fixed_img.mode not in ("RGB", "RGBA"):
                self.manual_fixed_img = self.manual_fixed_img.convert("RGB")
            if self.manual_fixed_img.mode == "RGBA":
                self.manual_fixed_img = self.manual_fixed_img.convert("RGB")
            if self.manual_moving_img.mode not in ("RGB", "RGBA"):
                self.manual_moving_img = self.manual_moving_img.convert("RGB")
            if self.manual_moving_img.mode == "RGBA":
                self.manual_moving_img = self.manual_moving_img.convert("RGB")
        except Exception:
            self.manual_fixed_img = None
            self.manual_moving_img = None
        self.manual_fixed_zoom = None
        self.manual_fixed_offset_x = 0.0
        self.manual_fixed_offset_y = 0.0
        self.manual_moving_zoom = None
        self.manual_moving_offset_x = 0.0
        self.manual_moving_offset_y = 0.0
        self._manual_clear_points()
        self._manual_render()

    def _manual_clear_points(self) -> None:
        self.manual_points_fixed = []
        self.manual_points_moving = []
        self.manual_pending_fixed = None
        self.manual_pending_moving = None
        self._manual_update_points_text()
        self._manual_render()

    def _manual_apply(self) -> None:
        if hasattr(self, "ransac_thresh_var"):
            self._on_ransac_params_commit()
        if self.manual_fixed_img is None or self.manual_moving_img is None:
            messagebox.showerror("Error", "Please select a valid pair first.")
            return
        if len(self.manual_points_fixed) < 4 or len(self.manual_points_moving) < 4:
            messagebox.showwarning("匹配点不足", "至少需要 4 对匹配点才能估计变换，请继续标注。")
            return
        fixed, moving, key = self._get_current_pair_paths_and_key()
        out_root = self.out_path_var.get()
        out_dir = str((Path(out_root) / "MANUAL" / key).resolve())
        inputs = ManualInputs(
            fixed_path=fixed,
            moving_path=moving,
            output_dir=out_dir,
            transform_model=self.transform_model_var.get(),
            ransac_thresh_px=self.config.ransac_thresh_px,
            ransac_max_iters=self.config.ransac_max_iters,
            ransac_confidence=self.config.ransac_confidence,
            ransac_refine_iters=self.config.ransac_refine_iters,
            checker_tile_px=self.config.checker_tile_px,
            points_fixed=np.asarray(self.manual_points_fixed, dtype=np.float32),
            points_moving=np.asarray(self.manual_points_moving, dtype=np.float32),
        )
        self._start_manual_pipeline(inputs, header=f"Manual: {key}")

    def _get_current_pair_paths_and_key(self) -> tuple[str, str, str]:
        mode = self.mode_var.get()
        if mode == "folder":
            sel = self.pair_list.curselection()
            if sel and sel[0] < len(self.pairs):
                pair = self.pairs[sel[0]]
                return pair.fixed_path, pair.moving_path, pair.key
        if mode == "txt":
            sel = self.txt_pair_list.curselection()
            if sel and sel[0] < len(self.pairs):
                pair = self.pairs[sel[0]]
                return pair.fixed_path, pair.moving_path, pair.key
        fixed = self.fixed_path_var.get()
        moving = self.moving_path_var.get()
        return fixed, moving, "custom_pair"

    def _start_manual_pipeline(self, inputs: ManualInputs, header: str) -> None:
        self.batch_active = False
        self.cancel_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.delete(1.0, tk.END)
        self.txt_log.insert(tk.END, header + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

        def on_log(s): self._log(s)
        def on_success(o): self.queue.put(("success", o))
        def on_error(e): self.queue.put(("error", e))
        def is_cancelled(): return self.cancel_flag

        p = ManualRegistrationPipeline(inputs, on_log, on_success, on_error, is_cancelled)
        threading.Thread(target=p.run, daemon=True).start()

    def _manual_on_click(self, side: str, cx: int, cy: int) -> None:
        if side == "fixed":
            img = self.manual_fixed_img
            zoom, ox, oy = self._manual_get_view("fixed")
        else:
            img = self.manual_moving_img
            zoom, ox, oy = self._manual_get_view("moving")
        if img is None:
            return
        if zoom is None or zoom <= 0:
            self._manual_fit_view(side)
            zoom, ox, oy = self._manual_get_view(side)
        ix = (cx - ox) / zoom
        iy = (cy - oy) / zoom
        if ix < 0 or iy < 0 or ix >= img.size[0] or iy >= img.size[1]:
            return
        pt = (float(ix), float(iy))
        if side == "fixed":
            self.manual_pending_fixed = pt
        else:
            self.manual_pending_moving = pt
        if self.manual_pending_fixed is not None and self.manual_pending_moving is not None:
            self.manual_points_fixed.append(self.manual_pending_fixed)
            self.manual_points_moving.append(self.manual_pending_moving)
            self.manual_pending_fixed = None
            self.manual_pending_moving = None
        self._manual_update_points_text()
        self._manual_render()

    def _manual_on_double_click(self, side: str, cx: int, cy: int) -> None:
        if side == "fixed":
            img = self.manual_fixed_img
            pts = self.manual_points_fixed
            pending = self.manual_pending_fixed
        else:
            img = self.manual_moving_img
            pts = self.manual_points_moving
            pending = self.manual_pending_moving
        if img is None:
            return

        zoom, ox, oy = self._manual_get_view(side)
        if zoom is None or zoom <= 0:
            self._manual_fit_view(side)
            zoom, ox, oy = self._manual_get_view(side)

        if pending is not None:
            px = pending[0] * zoom + ox
            py = pending[1] * zoom + oy
            if (px - cx) ** 2 + (py - cy) ** 2 <= 12 ** 2:
                if side == "fixed":
                    self.manual_pending_fixed = None
                else:
                    self.manual_pending_moving = None
                self._manual_update_points_text()
                self._manual_render()
                return

        best_i = -1
        best_d2 = 0.0
        for i, (x, y) in enumerate(pts):
            px = x * zoom + ox
            py = y * zoom + oy
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if best_i < 0 or d2 < best_d2:
                best_i = i
                best_d2 = d2
        if best_i >= 0 and best_d2 <= 12 ** 2:
            del self.manual_points_fixed[best_i]
            del self.manual_points_moving[best_i]
        self._manual_update_points_text()
        self._manual_render()

    def _manual_update_points_text(self) -> None:
        lines = []
        lines.append(f"匹配对数: {len(self.manual_points_fixed)}")
        if len(self.manual_points_fixed) < 4:
            lines.append("至少需要 4 对才能应用")
        if self.manual_pending_fixed is not None:
            lines.append(f"左图待配: ({self.manual_pending_fixed[0]:.1f}, {self.manual_pending_fixed[1]:.1f})")
        if self.manual_pending_moving is not None:
            lines.append(f"右图待配: ({self.manual_pending_moving[0]:.1f}, {self.manual_pending_moving[1]:.1f})")
        lines.append("")
        for i, (a, b) in enumerate(zip(self.manual_points_fixed, self.manual_points_moving), start=1):
            lines.append(f"{i:02d}  ({a[0]:.1f}, {a[1]:.1f})  ->  ({b[0]:.1f}, {b[1]:.1f})")
        self.txt_manual_points.config(state=tk.NORMAL)
        self.txt_manual_points.delete("1.0", tk.END)
        self.txt_manual_points.insert(tk.END, "\n".join(lines))
        self.txt_manual_points.config(state=tk.DISABLED)

    def _manual_render(self) -> None:
        self._manual_render_one("fixed")
        self._manual_render_one("moving")

    def _manual_get_view(self, side: str) -> tuple[float | None, float, float]:
        if side == "fixed":
            return self.manual_fixed_zoom, self.manual_fixed_offset_x, self.manual_fixed_offset_y
        return self.manual_moving_zoom, self.manual_moving_offset_x, self.manual_moving_offset_y

    def _manual_set_view(self, side: str, zoom: float | None, ox: float, oy: float) -> None:
        if side == "fixed":
            self.manual_fixed_zoom = zoom
            self.manual_fixed_offset_x = ox
            self.manual_fixed_offset_y = oy
        else:
            self.manual_moving_zoom = zoom
            self.manual_moving_offset_x = ox
            self.manual_moving_offset_y = oy

    def _manual_fit_view(self, side: str) -> None:
        if side == "fixed":
            canvas = self.manual_canvas_fixed
            img = self.manual_fixed_img
        else:
            canvas = self.manual_canvas_moving
            img = self.manual_moving_img
        if img is None:
            self._manual_set_view(side, None, 0.0, 0.0)
            return
        cw = max(int(canvas.winfo_width()), 1)
        ch = max(int(canvas.winfo_height()), 1)
        iw, ih = img.size
        zoom = min(cw / iw, ch / ih)
        zoom = max(min(float(zoom), 8.0), 0.05)
        ox = (cw - iw * zoom) * 0.5
        oy = (ch - ih * zoom) * 0.5
        self._manual_set_view(side, zoom, ox, oy)

    def _manual_on_wheel(self, side: str, event) -> None:
        if self._is_ctrl_pressed(event):
            return
        if side == "fixed":
            img = self.manual_fixed_img
        else:
            img = self.manual_moving_img
        if img is None:
            return
        zoom, ox, oy = self._manual_get_view(side)
        if zoom is None:
            self._manual_fit_view(side)
            zoom, ox, oy = self._manual_get_view(side)
        steps = int(event.delta / 120) if event.delta else 0
        if steps == 0:
            return
        factor = 1.15 ** steps
        old_zoom = float(zoom or 1.0)
        new_zoom = max(min(old_zoom * factor, 8.0), 0.05)
        if abs(new_zoom - old_zoom) < 1e-9:
            return
        mx = float(event.x)
        my = float(event.y)
        ix = (mx - ox) / old_zoom
        iy = (my - oy) / old_zoom
        new_ox = mx - ix * new_zoom
        new_oy = my - iy * new_zoom
        self._manual_set_view(side, new_zoom, new_ox, new_oy)
        self._manual_render_one(side)

    def _manual_on_pan_start(self, side: str, event) -> None:
        zoom, ox, oy = self._manual_get_view(side)
        if zoom is None:
            self._manual_fit_view(side)
            zoom, ox, oy = self._manual_get_view(side)
        if side == "fixed":
            self.manual_fixed_pan_start = (int(event.x), int(event.y), float(ox), float(oy))
        else:
            self.manual_moving_pan_start = (int(event.x), int(event.y), float(ox), float(oy))

    def _manual_on_pan_move(self, side: str, event) -> None:
        if side == "fixed":
            st = self.manual_fixed_pan_start
        else:
            st = self.manual_moving_pan_start
        if st is None:
            return
        sx, sy, ox, oy = st
        new_ox = ox + (float(event.x) - sx)
        new_oy = oy + (float(event.y) - sy)
        zoom, _, _ = self._manual_get_view(side)
        self._manual_set_view(side, zoom, new_ox, new_oy)
        self._manual_render_one(side)

    def _manual_render_one(self, side: str) -> None:
        if side == "fixed":
            canvas = self.manual_canvas_fixed
            img = self.manual_fixed_img
            pts = self.manual_points_fixed
            pending = self.manual_pending_fixed
            zoom = self.manual_fixed_zoom
            ox = self.manual_fixed_offset_x
            oy = self.manual_fixed_offset_y
        else:
            canvas = self.manual_canvas_moving
            img = self.manual_moving_img
            pts = self.manual_points_moving
            pending = self.manual_pending_moving
            zoom = self.manual_moving_zoom
            ox = self.manual_moving_offset_x
            oy = self.manual_moving_offset_y

        cw = int(canvas.winfo_width())
        ch = int(canvas.winfo_height())
        canvas.delete("all")
        if img is None or cw < 20 or ch < 20:
            canvas.create_text(cw // 2, ch // 2, text="No image", fill="white", font=self._get_font("Segoe UI", 12, "bold"))
            return

        iw, ih = img.size
        if zoom is None:
            zoom = min(cw / iw, ch / ih)
            zoom = max(min(float(zoom), 8.0), 0.05)
            ox = (cw - iw * zoom) * 0.5
            oy = (ch - ih * zoom) * 0.5
            self._manual_set_view(side, zoom, ox, oy)

        ix0 = (0 - ox) / zoom
        iy0 = (0 - oy) / zoom
        ix1 = (cw - ox) / zoom
        iy1 = (ch - oy) / zoom
        ix0_i = int(max(0, min(iw, ix0)))
        iy0_i = int(max(0, min(ih, iy0)))
        ix1_i = int(max(0, min(iw, ix1)))
        iy1_i = int(max(0, min(ih, iy1)))
        if ix1_i <= ix0_i or iy1_i <= iy0_i:
            return

        crop = img.crop((ix0_i, iy0_i, ix1_i, iy1_i))
        target_w = max(1, int(round((ix1_i - ix0_i) * zoom)))
        target_h = max(1, int(round((iy1_i - iy0_i) * zoom)))
        if crop.size[0] != target_w or crop.size[1] != target_h:
            crop = crop.resize((target_w, target_h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(crop)
        draw_x = ox + ix0_i * zoom
        draw_y = oy + iy0_i * zoom
        if side == "fixed":
            self.manual_fixed_photo = photo
        else:
            self.manual_moving_photo = photo
        canvas.create_image(draw_x, draw_y, anchor="nw", image=photo)

        for i, (x, y) in enumerate(pts, start=1):
            px = x * zoom + ox
            py = y * zoom + oy
            canvas.create_oval(px - 4, py - 4, px + 4, py + 4, outline="#22C55E", width=2)
            canvas.create_text(px + 10, py, text=str(i), fill="#22C55E", font=self._get_font("Segoe UI", 10, "bold"), anchor="w")

        if pending is not None:
            px = pending[0] * zoom + ox
            py = pending[1] * zoom + oy
            canvas.create_oval(px - 5, py - 5, px + 5, py + 5, outline="#F59E0B", width=2)

    def _cycle_compare_layer(self, delta: int) -> None:
        if not self.frame_compare.winfo_viewable():
            return
        layers = ["fixed", "warped"]
        curr = self.compare_layer_var.get()
        idx = layers.index(curr) if curr in layers else 0
        new_idx = (idx + delta) % len(layers)
        self.compare_layer_var.set(layers[new_idx])
        self._render_compare_view()

    def _refresh_matches_view(self) -> None:
        if not self.last_outputs or not self.last_outputs.matches_vis_path:
            self.lbl_matches_img.config(image="", text="No matches generated yet.")
            self.lbl_matches_img.image = None
            return
        self._show_image(self.lbl_matches_img, self.last_outputs.matches_vis_path)

    def _refresh_fusion_view(self) -> None:
        if not self.last_outputs or not self.last_outputs.checkerboard_path:
            self.lbl_checker_img.config(image="", text="No fusion result generated yet.")
            self.lbl_checker_img.image = None
            return
        self._show_image(self.lbl_checker_img, self.last_outputs.checkerboard_path)

    def _refresh_matrix_view(self) -> None:
        self.txt_matrix.delete(1.0, tk.END)
        if not self.last_outputs:
            self.txt_matrix.insert(tk.END, "No transform estimated yet.")
            return
        out = self.last_outputs
        self.txt_matrix.insert(tk.END, "RMSE: {:.4f}\nInliers: {}\n\n".format(out.rmse, out.inliers_count))
        self.txt_matrix.insert(tk.END, "Transform ({} 3x3):\n".format(getattr(out, "transform_model", "affine")))
        for row in out.H_3x3:
            self.txt_matrix.insert(tk.END, "[ {:.6f}, {:.6f}, {:.6f} ]\n".format(*row))

    def _refresh_result_tab(self, name: str) -> None:
        if name == "matches":
            self._refresh_matches_view()
        elif name == "fusion":
            self._refresh_fusion_view()
        elif name == "compare":
            self._render_compare_view()
        elif name == "matrix":
            self._refresh_matrix_view()

    def _switch_tab(self, name):
        self.current_tab = str(name)
        # Hide all
        self.frame_matches.pack_forget()
        self.frame_fusion.pack_forget()
        self.frame_compare.pack_forget()
        self.frame_matrix.pack_forget()
        self.frame_manual.pack_forget()
        
        # Reset button styles (optional, keeping simple flat colors)
        # Show selected
        if name == "matches":
            self.frame_matches.pack(fill=tk.BOTH, expand=True)
            self._refresh_matches_view()
        elif name == "fusion":
            self.frame_fusion.pack(fill=tk.BOTH, expand=True)
            self._refresh_fusion_view()
        elif name == "compare":
            self.frame_compare.pack(fill=tk.BOTH, expand=True)
            self._render_compare_view()
        elif name == "matrix":
            self.frame_matrix.pack(fill=tk.BOTH, expand=True)
            self._refresh_matrix_view()
        elif name == "manual":
            self.frame_manual.pack(fill=tk.BOTH, expand=True)
            self._manual_reload_pair()

    def _load_state(self):
        values = [a.name for a in self.config.algorithms]
        self.algo_combo['values'] = values
        if values:
            self.algo_combo.current(0)
        self._update_algo_hint()

        self.transform_model_var.set(self.config.last_transform_model or "affine")
        if hasattr(self, "ransac_thresh_var"):
            self.ransac_thresh_var.set(f"{float(self.config.ransac_thresh_px):g}")
            self.ransac_max_iters_var.set(str(int(self.config.ransac_max_iters)))
            self.ransac_confidence_var.set(f"{float(self.config.ransac_confidence):.4f}")
            self.ransac_refine_iters_var.set(str(int(self.config.ransac_refine_iters)))
            self._update_ransac_controls_state()
            
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

    def _selected_algorithm_name(self) -> str:
        idx = self.algo_combo.current()
        if idx >= 0 and idx < len(self.config.algorithms):
            return str(self.config.algorithms[idx].name or "")
        return ""

    def _algorithm_prefers_fsc_affine(self, algo_name: str) -> bool:
        name = (algo_name or "").strip().lower()
        return name in ("matlab_wssf", "matlab_wssf_tv_logtv")

    def _apply_algorithm_transform_preferences(self) -> None:
        algo_name = self._selected_algorithm_name()
        if not self._algorithm_prefers_fsc_affine(algo_name):
            return

        self.transform_model_var.set("fsc-affine")

        affine_defaults = AppConfig.opencv_ransac_defaults("affine")
        self.config.ransac_thresh_px = float(affine_defaults["ransac_thresh_px"])
        self.config.ransac_max_iters = int(affine_defaults["ransac_max_iters"])
        self.config.ransac_confidence = float(affine_defaults["ransac_confidence"])
        self.config.ransac_refine_iters = int(affine_defaults["ransac_refine_iters"])

        self.ransac_thresh_var.set(f"{float(self.config.ransac_thresh_px):g}")
        self.ransac_max_iters_var.set(str(int(self.config.ransac_max_iters)))
        self.ransac_confidence_var.set(f"{float(self.config.ransac_confidence):.4f}")
        self.ransac_refine_iters_var.set(str(int(self.config.ransac_refine_iters)))
        self._update_ransac_controls_state()

    def _on_algorithm_changed(self, _event=None) -> None:
        self._update_algo_hint()
        self._apply_algorithm_transform_preferences()
        self._save_config()

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
        self.config.last_transform_model = self.transform_model_var.get()
        save_config(self.config)

    def _sync_ransac_config_from_ui(self) -> None:
        model = (self.transform_model_var.get() or "").strip().lower()
        ransac_defaults = AppConfig.opencv_ransac_defaults(model)
        try:
            s = (self.ransac_thresh_var.get() or "").strip()
            thresh = float(s) if s else float(self.config.ransac_thresh_px)
        except Exception:
            thresh = float(self.config.ransac_thresh_px)
        if not np.isfinite(thresh) or thresh <= 0:
            thresh = float(ransac_defaults["ransac_thresh_px"])

        try:
            s = (self.ransac_max_iters_var.get() or "").strip()
            max_iters = int(s) if s else int(self.config.ransac_max_iters)
        except Exception:
            max_iters = int(self.config.ransac_max_iters)
        if max_iters < 1:
            max_iters = int(ransac_defaults["ransac_max_iters"])

        try:
            s = (self.ransac_confidence_var.get() or "").strip()
            confidence = float(s) if s else float(self.config.ransac_confidence)
        except Exception:
            confidence = float(self.config.ransac_confidence)
        if not np.isfinite(confidence) or not (0.0 < confidence < 1.0):
            confidence = float(ransac_defaults["ransac_confidence"])

        try:
            s = (self.ransac_refine_iters_var.get() or "").strip()
            refine_iters = int(s) if s else int(self.config.ransac_refine_iters)
        except Exception:
            refine_iters = int(self.config.ransac_refine_iters)
        if refine_iters < 0:
            refine_iters = int(ransac_defaults["ransac_refine_iters"])

        self.config.ransac_thresh_px = float(thresh)
        self.config.ransac_max_iters = int(max_iters)
        self.config.ransac_confidence = float(confidence)
        self.config.ransac_refine_iters = int(refine_iters)

        self.ransac_thresh_var.set(f"{float(self.config.ransac_thresh_px):g}")
        self.ransac_max_iters_var.set(str(int(self.config.ransac_max_iters)))
        self.ransac_confidence_var.set(f"{float(self.config.ransac_confidence):.4f}")
        self.ransac_refine_iters_var.set(str(int(self.config.ransac_refine_iters)))
        self._update_ransac_controls_state()

    def _on_ransac_params_commit(self) -> None:
        if self._ransac_syncing:
            return
        self._ransac_syncing = True
        try:
            self._sync_ransac_config_from_ui()
            save_config(self.config)
        finally:
            self._ransac_syncing = False

    def _reset_ransac_defaults(self) -> None:
        model = (self.transform_model_var.get() or "").strip().lower()
        d = AppConfig.opencv_ransac_defaults(model)
        self.ransac_thresh_var.set(f"{float(d['ransac_thresh_px']):g}")
        self.ransac_max_iters_var.set(str(int(d["ransac_max_iters"])))
        self.ransac_confidence_var.set(f"{float(d['ransac_confidence']):.4f}")
        self.ransac_refine_iters_var.set(str(int(d["ransac_refine_iters"])))
        self._on_ransac_params_commit()

    def _update_ransac_controls_state(self) -> None:
        m = (self.transform_model_var.get() or "").strip().lower()
        use_ransac = m in ("affine", "homography")
        state = "normal" if use_ransac else "disabled"
        refine_state = "normal" if m == "affine" else "disabled"
        if hasattr(self, "spin_ransac_thresh"):
            self.spin_ransac_thresh.config(state=state)
            self.spin_ransac_max_iters.config(state=state)
            self.spin_ransac_confidence.config(state=state)
            self.spin_ransac_refine_iters.config(state=refine_state if use_ransac else "disabled")

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
        if hasattr(self, "ransac_thresh_var"):
            self._on_ransac_params_commit()
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
            transform_model=self.transform_model_var.get(),
            fixed_path=fixed,
            moving_path=moving,
            output_dir=out_dir,
            repo_root=str(Path.cwd()),
            ransac_thresh_px=self.config.ransac_thresh_px,
            ransac_max_iters=self.config.ransac_max_iters,
            ransac_confidence=self.config.ransac_confidence,
            ransac_refine_iters=self.config.ransac_refine_iters,
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
            transform_model=self.transform_model_var.get(),
            fixed_path=pair.fixed_path,
            moving_path=pair.moving_path,
            output_dir=out_dir,
            repo_root=str(Path.cwd()),
            ransac_thresh_px=self.config.ransac_thresh_px,
            ransac_max_iters=self.config.ransac_max_iters,
            ransac_confidence=self.config.ransac_confidence,
            ransac_refine_iters=self.config.ransac_refine_iters,
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
        self._update_compare_controls()
        
        self.compare_layer_var.set("fixed")
        self._refresh_matches_view()
        self._refresh_fusion_view()
        self._refresh_matrix_view()
        self.after_idle(lambda: self._refresh_result_tab(self.current_tab))

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
