"""
Configuration and interactive prompt utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import Tk, filedialog, messagebox, simpledialog
except Exception:
    tk = Tk = filedialog = simpledialog = messagebox = None


def load_simple_config(config_path: Path | str) -> Dict[str, Any]:
    config = {}
    config_path = Path(config_path)

    if not config_path.exists():
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)
                config[key] = value
    return config


def save_simple_config(
    config: Dict[str, Any], config_path: Path | str, header: str = "Configuration File"
):
    config_path = Path(config_path)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        f.write("# Automatically generated - edit as needed\n\n")
        for key, value in config.items():
            if isinstance(value, bool):
                value = str(value).lower()
            f.write(f"{key}={value}\n")


def prompt_directory(
    title: str = "Select Directory",
    initial_dir: Optional[str] = None,
    use_terminal: bool = False,
) -> Optional[str]:
    if use_terminal:
        default = initial_dir or os.getcwd()
        result = input(f"{title} [{default}]: ").strip()
        path = result if result else default
        return path if os.path.isdir(path) else None

    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        directory = filedialog.askdirectory(
            title=title, initialdir=initial_dir or os.getcwd()
        )
        root.destroy()
        return directory if directory else None
    except Exception:
        return prompt_directory(title, initial_dir, use_terminal=True)


def prompt_file(
    title: str = "Select File",
    initial_dir: Optional[str] = None,
    filetypes: Optional[list] = None,
) -> Optional[str]:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    filetypes = filetypes or [("All files", "*.*")]
    file_path = filedialog.askopenfilename(
        title=title, initialdir=initial_dir or os.getcwd(), filetypes=filetypes
    )
    root.destroy()
    return file_path if file_path else None


def prompt_save_file(
    title: str = "Save File",
    initial_dir: Optional[str] = None,
    initial_file: Optional[str] = None,
    defaultextension: str = "",
    filetypes: Optional[list] = None,
) -> Optional[str]:
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        filetypes = filetypes or [("All files", "*.*")]
        file_path = filedialog.asksaveasfilename(
            title=title,
            initialdir=initial_dir or os.getcwd(),
            initialfile=initial_file,
            defaultextension=defaultextension,
            filetypes=filetypes,
        )
        root.destroy()
        return file_path if file_path else None
    except Exception:
        # Fallback to simple input in terminal if GUI fails
        print(f"[{title}] Enter save path:")
        path = input("> ").strip()
        return path if path else None


def prompt_options(title: str, prompt: str, options: list) -> Optional[str]:
    """
    Prompt user to select one of several options.
    """
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        # Create a custom dialog window
        dialog = tk.Toplevel(root)
        dialog.title(title)
        dialog.attributes("-topmost", True)
        dialog.lift()

        selected = {"value": None}

        tk.Label(dialog, text=prompt, pady=10).pack()

        def choose(val):
            selected["value"] = val
            dialog.destroy()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=20, pady=10)

        for opt in options:
            tk.Button(btn_frame, text=str(opt), command=lambda v=opt: choose(v)).pack(
                side=tk.LEFT, expand=True, fill=tk.X, padx=5
            )

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        root.wait_window(dialog)
        root.destroy()
        return selected["value"]
    except Exception:
        # Terminal fallback
        print(f"[{title}] {prompt}")
        for i, opt in enumerate(options):
            print(f"  {i + 1}: {opt}")
        res = input("Select number: ").strip()
        if res.isdigit():
            idx = int(res) - 1
            if 0 <= idx < len(options):
                return options[idx]
        return None


def prompt_text(
    title: str, prompt: str, initial_value: str = "", use_terminal: bool = False
) -> Optional[str]:
    if use_terminal:
        if initial_value:
            result = input(f"{prompt} [{initial_value}]: ").strip()
            return result if result else initial_value
        else:
            result = input(f"{prompt}: ").strip()
            return result if result else None

    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        result = simpledialog.askstring(
            title=title, prompt=prompt, initialvalue=initial_value
        )
        root.destroy()
        return result
    except Exception:
        return prompt_text(title, prompt, initial_value, use_terminal=True)


def prompt_yes_no(title: str, message: str) -> bool:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    result = messagebox.askyesno(title=title, message=message)
    root.destroy()
    return result


def get_or_prompt_value(
    arg_value: Any,
    config: Dict[str, Any],
    key: str,
    prompt_func,
    required: bool = True,
    **prompt_kwargs,
) -> Tuple[Any, bool]:
    if arg_value:
        return arg_value, False
    if key in config:
        return config[key], False
    value = prompt_func(**prompt_kwargs)
    if not value and required:
        raise ValueError(f"{key} is required but not provided")
    return value, True


def prompt_channel_grid(
    title: str, channels: list[str], defaults: Optional[list[bool]] = None
) -> Optional[list[str]]:
    """
    Shows a scrollable grid of channels.
    Allows click-drag selection (standard box selection to select multiple).
    """
    if not channels:
        return []

    if tk is None:
        return channels

    if defaults is None:
        defaults = [True] * len(channels)

    # State
    selected_mask = list(defaults)
    result_holder = {"selection": None}

    # UI Constants
    CELL_SIZE = 40
    GAP = 5
    COLS = 8

    root = Tk()
    root.title(title)
    root.attributes("-topmost", True)

    # Center Window
    try:
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        w = COLS * (CELL_SIZE + GAP) + 50  # padding
        h = 600
        x = int(ws / 2 - w / 2)
        y = int(hs / 2 - h / 2)
        root.geometry(f"{w}x{h}+{x}+{y}")
    except:
        pass

    # Header
    lbl = tk.Label(root, text=title, font=("Arial", 12, "bold"))
    lbl.pack(pady=10)

    # Scroll Frame
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=5)

    canvas = tk.Canvas(frame, bg="white")
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Calculate Scroll Region
    rows = (len(channels) + COLS - 1) // COLS
    content_height = rows * (CELL_SIZE + GAP) + GAP
    content_width = COLS * (CELL_SIZE + GAP) + GAP
    canvas.configure(scrollregion=(0, 0, content_width, content_height))

    # Helper: Draw Logic
    # We will use "tags" to identify items: "cell", "cell_i"
    item_ids = []  # map index -> (rect_id, text_id)

    def update_cell_color(idx, is_selected):
        rect_id, txt_id = item_ids[idx]
        color = "#4a90e2" if is_selected else "#e0e0e0"  # Blue vs Light Grey
        text_color = "white" if is_selected else "black"
        canvas.itemconfig(rect_id, fill=color)
        canvas.itemconfig(txt_id, fill=text_color)

    for i in range(len(channels)):
        r = i // COLS
        c = i % COLS
        x0 = GAP + c * (CELL_SIZE + GAP)
        y0 = GAP + r * (CELL_SIZE + GAP)
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE

        # Draw Rect
        rect = canvas.create_rectangle(
            x0, y0, x1, y1, outline="#bcbcbc", width=1, tags=("cell", f"cell_{i}")
        )

        # Draw Text (Channel Index 1-Based)
        txt = canvas.create_text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            text=str(i + 1),
            font=("Arial", 10),
            tags=("cell", f"cell_{i}"),
        )

        item_ids.append((rect, txt))
        update_cell_color(i, selected_mask[i])

    # Selection Logic (Rectangular Marquee)
    drag_data = {"start_x": 0, "start_y": 0, "rect": None, "start_scroll": 0}

    def get_indices_in_rect(x1, y1, x2, y2):
        # Convert coords to row/col ranges
        # Normalize
        sx, ex = min(x1, x2), max(x1, x2)
        sy, ey = min(y1, y2), max(y1, y2)

        # Grid logic
        # x = GAP + c*(CELL+GAP)  => c = (x - GAP) / (CELL+GAP)
        stride = CELL_SIZE + GAP
        c_start = int((sx - GAP) / stride)
        c_end = int((ex - GAP) / stride)
        r_start = int((sy - GAP) / stride)
        r_end = int((ey - GAP) / stride)

        indices = []
        for r in range(max(0, r_start), r_end + 1):
            for c in range(max(0, c_start), c_end + 1):
                idx = r * COLS + c
                if 0 <= idx < len(channels):
                    # Check finer intersection if needed, or just cell center
                    # Simple: if cell center is in rect
                    cx = GAP + c * stride + CELL_SIZE / 2
                    cy = GAP + r * stride + CELL_SIZE / 2
                    if sx <= cx <= ex and sy <= cy <= ey:
                        indices.append(idx)
        return indices

    def on_click(event):
        # Canvas coordinates
        cx = canvas.canvasx(event.x)
        cy = canvas.canvasy(event.y)
        drag_data["start_x"] = cx
        drag_data["start_y"] = cy

        # Create selection rect
        drag_data["rect"] = canvas.create_rectangle(
            cx, cy, cx, cy, outline="red", dash=(4, 4)
        )

    def on_drag(event):
        cx = canvas.canvasx(event.x)
        cy = canvas.canvasy(event.y)
        if drag_data["rect"]:
            canvas.coords(
                drag_data["rect"], drag_data["start_x"], drag_data["start_y"], cx, cy
            )

        # Optional: Auto-scroll if near edge? Skip for simplicity.

    def on_release(event):
        cx = canvas.canvasx(event.x)
        cy = canvas.canvasy(event.y)

        # Remove rect
        if drag_data["rect"]:
            canvas.delete(drag_data["rect"])
            drag_data["rect"] = None

        # Determine affected cells
        indices = get_indices_in_rect(
            drag_data["start_x"], drag_data["start_y"], cx, cy
        )

        # Toggle or Set? Usually "select" if mostly unselected, "deselect" if all selected?
        # Simple behavior: Toggle whatever is inside.
        # Or: Set to True. Box selection usually means "Select". Shift+Box means "Add". Alt+Box means "Subtract".

        # Let's implement logical behavior:
        # If single click (start ~ end), toggle single.
        # If drag, select all in box (Union).
        # But user might want to Deselect.
        # Let's use a mode based on the start cell?

        is_click = (
            abs(cx - drag_data["start_x"]) < 5 and abs(cy - drag_data["start_y"]) < 5
        )

        if is_click:
            # Toggle single
            clicked = get_indices_in_rect(cx - 1, cy - 1, cx + 1, cy + 1)
            for idx in clicked:
                selected_mask[idx] = not selected_mask[idx]
                update_cell_color(idx, selected_mask[idx])
        else:
            # Range Select -> Set to True (Union)
            # If user wants to deselect, they can click individually or maybe we add a mode button.
            # Default to "Add to selection".
            changed = False
            for idx in indices:
                if not selected_mask[idx]:
                    selected_mask[idx] = True
                    update_cell_color(idx, True)

    canvas.bind("<ButtonPress-1>", on_click)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=10)

    def do_ok():
        out = [channels[i] for i, s in enumerate(selected_mask) if s]
        result_holder["selection"] = out
        root.destroy()

    def do_cancel():
        root.destroy()

    def do_all():
        for i in range(len(selected_mask)):
            selected_mask[i] = True
            update_cell_color(i, True)

    def do_none():
        for i in range(len(selected_mask)):
            selected_mask[i] = False
            update_cell_color(i, False)

    tk.Button(btn_frame, text="Select All", command=do_all).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Select None", command=do_none).pack(side="left", padx=5)

    tk.Button(
        btn_frame, text="OK", command=do_ok, width=10, bg="#4a90e2", fg="white"
    ).pack(side="right", padx=5)
    tk.Button(btn_frame, text="Cancel", command=do_cancel, width=10).pack(
        side="right", padx=5
    )

    root.focus_force()
    root.wait_window()

    return result_holder["selection"]


def prompt_checked_list(
    title: str, options: list[str], defaults: Optional[list[bool]] = None
) -> list[str]:
    if not options:
        return []

    if tk is None:
        # Fallback for no GUI - return all or empty?
        # If we can't prompt, we can't filter safely without assumed defaults.
        # Returning all options is safer than returning none.
        return options

    if defaults is None:
        defaults = [True] * len(options)

    # Result container
    result_holder = {"selection": None}

    def on_ok(root, vars, options):
        selected = [opt for opt, var in zip(options, vars) if var.get()]
        result_holder["selection"] = selected
        root.destroy()

    def on_cancel(root):
        root.destroy()

    root = Tk()
    root.title(title)
    root.attributes("-topmost", True)

    # Center the window
    try:
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        w = 400
        h = 600
        x = int((ws / 2) - (w / 2))
        y = int((hs / 2) - (h / 2))
        root.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass

    # Description
    lbl = tk.Label(root, text=title, font=("Arial", 10, "bold"))
    lbl.pack(pady=10)

    # Frame for checklist
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=5)

    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    vars = []
    for i, option in enumerate(options):
        is_checked = defaults[i] if i < len(defaults) else False
        var = tk.BooleanVar(value=is_checked)
        vars.append(var)
        cb = tk.Checkbutton(scrollable_frame, text=option, variable=var, anchor="w")
        cb.pack(fill="x", expand=True)

    # Buttons
    frame_buttons = tk.Frame(root)
    frame_buttons.pack(fill="x", padx=10, pady=10)

    def select_all():
        for var in vars:
            var.set(True)

    def select_none():
        for var in vars:
            var.set(False)

    btn_all = tk.Button(frame_buttons, text="Select All", command=select_all)
    btn_all.pack(side="left", padx=5)

    btn_none = tk.Button(frame_buttons, text="Select None", command=select_none)
    btn_none.pack(side="left", padx=5)

    btn_ok = tk.Button(
        frame_buttons, text="OK", command=lambda: on_ok(root, vars, options), width=10
    )
    btn_ok.pack(side="right", padx=5)

    btn_cancel = tk.Button(
        frame_buttons, text="Cancel", command=lambda: on_cancel(root), width=10
    )
    btn_cancel.pack(side="right", padx=5)

    # Force focus
    root.focus_force()
    root.wait_window()

    # If cancelled (result_holder selection is None), return empty list or None?
    # Usually cancel means "abort" or "no selection".
    # Returning original list might be unexpected if user clicked cancel.
    # Returning empty list is consistent with "nothing selected".
    return result_holder["selection"] if result_holder["selection"] is not None else []


def prompt_channel_grid(
    title: str, channels: list[str], defaults: Optional[list[bool]] = None
) -> Optional[list[str]]:
    """
    Shows a scrollable grid of channels.
    Allows click-drag selection (paint/marquee).
    """
    if not channels:
        return []

    if tk is None:
        return channels

    if defaults is None:
        defaults = [True] * len(channels)

    # State
    # We'll use a local copy of selected states
    selected_mask = list(defaults)
    result_holder = {"selection": None}

    # UI Constants
    CELL_SIZE = 40
    GAP = 5
    COLS = 8

    root = Tk()
    root.title(title)
    root.attributes("-topmost", True)

    # Center Window
    try:
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        w = COLS * (CELL_SIZE + GAP) + 50  # padding + scrollbar
        h = 600
        x = int(ws / 2 - w / 2)
        y = int(hs / 2 - h / 2)
        root.geometry(f"{w}x{h}+{x}+{y}")
    except:
        pass

    # Header
    lbl = tk.Label(root, text=title, font=("Arial", 12, "bold"))
    lbl.pack(pady=10)

    # Scroll Frame
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=5)

    canvas = tk.Canvas(frame, bg="white")
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Calculate Scroll Region
    rows = (len(channels) + COLS - 1) // COLS
    content_height = rows * (CELL_SIZE + GAP) + GAP
    content_width = COLS * (CELL_SIZE + GAP) + GAP
    canvas.configure(scrollregion=(0, 0, content_width, content_height))

    # Helper: Draw Logic
    # We will use "tags" to identify items: "cell", "cell_i"
    item_ids = []  # map index -> (rect_id, text_id)

    def draw_cell(idx, is_selected):
        r = idx // COLS
        c = idx % COLS
        x0 = GAP + c * (CELL_SIZE + GAP)
        y0 = GAP + r * (CELL_SIZE + GAP)
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE

        fill_color = "#4a90e2" if is_selected else "#e0e0e0"  # Blue vs Light Grey
        text_color = "white" if is_selected else "black"

        rect = canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill=fill_color,
            outline="#bcbcbc",
            width=1,
            tags=("cell", f"cell_{idx}"),
        )
        txt = canvas.create_text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            text=str(idx + 1),
            fill=text_color,
            font=("Arial", 10),
            tags=("cell", f"cell_{idx}"),
        )
        return rect, txt

    # Initial Draw
    for i in range(len(channels)):
        rect, txt = draw_cell(i, selected_mask[i])
        item_ids.append((rect, txt))

    def update_visuals(indices):
        for idx in indices:
            if 0 <= idx < len(channels):
                is_sel = selected_mask[idx]
                r_id, t_id = item_ids[idx]

                fill_color = "#4a90e2" if is_sel else "#e0e0e0"
                text_color = "white" if is_sel else "black"

                canvas.itemconfig(r_id, fill=fill_color)
                canvas.itemconfig(t_id, fill=text_color)

    # Interaction State
    drag_state = {
        "start_idx": None,
        "is_selecting": True,  # True=Select, False=Deselect
        "last_dragged": set(),
    }

    def get_index_at(x, y):
        # account for scroll
        y_scroll = canvas.canvasy(y)
        x_scroll = canvas.canvasx(x)

        c = int((x_scroll - GAP) // (CELL_SIZE + GAP))
        r = int((y_scroll - GAP) // (CELL_SIZE + GAP))

        if 0 <= c < COLS and r >= 0:
            idx = r * COLS + c
            if idx < len(channels):
                return idx
        return None

    def on_press(event):
        idx = get_index_at(event.x, event.y)
        if idx is not None:
            drag_state["start_idx"] = idx
            # Mode based on clicked cell: if unselected -> selecting mode, else deselecting
            drag_state["is_selecting"] = not selected_mask[idx]

            # Apply to start cell immediately
            selected_mask[idx] = drag_state["is_selecting"]
            update_visuals([idx])
            drag_state["last_dragged"] = {idx}

    def on_drag(event):
        idx = get_index_at(event.x, event.y)
        if idx is not None and idx not in drag_state["last_dragged"]:
            # Apply mode
            selected_mask[idx] = drag_state["is_selecting"]
            update_visuals([idx])
            drag_state["last_dragged"].add(idx)

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)

    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=10)

    def do_ok():
        out = [channels[i] for i, s in enumerate(selected_mask) if s]
        result_holder["selection"] = out
        root.destroy()

    def do_cancel():
        root.destroy()

    def do_all():
        for i in range(len(selected_mask)):
            selected_mask[i] = True
        update_visuals(range(len(channels)))

    def do_none():
        for i in range(len(selected_mask)):
            selected_mask[i] = False
        update_visuals(range(len(channels)))

    tk.Button(btn_frame, text="Select All", command=do_all).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Select None", command=do_none).pack(side="left", padx=5)

    tk.Button(
        btn_frame,
        text="OK",
        command=do_ok,
        width=15,
        bg="#4a90e2",
        fg="white",
        font=("Arial", 10, "bold"),
    ).pack(side="right", padx=5)
    tk.Button(btn_frame, text="Cancel", command=do_cancel, width=10).pack(
        side="right", padx=5
    )

    root.focus_force()
    root.wait_window()

    return result_holder["selection"]
