"""Lightweight cue presentation GUI with synchronized LSL marker stream.

Purpose
-------
During HDEMG + IMU acquisition we want explicit, machine-readable timestamps
for every gesture cue so downstream analysis can slice EMG/IMU windows by cue
label instead of inferring gesture identity from the angle stream.  This module
provides a single-file Tkinter GUI that:

  1. Reads a protocol schedule from YAML (or uses a default block design).
  2. Presents large on-screen text + a countdown for each phase
     (``READY`` -> ``GO`` -> ``HOLD`` -> ``REST``).
  3. Pushes string markers on an LSL outlet at every phase transition with a
     consistent, parseable schema:

         "cue_start:<gesture>:rep_<n>"
         "cue_end:<gesture>:rep_<n>"
         "rest_start"
         "rest_end"
         "block_start:<block_name>"
         "block_end:<block_name>"
         "calibration_hold:<posture>:start"
         "calibration_hold:<posture>:end"
         "session_start"
         "session_end"

LSL outlet
----------
- Stream name:      ``Cues`` (override with ``--lsl_name``)
- Stream type:      ``Markers``
- Channel count:    1, channel_format ``string``
- nominal_srate:    0 (irregular)
- source_id:        ``semg-cues-<host>-<starttime>``

The marker timestamp uses ``pylsl.local_clock()`` at the moment of the
transition, so it shares a clock with all other LSL streams in the session.

Block design (defaults)
-----------------------
- READY phase:  5 s (operator-/participant-prep padding before each gesture).
- GO/HOLD:      configurable per gesture (default 5 s sustained contraction).
- REST:         3 s between gestures (configurable; default schedule uses 3 s).
- Calibration block at session start: 5 s each of {neutral, pronated,
  supinated} wrist postures with explicit markers.

Run
---
    python cue_player.py --fullscreen
    python cue_player.py --schedule gesture_protocol.json --fullscreen
    python cue_player.py --schedule path/to/protocol.json

If ``--schedule`` is omitted, the script looks for ``gesture_protocol.json``
next to the script before falling back to the hard-coded default block design.

Config file
-----------
JSON is the primary format (no extra deps).  YAML is also accepted if pyyaml
is installed.  The schema is the same for both -- either provide top-level
knobs (``gestures``, ``reps``, ``ready_s``, ``hold_s``, ``rest_s``, ...) or an
explicit ``events`` list of ``{phase, label, duration_s, rep}``.

Dependencies: pylsl (required), pyyaml (optional, YAML schedules only).
Tkinter ships with the standard CPython Windows installer.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Optional Pillow for gesture reference image loading.
try:
    from PIL import Image as _PILImage  # type: ignore
    from PIL import ImageTk as _PILImageTk

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
except ImportError as exc:  # pragma: no cover - hard dependency at runtime
    raise SystemExit("pylsl is required.  Install with:  pip install pylsl") from exc

import tkinter as tk
from tkinter import font as tkfont

# ---------------------------------------------------------------------------
# Protocol data model
# ---------------------------------------------------------------------------


@dataclass
class CueEvent:
    """A single timed phase the participant must complete.

    Attributes
    ----------
    phase : str
        One of ``"ready"``, ``"go"``, ``"rest"``, ``"calibration"``,
        ``"block_start"``, ``"block_end"``, ``"session_start"``,
        ``"session_end"``.
    label : str
        Gesture name (for ``go``/``ready``), posture name (for
        ``calibration``), or block name (for ``block_*``).  Empty for
        ``rest``/session events.
    duration_s : float
        Phase duration in seconds.  Zero for instantaneous markers.
    rep : int
        1-based repetition index inside the current block; ignored for
        non-gesture phases.
    color : str
        Tk color name for the background during this phase.  Defaults are
        chosen for high-contrast on-screen distinction.
    """

    phase: str
    label: str = ""
    duration_s: float = 0.0
    rep: int = 0
    color: str = "#202020"


# Default per-phase background colors -- muted, low-saturation palette so the
# screen is easy on the eyes during long sessions.
PHASE_COLORS = {
    "ready": "#3a4358",  # dusty slate-blue
    "go": "#3d5a47",  # muted sage green
    "rest": "#262626",  # warm dark gray
    "calibration": "#5a3f3f",  # dusty rose
    "block_start": "#1f1f1f",
    "block_end": "#1f1f1f",
    "session_start": "#181818",
    "session_end": "#181818",
}


def _color_for(phase: str) -> str:
    return PHASE_COLORS.get(phase, "#202020")


# ---------------------------------------------------------------------------
# Default schedule
# ---------------------------------------------------------------------------


DEFAULT_GESTURES: Sequence[str] = (
    "index_flex",
    "middle_flex",
    "ring_flex",
    "pinky_flex",
    "thumb_flex",
    "fist",
    "extend_all",
    "pinch_index_thumb",
)
DEFAULT_REPS = 5
DEFAULT_READY_S = 5.0
DEFAULT_HOLD_S = 5.0
DEFAULT_REST_S = 3.0
DEFAULT_CALIB_POSTURES: Sequence[str] = ("neutral", "pronated", "supinated")
DEFAULT_CALIB_HOLD_S = 5.0
DEFAULT_GESTURE_ORIENTATIONS: Sequence[str] = ("neutral",)
DEFAULT_POSTURE_SETUP_S = 8.0


def build_default_schedule(
    gestures: Sequence[str] = DEFAULT_GESTURES,
    reps: int = DEFAULT_REPS,
    ready_s: float = DEFAULT_READY_S,
    hold_s: float = DEFAULT_HOLD_S,
    rest_s: float = DEFAULT_REST_S,
    calibration_postures: Sequence[str] = DEFAULT_CALIB_POSTURES,
    calibration_hold_s: float = DEFAULT_CALIB_HOLD_S,
    gesture_orientations: Sequence[str] = DEFAULT_GESTURE_ORIENTATIONS,
    posture_setup_s: float = DEFAULT_POSTURE_SETUP_S,
    intro_pad_s: float = 5.0,
    outro_pad_s: float = 5.0,
    randomize_gestures: bool = False,
    seed: int | None = None,
) -> list[CueEvent]:
    """Build a typical block-design schedule.

    Layout
    ------
    1. ``session_start`` (instantaneous marker)
    2. Calibration block: for each posture -> ``ready`` (``ready_s``) ->
       ``calibration`` (``calibration_hold_s``) -> ``rest`` (``rest_s``)
    3. Gesture block: for each (gesture, rep) -> ``ready`` (``ready_s``) ->
       ``go`` (``hold_s``) -> ``rest`` (``rest_s``)
    4. ``session_end`` (instantaneous marker)

    Returns
    -------
    list[CueEvent]
    """
    rng = random.Random(seed) if randomize_gestures else None
    events: list[CueEvent] = []
    events.append(CueEvent("session_start", "", 0.0, 0, _color_for("session_start")))
    if intro_pad_s > 0:
        events.append(CueEvent("rest", "", float(intro_pad_s), 0, _color_for("rest")))

    # Calibration block
    if calibration_postures:
        events.append(CueEvent("block_start", "calibration", 0.0, 0))
        for posture in calibration_postures:
            events.append(
                CueEvent(
                    "ready", f"calib_{posture}", float(ready_s), 0, _color_for("ready")
                )
            )
            events.append(
                CueEvent(
                    "calibration",
                    posture,
                    float(calibration_hold_s),
                    0,
                    _color_for("calibration"),
                )
            )
            events.append(CueEvent("rest", "", float(rest_s), 0, _color_for("rest")))
        events.append(CueEvent("block_end", "calibration", 0.0, 0))

    # Gesture block(s) -- one per wrist orientation.
    orientations = tuple(gesture_orientations) if gesture_orientations else ("neutral",)
    multi_orient = len(orientations) > 1
    for orientation in orientations:
        block_name = f"gestures_{orientation}" if multi_orient else "gestures"
        events.append(CueEvent("block_start", block_name, 0.0, 0))
        # Posture-setup cue: long READY telling the participant which wrist
        # pose to assume before this block starts.  Only emit when there are
        # multiple orientations (otherwise it's redundant with the per-gesture
        # ready cues).
        if multi_orient:
            events.append(
                CueEvent(
                    "ready",
                    f"posture_{orientation}",
                    float(posture_setup_s),
                    0,
                    _color_for("ready"),
                )
            )
        gesture_order = list(gestures)
        if rng is not None:
            rng.shuffle(gesture_order)
        for gesture in gesture_order:
            label = f"{gesture}@{orientation}" if multi_orient else gesture
            for rep in range(1, int(reps) + 1):
                events.append(
                    CueEvent("ready", label, float(ready_s), rep, _color_for("ready"))
                )
                events.append(
                    CueEvent("go", label, float(hold_s), rep, _color_for("go"))
                )
                events.append(
                    CueEvent("rest", "", float(rest_s), rep, _color_for("rest"))
                )
        events.append(CueEvent("block_end", block_name, 0.0, 0))

    if outro_pad_s > 0:
        events.append(CueEvent("rest", "", float(outro_pad_s), 0, _color_for("rest")))
    events.append(CueEvent("session_end", "", 0.0, 0, _color_for("session_end")))
    return events


def _shuffle_gesture_blocks(
    events: list[CueEvent], rng: random.Random
) -> list[CueEvent]:
    """Shuffle the gesture order within every orientation block.

    Calibration events, posture-setup cues, block markers, and rest events
    are left in place.  Only the *gesture groups* (all reps of one gesture
    label, i.e. N×(ready+go+rest)) are shuffled within each block.
    """
    result: list[CueEvent] = []
    i = 0
    while i < len(events):
        ev = events[i]
        if ev.phase == "block_start" and ev.label.startswith("gestures"):
            result.append(ev)
            i += 1
            # Pass through posture-setup ready cue (label starts with "posture_").
            while (
                i < len(events)
                and events[i].phase == "ready"
                and events[i].label.startswith("posture_")
            ):
                result.append(events[i])
                i += 1
            # Collect gesture groups up to block_end.
            # A new group begins at each "ready" event with a gesture label.
            gesture_groups: list[list[CueEvent]] = []
            while i < len(events) and events[i].phase != "block_end":
                group: list[CueEvent] = [events[i]]
                i += 1
                while (
                    i < len(events)
                    and events[i].phase != "block_end"
                    and not (
                        events[i].phase == "ready"
                        and events[i].label
                        and not events[i].label.startswith("posture_")
                    )
                ):
                    group.append(events[i])
                    i += 1
                gesture_groups.append(group)
            rng.shuffle(gesture_groups)
            for group in gesture_groups:
                result.extend(group)
            if i < len(events):  # append block_end
                result.append(events[i])
                i += 1
        else:
            result.append(ev)
            i += 1
    return result


def _schedule_from_dict(cfg: dict) -> list[CueEvent]:
    """Convert a config dict into a list of ``CueEvent``.

    Two schemas are accepted (same shape for JSON and YAML):

    1. **Knob style** -- override any of the ``build_default_schedule``
       parameters::

            {
              "gestures": ["index_flex", "middle_flex", ...],
              "reps": 5,
              "ready_s": 5.0,
              "hold_s": 5.0,
              "rest_s": 3.0,
              "calibration_postures": ["neutral", "pronated", "supinated"],
              "calibration_hold_s": 5.0,
              "intro_pad_s": 5.0,
              "outro_pad_s": 5.0
            }

    2. **Explicit event list** -- full control::

            { "events": [ {"phase": "go", "label": "index_flex",
                           "duration_s": 5.0, "rep": 1}, ... ] }
    """
    if "events" in cfg:
        out: list[CueEvent] = []
        for item in cfg["events"]:
            phase = str(item["phase"])
            out.append(
                CueEvent(
                    phase=phase,
                    label=str(item.get("label", "")),
                    duration_s=float(item.get("duration_s", 0.0)),
                    rep=int(item.get("rep", 0)),
                    color=str(item.get("color", _color_for(phase))),
                )
            )
        return out

    calib = cfg.get("calibration_postures", DEFAULT_CALIB_POSTURES)
    if calib is None or cfg.get("no_calibration", False):
        calib = ()
    orientations = cfg.get("gesture_orientations", DEFAULT_GESTURE_ORIENTATIONS)
    if not orientations:
        orientations = ("neutral",)
    return build_default_schedule(
        gestures=cfg.get("gestures", DEFAULT_GESTURES),
        reps=int(cfg.get("reps", DEFAULT_REPS)),
        ready_s=float(cfg.get("ready_s", DEFAULT_READY_S)),
        hold_s=float(cfg.get("hold_s", DEFAULT_HOLD_S)),
        rest_s=float(cfg.get("rest_s", DEFAULT_REST_S)),
        calibration_postures=tuple(calib) if calib else (),
        calibration_hold_s=float(cfg.get("calibration_hold_s", DEFAULT_CALIB_HOLD_S)),
        gesture_orientations=tuple(orientations),
        posture_setup_s=float(cfg.get("posture_setup_s", DEFAULT_POSTURE_SETUP_S)),
        intro_pad_s=float(cfg.get("intro_pad_s", 5.0)),
        outro_pad_s=float(cfg.get("outro_pad_s", 5.0)),
        randomize_gestures=bool(cfg.get("randomize_gestures", False)),
        seed=int(cfg["seed"]) if "seed" in cfg else None,
    )


def load_schedule_json(path: Path) -> list[CueEvent]:
    """Load a schedule from a JSON config file."""
    with path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: top-level JSON must be an object.")
    return _schedule_from_dict(cfg)


def load_schedule_yaml(path: Path) -> list[CueEvent]:
    """Load a schedule from YAML (requires pyyaml)."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pyyaml is required for YAML schedules.  Install with:  pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: top-level YAML must be a mapping.")
    return _schedule_from_dict(cfg)


def load_schedule(path: Path) -> list[CueEvent]:
    """Dispatch by file extension.  ``.json`` -> JSON, ``.yaml``/``.yml`` -> YAML."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_schedule_json(path)
    if suffix in (".yaml", ".yml"):
        return load_schedule_yaml(path)
    raise SystemExit(
        f"{path}: unsupported schedule format '{suffix}'.  Use .json, .yaml, or .yml."
    )


# ---------------------------------------------------------------------------
# LSL marker outlet
# ---------------------------------------------------------------------------


def make_marker_outlet(name: str = "Cues") -> StreamOutlet:
    """Create the LSL ``Markers`` outlet used for cue events.

    The stream is irregular (``nominal_srate=0``) and string-typed so that
    pyxdf will load each event as a single string per sample, which is what
    `sleeve_preprocessing_imu.py` expects for label injection.
    """
    source_id = f"semg-cues-{socket.gethostname()}-{int(time.time())}"
    info = StreamInfo(
        name=name,
        type="Markers",
        channel_count=1,
        nominal_srate=0.0,
        channel_format="string",
        source_id=source_id,
    )
    try:
        desc = info.desc()
        desc.append_child_value("manufacturer", "sEMG-sleeve")
        desc.append_child_value("schema_version", "1")
        channels = desc.append_child("channels")
        ch = channels.append_child("channel")
        ch.append_child_value("label", "cue")
        ch.append_child_value("type", "Marker")
    except Exception:
        pass
    return StreamOutlet(info)


def format_marker(event: CueEvent, edge: str) -> str | None:
    """Translate a (event, edge) pair to a marker string.

    ``edge`` is ``"start"`` or ``"end"``.  Returns ``None`` if this
    (phase, edge) combination should not emit a marker (e.g. ``ready`` is
    operator-side preparation only).
    """
    phase = event.phase
    label = event.label
    rep = event.rep

    if phase == "go":
        suffix = f"{label}:rep_{rep}" if rep else label
        return f"cue_{edge}:{suffix}"
    if phase == "rest":
        return f"rest_{edge}"
    if phase == "calibration":
        return f"calibration_hold:{label}:{edge}"
    if phase == "block_start" and edge == "start":
        return f"block_start:{label}"
    if phase == "block_end" and edge == "start":
        return f"block_end:{label}"
    if phase == "session_start" and edge == "start":
        return "session_start"
    if phase == "session_end" and edge == "start":
        return "session_end"
    if phase == "ready":
        # Optional: emit a "ready" cue too so analysis can find prep windows.
        suffix = f"{label}:rep_{rep}" if rep else label
        return f"ready_{edge}:{suffix}"
    return None


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class GestureImageWindow:
    """Secondary Toplevel window showing a gesture reference image (or text).

    Images are loaded from *images_dir* using this lookup order:
      1. ``<gesture>@<orientation>.<ext>``  (e.g. ``index_flex@pronated.png``)
      2. ``<gesture>_<orientation>.<ext>``
      3. ``<gesture>.<ext>``                (generic, any orientation)

    Supported extensions: .png  .jpg  .jpeg  .webp  .gif
    PNG is supported without any extra package; other formats require Pillow.

    If no image file is found the window shows the gesture name as text.
    """

    _EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

    def __init__(self, root: tk.Tk, images_dir: Path) -> None:
        self._images_dir = images_dir
        self._photo: object | None = None  # strong ref prevents GC

        self.top = tk.Toplevel(root)
        self.top.title("Gesture Reference")
        self.top.configure(bg="#1a1a1a")
        self.top.protocol("WM_DELETE_WINDOW", lambda: None)  # prevent manual close

        # Position to the right of the main window once it's mapped.
        root.update_idletasks()
        rx = root.winfo_x() + root.winfo_width() + 12
        ry = root.winfo_y()
        self.top.geometry(f"420x480+{rx}+{ry}")

        self._img_label = tk.Label(self.top, bg="#1a1a1a", anchor="center")
        self._img_label.pack(expand=True, fill="both", padx=10, pady=(14, 4))

        name_font = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self._name_var = tk.StringVar(value="")
        tk.Label(
            self.top,
            textvariable=self._name_var,
            fg="#ffffff",
            bg="#1a1a1a",
            font=name_font,
            wraplength=400,
            justify="center",
        ).pack(pady=(0, 2))

        orient_font = tkfont.Font(family="Segoe UI", size=13)
        self._orient_var = tk.StringVar(value="")
        self._orient_lbl = tk.Label(
            self.top,
            textvariable=self._orient_var,
            fg="#ffffff",
            bg="#1a1a1a",
            font=orient_font,
        )
        self._orient_lbl.pack(pady=(0, 10))

        # Keep references to labels that need bg updates.
        self._bg_widgets: list[tk.Widget] = [self._img_label, self._orient_lbl]
        for w in self.top.pack_slaves():
            if w not in self._bg_widgets:
                self._bg_widgets.append(w)

    def update(self, label: str, phase: str, bg: str) -> None:
        """Refresh the window for the current cue label + phase."""
        # Update background on all child widgets.
        self.top.configure(bg=bg)
        for w in self.top.winfo_children():
            try:
                w.configure(bg=bg)
            except Exception:
                pass

        if not label or phase in (
            "rest",
            "session_start",
            "session_end",
            "block_start",
            "block_end",
        ):
            self._name_var.set("")
            self._orient_var.set("")
            self._img_label.configure(image="", text="")
            self._photo = None
            return

        # Decode "gesture@orientation" or bare "gesture".
        if "@" in label:
            gesture, orientation = label.split("@", 1)
        elif label.startswith("posture_"):
            gesture = ""
            orientation = label[len("posture_") :]
        else:
            gesture, orientation = label, ""

        if gesture:
            self._name_var.set(gesture.replace("_", " ").title())
        else:
            self._name_var.set("")
        self._orient_var.set(
            f"wrist: {orientation.replace('_', ' ')}" if orientation else ""
        )

        img_path = self._find_image(gesture, orientation) if gesture else None
        if img_path:
            self._load_image(img_path)
        else:
            self._photo = None
            self._img_label.configure(image="", text="")

    def _find_image(self, gesture: str, orientation: str) -> Path | None:
        for ext in self._EXTS:
            if orientation:
                for sep in ("@", "_"):
                    p = self._images_dir / f"{gesture}{sep}{orientation}{ext}"
                    if p.is_file():
                        return p
            p = self._images_dir / f"{gesture}{ext}"
            if p.is_file():
                return p
        return None

    def _load_image(self, path: Path) -> None:
        try:
            # Use the actual rendered size of the image label widget.
            self.top.update_idletasks()
            avail_w = max(200, self._img_label.winfo_width() - 20)
            avail_h = max(200, self._img_label.winfo_height() - 20)
            if _HAS_PIL:
                img = _PILImage.open(path).convert("RGBA")
                img.thumbnail((avail_w, avail_h), _PILImage.LANCZOS)
                self._photo = _PILImageTk.PhotoImage(img)
            else:
                # Built-in PhotoImage: PNG/GIF only, integer downscaling.
                self._photo = tk.PhotoImage(file=str(path))
                iw = self._photo.width()
                ih = self._photo.height()
                factor = max(1, max(iw // avail_w, ih // avail_h))
                if factor > 1:
                    self._photo = self._photo.subsample(factor, factor)
            self._img_label.configure(image=self._photo, text="")
        except Exception as exc:
            print(
                f"[image_window] Failed to load {path.name}: {exc}",
                file=sys.stderr,
            )
            self._photo = None
            self._img_label.configure(image="", text="")

    def destroy(self) -> None:
        try:
            self.top.destroy()
        except Exception:
            pass


class CuePlayerApp:
    """Tkinter cue presenter driven by an event list.

    Timing model
    ------------
    The GUI uses ``tk.after`` for scheduling, but every marker push captures
    ``pylsl.local_clock()`` at the moment of the transition rather than the
    scheduled time, so jitter in the Tk event loop does not accumulate into
    the marker stream.  Phase durations are honored using ``after`` callbacks
    so the GUI remains responsive (Space to pause, Esc to abort).
    """

    def __init__(
        self,
        events: Sequence[CueEvent],
        outlet: StreamOutlet | None,
        title: str = "sEMG Cue Player",
        fullscreen: bool = False,
        image_window: GestureImageWindow | None = None,
    ) -> None:
        self._original_events = list(events)
        self.events = list(events)
        self._image_window = image_window
        self.outlet = outlet
        self.idx = 0
        self.paused = False
        self.started = False
        self._after_id: str | None = None
        self._phase_started_at: float = 0.0
        self._phase_remaining: float = 0.0
        self._display_duration: float = 0.0
        self._total_duration_s = float(sum(e.duration_s for e in self.events))
        self._session_started_at: float = 0.0

        self.root = tk.Tk()
        self.root.title(title)
        if fullscreen:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry("1000x460")
            self.root.minsize(800, 380)
        self.root.configure(bg="#1a1a1a")
        self.root.bind("<Escape>", lambda _e: self._abort())
        self.root.bind("<space>", lambda _e: self._on_space())

        big = tkfont.Font(family="Segoe UI", size=42, weight="normal")
        label_font = tkfont.Font(family="Segoe UI", size=38, weight="bold")
        med = tkfont.Font(family="Segoe UI", size=20)
        small = tkfont.Font(family="Segoe UI", size=13)

        self.phase_var = tk.StringVar(value="READY TO START")
        self.label_var = tk.StringVar(value="Press SPACE to begin")
        self.timer_var = tk.StringVar(value="")
        self.next_var = tk.StringVar(value="")
        self.session_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(
            value="SPACE = start    (then SPACE = pause/resume,  ESC = abort)"
        )

        self.phase_lbl = tk.Label(
            self.root,
            textvariable=self.phase_var,
            fg="#ffffff",
            bg="#1a1a1a",
            font=big,
        )
        self.label_lbl = tk.Label(
            self.root,
            textvariable=self.label_var,
            fg="#ffffff",
            bg="#1a1a1a",
            font=label_font,
        )
        self.timer_lbl = tk.Label(
            self.root,
            textvariable=self.timer_var,
            fg="#ffffff",
            bg="#1a1a1a",
            font=med,
        )
        # Phase progress bar (depletes left -> right as the phase elapses).
        self.bar = tk.Canvas(
            self.root,
            height=18,
            bg="#141414",
            highlightthickness=1,
            highlightbackground="#3a3a3a",
        )
        self._bar_rect = self.bar.create_rectangle(0, 0, 0, 18, fill="#b8b3a8", width=0)
        self.next_lbl = tk.Label(
            self.root,
            textvariable=self.next_var,
            fg="#ffffff",
            bg="#1a1a1a",
            font=small,
        )
        self.session_lbl = tk.Label(
            self.root,
            textvariable=self.session_var,
            fg="#ffffff",
            bg="#1a1a1a",
            font=small,
        )
        self.status_lbl = tk.Label(
            self.root,
            textvariable=self.status_var,
            fg="#ffffff",
            bg="#1a1a1a",
            font=small,
        )

        # Randomize gesture order toggle (visible on the pre-start screen only).
        chk_font = tkfont.Font(family="Segoe UI", size=11)
        self._randomize_var = tk.BooleanVar(value=True)
        self._randomize_chk = tk.Checkbutton(
            self.root,
            text="Randomize gesture order",
            variable=self._randomize_var,
            fg="#ffffff",
            bg="#1a1a1a",
            selectcolor="#2a2a2a",
            activebackground="#1a1a1a",
            activeforeground="#ffffff",
            font=chk_font,
            bd=0,
            highlightthickness=0,
        )

        self.phase_lbl.pack(pady=(12, 2))
        self.label_lbl.pack(pady=(0, 6))
        self.timer_lbl.pack(pady=(0, 2))
        self.bar.pack(fill="x", padx=60, pady=(0, 6))
        self.next_lbl.pack(pady=(2, 0))
        self.session_lbl.pack(pady=(0, 0))
        self._randomize_chk.pack(pady=(4, 0))
        self.status_lbl.pack(side="bottom", pady=4)

        # Wait for SPACE before pushing the first marker so the operator can
        # arm LabRecorder while the participant reads the title screen.
        if outlet is not None:
            print(
                "[cue_player] Outlet is live -- arm your recorder, then press SPACE in the GUI."
            )

    # -- lifecycle ------------------------------------------------------

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            if self.outlet is not None:
                # Best-effort: pylsl outlets close when garbage-collected.
                try:
                    del self.outlet
                except Exception:
                    pass

    def _abort(self) -> None:
        if self.started:
            self._push("session_aborted")
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
        self.root.destroy()

    def _on_space(self) -> None:
        if not self.started:
            self.started = True
            # Apply gesture-order randomization if the toggle is on.
            if self._randomize_var.get():
                self.events = _shuffle_gesture_blocks(
                    self._original_events, random.Random()
                )
                self._total_duration_s = float(sum(e.duration_s for e in self.events))
                print("[cue_player] Gesture order randomized for this run.")
            # Hide the toggle — it's irrelevant once running.
            self._randomize_chk.pack_forget()
            self._session_started_at = local_clock()
            self.status_var.set("SPACE = pause/resume    ESC = abort")
            self._next_phase()
            return
        self._toggle_pause()

    def _toggle_pause(self) -> None:
        if self.paused:
            self.paused = False
            self._push("resume")
            self.status_var.set("Space = pause/resume    Esc = abort")
            # Reset the tick baseline to the remaining duration so the bar
            # and countdown continue from where they paused, not from full.
            self._display_duration = self._phase_remaining
            self._phase_started_at = local_clock()
            self._after_id = self.root.after(
                int(self._phase_remaining * 1000), self._end_current_phase
            )
            self._tick()  # restart the visual update loop
        else:
            self.paused = True
            elapsed = local_clock() - self._phase_started_at
            self._phase_remaining = max(0.0, self._phase_remaining - elapsed)
            if self._after_id is not None:
                try:
                    self.root.after_cancel(self._after_id)
                except Exception:
                    pass
                self._after_id = None
            self._push("pause")
            self.status_var.set("PAUSED  -- Space = resume    Esc = abort")

    # -- phase machine --------------------------------------------------

    def _next_phase(self) -> None:
        if self.idx >= len(self.events):
            self.phase_var.set("DONE")
            self.label_var.set("")
            self.timer_var.set("")
            self.next_var.set("")
            self.root.configure(bg="#1a1a1a")
            if self._image_window is not None:
                self._image_window.update("", "rest", "#1a1a1a")
            self.root.after(1500, self.root.destroy)
            return

        event = self.events[self.idx]
        self.root.configure(bg=event.color)
        for widget in (
            self.phase_lbl,
            self.label_lbl,
            self.timer_lbl,
            self.next_lbl,
            self.session_lbl,
            self.status_lbl,
        ):
            widget.configure(bg=event.color)
        self.bar.itemconfigure(self._bar_rect, fill=self._bar_color(event.phase))

        self.phase_var.set(event.phase.upper())
        if event.label and event.rep:
            self.label_var.set(f"{event.label}   #{event.rep}")
        else:
            self.label_var.set(event.label)

        nxt = self._format_next()
        self.next_var.set(f"next: {nxt}" if nxt else "")
        self._update_session_progress()

        # Update the gesture reference window (if open).
        if self._image_window is not None:
            self._image_window.update(event.label, event.phase, event.color)

        marker = format_marker(event, "start")
        if marker is not None:
            self._push(marker)

        if event.duration_s <= 0:
            # Instantaneous marker -- emit end immediately and advance.
            end_marker = format_marker(event, "end")
            if end_marker is not None:
                self._push(end_marker)
            self.idx += 1
            self.root.after(1, self._next_phase)
            return

        self._phase_started_at = local_clock()
        self._phase_remaining = float(event.duration_s)
        self._display_duration = float(event.duration_s)
        self._tick()
        self._after_id = self.root.after(
            int(event.duration_s * 1000), self._end_current_phase
        )

    def _end_current_phase(self) -> None:
        self._after_id = None
        if self.paused:
            return
        event = self.events[self.idx]
        end_marker = format_marker(event, "end")
        if end_marker is not None:
            self._push(end_marker)
        self.idx += 1
        self._next_phase()

    def _tick(self) -> None:
        if self.paused:
            return
        if self.idx >= len(self.events):
            return
        event = self.events[self.idx]
        elapsed = local_clock() - self._phase_started_at
        dur = max(1e-3, self._display_duration)
        remaining = max(0.0, dur - elapsed)
        self.timer_var.set(f"{remaining:0.1f} s")
        frac = max(0.0, min(1.0, remaining / dur))
        try:
            width = max(1, int(self.bar.winfo_width()))
            self.bar.coords(self._bar_rect, 0, 0, int(width * frac), 28)
        except Exception:
            pass
        self._update_session_progress()
        if remaining > 0.05:
            self.root.after(50, self._tick)
        else:
            self.timer_var.set("0.0 s")
            try:
                self.bar.coords(self._bar_rect, 0, 0, 0, 28)
            except Exception:
                pass

    def _bar_color(self, phase: str) -> str:
        return {
            "go": "#9ec9a6",
            "ready": "#a8b8d4",
            "rest": "#7a7468",
            "calibration": "#d4a8a8",
        }.get(phase, "#b8b3a8")

    def _update_session_progress(self) -> None:
        if not self.started or self._total_duration_s <= 0:
            self.session_var.set("")
            return
        elapsed = local_clock() - self._session_started_at
        remaining_s = max(0.0, self._total_duration_s - elapsed)

        # Count remaining GO-phase gestures in the current orientation block.
        # Find the enclosing block_start (gestures_*) at or before self.idx.
        block_label = ""
        for k in range(self.idx, -1, -1):
            if self.events[k].phase == "block_start" and self.events[
                k
            ].label.startswith("gestures"):
                block_label = self.events[k].label
                break

        block_info = ""
        if block_label:
            # Orientation name from block label ("gestures_pronated" -> "pronated").
            orient = block_label.split("_", 1)[1] if "_" in block_label else ""
            # Count remaining GO events (individual reps) in this block.
            remaining_reps = 0
            for k in range(self.idx, len(self.events)):
                ev = self.events[k]
                if ev.phase == "block_end" and ev.label == block_label:
                    break
                if ev.phase == "go":
                    remaining_reps += 1
            orient_str = f" ({orient})" if orient else ""
            block_info = f"  |  {remaining_reps} rep{'s' if remaining_reps != 1 else ''} left in block{orient_str}"

        self.session_var.set(
            f"session  {elapsed:0.0f}s elapsed / {remaining_s:0.0f}s left{block_info}"
        )

    def _format_next(self) -> str:
        for j in range(self.idx + 1, len(self.events)):
            ev = self.events[j]
            if ev.phase in ("go", "calibration"):
                tag = ev.label
                if ev.rep:
                    tag = f"{tag} #{ev.rep}"
                return f"{ev.phase} -> {tag}"
        return ""

    # -- LSL push -------------------------------------------------------

    def _push(self, marker: str) -> None:
        ts = local_clock()
        if self.outlet is not None:
            try:
                self.outlet.push_sample([marker], timestamp=ts)
            except Exception as exc:  # pragma: no cover
                print(
                    f"[cue_player] LSL push failed for {marker!r}: {exc}",
                    file=sys.stderr,
                )
        # Also echo to stdout for log capture.
        print(f"{ts:.6f}\t{marker}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Cue presentation GUI that pushes synchronized LSL markers "
            "for sEMG/IMU acquisition."
        ),
    )
    p.add_argument(
        "--schedule",
        default=None,
        help=(
            "Path to a .json (or .yaml) schedule, or 'default' "
            "for the built-in block design.  If omitted, looks "
            "for 'gesture_protocol.json' next to this script."
        ),
    )
    p.add_argument(
        "--lsl_name", default="Cues", help="LSL stream name for the marker outlet."
    )
    p.add_argument(
        "--no_lsl",
        action="store_true",
        help="Run without creating an LSL outlet (dry-run/testing).",
    )
    p.add_argument(
        "--fullscreen", action="store_true", help="Display the cue window fullscreen."
    )
    p.add_argument(
        "--print_schedule",
        action="store_true",
        help="Print the resolved schedule and exit (no GUI).",
    )
    # Knobs for the default schedule (ignored for YAML schedules).
    p.add_argument(
        "--gestures",
        nargs="+",
        default=None,
        help="Override gesture list for the default schedule.",
    )
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--ready_s", type=float, default=DEFAULT_READY_S)
    p.add_argument("--hold_s", type=float, default=DEFAULT_HOLD_S)
    p.add_argument("--rest_s", type=float, default=DEFAULT_REST_S)
    p.add_argument("--calib_hold_s", type=float, default=DEFAULT_CALIB_HOLD_S)
    p.add_argument(
        "--no_calibration", action="store_true", help="Skip the calibration block."
    )
    p.add_argument(
        "--randomize_gestures",
        action="store_true",
        help="Shuffle gesture order independently for each orientation block.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --randomize_gestures (omit for a different order each run).",
    )
    p.add_argument(
        "--no_image_window",
        action="store_true",
        help="Disable the gesture reference image window.",
    )
    p.add_argument(
        "--images_dir",
        default=None,
        help=(
            "Directory containing gesture reference images. "
            "Defaults to an 'images/' folder next to this script."
        ),
    )
    return p


def _resolve_schedule(args: argparse.Namespace) -> list[CueEvent]:
    # Explicit "default" forces the built-in block design (CLI knobs apply).
    if args.schedule == "default":
        gestures = tuple(args.gestures) if args.gestures else DEFAULT_GESTURES
        calib = () if args.no_calibration else DEFAULT_CALIB_POSTURES
        return build_default_schedule(
            gestures=gestures,
            reps=args.reps,
            ready_s=args.ready_s,
            hold_s=args.hold_s,
            rest_s=args.rest_s,
            calibration_postures=calib,
            calibration_hold_s=args.calib_hold_s,
            randomize_gestures=args.randomize_gestures,
            seed=args.seed,
        )

    # No --schedule: auto-discover gesture_protocol.yaml/.json next to the script.
    if args.schedule is None:
        here = Path(__file__).resolve().parent
        for _name in (
            "gesture_protocol.yaml",
            "gesture_protocol.yml",
            "gesture_protocol.json",
        ):
            candidate = here / _name
            if candidate.is_file():
                print(f"[cue_player] Using auto-discovered schedule: {candidate}")
                return load_schedule(candidate)
        print(
            "[cue_player] No --schedule given and no gesture_protocol.yaml/.json "
            "found next to script -- using built-in default."
        )
        gestures = tuple(args.gestures) if args.gestures else DEFAULT_GESTURES
        calib = () if args.no_calibration else DEFAULT_CALIB_POSTURES
        return build_default_schedule(
            gestures=gestures,
            reps=args.reps,
            ready_s=args.ready_s,
            hold_s=args.hold_s,
            rest_s=args.rest_s,
            calibration_postures=calib,
            calibration_hold_s=args.calib_hold_s,
            randomize_gestures=args.randomize_gestures,
            seed=args.seed,
        )

    return load_schedule(Path(args.schedule))


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    events = _resolve_schedule(args)

    total = sum(e.duration_s for e in events)
    print(
        f"[cue_player] Resolved schedule: {len(events)} events, "
        f"~{total:0.1f} s total ({total / 60:0.1f} min)."
    )

    if args.print_schedule:
        for i, ev in enumerate(events):
            print(
                f"  {i:3d}  {ev.phase:14s}  {ev.label:24s}  "
                f"{ev.duration_s:5.2f}s  rep={ev.rep}"
            )
        return 0

    outlet = None if args.no_lsl else make_marker_outlet(args.lsl_name)
    if outlet is not None:
        print(
            f"[cue_player] LSL outlet '{args.lsl_name}' (Markers, "
            f"string, irregular) is live.  Start your recorder and press "
            f"any key in the GUI to begin."
        )

    # Gesture reference image window.
    image_window: GestureImageWindow | None = None
    if not args.no_image_window:
        # Defer actual construction until the Tk root exists (done in CuePlayerApp).
        if args.images_dir:
            _imgs_dir = Path(args.images_dir)
        else:
            _imgs_dir = Path(__file__).resolve().parent / "images"
        _imgs_dir.mkdir(parents=True, exist_ok=True)
    else:
        _imgs_dir = None  # type: ignore[assignment]

    app = CuePlayerApp(events, outlet, fullscreen=args.fullscreen)
    if _imgs_dir is not None:
        image_window = GestureImageWindow(app.root, _imgs_dir)
        app._image_window = image_window
    app.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
