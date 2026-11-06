"""
diagnostics.py — live motor diagnostics window (tkinter), used by xbox_live.py
and playback.py.

The control loop owns the BEAR bus: it reads each motor's status registers
(read_stats, one packet per motor) and passes the latest values to the window
a few times a second.  The window never talks to the motors itself.

The window runs in its own process (this file, started with --window), so
drawing it never delays the control loop.  The loop sends one JSON line per
refresh down a pipe without waiting (a refresh the window has not caught up
with is dropped); the window shows the newest.  Pressing STOP or closing the
window ends that process; the loop's update() then returns False and the robot
stops.  So does the window process dying for any other reason.

If tkinter or a display is missing, open_window() says why and returns None,
and the caller carries on without the window.

The window, top to bottom:
  STOP, and what the control loop is doing
  Graphs of the last GRAPH_SECONDS, with Tk alone so nothing else need be
  installed.  Each motor keeps one colour on every graph (legend above them),
  and the graphs resize with the window.
    Velocity     the wheels (WHEEL_MOTORS)
    Position     every other motor
    Error        command - delivered: one graph per mode in use (ERROR_GRAPHS),
                 each motor by the mode it is in.  Both sides come from the
                 same reading, so a motor's lag is real and not a timing
                 artefact.  Direct-force mode has no single error to plot.
    Iq           every motor (torque current)
    Winding      every motor
    Supply       the lowest voltage any motor reports
  A table with one column per motor, headed by its colour and its mode (read
  from the motor at startup).
    Status       OK, WARN (warning bit, current near its cap, hot) or ERROR
                 (error bit, disabled, no reply) with the reason, whether it
                 is enabled, and the error code returned with every reply
                 (128 = normal)
    Dropdowns, closed at the start; one click opens a section for every motor:
    Motion       Command (what the script sent), Goal velocity and Goal
                 position (what the motor holds — differs from Command if a
                 write was lost, as writes are not acknowledged), measured
                 Velocity and Position, at the BEAR output
    Electrical   Iq with its cap, Goal Iq, Id, Supply
    Thermal      Winding, Power stage, IC, the motor's temperature limits
    Comms        Last reply, Missed replies: how fresh the values are
"""

import json
import math
import os
import select
import subprocess
import sys
import time
from collections import deque

try:
    import tkinter as tk
except ImportError:          # Python built without tkinter (Debian/Ubuntu: apt install python3-tk)
    tk = None

# Status registers read in one packet per motor.  goal_* is what the motor
# holds as its goal, which confirms the last command arrived.
STATS = ["torque_enable", "goal_position", "goal_velocity", "goal_iq", "present_position", "present_velocity",
         "present_iq", "present_id", "input_voltage", "winding_temperature",
         "powerstage_temperature", "ic_temperature"]

MODE_NAMES = {0: "torque", 1: "velocity", 2: "position", 3: "direct force"}     # the BEAR's mode register

# Error-code bits (BEAR SDK Table 5); bit 7 is always 1.
CODE_BITS = {0: "communication", 1: "overheat", 2: "absolute position", 3: "watchdog/E-stop",
             4: "joint limit", 5: "hardware fault", 6: "initialization"}
WARNING_BITS = 0b0000_0011
ERROR_BITS   = 0b0111_1100

TEMP_WARN_C     = 70.0    # °C; the BEAR starts limiting power at 80 °C by default (temp_limit_low)
IQ_WARN_FRACTION = 0.9    # of the motor's current cap
NO_REPLY_S      = 1.0     # s without a reply before a motor shows as ERROR

COLORS = {"OK": "#2e7d32", "WARN": "#ef6c00", "ERROR": "#c62828", "WAIT": "#757575"}

WHEEL_MOTORS  = (1, 8)    # BEAR IDs graphed by velocity; every other motor by position
GRAPH_SECONDS = 30        # s of history on the graphs
GRAPH_SIZE    = (530, 130)     # px of a half-width graph at the start, before fitting the screen

# Command - delivered, per mode: {mode: (graph title, goal register, measured register, smallest y-range)}.
ERROR_GRAPHS = {1: ("Velocity error (rad/s)", "goal_velocity", "present_velocity", 0.5),
                2: ("Position error (rad at the motor)", "goal_position", "present_position", 0.2),
                0: ("Torque error, Iq (A)", "goal_iq", "present_iq", 0.2)}

# Line colours in display order, one per motor (categorical palette, fixed order).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, GRID, INK, INK_MUTED = "#fcfcfb", "#e8e7e3", "#0b0b0b", "#52514e"

# The dropdowns: (title, [(value, row label), ...]).
SECTIONS = [("Motion",     [("command", "Command"), ("goal_velocity", "Goal velocity (rad/s)"),
                            ("goal_position", "Goal position (rad)"), ("present_velocity", "Velocity (rad/s)"),
                            ("present_position", "Position (rad)")]),
            ("Electrical", [("present_iq", "Iq (A) / cap"), ("goal_iq", "Goal Iq (A)"), ("present_id", "Id (A)"),
                            ("input_voltage", "Supply (V)")]),
            ("Thermal",    [("winding_temperature", "Winding (°C)"), ("powerstage_temperature", "Power stage (°C)"),
                            ("ic_temperature", "IC (°C)"), ("temp_limit_low", "Limits power at (°C)"),
                            ("temp_limit_high", "Shuts down at (°C)")]),
            ("Comms",      [("age", "Last reply"), ("missed", "Missed replies")])]


def read_stats(bear, motor_id):
    """{register: value for STATS, "error": error code} for one motor in one
    packet, or None if it did not answer."""
    try:
        values, err = bear.single_read(motor_id, STATS)
    except Exception:
        return None
    if err is None or any(v is None for v in values):
        return None
    return dict(zip(STATS, values), error=err)


def motor_entry(bear, label, motor_id, iq_cap):
    """One motor's entry for open_window's `motors`: (label, motor_id, iq_cap,
    mode, temperature limit low, high), the last three read from the motor
    (None if it does not answer).  Call it after setting the motor's mode."""
    try:
        values, err = bear.single_read(motor_id, ["mode", "temp_limit_low", "temp_limit_high"])
    except Exception:
        values = [None, None, None]
    return (label, motor_id, iq_cap, *values)


def decode_code(code):
    """Names of the bits set in an error code, e.g. ['watchdog/E-stop']."""
    return [name for bit, name in CODE_BITS.items() if code & (1 << bit)]


def health(m, iq_cap):
    """(level, reason) for one motor's entry of the snapshot."""
    st = m["stats"]
    if st is None:
        return ("ERROR", "no reply") if m["age"] > NO_REPLY_S else ("WAIT", "waiting")
    if m["age"] > NO_REPLY_S:
        return "ERROR", f"no reply for {m['age']:.1f} s"
    if st["error"] & ERROR_BITS:
        return "ERROR", ", ".join(decode_code(st["error"] & ERROR_BITS))
    if st["torque_enable"] != 1:
        return "ERROR", "disabled"
    reasons = decode_code(st["error"] & WARNING_BITS)
    if abs(st["present_iq"]) >= IQ_WARN_FRACTION * iq_cap:
        reasons.append("current near cap")
    hottest = max(st["winding_temperature"], st["powerstage_temperature"], st["ic_temperature"])
    if hottest >= TEMP_WARN_C:
        reasons.append(f"{hottest:.0f} °C")
    return ("WARN", ", ".join(reasons)) if reasons else ("OK", "")


def status_text(m, level, reason):
    """The Status cell: level and reason, enabled state, error code."""
    first = f"{level}: {reason}" if reason and reason not in ("disabled", "waiting") else level
    st = m["stats"]
    if st is None:
        return first
    enabled = {1: "enabled", 0: "disabled"}.get(st["torque_enable"], f"torque_enable {st['torque_enable']}")
    return f"{first}\n{enabled}\ncode {st['error']}"


def nice_step(span, ticks=4):
    """A round gridline spacing (1, 2 or 5 × 10^n) giving about `ticks` lines over `span`."""
    raw = span / ticks
    power = 10.0 ** math.floor(math.log10(raw))
    return next(m * power for m in (1, 2, 5, 10) if m * power >= raw)


class Graph:
    """
    A live line chart of the last GRAPH_SECONDS on a Tk canvas, one y-axis,
    scaled to what is on it, sized to its canvas.  series: list of (key,
    colour).  min_span: the smallest y-range shown, so sensor noise does not
    fill the graph.  A single series gives its latest value in the title line.
    """
    LEFT, RIGHT, TOP, BOTTOM = 46, 10, 20, 16

    def __init__(self, parent, title, series, width, height, min_span):
        self.title, self.series, self.min_span = title, series, min_span
        self.samples = {key: deque() for key, _ in series}
        self.canvas = tk.Canvas(parent, width=width, height=height, bg=SURFACE, highlightthickness=0)

    def add(self, key, t, value):
        self.samples[key].append((t, value))

    def draw(self, now):
        c, t0 = self.canvas, now - GRAPH_SECONDS
        for points in self.samples.values():
            while points and points[0][0] < t0:
                points.popleft()
        c.delete("all")
        latest = {key: points[-1][1] for key, points in self.samples.items() if points}
        heading = self.title
        if len(self.series) == 1 and latest:
            heading += f":  {next(iter(latest.values())):.2f}"
        c.create_text(self.LEFT, 4, text=heading, anchor="nw", fill=INK, font=("TkDefaultFont", 9, "bold"))
        left, right, top, bottom = self.LEFT, c.winfo_width() - self.RIGHT, self.TOP, c.winfo_height() - self.BOTTOM
        for age in range(0, GRAPH_SECONDS + 1, 10):
            x = right - (right - left) * age / GRAPH_SECONDS
            c.create_line(x, top, x, bottom, fill=GRID)
            c.create_text(x, bottom + 2, text=f"-{age} s" if age else "now", anchor="n", fill=INK_MUTED,
                          font=("TkDefaultFont", 8))
        values = [v for points in self.samples.values() for _, v in points]
        if not values:
            return
        lo, hi = min(values), max(values)
        if hi - lo < self.min_span:
            lo, hi = (lo + hi - self.min_span) / 2, (lo + hi + self.min_span) / 2
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        step = nice_step(hi - lo)

        def y_of(v):
            return bottom - (bottom - top) * (v - lo) / (hi - lo)

        tick = math.ceil(lo / step) * step
        while tick <= hi:
            c.create_line(left, y_of(tick), right, y_of(tick), fill=GRID)
            c.create_text(left - 4, y_of(tick), text=f"{tick:g}", anchor="e", fill=INK_MUTED, font=("TkDefaultFont", 8))
            tick += step
        for key, color in self.series:
            coords = [xy for t, v in self.samples[key]
                      for xy in (right - (right - left) * (now - t) / GRAPH_SECONDS, y_of(v))]
            if len(coords) >= 4:
                c.create_line(*coords, fill=color, width=2, joinstyle="round")
            elif coords:
                c.create_oval(coords[0] - 2, coords[1] - 2, coords[0] + 2, coords[1] + 2, fill=color, outline="")


class DiagnosticsWindow:
    """
    The window (see the module docstring).  motors: list of motor_entry()
    tuples in display order.  stop_hint: the line under the loop information.
    """

    def __init__(self, title, motors, stop_hint="STOP, Ctrl-C or closing this window stop the robot."):
        self.motors = motors
        self.stop_requested = False
        self.root = tk.Tk()
        self.root.title(title)
        self.root.tk_setPalette(background=SURFACE, foreground=INK)
        self.root.protocol("WM_DELETE_WINDOW", self._stop)
        self.color = {m[1]: SERIES_COLORS[i % len(SERIES_COLORS)] for i, m in enumerate(motors)}
        self.seen = {m[1]: None for m in motors}      # time of the reading last graphed, per motor

        # The table is packed before the graphs, so if the window is short it
        # is the graphs that give way.
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(8, 4))
        tk.Button(top, text="STOP", command=self._stop, bg=COLORS["ERROR"], fg="white", width=10,
                  activebackground="#8e0000", activeforeground="white",
                  font=("TkDefaultFont", 16, "bold")).pack(side="right", padx=(8, 0))
        self.info = tk.Label(top, text="starting…", font="TkFixedFont", anchor="w")
        self.info.pack(fill="x")
        tk.Label(top, text=stop_hint, font=("TkDefaultFont", 9), fg=INK_MUTED, anchor="w").pack(fill="x")

        legend = tk.Frame(self.root)
        legend.pack(fill="x", padx=8, pady=(0, 4))
        for label, mid, *_ in motors:
            tk.Frame(legend, bg=self.color[mid], width=10, height=10).pack(side="left", padx=(0, 4))
            tk.Label(legend, text=label, font=("TkDefaultFont", 9)).pack(side="left", padx=(0, 14))

        table = tk.Frame(self.root)
        table.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        graphs = tk.Frame(self.root)
        graphs.pack(fill="both", expand=True, padx=8)
        width = GRAPH_SIZE[0]
        height = max(80, min(GRAPH_SIZE[1], (self.root.winfo_screenheight() - 360) // 4))

        def series(selected):
            return [(m[1], self.color[m[1]]) for m in motors if selected(m)]

        everyone = series(lambda m: True)
        self.velocity = Graph(graphs, "Velocity (rad/s)", series(lambda m: m[1] in WHEEL_MOTORS), width, height, 1.0)
        self.position = Graph(graphs, "Position (rad at the motor)", series(lambda m: m[1] not in WHEEL_MOTORS),
                              width, height, 1.0)
        self.iq = Graph(graphs, "Iq (A)", everyone, width, height, 0.5)
        self.winding = Graph(graphs, "Winding (°C)", everyone, width, height, 5.0)
        self.supply = Graph(graphs, "Supply, lowest of all motors (V)", [("lowest", INK_MUTED)],
                            2 * width + 8, height * 3 // 4, 1.0)
        # One error graph per mode in use, sharing a row; the grid has six
        # equal columns so one, two or three graphs divide it evenly.
        modes = [mode for mode in ERROR_GRAPHS if any(m[3] == mode for m in motors)]
        self.errors = {}                              # motor_id: (graph, goal register, measured register)
        error_graphs = []
        for mode in modes:
            title, goal, measured, min_span = ERROR_GRAPHS[mode]
            graph = Graph(graphs, title, series(lambda m: m[3] == mode),
                          (2 * width + 8) // len(modes), height, min_span)
            error_graphs.append(graph)
            for m in motors:
                if m[3] == mode:
                    self.errors[m[1]] = (graph, goal, measured)
        self.graphs = [self.velocity, self.position, *error_graphs, self.iq, self.winding, self.supply]

        for c in range(6):
            graphs.columnconfigure(c, weight=1, uniform="graphs")
        placed = [(self.velocity, 0, 0, 3), (self.position, 0, 3, 3)]
        placed += [(g, 1, i * 6 // len(modes), 6 // len(modes)) for i, g in enumerate(error_graphs)]
        placed += [(self.iq, 2, 0, 3), (self.winding, 2, 3, 3), (self.supply, 3, 0, 6)]
        for graph, row, column, span in placed:
            graph.canvas.grid(row=row, column=column, columnspan=span, sticky="nsew",
                              padx=(0 if column == 0 else 8, 0), pady=(0, 8))
        for row, weight in ((0, 4), (1, 4 if modes else 0), (2, 4), (3, 3)):
            graphs.rowconfigure(row, weight=weight)

        font, bold = "TkFixedFont", ("TkDefaultFont", 9, "bold")
        table.columnconfigure(0, minsize=175)         # wide enough for every row label, open or closed
        for c, m in enumerate(motors, start=1):
            head = tk.Frame(table)
            head.grid(row=0, column=c, sticky="we", padx=3)
            tk.Frame(head, bg=self.color[m[1]], height=4).pack(fill="x")
            tk.Label(head, text=f"{m[0]}\nBEAR {m[1]} · {MODE_NAMES.get(m[3], 'mode ?')}", font=bold).pack()
        tk.Label(table, text="Status", font=font, anchor="nw").grid(row=1, column=0, sticky="nw")
        self.cells = {}
        for c, m in enumerate(motors, start=1):
            self.cells["status", m[1]] = tk.Label(table, text="—", font=font, width=14, anchor="nw",
                                                  justify="left", wraplength=108)
            self.cells["status", m[1]].grid(row=1, column=c, sticky="nw", padx=3)
        self.sections = {}
        row = 2
        for name, rows in SECTIONS:
            button = tk.Button(table, text=f"[+] {name}", font=bold, anchor="w", relief="flat", bd=0,
                               command=lambda n=name: self.toggle(n))
            button.grid(row=row, column=0, columnspan=len(motors) + 1, sticky="we", pady=(4, 0))
            row += 1
            widgets = []
            for key, label in rows:
                widgets.append(tk.Label(table, text=label, font=font, anchor="w"))
                widgets[-1].grid(row=row, column=0, sticky="w", padx=(12, 0))
                for c, m in enumerate(motors, start=1):
                    self.cells[key, m[1]] = tk.Label(table, text="—", font=font, width=14, anchor="w")
                    self.cells[key, m[1]].grid(row=row, column=c, sticky="w", padx=3)
                    widgets.append(self.cells[key, m[1]])
                row += 1
            for w in widgets:
                w.grid_remove()
            self.sections[name] = {"button": button, "widgets": widgets, "keys": [k for k, _ in rows], "open": False}
        self.root.update()

    def _stop(self):
        self.stop_requested = True

    def toggle(self, name):
        """Open or close one dropdown."""
        section = self.sections[name]
        section["open"] = not section["open"]
        section["button"].config(text=f"[{'-' if section['open'] else '+'}] {name}")
        for w in section["widgets"]:
            if section["open"]:
                w.grid()
            else:
                w.grid_remove()

    def update(self, snap):
        """
        Show the snapshot and handle window events.  Returns False once STOP
        was pressed or the window closed.

        snap: {"info": str, "motors": {motor_id: {"command", "unit" (of the
        command, default rad/s), "stats" (dict from read_stats or None),
        "last_reply" (time of that reading), "age" (s since it), "missed"}}}
        """
        if self.stop_requested:
            return False
        now = time.time()
        try:
            self.info.config(text=snap["info"])
            voltages = []
            for label, mid, iq_cap, mode, temp_low, temp_high in self.motors:
                m = snap["motors"][mid]
                st = m["stats"]
                self.cells["status", mid].config(text=status_text(m, *health(m, iq_cap)),
                                                 fg=COLORS[health(m, iq_cap)[0]])
                if st is not None:
                    voltages.append(st["input_voltage"])
                    if m["last_reply"] != self.seen[mid]:
                        self.seen[mid] = m["last_reply"]
                        t = m["last_reply"]
                        if mid in WHEEL_MOTORS:
                            self.velocity.add(mid, t, st["present_velocity"])
                        else:
                            self.position.add(mid, t, st["present_position"])
                        self.iq.add(mid, t, st["present_iq"])
                        self.winding.add(mid, t, st["winding_temperature"])
                        if mid in self.errors:
                            graph, goal, measured = self.errors[mid]
                            graph.add(mid, t, st[goal] - st[measured])
                text = {"command": f"{m['command']:+.3f} {m.get('unit', 'rad/s')}",
                        "age": f"{m['age']:.2f} s ago" if st is not None else "never",
                        "missed": str(m["missed"]),
                        "temp_limit_low": "—" if temp_low is None else f"{temp_low:.0f}",
                        "temp_limit_high": "—" if temp_high is None else f"{temp_high:.0f}"}
                if st is not None:
                    text.update({k: f"{st[k]:+.3f}" for k in ("goal_velocity", "goal_position", "goal_iq",
                                                              "present_velocity", "present_position", "present_id")})
                    text["present_iq"] = f"{st['present_iq']:+.2f} / {iq_cap:g}"
                    text["input_voltage"] = f"{st['input_voltage']:.1f}"
                    for k in ("winding_temperature", "powerstage_temperature", "ic_temperature"):
                        text[k] = f"{st[k]:.1f}"
                for section in self.sections.values():
                    if section["open"]:
                        for key in section["keys"]:
                            self.cells[key, mid].config(text=text.get(key, "—"))
            if voltages:
                self.supply.add("lowest", now, min(voltages))
            self.root.update_idletasks()     # so the graphs draw to their current size
            for graph in self.graphs:
                graph.draw(now)
            self.root.update()
        except tk.TclError:              # window destroyed
            return False
        return not self.stop_requested

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


class DiagnosticsProcess:
    """The control loop's handle on the window process (see the module docstring)."""

    def __init__(self, proc):
        self.proc = proc
        os.set_blocking(proc.stdin.fileno(), False)

    def update(self, snap):
        """Send a snapshot (see DiagnosticsWindow.update) without waiting.
        Returns False once the window is gone (STOP, closed, or crashed)."""
        if self.proc.poll() is not None:
            return False
        try:
            os.write(self.proc.stdin.fileno(), (json.dumps(snap) + "\n").encode())
        except BlockingIOError:          # the window is behind: drop this refresh
            pass
        except (BrokenPipeError, OSError):
            return False
        return True

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def open_window(title, motors, stop_hint=None, timeout=10.0):
    """
    Start the window process and wait for it to come up.  motors: list of
    motor_entry() tuples in display order.  stop_hint: the line under the
    loop information (naming the controller's stop button, say).  Returns a
    DiagnosticsProcess, or None (with the reason printed) if the window
    cannot be opened.
    """
    config = {"title": title, "motors": motors}
    if stop_hint:
        config["stop_hint"] = stop_hint
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--window", json.dumps(config)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    line = proc.stdout.readline().decode().strip() if ready else f"no answer in {timeout:g} s"
    if line != "ready":
        line = line or "the window process exited"
        proc.kill()
        print(f"Diagnostics window unavailable ({line.removeprefix('error ')}). Continuing without it.")
        return None
    return DiagnosticsProcess(proc)


def _window_main(config):
    """The window process: show the newest snapshot from stdin until STOP, close or end of input."""
    if tk is None:
        print("error this Python has no tkinter (Debian/Ubuntu: apt install python3-tk)", flush=True)
        return
    try:
        window = DiagnosticsWindow(config["title"], [tuple(m) for m in config["motors"]],
                                   **({"stop_hint": config["stop_hint"]} if "stop_hint" in config else {}))
    except tk.TclError as e:
        print(f"error {e}", flush=True)
        return
    print("ready", flush=True)
    fd = sys.stdin.fileno()
    os.set_blocking(fd, False)
    buffer = b""
    while True:
        try:
            chunk = os.read(fd, 1 << 16)
            if not chunk:                # the control loop has finished
                break
            buffer += chunk
        except BlockingIOError:
            pass
        *lines, buffer = buffer.split(b"\n")
        if lines:                        # redraw only when there is something new
            snap = json.loads(lines[-1])
            snap["motors"] = {int(k): v for k, v in snap["motors"].items()}
            if not window.update(snap):
                break
        else:
            try:
                window.root.update()     # still handle clicks and window events
            except tk.TclError:
                break
            if window.stop_requested:
                break
        time.sleep(0.02)
    window.close()


if __name__ == "__main__" and "--window" in sys.argv:
    _window_main(json.loads(sys.argv[sys.argv.index("--window") + 1]))
