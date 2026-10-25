"""
diagnostics.py — live motor diagnostics window (tkinter), used by xbox_live.py.

The control loop owns the BEAR bus: it reads each motor's status registers
(read_stats, one packet per motor) and passes the latest values to the window
a few times a second.  The window never talks to the motors itself.

The window runs in its own process (this file, started with --window), so
drawing it — 15–35 ms per refresh — never delays the control loop.  The loop
sends one JSON line per refresh down a pipe without waiting (a refresh the
window has not caught up with is dropped); the window shows the newest.
Pressing STOP or closing the window ends that process; the loop's update()
then returns False and the robot stops.  So does the window process dying
for any other reason.

If tkinter or a display is missing, open_window() says why and returns None,
and the caller carries on without the window.

Motors are shown in blocks of COLUMNS_PER_BLOCK columns (two rows of four for
the 8-motor robot, front to back); each column's rows are:
  Status          OK, WARN (warning bit, current near its cap, hot) or
                  ERROR (error bit, disabled, no reply), with the reason
  Enabled         torque_enable (1 = enabled)
  Input           controller input driving the motor, -1 to 1
  Command         speed sent this tick (rad/s)
  Goal held       goal_velocity the motor holds — differs from Command if a
                  write was lost (writes are not acknowledged)
  Velocity, Position          measured, at the BEAR output
  Current iq, Current id      torque and excitation current (A), iq with its cap
  Supply          input voltage (V)
  Winding, Power stage, IC    temperatures (°C)
  Error code      8-bit code returned with every reply (128 = normal)
  Last reply, Missed replies  how fresh the values are
"""

import json
import os
import select
import subprocess
import sys
import time

try:
    import tkinter as tk
except ImportError:          # Python built without tkinter (Debian/Ubuntu: apt install python3-tk)
    tk = None

# Status registers read in one packet per motor.  goal_* is what the motor
# holds as its goal, which confirms the last command arrived.
STATS = ["torque_enable", "goal_position", "goal_velocity", "present_position", "present_velocity",
         "present_iq", "present_id", "input_voltage", "winding_temperature",
         "powerstage_temperature", "ic_temperature"]

# Error-code bits (BEAR SDK Table 5); bit 7 is always 1.
CODE_BITS = {0: "communication", 1: "overheat", 2: "absolute position", 3: "watchdog/E-stop",
             4: "joint limit", 5: "hardware fault", 6: "initialization"}
WARNING_BITS = 0b0000_0011
ERROR_BITS   = 0b0111_1100

TEMP_WARN_C     = 70.0    # °C; the BEAR starts limiting power at 80 °C by default (temp_limit_low)
IQ_WARN_FRACTION = 0.9    # of the motor's current cap
NO_REPLY_S      = 1.0     # s without a reply before a motor shows as ERROR

COLORS = {"OK": "#2e7d32", "WARN": "#ef6c00", "ERROR": "#c62828", "WAIT": "#757575"}
COLUMNS_PER_BLOCK = 4     # motors side by side before starting a new block below

ROWS = [("status", "Status"), ("torque_enable", "Enabled"), ("input", "Input (-1 to 1)"),
        ("command", "Command (rad/s)"), ("goal_velocity", "Goal held (rad/s)"),
        ("present_velocity", "Velocity (rad/s)"), ("present_position", "Position (rad)"),
        ("present_iq", "Current iq (A)"), ("present_id", "Current id (A)"),
        ("input_voltage", "Supply (V)"), ("winding_temperature", "Winding (°C)"),
        ("powerstage_temperature", "Power stage (°C)"), ("ic_temperature", "IC (°C)"),
        ("error", "Error code"), ("age", "Last reply"), ("missed", "Missed replies")]


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


class DiagnosticsWindow:
    """
    A window with one column per motor, in blocks of COLUMNS_PER_BLOCK.
    motors: list of (label, motor_id, iq_cap) in display order.  stop_hint:
    the line under the STOP button.
    """

    def __init__(self, title, motors, stop_hint="STOP, Ctrl-C or closing this window stop the robot."):
        self.motors = motors
        self.stop_requested = False
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._stop)
        font, bold = "TkFixedFont", ("TkDefaultFont", 10, "bold")
        span = min(len(motors), COLUMNS_PER_BLOCK) + 1

        self.info = tk.Label(self.root, text="starting…", font=font, anchor="w")
        self.info.grid(row=0, column=0, columnspan=span, sticky="we", padx=8, pady=(8, 4))
        self.cells = {}
        row = 1
        for start in range(0, len(motors), COLUMNS_PER_BLOCK):
            block = motors[start:start + COLUMNS_PER_BLOCK]
            for c, (label, mid, _) in enumerate(block, start=1):
                tk.Label(self.root, text=f"{label}\nBEAR {mid}", font=bold, width=24).grid(
                    row=row, column=c, padx=4, pady=(8 if start else 0, 0))
            for r, (key, name) in enumerate(ROWS, start=row + 1):
                tk.Label(self.root, text=name, font=font, anchor="w").grid(row=r, column=0, sticky="w", padx=8)
                for c, (_, mid, _) in enumerate(block, start=1):
                    cell = tk.Label(self.root, text="—", font=font, width=24, anchor="w")
                    cell.grid(row=r, column=c, sticky="w", padx=4)
                    self.cells[key, mid] = cell
            row += len(ROWS) + 1
        tk.Button(self.root, text="STOP", command=self._stop, bg=COLORS["ERROR"], fg="white",
                  activebackground="#8e0000", activeforeground="white", font=("TkDefaultFont", 16, "bold"),
                  height=2).grid(row=row, column=0, columnspan=span, sticky="we", padx=8, pady=(8, 2))
        tk.Label(self.root, text=stop_hint, font=("TkDefaultFont", 9)).grid(
            row=row + 1, column=0, columnspan=span, pady=(0, 8))
        self.root.update()

    def _stop(self):
        self.stop_requested = True

    def update(self, snap):
        """
        Show the snapshot and handle window events.  Returns False once STOP
        was pressed or the window closed.

        snap: {"info": str, "motors": {motor_id: {"input", "command", "stats"
        (dict from read_stats or None), "age" (s since last reply), "missed"}}}
        """
        if self.stop_requested:
            return False
        try:
            self.info.config(text=snap["info"])
            for label, mid, iq_cap in self.motors:
                m = snap["motors"][mid]
                level, reason = health(m, iq_cap)
                st = m["stats"] or {}
                text = {"status": f"{level}" + (f": {reason}" if reason and level != "WAIT" else ""),
                        "input": f"{m['input']:+.2f}", "command": f"{m['command']:+.3f}",
                        "age": f"{m['age']:.2f} s ago" if m["stats"] is not None else "never",
                        "missed": str(m["missed"])}
                if st:
                    text.update({k: f"{st[k]:+.3f}" for k in ("goal_velocity", "present_velocity",
                                                              "present_position", "present_id")})
                    text["present_iq"] = f"{st['present_iq']:+.2f}  (cap {iq_cap:g})"
                    text["torque_enable"] = str(st["torque_enable"])
                    text["input_voltage"] = f"{st['input_voltage']:.1f}"
                    for k in ("winding_temperature", "powerstage_temperature", "ic_temperature"):
                        text[k] = f"{st[k]:.1f}"
                    names = decode_code(st["error"])
                    text["error"] = f"{st['error']}" + (f"  ({', '.join(names)})" if names else "  (none)")
                for key, _ in ROWS:
                    cell = self.cells[key, mid]
                    cell.config(text=text.get(key, "—"),
                                fg=COLORS[level] if key == "status" else "black")
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
    Start the window process and wait for it to come up.  Returns a
    DiagnosticsProcess, or None (with the reason printed) if the window
    cannot be opened.  stop_hint: the line under the STOP button (naming
    the controller's stop button, say).
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
