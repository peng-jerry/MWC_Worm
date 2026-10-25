"""
xbox_test.py — verify Xbox controller axes/buttons.

Run straight through with:
    python xbox_test.py

Tests in order:
  1. Detect Xbox controller and list axes/buttons
  2. Walk through each axis one at a time with prompts
"""

import sys

try:
    import evdev
    from evdev import ecodes
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "evdev"], check=True)
    import evdev
    from evdev import ecodes

# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def ok(msg):  print(f"  ✓  {msg}")
def warn(msg): print(f"  !  {msg}")
def fail(msg): print(f"  ✗  {msg}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Detect Xbox controller
# ═══════════════════════════════════════════════════════════════════════════════
section("1 / 2  Detect Xbox controller")

candidates = []
for path in evdev.list_devices():
    dev = evdev.InputDevice(path)
    if any(k in dev.name for k in ("Xbox", "X-Box", "Microsoft")):
        candidates.append(dev)
    else:
        dev.close()

if not candidates:
    print("\n  No Xbox controller found. All visible input devices:")
    for path in evdev.list_devices():
        d = evdev.InputDevice(path)
        print(f"    {d.path}  {d.name!r}")
        d.close()
    print("\n  Plug in the controller and re-run.")
    sys.exit(1)

joy = candidates[0]
ok(f"Found: {joy.name}  ({joy.path})")

caps     = joy.capabilities(verbose=False)
abs_info = {c: joy.absinfo(c) for c, _ in caps.get(ecodes.EV_ABS, [])}
btn_codes = set(caps.get(ecodes.EV_KEY, []))

abs_names = []
for c in sorted(abs_info):
    n = ecodes.ABS[c]
    abs_names.append(n if isinstance(n, str) else n[0])
print(f"\n  ABS axes  ({len(abs_info)}): {abs_names}")
print(f"  Buttons   ({len(btn_codes)}): present")

AXIS_LEFT_Y  = ecodes.ABS_Y
AXIS_RIGHT_X = ecodes.ABS_RX
AXIS_LT      = ecodes.ABS_Z
AXIS_RT      = ecodes.ABS_RZ
BTN_KILL     = ecodes.BTN_EAST   # B button

needed = {AXIS_LEFT_Y, AXIS_RIGHT_X, AXIS_LT, AXIS_RT}
missing = needed - set(abs_info)
if missing:
    warn(f"Expected axes not found: {[ecodes.ABS[c] for c in missing]}")
    warn("Check axis mapping and update AXIS_* constants if needed.")
else:
    ok("All expected axes present (ABS_Y, ABS_RX, ABS_Z, ABS_RZ)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Axis isolation tests (one at a time)
# ═══════════════════════════════════════════════════════════════════════════════
section("2 / 2  Axis isolation tests")
print("  You will be prompted to move each control. Keep everything else still.")

def read_all_abs():
    return {c: joy.absinfo(c).value for c in abs_info}

def abs_name(c):
    n = ecodes.ABS[c]
    return n if isinstance(n, str) else n[0]

def diff(before, after):
    return {c: (before[c], after[c]) for c in before if abs(after[c] - before[c]) > 5}

input("\n  Release ALL sticks and triggers, then press Enter...")
baseline = read_all_abs()

tests = [
    ("Left stick  UP/DOWN    (ABS_Y)",  AXIS_LEFT_Y,  "push UP then DOWN"),
    ("Right stick LEFT/RIGHT (ABS_RX)", AXIS_RIGHT_X, "push RIGHT then LEFT"),
    ("Left trigger  LT       (ABS_Z)",  AXIS_LT,      "pull LT fully"),
    ("Right trigger RT       (ABS_RZ)", AXIS_RT,      "pull RT fully"),
]

axis_results = {}
for label, expected_code, instruction in tests:
    input(f"\n  {label} — {instruction}, then press Enter...")
    current = read_all_abs()
    changed = diff(baseline, current)
    if not changed:
        warn(f"No axis moved. Try again and hold while pressing Enter.")
    else:
        for c, (b, a) in changed.items():
            flag = "  <-- EXPECTED" if c == expected_code else ""
            print(f"    code {c:3d} ({abs_name(c)}): {b} -> {a}{flag}")
        if expected_code in changed:
            ok(f"Confirmed on expected axis ({abs_name(expected_code)})")
            axis_results[expected_code] = True
        else:
            warn(f"Expected axis ({abs_name(expected_code)}) didn't move — update AXIS_* constants if needed")
            axis_results[expected_code] = False

# B button test
input("\n  Press and HOLD the B button, then press Enter...")
btn_ok = BTN_KILL in btn_codes
if btn_ok:
    ok("B button code present in capabilities")
else:
    warn("B button (BTN_EAST) not in capabilities — kill may not work")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
section("Summary")

summary = [("Expected axes present", not missing)]
for label, expected_code, _ in tests:
    summary.append((label, axis_results.get(expected_code, False)))
summary.append(("B button (BTN_EAST) present", btn_ok))

for label, passed in summary:
    (ok if passed else fail)(label)

if all(passed for _, passed in summary):
    print("\nAll tests passed.")
else:
    print("\nSome tests failed — see ✗ items above.")
