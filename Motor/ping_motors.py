"""
ping_motors.py — scan the BEAR bus and report which motor IDs respond.

Usage:
    python ping_motors.py            # scan IDs 0-20 (default range)
    python ping_motors.py 1-8        # scan a narrower range, e.g. IDs 1 through 8

Read-only: it only pings and reads present_position/input_voltage, never
writes torque_enable, mode, or goals, so it's safe to run at any time without
clearing the robot first.

A 0-20 scan takes a few seconds (3 retries x 50ms timeout per empty ID) — it
prints a running "." every 10 IDs so it doesn't look stuck.
PyBEAR prints its own "timed out, try again" warning for every empty ID,
which would otherwise bury that progress output, so this script suppresses
PyBEAR's prints during the scan.
"""

import contextlib
import io
import sys
import time

from pybear import Manager

PORT = "/dev/serial/by-id/usb-Westwood_Robotics_USB2BEAR_Boosted_UB0104-if00-port0"
BAUD = 8_000_000

if len(sys.argv) == 1:
    lo, hi = 0, 20
elif len(sys.argv) == 2 and "-" in sys.argv[1]:
    try:
        lo, hi = (int(x) for x in sys.argv[1].split("-", 1))
    except ValueError:
        print("Usage: python ping_motors.py [lo-hi]   e.g. python ping_motors.py 1-8")
        sys.exit(1)
else:
    print("Usage: python ping_motors.py [lo-hi]   e.g. python ping_motors.py 1-8")
    sys.exit(1)

def read1(getter, motor_id):
    data_list, err = getter(motor_id)[0]
    return data_list[0], err

bear = Manager.BEAR(port=PORT, baudrate=BAUD, single_timeout=0.05, single_try_num=3)

n_ids = hi - lo + 1
print(f"Scanning IDs {lo}-{hi} ({n_ids} IDs, ~{n_ids * 0.15:.0f}s worst case)...\n")
t0 = time.time()
found = []
_devnull = io.StringIO()
for i, mid in enumerate(range(lo, hi + 1)):
    with contextlib.redirect_stdout(_devnull):
        _, err = bear.ping(mid)[0]
        if err is not None:
            pos, _  = read1(bear.get_present_position, mid)
            volt, _ = read1(bear.get_input_voltage, mid)
    if err is None:
        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)
        continue
    found.append(mid)
    print(f"\n  BEAR {mid:3d}: online   position={pos:+7.3f} rad   voltage={volt:5.2f} V", end="", flush=True)

bear.close_port()

print(f"\n\nScan took {time.time() - t0:.1f}s.")
print(f"Found {len(found)} motor(s): {found}" if found else "No motors responded.")
if len(set(found)) != len(found):
    print("WARNING: duplicate IDs detected — this should be impossible from a scan; re-run.")
