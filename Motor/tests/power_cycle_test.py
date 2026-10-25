"""
power_cycle_test.py — confirm motor IDs survive a power cycle.

Run with:
    python power_cycle_test.py

Scans the bus, has you power-cycle the motors, then scans again and reports:
  - any ID that disappeared. A saved ID (assign_motor_id.py's save_config
    call) should survive; an ID that was only set in RAM reverts on
    power-off and won't show up in the second scan — that's the bug this
    test catches.
  - any motor reporting an Absolute Position fault (error bit 2) after the
    cycle — the fault the BEAR SDK manual ties to unstable power or a
    disconnected encoder battery. Koala-series BEARs (ours) have no battery
    backing the encoder and no set_posi() to clear this fault in software;
    the manual's fix is to power off, move the output shaft, and try again.

Read-only: only pings, never writes torque_enable, mode, id, or goals.
"""

import contextlib
import io
import sys
import time

from pybear import Manager

PORT = "/dev/serial/by-id/usb-Westwood_Robotics_USB2BEAR_Boosted_UB0104-if00-port0"
BAUD = 8_000_000
LO, HI = 0, 20
ABSOLUTE_POSITION_BIT = 1 << 2   # error code bit 2, per the BEAR SDK manual's error table

bear = Manager.BEAR(port=PORT, baudrate=BAUD, single_timeout=0.05, single_try_num=3)

def scan():
    """Return {id: error_code} for every motor that responds."""
    found = {}
    _devnull = io.StringIO()
    for mid in range(LO, HI + 1):
        with contextlib.redirect_stdout(_devnull):
            _, err = bear.ping(mid)[0]
        if err is not None:
            found[mid] = err
    return found

def report_absolute_position_faults(results):
    faulted = {mid: err for mid, err in results.items() if err & ABSOLUTE_POSITION_BIT}
    for mid, err in faulted.items():
        print(f"  WARNING: BEAR {mid} has an Absolute Position fault (err={err}, "
              f"{bear.decode_error(err)}). Power off, move the output shaft to a "
              f"different position, then power back on — set_posi() isn't available "
              f"on Koala-series BEARs, so there's no software fix.")
    return faulted

print(f"Scanning IDs {LO}-{HI} before power cycle...")
before = scan()
print(f"  Found: {sorted(before)}" if before else "  Found: none")
if not before:
    print("\nNo motors responded — check power and wiring before testing persistence.")
    bear.close_port()
    sys.exit(1)
report_absolute_position_faults(before)

input("\nPower-cycle the motor(s) now (unplug and reconnect power), "
      "then press Enter once they're back up...")

print(f"\nScanning IDs {LO}-{HI} after power cycle...")
after = scan()
print(f"  Found: {sorted(after)}" if after else "  Found: none")
faulted_after = report_absolute_position_faults(after)

bear.close_port()

lost  = sorted(set(before) - set(after))
new   = sorted(set(after) - set(before))

print()
if not lost and not new:
    print(f"All IDs persisted: {sorted(after)}")
elif lost and not new:
    print(f"WARNING: {lost} responded before the power cycle but not after — "
          f"their ID likely wasn't saved (reverted to factory default, probably colliding "
          f"with another motor now). Re-run assign_motor_id.py for them.")
else:
    print(f"Before: {sorted(before)}")
    print(f"After:  {sorted(after)}")
    if lost:
        print(f"Lost: {lost} — ID likely not saved, check those motors.")
    if new:
        print(f"New:   {new} — unexpected; a motor may have reverted to a different ID.")

if faulted_after:
    print(f"\nAbsolute Position fault on: {sorted(faulted_after)} — see warnings above.")
else:
    print("\nNo Absolute Position faults after the power cycle.")
