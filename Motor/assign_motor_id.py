"""
assign_motor_id.py — give one isolated BEAR a new permanent ID.

RS-485 addressing is by ID only: if two motors currently share an ID, a
command sent to that ID is heard and acted on by BOTH of them at once —
there is no way to single one out over the wire.  So reassigning IDs (e.g.
renumbering every motor to a clean 1-8 scheme) has to happen one motor at a
time, in isolation:

  1. Disconnect EVERYTHING else from the signal chain — every other motor,
     old or new — leaving only the ONE motor you're about to renumber wired
     to the USB2BEAR adapter.  (A motor with an ID nothing else on the bus
     currently shares doesn't strictly need isolating from the others, but
     isolating one at a time is the simplest way to never get this wrong.)
  2. Run:  python assign_motor_id.py <new_id>
     e.g.  python assign_motor_id.py 4
     It scans first and refuses unless exactly one motor answers — if more
     than one responds, something upstream is still sharing an ID.
  3. Disconnect it, wire in the next motor, repeat with the next ID.
  4. Once every motor has its final unique ID, reconnect the full chain and
     re-run motor_test.py / a ping scan to confirm every ID responds together.

This only writes the id register — it does not touch gains, modes, or
MOTOR_IDS in homing.py / playback.py / motor_calibration.py (update those
yourself once you know the final wiring).
"""

import sys
import time

from pybear import Manager

PORT = "/dev/serial/by-id/usb-Westwood_Robotics_USB2BEAR_Boosted_UB0104-if00-port0"
BAUD = 8_000_000

if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    print("Usage: python assign_motor_id.py <new_id>")
    print("  new_id: 1-252 (0xFC), and not already used on this bus")
    sys.exit(1)

new_id = int(sys.argv[1])
if not (1 <= new_id <= 0xFC):
    print(f"new_id must be between 1 and {0xFC} (0xFC)")
    sys.exit(1)

bear = Manager.BEAR(port=PORT, baudrate=BAUD, single_timeout=0.05, single_try_num=3)

def read1(getter, motor_id):
    data_list, err = getter(motor_id)[0]
    return data_list[0], err

print("Scanning IDs 1-20 to check what's currently on the bus...")
seen = []
for mid in range(1, 21):
    _, err = bear.ping(mid)[0]
    if err is not None:
        seen.append(mid)
print(f"  Responding: {seen}")

if len(seen) != 1:
    print(f"\nExpected exactly one motor wired in, but found {seen}.")
    print("Disconnect every other motor so only the one you're renumbering "
          "is wired in, then re-run.")
    bear.close_port()
    sys.exit(1)

current_id = seen[0]

if current_id == new_id:
    print(f"\nMotor is already at ID {new_id} — nothing to do.")
    bear.close_port()
    sys.exit(0)

print(f"\nFound one motor at ID {current_id}. Assigning it ID {new_id}...")
bear.set_id((current_id, new_id))
time.sleep(0.1)

_, err_old = bear.ping(current_id)[0]
pos, err_new = read1(bear.get_present_position, new_id)

if err_new is not None and err_old is None:
    print(f"Confirmed: motor now responds at ID {new_id} (present_position={pos:+.4f} rad) "
          f"and no longer at ID {current_id}.")
    print("Disconnect it, wire in the next motor, and run this script again with the next ID.")
else:
    print(f"WARNING: could not confirm the new ID (ID {new_id} err={err_new}, ID {current_id} err={err_old}). "
          "Re-run the scan before trusting this motor's ID.")

bear.close_port()
