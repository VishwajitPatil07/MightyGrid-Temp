import argparse
import os
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hands.adb_client import execute_tap, get_device_for_serial


def get_connected_devices():
    try:
        output = subprocess.check_output(["adb", "devices"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Could not list ADB devices: %s" % exc)
    return [line.split()[0] for line in output.splitlines()[1:]
            if len(line.split()) >= 2 and line.split()[1] == "device"]


def resolve_serial(requested_serial):
    devices = get_connected_devices()
    if requested_serial:
        if requested_serial not in devices:
            raise SystemExit("Device %s is not connected and ready." % requested_serial)
        return requested_serial
    if len(devices) == 1:
        return devices[0]
    if not devices:
        raise SystemExit("No ready ADB devices found.")
    raise SystemExit("Multiple devices found; choose one with --serial SERIAL.")


def main():
    parser = argparse.ArgumentParser(description="Tap a normalized point on one Android device.")
    parser.add_argument("--serial", help="ADB serial of the device to tap")
    parser.add_argument("--x", type=float, default=500, help="Normalized x coordinate (0-1000)")
    parser.add_argument("--y", type=float, default=500, help="Normalized y coordinate (0-1000)")
    args = parser.parse_args()

    serial = resolve_serial(args.serial)
    get_device_for_serial(serial)
    print("[%s] Tapping normalized (%.0f, %.0f)." % (serial, args.x, args.y))
    execute_tap(serial, args.x, args.y)


if __name__ == "__main__":
    main()
