#!/usr/bin/env python3
"""
prototype_maatouch.py  --  STANDALONE viability probe for MaaTouch.

This file imports NOTHING from MightyGrid and changes NOTHING in your code.
It exists only to answer one question before we invest in an integration:

    Does MaaTouch actually drive touches on THIS Realme / ColorOS / Android 16 phone?

What it does, in order:
  1. Pushes the `maatouch` binary to /data/local/tmp/ .
  2. Launches it via app_process and reads its startup HEADER.
        -> If the header prints, the binary RUNS on your device (the big unknown).
  3. Plays a real TAP and a visible SWIPE using the maatouch protocol
     (down / move / up / commit, with pressure).
        -> If uiautomator2 is installed, it also checks the screen actually changed,
           giving an objective PASS/FAIL. Otherwise you just watch the screen.

Run (unlock the phone first so the screen is on):
    python prototype_maatouch.py                 # auto-picks the only connected device
    python prototype_maatouch.py S8V8UGEAUW4LSCLV # or pass the serial explicitly

Delete this file afterwards and nothing remains. Zero risk to the working code.
"""
import subprocess, sys, time, threading, queue, os, re

HERE       = os.path.dirname(os.path.abspath(__file__))
BIN_LOCAL  = os.path.join(HERE, "maatouch")     # put the maatouch binary next to this script
BIN_REMOTE = "/data/local/tmp/maatouch"
# The well-known MaaTouch entry class. If the header never prints, the alt launch form
# below (with /system/bin) is worth trying -- see the note where we build the command.
MAA_CLASS  = "com.shxyke.MaaTouch.App"


def adb(args, serial=None):
    cmd = ["adb"] + (["-s", serial] if serial else []) + args
    return subprocess.run(cmd, capture_output=True, text=True)


def list_devices():
    lines = adb(["devices"]).stdout.strip().splitlines()[1:]
    return [l.split()[0] for l in lines if l.strip() and l.split()[-1] == "device"]


def screen_size(serial):
    out = adb(["shell", "wm", "size"], serial).stdout          # "Physical size: 1080x2400"
    m = re.search(r'(\d+)\s*x\s*(\d+)', out)
    return (int(m.group(1)), int(m.group(2))) if m else (1080, 2400)


def main():
    serial = sys.argv[1] if len(sys.argv) > 1 else None
    if not serial:
        devs = list_devices()
        if len(devs) != 1:
            print("Connect exactly ONE device (or pass the serial). adb sees:", devs)
            return
        serial = devs[0]
    print("Device:", serial)

    if not os.path.exists(BIN_LOCAL):
        print("ERROR: the 'maatouch' binary must sit next to this script:\n   %s" % BIN_LOCAL)
        return

    # wake the screen so the gesture is visible (harmless keyevent, not maatouch yet)
    adb(["shell", "input", "keyevent", "224"], serial)   # KEYCODE_WAKEUP
    time.sleep(0.4)

    # 1) push the binary
    print("Pushing maatouch to the device ...")
    r = adb(["push", BIN_LOCAL, BIN_REMOTE], serial)
    if r.returncode != 0:
        print("  push FAILED:", r.stderr.strip()); return
    adb(["shell", "chmod", "755", BIN_REMOTE], serial)

    w, h = screen_size(serial)
    print("Screen size: %dx%d" % (w, h))

    # 2) launch maatouch with stdin kept open.
    #    No PTY is allocated because we pass a command AND stdin is a pipe, so bytes are raw.
    #    If the header never appears, try changing "app_process", "/" to
    #    "app_process", "/system/bin"  (some ROMs want that) -- everything else stays.
    cmd = ["adb", "-s", serial, "shell",
           "CLASSPATH=%s" % BIN_REMOTE, "app_process", "/", MAA_CLASS]
    print("Launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=0)

    # cross-platform non-blocking reads: a reader thread feeds a queue
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(ln) for ln in iter(proc.stdout.readline, b"")],
                     daemon=True).start()

    def read_line(timeout):
        try:
            return q.get(timeout=timeout).decode(errors="replace").strip("\r\n ")
        except queue.Empty:
            return None

    # 3) read the header:  v <ver> | ^ <contacts> <max_x> <max_y> <max_pressure> | $ <pid>
    max_x = max_y = max_p = None
    t0 = time.time()
    while time.time() - t0 < 6.0:
        ln = read_line(1.0)
        if ln is None:
            continue
        print("  maatouch>", ln)
        if ln.startswith("^"):
            p = ln.split()
            if len(p) >= 5:
                max_x, max_y, max_p = int(p[2]), int(p[3]), int(p[4])
        if ln.startswith("$"):
            break

    if max_x is None:
        print("\n>>> FAIL: maatouch printed no header -- it likely can't run on this device.")
        print("    Read any Java/SELinux error above. Things to try:")
        print("      - change  app_process /  to  app_process /system/bin  in this script")
        print("      - confirm the binary pushed:  adb shell ls -l %s" % BIN_REMOTE)
        try: proc.terminate()
        except Exception: pass
        return

    print("\n>>> HEADER OK -- maatouch is RUNNING on this device.")
    print("    max_x=%d  max_y=%d  max_pressure=%d" % (max_x, max_y, max_p))

    press = max(1, max_p // 2)                      # a "normal" press relative to this device
    sx = lambda px: max(0, min(max_x, round(px / w * max_x)))
    sy = lambda py: max(0, min(max_y, round(py / h * max_y)))

    def send(line):
        proc.stdin.write((line + "\n").encode()); proc.stdin.flush()

    def gesture(lines):                             # send a full gesture, ONE commit at the end
        for ln in lines:
            send(ln)
        send("c")

    # optional objective check via uiautomator2 (you already have it)
    d = None
    try:
        import uiautomator2 as u2
        d = u2.connect(serial)
    except Exception:
        pass
    snap = lambda: (d.dump_hierarchy() if d else None)

    # ---- TEST A: a real tap in the centre (harmless -- taps whatever is there) ----
    print("\n[Test A] tapping the centre of the screen ...")
    cx, cy = sx(w * 0.5), sy(h * 0.5)
    gesture(["d 0 %d %d %d" % (cx, cy, press), "w 60", "u 0"])
    time.sleep(0.8)

    # ---- TEST B: a visible SWIPE -- pull the notification shade DOWN from the top ----
    #      (works from any screen; the tree changes a lot when the shade opens)
    print("[Test B] swiping DOWN from the top edge (should pull the notification shade) ...")
    before = snap()
    x = sx(w * 0.5)
    y0, y1 = h * 0.01, h * 0.55
    steps = 18
    lines = ["d 0 %d %d %d" % (x, sy(y0), press)]
    for i in range(1, steps + 1):
        yy = sy(y0 + (y1 - y0) * i / steps)
        lines += ["w 12", "m 0 %d %d %d" % (x, yy, press)]
    lines += ["u 0"]
    gesture(lines)
    time.sleep(1.3)
    after = snap()

    if before is not None and after is not None:
        changed = before != after
        print("    screen changed after the swipe:", changed)
        if changed:
            print("\n==================  PASS  ==================")
            print("maatouch injected a real gesture and the UI reacted. It works here.")
            print("===========================================")
        else:
            print("\n===============  INCONCLUSIVE  =============")
            print("Header worked (binary runs) but the shade didn't open. WATCH the screen")
            print("and re-run; if the tap in Test A moved something, injection still works.")
            print("===========================================")
    else:
        print("\n(uiautomator2 not importable here -- WATCH THE SCREEN instead:)")
        print("    Did the notification shade pull down, or anything visibly move?")

    # tidy up: swipe the shade back up if it opened
    lines = ["d 0 %d %d %d" % (x, sy(y1), press)]
    for i in range(1, steps + 1):
        yy = sy(y1 - (y1 - y0) * i / steps)
        lines += ["w 12", "m 0 %d %d %d" % (x, yy, press)]
    lines += ["u 0"]
    gesture(lines)
    time.sleep(0.4)

    print("\nDone. Closing maatouch.")
    try:
        proc.stdin.close(); proc.terminate()
    except Exception:
        pass


if __name__ == "__main__":
    main()
    