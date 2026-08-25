"""
MaaTouch transport -- preferred, isolated, fail-safe.

Streams touch events to the MaaTouch daemon (minitouch protocol) over adb, giving
ON-DEVICE gesture timing (no host sleep jitter) and real pressure. It is preferred
when available; set MG_TOUCH=u2 to force the uiautomator2 path. Crucially, EVERY
failure degrades to u2 -- a missing binary, a connect failure, a dead daemon, or a
write error all just return False so the caller falls back. Nothing in this module can
break the working flows.

Protocol (MaaTouch / minitouch):
    d id x y pressure   touch down
    m id x y pressure   move
    u id                touch up
    c                   commit
    w ms                wait ON THE DEVICE (this is what removes host sleep jitter)
Coordinates are device pixels (this phone reports max_x/max_y == screen size, so 1:1).
Pressure is 0..max_pressure (255 here).
"""
import os
import subprocess
import threading
import queue
import time
import atexit

_BIN_REMOTE = "/data/local/tmp/maatouch"
_MAA_CLASS = "com.shxyke.MaaTouch.App"

_instances = {}          # serial -> MaaTouch | None  (one connect attempt per run)
_lock = threading.Lock()


def _bin_candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))        # src/hands -> src -> project root
    return [
        os.environ.get("MG_MAATOUCH_BIN", ""),           # explicit override wins
        os.path.join(os.getcwd(), "maatouch"),           # run-from-root (the usual case)
        os.path.join(root, "maatouch"),
        os.path.join(here, "maatouch"),
    ]


class MaaTouch:
    """One persistent connection to the MaaTouch daemon on a device. Construction tries
    to connect; on any problem it stays .ok == False and the caller uses u2 instead."""

    def __init__(self, serial):
        self.serial = serial
        self.ok = False
        self.max_x = self.max_y = self.max_p = None
        self.proc = None
        self._q = queue.Queue()
        self._wlock = threading.Lock()
        try:
            self._connect()
        except Exception as e:
            self._err = str(e)
            self.close()

    def _adb(self, args, timeout=30):
        return subprocess.run(["adb", "-s", self.serial] + args,
                              capture_output=True, text=True, timeout=timeout)

    def _connect(self):
        binp = next((p for p in _bin_candidates() if p and os.path.exists(p)), None)
        if not binp:
            raise RuntimeError("maatouch binary not found (project root or MG_MAATOUCH_BIN)")
        # push is idempotent; chmod so app_process can read it
        self._adb(["push", binp, _BIN_REMOTE])
        self._adb(["shell", "chmod", "755", _BIN_REMOTE])
        cmd = ["adb", "-s", self.serial, "shell",
               "CLASSPATH=%s" % _BIN_REMOTE, "app_process", "/", _MAA_CLASS]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=0)
        # reader thread -> queue (cross-platform; no select on pipes)
        threading.Thread(
            target=lambda: [self._q.put(l) for l in iter(self.proc.stdout.readline, b"")],
            daemon=True).start()
        # read the header: ^ <contacts> <max_x> <max_y> <max_pressure> ; $ <pid>
        t0 = time.time()
        while time.time() - t0 < 6.0:
            try:
                ln = self._q.get(timeout=1.0).decode(errors="replace").strip("\r\n ")
            except queue.Empty:
                continue
            if ln.startswith("^"):
                p = ln.split()
                if len(p) >= 5:
                    self.max_x, self.max_y, self.max_p = int(p[2]), int(p[3]), int(p[4])
            if ln.startswith("$"):
                break
        if not self.max_x:
            raise RuntimeError("no maatouch header (daemon may not run on this device)")
        self.ok = True

    def alive(self):
        return bool(self.ok and self.proc and self.proc.poll() is None)

    def _clampp(self, pressure):
        m = self.max_p or 255
        if pressure is None:
            pressure = m // 4
        return max(1, min(m, int(pressure)))

    def _write(self, lines):
        data = ("\n".join(lines) + "\n").encode()
        with self._wlock:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    def tap(self, x, y, hold_ms=55, pressure=None):
        """A single tap: down, hold hold_ms on the device, up. Returns True if sent."""
        if not self.alive():
            return False
        p = self._clampp(pressure)
        try:
            self._write(["d 0 %d %d %d" % (int(x), int(y), p),
                         "w %d" % max(1, int(hold_ms)),
                         "u 0", "c"])
            return True
        except Exception:
            self.ok = False
            return False

    def long_press(self, x, y, hold_ms=700, moves=None, pressure=None):
        """Hold one contact on-device, optionally including small drift moves.
        `moves` contains (x, y, wait_before_move_ms) entries and the remaining hold
        time is added before release. Returns True if the gesture was sent."""
        if not self.alive():
            return False
        p = self._clampp(pressure)
        total_ms = max(1, int(hold_ms))
        elapsed_ms = 0
        try:
            cmds = ["d 0 %d %d %d" % (int(x), int(y), p)]
            for mx, my, wait_ms in moves or ():
                wait_ms = max(0, int(wait_ms))
                if wait_ms:
                    cmds.append("w %d" % wait_ms)
                    elapsed_ms += wait_ms
                cmds.append("m 0 %d %d %d" % (int(mx), int(my), p))
            if elapsed_ms < total_ms:
                cmds.append("w %d" % (total_ms - elapsed_ms))
            cmds += ["u 0", "c"]
            self._write(cmds)
            return True
        except Exception:
            self.ok = False
            return False

    def swipe(self, downxy, moves, hold_after_down_ms=0, hold_before_up_ms=0, pressure=None):
        """A gesture. downxy=(x,y) is the touch-down; moves is a list of
        (x, y, wait_before_ms) played in order; the wait happens ON THE DEVICE (no host
        sleep). One commit at the end. Returns True if sent, False to fall back to u2."""
        if not self.alive() or not moves:
            return False
        p = self._clampp(pressure)
        try:
            cmds = ["d 0 %d %d %d" % (int(downxy[0]), int(downxy[1]), p)]
            if hold_after_down_ms > 0:
                cmds.append("w %d" % int(hold_after_down_ms))
            for (x, y, wms) in moves:
                if wms and wms > 0:
                    cmds.append("w %d" % int(wms))
                cmds.append("m 0 %d %d %d" % (int(x), int(y), p))
            if hold_before_up_ms > 0:
                cmds.append("w %d" % int(hold_before_up_ms))
            cmds += ["u 0", "c"]
            self._write(cmds)
            return True
        except Exception:
            self.ok = False
            return False

    def close(self):
        self.ok = False
        try:
            if self.proc:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
                self.proc.terminate()
        except Exception:
            pass


def get(serial):
    """Return a cached, connected MaaTouch for this serial, or None. Makes at most ONE
    connect attempt per run (so a failure doesn't re-cost the ~seconds connect on every
    action); never raises. The caller decides whether MaaTouch is enabled."""
    with _lock:
        if serial not in _instances:
            try:
                mt = MaaTouch(serial)
            except Exception:
                mt = None
            _instances[serial] = mt if (mt is not None and mt.ok) else None
        mt = _instances.get(serial)
        return mt if (mt is not None and mt.alive()) else None


def shutdown():
    with _lock:
        for mt in _instances.values():
            if mt is not None:
                mt.close()
        _instances.clear()


atexit.register(shutdown)