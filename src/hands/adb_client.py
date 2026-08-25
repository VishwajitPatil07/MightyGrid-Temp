import os
import time
import math
import random
import re
from typing import Tuple, List, Callable, Optional
import uiautomator2 as u2

# Imports the DNA factory for bot personas
from .personas import BiometricPersona, generate_random_persona
# Screen-aware input: read where keys actually are instead of guessing a grid
from src.eyes.screen import parse_elements, keyboard_keys

# OPT-IN low-level touch via MaaTouch (on-device timing + real pressure). Fully ISOLATED:
# wrapped in try/except so a broken/absent maatouch.py can never break this module, and
# every call below is gated by MG_TOUCH and falls back to uiautomator2 on any failure.
try:
    from . import maatouch as _maatouch
except Exception:
    _maatouch = None

Vec = Tuple[float, float]

_device_cache = {}
_DEVICE_SIZE = {}
_PERSONAS = {}
_LAST_XY = {}          # last place each device's finger touched (for Fitts travel)
_ACTION_COUNT = {}     # how many gestures a device has done (drives fatigue)

# ---------------------------------------------------------------------------
# Measured human motion, from the recorder's 1685 real gestures (p10/p50/p90).
# These are the ground truth every primitive below is tuned to hit.
#   tap hold        17 / 55  / 89   ms
#   long_press     448 / 702 / 1667 ms
#   flick   dur     58 / 88  / 160  ms  | straightness .966/.988/.998
#                   -> BACK-loaded: 50% of the distance is covered at t/T = 0.59
#                      (slow wind-up, fast ballistic release -> gives momentum)
#   scroll  dur    241 / 316 / 490  ms  | straightness .872/.989/.997
#                   -> FRONT-loaded: 50% of the distance is covered at t/T = 0.38
#                      (fast start, then decelerate and settle -> content stops
#                       with the finger, no skid)
# lognormal sigma is fitted from the spread: sigma = ln(p90/p10) / 2.56
# ---------------------------------------------------------------------------
_TAP_HOLD    = (0.055, 0.647, 0.017, 0.140)   # median, sigma, lo, hi  (seconds)
_LONGPRESS   = (0.702, 0.513, 0.400, 1.700)
_FLICK_DUR   = (0.088, 0.396, 0.050, 0.200)
_SCROLL_DUR  = (0.316, 0.277, 0.170, 0.600)


# ---------------------------------------------------------------------------
# Biomechanical Mathematics & Kinematic Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _unit(dx: float, dy: float) -> Vec:
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def _min_jerk(tau: float) -> float:
    # Symmetric minimum-jerk: the natural profile for a deliberate point-to-point
    # move (drag). 50% distance at t/T = 0.5.
    t = _clamp(tau, 0.0, 1.0)
    return 10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5)


def _ease_in(tau: float, power: float) -> float:
    # Back-loaded (distance builds late) -> a flick's ballistic whip.
    # power ~1.3 puts 50% distance near t/T = 0.59, matching the recorded flicks.
    return _clamp(tau, 0.0, 1.0) ** power


def _ease_out(tau: float, power: float) -> float:
    # Front-loaded (distance builds early, then decelerates) -> a controlled
    # scroll that settles. power ~1.45 puts 50% near t/T = 0.38, matching data.
    return 1.0 - (1.0 - _clamp(tau, 0.0, 1.0)) ** power


def _cubic_bezier(p0: Vec, c1: Vec, c2: Vec, p3: Vec, t: float) -> Vec:
    mt = 1.0 - t
    a, b, c, d = mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t
    return (
        a * p0[0] + b * c1[0] + c * c2[0] + d * p3[0],
        a * p0[1] + b * c1[1] + c * c2[1] + d * p3[1]
    )


def _control_points(start: Vec, end: Vec, curvature: float, bias: float, rng: random.Random) -> Tuple[Vec, Vec]:
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    dist = math.hypot(dx, dy) or 1.0
    px, py = -dy / dist, dx / dist

    # A real thumb-stroke bows to ONE side (recorded human scrolls have ~0.1 reversals --
    # a single clean arc, never a self-cancelling S). So both control points sit on the
    # SAME side: pick the bow direction ONCE (leaning with the persona's handedness), then
    # offset both control points that way. Magnitude ~= curvature*dist puts the arc at the
    # visible, curvier end of the measured human range (scroll/flick bow up to ~9-11% of
    # length) instead of the near-straight low end, so the curve actually reads on screen.
    amp = curvature * dist
    lean = max(-1.0, min(1.0, bias))
    sgn = 1.0 if rng.random() < (0.5 + 0.45 * lean) else -1.0
    o1 = sgn * amp * rng.uniform(0.80, 1.10)
    o2 = sgn * amp * rng.uniform(0.80, 1.10)

    f1, f2 = rng.uniform(0.2, 0.4), rng.uniform(0.6, 0.8)
    return (sx + dx * f1 + px * o1, sy + dy * f1 + py * o1), (sx + dx * f2 + px * o2, sy + dy * f2 + py * o2)


def _tremor_series(n: int, amp: float, rng: random.Random, smooth: float = 0.6) -> List[float]:
    out, v = [], 0.0
    for _ in range(n):
        v = smooth * v + (1.0 - smooth) * rng.gauss(0, amp)
        out.append(v)
    return out


def _build_trajectory(start: Vec, end: Vec, duration_ms: float, easing: Callable[[float], float],
                      curvature: float, tremor_px: float, bias: float, sample_hz: float, rng: random.Random) -> Tuple[List[Vec], List[float]]:
    n = max(6, int(round(duration_ms / (1000.0 / sample_hz))))
    c1, c2 = _control_points(start, end, curvature, bias, rng)

    times = sorted([duration_ms * i / (n - 1) + (rng.gauss(0, duration_ms * 0.02) if 0 < i < n - 1 else 0) for i in range(n)])

    tang = _tremor_series(n, tremor_px, rng)
    perp = _tremor_series(n, tremor_px, rng)
    dirx, diry = _unit(end[0] - start[0], end[1] - start[1])
    nx, ny = -diry, dirx

    points = []
    for i in range(n):
        s = easing(times[i] / duration_ms)
        bx, by = _cubic_bezier(start, c1, c2, end, s)
        env = math.sin(math.pi * (i / (n - 1)))
        points.append((bx + (tang[i] * dirx + perp[i] * nx) * env, by + (tang[i] * diry + perp[i] * ny) * env))

    return points, [0.0] + [times[i] - times[i - 1] for i in range(1, n)]


def _lognormal(median: float, sigma: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, random.lognormvariate(math.log(median), sigma)))


# ---------------------------------------------------------------------------
# Human cadence: the PAUSE between actions. Calibrated to REAL recorded idle_time_ms
# (across 3 human sessions: p10 ~0.15s, median ~0.40s, fat tail to ~2.5s). Bots act at
# constant intervals; humans vary. note_action() timestamps each touch-up;
# human_idle_gap() pads the gap before the NEXT action up to a sampled human idle, but
# COUNTS the perceive/decide time already elapsed since the last touch -- so a slow
# decision (already a human-length pause) adds nothing, and only a fast deterministic
# tap gets padded. Latency-safe by construction. Gated by MG_HUMAN_CADENCE (default on).
# ---------------------------------------------------------------------------
_last_action_ts = {}


def note_action(serial):
    """Record when the phone's last touch finished (called at each touch-up)."""
    _last_action_ts[serial] = time.time()


def human_idle_gap(serial):
    """Sleep only enough to make the gap since the last touch match a human idle."""
    if os.environ.get("MG_HUMAN_CADENCE", "1") in ("0", "", "false", "off"):
        return
    last = _last_action_ts.get(serial)
    if last is None:
        return   # first action of the task -> no prior gap to shape
    target = _lognormal(0.40, 0.85, 0.12, 2.5)                     # seconds, fit to real idle data
    persona = _PERSONAS.get(serial)
    if persona is not None:
        target *= (0.75 + 0.5 * getattr(persona, "patience", 0.5))  # patient linger, impatient sooner
    elapsed = time.time() - last
    if elapsed < target:
        time.sleep(target - elapsed)


# ---------------------------------------------------------------------------
# Session state: fatigue and last-touch position
# ---------------------------------------------------------------------------

def _fatigue(serial: str, persona: BiometricPersona) -> float:
    """A human slows down and drifts over a long session. Returns a multiplier
    from 1.00 (fresh) up to ~1.25 (tired), reached gradually over ~60 actions
    and scaled by how tire-prone this persona is."""
    n = _ACTION_COUNT.get(serial, 0)
    return 1.0 + persona.fatigue_rate * min(n / 60.0, 1.0) * 0.25


def _bump(serial: str, px: float, py: float):
    _LAST_XY[serial] = (float(px), float(py))
    _ACTION_COUNT[serial] = _ACTION_COUNT.get(serial, 0) + 1


def _travel_delay(serial: str, px: float, py: float, persona: BiometricPersona) -> float:
    """Fitts's law, loosely: the further the thumb has to travel to the next
    target, the longer it takes to get there. Short hop ~30ms, screen-crossing
    reach ~150ms, jittered per persona."""
    lx, ly = _LAST_XY.get(serial) or (px, py)
    dist = math.hypot(px - lx, py - ly)
    w, h = get_device_size(serial)
    diag = math.hypot(w, h)
    idx = math.log2(1.0 + 8.0 * dist / diag)           # 0 (no move) .. ~1 (far)
    base = 0.030 + 0.075 * idx
    return _lognormal(base, 0.30, 0.010, 0.40) * persona.dwell_scale


def _aim_sigma(serial: str, persona: BiometricPersona, target_w_px: Optional[float]) -> float:
    """How far a tap lands from dead-centre. Humans are TIGHTER on small careful
    targets and looser on big ones (you don't aim a fingertip at a huge button).
    Steadier personas (higher fitts_precision) tighten it; fatigue loosens it."""
    base = persona.scatter_px
    if target_w_px:
        w, _ = get_device_size(serial)
        ref = 0.12 * w                                  # a ~12%-wide button = reference
        size_factor = _clamp((target_w_px / ref) ** 0.5, 0.55, 1.9)
        base = base * size_factor / persona.fitts_precision
    base *= _fatigue(serial, persona) ** 0.5
    return _clamp(base, 0.8, 14.0)


def _handedness_lean(serial: str, persona: BiometricPersona, py: float) -> float:
    """A right thumb reaching the TOP of the screen pulls a hair left; the bottom
    pulls right (and mirror for a left thumb). A tiny, CONSISTENT spatial bias --
    the kind of systematic signature a real hand has and a random jitter doesn't."""
    _, h = get_device_size(serial)
    reach = (py / h) - 0.5                              # -0.5 top .. +0.5 bottom
    return -persona.curve_bias * reach * 3.0            # up to ~1.5 px


# ---------------------------------------------------------------------------
# Device Connection and Resolution Management
# ---------------------------------------------------------------------------

def reconnect_device(serial: str):
    """Refresh the uiautomator2 handle after a transient agent drop (e.g. a screenshot
    RemoteDisconnected), keeping the persona and per-device state intact. Returns the new
    handle, or the old one if reconnect fails."""
    try:
        _device_cache[serial] = u2.connect(serial)
    except Exception:
        pass
    return _device_cache.get(serial)


def get_device_for_serial(serial: str):
    if serial not in _device_cache:
        d = u2.connect(serial)
        _device_cache[serial] = d
        _PERSONAS[serial] = generate_random_persona()
        _LAST_XY[serial] = None
        _ACTION_COUNT[serial] = 0
        # Persona is printed by the run loop AFTER any seed/MG_PERSONA_TYPE is applied --
        # printing it here would show a throwaway random one that gets overwritten (was a
        # misleading banner during A/B tests). Keep this line to just "Connected."
        print("[%s] Connected." % serial)
        try:
            d.set_input_ime(False)
            gboard = "com.google.android.inputmethod.latin/com.android.inputmethod.latin.LatinIME"
            d.shell("ime enable %s" % gboard)
            d.shell("ime set %s" % gboard)
        except Exception:
            pass
    return _device_cache[serial]


def get_device_size(serial: str) -> Tuple[int, int]:
    if serial not in _DEVICE_SIZE:
        w, h = _device_cache[serial].window_size()
        _DEVICE_SIZE[serial] = (int(w), int(h))
    return _DEVICE_SIZE[serial]


def norm_to_px(serial: str, nx: float, ny: float) -> Tuple[int, int]:
    w, h = get_device_size(serial)
    return int(round(w * nx / 1000.0)), int(round(h * ny / 1000.0))


# ---------------------------------------------------------------------------
# Contact helper: a finger-down that is never perfectly still
# ---------------------------------------------------------------------------

def _held_contact(serial: str, px: int, py: int, hold_s: float, persona: BiometricPersona):
    """Press, hold, release. A short tap just holds; a long hold gets real
    micro-drift, because a human finger never sits frozen for 700ms."""
    d = _device_cache[serial]
    w, h = get_device_size(serial)
    d.touch.down(px, py)
    if hold_s <= 0.22:
        time.sleep(hold_s)
        d.touch.up(px, py)
        return
    n = max(2, int(hold_s / 0.12))
    amp = 1.2 * persona.tremor_scale
    cx, cy = float(px), float(py)
    for _ in range(n):
        time.sleep(hold_s / n)
        cx += random.gauss(0, amp)
        cy += random.gauss(0, amp)
        d.touch.move(int(_clamp(cx, 0, w - 1)), int(_clamp(cy, 0, h - 1)))
    d.touch.up(int(_clamp(cx, 0, w - 1)), int(_clamp(cy, 0, h - 1)))


# ---------------------------------------------------------------------------
# Persona-Driven Physical Interaction Primitives
# ---------------------------------------------------------------------------

def _use_maatouch() -> bool:
    """True unless the maatouch module failed to load or MG_TOUCH=u2. maatouch is now the
    DEFAULT (on-device timing + real pressure); set MG_TOUCH=u2 to force the old path. Any
    maatouch failure still falls back to u2 automatically, so this is safe as a default."""
    return _maatouch is not None and os.environ.get("MG_TOUCH", "maatouch").lower() != "u2"


def _mt_pressure(persona) -> int:
    # a normal press with slight variation -- constant pressure is a bot tell, and unlike
    # uiautomator2 we can actually vary it now (small extra realism, on 0..255).
    return int(_clamp(random.gauss(0.26, 0.05) * 255, 24, 200))


def _mt_tap(serial: str, px: int, py: int, hold_s: float, persona: BiometricPersona) -> bool:
    """Try to land the tap via maatouch. Returns False (no-op) unless MG_TOUCH=maatouch AND
    a live daemon is connected -- so the u2 path runs unchanged by default / on any failure."""
    if not _use_maatouch():
        return False
    mt = _maatouch.get(serial)
    if mt is None:
        return False
    return mt.tap(px, py, int(hold_s * 1000), _mt_pressure(persona))


def _mt_gesture(serial: str, pts, dts, start_dwell: float, end_dwell: float,
                persona: BiometricPersona) -> bool:
    """Play the SAME computed trajectory (pts + per-point ms in dts) via maatouch, with the
    waits happening on-device (no host sleep jitter). Returns False unless MG_TOUCH=maatouch
    and a live daemon is connected, so the u2 move-loop runs unchanged by default."""
    if not _use_maatouch():
        return False
    mt = _maatouch.get(serial)
    if mt is None:
        return False
    w, h = get_device_size(serial)
    cx = lambda v: int(_clamp(round(v), 0, w - 1))
    cy = lambda v: int(_clamp(round(v), 0, h - 1))
    downxy = (cx(pts[0][0]), cy(pts[0][1]))
    moves = [(cx(pts[i][0]), cy(pts[i][1]), dts[i]) for i in range(1, len(pts))]
    return mt.swipe(downxy, moves, int(start_dwell * 1000), int(end_dwell * 1000),
                    _mt_pressure(persona))


def execute_tap(serial: str, nx: float, ny: float, target_w_px: Optional[float] = None, target_h_px: Optional[float] = None):
    """Tap a 0-1000 point like a human: decide, move the thumb there (time scales
    with distance), land with size-aware scatter + a consistent hand lean, and
    hold the contact ~55ms. Pass the target's pixel size when known for
    Fitts-accurate aim; leave it None for a generic tap."""
    d = _device_cache[serial]
    w, h = get_device_size(serial)
    persona = _PERSONAS[serial]

    base_x, base_y = norm_to_px(serial, nx, ny)
    sigma = _aim_sigma(serial, persona, target_w_px)
    lean = _handedness_lean(serial, persona, base_y)

    px = int(_clamp(base_x + lean + random.gauss(0, sigma), 0, w - 1))
    py = int(_clamp(base_y + random.gauss(0, sigma), 0, h - 1))

    fatg = _fatigue(serial, persona)
    time.sleep(_lognormal(persona.reaction_ms / 1000.0, 0.35, 0.09, 0.90) * fatg)  # decide
    time.sleep(_travel_delay(serial, px, py, persona))                             # reach

    hold = _lognormal(*_TAP_HOLD) * persona.dwell_scale * (fatg ** 0.5)
    if not _mt_tap(serial, px, py, hold, persona):     # maatouch if MG_TOUCH=maatouch, else:
        _held_contact(serial, px, py, hold, persona)   # <-- unchanged u2 path (the default)
    _bump(serial, px, py)
    note_action(serial)
    print("[%s] [tap] %s at (%d,%d) held %dms" % (serial, persona.name, px, py, int(hold * 1000)))


def long_press(serial: str, nx: float, ny: float, target_w_px: Optional[float] = None):
    """A deliberate long-press (~700ms with micro-drift) -- the recorder saw 45
    of these; now the Hands can make them. Wire it in where a long-press is the
    right gesture (context menus, drag-to-reorder handles, map pins)."""
    d = _device_cache[serial]
    w, h = get_device_size(serial)
    persona = _PERSONAS[serial]

    base_x, base_y = norm_to_px(serial, nx, ny)
    sigma = _aim_sigma(serial, persona, target_w_px)
    lean = _handedness_lean(serial, persona, base_y)
    px = int(_clamp(base_x + lean + random.gauss(0, sigma), 0, w - 1))
    py = int(_clamp(base_y + random.gauss(0, sigma), 0, h - 1))

    fatg = _fatigue(serial, persona)
    time.sleep(_lognormal(persona.reaction_ms / 1000.0, 0.35, 0.09, 0.90) * fatg)
    time.sleep(_travel_delay(serial, px, py, persona))

    hold = _lognormal(*_LONGPRESS) * persona.dwell_scale
    if not _mt_tap(serial, px, py, hold, persona):     # maatouch long hold, else u2 (with drift)
        _held_contact(serial, px, py, hold, persona)
    _bump(serial, px, py)
    print("[%s] [long-press] %s at (%d,%d) held %dms" % (serial, persona.name, px, py, int(hold * 1000)))


def _gesture(serial: str, start: Vec, end: Vec, kind: str,
             bow_frac: Optional[float] = None, dur: Optional[float] = None,
             steps: int = 14, allow_overshoot: bool = False):
    """One swipe of a given KIND, with that kind's real physics:
       flick  -> back-loaded ballistic, fast, straight  (throws content)
       scroll -> front-loaded, decelerates and settles   (content stops clean)
       drag   -> symmetric min-jerk, precise             (deliberate move)
    Duration and straightness are sampled from the recorder's measured spread,
    then bent by the persona (vigour, tremor, handedness, fatigue)."""
    d = _device_cache[serial]
    w, h = get_device_size(serial)
    persona = _PERSONAS[serial]
    rng = random.Random()
    fatg = _fatigue(serial, persona)

    start = (float(start[0]), float(start[1]))
    end = (float(end[0]), float(end[1]))

    # optional overshoot: push the end a little past the target, to be nudged
    # back after -- a human scroll that goes one flick too far.
    corrected = None
    if allow_overshoot and kind in ("flick", "scroll") and rng.random() < persona.overshoot_rate:
        ox = rng.uniform(0.04, 0.09)
        over = (end[0] + (end[0] - start[0]) * ox, end[1] + (end[1] - start[1]) * ox)
        corrected, end = end, over

    if kind == "flick":
        if dur is None:
            dur = _lognormal(*_FLICK_DUR) / persona.flick_vigor
        easing = lambda tau, p=rng.uniform(1.20, 1.45): _ease_in(tau, p)
        bow = bow_frac if bow_frac is not None else rng.uniform(0.05, 0.18)
        start_dwell, end_dwell = rng.uniform(0.008, 0.025) * persona.dwell_scale, 0.0
    elif kind == "drag":
        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        if dur is None:
            dur = _clamp(0.18 + (dist / h) * 0.40, 0.16, 0.60) * rng.uniform(0.9, 1.15)
        easing = _min_jerk
        bow = bow_frac if bow_frac is not None else rng.uniform(0.03, 0.08)
        start_dwell = rng.uniform(0.020, 0.060) * persona.dwell_scale
        end_dwell = rng.uniform(0.030, 0.080) * persona.dwell_scale
    else:  # scroll
        if dur is None:
            dur = _lognormal(*_SCROLL_DUR)
        easing = lambda tau, p=rng.uniform(1.35, 1.55): _ease_out(tau, p)
        # scroll straightness has a fatter low tail (0.872) -> occasionally curvier
        bow = bow_frac if bow_frac is not None else _lognormal(0.10, 0.55, 0.05, 0.32)
        start_dwell = rng.uniform(0.015, 0.045) * persona.dwell_scale
        end_dwell = rng.uniform(0.025, 0.070) * persona.dwell_scale

    dur *= fatg                                          # tired = slower
    duration_ms = dur * 1000.0
    pts, dts = _build_trajectory(start, end, duration_ms, easing, bow,
                                 0.8 * persona.tremor_scale, persona.curve_bias,
                                 max(60.0, float(steps) / max(0.01, dur)), rng)

    # Inject the trajectory. Default (u2) is the exact loop below; with MG_TOUCH=maatouch
    # the SAME pts/dts are streamed to the daemon with on-device timing instead.
    if not _mt_gesture(serial, pts, dts, start_dwell, end_dwell, persona):
        d.touch.down(int(_clamp(round(pts[0][0]), 0, w - 1)), int(_clamp(round(pts[0][1]), 0, h - 1)))
        if start_dwell > 0:
            time.sleep(start_dwell)
        for i in range(1, len(pts)):
            d.touch.move(int(_clamp(round(pts[i][0]), 0, w - 1)), int(_clamp(round(pts[i][1]), 0, h - 1)))
            if (dt := dts[i] / 1000.0) > 0.001:
                time.sleep(dt)
        if end_dwell > 0:
            time.sleep(end_dwell)
        d.touch.up(int(_clamp(round(pts[-1][0]), 0, w - 1)), int(_clamp(round(pts[-1][1]), 0, h - 1)))
    _bump(serial, pts[-1][0], pts[-1][1])
    print("[%s] [%s] %s" % (serial, kind, persona.name))

    # settle the overshoot with a small nudge back toward the true target
    if corrected is not None:
        time.sleep(_lognormal(0.22, 0.30, 0.10, 0.45) * persona.dwell_scale)
        back_start = (end[0], end[1])
        _gesture(serial, back_start, corrected, "scroll",
                 bow_frac=rng.uniform(0.04, 0.10), dur=rng.uniform(0.16, 0.26),
                 steps=8, allow_overshoot=False)
    note_action(serial)


def curved_swipe(serial: str, x0: float, y0: float, x1: float, y1: float,
                 dur: Optional[float] = None, bow_frac: Optional[float] = None,
                 steps: int = 14, is_flick: bool = False):
    """Back-compat wrapper. is_flick=True -> ballistic flick; otherwise a
    controlled scroll. Kept so open_app_organically and older callers work
    unchanged."""
    _gesture(serial, (x0, y0), (x1, y1), "flick" if is_flick else "scroll",
             bow_frac=bow_frac, dur=dur, steps=steps, allow_overshoot=False)


def _swipe_vector(serial, direction, dist_frac, rng):
    """Start/end points for a swipe. A real thumb swipe is a slightly-DIAGONAL,
    curved stroke: the finger drifts sideways as it travels (an arc around the thumb
    joint), so it starts and ends at DIFFERENT x (for a vertical swipe) -- never a
    dead-straight centre-to-centre vertical line, which is a bot tell. The start
    point is jittered per gesture; the sideways drift is a fraction of the travel
    distance, leaning with the persona's handedness, and sampled fresh every call.
    The intended direction stays dominant (drift <= ~30% of travel) so a 'down'
    swipe still scrolls down rather than reading as a horizontal page-swipe."""
    w, h = get_device_size(serial)
    persona = _PERSONAS[serial]
    lean = getattr(persona, "curve_bias", 0.0)   # handedness: >0 right thumb, <0 left

    def _cx(x):
        return max(int(w * 0.05), min(int(w * 0.95), int(x)))

    def _cy(y):
        return max(int(h * 0.08), min(int(h * 0.92), int(y)))

    cx = int(w * (0.5 + rng.uniform(-0.09, 0.09) + 0.02 * lean))
    cy = int(h * (0.5 + rng.uniform(-0.07, 0.07)))
    half_v = int(h * dist_frac / 2.0)
    half_h = int(w * dist_frac / 2.0)

    if direction in ("down", "up"):
        travel = half_v * 2
        mag = rng.uniform(0.08, 0.30)                                  # sideways : vertical
        sign = 1 if rng.random() < (0.5 + 0.35 * lean) else -1          # handedness leans the drift
        drift = int(travel * mag * sign)
        sx, ex = _cx(cx - drift // 2), _cx(cx + drift // 2)             # start x != end x -> diagonal
        if direction == "down":
            return (sx, _cy(cy + half_v)), (ex, _cy(cy - half_v))
        return (sx, _cy(cy - half_v)), (ex, _cy(cy + half_v))

    # horizontal swipes get a small vertical drift for the same reason
    travel = half_h * 2
    mag = rng.uniform(0.06, 0.22)
    sign = 1 if rng.random() < 0.5 else -1
    drift = int(travel * mag * sign)
    sy, ey = _cy(cy - drift // 2), _cy(cy + drift // 2)
    if direction == "left":
        return (_cx(cx + half_h), sy), (_cx(cx - half_h), ey)
    return (_cx(cx - half_h), sy), (_cx(cx + half_h), ey)


def execute_scroll(serial: str, direction: str = "down"):
    """One scroll, but the STYLE varies every call and is biased by the persona:
       - fast fling  : ballistic, covers a lot of screen, may overshoot
       - normal      : a comfortable controlled swipe
       - slow careful: short, front-loaded, precise
    After a hard fling a human often adds a short slow corrective scroll to settle
    on the target ("flip fast, then slow") -- so we sometimes append that. Distance,
    duration, start point and left/right drift are all sampled fresh each time."""
    persona = _PERSONAS[serial]
    rng = random.Random()
    fb = getattr(persona, "fling_bias", 0.5)
    roll = rng.random()

    if roll < 0.25 + 0.50 * fb:
        s, e = _swipe_vector(serial, direction, rng.uniform(0.62, 0.88), rng)
        _gesture(serial, s, e, "flick", allow_overshoot=True)
        style = "fling"
    elif roll > 1.0 - (0.18 + 0.42 * (1.0 - fb)):
        s, e = _swipe_vector(serial, direction, rng.uniform(0.24, 0.40), rng)
        _gesture(serial, s, e, "scroll", dur=rng.uniform(0.42, 0.66), allow_overshoot=False)
        style = "slow"
    else:
        s, e = _swipe_vector(serial, direction, rng.uniform(0.44, 0.62), rng)
        _gesture(serial, s, e, "scroll", allow_overshoot=True)
        style = "scroll"

    # human "flip fast then settle": short slow corrective scroll after a fling
    if style == "fling" and rng.random() < 0.35 + 0.30 * fb:
        time.sleep(_lognormal(0.35, 0.35, 0.18, 0.70) * persona.dwell_scale)
        s2, e2 = _swipe_vector(serial, direction, rng.uniform(0.10, 0.20), rng)
        _gesture(serial, s2, e2, "scroll", dur=rng.uniform(0.40, 0.60), allow_overshoot=False)
        style = "fling+settle"

    print("[%s] [scroll:%s] %s" % (serial, style, direction))


def execute_flick(serial: str, direction: str = "down"):
    """A hard, fast fling for blasting through a long list -- distance and speed
    still sampled fresh each time so repeated flings don't look identical."""
    rng = random.Random()
    s, e = _swipe_vector(serial, direction, rng.uniform(0.66, 0.90), rng)
    _gesture(serial, s, e, "flick", allow_overshoot=False)
    print("[%s] [flick] %s" % (serial, direction))


def human_thinking_motion(serial: str, is_busy):
    """Fill the model's 'thinking' wait with gentle, SELF-RESTORING idle motion so
    the phone doesn't sit frozen while the grounder decides -- the way a person
    nudges/reads the screen before acting. Safe by construction:
      - only SWIPES (a small scroll down, then back up by the SAME amount) -- swipes
        never register as a tap, so nothing gets accidentally clicked;
      - each nudge returns to the origin, so the screen the model saw is restored
        and the decision's coordinates stay valid (no re-perceive needed);
      - persona-scaled: curious/restless personas fidget more, careful ones hold
        stiller.
    `is_busy` is a zero-arg callable that returns True while the model is still
    running (e.g. thread.is_alive).
    LATENCY-SAFE: it first waits a short threshold (MG_THINK_AFTER_MS, default 1200ms)
    checking constantly, and RETURNS IMMEDIATELY if the decision comes back fast --
    so a quick decision gets ZERO added motion/latency; only a genuinely long wait
    (e.g. a cold worker) is filled with nudges."""
    persona = _PERSONAS[serial]
    rng = random.Random()
    w, h = get_device_size(serial)

    # Don't add ANY motion to a fast decision: wait up to the threshold, checking
    # often, and bail the moment the model returns.
    try:
        threshold = float(os.environ.get("MG_THINK_AFTER_MS", "1200")) / 1000.0
    except Exception:
        threshold = 1.2
    waited = 0.0
    while is_busy() and waited < threshold:
        time.sleep(0.08)
        waited += 0.08
    if not is_busy():
        return   # fast decision -> no fidget, no added latency

    # The wait is genuinely long. A real person mostly just READS / holds still here
    # -- they do NOT scroll every screen. So fidget only RARELY: decide ONCE, with a
    # low probability (a little higher for curious personas), whether to do a single
    # small self-restoring nudge; otherwise wait quietly (no scrolling) until the
    # model returns. Tunable via MG_THINK_FIDGET_P (default 0.18 = ~1 in 6 screens).
    try:
        base_p = float(os.environ.get("MG_THINK_FIDGET_P", "0.18"))
    except Exception:
        base_p = 0.18
    p_fidget = base_p * (0.6 + getattr(persona, "curiosity", 0.4))
    if rng.random() < p_fidget:
        amp = int(h * rng.uniform(0.04, 0.09))                 # one small nudge
        cx = int(w * (0.5 + rng.uniform(-0.05, 0.05)))
        cy = int(h * 0.5)
        _gesture(serial, (cx, cy), (cx, cy - amp), "scroll",
                 dur=rng.uniform(0.35, 0.55), allow_overshoot=False)   # content nudges up
        if is_busy():
            time.sleep(_lognormal(0.4, 0.4, 0.20, 0.90))
        _gesture(serial, (cx, cy - amp), (cx, cy), "scroll",
                 dur=rng.uniform(0.35, 0.55), allow_overshoot=False)   # restore to origin
    # otherwise (and after any single nudge) just wait quietly -- no scrolling
    while is_busy():
        time.sleep(0.15)


def human_open_drawer(serial: str):
    """Open the app drawer with the SAME curved, per-persona swipe engine as execute_scroll
    (the motion tuned to the recorded human data) -- a real Bezier ARC via _gesture, styled
    by the persona: a fling-happy persona mostly throws a fast ballistic flick, a careful or
    shaky one more often does a slower, curvier swipe. Distance, curvature, diagonal lean and
    speed are all sampled fresh, so different personas open the drawer differently and no two
    opens look alike. Direction 'down' here = _swipe_vector sends the FINGER UP the screen
    (from ~0.9h to ~0.1h) with a long throw so it carries velocity to fling the drawer. Biased
    toward the fast flick so it reliably opens; if a given open doesn't fling it, the
    search/scroll-hunt + package-launch recovery still gets the app open (unchanged)."""
    persona = _PERSONAS[serial]
    rng = random.Random()
    fb = getattr(persona, "fling_bias", 0.5)
    roll = rng.random()

    # Style weighted by the persona -- same shape as execute_scroll, biased a bit more toward
    # the ballistic flick (a drawer needs throw to open). 'down' => finger travels UP.
    if roll < 0.45 + 0.35 * fb:
        s, e = _swipe_vector(serial, "down", rng.uniform(0.70, 0.86), rng)
        _gesture(serial, s, e, "flick", allow_overshoot=False)                    # fast ballistic throw
        style = "flick"
    elif roll > 1.0 - (0.20 + 0.30 * (1.0 - fb)):
        s, e = _swipe_vector(serial, "down", rng.uniform(0.64, 0.80), rng)
        _gesture(serial, s, e, "scroll", dur=rng.uniform(0.26, 0.42), allow_overshoot=False)  # slower, curvier
        style = "slow-swipe"
    else:
        s, e = _swipe_vector(serial, "down", rng.uniform(0.66, 0.82), rng)
        _gesture(serial, s, e, "flick", dur=rng.uniform(0.11, 0.18), allow_overshoot=False)   # moderate flick
        style = "swipe"

    time.sleep(0.6 * persona.dwell_scale)
    print("[%s] [drawer-open:%s]" % (serial, style))


def _try_drawer_search(serial: str, app_name: str) -> bool:
    """UNIVERSAL app-drawer search (optional accelerator). If a search field or a
    magnifying-glass is visible, tap it, type the app name the HUMAN way, then tap the
    matching RESULT -- far more reliable than scroll-hunting a drawer full of duplicate
    labels, and still human (people search for apps). Returns True if it opened the app,
    else False so the caller falls back to the scroll-hunt. Cleans up after itself
    (dismisses the keyboard) if it typed but found nothing, so a fallback scroll-hunt
    starts clean. App-agnostic: keys off generic 'search' field/desc patterns, never a
    specific launcher; the scroll-hunt + package-launch remain the universal fallback."""
    d = _device_cache[serial]
    w, h = get_device_size(serial)
    persona = _PERSONAS[serial]

    # Find a search entry universally: an EditText, or a clickable element whose
    # content-desc/text says "search" (a magnifying glass usually has desc 'Search').
    search_el = None
    for sel in (dict(className="android.widget.EditText"),
                dict(descriptionMatches="(?i).*search.*"),
                dict(textMatches="(?i)search.*")):
        try:
            cand = d(**sel)
            if cand.exists:
                search_el = cand
                break
        except Exception:
            continue
    if search_el is None:
        return False   # no search here -> fall back to scroll-hunt (no state changed)

    try:
        b = search_el.info["bounds"]
    except Exception:
        return False
    tw = b["right"] - b["left"]
    execute_tap(serial, int((b["left"] + b["right"]) / 2 / w * 1000),
                int((b["top"] + b["bottom"]) / 2 / h * 1000), target_w_px=tw)
    time.sleep(0.8 * persona.dwell_scale)

    # Type the app name the HUMAN way (engine v5 taps keys; its own inject is only the
    # last resort inside that function). If typing errors, bail to the scroll-hunt.
    try:
        execute_human_type(serial, app_name)
    except Exception:
        try:
            d.press("back"); time.sleep(0.4)
        except Exception:
            pass
        return False
    time.sleep(1.5 * persona.dwell_scale)   # let the results filter

    # Tap the matching RESULT: a match whose top is BELOW the search box bottom. (The
    # text we just typed also lives inside the search box and matches app_name, so any
    # match at the box itself is the echo, not a result -- skip it.)
    sb_bottom = b["bottom"]
    for _ in range(3):
        target = None
        try:
            for el in d(textMatches="(?i).*%s.*" % re.escape(app_name)):
                try:
                    bb = el.info["bounds"]
                except Exception:
                    continue
                if bb["top"] >= sb_bottom - 5:
                    target = bb
                    break
        except Exception:
            target = None
        if target:
            tw2 = target["right"] - target["left"]
            execute_tap(serial, int((target["left"] + target["right"]) / 2 / w * 1000),
                        int((target["top"] + target["bottom"]) / 2 / h * 1000), target_w_px=tw2)
            time.sleep(3.5)
            return True
        time.sleep(0.8)

    # Typed but no result -> dismiss the keyboard so a fallback scroll-hunt starts clean.
    try:
        d.press("back"); time.sleep(0.4)
    except Exception:
        pass
    return False


def open_app_organically(serial: str, app_name: str) -> bool:
    d = _device_cache[serial]
    w, h = get_device_size(serial)
    persona = _PERSONAS[serial]

    try:
        if app_name.replace(" ", "").lower() in d.app_current().get('package', '').lower():
            return True
    except Exception:
        pass

    d.press("home")
    time.sleep(1.5 * persona.dwell_scale)

    if (app_element := d(textMatches="(?i).*%s.*" % re.escape(app_name))).exists:
        b = app_element.info['bounds']
        tw = b['right'] - b['left']
        execute_tap(serial, int((b['left'] + b['right']) / 2 / w * 1000),
                    int((b['top'] + b['bottom']) / 2 / h * 1000), target_w_px=tw)
        time.sleep(3.5)
        return True

    human_open_drawer(serial)
    time.sleep(2.0 * persona.dwell_scale)

    search_on = os.environ.get("MG_DRAWER_SEARCH", "1") not in ("0", "", "false", "off")

    def _do_search():
        if not search_on:
            return False
        try:
            if _try_drawer_search(serial, app_name):
                print("[%s] [drawer-search] opened '%s' via search" % (serial, app_name))
                return True
        except Exception as _e:
            print("[%s] [drawer-search] error (%s)" % (serial, _e))
        return False

    def _scroll_hunt():
        for _ in range(12):
            if (app_element := d(textMatches="(?i).*%s.*" % re.escape(app_name))).exists:
                b = app_element.info['bounds']
                tw = b['right'] - b['left']
                execute_tap(serial, int((b['left'] + b['right']) / 2 / w * 1000),
                            int((b['top'] + b['bottom']) / 2 / h * 1000), target_w_px=tw)
                time.sleep(4.0)
                return True
            # Hunt for the app like a human: the swipe VARIES every time (mood/persona)
            # -- quick flick / slow careful nudge / moderate scroll -- and it's diagonal
            # (via _swipe_vector), distance/speed/pause sampled fresh. Kept controlled so
            # it rarely sails past the app (package-launch is the final fallback).
            fb = getattr(persona, "fling_bias", 0.5)
            roll = random.random()
            if roll < 0.12 + 0.28 * fb:
                s, e = _swipe_vector(serial, "down", random.uniform(0.40, 0.54), random)
                _gesture(serial, s, e, "flick", dur=random.uniform(0.16, 0.28), allow_overshoot=False)
            elif roll > 1.0 - (0.22 + 0.28 * (1.0 - fb)):
                s, e = _swipe_vector(serial, "down", random.uniform(0.20, 0.32), random)
                _gesture(serial, s, e, "scroll", dur=random.uniform(0.50, 0.78), allow_overshoot=False)
            else:
                s, e = _swipe_vector(serial, "down", random.uniform(0.32, 0.48), random)
                _gesture(serial, s, e, "scroll", dur=random.uniform(0.42, 0.62), allow_overshoot=False)
            time.sleep(random.uniform(1.1, 2.3) * persona.dwell_scale)
        return False

    # BROWSE FIRST, LIKE A PERSON. A human opening the drawer looks at what is there and
    # gives it a few small scrolls to spot the icon BEFORE resorting to the search box --
    # they do not jump straight to search. So we scroll-and-look for a handful of screens
    # first; only if the icon has not shown up do we fall back to the drawer search, and
    # finally to a package launch (the reliability net). A more curious persona browses a
    # bit longer before giving up and searching. MG_DRAWER_SEARCH=0 keeps scrolling only.
    def _browse_and_look(max_scrolls):
        # check the CURRENT screen first (the icon may already be visible), then scroll a
        # little, look again -- the gentle "scroll a bit and see" rhythm, not a 12x hunt.
        for i in range(max_scrolls):
            if (app_element := d(textMatches="(?i).*%s.*" % re.escape(app_name))).exists:
                b = app_element.info['bounds']
                tw = b['right'] - b['left']
                execute_tap(serial, int((b['left'] + b['right']) / 2 / w * 1000),
                            int((b['top'] + b['bottom']) / 2 / h * 1000), target_w_px=tw)
                time.sleep(4.0)
                return True
            if i == max_scrolls - 1:
                break            # looked on the last screen; do not scroll past uselessly
            # one small, human scroll (short throw), then pause and look again
            sv, ev = _swipe_vector(serial, "down", random.uniform(0.28, 0.42), random)
            _gesture(serial, sv, ev, "scroll", dur=random.uniform(0.44, 0.66), allow_overshoot=False)
            time.sleep(random.uniform(0.9, 1.8) * persona.dwell_scale)
        return False

    # how many screens to browse before falling back to search: a few, a touch more if curious
    browse_screens = 3 + int(round(2 * getattr(persona, "curiosity", 0.3)))
    print("[%s] [open] browsing the drawer (scroll a bit and look) first" % serial)
    if _browse_and_look(browse_screens):
        return True
    # not spotted by browsing -> now a person uses the search box (if there is one)
    if _do_search():
        return True
    # still not open -> a longer scroll-hunt as the last visual attempt before package launch
    if _scroll_hunt():
        return True
    return False


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Screen-Aware Keyboard Input
# The typer READS the live keyboard from the accessibility tree and taps each
# key's real centre. No more guessing a fixed grid (which landed 'p' on the
# voice icon). The persona still drives typos, rhythm, bursts, and fatigue.
# ---------------------------------------------------------------------------

_QWERTY = {
    'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
    'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz',
    't': 'rfgy', 'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
}


# Some keyboards label a symbol key by NAME in its content-desc (e.g. "Dollar sign") with
# an empty text glyph, so it isn't seen as a single-character key. Map those names to the
# real character so the key can be TAPPED by hand instead of injected. Lowercase keys.
_SYM_NAMES = {
    "dollar sign": "$", "dollar": "$",
    "percent sign": "%", "percent": "%",
    "at sign": "@", "at": "@",
    "ampersand": "&",
    "asterisk": "*", "star": "*",
    "number sign": "#", "pound sign": "#", "hash": "#", "hashtag": "#", "hash key": "#",
    "exclamation mark": "!", "exclamation point": "!", "exclamation": "!",
    "question mark": "?",
    "hyphen": "-", "minus sign": "-", "minus": "-", "dash": "-",
    "plus sign": "+", "plus": "+",
    "equals sign": "=", "equal sign": "=", "equals": "=",
    "slash": "/", "forward slash": "/",
    "backslash": "\\", "back slash": "\\",
    "left parenthesis": "(", "open parenthesis": "(",
    "right parenthesis": ")", "close parenthesis": ")",
    "colon": ":", "semicolon": ";",
    "apostrophe": "'", "single quote": "'", "single quotation mark": "'",
    "quotation mark": '"', "double quote": '"', "quote": '"',
    "comma": ",", "period": ".", "full stop": ".", "dot": ".",
    "underscore": "_",
    "tilde": "~",
    "grave accent": "`", "backtick": "`",
    "caret": "^", "circumflex": "^",
    "vertical bar": "|", "pipe": "|",
    "less-than sign": "<", "less than sign": "<",
    "greater-than sign": ">", "greater than sign": ">",
    "left curly bracket": "{", "left brace": "{", "open brace": "{",
    "right curly bracket": "}", "right brace": "}", "close brace": "}",
    "left square bracket": "[", "open bracket": "[",
    "right square bracket": "]", "close bracket": "]",
}


def _kbd_full(serial: str):
    """Read EVERY key currently on the keyboard from the tree:
       chars -> {char: (x,y)}  single-character keys. Digits/symbols report their
                glyph in `text`; letters report it in `content-desc` -- we read both.
       ctrl  -> {'SYM','ABC','SHIFT','BKSP'} page-toggle and control keys.
    Uses a generous keyboard region (lower ~60% of the screen) so the high number
    row is never dropped (the old bottom-half filter was cutting it)."""
    d = _device_cache[serial]
    _, h = get_device_size(serial)
    try:
        els = parse_elements(d.dump_hierarchy())
    except Exception:
        return {}, {}
    chars, ctrl = {}, {}
    for e in els:
        if e["cy"] < h * 0.40:
            continue
        t, dsc = e["text"].strip(), e["desc"].strip()
        low = (dsc or t).lower()
        # single-character key (digit/symbol via text, letter via desc)
        matched_single = False
        for lab in (t, dsc):
            if len(lab) == 1:
                chars.setdefault(lab.lower(), (e["cx"], e["cy"]))
                matched_single = True
                break
        # symbol key labelled by NAME (e.g. desc "Dollar sign", empty text) -> register it
        # under the real glyph so it can be tapped, not injected.
        if not matched_single and low in _SYM_NAMES:
            chars.setdefault(_SYM_NAMES[low], (e["cx"], e["cy"]))
        # control / page-toggle keys. Check "more symbols" (=\<, symbols page 1 -> page 2)
        # BEFORE the generic ?123/symbol match, since its label "More symbols" contains
        # "symbol" and would otherwise be misread as the ?123 key.
        if "more symbol" in low or ("=" in t and "<" in t):
            ctrl.setdefault("SYM2", (e["cx"], e["cy"]))
        elif "?123" in low or low in ("123", "symbol keyboard", "symbols") or "symbol" in low:
            ctrl.setdefault("SYM", (e["cx"], e["cy"]))
        elif low in ("abc", "letters", "alphabet") or "to letter" in low or "to alphabet" in low:
            ctrl.setdefault("ABC", (e["cx"], e["cy"]))
        elif "shift" in low or "caps" in low or "capital" in low:
            ctrl.setdefault("SHIFT", (e["cx"], e["cy"]))
        elif "delete" in low or "backspace" in low:
            ctrl.setdefault("BKSP", (e["cx"], e["cy"]))
        elif low in ("search", "go", "enter", "return", "send", "next", "done", "submit",
                     "search key", "enter key", "go key", "done key", "return key"):
            ctrl.setdefault("ENTER", (e["cx"], e["cy"]))   # the bottom-right action key
        elif low == "space" or "spacebar" in low or low.endswith(" space"):
            chars.setdefault(" ", (e["cx"], e["cy"]))
    return chars, ctrl


def _read_keyboard(serial: str):
    """Dump the screen and locate every visible key by its real position."""
    d = _device_cache[serial]
    _, h = get_device_size(serial)
    try:
        xml = d.dump_hierarchy()
    except Exception:
        return {}, {}
    return keyboard_keys(parse_elements(xml), h)


def _find_key_by_text(serial: str, ch: str):
    """Find a keyboard key by its exact `text` attribute and return
    (cx, cy, w, h) or None. This Gboard labels number-row digit keys with
    text='1'..'0'. We pick the SMALLEST box whose text matches (that's the real
    key, not a container) anywhere in the lower part of the screen -- no tight
    size caps, since those were guessed and may have been excluding the key."""
    d = _device_cache[serial]
    w, h = get_device_size(serial)
    best = None
    try:
        xml = d.dump_hierarchy()
    except Exception:
        return None
    for e in parse_elements(xml):
        if e["cy"] < h * 0.30:                 # anywhere in the lower ~2/3
            continue
        if e["text"].strip() != ch:
            continue
        if e["w"] > w * 0.5 or e["h"] > h * 0.25:   # ignore huge containers only
            continue
        area = e["w"] * e["h"]
        if best is None or area < best[4]:
            best = (e["cx"], e["cy"], e["w"], e["h"], area)
    return best[:4] if best else None


def _key_tap(serial: str, cx: float, cy: float, is_control: bool = False):
    """Tap a real key centre with a little persona scatter (small, so it stays on the key)
    and a quick keystroke hold. When MG_TOUCH=maatouch this goes through maatouch too --
    so EVERY keystroke, including the fat-finger wrong-key taps and the backspace that
    corrects them, carries on-device timing + pressure -- else the u2 tap; falls back to
    u2 on any maatouch failure. The key-selection / typo / verify logic is unchanged; only
    the final press is swapped."""
    d = _device_cache[serial]
    w, h = get_device_size(serial)
    persona = _PERSONAS[serial]
    px = int(_clamp(cx + random.gauss(0, min(persona.scatter_px, 4.0)), 0, w - 1))
    py = int(_clamp(cy + random.gauss(0, min(persona.scatter_px, 4.0)), 0, h - 1))
    dwell = (_lognormal(0.040, 0.30, 0.020, 0.080) if is_control
             else _lognormal(0.052, 0.38, 0.020, 0.110)) * persona.dwell_scale
    if not _mt_tap(serial, px, py, dwell, persona):
        d.touch.down(px, py)
        time.sleep(dwell)
        d.touch.up(px, py)


# Bullet/mask glyphs a masked field shows instead of the real characters. Some apps
# (e.g. Instagram) DON'T set the a11y password flag but still render dots, so we detect
# "visually masked" and treat those as password fields too.
_MASK_CHARS = set("\u2022\u25cf\u2219\u00b7\u2024\u2027\u26ab\u2b24\u25aa\u25cb*")

def _looks_masked(s: str) -> bool:
    """True if the field text is mostly bullet/mask glyphs (a masked password field). The
    threshold is 60% so a briefly-revealed last char (e.g. '.........@') still counts."""
    if not s:
        return False
    masked = sum(1 for c in s if c in _MASK_CHARS)
    return masked >= max(1, int(len(s) * 0.6))


def _read_focused_field(serial):
    """Return (text_currently_in_the_focused_field, is_password), or (None, False) if it
    can't be read. is_password is True if the a11y flag is set OR the text is visibly
    masked (bullets) -- so a masked field that doesn't set the flag is still verified by
    length / show-password reveal instead of an exact match against dots."""
    d = _device_cache[serial]
    try:
        info = d(focused=True).info
        text = info.get("text") or ""
        return text, (bool(info.get("password", False)) or _looks_masked(text))
    except Exception:
        return None, False


def _clear_focused(serial):
    d = _device_cache[serial]
    try:
        d(focused=True).clear_text()
    except Exception:
        pass


def _type_norm(s):
    return "".join((s or "").split()).lower()


def _needs_sym_page(ch, chars):
    """True if this char should be typed on the ?123 page rather than the letters page.
    DIGITS go here: on this keyboard the number row is a LONG-PRESS layer (a tap on the
    top row gives the letter under the digit -> '1'->'q' etc.), so digits are typed on
    the ?123 page where they're normal keys. Symbols not visible on the letters page also
    go here. Letters, space, and symbols already on the letters page do NOT."""
    if ch == " " or ch.isalpha():
        return False
    if ch.isdigit():
        return True
    return ch.lower() not in chars


def _tap_out_text(serial, text, chars, ctrl, persona, rng, fatg):
    """Tap each character on the on-screen keyboard -- the moat (a bot that sets text
    with no keypress is detectable). Letters + letters-page symbols are tapped directly;
    a RUN of digits/?123-symbols is typed by switching to ?123 ONCE, tapping them all,
    then switching back to ABC -- like a person who flips to the number page, types the
    digits, and flips back (and it avoids the number-row long-press misread)."""
    target_gap = 60.0 / (persona.wpm * 5.0)

    def _rhythm(idx, ch):
        gap = _lognormal(target_gap, 0.40, target_gap * 0.5, target_gap * 1.5)
        if persona.burstiness > 1.0 and (idx % max(2, int(6 / (persona.burstiness - 0.9)))) == 0:
            gap *= rng.uniform(1.2, 1.9)
        if ch == " ":
            gap += rng.uniform(0.10, 0.20)
        if rng.random() < 0.03:
            gap += _lognormal(0.45, 0.30, 0.25, 1.10)
        time.sleep(gap * fatg)

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        low = ch.lower()

        if not _needs_sym_page(ch, chars):
            # letter / space / letters-page symbol -> tap directly on the current page
            if ch == " ":
                if " " in chars:
                    _key_tap(serial, chars[" "][0], chars[" "][1])
                else:
                    _shell_type(serial, " ")
            elif low in chars:
                if ch.isupper() and "SHIFT" in ctrl:
                    _key_tap(serial, ctrl["SHIFT"][0], ctrl["SHIFT"][1], is_control=True)
                    time.sleep(_lognormal(0.12, 0.25, 0.08, 0.25))
                if ch.isalpha() and rng.random() < persona.typo_rate:
                    for wrong in _QWERTY.get(low, ""):
                        if wrong in chars:
                            _key_tap(serial, chars[wrong][0], chars[wrong][1])
                            time.sleep(_lognormal(0.28, 0.35, 0.12, 0.60))
                            if "BKSP" in ctrl:
                                _key_tap(serial, ctrl["BKSP"][0], ctrl["BKSP"][1], is_control=True)
                                time.sleep(_lognormal(0.14, 0.35, 0.06, 0.30))
                            break
                _key_tap(serial, chars[low][0], chars[low][1])
            else:
                _shell_type(serial, ch)   # not found on the letters page -> inject this one char
            _rhythm(i, ch)
            i += 1
            continue

        # ch needs the ?123 page. Switch ONCE, type the whole consecutive run of
        # ?123 chars (digits + symbols), then switch back to ABC.
        if "SYM" not in ctrl:
            print("[%s] [type] no ?123 key found; injecting '%s'" % (serial, ch))
            _shell_type(serial, ch)
            _rhythm(i, ch)
            i += 1
            continue
        # ch needs the ?123 pages. Switch to ?123 ONCE, then type the whole consecutive run
        # of digits/symbols -- hopping to the SECOND symbols page (=\<) for the rarer symbols
        # that aren't on the first (e.g. %), exactly like a person taps ?123 then =\< -- and
        # only injecting a char that's on NEITHER page. Switch back to ABC at the end.
        if "SYM" not in ctrl:
            print("[%s] [type] no ?123 key found; injecting '%s'" % (serial, ch))
            _shell_type(serial, ch)
            _rhythm(i, ch)
            i += 1
            continue
        _key_tap(serial, ctrl["SYM"][0], ctrl["SYM"][1], is_control=True)
        time.sleep(_lognormal(0.34, 0.22, 0.20, 0.55))
        cur_map, cur_ctrl = _kbd_full(serial)
        page = 1
        while i < n and _needs_sym_page(text[i], chars):
            c = text[i]
            cl = c.lower()
            if cl not in cur_map:
                # not on this symbol page -> flip to the OTHER symbol page and re-read
                # (page 1 -> page 2 via =\<; page 2 -> page 1 via ?123) like a real thumb
                hop = cur_ctrl.get("SYM2") if page == 1 else cur_ctrl.get("SYM")
                if hop:
                    _key_tap(serial, hop[0], hop[1], is_control=True)
                    time.sleep(_lognormal(0.30, 0.22, 0.18, 0.50))
                    cur_map, cur_ctrl = _kbd_full(serial)
                    page = 2 if page == 1 else 1
            if cl in cur_map:
                _key_tap(serial, cur_map[cl][0], cur_map[cl][1])
                kind = "digit" if c.isdigit() else "sym"
                pg = "?123" if page == 1 else "=\\<"
                print("[%s] [%s] '%s' tapped via %s at (%d,%d)"
                      % (serial, kind, c, pg, cur_map[cl][0], cur_map[cl][1]))
            else:
                print("[%s] [type] '%s' not on either symbol page; injected" % (serial, c))
                _shell_type(serial, c)
            _rhythm(i, c)
            i += 1
        back = cur_ctrl.get("ABC") or ctrl.get("ABC")
        if back:
            _key_tap(serial, back[0], back[1], is_control=True)
            time.sleep(_lognormal(0.28, 0.22, 0.16, 0.45))
        chars, ctrl = _kbd_full(serial)   # refresh the letters-page map after coming back


def submit_search(serial, target, suggestion_only=False):
    """After a search query is typed, act like a person: if a suggestion/result matching the
    query is on screen, TAP it; otherwise press the keyboard's Search/Enter key to run the
    search (falling back to the IME enter action if that key isn't labelled). UNIVERSAL --
    reads the a11y tree + keyboard, no app-specific coordinates. Returns a short description
    of what it did so the caller can log it.

    suggestion_only=True stops after the suggestion tap and returns None if there is no
    matching row -- for an INTERMEDIATE field (e.g. the 'From' city of a flight search),
    where a city must be chosen from its dropdown but pressing Enter would wrongly submit a
    half-filled form. Default False keeps the original behaviour for every existing caller."""
    d = _device_cache[serial]
    w, h = get_device_size(serial)
    tgt = (target or "").lower().strip()
    words = [x for x in re.split(r"\s+", tgt) if len(x) >= 2]

    # 1) TAP a matching suggestion row the way a person taps the right autocomplete row.
    #    Handles the common Android pattern where the row is a CLICKABLE CONTAINER with no
    #    text of its own, wrapping NON-CLICKABLE TextViews that hold the label -- so we
    #    match the label on ANY node and, when it isn't itself clickable, tap the smallest
    #    clickable container that contains it. Skips the top app-bar/search-box (which just
    #    echoes the query) and the keyboard's autocorrect strip, and never taps a whole-list
    #    container. Suggestion lists are network-backed and may lag typing, so poll a few
    #    times (~2s) before concluding there is nothing to tap. Tap only ONE row.
    def _matches(lab):
        labwords = [x for x in re.split(r"[^a-z0-9]+", lab) if x]
        if tgt and tgt in labwords:
            return True                      # whole-word hit ("pune" in "pune, india")
        if words and all(x in labwords for x in words):
            return True                      # every query word present as a word
        if tgt and " " in tgt and tgt in lab:
            return True                      # multi-word query as a phrase
        return False

    def _pick_suggestion():
        """Return (cx, cy, w, h, label) of the tappable suggestion row, or None."""
        try:
            els = parse_elements(d.dump_hierarchy())
        except Exception:
            return None
        # EXCLUDE THE KEYBOARD. Gboard's autocorrect strip sits directly above the keys and
        # its chips ("Pune" -> Pune / Puneet / punepu) are clickable AND match the query;
        # tapping one just autocorrects the text instead of choosing the app's row. The
        # app's real dropdown is ABOVE the keyboard, so drop everything from a margin above
        # the topmost key downwards (margin scales with screen height, so it is not ~12px on
        # a tall screen). No keyboard visible -> no limit.
        kbd_floor = h
        try:
            _kchars, _kctrl = _kbd_full(serial)
            key_ys = [xy[1] for xy in list(_kchars.values()) + list(_kctrl.values())]
            if key_ys:
                kbd_floor = min(key_ys) - int(h * 0.09)
        except Exception:
            pass

        def _too_big(c):
            # a whole-list / whole-screen container, not one row
            return c["h"] > h * 0.25 or (c["w"] * c["h"]) > (w * h) * 0.35

        def _contains(c, px, py):
            return c["x1"] <= px <= c["x2"] and c["y1"] <= py <= c["y2"]

        clickables = [e for e in els if e.get("clickable") and not _too_big(e)]
        cands = []
        for e in els:                                    # match on ANY node, not only clickable
            cy = e["cy"]
            if cy < h * 0.18 or cy >= kbd_floor:         # skip app-bar echo + keyboard strip
                continue
            lab = (e.get("label") or "").lower().strip()
            if len(lab) < 2 or not _matches(lab):
                continue
            if e.get("clickable") and not _too_big(e):
                row = e                                  # the matching node is itself tappable
            else:
                # label lives on a non-clickable TextView -> tap its clickable CONTAINER:
                # the smallest clickable box whose bounds contain this label's centre.
                holders = [c for c in clickables if _contains(c, e["cx"], e["cy"])]
                if not holders:
                    continue
                row = min(holders, key=lambda c: c["w"] * c["h"])
            cands.append((len(lab), row, lab))           # rank by matched-label length (tightest)
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        _, row, lab = cands[0]
        return row["cx"], row["cy"], row.get("w"), row.get("h"), lab

    for _ in range(3):
        hit = _pick_suggestion()
        if hit:
            cx, cy, cw, ch, lab = hit
            execute_tap(serial, cx / w * 1000.0, cy / h * 1000.0,
                        target_w_px=cw, target_h_px=ch)
            return "tapped matching result '%s'" % ((lab or "")[:30])
        time.sleep(0.7)

    # An intermediate field only wanted its dropdown row -- no row means "leave it alone"
    # (pressing Enter here would submit a half-filled form).
    if suggestion_only:
        return None

    # 2) no matching suggestion -> press the on-screen Search/Enter key by HAND (the moat),
    #    else fall back to the IME enter action.
    _, ctrl = _kbd_full(serial)
    if "ENTER" in ctrl:
        _key_tap(serial, ctrl["ENTER"][0], ctrl["ENTER"][1], is_control=True)
        return "pressed the keyboard Search/Enter key"
    try:
        d.press("enter")
        return "pressed enter (IME search action)"
    except Exception:
        return "could not submit the search"


# Show/reveal-password "eye" toggle -- matched UNIVERSALLY by a11y patterns, never a
# specific app. Content-desc usually says something like "Show password"/"Hide password"/
# "Reveal password"/"Toggle password visibility"; resource-id is often a Material end-icon
# or a *_password_toggle. These patterns require the word "password" (or a known toggle id)
# so we never tap an unrelated button by mistake.
_PW_REVEAL_DESC = [
    r"(?i).*(show|reveal|view|unmask|display).{0,16}password.*",
    r"(?i).*password.{0,16}(show|reveal|visib|unmask|hidden).*",
    r"(?i).*toggle.{0,16}password.*",
    r"(?i).*password.{0,8}toggle.*",
]
_PW_REVEAL_ID = (r"(?i).*(password.?toggle|toggle.?password|text_input_end_icon|"
                 r"password.?visib|show.?password|reveal.?password|password.?eye).*")


def _reveal_password(serial) -> bool:
    """Tap a show/reveal-password 'eye' toggle so a masked field shows its REAL text. That
    lets us verify the hand-typed password against what's actually on screen and keep typing
    by hand (the moat) instead of a blind inject. UNIVERSAL (a11y patterns, no app-specific
    coords). Taps the toggle's centre with human motion (execute_tap). Returns True if a
    toggle was found and tapped; if none exists the caller falls back to the length check +
    inject, unchanged. Gated by MG_REVEAL_PASSWORD at the call site."""
    d = _device_cache[serial]
    w, h = get_device_size(serial)

    def _tap_center(el) -> bool:
        try:
            b = el.info.get("bounds") or {}
        except Exception:
            return False
        cx = (int(b.get("left", 0)) + int(b.get("right", 0))) // 2
        cy = (int(b.get("top", 0)) + int(b.get("bottom", 0))) // 2
        if cx <= 0 and cy <= 0:
            return False
        execute_tap(serial, cx * 1000.0 / w, cy * 1000.0 / h)   # 0-1000 space, human tap
        return True

    # clickable matches first (the real toggle), then any match; desc before resource-id.
    for want_click in (True, False):
        for pat in _PW_REVEAL_DESC:
            try:
                el = d(descriptionMatches=pat, clickable=True) if want_click else d(descriptionMatches=pat)
                if el.exists:
                    return _tap_center(el)
            except Exception:
                pass
        try:
            el = d(resourceIdMatches=_PW_REVEAL_ID, clickable=True) if want_click else d(resourceIdMatches=_PW_REVEAL_ID)
            if el.exists:
                return _tap_center(el)
        except Exception:
            pass
    return False


def execute_human_type(serial: str, text: str):
    d = _device_cache[serial]
    w, h = get_device_size(serial)
    persona = _PERSONAS[serial]
    rng = random.Random()
    fatg = _fatigue(serial, persona)

    print("[%s] [type] engine v5 (taps every key: number row + ?123 for symbols)" % serial)
    time.sleep(0.9)  # let the keyboard slide up before we read it
    chars, ctrl = _kbd_full(serial)

    # If the keyboard doesn't expose its keys at all, we can't tap it -- last-resort
    # type via the system input command (types the exact text, keeps Gboard active).
    if len([k for k in chars if k.isalpha()]) < 8:
        print("[%s] [type] keyboard not readable; using 'input text' (keeps Gboard)" % serial)
        _shell_type(serial, text)
        print("[%s] [type] %s typed: %s" % (serial, persona.name, text))
        return

    time.sleep(_lognormal(0.55, 0.35, 0.30, 1.10) * persona.dwell_scale)

    # SELF-VERIFYING TYPE: tap the text by hand (the moat), then READ BACK what actually
    # landed. If the keyboard misread a key (e.g. a number-row digit hitting the letter
    # under it -> "Akash1306" became "Akashqepy"), clear and retry; if two hand attempts
    # still don't match, inject the exact text so the VALUE is never wrong (correctness
    # beats the moat on a field that would otherwise hold garbage). Passwords render as
    # dots (or hide entirely), so they're verified by LENGTH, and a field we can't read
    # at all is trusted (never falsely overwrite a correct hidden password).
    reveal_on = os.environ.get("MG_REVEAL_PASSWORD", "1") not in ("0", "", "false", "False", "off")
    revealed = False
    for attempt in range(2):
        _tap_out_text(serial, text, chars, ctrl, persona, rng, fatg)
        actual, is_pw = _read_focused_field(serial)
        if actual is None:
            print("[%s] [type] field not readable to verify; accepting typed input" % serial)
            print("[%s] [type] %s typed: %s" % (serial, persona.name, text))
            return

        # PASSWORD: don't trust a masked field or inject blindly. Tap the show-password
        # "eye" toggle so the field reveals its REAL text -- then verify the hand-typed
        # characters EXACTLY, and if they're wrong retype BY HAND (the moat stays intact).
        # Reveal once; after that the field reads as plain text on this and later attempts.
        if is_pw and reveal_on and not revealed and _reveal_password(serial):
            revealed = True
            print("[%s] [type] tapped show-password (eye) to verify the password on screen" % serial)
            time.sleep(0.5)
            actual, is_pw = _read_focused_field(serial)   # re-read: now plain text if it unmasked
            if actual is None:
                print("[%s] [type] %s typed: %s (field unreadable after reveal; accepted)" % (serial, persona.name, text))
                return

        if is_pw:
            # no eye toggle, or it stayed masked -> verify by LENGTH as before (safe fallback)
            if not actual:      # fully hidden -> can't verify; keep the moat, trust it
                print("[%s] [type] %s typed: %s (password field, not inspectable)" % (serial, persona.name, text))
                return
            ok = (len(actual) == len(text))
            vlabel = "length"
        else:
            ok = (_type_norm(actual) == _type_norm(text))
            vlabel = "exact via show-password" if revealed else "exact"
        if ok:
            print("[%s] [type] %s typed: %s (verified %s)" % (serial, persona.name, text, vlabel))
            return
        shown = ("*" * len(actual)) if (is_pw and not revealed) else actual
        print("[%s] [type] field shows %r but wanted %r -- clearing and retyping by hand"
              % (serial, shown, text))
        _clear_focused(serial)
        time.sleep(0.4)
        chars, ctrl = _kbd_full(serial)   # refresh the keyboard read for the retry

    # Hand-typed attempts still wrong -> last resort, guarantee the value via injection.
    _clear_focused(serial)
    time.sleep(0.2)
    print("[%s] [type] per-key kept mismatching; injecting exact text (value integrity)" % serial)
    _shell_type(serial, text)
    print("[%s] [type] %s typed: %s (injected after retries)" % (serial, persona.name, text))


_SHELL_SPECIAL = set(" ()&<>|;$`\"'\\")


def _shell_type(serial: str, s: str):
    """Type via the system input command (does NOT switch the IME, so Gboard stays
    active -- unlike uiautomator2 send_keys, and no keyboard page-switch). Used for
    digits/symbols and as a whole-string fallback. Escapes shell-special chars."""
    d = _device_cache[serial]
    for ch in s:
        try:
            if ch == " ":
                d.shell("input keyevent 62")           # space
            elif ch in _SHELL_SPECIAL:
                d.shell("input text \\%s" % ch)        # escape for the shell
            else:
                d.shell("input text %s" % ch)
        except Exception:
            pass
        time.sleep(0.03)