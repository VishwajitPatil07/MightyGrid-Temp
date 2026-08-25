import time
import sys
import os
import re
import random
import subprocess
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.brain.vllm_client import ask_ui_tars_7b
from src.brain import planner
from src.eyes.screen import parse_elements, nearest_clickable, deterministic_button, find_input_fields, elements_for_prompt, detect_interruption, clickable_still_visible
from src.eyes import verify
from src.hands import adb_client
from src.hands.adb_client import (
    get_device_for_serial,
    get_device_size,
    execute_tap,
    execute_human_type,
    execute_scroll,
    open_app_organically,
    human_idle_gap,
    submit_search,
)
from src.hands.personas import generate_persona


def _free_intent_model():
    """Best-effort: make sure the intent model isn't parked in GPU memory from a
    previous run, so UI-TARS keeps the whole GPU. Harmless if Ollama isn't used."""
    import requests as _rq
    model = getattr(planner, "INTENT_MODEL", "qwen2.5:3b-instruct")
    bases = {planner.INTENT_API_URL.split("/v1/")[0], "http://localhost:11434"}
    for base in bases:
        try:
            _rq.post(base + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=4)
        except Exception:
            pass

_EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')


def _clean_ascii(s):
    return "".join(ch for ch in (s or "") if 32 <= ord(ch) < 127).strip()


def resolve_type_text(query, model_text):
    """Fallback text only for when we have no queued value: strip non-ASCII, and
    recover an email from the request if the model was aiming at one."""
    raw = model_text or ""
    ascii_text = _clean_ascii(raw)
    emails = _EMAIL_RE.findall(query or "")
    if emails and (("@" in raw) or ("@" in ascii_text) or not ascii_text):
        return emails[0]
    return ascii_text


def perceive(serial):
    d = get_device_for_serial(serial)
    path = "screen_%s.png" % serial
    last_err = None
    # uiautomator2's screenshot/dump can transiently drop the connection on ColorOS
    # (RemoteDisconnected). Don't let one blip kill the whole run -- retry a few times,
    # refreshing the device handle in between, before giving up.
    for attempt in range(3):
        try:
            xml = d.dump_hierarchy()
            d.screenshot(path)
            return xml, path, parse_elements(xml)
        except Exception as e:
            last_err = e
            print("[%s]   perceive hiccup (%s) -- retry %d/3" % (serial, type(e).__name__, attempt + 1))
            time.sleep(1.2)
            d = adb_client.reconnect_device(serial)
    raise last_err


def _dump(serial):
    try:
        return get_device_for_serial(serial).dump_hierarchy()
    except Exception:
        return ""


def _keyboard_shown(serial):
    """True if the soft keyboard (IME) is currently on screen."""
    try:
        d = get_device_for_serial(serial)
        out = d.shell("dumpsys input_method")
        s = out.output if hasattr(out, "output") else str(out)
        return "mInputShown=true" in s
    except Exception:
        return False


def _set_focused_text(serial, value):
    """Set the value straight into the focused field via the accessibility API
    (uiautomator2 set_text). Page-independent (no ?123 dance), clears existing
    text, and does NOT swap the keyboard away from Gboard -- unlike send_keys and
    unlike tapping keys, which kept breaking on digits/@/symbols. Returns success."""
    try:
        get_device_for_serial(serial)(focused=True).set_text(value)
        return True
    except Exception:
        return False


def _clear_focused_field(serial, target=None):
    """Erase whatever is already in the field before we type a value.
    Uses direct node targeting first to bypass sluggish accessibility focus states."""
    d = get_device_for_serial(serial)
    
    # 1. Bypass `focused=True` delay by using the exact resource ID if available
    if target and target.get("rid"):
        try:
            d(resourceId=target["rid"]).clear_text()
            return
        except Exception:
            pass
            
    # 2. Your original best-effort fallback
    try:
        d(focused=True).clear_text()
        return
    except Exception:
        try:
            d.clear_text()
            return
        except Exception:
            pass
            
    # 3. The bulletproof moat fallback: Select All (Ctrl+A) + Backspace
    # 113 = KEYCODE_CTRL_LEFT, 29 = KEYCODE_A, 67 = KEYCODE_DEL
    try:
        d.shell("input keycombination 113 29") 
        d.shell("input keyevent 67")
    except Exception:
        pass


def _dismiss_keyboard(serial):
    """Drop the keyboard (like a human does after typing) so the buttons it was
    covering -- Continue / Next / Search -- become visible to the next decision.
    Pressing Back closes the IME first without navigating away. No-op if the
    keyboard isn't up (so we never accidentally go back)."""
    if not _keyboard_shown(serial):
        return False
    try:
        get_device_for_serial(serial).press("back")
        time.sleep(0.6)
        return True
    except Exception:
        return False


def tap_on_screen(serial, elements, nx, ny):
    w, h = get_device_size(serial)
    px, py = int(w * nx / 1000.0), int(h * ny / 1000.0)
    diag = (w * w + h * h) ** 0.5
    # If the model's point is already inside a clickable element, TAP RIGHT THERE.
    # Tapping anywhere inside a control clicks it, so there's no reason to jerk the
    # tap to the element's centre -- that redirect was the repeated wrong-tap (the
    # point sat inside a wide bar and the tap jumped to its far centre). Tapping the
    # raw point clicks the same control and respects where the model aimed.
    inside = [e for e in elements if e["clickable"]
              and e["x1"] <= px <= e["x2"] and e["y1"] <= py <= e["y2"]]
    if inside:
        el = min(inside, key=lambda e: e["w"] * e["h"])
        label = (el["label"] or "element")[:24]
        print("[%s]   tap '%s' at model point (%d,%d)" % (serial, label, px, py))
        execute_tap(serial, nx, ny, target_w_px=el["w"], target_h_px=el["h"])
        return el
    # Dead space: snap to the nearest SMALL clickable (never a whole-screen box).
    el = nearest_clickable(elements, px, py, max_dist=0.09 * diag, w=w, h=h)
    if el:
        label = (el["label"] or "element")[:24]
        print("[%s]   snap (%d,%d) -> '%s' (%d,%d)" % (serial, px, py, label, el["cx"], el["cy"]))
        execute_tap(serial, el["cx"] / w * 1000.0, el["cy"] / h * 1000.0,
                    target_w_px=el["w"], target_h_px=el["h"])
        return el
    else:
        print("[%s]   no small element near (%d,%d); tapping raw" % (serial, px, py))
        execute_tap(serial, nx, ny)
        return None


def _explore_once(serial):
    """A highly curious persona pokes one interesting element after finishing --
    logged as exploration, never counted as task progress or success."""
    _, _, elements = perceive(serial)
    w, h = get_device_size(serial)
    picks = [e for e in elements if e["clickable"] and e["label"]]
    if not picks:
        return
    e = random.choice(picks)
    print("[%s] [curiosity] exploring '%s'" % (serial, (e["label"] or "?")[:20]))
    execute_tap(serial, e["cx"] / w * 1000.0, e["cy"] / h * 1000.0, target_w_px=e["w"])


def _finish_typing(serial, do_search, value, pick_suggestion=False):
    """After typing:
    1. If it's an explicit search or the final value entered, submit it (pick suggestion or press keyboard Enter/Search).
    2. If picking suggestions only (intermediate field), tap the suggestion if present.
    3. If neither applies and no action key was pressed, drop the keyboard to reveal UI buttons underneath."""
    if do_search and value:
        how = submit_search(serial, value)
        print("[%s]   search submitted -> %s" % (serial, how))
        return
    if pick_suggestion and value:
        how = submit_search(serial, value, suggestion_only=True)
        if how:
            print("[%s]   picked the suggestion -> %s" % (serial, how))
            return
    # If we have typed the value but no dropdown was clicked, press the keyboard's Search/Proceed key
    if value:
        how = submit_search(serial, value, suggestion_only=False)
        if how and "could not submit" not in how:
            print("[%s]   proceed/search submitted -> %s" % (serial, how))
            return
    if _dismiss_keyboard(serial):
        print("[%s]   keyboard dismissed (revealing buttons underneath)" % serial)


def _remember(history, desc):
    """Append a short, HUMAN-READABLE record of what we just did (e.g. "tapped 'Log in'")
    and keep the last few. Semantic entries -- not raw coordinates -- let the grounder see
    what it has already done, so it stops re-tapping the same button (the re-tap loop)."""
    history.append(desc)
    history[:] = history[-6:]


# Labels of buttons that SUBMIT a form and should DISAPPEAR on success (a login/signup
# button stays on screen after a failed attempt). Used to verify a finish is real.
_SUBMIT_WORDS = ("log in", "login", "log-in", "sign in", "signin", "sign-in", "log on",
                 "logon", "submit", "continue", "next")


def _is_submit(label):
    lab = (label or "").strip().lower().rstrip(" >").strip()
    return bool(lab) and len(lab) <= 24 and any(lab == w or lab.startswith(w) for w in _SUBMIT_WORDS)


# ---- code-maintained PROGRESS MEMORY (opt-in via MG_PROGRESS_MEMORY, default off) --------
# A running, high-level summary of the task so far -- what's accomplished, what's left, and
# what did NOT work -- built by CODE from real events (not the model) and fed to the model
# each turn so it can reason across a long flow instead of only the last few actions. First
# pass; expected to grow once long-flow failure data is in. Off by default so the current
# behaviour (what the testers measure) is unchanged.
def _new_progress(app, values):
    return {"opened": False, "values_total": len(values), "dead_ends": []}


def _note_dead_end(progress, what):
    if what and what not in progress["dead_ends"]:
        progress["dead_ends"].append(what)
        progress["dead_ends"][:] = progress["dead_ends"][-5:]


def _goal_with_progress(objective, progress, vi):
    """Objective unchanged unless MG_PROGRESS_MEMORY is on; then append a compact summary of
    the task so far (accomplished / still-to-do / do-not-repeat) to what the model sees."""
    if os.environ.get("MG_PROGRESS_MEMORY", "0") in ("0", "", "false", "False", "off"):
        return objective
    done = []
    if progress["opened"]:
        done.append("opened the app")
    if progress["values_total"]:
        done.append("entered %d of %d values" % (min(vi, progress["values_total"]), progress["values_total"]))
    note = "Progress so far -- accomplished: %s." % ("; ".join(done) if done else "nothing yet")
    left = progress["values_total"] - vi
    if left > 0:
        note += " Still to enter: %d value(s)." % left
    if progress["dead_ends"]:
        note += " Do NOT repeat these (they did nothing): %s." % "; ".join(progress["dead_ends"])
    return objective + "\n\n" + note


def _click_on_input_field(elements, px, py, pad=14):
    """True if the tap (px,py) lands on an editable text field -- so the model 'clicking'
    there is really an intent to type into that box, not to press a button."""
    for e in find_input_fields(elements):
        if (e.get("x1", 1) - pad) <= px <= (e.get("x2", -1) + pad) \
           and (e.get("y1", 1) - pad) <= py <= (e.get("y2", -1) + pad):
            return True
    return False


def _type_queued_value(serial, elements, values, vi, before_xml):
    """Type the vi-th queued value into the vi-th input field the way a person taps a box
    and types: focus the field (human motion), clear it, type it by HAND (self-verifying;
    a masked password is revealed via the eye icon inside execute_human_type). Returns
    (new_vi, advanced). SHARED by the 'type' action and the click-on-an-empty-field
    short-circuit, so both behave identically."""
    text = values[vi]
    fields = find_input_fields(elements)
    target = fields[min(vi, len(fields) - 1)] if fields else None
    used = "focused field"
    if target:
        w, h = get_device_size(serial)
        execute_tap(serial, target["cx"] / w * 1000.0, target["cy"] / h * 1000.0,
                    target_w_px=target.get("w"), target_h_px=target.get("h"))
        used = "field '%s'" % ((target.get("label") or "input")[:24])
        time.sleep(0.6)
        
    # FIX: Pass the target element down to bypass focus delays
    _clear_focused_field(serial, target) 
    execute_human_type(serial, text)
    after = _dump(serial)
    # Password fields render dots, so element_present can't see them -- accept a screen
    # change as proof the value went in.
    seen = verify.element_present(serial, text)
    if seen or (after and before_xml and after != before_xml):
        how = "confirmed on screen" if seen else "entered (masked or changed)"
        print("[%s]   typed value %d/%d into %s: %r (%s)"
              % (serial, vi + 1, len(values), used, text, how))
        return vi + 1, True
    print("[%s]   typed %r but nothing changed; not advancing" % (serial, text))
    return vi, False

# ---------------------------------------------------------------------------
# Reactive intent-driven loop: understand the intent once, then decide each next
# step live from the CURRENT screen -- no pre-planned UI steps.
# ---------------------------------------------------------------------------

def run_device_workflow(serial, query, seed=None):
    get_device_for_serial(serial)
    # MG_PERSONA_TYPE forces an archetype (speedy/careful/average/shaky/restless);
    # unset -> the seed picks the type, so a cast of seeds spans all types.
    ptype = os.environ.get("MG_PERSONA_TYPE") or None
    if seed is not None or ptype is not None:
        base_seed = seed if seed is not None else adb_client._PERSONAS[serial].seed
        adb_client._PERSONAS[serial] = generate_persona(base_seed, archetype=ptype)
    persona = adb_client._PERSONAS[serial]

    # PERSONA DECISIONS (patience -> give up early, curiosity -> wander) are a
    # diverse-user SIMULATION feature, OFF by default. Default = a focused,
    # persistent executor that finishes the task like the original loop did.
    # The persona's MOTION (tremor/flick/typing = the moat) is always on -- it
    # lives in the Hands and is unaffected by this switch.
    persona_decisions = os.environ.get("MG_PERSONA_BEHAVIOR", "0") not in ("0", "", "false", "False", "off")
    # How many no-progress (screen-unchanged) steps before abandoning as stuck. Default 6;
    # raise via MG_STUCK_BUDGET to give a long/complex flow more slack before giving up.
    try:
        _default_stuck = max(1, int(os.environ.get("MG_STUCK_BUDGET", "6")))
    except ValueError:
        _default_stuck = 6
    stuck_budget = (persona.retry_budget + 1) if persona_decisions else _default_stuck
    mode = "persona-decisions ON (may quit early / explore)" if persona_decisions else "reliable executor"

    print("\n[%s] Persona %s (seed %d) | %s | stuck budget %d"
          % (serial, persona.name, persona.seed, mode, stuck_budget))
    print("[%s] REQUEST: %s" % (serial, query))

    intent = planner.parse_intent(query, persona=persona, verbose=True)
    app = intent["app"]
    objective = intent["objective"]
    values = list(intent["values"])
    success_when = intent["success_when"]
    # Progress memory must exist BEFORE the app-open step below (which records the
    # 'opened' milestone) -- it was previously created after the loop start, which
    # crashed the run with UnboundLocalError.
    progress = _new_progress(app, values)  # code-maintained (MG_PROGRESS_MEMORY, default off)
    print("[%s] INTENT: app=%r objective=%r values=%r done_when=%r"
          % (serial, app, objective, values, success_when))

    # An "install ..." request must go through the Play Store -- client apps live
    # there, not in ColorOS's App Market (which the model would otherwise wander
    # into). Launch Play Store directly by package (reliable, no store-icon guess).
    is_install = "install" in (objective or "").lower()
    # A search task must SUBMIT after typing (tap a matching suggestion or press the
    # keyboard Search key) -- otherwise it types the query and just sits there.
    is_search = bool(re.search(r"\bsearch(ing|ed|es)?\b|\blook\s+(for|up)\b", (objective or "").lower()))
    PLAY_STORE_PKG = "com.android.vending"
    if is_install:
        # Open the Play Store the HUMAN way -- go home, find its icon, tap it -- like
        # a person would, instead of an instant package launch. Fall back to the
        # direct launch ONLY if the icon can't be found (ColorOS drawer is unreliable),
        # so a hard-to-find store is never permanently stuck.
        print("[%s] --- install request: opening Play Store (human: find + tap icon) ---" % serial)
        dev = get_device_for_serial(serial)
        open_app_organically(serial, "Play Store")
        time.sleep(1.5)
        try:
            cur = dev.app_current().get("package", "")
        except Exception:
            cur = ""
        if cur != PLAY_STORE_PKG:
            print("[%s]   couldn't find the Play Store icon; direct-launch fallback" % serial)
            try:
                dev.app_start(PLAY_STORE_PKG)
                time.sleep(2.5)
                try:
                    cur = dev.app_current().get("package", "")
                except Exception:
                    cur = ""
            except Exception as e:
                print("[%s]   couldn't launch Play Store: %s" % (serial, e))
        print("[%s]   Play Store open -> %s (foreground %s)"
              % (serial, cur == PLAY_STORE_PKG, cur or "?"))
    # Otherwise open the app the HUMAN way first: look on the home screen, and if
    # it's not there, open the app drawer and scroll to find and tap its icon. Only
    # if that genuinely fails to foreground the app do we fall back to a direct
    # package launch, so a hard-to-find app is never permanently stuck.
    elif app:
        print("[%s] --- opening %s (home + drawer scroll) ---" % (serial, app))
        open_app_organically(serial, app)
        ok, detail = verify.wait_foreground(serial, app, timeout=5.0)
        if ok is False:
            print("[%s]   couldn't find %s by scrolling; package-launch fallback" % (serial, app))
            try:
                pkg = verify.resolve_package(serial, app)
            except Exception:
                pkg = None
            if pkg:
                try:
                    get_device_for_serial(serial).app_start(pkg)
                    print("[%s]   launched %s" % (serial, pkg))
                    ok, detail = verify.wait_foreground(serial, app, timeout=5.0)
                except Exception as e:
                    print("[%s]   app_start also failed (%s)" % (serial, e))
        print("[%s]   verify open -> %s (%s)" % (serial, ok, detail))
        if ok:
            # Defensive: never let progress-memory bookkeeping break a real run.
            try:
                progress["opened"] = True
            except Exception:
                pass

    history = []
    vi = 0                 # index of the next value to type
    no_progress = 0
    outcome = "incomplete"
    # Run until the task reaches a NATURAL stop -- finished (done), or the guards below
    # detect a genuine stall (stuck_budget) / fixation (same-spot re-taps) / abort. There
    # is NO hard step cap by default, so the model keeps going until the task is actually
    # done. MG_MAX_STEPS=N imposes an optional backstop; 0 or unset = unlimited. (Press
    # Ctrl+C to stop a run by hand.)
    try:
        max_steps = int(os.environ.get("MG_MAX_STEPS", "0"))   # 0 = unlimited
    except ValueError:
        max_steps = 0
    first_xml = None       # the screen at the start of the task (progress baseline)
    interrupts_handled = 0 # blocking popups auto-dismissed (capped, to avoid loops)
    last_submit = None     # the last submit/login button tapped, to verify it worked
    wait_streak = 0        # consecutive 'wait' decisions (stuck-on-a-popup guard)
    unclear_streak = 0     # consecutive unparseable model replies (prose instead of Action)
    finish_rejects = 0     # premature 'finished' calls rejected (capped, to avoid loops)
    recent_clicks = []     # recent model tap points (px) -- to catch it fixating on one spot
    model_repeat = 0       # times we've had to break a same-spot repeat loop
    det_blocked = set()    # deterministic buttons tapped that didn't move the screen
                           # (cleared on any screen change -- short-term guard)
    det_counts = {}        # how many times each deterministic label has been tapped
    det_permablock = set() # labels tapped too many times overall -> hand to the model
                           # for good (breaks home<->screen oscillation loops that the
                           # screen-change guard above can't catch)
    DET_MAX = 3

    step = 0
    while True:
        step += 1
        if max_steps and step > max_steps:
            print("[%s]   reached MG_MAX_STEPS=%d -- stopping" % (serial, max_steps))
            break
        print("\n[%s] --- Step %d%s ---" % (serial, step, ("/%d" % max_steps) if max_steps else ""))
        before_xml, img, elements = perceive(serial)
        if first_xml is None:
            first_xml = before_xml   # baseline to measure real progress against

        # ---- universal interruption handling (app-agnostic) ----
        # Clear any blocking popup -- a system permission dialog, a "rate us"/
        # notification nag, a no-internet error -- BEFORE the model looks at the
        # screen, so an unexpected interstitial can't derail the task. Conservative:
        # only fires on high-confidence patterns (see detect_interruption). Gated by
        # MG_HANDLE_INTERRUPTS (default on); capped so it can never loop forever.
        if (os.environ.get("MG_HANDLE_INTERRUPTS", "1") not in ("0", "", "false", "off")
                and interrupts_handled < 8):
            try:
                _fg = get_device_for_serial(serial).app_current().get("package", "")
            except Exception:
                _fg = ""
            iel, ikind = detect_interruption(elements, _fg)
            if iel is not None:
                w, h = get_device_size(serial)
                print("[%s]   [interrupt:%s] dismissing '%s'"
                      % (serial, ikind, (iel.get("label") or "?")[:24]))
                execute_tap(serial, iel["cx"] / w * 1000.0, iel["cy"] / h * 1000.0,
                            target_w_px=iel["w"], target_h_px=iel["h"])
                interrupts_handled += 1
                history.append("dismissed a %s popup" % ikind)
                history[:] = history[-6:]
                time.sleep(1.0)
                continue

        # Already done? (evidence-based, and only when we have a real condition
        # -- an empty success_when must never auto-complete).
        if success_when:
            sok, sreason = verify.check_success(serial, success_when)
            if sok:
                outcome = "completed"
                print("[%s]   success detected: %s" % (serial, sreason))
                break

        t0 = time.time()

        # ---- deterministic-first ----
        # If the next step is an obvious labeled button (Log in / Continue / Search
        # / Allow ...) that's on screen, tap it straight from the accessibility tree
        # with NO vision-model call: exact coordinates, ~milliseconds, and it can't
        # mis-snap. Gated so it never fires while values still need typing. Set
        # MG_DETERMINISTIC=0 to disable.
        if os.environ.get("MG_DETERMINISTIC", "1") not in ("0", "", "false", "off"):
            det = deterministic_button(objective, elements, values, vi)
            det_label = det["label"].strip().lower() if det else None
            # Tap deterministically only if this exact button isn't short-term blocked
            # (did nothing last time) AND isn't permanently blocked (tapped too many
            # times across this task -- an oscillation). Otherwise fall to the model.
            if det is not None and det_label not in det_blocked and det_label not in det_permablock:
                w, h = get_device_size(serial)
                print("[%s]   [deterministic] tap '%s' -- no model call (%.0fms)"
                      % (serial, det["label"][:30], (time.time() - t0) * 1000))
                human_idle_gap(serial)
                execute_tap(serial, det["cx"] / w * 1000.0, det["cy"] / h * 1000.0,
                            target_w_px=det["w"], target_h_px=det["h"])
                det_counts[det_label] = det_counts.get(det_label, 0) + 1
                if det_counts[det_label] >= DET_MAX:
                    det_permablock.add(det_label)
                    print("[%s]   tapped '%s' %d times with no real progress; leaving it to the model"
                          % (serial, det["label"][:30], DET_MAX))
                history.append("click %s" % det["label"][:30])
                history[:] = history[-6:]
                if _is_submit(det.get("label")):
                    last_submit = det   # remember the submit button, to verify it worked
                time.sleep(1.0)
                after_xml = _dump(serial)
                if after_xml and before_xml and after_xml == before_xml:
                    no_progress += 1
                    det_blocked.add(det_label)   # it did nothing -- stop retrying it
                    _note_dead_end(progress, "'%s'" % det_label)
                    print("[%s]   '%s' didn't change the screen; handing back to the model"
                          % (serial, det["label"][:30]))
                else:
                    no_progress = 0
                    det_blocked.clear()          # screen moved -- fresh start (permablock stays)
                if no_progress > stuck_budget:
                    outcome = "abandoned"
                    print("[%s]   giving up (stuck %d, out of patience)" % (serial, no_progress))
                    break
                continue

        _w, _h = get_device_size(serial)
        screen_list = elements_for_prompt(elements, _w, _h)
        # MG_THINK_MOTION (default OFF now): the idle "thinking" fidget was a small
        # up/down scroll while the grounder decides. On the local slow setup every
        # decision is ~16s so it fidgeted on nearly every screen (a dead-vertical scroll
        # that read as robotic), so it's OFF by default -- the phone holds still and we
        # make the exact single blocking decision call. Set MG_THINK_MOTION=1 to bring
        # the (rare, self-restoring) fidget back.
        if os.environ.get("MG_THINK_MOTION", "0") not in ("0", "", "false", "off"):
            _res = {}

            def _decide():
                _res["d"], _res["r"] = ask_ui_tars_7b(
                    goal=_goal_with_progress(objective, progress, vi), screen_data_json=screen_list,
                    image_path=img, action_history=history)

            _th = threading.Thread(target=_decide, daemon=True)
            _th.start()
            try:
                adb_client.human_thinking_motion(serial, _th.is_alive)
            except Exception:
                pass
            _th.join()
            decision = _res.get("d", {"action": "none", "target": "error"})
            raw = _res.get("r", "")
        else:
            decision, raw = ask_ui_tars_7b(goal=_goal_with_progress(objective, progress, vi), screen_data_json=screen_list,
                                           image_path=img, action_history=history)
        print("[%s]   decision: %s (%.1fs)" % (serial, decision, time.time() - t0))
        act = decision.get("action")

        if act == "none":
            print("[%s]   grounder unclear: %r" % (serial, (raw or "")[:120]))
            no_progress += 1
            unclear_streak += 1
            # The model runs at temperature 0, so re-asking the SAME prompt returns the
            # SAME non-answer forever. Change the prompt: tell it plainly to reply with one
            # Action line. If it still can't, press back to change the SCREEN itself.
            _remember(history, "the last reply was not an action -- reply with exactly ONE "
                               "'Action:' line (click/type/scroll/back/finished), no prose")
            if unclear_streak >= 2:
                print("[%s]   still unclear -- pressing back to leave this screen" % serial)
                try:
                    get_device_for_serial(serial).press("back")
                except Exception:
                    pass
                unclear_streak = 0
                time.sleep(1.0)
            if no_progress > stuck_budget:
                outcome = "abandoned"
                print("[%s]   giving up (no progress %d)" % (serial, no_progress))
                break
            time.sleep(1.5)
            continue
        unclear_streak = 0

        if act == "abort":
            outcome = "abandoned"
            print("[%s]   grounder aborted: required information is missing -- stopping" % serial)
            break

        if act == "finished":
            sok, sreason = verify.check_success(serial, success_when)
            if sok:
                outcome = "completed"
                print("[%s]   grounder says finished; success check: True (%s)" % (serial, sreason))
                break
            # No explicit success token matched -> verify the step by EVIDENCE before
            # trusting the model's word: every queued value must have been entered AND
            # the screen must have actually moved from where the task started. This
            # stops a premature 'finished' from being accepted before the work is done
            # (e.g. the model claiming done after filling only one of two fields).
            typed_all = vi >= len(values)
            progressed = (first_xml is None) or (before_xml != first_xml)
            # A still-visible submit/login button means the form was NOT actually
            # submitted -- a failed login shows an error but KEEPS the 'Log in' button,
            # so 'screen moved' alone is a FALSE success. Require the button we pressed
            # to be gone before trusting a values-based finish.
            submit_stuck = last_submit is not None and clickable_still_visible(last_submit, elements)
            if typed_all and progressed and not submit_stuck:
                outcome = "completed"
                print("[%s]   grounder says finished; verified by evidence (values %d/%d entered, screen moved)"
                      % (serial, vi, len(values)))
                break
            finish_rejects += 1
            why = ("the submit button is still on screen (form not submitted)" if submit_stuck
                   else "values %d/%d, screen moved: %s" % (vi, len(values), progressed))
            print("[%s]   grounder says finished but NOT verified (%s) -- continuing" % (serial, why))
            if finish_rejects >= 3:
                outcome = "completed_unverified"
                print("[%s]   accepting finished after %d checks (avoid loop)" % (serial, finish_rejects))
                break
            history.append("thought done but wasn't -- keep working")
            history[:] = history[-6:]
            continue

        # ---- carry out the action ----
        # Human pause before acting -- padded only if perceive+decide was faster than a
        # sampled human idle (so a slow decision adds nothing). Covers all touch actions.
        human_idle_gap(serial)
        acted = None   # a human-readable record of THIS step, for semantic history below
        if act == "type":
            typed_text = None
            if vi < len(values):
                typed_text = values[vi]
                vi, advanced = _type_queued_value(serial, elements, values, vi, before_xml)
                no_progress = 0 if advanced else no_progress + 1
            else:
                text = resolve_type_text(query, decision.get("text", ""))
                if text:
                    execute_human_type(serial, text)
                    typed_text = text
                else:
                    print("[%s]   no value left to type; skipping" % serial)
            # A person now either RUNS the search or drops the keyboard to reveal buttons.
            # All values in -> run the search. Still values pending in a search flow ->
            # pick this field's dropdown row (city pickers need the row tapped).
            _finish_typing(serial, is_search and vi >= len(values), typed_text,
                           pick_suggestion=True)
            if typed_text:
                acted = ("searched '%s'" % typed_text[:30]) if is_search else "entered a value into the form"
        elif act == "open":
            open_app_organically(serial, decision.get("text", app or objective))
            acted = "opened %s" % str(decision.get("text", app or objective))[:30]
        elif act == "click" and isinstance(decision.get("target"), list):
            nx, ny = decision["target"]
            w, h = get_device_size(serial)
            px, py = int(w * nx / 1000.0), int(h * ny / 1000.0)
            # HUMAN SHORT-CIRCUIT: the model can't see focus, so on a login it clicks the
            # empty box over and over instead of typing (the tap-tap-tap loop in the log).
            # If the click lands on an INPUT FIELD, a value is still pending, and the next
            # field to fill is EMPTY -> just type that value, the way a person taps a box
            # and types. This ends the loop; it's exactly what the 'type' action would do.
            if vi < len(values) and _click_on_input_field(elements, px, py):
                nf = find_input_fields(elements)
                nf = nf[min(vi, len(nf) - 1)] if nf else None
                if nf is not None:
                    print("[%s]   click landed on an input field -> typing the pending value" % serial)
                    typed_text = values[vi]
                    vi, advanced = _type_queued_value(serial, elements, values, vi, before_xml)
                    no_progress = 0 if advanced else no_progress + 1
                    if advanced:
                        recent_clicks.clear()
                        model_repeat = 0
                    # All values in -> run the search. Still values pending in a search
                    # flow -> pick this field's dropdown row (city pickers need the tap).
                    _finish_typing(serial, is_search and vi >= len(values), typed_text,
                                   pick_suggestion=True)
                    _remember(history, "entered a value into a field")
                    continue
            # Anti-repeat: UI-TARS can FIXATE -- clicking essentially the same spot
            # over and over (the screen may flicker so the no-progress guard misses
            # it). If this target matches 2+ of the last 3 taps, don't tap it again:
            # tell the model to pick a DIFFERENT element and press back to escape the
            # dead-end. If it just keeps doing it, abandon (genuinely stuck).
            near = sum(1 for (qx, qy) in recent_clicks[-3:]
                       if abs(qx - px) <= 45 and abs(qy - py) <= 45)
            if near >= 2:
                model_repeat += 1
                print("[%s]   model repeating the same tap (%d,%d) [%d] -- breaking the loop"
                      % (serial, px, py, model_repeat))
                _note_dead_end(progress, "that repeated spot")
                history.append("already tapped that spot several times with no result -- pick a DIFFERENT element")
                history[:] = history[-6:]
                recent_clicks.clear()
                if model_repeat >= 3:
                    outcome = "abandoned"
                    print("[%s]   giving up (stuck repeating one tap)" % serial)
                    break
                get_device_for_serial(serial).press("back")   # escape the stuck screen
                time.sleep(1.0)
                continue
            model_repeat = 0
            recent_clicks.append((px, py))
            recent_clicks[:] = recent_clicks[-6:]
            el = tap_on_screen(serial, elements, nx, ny)
            lbl = (el.get("label") or "element")[:24] if el else None
            acted = ("tapped '%s'" % lbl) if lbl else "tapped an element"
            if el and _is_submit(el.get("label")):
                last_submit = el   # remember the submit button, to verify it worked
        elif act == "scroll":
            execute_scroll(serial, decision.get("direction", "down"))
            acted = "scrolled %s" % decision.get("direction", "down")
        elif act == "back":
            dev = get_device_for_serial(serial)
            dev.press("back")
            # Foreground guard: if Back went too far and kicked us out to the launcher/home,
            # re-open the target app so the task can continue (a person wouldn't just abandon
            # the app). Only fires when we actually landed on a launcher.
            if app:
                time.sleep(0.8)
                try:
                    cur = dev.app_current().get("package", "").lower()
                except Exception:
                    cur = ""
                if cur and ("launcher" in cur or "home" in cur):
                    print("[%s]   back exited to home -> re-opening %s" % (serial, app))
                    open_app_organically(serial, app)
                    time.sleep(1.0)
            acted = "pressed back"
        elif act == "home":
            get_device_for_serial(serial).press("home")
            acted = "pressed home"
        elif act == "enter":
            get_device_for_serial(serial).press("enter")
            acted = "pressed enter"
        elif act == "wait":
            time.sleep(4)
            # The model can get stuck answering 'wait' forever on a screen that will never
            # change on its own (an unexpected popup, a dead-end). A person waits once or
            # twice, then DOES something. After a few waits in a row, tell the model that
            # waiting isn't working and count it as no-progress so the stall guard can end
            # the task instead of waiting all day. Any real action resets the streak.
            wait_streak += 1
            if wait_streak >= 3:
                no_progress += 1
                print("[%s]   waited %d times with nothing changing -- asking for a real action"
                      % (serial, wait_streak))
                _remember(history, "waiting has not helped -- do NOT wait again; take a real "
                                   "action on what is on screen (or press back to leave it)")
                wait_streak = 0
        if act != "wait":
            wait_streak = 0

        # Record what we did in HUMAN terms (semantic) so the grounder can see it and stop
        # redoing finished steps (the re-tap-'Log in' loop). Prefer the semantic 'acted';
        # fall back to the model's own action text only if we didn't set one.
        if acted:
            _remember(history, acted)
        elif raw:
            m = re.search(r'(?:Action:|^\d+\.)\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
            _remember(history, m.group(1).strip() if m else raw.strip())

        # ---- did the screen move at all? (patience-bounded) ----
        time.sleep(1.0)
        after_xml = _dump(serial)
        if after_xml and before_xml and after_xml != before_xml:
            det_blocked.clear()          # screen moved -- deterministic buttons worth retrying
        if act != "type":
            if after_xml and before_xml and after_xml == before_xml:
                no_progress += 1
                print("[%s]   screen unchanged (no progress %d/%d)" % (serial, no_progress, stuck_budget))
            else:
                no_progress = 0
        if no_progress > stuck_budget:
            outcome = "abandoned"
            print("[%s]   giving up (stuck %d, out of patience)" % (serial, no_progress))
            break

    if persona_decisions and outcome == "completed" and persona.curiosity > 0.7:
        try:
            _explore_once(serial)
        except Exception:
            pass

    print("\n[%s] OUTCOME: %s  (%s)" % (serial, outcome.upper(), persona.name))
    return outcome


def _log_result(serial, query, outcome, steps, seconds):
    """Append one row per run to MG_RESULTS_CSV (default results.csv) so a tester's runs
    are captured automatically instead of hand-copied from the console. Pure record-keeping
    -- it never changes what the agent does, and a failure to write is ignored. Columns:
    when, serial, persona, task, outcome, steps, seconds."""
    path = os.environ.get("MG_RESULTS_CSV", "results.csv")
    try:
        persona = adb_client._PERSONAS.get(serial)
        pname = persona.name if persona else ""
    except Exception:
        pname = ""
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), serial, pname,
           (query or "").replace("\n", " ").strip(), str(outcome), str(steps), "%.0f" % seconds]
    try:
        import csv
        newfile = not os.path.exists(path)
        with open(path, "a", newline="", encoding="ascii", errors="replace") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(["when", "serial", "persona", "task", "outcome", "steps", "seconds"])
            w.writerow(row)
    except Exception as e:
        print("[%s]   (could not write result row: %s)" % (serial, e))


def _run_and_log(serial, query, seed=None):
    """Thread target: run the workflow, time it, and record ONE result row per device.
    Wraps run_workflow without changing it. Handles both return shapes: a single-step run
    returns an outcome string; a multi-step run returns a list of (subtask, outcome) -- for
    which we log 'completed' only if every step passed, else the first failing outcome."""
    t0 = time.time()
    try:
        res = run_workflow(serial, query, seed)
    except Exception as e:
        _log_result(serial, query, "error", 0, time.time() - t0)
        print("[%s]   workflow crashed: %s" % (serial, e))
        return
    if isinstance(res, list):
        ok = ("completed", "completed_unverified")
        bad = next((o for _, o in res if o not in ok), None)
        outcome = bad if bad else "completed"
    else:
        outcome = res if isinstance(res, str) else "completed"
    _log_result(serial, query, outcome, "", time.time() - t0)
    print("[%s]   >>> RESULT: %s (logged to %s)"
          % (serial, str(outcome).upper(), os.environ.get("MG_RESULTS_CSV", "results.csv")))


def run_workflow(serial, query, seed=None):
    """Client workflow runner. Splits the request into ordered steps and runs each
    through the reactive loop above, keeping the SAME persona across all steps, then
    prints a per-step pass/fail report. A single-step request behaves EXACTLY like
    run_device_workflow (no change). Env MG_MULTI_STEP=0 forces one-shot behavior.

    Each step gets the loop's full step budget, so a workflow can take far more than
    20 actions total (N steps x budget) -- and a step that can't complete stops the
    workflow, since later steps usually depend on it."""
    multi = os.environ.get("MG_MULTI_STEP", "1") not in ("0", "", "false", "off")
    steps = planner.plan_steps(query) if multi else [{"objective": query}]
    if len(steps) <= 1:
        return run_device_workflow(serial, query, seed)

    print("\n[%s] ===== CLIENT WORKFLOW: %d steps =====" % (serial, len(steps)))
    for i, st in enumerate(steps):
        print("[%s]   step %d: %s" % (serial, i + 1, st["objective"][:70]))

    ok_states = ("completed", "completed_unverified")
    results = []
    for i, st in enumerate(steps):
        subq = st["objective"]
        s = seed if i == 0 else None    # step 1 sets the persona; later steps keep it
        print("\n[%s] ========== STEP %d/%d ==========" % (serial, i + 1, len(steps)))
        try:
            outcome = run_device_workflow(serial, subq, s)
            outcome = outcome if isinstance(outcome, str) else "completed"
        except Exception as e:
            outcome = "error"
            print("[%s]   step %d crashed: %s" % (serial, i + 1, e))
        results.append((subq, outcome))
        if outcome not in ok_states:
            print("[%s]   step %d did not complete (%s) -- stopping workflow"
                  % (serial, i + 1, outcome))
            break

    passed = sum(1 for _, o in results if o in ok_states)
    print("\n[%s] ===== WORKFLOW REPORT: %d/%d steps completed =====" % (serial, passed, len(steps)))
    for i, (q, o) in enumerate(results):
        mark = "OK " if o in ok_states else "XX "
        print("[%s]   %s step %d: %s -> %s" % (serial, mark, i + 1, q[:48], o.upper()))
    for j in range(len(results), len(steps)):
        print("[%s]   -- step %d: %s -> SKIPPED" % (serial, j + 1, steps[j]["objective"][:48]))
    return results


def get_connected_devices():
    try:
        out = subprocess.check_output(["adb", "devices"]).decode()
        serials = []
        for line in out.splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device":
                serials.append(parts[0])
        return serials
    except Exception as e:
        print("Error listing ADB devices: %s" % e)
        return []


if __name__ == "__main__":
    task_goal = sys.argv[1] if len(sys.argv) > 1 else (
        "Open the Agoda app, go to the login screen, enter the email test@test.com, and tap continue."
    )

    devices = get_connected_devices()
    if not devices:
        print("No devices found. Connect a phone and check `adb devices`.")
        sys.exit(1)

    seeds = [s.strip() for s in os.environ.get("MG_SEEDS", "").split(",") if s.strip()]

    _free_intent_model()
    print("Found %d device(s): %s   [build: browse-first-open]" % (len(devices), devices))
    print("REQUEST FOR ALL DEVICES: %s\n" % task_goal)

    threads = []
    for idx, serial in enumerate(devices):
        seed = int(seeds[idx]) if idx < len(seeds) else None
        t = threading.Thread(target=_run_and_log, args=(serial, task_goal, seed))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print("\nAll device workflows finished.")