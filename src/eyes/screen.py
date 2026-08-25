import re

# Turns a raw uiautomator2 dump_hierarchy() XML string into a plain list of
# on-screen elements, and gives two screen-aware helpers the agent leans on:
#   - nearest_clickable(): snap the model's approximate coordinate onto the real
#     button underneath (or nearest to) it, so a tap never lands on empty space.
#   - keyboard_keys(): read where each key ACTUALLY is right now, so typing taps
#     the real 'p' instead of a guessed grid cell that might be the voice icon.

_BOUNDS = re.compile(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"')


def _attr(node, name):
    key = name + '="'
    if key in node:
        return node.split(key, 1)[1].split('"', 1)[0]
    return ""


def parse_elements(xml):
    """Every node with a real box -> a dict with its pixel box, centre, label,
    and whether it is clickable."""
    els = []
    for node in xml.split("<node")[1:]:
        m = _BOUNDS.search(node)
        if not m:
            continue
        x1, y1, x2, y2 = map(int, m.groups())
        if x2 <= x1 or y2 <= y1:
            continue
        text = _attr(node, "text").strip()
        desc = _attr(node, "content-desc").strip()
        els.append({
            "label": text or desc,
            "text": text,
            "desc": desc,
            "class": _attr(node, "class"),
            "rid": _attr(node, "resource-id"),
            "focused": _attr(node, "focused") == "true",
            "password": _attr(node, "password") == "true",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
            "w": x2 - x1, "h": y2 - y1,
            "clickable": _attr(node, "clickable") == "true",
        })
    return els


def find_input_fields(elements):
    """Editable text fields on screen, ordered top-to-bottom. Lets us target the
    right box deterministically (find it in the tree) instead of letting the vision
    model guess a coordinate and mis-tap a nearby link. Each field keeps its
    'password' flag so a login's two boxes can be told apart."""
    fields = [e for e in elements
              if "EditText" in (e.get("class") or "") or e.get("password")]
    fields.sort(key=lambda e: e["cy"])
    return fields


# Whether a box is a SEARCH box has to be read off the FIELD, not off the user's
# wording: "add wireless earbuds to your cart" never says "search", yet it still has
# to be typed into Amazon's search bar and RUN. Typing without submitting leaves the
# query sitting in the box, so the model taps the bar and retypes it forever. Read
# app-agnostic signals from the a11y tree: the resource-id (…:id/rs_search_src_text),
# or a search-ish label/hint on an editable box.
_SEARCH_HINTS = ("search", "find", "explore", "query")


def looks_like_search_field(el):
    """True if this element is an editable box that is a SEARCH box."""
    if not el:
        return False
    if not ("EditText" in (el.get("class") or "") or el.get("password")):
        return False
    if el.get("password"):
        return False                     # a password box is never a search box
    rid = (el.get("rid") or "").lower().rsplit("/", 1)[-1]
    if any(h in rid for h in _SEARCH_HINTS):
        return True
    blob = " ".join([(el.get("label") or ""), (el.get("desc") or "")]).lower()
    return any(h in blob for h in _SEARCH_HINTS)


def active_search_field(elements):
    """The search box the user is typing into right now: the FOCUSED one if the tree
    marks one, else the only search box on screen. None if this screen has no search
    box (a login form, a checkout page) -- so nothing is ever wrongly submitted."""
    fields = [e for e in (elements or []) if looks_like_search_field(e)]
    if not fields:
        return None
    for e in fields:
        if e.get("focused"):
            return e
    return fields[0] if len(fields) == 1 else None


# ---- Universal (app-agnostic) interruption handling --------------------------
# Unexpected popups block a task on ANY app: system permission dialogs, "rate us" /
# notification nags, no-internet errors. These are detected from the accessibility
# tree by generic patterns -- never app-specific -- and only on HIGH-CONFIDENCE
# signals, so a real task screen is never mistaken for an interruption.

# Labels that GRANT a system permission (only used inside a real permission dialog).
_GRANT_LABELS = ["while using the app", "allow only while using", "only this time",
                 "allow all the time", "allow", "install", "update", "got it",
                 "continue", "ok", "yes"]
# Labels that clearly DISMISS a nag/popup. Deliberately only the unambiguous ones --
# a normal task screen almost never shows these, so dismissing them is safe. NOT
# included: "cancel"/"skip"/"later"/"dismiss" alone (too ambiguous -> left to model).
_DISMISS_LABELS = ["no thanks", "no, thanks", "no thank you", "maybe later",
                   "remind me later", "ask me later", "not now", "not right now"]
_RETRY_LABELS = ["retry", "try again", "reload"]
_NOINTERNET_HINTS = ["no internet", "no connection", "couldn't connect", "cannot connect",
                     "check your connection", "not connected", "network error",
                     "unable to connect", "connection error", "no network"]


def _find_label(elements, wanted):
    """First CLICKABLE element whose label equals or starts with a wanted phrase,
    scanning in wanted-priority order (so 'while using the app' beats 'allow')."""
    for phrase in wanted:
        for e in elements:
            if not e.get("clickable"):
                continue
            lab = (e.get("label") or "").strip().lower()
            if not lab or len(lab) > 30:
                continue
            norm = lab.rstrip(" >").strip()
            if norm == phrase or norm.startswith(phrase):
                return e
    return None


def detect_interruption(elements, foreground_pkg=""):
    """Universal detector for a blocking interruption. Returns (element_to_tap, kind)
    to handle it, or (None, None). Conservative by design -- fires only on:
      1. a system PERMISSION / install dialog (foreground is the permission
         controller / package installer) -> grant it so the task can proceed;
      2. a NO-INTERNET / connection error with a retry button -> retry;
      3. a nag/popup showing an UNAMBIGUOUS dismiss label -> dismiss.
    Everything else is left to the model, so normal task screens are untouched."""
    pkg = (foreground_pkg or "").lower()

    if "permissioncontroller" in pkg or "packageinstaller" in pkg or "permission" in pkg:
        el = _find_label(elements, _GRANT_LABELS)
        if el:
            return el, "permission"

    alltext = " ".join((e.get("label") or "").lower() for e in elements)
    # Some permission overlays (Android 11+) don't run under the permission-controller
    # package, so the package check above misses them. Also treat a dialog that clearly
    # offers BOTH Allow and Deny as a permission prompt and grant it. Conservative: it
    # needs an allow AND a deny/don't-allow present, so ordinary screens don't match.
    if ("allow" in alltext) and ("deny" in alltext or "don't allow" in alltext or "dont allow" in alltext):
        el = _find_label(elements, _GRANT_LABELS)
        if el:
            return el, "permission"

    if any(h in alltext for h in _NOINTERNET_HINTS):
        el = _find_label(elements, _RETRY_LABELS)
        if el:
            return el, "no-internet"

    el = _find_label(elements, _DISMISS_LABELS)
    if el:
        return el, "dismiss-nag"

    return None, None


def elements_for_prompt(elements, w, h, max_n=28):
    """Compact list of the ACTIONABLE on-screen elements -- label + centre in the
    model's own 0-1000 normalized space -- to hand the grounder as text alongside
    the screenshot. It can then copy the coordinate of the element whose label
    matches the goal, instead of eyeballing a point on a shrunk image and landing
    on the wrong thing. Clickable or editable, labeled, deduped, top-to-bottom,
    capped so the prompt stays short."""
    if not w or not h:
        return ""
    seen, rows = set(), []
    for e in elements:
        label = (e.get("label") or "").strip()
        if not label or len(label) > 40:
            continue
        if not (e.get("clickable") or "EditText" in (e.get("class") or "") or e.get("password")):
            continue
        nx = int(e["cx"] / w * 1000)
        ny = int(e["cy"] / h * 1000)
        key = (label.lower(), nx // 20, ny // 20)
        if key in seen:
            continue
        seen.add(key)
        kind = "field" if ("EditText" in (e.get("class") or "") or e.get("password")) else "button"
        rows.append((e["cy"], '%s "%s" (%d,%d)' % (kind, label, nx, ny)))
    rows.sort()
    return "\n".join("- %s" % r[1] for r in rows[:max_n])


def nearest_clickable(elements, px, py, max_dist, w=None, h=None):
    """Given the model's guessed pixel (px,py), return the real element to tap.
    Prefers the SMALLEST clickable box the point already sits inside (the most
    specific button); if the point is in dead space, falls back to the nearest
    clickable centre within max_dist; returns None if nothing is close enough
    (caller then taps the raw coordinate). When the screen size is known, the
    dead-space fallback ignores oversized boxes (whole-screen containers/overlays)
    so a stray point doesn't snap to a giant element's far centre."""
    def _too_big(e):
        return bool(w and h) and (e["w"] * e["h"] > 0.35 * w * h or e["h"] > 0.55 * h)

    inside = [e for e in elements if e["x1"] <= px <= e["x2"] and e["y1"] <= py <= e["y2"]]
    if inside:
        clickable = [e for e in inside if e["clickable"]]
        pool = clickable or inside
        return min(pool, key=lambda e: e["w"] * e["h"])

    best, best_d = None, None
    for e in elements:
        if not e["clickable"] or _too_big(e):
            continue
        d = ((e["cx"] - px) ** 2 + (e["cy"] - py) ** 2) ** 0.5
        if d <= max_dist and (best_d is None or d < best_d):
            best, best_d = e, d
    return best


def clickable_still_visible(target, elements):
    """True when the same labeled clickable remains in nearly the same place.
    This is useful after submitting a form: a screen can change because it shows an
    error or moves focus, but a still-visible submit button is evidence that pressing
    it again would be a loop rather than progress."""
    if not target:
        return False
    label = (target.get("label") or "").strip().lower().rstrip(" >").strip()
    if not label:
        return False
    tx1, ty1 = target.get("x1", 0), target.get("y1", 0)
    tx2, ty2 = target.get("x2", 0), target.get("y2", 0)
    target_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    for candidate in elements:
        candidate_label = (candidate.get("label") or "").strip().lower().rstrip(" >").strip()
        if not candidate.get("clickable") or candidate_label != label:
            continue
        cx1, cy1 = candidate.get("x1", 0), candidate.get("y1", 0)
        cx2, cy2 = candidate.get("x2", 0), candidate.get("y2", 0)
        overlap = max(0, min(tx2, cx2) - max(tx1, cx1)) * max(0, min(ty2, cy2) - max(ty1, cy1))
        candidate_area = max(1, (cx2 - cx1) * (cy2 - cy1))
        if overlap / min(target_area, candidate_area) >= 0.5:
            return True
        distance = ((candidate.get("cx", 0) - target.get("cx", 0)) ** 2
                    + (candidate.get("cy", 0) - target.get("cy", 0)) ** 2) ** 0.5
        if distance <= max(24, min(target.get("w", 0), target.get("h", 0)) * 0.75):
            return True
    return False


def point_hits_element(element, px, py, pad=24):
    """Whether a model coordinate would re-tap this element with a small aim margin."""
    return bool(element) and (
        element.get("x1", 0) - pad <= px <= element.get("x2", 0) + pad
        and element.get("y1", 0) - pad <= py <= element.get("y2", 0) + pad
    )


def keyboard_keys(elements, h):
    """Read the visible keyboard. Returns (keys, ctrl):
       keys  -> {'p': (x,y), 'a': (x,y), ' ': (x,y), '1': (x,y), ...}
       ctrl  -> {'SHIFT':(x,y), 'BKSP':(x,y), 'TOGGLE_SYM':(x,y),
                 'TOGGLE_ABC':(x,y), 'ENTER':(x,y)}
    Only looks in the bottom ~half of the screen where the keyboard lives."""
    keys, ctrl = {}, {}
    top = h * 0.48
    punct = "@#$_&-+()/*\"':;!?.,="
    for e in elements:
        if e["cy"] < top:
            continue
        lab = (e["desc"] or e["text"]).strip()
        low = lab.lower()
        if len(lab) == 1 and (lab.isalpha() or lab.isdigit() or lab in punct):
            keys.setdefault(low, (e["cx"], e["cy"]))
            continue
        if any(k in low for k in ("delete", "backspace")):
            ctrl.setdefault("BKSP", (e["cx"], e["cy"]))
        elif low == "space" or low.endswith(" space") or "spacebar" in low:
            keys.setdefault(" ", (e["cx"], e["cy"]))
        elif "shift" in low or "capital" in low or "caps" in low:
            ctrl.setdefault("SHIFT", (e["cx"], e["cy"]))
        elif low in ("?123", "123") or "switch to symbols" in low or "symbol" in low:
            ctrl.setdefault("TOGGLE_SYM", (e["cx"], e["cy"]))
        elif low == "abc" or "switch to letters" in low or "to alphabet" in low:
            ctrl.setdefault("TOGGLE_ABC", (e["cx"], e["cy"]))
        elif any(k in low for k in ("enter", "search", "done", "send", "next", "go")):
            ctrl.setdefault("ENTER", (e["cx"], e["cy"]))
        # voice / microphone keys are intentionally ignored
    return keys, ctrl


# --- deterministic-first: tap an obvious progression button straight from the
# accessibility tree, with NO vision-model call. Objective keyword -> the button
# labels that satisfy it. ---
_GOAL_BUTTONS = [
    (("log in", "login", "sign in", "signin", "log-in"), ("log in", "login", "sign in", "log-in", "signin")),
    (("continue",), ("continue",)),
    (("next",), ("next",)),
    (("submit",), ("submit",)),
    (("search",), ("search flights", "search hotels", "search flight", "search hotel", "search buses", "search trains")),
    (("get started", "getstarted"), ("get started",)),
    (("done", "finish"), ("done", "finish")),
    (("allow",), ("allow", "while using the app", "allow only while using")),
    (("accept", "agree"), ("accept", "agree", "accept all", "i agree")),
]


def deterministic_button(objective, elements, values, vi):
    """If the task's next step is clearly a labeled progression button that is on
    screen, return that element so the caller can tap it with NO model call
    (faster and exact). Conservative on purpose:
      - returns None while queued values remain (vi < len(values)) so we NEVER
        press submit/login/continue on a half-filled form;
      - only fires when EXACTLY ONE on-screen clickable matches a button the
        objective actually asks for (0 or several -> fall back to the model).
    Returns the element dict, or None."""
    if values and vi < len(values):
        return None
    obj = (objective or "").lower()
    is_flight_route = bool(re.search(r"\bflights?\s+from\b", obj))
    wanted = []
    for triggers, labels in _GOAL_BUTTONS:
        if any(t in obj for t in triggers):
            wanted.extend(labels)
    if not wanted:
        return None
    matches = []
    for e in elements:
        if not e.get("clickable") or not e.get("label"):
            continue
        lab = e["label"].strip().lower()
        if len(lab) > 24:
            continue
        norm = lab.rstrip(" >").strip()
        # A dated flight search still needs a calendar selection after its two
        # airport text fields are complete. Leave the final form submission to
        # the live grounder, which can see whether the date is selected.
        if is_flight_route and norm.startswith("search flight"):
            continue
        if any(norm == w or norm.startswith(w) for w in wanted):
            matches.append(e)
    if len(matches) == 1:
        return matches[0]
    return None

def clickable_still_visible(target, elements, tol_frac=0.04):
    """True if the SAME clickable element as `target` is still on screen -- used to tell a
    real success from a false one (e.g. after tapping 'Log in', if a clickable 'Log in' is
    still sitting in the same place, the submit did NOT take effect, so a values-based
    'finished' is a false pass). Matches by label when the target has one, else by position:
    a clickable element whose centre is within a small tolerance of the target's. Pure and
    defensive -- any bad/missing field returns False, so it can never throw inside the loop."""
    try:
        if not target:
            return False
        tlab = (target.get("label") or "").strip().lower()
        tcx, tcy = target.get("cx"), target.get("cy")
        xs = [e.get("x2", 0) for e in elements] or [1080]
        ys = [e.get("y2", 0) for e in elements] or [2400]
        tol = max(24, int(max(max(xs), max(ys)) * tol_frac))
        for e in elements:
            if not e.get("clickable"):
                continue
            elab = (e.get("label") or "").strip().lower()
            if tlab and elab:
                if elab == tlab:
                    return True
                continue
            ecx, ecy = e.get("cx"), e.get("cy")
            if None in (tcx, tcy, ecx, ecy):
                continue
            if abs(ecx - tcx) <= tol and abs(ecy - tcy) <= tol:
                return True
        return False
    except Exception:
        return False