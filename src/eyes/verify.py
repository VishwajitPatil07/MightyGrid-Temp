import re
import time
from src.hands.adb_client import get_device_for_serial


def _d(serial):
    return get_device_for_serial(serial)


def foreground_app(serial):
    """Current foreground package (u2 app_current uses dumpsys under the hood)."""
    try:
        return _d(serial).app_current().get("package")
    except Exception:
        return None


def resolve_package(serial, name):
    """Map an app name to an installed package, best match first.
    An exact tail match (com.android.chrome) beats a loose one."""
    res = _d(serial).shell("pm list packages")
    out = res.output if hasattr(res, "output") else str(res)
    pkgs = [ln.split(":", 1)[1].strip() for ln in out.splitlines() if ":" in ln]
    tokens = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) > 2]
    if not tokens:
        return None
    best, best_score = None, None
    for pkg in pkgs:
        low = pkg.lower()
        tail = low.rsplit(".", 1)[-1]
        score, matched = 0, False
        for tok in tokens:
            if tail == tok:
                score += 100; matched = True
            elif tail.startswith(tok):
                score += 60; matched = True
            elif tok in tail:
                score += 30; matched = True
            elif tok in low:
                score += 10; matched = True
        if not matched:
            continue
        if low.startswith("com.example"):
            score -= 50
        if low.startswith(("com.android.", "com.google.", "com.oplus.", "com.coloros.")):
            score += 5
        if best_score is None or score > best_score:
            best, best_score = pkg, score
    return best


def wait_foreground(serial, name, timeout=3.0):
    """Independent proof the named app is in front. Returns:
    (True, detail) verified / (False, detail) not / (None, detail) unresolvable."""
    pkg = resolve_package(serial, name)
    if pkg is None:
        return None, "no package resolved for '%s'" % name
    deadline = time.time() + timeout
    fg = None
    while time.time() < deadline:
        fg = foreground_app(serial)
        if fg == pkg:
            return True, "foreground is %s" % pkg
        time.sleep(0.4)
    return False, "expected %s, foreground is %s" % (pkg, fg)


def element_present(serial, text):
    """Structural check: is an element with matching text/desc on screen?"""
    try:
        xml = _d(serial).dump_hierarchy()
    except Exception:
        return False
    needle = text.lower()
    for attr in ('text="', 'content-desc="'):
        for part in xml.split(attr)[1:]:
            if needle in part.split('"', 1)[0].lower():
                return True
    return False


_STOP = {
    "the", "is", "are", "screen", "field", "shows", "showing", "open", "opens",
    "opened", "visible", "page", "with", "and", "for", "into", "this", "that",
    "app", "appears", "a", "an", "of", "to", "on", "in", "now", "has", "been",
    "current", "correct", "search", "result", "results", "view", "displayed",
    # generic UI chrome -- too common to prove anything, so never a success match
    "button", "buttons", "highlighted", "clickable", "enabled", "input", "icon",
    "option", "options", "bar", "menu", "tab", "toggle", "checkbox", "label",
    "text", "box", "list", "item", "items", "available", "loaded", "ready",
}


def _keywords(done_when):
    """Pull the checkable tokens out of a natural-language done_when: quoted
    phrases first, then any content word long enough to look up on screen."""
    kws = []
    for a, b in re.findall(r'"([^"]+)"|\'([^\']+)\'', done_when):
        kws.append(a or b)
    for w in re.findall(r"[A-Za-z0-9@._+\-]+", done_when):
        if len(w) > 3 and w.lower() not in _STOP:
            kws.append(w)
    seen, out = set(), []
    for k in kws:
        k = k.strip()
        if k and k.lower() not in seen:
            seen.add(k.lower())
            out.append(k)
    return out[:5]


def verify_step(serial, done_when, before_xml, settle=1.2):
    """Decide whether a sub-goal's done_when is met. Returns (ok, reason).
    ok is True only on positive evidence -- an app in the foreground, or a
    keyword from done_when found on screen. A bare screen-change is accepted
    but flagged 'weak' so a false pass is visible in the log, never silent."""
    time.sleep(settle)
    d = _d(serial)
    try:
        after = d.dump_hierarchy()
    except Exception:
        after = ""

    dw = (done_when or "").lower()

    if any(k in dw for k in ("foreground", "is open", "opens", "opened", "launches")):
        for cand in re.findall(r"[A-Z][A-Za-z0-9]+", done_when):
            ok, detail = wait_foreground(serial, cand, timeout=2.0)
            if ok:
                return True, "foreground: %s" % detail

    for kw in _keywords(done_when):
        if element_present(serial, kw):
            return True, "found '%s' on screen" % kw

    if before_xml and after and after != before_xml:
        return True, "screen changed (weak)"

    return False, "no evidence done_when met"


def check_success(serial, success_when):
    """Loose, evidence-based check that the objective looks done: is any keyword
    from success_when present on the current screen? Returns (ok, reason).
    Used to confirm (or doubt) the grounder's finished() rather than trust it."""
    if not success_when:
        return False, "no success condition given"
    for kw in _keywords(success_when):
        if element_present(serial, kw):
            return True, "found '%s' on screen" % kw
    return False, "no success evidence on screen"