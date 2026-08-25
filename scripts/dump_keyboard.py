# Diagnostic ONLY -- no changes to the agent. Shows how this Gboard exposes its
# keys so we can fix digit typing. Paste the ENTIRE output back.
import re, time
import uiautomator2 as u2

SERIAL = "S8V8UGEAUW4LSCLV"
d = u2.connect(SERIAL)
w, h = d.window_size()
print("== screen:", w, h, "==")
print("== keyboard IME:", d.shell("settings get secure default_input_method").output.strip(), "==")

def attr(s, n):
    m = re.search(n + r'="([^"]*)"', s)
    return m.group(1) if m else ""

def dump(tag):
    xml = d.dump_hierarchy()
    print("\n===== %s =====" % tag)
    found = []
    for m in re.finditer(r'<node[^>]*?/?>', xml):
        s = m.group(0)
        b = re.search(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', s)
        if not b:
            continue
        x1, y1, x2, y2 = map(int, b.groups())
        cy = (y1 + y2) // 2
        if cy < h * 0.45:            # keyboard lives in the bottom half
            continue
        txt, desc = attr(s, "text").strip(), attr(s, "content-desc").strip()
        label = txt or desc
        if not label or len(label) > 20:
            continue
        found.append((cy, (x1 + x2) // 2, txt, desc))
    # print grouped into rows by y
    found.sort()
    rowy = None
    line = []
    for cy, cx, txt, desc in found:
        if rowy is None or abs(cy - rowy) > 45:
            if line:
                print("  " + "   ".join(line))
            line = []
            rowy = cy
        line.append("[x%d t=%r d=%r]" % (cx, txt, desc))
    if line:
        print("  " + "   ".join(line))

# open Instagram, tap the username field so Gboard shows
print("\nOpening Instagram login field...")
d.app_start("com.instagram.android")
time.sleep(4)
# tap the first editable field if we can find one
xml = d.dump_hierarchy()
m = re.search(r'<node[^>]*class="android.widget.EditText"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', xml)
if m:
    x1, y1, x2, y2 = map(int, m.groups())
    d.click((x1 + x2) // 2, (y1 + y2) // 2)
    time.sleep(2)
    print("tapped a text field")
else:
    print("!! no EditText found -- tap a username/search box MANUALLY now")
    time.sleep(5)

dump("LETTERS PAGE (as shown)")

# find and tap the ?123 / symbols toggle
xml = d.dump_hierarchy()
tog = None
for m in re.finditer(r'<node[^>]*?/?>', xml):
    s = m.group(0)
    lab = (attr(s, "text") or attr(s, "content-desc")).lower()
    if any(k in lab for k in ("123", "symbol", "number", "digit")):
        b = re.search(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', s)
        if b:
            x1, y1, x2, y2 = map(int, b.groups())
            tog = ((x1 + x2) // 2, (y1 + y2) // 2, attr(s, "text"), attr(s, "content-desc"))
            break
if tog:
    print("\n>> Found symbols toggle: text=%r desc=%r at (%d,%d) -- tapping it" % (tog[2], tog[3], tog[0], tog[1]))
    d.click(tog[0], tog[1])
    time.sleep(1.5)
    dump("NUMBER/SYMBOL PAGE (after tapping toggle)")
else:
    print("\n!! Could NOT find a ?123/symbols toggle in the tree. The letters dump above is what we have.")
print("\n== DONE -- copy everything from '== screen' down and paste it back ==")