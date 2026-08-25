# Latency breakdown probe -- finds WHERE the ~4-5s per decision goes, so we fix the
# real part instead of guessing. Times: (a) screenshot capture on the phone,
# (b) JPEG+base64 encode on the PC, (c) the full HTTP round-trip to the model server.
# Then compare (c) against the vLLM inference time shown in the RunPod Logs tab:
#   round-trip (c)  -  vLLM inference (from logs)  =  network + serverless overhead.
#
# Run:  python scripts\probe_latency.py     (uses the same MG_LLM_* env vars)
import os, time, base64, io, statistics
import requests
import uiautomator2 as u2
from PIL import Image

SERIAL = os.environ.get("MG_SERIAL", "S8V8UGEAUW4LSCLV")
URL = os.environ.get("MG_LLM_URL", "http://localhost:1234/v1/chat/completions")
MODEL = os.environ.get("MG_LLM_MODEL", "ByteDance-Seed/UI-TARS-7B-DPO")
KEY = os.environ.get("MG_LLM_KEY", "")
WIDTH = int(os.environ.get("MG_IMG_WIDTH", "336"))
N = int(os.environ.get("MG_PROBE_N", "5"))

def hdrs():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer %s" % KEY
    return h

print("endpoint:", URL)
print("model:", MODEL, "| image width:", WIDTH, "| samples:", N)
d = u2.connect(SERIAL)

# (a) screenshot capture
cap = []
raw = None
for _ in range(N):
    t = time.time()
    raw = d.screenshot(format="pillow")
    cap.append((time.time() - t) * 1000)

# (b) encode (resize to WIDTH, JPEG, base64) -- what actually gets sent
enc = []
data_url = None
for _ in range(N):
    t = time.time()
    im = raw.convert("RGB")
    w, h = im.size
    im = im.resize((WIDTH, int(h * WIDTH / w)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_url = "data:image/jpeg;base64," + b64
    enc.append((time.time() - t) * 1000)
payload_kb = len(data_url) / 1024.0

# (c) full HTTP round-trip (image + short prompt, tiny output)
body = {
    "model": MODEL,
    "max_tokens": 20,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is the next action? Reply with one short line."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}],
}
rt = []
first_err = None
for i in range(N):
    t = time.time()
    try:
        r = requests.post(URL, json=body, headers=hdrs(), timeout=180)
        ms = (time.time() - t) * 1000
        rt.append(ms)
        if r.status_code != 200 and first_err is None:
            first_err = "HTTP %s: %s" % (r.status_code, r.text[:200])
    except Exception as e:
        if first_err is None:
            first_err = str(e)
    print("  request %d/%d: %.0f ms" % (i + 1, N, rt[-1] if rt else -1))

def line(name, xs):
    if not xs:
        print("  %-26s (no data)" % name); return
    print("  %-26s min %.0f  median %.0f  max %.0f  ms" % (name, min(xs), statistics.median(xs), max(xs)))

print("\n===== LATENCY BREAKDOWN =====")
line("(a) screenshot capture", cap)
line("(b) encode+base64", enc)
print("  payload size: %.1f KB" % payload_kb)
line("(c) full round-trip", rt)
if rt and cap and enc:
    server_side = statistics.median(rt)
    client_side = statistics.median(cap) + statistics.median(enc)
    print("\n  client-side (a+b): ~%.0f ms   (runs on your PC, no GPU/network)" % client_side)
    print("  round-trip (c):    ~%.0f ms   (network + serverless delay + GPU inference)" % server_side)
    print("\n  NEXT: open the RunPod Logs tab, find these requests' vLLM timing, and subtract:")
    print("        round-trip (~%.0f) - vLLM_inference(from logs) = network+serverless overhead." % server_side)
if first_err:
    print("\n  NOTE first error:", first_err)