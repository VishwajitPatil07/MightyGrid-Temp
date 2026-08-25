# Times a UI-TARS call at several image widths to prove whether the screenshot
# (prefill/vision tokens) is the ~60s bottleneck. Run from the project root:
#   python probe_image_latency.py
import time, base64, io, requests
from PIL import Image
import uiautomator2 as u2

URL = "http://localhost:1234/v1/chat/completions"
MODEL = "ui-tars-7b-dpo"

d = u2.connect("S8V8UGEAUW4LSCLV")
d.screenshot("t.png")
raw = Image.open("t.png").convert("RGB")
print("native screenshot:", raw.size)

def enc(w):
    h = int(raw.size[1] * (w / raw.size[0]))
    im = raw.resize((w, h), Image.Resampling.LANCZOS)
    b = io.BytesIO(); im.save(b, format="JPEG", quality=85)
    return base64.b64encode(b.getvalue()).decode()

for w in (512, 336, 256, 192):
    b = enc(w)
    t = time.time()
    r = requests.post(URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What app is on screen? One word."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,%s" % b}},
        ]}],
        "max_tokens": 20,
    }, timeout=600)
    dt = time.time() - t
    ans = r.json()["choices"][0]["message"]["content"].strip().replace("\n", " ")[:30]
    print("width %4d px  ->  %5.1f s   (answer: %s)" % (w, dt, ans))