import time, base64, io, requests
from PIL import Image
import uiautomator2 as u2
d = u2.connect('S8V8UGEAUW4LSCLV'); d.screenshot('t.png')
raw = Image.open('t.png').convert('RGB'); print('native', raw.size)
for w in (512, 336, 256, 192):
    h = int(raw.size[1]*w/raw.size[0])
    im = raw.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format='JPEG', quality=85)
    b = base64.b64encode(buf.getvalue()).decode()
    t = time.time()
    r = requests.post('http://localhost:1234/v1/chat/completions', json={'model':'ui-tars-7b-dpo','messages':[{'role':'user','content':[{'type':'text','text':'What app? One word.'},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+b}}]}],'max_tokens':20}, timeout=600)
    ans = r.json()['choices'][0]['message']['content'].strip().replace(chr(10),' ')[:25]
    print('width %4d -> %5.1f s  (%s)' % (w, time.time()-t, ans))
