import sys, time
sys.path.append('.')
from src.hands.adb_client import get_device, curved_swipe, get_device_size
d = get_device()
w, h = get_device_size()
# EXPLICIT finger-moves-UP: start low (80% down), end high (25% down)
print("Swiping finger UPWARD: bottom -> top")
curved_swipe(int(w*0.5), int(h*0.80), int(w*0.5), int(h*0.25), dur=0.10, bow_frac=0.05, steps=10)
time.sleep(1.2)
xml = d.dump_hierarchy()
has_search = 'search' in xml.lower()
has_kbd = "'q'" in xml.lower() or 'keyboard' in xml.lower()
print("Drawer/search opened:", has_search, "| keyboard up:", has_kbd)
