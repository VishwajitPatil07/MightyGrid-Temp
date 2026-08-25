import sys
sys.path.append('.')
from src.hands.adb_client import get_device, execute_scroll
import time
d = get_device()
execute_scroll("up")          # try to open the drawer
time.sleep(1.2)
xml = d.dump_hierarchy()
labels = []
for node in xml.split("<node")[1:]:
    for attr in ('text="', 'content-desc="'):
        if attr in node:
            v = node.split(attr,1)[1].split('"',1)[0].strip()
            if v:
                labels.append(v); break
print("After swipe-up, %d labels. Search box present:" % len(labels),
      any("search" in l.lower() for l in labels))
print("Chrome present:", any("chrome" == l.lower() for l in labels))
print("Sample:", labels[:25])
