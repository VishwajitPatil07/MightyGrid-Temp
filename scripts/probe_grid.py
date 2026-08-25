import sys
sys.path.append('.')
from src.hands.adb_client import get_device
d = get_device()
xml = d.dump_hierarchy()
labels = []
for node in xml.split("<node")[1:]:
    for attr in ('text="', 'content-desc="'):
        if attr in node:
            v = node.split(attr,1)[1].split('"',1)[0].strip()
            if v:
                labels.append(v); break
print("TOTAL labels:", len(labels))
print("keyboard up (q key present):", "q" in labels)
print("app-like labels (first 40):")
for l in labels[:40]:
    print("  ", repr(l))
