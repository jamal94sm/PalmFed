# inspect_xjtu.py
import os
from collections import defaultdict

data_root = "/home/pai-ng/Jamal/XJTU-UP"
VARIATIONS = [("iPhone", "Flash"), ("iPhone", "Nature"),
              ("huawei", "Flash"), ("huawei", "Nature")]
IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}

print(f"Scanning: {data_root}\n")

for device, condition in VARIATIONS:
    var_dir = os.path.join(data_root, device, condition)
    print(f"{'─'*50}")
    print(f"  {device}/{condition}  →  {var_dir}")
    if not os.path.isdir(var_dir):
        print("  NOT FOUND"); continue

    id_folders = sorted(os.listdir(var_dir))
    print(f"  Total subfolders: {len(id_folders)}")
    print(f"  First 10: {id_folders[:10]}")

    # count images per folder
    counts = {}
    for folder in id_folders:
        fdir = os.path.join(var_dir, folder)
        if not os.path.isdir(fdir): continue
        n = sum(1 for f in os.listdir(fdir)
                if os.path.splitext(f)[1].lower() in IMG_EXTS)
        counts[folder] = n

    if counts:
        vals = list(counts.values())
        print(f"  Images/folder: min={min(vals)}  max={max(vals)}  "
              f"mean={sum(vals)/len(vals):.1f}")
        # show folders with unexpected counts
        unusual = {k: v for k, v in counts.items() if v != 10}
        if unusual:
            print(f"  Folders with != 10 images ({len(unusual)}): "
                  f"{dict(list(unusual.items())[:10])}")
    print()

# now check overlap logic
print("="*50)
print("Overlap analysis")
data = defaultdict(lambda: defaultdict(list))
for device, condition in VARIATIONS:
    var_dir = os.path.join(data_root, device, condition)
    if not os.path.isdir(var_dir): continue
    for folder in sorted(os.listdir(var_dir)):
        fdir = os.path.join(var_dir, folder)
        if not os.path.isdir(fdir): continue
        parts = folder.split("_")
        print(f"  Sample folder name: {folder!r}  →  parts={parts}")
        break   # just show one example per domain

print()
print("Checking what parse_xjtu_domains currently keeps/skips:")
kept = defaultdict(set)
skipped = []
for device, condition in VARIATIONS:
    var_dir = os.path.join(data_root, device, condition)
    if not os.path.isdir(var_dir): continue
    for folder in sorted(os.listdir(var_dir)):
        fdir = os.path.join(var_dir, folder)
        if not os.path.isdir(fdir): continue
        parts = folder.split("_")
        if len(parts) < 2 or parts[0].upper() not in ("L", "R"):
            skipped.append((device, condition, folder, parts))
        else:
            kept[(device, condition)].add(folder)

print(f"  Kept per domain:    { {f'{d}/{c}': len(v) for (d,c),v in kept.items()} }")
print(f"  Skipped total: {len(skipped)}")
if skipped:
    print(f"  Skipped examples: {skipped[:5]}")