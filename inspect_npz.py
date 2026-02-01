# inspect_npz.py
import glob, numpy as np, os

DATA_DIR = "data"
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
if not files:
    print("No .npz files found in", DATA_DIR)
    raise SystemExit(0)

for fn in files:
    try:
        with np.load(fn, allow_pickle=True) as dd:
            print(os.path.basename(fn), "-> keys:", dd.files)
            for k in dd.files:
                try:
                    arr = dd[k]
                    print("   ", k, "shape:", getattr(arr, "shape", type(arr)))
                except Exception as e:
                    print("   ", k, "type:", type(dd[k]), " (error reading shape:", e, ")")
    except Exception as e:
        print("Failed to open", fn, ":", e)

print("Done inspecting", len(files), "files.")
