from PIL import Image
import numpy as np
import glob
import os

width, height = 512, 341
try:
    raw_files = sorted(glob.glob("output/raw/*.raw"))
    for raw_path in raw_files:
        name = os.path.splitext(os.path.basename(raw_path))[0]
        raw_data = np.fromfile(raw_path, dtype=np.uint8).reshape((height, width))
        img = Image.fromarray(raw_data)
        img.save(f"output/{name}.png")
        img.show()
    print("success")
except Exception as e:
    print("[Fail]", e)
