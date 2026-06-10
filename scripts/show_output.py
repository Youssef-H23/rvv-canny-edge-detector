from PIL import Image
import numpy as np
names = [
    "output_blurred"
]
width, height = 512, 341
try:
    for name in names:
        raw_path = f"../output/{name}.raw"
        raw_data = np.fromfile(raw_path, dtype=np.uint8).reshape((height, width))
        img = Image.fromarray(raw_data)
        img.save(f"../output/{name}.png")
        img.show()
    print("success")
except Exception as e:
    print("[Fail]", e)
