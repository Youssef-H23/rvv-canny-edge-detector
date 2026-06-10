from PIL import Image
import numpy as np
import os

image_path = "../images/car_original.png"
raw_path = "../images/car.raw"

try:
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    img_array.tofile(raw_path)
    print(f"successfully converted and saved as raw. Extracted image size: {img.size[0]}x{img.size[1]}")
except Exception as e:
    print("Failed to convert image to raw format:", e)
