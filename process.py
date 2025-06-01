from PIL import Image
import os
import numpy as np
imgs_path = "ver_mech/"
masks_path = "mechrxn_arrowmask/"
processed_path = "mechrxn_processed/"

imgs_files = os.listdir(imgs_path)
for img_file in imgs_files:
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        processed_file = os.path.join(processed_path, img_file)

        if os.path.exists(processed_file):
            # print(f"Skipping {img_file}, already processed.")
            continue

        mask_file = os.path.join(masks_path, img_file)
        mask_img = Image.open(mask_file)
        mask_data = mask_img.getdata()
        img_file_path = os.path.join(imgs_path, img_file)
        img = Image.open(img_file_path)
        img_data = img.load()

        for i in range(len(mask_data)):
            if mask_data[i] != (0, 0, 0):
                x = i % img.size[0]
                y = i // img.size[0]
                img_data[x, y] = (255, 255, 255)
        img.convert("L").save(processed_file)
