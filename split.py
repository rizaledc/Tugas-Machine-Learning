import os
import shutil
import random
from pathlib import Path

# Set path dasar
SOURCE_DIR = "DisasterModel/Cyclone_Wildfire_Flood_Earthquake_Dataset"
TARGET_DIR = SOURCE_DIR  # Simpan hasil di folder yang sama

CLASSES = ["Cyclone", "Earthquake", "Flood", "Wildfire"]
SPLITS = ["train", "validation", "test"]
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # 70% train, 20% val, 10% test

def make_dirs():
    for split in SPLITS:
        for cls in CLASSES:
            path = os.path.join(TARGET_DIR, split, cls)
            os.makedirs(path, exist_ok=True)

def split_and_copy():
    for cls in CLASSES:
        class_dir = os.path.join(SOURCE_DIR, cls)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIOS[0])
        n_val = int(n_total * SPLIT_RATIOS[1])

        splits = {
            "train": images[:n_train],
            "validation": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, split_files in splits.items():
            for file_name in split_files:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(TARGET_DIR, split_name, cls, file_name)
                shutil.copy2(src, dst)

if __name__ == "__main__":
    make_dirs()
    split_and_copy()
    print("✅ Dataset berhasil dibagi ke folder train, validation, dan test.")
