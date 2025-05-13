# Module for splitting the raw dataset into train, validation, and test sets.

import os
import shutil
import random
import logging
import config # Import configuration

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_target_directories():
    """Creates train, validation, and test directories for each class."""
    logging.info(f"Creating target directories under: {config.SPLIT_DATASET_BASE_DIR}")
    for split_name in ["train", "validation", "test"]:
        split_folder_path = getattr(config, f"{split_name.upper()}_DIR")
        for class_name in config.CLASSES:
            path = os.path.join(split_folder_path, class_name)
            os.makedirs(path, exist_ok=True)
    logging.info("Target directories created/verified.")

def split_and_copy_files():
    """Splits image files from source directories and copies them to target directories."""
    if not os.path.exists(config.RAW_DATASET_BASE_DIR):
        logging.error(f"Raw dataset directory not found: {config.RAW_DATASET_BASE_DIR}")
        logging.error("Ensure the path in config.py is correct and the dataset is available.")
        return False

    for class_name in config.CLASSES:
        source_class_dir = os.path.join(config.RAW_DATASET_BASE_DIR, class_name)
        logging.info(f"Processing class: {class_name} from {source_class_dir}")

        if not os.path.isdir(source_class_dir):
            logging.warning(f"Directory for class '{class_name}' not found at {source_class_dir}. Skipping.")
            continue

        try:
            images = [f for f in os.listdir(source_class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        except Exception as e:
            logging.error(f"Failed to read files from {source_class_dir}: {e}")
            continue

        if not images:
            logging.warning(f"No images found in {source_class_dir} for class '{class_name}'.")
            continue

        random.seed(config.RANDOM_SEED) # Ensure reproducible splits
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * config.SPLIT_RATIOS[0])
        n_val = int(n_total * config.SPLIT_RATIOS[1])
        # Test size is the remainder

        split_definitions = {
            "train": images[:n_train],
            "validation": images[n_train : n_train + n_val],
            "test": images[n_train + n_val:]
        }

        logging.info(f"Class '{class_name}': Total={n_total}, Train={len(split_definitions['train'])}, Validation={len(split_definitions['validation'])}, Test={len(split_definitions['test'])}")

        for split_name, split_files in split_definitions.items():
            target_split_dir_base = getattr(config, f"{split_name.upper()}_DIR")
            destination_dir = os.path.join(target_split_dir_base, class_name)

            for file_name in split_files:
                source_file_path = os.path.join(source_class_dir, file_name)
                destination_file_path = os.path.join(destination_dir, file_name)
                try:
                    shutil.copy2(source_file_path, destination_file_path)
                except Exception as e:
                    logging.error(f"Failed to copy {source_file_path} to {destination_file_path}: {e}")
        logging.info(f"Files for class '{class_name}' successfully copied to split folders.")
    return True

if __name__ == "__main__":
    logging.info("Starting dataset splitting process...")
    # Ensure the base directory for the split dataset exists
    os.makedirs(config.SPLIT_DATASET_BASE_DIR, exist_ok=True)
    
    create_target_directories()
    
    if split_and_copy_files():
        logging.info("✅ Dataset splitting process completed.")
    else:
        logging.error("❌ Dataset splitting process failed. Check logs.")