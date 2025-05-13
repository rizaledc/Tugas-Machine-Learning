# split_data.py
import os
import shutil
import random
import logging
import config # Impor konfigurasi

# Setup logging dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_target_directories():
    """Membuat direktori train, validation, dan test untuk setiap kelas."""
    logging.info(f"Membuat direktori target di: {config.SPLIT_DATASET_BASE_DIR}")
    for split_name in ["train", "validation", "test"]:
        split_folder_path = getattr(config, f"{split_name.upper()}_DIR") # contoh: config.TRAIN_DIR
        for class_name in config.CLASSES:
            path = os.path.join(split_folder_path, class_name)
            os.makedirs(path, exist_ok=True)
    logging.info("Direktori target berhasil dibuat/diverifikasi.")

def split_and_copy_files():
    """Membagi dan menyalin file gambar dari direktori sumber ke direktori target."""
    if not os.path.exists(config.RAW_DATASET_BASE_DIR):
        logging.error(f"Direktori dataset mentah tidak ditemukan: {config.RAW_DATASET_BASE_DIR}")
        logging.error("Pastikan path di config.py sudah benar dan dataset tersedia.")
        return False

    for class_name in config.CLASSES:
        source_class_dir = os.path.join(config.RAW_DATASET_BASE_DIR, class_name)
        logging.info(f"Memproses kelas: {class_name} dari {source_class_dir}")

        if not os.path.isdir(source_class_dir):
            logging.warning(f"Direktori untuk kelas '{class_name}' tidak ditemukan di {source_class_dir}. Melanjutkan ke kelas berikutnya.")
            continue

        try:
            images = [f for f in os.listdir(source_class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        except Exception as e:
            logging.error(f"Gagal membaca file dari {source_class_dir}: {e}")
            continue

        if not images:
            logging.warning(f"Tidak ada gambar ditemukan di {source_class_dir} untuk kelas '{class_name}'.")
            continue

        random.seed(config.RANDOM_SEED) # Untuk hasil shuffle yang konsisten
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * config.SPLIT_RATIOS[0])
        n_val = int(n_total * config.SPLIT_RATIOS[1])
        # n_test adalah sisanya

        split_definitions = {
            "train": images[:n_train],
            "validation": images[n_train : n_train + n_val],
            "test": images[n_train + n_val:]
        }

        logging.info(f"Kelas '{class_name}': Total={n_total}, Train={len(split_definitions['train'])}, Validation={len(split_definitions['validation'])}, Test={len(split_definitions['test'])}")

        for split_name, split_files in split_definitions.items():
            target_split_dir_base = getattr(config, f"{split_name.upper()}_DIR")
            destination_dir = os.path.join(target_split_dir_base, class_name)

            for file_name in split_files:
                source_file_path = os.path.join(source_class_dir, file_name)
                destination_file_path = os.path.join(destination_dir, file_name)
                try:
                    shutil.copy2(source_file_path, destination_file_path)
                except Exception as e:
                    logging.error(f"Gagal menyalin {source_file_path} ke {destination_file_path}: {e}")
        logging.info(f"File untuk kelas '{class_name}' berhasil disalin ke masing-masing folder split.")
    return True

if __name__ == "__main__":
    logging.info("Memulai proses pembagian dataset...")
    # Pastikan direktori dasar untuk dataset yang sudah di-split ada
    os.makedirs(config.SPLIT_DATASET_BASE_DIR, exist_ok=True)
    create_target_directories()
    if split_and_copy_files():
        logging.info("✅ Proses pembagian dataset ke folder train, validation, dan test selesai.")
    else:
        logging.error("❌ Proses pembagian dataset gagal. Periksa log di atas.")