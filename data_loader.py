# data_loader.py
import tensorflow as tf
import logging
import config # Impor konfigurasi

# Setup logging dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_single_dataset(directory, image_size, batch_size, label_mode, dataset_name, shuffle_data=False):
    """Fungsi helper untuk memuat satu bagian dataset (train/val/test)."""
    if not tf.io.gfile.exists(directory) or not tf.io.gfile.listdir(directory):
        logging.error(f"Direktori {dataset_name} kosong atau tidak ditemukan: {directory}")
        logging.error("Pastikan skrip 'split_data.py' sudah dijalankan dan path di 'config.py' benar.")
        return None
    try:
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels='inferred', # Otomatis mengambil label dari nama subdirektori
            label_mode=label_mode, # 'categorical' untuk one-hot encoding
            image_size=image_size,
            interpolation='nearest', # Atau 'bilinear', sesuaikan kebutuhan
            batch_size=batch_size,
            shuffle=shuffle_data, # Hanya shuffle data training saat loading awal
            seed=config.RANDOM_SEED if shuffle_data else None
        )
        logging.info(f"Dataset {dataset_name} berhasil dimuat dari: {directory}")
        return dataset # Objek ini memiliki atribut class_names
    except Exception as e:
        logging.error(f"Gagal memuat dataset {dataset_name} dari {directory}: {e}")
        return None

def _preprocess_dataset(dataset, dataset_name):
    """Menerapkan normalisasi dan prefetching pada dataset."""
    if dataset is None:
        return None

    # Normalisasi gambar (skala piksel 0-1)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Optimasi performa dengan cache dan prefetch
    dataset = dataset.cache()
    if dataset_name == "train": # Shuffle data training lebih lanjut setelah caching
        dataset = dataset.shuffle(buffer_size=1000, seed=config.RANDOM_SEED, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    logging.info(f"Pra-pemrosesan (normalisasi & prefetch) diterapkan pada dataset {dataset_name}.")
    return dataset

def get_data_generators():
    """Memuat dan melakukan pra-pemrosesan pada dataset train, validation, dan test.
    Juga mengembalikan class_names dari dataset training."""
    image_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
    label_mode = 'categorical' # Sesuai dengan loss 'categorical_crossentropy'
    class_names_list = None

    logging.info("Memuat dataset training...")
    # Muat dataset mentah terlebih dahulu untuk mendapatkan class_names
    train_ds_raw = _load_single_dataset(
        config.TRAIN_DIR, image_size, config.BATCH_SIZE, label_mode, "train_raw", shuffle_data=True
    )
    if train_ds_raw:
        class_names_list = train_ds_raw.class_names # Ambil class_names di sini
        logging.info(f"Nama kelas yang terdeteksi dari dataset training: {class_names_list}")
        train_ds = _preprocess_dataset(train_ds_raw, "train")
    else:
        train_ds = None

    logging.info("Memuat dataset validasi...")
    val_ds_raw = _load_single_dataset(
        config.VALIDATION_DIR, image_size, config.BATCH_SIZE, label_mode, "validation_raw", shuffle_data=False
    )
    val_ds = _preprocess_dataset(val_ds_raw, "validation")


    logging.info("Memuat dataset test...")
    test_ds_raw = _load_single_dataset(
        config.TEST_DIR, image_size, config.BATCH_SIZE, label_mode, "test_raw", shuffle_data=False
    )
    test_ds = _preprocess_dataset(test_ds_raw, "test")


    if train_ds and val_ds: # Test_ds bersifat opsional untuk tahap training utama
        logging.info("Semua dataset yang diperlukan berhasil dimuat dan diproses.")
    else:
        logging.warning("Gagal memuat satu atau lebih dataset. Periksa log di atas.")

    return train_ds, val_ds, test_ds, class_names_list # Kembalikan juga class_names

if __name__ == "__main__":
    logging.info("Menguji coba pemuat data...")
    # Tangkap class_names yang dikembalikan
    train_dataset, val_dataset, test_dataset, loaded_class_names = get_data_generators()

    if train_dataset and loaded_class_names: # Periksa juga loaded_class_names
        logging.info(f"Nama kelas dari dataset training: {loaded_class_names}") # Gunakan variabel yang sudah diambil
        for images, labels in train_dataset.take(1): # Ambil satu batch untuk inspeksi
            logging.info(f"Bentuk batch gambar training: {images.shape}")
            logging.info(f"Bentuk batch label training: {labels.shape}")
            logging.info(f"Contoh label training (one-hot): {labels[0]}")
            break
    elif train_dataset:
        logging.warning("Dataset training dimuat, tetapi nama kelas tidak dapat diambil.")
        
    if val_dataset:
        logging.info(f"Dataset validasi berhasil dimuat.")
    if test_dataset:
        logging.info(f"Dataset test berhasil dimuat.")