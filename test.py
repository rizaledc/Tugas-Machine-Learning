import tensorflow as tf
import logging
import os
import config

# Setup logging dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memuat dataset test
def load_test_data():
    logging.info("Memuat dataset test...")
    # Menggunakan image_dataset_from_directory untuk memuat dataset dari folder test
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.TEST_DIR,
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        label_mode='categorical'  # Untuk one-hot encoding
    )
    return test_ds

# Mengevaluasi Model
def evaluate_model(model, test_ds):
    logging.info("Mengevaluasi model pada dataset test...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    logging.info(f"Hasil Evaluasi Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

# Muat model terbaik berdasarkan nama file yang sudah disimpan
def load_best_model(model_name):
    logging.info(f"Memuat model terbaik dari: {model_name}")
    model = tf.keras.models.load_model(model_name)
    return model

if __name__ == "__main__":
    # Nama file model yang terbaik (epoch ke-11)
    best_model_file = os.path.join(config.MODEL_SAVE_DIR, 'best_model_epoch_11_val_acc_0.874.keras')

    # Muat model terbaik (epoch ke-11)
    model = load_best_model(best_model_file)
    logging.info("Model berhasil dimuat.")

    # Muat data test
    test_ds = load_test_data()

    # Evaluasi model pada data test
    evaluate_model(model, test_ds)
