# model_builder.py
import tensorflow as tf
import logging
import config # Impor konfigurasi

# Setup logging dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_cnn_model():
    """
    Membuat dan mengompilasi model Convolutional Neural Network (CNN)
    untuk klasifikasi gambar bencana.
    """
    model = tf.keras.Sequential([
        # Lapisan Input (implisit dari input_shape di lapisan Conv2D pertama)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=config.INPUT_SHAPE, name="conv2d_1"),
        tf.keras.layers.MaxPooling2D(2, 2, name="maxpool_1"),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv2d_2"),
        tf.keras.layers.MaxPooling2D(2, 2, name="maxpool_2"),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv2d_3"),
        tf.keras.layers.MaxPooling2D(2, 2, name="maxpool_3"),

        # Anda bisa menambahkan lebih banyak lapisan Conv/Pool jika diperlukan
        # tf.keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv2d_4"),
        # tf.keras.layers.MaxPooling2D(2, 2, name="maxpool_4"),

        tf.keras.layers.Flatten(name="flatten_layer"),
        tf.keras.layers.Dense(128, activation='relu', name="dense_layer_1"), # Jumlah unit bisa disesuaikan
        tf.keras.layers.Dropout(0.5, name="dropout_layer"), # Dropout untuk regularisasi
        tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax', name="output_layer") # Lapisan output
    ])

    # Kompilasi model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', # Karena label one-hot dan klasifikasi multi-kelas
        metrics=['accuracy']
    )

    logging.info("Model CNN berhasil dibuat dan dikompilasi.")
    # Cetak ringkasan model ke log
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    logging.info(f"Ringkasan Model:\n{model_summary}")

    return model

if __name__ == "__main__":
    logging.info("Menguji coba pembuatan model...")
    cnn_model = create_cnn_model()
    if cnn_model:
        logging.info("Instance model berhasil dibuat.")
    else:
        logging.error("Gagal membuat instance model.")