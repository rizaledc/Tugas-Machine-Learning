# train_model.py
import tensorflow as tf
import logging
import os
import matplotlib.pyplot as plt # Untuk plotting (opsional)

# Impor modul lokal
import config
from data_loader import get_data_generators
from model_builder import create_cnn_model

# Setup logging dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_training_history(history, save_path_base):
    """Membuat plot akurasi dan loss dari history pelatihan."""
    try:
        # Plot Akurasi
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        acc_plot_path = os.path.join(save_path_base, "training_accuracy_plot.png")
        plt.savefig(acc_plot_path)
        logging.info(f"Plot akurasi disimpan di: {acc_plot_path}")
        plt.close() # Tutup plot agar tidak ditampilkan jika dijalankan di server

        # Plot Loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(save_path_base, "training_loss_plot.png")
        plt.savefig(loss_plot_path)
        logging.info(f"Plot loss disimpan di: {loss_plot_path}")
        plt.close()
    except Exception as e:
        logging.warning(f"Gagal membuat plot histori pelatihan: {e}")


# train_model.py

# ... (impor dan kode lainnya) ...

def train():
    """Fungsi utama untuk melatih model deteksi bencana."""
    logging.info("Memulai proses pelatihan model...")

    # 1. Muat Dataset
    logging.info("Memuat dataset...")
    # PERBAIKI BARIS INI:
    # train_ds, val_ds, test_ds = get_data_generators() # BARIS LAMA YANG SALAH
    train_ds, val_ds, test_ds, class_names = get_data_generators() # BARIS BARU YANG BENAR

    if not train_ds or not val_ds:
        logging.error("Dataset training atau validasi tidak berhasil dimuat. Proses pelatihan dihentikan.")
        return
    
    # Jika Anda ingin memastikan class_names juga termuat:
    if not class_names:
        logging.warning("Nama kelas tidak berhasil dimuat dari data_loader. Mungkin akan ada masalah jika nama kelas diperlukan nanti.")
    else:
        logging.info(f"Nama kelas yang berhasil dimuat: {class_names}")


    # 2. Buat Model
    # ... (sisa kode fungsi train() tetap sama) ...

    # 2. Buat Model
    logging.info("Membuat model CNN...")
    model = create_cnn_model()
    if not model:
        logging.error("Gagal membuat model. Proses pelatihan dihentikan.")
        return

    # 3. Definisikan Callbacks (Opsional tapi sangat direkomendasikan)
    callbacks = []

    # EarlyStopping: Menghentikan pelatihan jika tidak ada peningkatan
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',       # Metrik yang dipantau
        patience=5,               # Jumlah epoch tanpa peningkatan sebelum berhenti
        verbose=1,
        restore_best_weights=True # Kembalikan bobot terbaik saat berhenti
    )
    callbacks.append(early_stopping_cb)

    # ModelCheckpoint: Menyimpan model terbaik selama pelatihan
    # Pastikan direktori penyimpanan model ada (sudah di config.py)
    checkpoint_filepath = os.path.join(config.MODEL_SAVE_DIR, 'best_model_epoch_{epoch:02d}_val_acc_{val_accuracy:.3f}.keras')
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',               # Simpan model dengan val_accuracy tertinggi
        save_best_only=True,      # Hanya simpan yang terbaik
        verbose=1
    )
    callbacks.append(model_checkpoint_cb)

    # ReduceLROnPlateau: Mengurangi learning rate jika progress melambat
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,              # Faktor pengurangan learning rate (new_lr = lr * factor)
        patience=3,
        min_lr=0.00001,          # Batas bawah learning rate
        verbose=1
    )
    callbacks.append(reduce_lr_cb)

    # 4. Latih Model
    logging.info(f"Memulai pelatihan model untuk {config.EPOCHS} epoch...")
    try:
        history = model.fit(
            train_ds,
            epochs=config.EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks # Gunakan callbacks yang sudah didefinisikan
        )
        logging.info("Pelatihan model selesai.")

        # 5. Plot histori pelatihan (opsional)
        plot_training_history(history, config.MODEL_SAVE_DIR)


        # 6. Evaluasi Model pada data Test (Opsional, tapi baik untuk dilakukan)
        if test_ds:
            logging.info("Mengevaluasi model pada dataset test...")
            test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
            logging.info(f"Hasil Evaluasi Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        else:
            logging.info("Dataset test tidak tersedia untuk evaluasi akhir.")

        # 7. Simpan Model Final
        # Jika restore_best_weights=True pada EarlyStopping, model sudah memiliki bobot terbaik.
        # ModelCheckpoint mungkin sudah menyimpan model terbaik secara terpisah.
        # Penyimpanan ini adalah untuk model pada kondisi akhir pelatihan (atau bobot terbaik jika EarlyStopping aktif).
        model.save(config.MODEL_SAVE_PATH)
        logging.info(f"Model final berhasil disimpan di: {config.MODEL_SAVE_PATH}")

    except Exception as e:
        logging.error(f"Terjadi kesalahan selama proses pelatihan: {e}", exc_info=True)

if __name__ == "__main__":
    train()