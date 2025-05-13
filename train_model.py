# Main module for training the disaster image classification model with Fine-tuning.

import tensorflow as tf
import logging
import os
import matplotlib.pyplot as plt

# Import local modules
import config
from data_loader import get_data_generators
from model_builder import create_cnn_model # Ini sekarang memanggil build_augmented_transfer_model

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ... (fungsi plot_training_history tetap sama) ...

def train():
    """Main function to train the disaster detection model, including fine-tuning."""
    logging.info("Starting model training process...")

    # 1. Load Dataset
    logging.info("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = get_data_generators()

    if not train_ds or not val_ds:
        logging.error("Training or validation dataset could not be loaded. Training aborted.")
        return

    if class_names:
        logging.info(f"Loaded class names: {class_names}")
    else:
        logging.warning("Class names could not be loaded from data_loader.")

    # 2. Build Base Model (Head is trainable, Base is frozen initially)
    logging.info("Creating Augmented Transfer Learning model...")
    model = create_cnn_model() # Ini memanggil build_augmented_transfer_model

    if not model:
        logging.error("Failed to create the model. Training aborted.")
        return

    # Define Callbacks (tetap sama, akan digunakan di kedua fase)
    callbacks = []

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True # Penting untuk mengambil bobot terbaik
    )
    callbacks.append(early_stopping_cb)

    # Checkpoint untuk fase pelatihan awal (frozen base)
    checkpoint_filepath_frozen = os.path.join(config.MODEL_SAVE_DIR, 'best_model_frozen_base_epoch_{epoch:02d}_val_acc_{val_accuracy:.3f}.keras')
    model_checkpoint_cb_frozen = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_frozen,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(model_checkpoint_cb_frozen)

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.000001, # Turunkan batas bawah learning rate untuk fine-tuning
        verbose=1
    )
    callbacks.append(reduce_lr_cb)


    # --- Fase 1: Latih Hanya Lapisan Klasifikasi Baru (Base Frozen) ---
    logging.info(f"Starting training phase 1: Training classification head for {config.EPOCHS} epochs (Base frozen)...")

    try:
        history_frozen = model.fit(
            train_ds,
            epochs=config.EPOCHS, # Gunakan epoch dari config
            validation_data=val_ds,
            callbacks=callbacks # Gunakan callbacks yang sama
        )
        logging.info("Phase 1 training completed.")

    except Exception as e:
        logging.error(f"An error occurred during Phase 1 training: {e}", exc_info=True)
        return # Berhenti jika Fase 1 gagal


    # --- Fase 2: Fine-tuning (Unfreeze Top Layers of Base Model) ---
    logging.info("\nStarting training phase 2: Fine-tuning the model (Unfreezing base layers)...")

    # Unfreeze the base model (or a portion of it)
    base_model = model.get_layer("mobilenetv2_1.00_224") # Dapatkan layer MobileNetV2 berdasarkan namanya di model_builder

    # Berapa banyak layer yang ingin Anda unfreeze dari akhir base model?
    # Lebih sedikit layer diunfreeze = lebih aman dari overfitting tapi potensial improvement lebih kecil.
    # Seluruh base model diunfreeze = potensial improvement lebih besar tapi risiko overfitting lebih tinggi.
    # MobileNetV2 punya ~150-200 layer tergantung versi. Contoh: unfreeze 20 layer terakhir.
    fine_tune_from_layer = -20 # Ganti angka ini sesuai eksperimen Anda

    logging.info(f"Unfreezing the last {abs(fine_tune_from_layer)} layers of the base model for fine-tuning.")

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_from_layer]:
        layer.trainable = False # Bekukan lapisan di awal base model

    # Re-compile the model for fine-tuning
    # Penting: Gunakan learning rate yang SANGAT RENDAH untuk fine-tuning
    fine_tune_learning_rate = config.LEARNING_RATE * 0.01 # Contoh: 1/100th dari LR awal
    fine_tune_optimizer = Adam(learning_rate=fine_tune_learning_rate) # Atau SGD dengan momentum

    model.compile(
        optimizer=fine_tune_optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logging.info(f"Model re-compiled with lower learning rate ({fine_tune_learning_rate}) for fine-tuning.")
    # Log summary model lagi untuk melihat layer mana yang trainable
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary_fine_tune = "\n".join(stringlist)
    logging.info(f"Fine-tuning Model Summary (Trainable Layers):\n{model_summary_fine_tune}")


    # Latih lagi seluruh model dengan data_ds *dimulai dari epoch terakhir*
    total_fine_tune_epochs = config.EPOCHS # Jumlah epoch tambahan untuk fine-tuning
    initial_epoch = history_frozen.epoch[-1] + 1 # Mulai dari epoch setelah fase 1 selesai
    total_epochs = initial_epoch + total_fine_tune_epochs # Total epoch keseluruhan

    # Checkpoint untuk fase fine-tuning
    checkpoint_filepath_finetune = os.path.join(config.MODEL_SAVE_DIR, 'best_model_finetuned_epoch_{epoch:02d}_val_acc_{val_accuracy:.3f}.keras')
    model_checkpoint_cb_finetune = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_finetune,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    # Ganti checkpoint di callbacks dengan yang baru untuk fase fine-tuning
    callbacks_finetune = [cb for cb in callbacks if not isinstance(cb, tf.keras.callbacks.ModelCheckpoint)] # Keep other callbacks
    callbacks_finetune.append(model_checkpoint_cb_finetune)


    logging.info(f"Starting fine-tuning for {total_fine_tune_epochs} additional epochs (Total epochs: {total_epochs})...")

    try:
        history_finetune = model.fit(
            train_ds,
            epochs=total_epochs, # Jalankan hingga total_epochs
            initial_epoch=initial_epoch, # Mulai dari epoch terakhir fase 1
            validation_data=val_ds,
            callbacks=callbacks_finetune # Gunakan callbacks fase fine-tuning
        )
        logging.info("Phase 2 (Fine-tuning) training completed.")

        # Gabungkan history untuk plotting
        history_frozen_dict = history_frozen.history
        history_finetune_dict = history_finetune.history

        combined_history = {
            key: history_frozen_dict[key] + history_finetune_dict[key]
            for key in history_frozen_dict.keys()
        }
        # 5. Plot histori pelatihan gabungan
        plot_training_history(type('obj', (object,), {'history': combined_history}), config.MODEL_SAVE_DIR)


    except Exception as e:
        logging.error(f"An error occurred during Phase 2 (Fine-tuning) training: {e}", exc_info=True)
        # Lanjutkan ke evaluasi meskipun fine-tuning gagal? Tergantung kebutuhan.
        # Untuk saat ini, biarkan lanjut tapi model mungkin bukan yang terbaik.


    # 6. Evaluasi Model pada data Test (Opsional)
    # Model terbaik dari fase 1 atau fase 2 akan dimuat secara otomatis oleh restore_best_weights
    # jika EarlyStopping dipicu, atau Anda bisa memuat file checkpoint terbaik secara manual.
    if test_ds:
        logging.info("Mengevaluasi model final pada dataset test...")
        try:
            test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
            logging.info(f"Hasil Evaluasi Test Final - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        except Exception as e:
             logging.error(f"An error occurred during final test evaluation: {e}", exc_info=True)

    else:
        logging.info("Dataset test tidak tersedia untuk evaluasi akhir.")

    # 7. Simpan Model Final (model saat ini sudah memiliki bobot terbaik jika restore_best_weights aktif)
    # Anda juga bisa memuat file checkpoint terbaik secara eksplisit di sini jika ingin yakin.
    try:
        model.save(config.MODEL_SAVE_PATH)
        logging.info(f"Model final berhasil disimpan di: {config.MODEL_SAVE_PATH}")
    except Exception as e:
        logging.error(f"Failed to save final model to {config.MODEL_SAVE_PATH}: {e}", exc_info=True)


if __name__ == "__main__":
    train()