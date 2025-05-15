# train.py

import tensorflow as tf
from getData import train_ds, val_ds
from model import create_disaster_cnn

# Cek apakah GPU tersedia
if tf.config.list_physical_devices('GPU'):
    print("CUDA tersedia! Menggunakan GPU untuk pelatihan.")
else:
    print("CUDA tidak tersedia. Menggunakan CPU untuk pelatihan.")

def main():
    # Cek apakah data tersedia
    if train_ds is None or val_ds is None:
        raise ValueError("train_ds atau val_ds tidak boleh None. Periksa kembali getData.py.")

    # Buat model dengan transfer learning dan augmentasi
    model = create_disaster_cnn()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Setup callback untuk menghentikan pelatihan saat model berhenti meningkat
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    # Latih model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[early_stop, model_checkpoint]
    )

    # Simpan model akhir
    model.save("disaster_cnn_model.keras")
    print("✅ Model disimpan ke disaster_cnn_model.keras")

if __name__ == "__main__":
    main()
