from getData import train_ds, val_ds
from model import create_disaster_cnn
import tensorflow as tf

def main():
    # Cek apakah data tersedia
    if train_ds is None or val_ds is None:
        raise ValueError("train_ds atau val_ds tidak boleh None. Periksa kembali getData.py.")

    # Buat model
    model = create_disaster_cnn()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Latih model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    # Simpan model
    model.save("disaster_cnn_model.h5")
    print("✅ Model disimpan ke disaster_cnn_model.h5")

if __name__ == "__main__":
    main()
