# preprocessing.py

import tensorflow as tf
import os

def create_datasets(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2, seed=42):
    # Buat training dan validation datasets dari folder
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    # Normalisasi piksel (0-1)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Prefetch untuk efisiensi
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

if __name__ == "__main__":
    data_dir = "C:/Machine Learning/DisasterModel"  # Ganti dengan folder dataset
    train_ds, val_ds = create_datasets(data_dir)
    print("Train and validation datasets loaded successfully.")
