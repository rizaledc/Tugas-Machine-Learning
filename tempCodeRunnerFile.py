# model.py

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout

def create_disaster_cnn(input_shape=(224, 224, 3), num_classes=4):
    # Gunakan VGG16 sebagai base model untuk transfer learning
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False  # Freeze pre-trained layers

    model = tf.keras.Sequential([
        base_model,
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
