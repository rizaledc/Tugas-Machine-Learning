# Module for creating the Convolutional Neural Network model using Transfer Learning and Data Augmentation.

import tensorflow as tf
import logging
# Import a pre-trained model base
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import config # Import configuration

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_augmented_transfer_model():
    """
    Builds and compiles a CNN model using a pre-trained MobileNetV2 base,
    including Data Augmentation and Batch Normalization.
    """
    # --- Data Augmentation Layers ---
    # These layers apply random transformations *during training*.
    # Placed at the beginning of the model for efficiency on GPU.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        # tf.keras.layers.RandomContrast(0.2), # Can add more augmentation layers
    ], name="data_augmentation")


    # --- Pre-trained Base Model (Transfer Learning) ---
    # Load MobileNetV2 pre-trained on ImageNet.
    # include_top=False removes the classification layer at the end.
    # weights='imagenet' loads the weights from ImageNet.
    base_model = MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model layers.
    # This prevents their weights from being updated during the first phase of training.
    base_model.trainable = False

    # --- Build the new classification head ---
    # Add layers on top of the base model's output
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")
    
    # Add Batch Normalization and Dropout for regularization
    prediction_layer = Sequential([
        BatchNormalization(name="batch_norm_1"),
        Dense(128, activation='relu', name="dense_layer_1"),
        Dropout(0.5, name="dropout_layer_1"),
        # Optional: add more dense/batchnorm/dropout layers
        # BatchNormalization(name="batch_norm_2"),
        # Dense(64, activation='relu', name="dense_layer_2"),
        # Dropout(0.3, name="dropout_layer_2"),
        Dense(config.NUM_CLASSES, activation='softmax', name="output_layer") # Output layer
    ], name="prediction_head")


    # --- Combine Base Model and Head ---
    model = Sequential([
        tf.keras.layers.Input(shape=config.INPUT_SHAPE), # Explicit Input layer
        data_augmentation,       # Apply data augmentation first
        tf.keras.layers.Rescaling(1./255), # Rescale pixel values (MobileNetV2 expects input in [-1, 1] range,
                                          # but Rescaling 1/255 then applying preprocess_input is common)
        # tf.keras.applications.mobilenet_v2.preprocess_input, # Alternative preprocessor
        base_model,              # Add the pre-trained base
        global_average_layer,    # Pooling layer
        prediction_layer         # Add the new classification head
    ], name="disaster_cnn_transfer")


    # --- Compile the model ---
    # Use a slightly lower learning rate often works well with transfer learning
    optimizer = Adam(learning_rate=config.LEARNING_RATE * 0.1) # Example: 1/10th of original LR

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logging.info("Augmented Transfer Learning model built and compiled.")
    # Log model summary
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    logging.info(f"Model Summary:\n{model_summary}")

    return model

# Note: We are replacing the original create_cnn_model for this example.
# If you want to keep both, rename the function above and update train_model.py
# to call the desired function.
create_cnn_model = build_augmented_transfer_model

if __name__ == "__main__":
    logging.info("Testing model creation (Augmented Transfer Learning)...")
    cnn_model = create_cnn_model()
    if cnn_model:
        logging.info("Model instance created successfully.")
    else:
        logging.error("Failed to create model instance.")