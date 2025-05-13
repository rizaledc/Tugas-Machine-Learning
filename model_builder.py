# Module for creating the Convolutional Neural Network model.

import tensorflow as tf
import logging
import config # Import configuration

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_cnn_model():
    """
    Creates and compiles a Convolutional Neural Network (CNN) model
    for image classification.
    """
    model = tf.keras.Sequential([
        # Input layer (implicitly defined by input_shape in the first Conv2D)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=config.INPUT_SHAPE, name="conv2d_1"),
        tf.keras.layers.MaxPooling2D(2, 2, name="maxpool_1"),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv2d_2"),
        tf.keras.layers.MaxPooling2D(2, 2, name="maxpool_2"),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv2d_3"),
        tf.keras.layers.MaxPooling2D(2, 2, name="maxpool_3"),

        # Optional: Add more Conv/Pool layers
        # tf.keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv2d_4"),
        # tf.keras.layers.MaxPooling2D(2, 2, name="maxpool_4"),

        tf.keras.layers.Flatten(name="flatten_layer"),
        tf.keras.layers.Dense(128, activation='relu', name="dense_layer_1"),
        tf.keras.layers.Dropout(0.5, name="dropout_layer"), # Dropout for regularization
        tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax', name="output_layer") # Output layer
    ])

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', # Use for one-hot encoded labels
        metrics=['accuracy']
    )

    logging.info("CNN model created and compiled.")
    # Log model summary
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    logging.info(f"Model Summary:\n{model_summary}")

    return model

if __name__ == "__main__":
    logging.info("Testing model creation...")
    cnn_model = create_cnn_model()
    if cnn_model:
        logging.info("Model instance created successfully.")
    else:
        logging.error("Failed to create model instance.")