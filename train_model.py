# Main module for training the disaster image classification model.

import tensorflow as tf
import logging
import os
import matplotlib.pyplot as plt # For plotting training history

# Import local modules
import config
from data_loader import get_data_generators
from model_builder import create_cnn_model

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_training_history(history, save_path_base):
    """Plots and saves the training and validation accuracy and loss."""
    try:
        # Plot Accuracy
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
        logging.info(f"Accuracy plot saved to: {acc_plot_path}")
        plt.close()

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
        logging.info(f"Loss plot saved to: {loss_plot_path}")
        plt.close()
    except Exception as e:
        logging.warning(f"Failed to generate or save training history plots: {e}")

def train():
    """Main function to train the disaster detection model."""
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

    # 2. Build Model
    logging.info("Creating CNN model...")
    model = create_cnn_model()
    if not model:
        logging.error("Failed to create the model. Training aborted.")
        return

    # 3. Define Callbacks
    callbacks = []

    # EarlyStopping: Stop training when a monitored metric has stopped improving
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',      # Metric to monitor
        patience=5,              # Number of epochs with no improvement
        verbose=1,
        restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored metric
    )
    callbacks.append(early_stopping_cb)

    # ModelCheckpoint: Save the best model during training
    # The directory for saving models is created in config.py
    checkpoint_filepath = os.path.join(config.MODEL_SAVE_DIR, 'best_model_epoch_{epoch:02d}_val_acc_{val_accuracy:.3f}.keras')
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',  # Monitor validation accuracy
        mode='max',              # Save model with the highest val_accuracy
        save_best_only=True,     # Only save the best model
        verbose=1
    )
    callbacks.append(model_checkpoint_cb)

    # ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,              # Factor by which the learning rate will be reduced (new_lr = lr * factor)
        patience=3,
        min_lr=0.00001,          # Lower bound on the learning rate
        verbose=1
    )
    callbacks.append(reduce_lr_cb)


    # 4. Train Model
    logging.info(f"Starting model training for {config.EPOCHS} epochs...")
    try:
        history = model.fit(
            train_ds,
            epochs=config.EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks # Use defined callbacks
        )
        logging.info("Model training completed.")

        # 5. Plot training history
        plot_training_history(history, config.MODEL_SAVE_DIR)

        # 6. Evaluate Model on Test Data (Optional)
        if test_ds:
            logging.info("Evaluating model on test dataset...")
            test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
            logging.info(f"Test Evaluation Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        else:
            logging.info("Test dataset not available for final evaluation.")

        # 7. Save Final Model
        # If restore_best_weights=True is used, this model will have the best weights.
        # ModelCheckpoint saves the best model separately.
        model.save(config.MODEL_SAVE_PATH)
        logging.info(f"Final model saved to: {config.MODEL_SAVE_PATH}")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)

if __name__ == "__main__":
    train()