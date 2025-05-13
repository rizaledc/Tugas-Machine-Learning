# Module for evaluating the trained model on the test dataset.

import tensorflow as tf
import logging
import os
import config

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_test_data():
    """Loads the test dataset using image_dataset_from_directory."""
    logging.info("Loading test dataset...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.TEST_DIR,
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        label_mode='categorical' # One-hot encoding
    )
    return test_ds

def evaluate_model(model, test_ds):
    """Evaluates the model on the provided test dataset."""
    logging.info("Evaluating model on test dataset...")
    if test_ds is None:
        logging.warning("Test dataset not available for evaluation.")
        return

    try:
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
        logging.info(f"Test Evaluation Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    except Exception as e:
         logging.error(f"An error occurred during model evaluation: {e}", exc_info=True)


def load_best_model(model_path):
    """Loads a saved Keras model from the specified path."""
    logging.info(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Example path to the best model file saved during training.
    # The actual filename will depend on your training output and ModelCheckpoint configuration.
    best_model_example_file = os.path.join(config.MODEL_SAVE_DIR, 'best_model_frozen_base_epoch_08_val_acc_0.627.keras') # Replace XX and X.XXX
    
    # Consider adding logic here to find the actual best model file
    # if you don't know the exact filename beforehand.

    model = load_best_model(best_model_example_file)
    
    if model:
        test_ds = load_test_data()
        evaluate_model(model, test_ds)
    else:
        logging.error("Could not load the model. Test evaluation skipped.")