# Module for loading and preprocessing image datasets using TensorFlow.

import tensorflow as tf
import logging
import os
import config # Import configuration

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_single_dataset(directory, image_size, batch_size, label_mode, dataset_name, shuffle_data=False):
    """Helper function to load a single dataset split (train/val/test)."""
    if not tf.io.gfile.exists(directory) or not tf.io.gfile.listdir(directory):
        logging.error(f"Directory for {dataset_name} is empty or not found: {directory}")
        logging.error("Please ensure 'split_data.py' has been run and paths in 'config.py' are correct.")
        return None
    try:
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels='inferred', # Labels inferred from subdirectory names
            label_mode=label_mode, # 'categorical' for one-hot encoding
            image_size=image_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=shuffle_data, # Only shuffle training data initially
            seed=config.RANDOM_SEED if shuffle_data else None
        )
        logging.info(f"Dataset {dataset_name} successfully loaded from: {directory}")
        return dataset # Dataset object includes class_names attribute
    except Exception as e:
        logging.error(f"Failed to load dataset {dataset_name} from {directory}: {e}")
        return None

def _preprocess_dataset(dataset, dataset_name):
    """Applies normalization and prefetching to a dataset."""
    if dataset is None:
        return None

    # Normalize pixel values to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Optimize performance
    dataset = dataset.cache()
    if dataset_name == "train":
        dataset = dataset.shuffle(buffer_size=1000, seed=config.RANDOM_SEED, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    logging.info(f"Preprocessing (normalization & prefetch) applied to dataset {dataset_name}.")
    return dataset

def get_data_generators():
    """Loads and preprocesses train, validation, and test datasets.
    Returns the datasets and the detected class names.
    """
    image_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
    label_mode = 'categorical' # Matches 'categorical_crossentropy' loss

    logging.info("Loading training dataset...")
    train_ds_raw = _load_single_dataset(
        config.TRAIN_DIR, image_size, config.BATCH_SIZE, label_mode, "train_raw", shuffle_data=True
    )
    
    class_names_list = None
    if train_ds_raw:
        class_names_list = train_ds_raw.class_names # Get class names from training data
        logging.info(f"Detected class names from training dataset: {class_names_list}")
        train_ds = _preprocess_dataset(train_ds_raw, "train")
    else:
        train_ds = None

    logging.info("Loading validation dataset...")
    val_ds_raw = _load_single_dataset(
        config.VALIDATION_DIR, image_size, config.BATCH_SIZE, label_mode, "validation_raw", shuffle_data=False
    )
    val_ds = _preprocess_dataset(val_ds_raw, "validation")

    logging.info("Loading test dataset...")
    test_ds_raw = _load_single_dataset(
        config.TEST_DIR, image_size, config.BATCH_SIZE, label_mode, "test_raw", shuffle_data=False
    )
    test_ds = _preprocess_dataset(test_ds_raw, "test")

    if train_ds and val_ds:
        logging.info("Required datasets (train, validation) loaded successfully.")
    else:
        logging.warning("Failed to load one or more required datasets. Check logs.")

    return train_ds, val_ds, test_ds, class_names_list # Return class names

if __name__ == "__main__":
    logging.info("Testing data loader...")
    
    # Get datasets and class names
    train_dataset, val_dataset, test_dataset, loaded_class_names = get_data_generators()

    if train_dataset:
        if loaded_class_names:
             logging.info(f"Class names from training dataset: {loaded_class_names}")
        else:
             logging.warning("Training dataset loaded, but class names could not be retrieved.")

        for images, labels in train_dataset.take(1): # Inspect one batch
            logging.info(f"Shape of training image batch: {images.shape}")
            logging.info(f"Shape of training label batch: {labels.shape}")
            logging.info(f"Example training label (one-hot): {labels[0]}")
            break

    if val_dataset:
        logging.info("Validation dataset loaded successfully.")
    if test_dataset:
        logging.info("Test dataset loaded successfully.")