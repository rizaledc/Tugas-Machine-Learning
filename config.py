import os

# --- PROJECT ROOT ---
# Assumes this script is at the project root. Adjust if needed.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- DATASET PATHS ---
# Directory for raw dataset
RAW_DATASET_BASE_DIR = os.path.join(PROJECT_ROOT, "DisasterModel_Raw", "Cyclone_Wildfire_Flood_Earthquake_Dataset")

# Base directory for split dataset (train/validation/test)
SPLIT_DATASET_BASE_DIR = os.path.join(PROJECT_ROOT, "DisasterModel_Split")

TRAIN_DIR = os.path.join(SPLIT_DATASET_BASE_DIR, "train")
VALIDATION_DIR = os.path.join(SPLIT_DATASET_BASE_DIR, "validation")
TEST_DIR = os.path.join(SPLIT_DATASET_BASE_DIR, "test")

# --- DATASET PARAMETERS ---
CLASSES = ["Cyclone", "Earthquake", "Flood", "Wildfire"]
# Split ratios: (train, validation, test)
SPLIT_RATIOS = (0.7, 0.2, 0.1)

# --- MODEL PARAMETERS ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
NUM_CLASSES = len(CLASSES)

# --- TRAINING PARAMETERS ---
EPOCHS = 25
LEARNING_RATE = 0.001

# --- MODEL SAVING PATH ---
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
MODEL_NAME = "disaster_detection_cnn.h5"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

# --- MISC ---
RANDOM_SEED = 42 # For reproducibility