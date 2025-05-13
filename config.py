# config.py
import os

# --- PATH DASAR PROYEK ---
# Asumsikan skrip ini berada di root proyek, atau sesuaikan path-nya.
# Jika skrip-skrip (seperti train_model.py) ada di subfolder (misal 'src'),
# Anda mungkin perlu menyesuaikan PROJECT_ROOT, contoh:
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- PATH DATASET ---
# Direktori sumber dataset mentah (sebelum dibagi)
# Contoh: DisasterModel_Raw/Cyclone_Wildfire_Flood_Earthquake_Dataset/Cyclone/img1.jpg
RAW_DATASET_BASE_DIR = os.path.join(PROJECT_ROOT, "DisasterModel_Raw", "Cyclone_Wildfire_Flood_Earthquake_Dataset")

# Direktori target untuk dataset yang sudah dibagi (train/validation/test)
# Contoh: DisasterModel_Split/train/Cyclone/img1.jpg
SPLIT_DATASET_BASE_DIR = os.path.join(PROJECT_ROOT, "DisasterModel_Split")

TRAIN_DIR = os.path.join(SPLIT_DATASET_BASE_DIR, "train")
VALIDATION_DIR = os.path.join(SPLIT_DATASET_BASE_DIR, "validation")
TEST_DIR = os.path.join(SPLIT_DATASET_BASE_DIR, "test")

# --- PARAMETER DATASET ---
CLASSES = ["Cyclone", "Earthquake", "Flood", "Wildfire"]
# Rasio pembagian: (train, validation, test)
SPLIT_RATIOS = (0.7, 0.2, 0.1)

# --- PARAMETER MODEL ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
NUM_CLASSES = len(CLASSES)

# --- PARAMETER PELATIHAN ---
EPOCHS = 25  # Jumlah epoch bisa disesuaikan
LEARNING_RATE = 0.001

# --- PATH PENYIMPANAN MODEL ---
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
MODEL_NAME = "disaster_detection_cnn.h5"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

# Pastikan direktori penyimpanan model ada
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- LAIN-LAIN ---
RANDOM_SEED = 42 # Untuk reproduktifitas