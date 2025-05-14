import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple

class ModelTester:
    def __init__(self, model_path: str, test_dir: str, img_size: Tuple[int, int]=(224, 224), batch_size: int=32):
        self.model_path = model_path
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.test_ds = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file tidak ditemukan: {self.model_path}")
        print(f"🔄 Loading model dari {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)

    def load_test_data(self):
        if not os.path.exists(self.test_dir):
            raise FileNotFoundError(f"Direktori test tidak ditemukan: {self.test_dir}")

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
        self.test_ds = datagen.flow_from_directory(
            directory=self.test_dir,
            target_size=self.img_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False
        )

    def evaluate(self):
        if self.model is None or self.test_ds is None:
            raise ValueError("Model dan dataset harus dimuat terlebih dahulu.")

        print("🔍 Evaluasi model...")
        loss, acc = self.model.evaluate(self.test_ds, verbose=0)
        print(f"✅ Test Loss     : {loss:.4f}")
        print(f"✅ Test Accuracy : {acc:.4f}")

        y_true = self.test_ds.classes
        y_pred_probs = self.model.predict(self.test_ds, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        class_labels = list(self.test_ds.class_indices.keys())

        print("\n📊 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_labels))

        print("🧩 Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

def main():
    MODEL_PATH = r"C:\Machine Learning\disaster_cnn_model.h5"
    TEST_DIR   = r"C:\Machine Learning\DisasterModel\Cyclone_Wildfire_Flood_Earthquake_Dataset\test"

    tester = ModelTester(model_path=MODEL_PATH, test_dir=TEST_DIR)
    tester.load_model()
    tester.load_test_data()
    tester.evaluate()

if __name__ == "__main__":
    main()
