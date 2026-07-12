import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """Loads the preprocessed training datasets."""
    print("Loading preprocessed training data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train_bin = np.load(os.path.join(DATA_DIR, "y_train_bin.npy"))
    y_train_multi = np.load(os.path.join(DATA_DIR, "y_train_multi.npy"))
    return X_train, y_train_bin, y_train_multi

def train_random_forest(X, y_bin, y_multi):
    """Trains Random Forest models for binary and multi-class classification."""
    print("\n--- Training Random Forest Models ---")
    
    # 1. Binary Classification
    print("Training Random Forest Binary Classifier (Full Dataset)...")
    start_time = time.time()
    rf_bin = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_bin.fit(X, y_bin)
    duration = time.time() - start_time
    print(f"Random Forest Binary Classifier trained in {duration:.2f} seconds.")
    joblib.dump(rf_bin, os.path.join(MODELS_DIR, "rf_binary.joblib"))

    # 2. Multi-class Classification
    print("Training Random Forest Multi-class Classifier (Full Dataset)...")
    start_time = time.time()
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_multi.fit(X, y_multi)
    duration = time.time() - start_time
    print(f"Random Forest Multi-class Classifier trained in {duration:.2f} seconds.")
    joblib.dump(rf_multi, os.path.join(MODELS_DIR, "rf_multi.joblib"))

def train_svm(X, y_bin, y_multi, subset_size=10000):
    """Trains Support Vector Machine models on a stratified subset to save time."""
    print(f"\n--- Training SVM Models (using a stratified subset of {subset_size} samples) ---")
    
    # Create stratified subsets for SVM training
    _, X_subset, _, y_bin_subset = train_test_split(
        X, y_bin, test_size=subset_size, stratify=y_bin, random_state=42
    )
    
    _, _, _, y_multi_subset = train_test_split(
        X, y_multi, test_size=subset_size, stratify=y_multi, random_state=42
    )
    
    # 1. Binary Classification SVM
    print("Training SVM Binary Classifier...")
    start_time = time.time()
    svm_bin = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm_bin.fit(X_subset, y_bin_subset)
    duration = time.time() - start_time
    print(f"SVM Binary Classifier trained in {duration:.2f} seconds.")
    joblib.dump(svm_bin, os.path.join(MODELS_DIR, "svm_binary.joblib"))

    # 2. Multi-class Classification SVM
    print("Training SVM Multi-class Classifier...")
    start_time = time.time()
    svm_multi = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm_multi.fit(X_subset, y_multi_subset)
    duration = time.time() - start_time
    print(f"SVM Multi-class Classifier trained in {duration:.2f} seconds.")
    joblib.dump(svm_multi, os.path.join(MODELS_DIR, "svm_multi.joblib"))

def main():
    X_train, y_train_bin, y_train_multi = load_data()
    
    # Train Models
    train_random_forest(X_train, y_train_bin, y_train_multi)
    train_svm(X_train, y_train_bin, y_train_multi)
    
    print("\nAll models trained and saved to the 'models/' directory successfully.")

if __name__ == "__main__":
    main()