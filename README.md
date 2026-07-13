# Smart Network Intrusion Detection System (NIDS)
A Machine Learning-powered Network Intrusion Detection System (NIDS) trained on the **NSL-KDD** dataset. The system features preprocessing, evaluation analytics, a command-line interface (CLI) diagnostic tool, and an interactive dark-themed web dashboard.
---
## 🚀 Features
- **Binary Classification:** Classify connection traffic as `Normal` or `Anomaly` (threat).
- **Multi-class Classification:** Categorize attacks into 5 distinct classes: `Normal`, `DoS` (Denial of Service), `Probe` (Port scanning/probing), `R2L` (Remote-to-Local unauthorized access), and `U2R` (User-to-Root privilege escalation).
- **Dual Classifiers:** Employs both **Random Forest** (high accuracy) and **Support Vector Machine (SVM)** models.
- **Diagnostics CLI:** Run immediate threat diagnostics on specific test set indexes or mock HTTP traffic.
- **Glassmorphic Web Dashboard:** Interactive web user interface with batch CSV log uploading, live dataset inspector, and visual evaluation graph galleries.
---

## 📁 Project Structure
```text
nids/
│
├── data/                       # Dataset storage
│   ├── KDDTrain+.txt           # Raw train set (auto-downloaded)
│   ├── KDDTest+.txt            # Raw test set (auto-downloaded)
│   └── preprocessed/           # Preprocessed numpy arrays (.npy)
│
├── models/                     # Saved models & preprocessor state
│   ├── preprocessors.joblib    # StandardScaler, OneHotEncoder, column maps
│   ├── rf_binary.joblib        # Random Forest Binary Classifier
│   ├── rf_multi.joblib         # Random Forest Multi-class Classifier
│   ├── svm_binary.joblib       # SVM Binary Classifier
│   └── svm_multi.joblib        # SVM Multi-class Classifier
│
├── plots/                      # Generated performance graphs
│   ├── class_distribution.png
│   ├── confusion_matrix_binary.png
│   ├── confusion_matrix_multi.png
│   ├── roc_curve_binary.png
│   └── feature_importance.png
│
├── static/                     # Web app assets
│   ├── style.css               # Glassmorphism visual layout stylesheet
│   └── app.js                  # Frontend API caller & GUI manager
│
├── templates/
│   └── index.html              # Dashboard dashboard index
│
├── requirements.txt            # Project dependencies
├── utils.py                    # Dataset downloader utility
├── preprocess.py               # Preprocessing & encoding pipeline
├── train.py                    # Model training coordinator
├── visualize.py                # Visual exporter and metric generator
├── detect.py                   # CLI diagnostics utility
└── app.py                      # Flask web server entrypoint

## ⚙️ Running the Pipeline Step-by-Step
### Step 1: Download the Datasets
Downloads the official NSL-KDD train and test datasets from the UNB repository:
```powershell
python utils.py
```
### Step 2: Preprocess the Features
Normalizes numerical scales, encodes categorical features, maps labels, and saves parameters:
```powershell
python preprocess.py
```
### Step 3: Train the Models
Trains Random Forest models (on the full dataset) and SVM classifiers (on a stratified subset):
```powershell
python train.py
```
### Step 4: Export Visual Analytics
Evaluates test accuracy, prints metric reports, and saves performance curves:
```powershell
python visualize.py
```
---
## 🖥️ Running the NIDS Interfaces
### 1. The Command-Line Interface (CLI)
Diagnose live or saved traffic connection logs:
- **Demonstration check (Standard HTTP):**
  ```powershell
  python detect.py
  ```
- **Test Set inspector (Specify index row, e.g., index 20):**
  ```powershell
  python detect.py --index 20
  ```
### 2. The Interactive Web Dashboard
Launches a responsive local Flask dashboard server:
```powershell
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000/`**
---
## 📊 Summary of Model Performance
### Binary Status (Anomaly vs Normal)
|
 Model 
|
 Test Accuracy 
|
 Precision (Anomaly) 
|
 Recall (Anomaly) 
|
 F1-Score (Anomaly) 
|
|
:---
|
:---:
|
:---:
|
:---:
|
:---:
|
|
**
Random Forest
**
|
 77.0% 
|
 97.0% 
|
 61.0% 
|
 75.0% 
|
|
**
SVM (stratified)
**
|
 78.0% 
|
 97.0% 
|
 63.0% 
|
 76.0% 
|
