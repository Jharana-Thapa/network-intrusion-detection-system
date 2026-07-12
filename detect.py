import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Mappings for output printing
INT_TO_CLASS = {0: 'normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

def load_nids_system():
    """Loads models and preprocessors."""
    preprocessors_path = os.path.join(MODELS_DIR, "preprocessors.joblib")
    if not os.path.exists(preprocessors_path):
        raise FileNotFoundError("Preprocessors not found. Run preprocess.py first.")
    
    print("Loading preprocessing pipeline...")
    preprocessors = joblib.load(preprocessors_path)
    
    print("Loading trained classifiers...")
    models = {
        'rf_binary': joblib.load(os.path.join(MODELS_DIR, "rf_binary.joblib")),
        'rf_multi': joblib.load(os.path.join(MODELS_DIR, "rf_multi.joblib")),
        'svm_binary': joblib.load(os.path.join(MODELS_DIR, "svm_binary.joblib")),
        'svm_multi': joblib.load(os.path.join(MODELS_DIR, "svm_multi.joblib"))
    }
    
    return preprocessors, models

def preprocess_single_record(record_df, preprocessors):
    """Processes a single raw record DataFrame into feature array of shape (1, 122)."""
    numeric_cols = preprocessors['numeric_cols']
    categorical_cols = preprocessors['categorical_cols']
    binary_cols = preprocessors['binary_cols']
    
    scaler = preprocessors['scaler']
    encoder = preprocessors['encoder']
    
    # Extract & preprocess features
    num_features = record_df[numeric_cols].copy()
    cat_features = record_df[categorical_cols].copy()
    bin_features = record_df[binary_cols].copy()
    
    # Standard scale numerical columns
    num_scaled = scaler.transform(num_features)
    
    # One-hot encode categorical columns
    cat_encoded = encoder.transform(cat_features)
    
    # Stack together
    features_vector = np.hstack([num_scaled, cat_encoded, bin_features.values])
    return features_vector

def predict_record(raw_record_dict, preprocessors, models):
    """Predicts binary status and class for a raw record dictionary."""
    # Convert dictionary to 1-row DataFrame
    record_df = pd.DataFrame([raw_record_dict])
    
    # Preprocess
    features = preprocess_single_record(record_df, preprocessors)
    
    # Get predictions
    rf_bin_pred = models['rf_binary'].predict(features)[0]
    rf_bin_prob = models['rf_binary'].predict_proba(features)[0][1]
    
    svm_bin_pred = models['svm_binary'].predict(features)[0]
    svm_bin_prob = models['svm_binary'].predict_proba(features)[0][1]
    
    rf_multi_pred = models['rf_multi'].predict(features)[0]
    rf_multi_prob = models['rf_multi'].predict_proba(features)[0]
    
    svm_multi_pred = models['svm_multi'].predict(features)[0]
    svm_multi_prob = models['svm_multi'].predict_proba(features)[0]
    
    results = {
        'rf_binary': {'class': 'Anomaly' if rf_bin_pred == 1 else 'Normal', 'prob': rf_bin_prob},
        'svm_binary': {'class': 'Anomaly' if svm_bin_pred == 1 else 'Normal', 'prob': svm_bin_prob},
        'rf_multi': {'class': INT_TO_CLASS[rf_multi_pred], 'prob': rf_multi_prob[rf_multi_pred]},
        'svm_multi': {'class': INT_TO_CLASS[svm_multi_pred], 'prob': svm_multi_prob[svm_multi_pred]}
    }
    return results

def get_record_from_file(index):
    """Loads a specific record from the KDDTest+.txt dataset file."""
    test_file_path = os.path.join(DATA_DIR, "KDDTest+.txt")
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test dataset not found at {test_file_path}. Run utils.py first.")
    
    # Column names of the raw dataset
    from preprocess import COLUMNS
    
    print(f"Reading record at line index {index} from KDDTest+.txt...")
    df = pd.read_csv(test_file_path, header=None, names=COLUMNS)
    if index < 0 or index >= len(df):
        raise ValueError(f"Index {index} out of range (dataset size: {len(df)})")
        
    row = df.iloc[index]
    actual_label = row['label']
    actual_difficulty = row['difficulty_level']
    
    # Convert row to dictionary excluding target columns
    record_dict = row.drop(['label', 'difficulty_level']).to_dict()
    return record_dict, actual_label, actual_difficulty

def main():
    parser = argparse.ArgumentParser(description="NIDS Traffic Predictor")
    parser.add_argument("--index", type=int, default=None, 
                        help="Index of the connection record in KDDTest+.txt to predict on")
    args = parser.parse_args()
    
    try:
        preprocessors, models = load_nids_system()
    except Exception as e:
        print(f"Initialization Error: {e}")
        sys.exit(1)
        
    if args.index is not None:
        try:
            record_dict, actual_label, actual_diff = get_record_from_file(args.index)
        except Exception as e:
            print(f"Error loading record: {e}")
            sys.exit(1)
    else:
        # Standard Demo Record: Normal TCP traffic sample
        print("No index specified. Running prediction on demo traffic records...")
        record_dict = {
            "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF", 
            "src_bytes": 220, "dst_bytes": 850, "land": 0, "wrong_fragment": 0, 
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1, 
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0, 
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, 
            "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0, 
            "count": 4, "srv_count": 4, "serror_rate": 0.0, "srv_serror_rate": 0.0, 
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0, 
            "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 10, 
            "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0, 
            "dst_host_same_src_port_rate": 0.1, "dst_host_srv_diff_host_rate": 0.04, 
            "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0, 
            "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
        }
        actual_label = "normal"
        actual_diff = "N/A"

    print("\n" + "="*50)
    print("Network Connection Traffic Details:")
    print("="*50)
    print(f"Protocol: {record_dict['protocol_type']} | Service: {record_dict['service']} | Connection Status Flag: {record_dict['flag']}")
    print(f"Bytes Sent: {record_dict['src_bytes']} | Bytes Received: {record_dict['dst_bytes']}")
    print(f"Logged In: {'Yes' if record_dict['logged_in'] == 1 else 'No'}")
    print(f"Same Service Rate: {record_dict['same_srv_rate']} | Destination Host Connections: {record_dict['dst_host_count']}")
    print(f"Actual Classification: {actual_label} (Difficulty score: {actual_diff})")
    print("="*50)
    
    # Run predictions
    print("\nRunning Machine Learning Diagnostics...")
    res = predict_record(record_dict, preprocessors, models)
    
    print("\n--- Model Classification Results ---")
    print(f"Random Forest (Binary): {res['rf_binary']['class']} (Malicious Prob: {res['rf_binary']['prob']:.4f})")
    print(f"SVM (Binary):           {res['svm_binary']['class']} (Malicious Prob: {res['svm_binary']['prob']:.4f})")
    print(f"Random Forest (Multi):  {res['rf_multi']['class'].upper()} (Confidence: {res['rf_multi']['prob']:.4f})")
    print(f"SVM (Multi):            {res['svm_multi']['class'].upper()} (Confidence: {res['svm_multi']['prob']:.4f})")
    print("="*50)

if __name__ == "__main__":
    main()
