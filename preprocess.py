import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "preprocessed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Column names of the NSL-KDD dataset
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
    "num_compromised", "root_shell", "su_attempted", "num_root", 
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", 
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

# Feature lists
NUMERIC_COLS = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot", 
    "num_failed_logins", "num_compromised", "root_shell", "su_attempted", 
    "num_root", "num_file_creations", "num_shells", "num_access_files", 
    "num_outbound_cmds", "count", "srv_count", "serror_rate", "srv_serror_rate", 
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
BINARY_COLS = ["land", "logged_in", "is_host_login", "is_guest_login"]

# Mapping of specific attacks to their broader category
# 5 classes: normal, DoS, Probe, R2L, U2R
ATTACK_MAP = {
    # DoS
    'neptune': 'DoS', 'smurf': 'DoS', 'back': 'DoS', 'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS', 
    'apache2': 'DoS', 'mailbomb': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'portsweep': 'Probe', 'nmap': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L', 
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 
    'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 
    'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
}

CLASS_LABEL_TO_INT = {
    'normal': 0,
    'DoS': 1,
    'Probe': 2,
    'R2L': 3,
    'U2R': 4
}

def load_raw_dataset(path):
    """Loads raw NSL-KDD text files into Pandas DataFrame."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df

def preprocess_datasets():
    train_file = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_file = os.path.join(DATA_DIR, "KDDTest+.txt")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError("Raw datasets not found. Run utils.py first.")

    print("Loading raw training and testing data...")
    df_train = load_raw_dataset(train_file)
    df_test = load_raw_dataset(test_file)

    # 1. Target Variables Preprocessing
    # Remove trailing dot if present in label
    df_train['label'] = df_train['label'].astype(str).str.strip('.')
    df_test['label'] = df_test['label'].astype(str).str.strip('.')

    # Binary Targets (0 = normal, 1 = anomaly/attack)
    y_train_bin = (df_train['label'] != 'normal').astype(int).values
    y_test_bin = (df_test['label'] != 'normal').astype(int).values

    # Multi-class Targets (5 classes)
    def map_attack_class(label):
        if label == 'normal':
            return 'normal'
        return ATTACK_MAP.get(label, 'DoS') # Default unknown attacks to DoS or generalized class

    train_mapped = df_train['label'].apply(map_attack_class)
    test_mapped = df_test['label'].apply(map_attack_class)

    y_train_multi = train_mapped.map(CLASS_LABEL_TO_INT).fillna(1).astype(int).values
    y_test_multi = test_mapped.map(CLASS_LABEL_TO_INT).fillna(1).astype(int).values

    # 2. Preprocess features
    # Separate numeric and categorical features
    X_train_num = df_train[NUMERIC_COLS].copy()
    X_test_num = df_test[NUMERIC_COLS].copy()

    X_train_cat = df_train[CATEGORICAL_COLS].copy()
    X_test_cat = df_test[CATEGORICAL_COLS].copy()

    X_train_bin = df_train[BINARY_COLS].copy()
    X_test_bin = df_test[BINARY_COLS].copy()

    # Scaling numeric features
    print("Scaling numeric features...")
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    # One-hot encoding categorical features
    print("Encoding categorical features...")
    # handle_unknown='ignore' ensures test values not seen in training are encoded as all zeros.
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat_encoded = encoder.fit_transform(X_train_cat)
    X_test_cat_encoded = encoder.transform(X_test_cat)

    # Combine all preprocessed features
    # Order: Numeric features, encoded categorical features, raw binary features
    X_train_preprocessed = np.hstack([
        X_train_num_scaled, 
        X_train_cat_encoded, 
        X_train_bin.values
    ])
    X_test_preprocessed = np.hstack([
        X_test_num_scaled, 
        X_test_cat_encoded, 
        X_test_bin.values
    ])

    # Get feature names for references (e.g. feature importance plotting)
    cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERIC_COLS + cat_feature_names + BINARY_COLS

    print(f"Features dimension: {X_train_preprocessed.shape[1]}")
    print(f"Training shape: {X_train_preprocessed.shape}")
    print(f"Testing shape: {X_test_preprocessed.shape}")

    # 3. Save preprocessed data
    print("Saving preprocessed numpy arrays...")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_preprocessed)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_preprocessed)
    np.save(os.path.join(OUTPUT_DIR, "y_train_bin.npy"), y_train_bin)
    np.save(os.path.join(OUTPUT_DIR, "y_test_bin.npy"), y_test_bin)
    np.save(os.path.join(OUTPUT_DIR, "y_train_multi.npy"), y_train_multi)
    np.save(os.path.join(OUTPUT_DIR, "y_test_multi.npy"), y_test_multi)

    # Save preprocessing objects and configuration
    preprocessors = {
        'scaler': scaler,
        'encoder': encoder,
        'numeric_cols': NUMERIC_COLS,
        'categorical_cols': CATEGORICAL_COLS,
        'binary_cols': BINARY_COLS,
        'feature_names': feature_names,
        'class_mapping': CLASS_LABEL_TO_INT,
        'attack_map': ATTACK_MAP
    }
    
    preprocessors_path = os.path.join(MODELS_DIR, "preprocessors.joblib")
    print(f"Saving preprocessors to {preprocessors_path}...")
    joblib.dump(preprocessors, preprocessors_path)
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_datasets()
