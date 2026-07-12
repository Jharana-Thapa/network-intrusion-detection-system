import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

app = Flask(__name__)

# Load models and preprocessors
print("Loading models and preprocessors...")
try:
    preprocessors = joblib.load(os.path.join(MODELS_DIR, "preprocessors.joblib"))
    rf_binary = joblib.load(os.path.join(MODELS_DIR, "rf_binary.joblib"))
    rf_multi = joblib.load(os.path.join(MODELS_DIR, "rf_multi.joblib"))
    svm_binary = joblib.load(os.path.join(MODELS_DIR, "svm_binary.joblib"))
    svm_multi = joblib.load(os.path.join(MODELS_DIR, "svm_multi.joblib"))
    print("All components loaded successfully.")
except Exception as e:
    print(f"Error loading models or preprocessors: {e}")
    # We will initialize them as None, and check during requests to avoid server startup crash.
    preprocessors = None
    rf_binary = None
    rf_multi = None
    svm_binary = None
    svm_multi = None

# Mappings for predictions
INT_TO_CLASS = {0: 'normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

DEFAULT_RECORD = {
    # Numeric
    "duration": 0, "src_bytes": 0, "dst_bytes": 0, "wrong_fragment": 0, "urgent": 0, 
    "hot": 0, "num_failed_logins": 0, "num_compromised": 0, "root_shell": 0, "su_attempted": 0, 
    "num_root": 0, "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, 
    "num_outbound_cmds": 0, "count": 1, "srv_count": 1, "serror_rate": 0.0, 
    "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0, 
    "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 1, "dst_host_srv_count": 1, 
    "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0, 
    "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0, 
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
    # Categorical
    "protocol_type": "tcp", "service": "http", "flag": "SF",
    # Binary
    "land": 0, "logged_in": 0, "is_host_login": 0, "is_guest_login": 0
}

def preprocess_df(df_input):
    """Safely aligns and preprocesses input DataFrame using saved scaler/encoder."""
    scaler = preprocessors['scaler']
    encoder = preprocessors['encoder']
    numeric_cols = preprocessors['numeric_cols']
    categorical_cols = preprocessors['categorical_cols']
    binary_cols = preprocessors['binary_cols']
    
    # Copy to avoid modifying original
    df = df_input.copy()
    
    # 1. Fill missing columns with defaults if necessary
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = DEFAULT_RECORD[col]
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = DEFAULT_RECORD[col]
    for col in binary_cols:
        if col not in df.columns:
            df[col] = DEFAULT_RECORD[col]
            
    # 2. Extract and transform
    X_num = scaler.transform(df[numeric_cols])
    X_cat = encoder.transform(df[categorical_cols])
    X_bin = df[binary_cols].values
    
    # 3. Stack features
    return np.hstack([X_num, X_cat, X_bin])

def get_predictions_for_features(features):
    """Helper to run predictions on preprocessed features (2D numpy array)."""
    # Random Forest Binary
    rf_bin_pred = rf_binary.predict(features)[0]
    rf_bin_prob = rf_binary.predict_proba(features)[0][1]
    
    # SVM Binary
    svm_bin_pred = svm_binary.predict(features)[0]
    svm_bin_prob = svm_binary.predict_proba(features)[0][1]
    
    # Random Forest Multi-class
    rf_multi_pred = rf_multi.predict(features)[0]
    
    # SVM Multi-class
    svm_multi_pred = svm_multi.predict(features)[0]
    
    return {
        'rf_binary': {'class': 'Anomaly' if rf_bin_pred == 1 else 'Normal', 'prob': float(rf_bin_prob)},
        'svm_binary': {'class': 'Anomaly' if svm_bin_pred == 1 else 'Normal', 'prob': float(svm_bin_prob)},
        'rf_multi': {'class': INT_TO_CLASS[rf_multi_pred]},
        'svm_multi': {'class': INT_TO_CLASS[svm_multi_pred]}
    }

@app.route('/')
def index():
    return render_template('index.html')

# Custom route to serve generated plots from plots directory
@app.route('/static/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

@app.route('/api/inspect', methods=['GET'])
def inspect():
    if not preprocessors:
        return jsonify({'error': 'Models not loaded'}), 500
        
    index = request.args.get('index', default=None, type=int)
    if index is None:
        return jsonify({'error': 'Missing row index parameter'}), 400
        
    test_file_path = os.path.join(DATA_DIR, "KDDTest+.txt")
    if not os.path.exists(test_file_path):
        return jsonify({'error': 'Test dataset file not found. Run utils.py first.'}), 404
        
    # Read row N
    try:
        from preprocess import COLUMNS
        df = pd.read_csv(test_file_path, header=None, names=COLUMNS)
        if index < 0 or index >= len(df):
            return jsonify({'error': f'Index must be between 0 and {len(df)-1}'}), 400
            
        row = df.iloc[index]
        actual_label = str(row['label']).strip('.')
        actual_difficulty = int(row['difficulty_level'])
        
        record_dict = row.drop(['label', 'difficulty_level']).to_dict()
        
        # Preprocess & Predict
        record_df = pd.DataFrame([record_dict])
        features = preprocess_df(record_df)
        predictions = get_predictions_for_features(features)
        
        # Format connection fields to return to frontend
        record_info = {
            'protocol_type': record_dict['protocol_type'],
            'service': record_dict['service'],
            'flag': record_dict['flag']
        }
        
        return jsonify({
            'record': record_info,
            'actual_label': actual_label,
            'actual_difficulty': actual_difficulty,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process record: {str(e)}'}), 500

@app.route('/api/predict_manual', methods=['POST'])
def predict_manual():
    if not preprocessors:
        return jsonify({'error': 'Models not loaded'}), 500
        
    user_data = request.get_json()
    if not user_data:
        return jsonify({'error': 'Empty JSON request'}), 400
        
    # Merge user data with defaults to create a full row
    record_dict = DEFAULT_RECORD.copy()
    for key, val in user_data.items():
        if key in record_dict:
            record_dict[key] = val
            
    try:
        record_df = pd.DataFrame([record_dict])
        features = preprocess_df(record_df)
        predictions = get_predictions_for_features(features)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict_csv', methods=['POST'])
def predict_csv():
    if not preprocessors:
        return jsonify({'error': 'Models not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400
        
    try:
        df = pd.read_csv(file)
        if len(df) == 0:
            return jsonify({'error': 'Uploaded CSV is empty'}), 400
            
        # Preprocess all rows
        features = preprocess_df(df)
        
        # Batch Predict
        bin_preds = rf_binary.predict(features)
        multi_preds = rf_multi.predict(features)
        
        bin_labels = ["Normal" if x == 0 else "Anomaly" for x in bin_preds]
        multi_labels = [INT_TO_CLASS[x] for x in multi_preds]
        
        results = []
        for i in range(len(df)):
            results.append({
                "Row": i + 1,
                "Binary": bin_labels[i],
                "Multi-class": multi_labels[i]
            })
            
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Failed to parse and predict CSV: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask NIDS Web Server at http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
