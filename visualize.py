import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Artifacts directory for embedding plots in walkthrough
ARTIFACT_DIR = r"C:\Users\jhara\.gemini\antigravity\brain\e79a4c87-c297-4d8c-861c-0b1603e489af"

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    """Loads preprocessed testing data."""
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test_bin = np.load(os.path.join(DATA_DIR, "y_test_bin.npy"))
    y_test_multi = np.load(os.path.join(DATA_DIR, "y_test_multi.npy"))
    return X_test, y_test_bin, y_test_multi

def save_plot(filename):
    """Saves plot to project plots directory and the artifacts directory."""
    local_path = os.path.join(PLOTS_DIR, filename)
    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    
    plt.savefig(local_path, bbox_inches='tight', dpi=150)
    try:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        plt.savefig(artifact_path, bbox_inches='tight', dpi=150)
        print(f"Saved plot to: {local_path} and {artifact_path}")
    except Exception as e:
        print(f"Saved plot to local project only. Artifact save failed: {e}")
    plt.close()

def plot_class_distribution(y_test_multi, class_mapping):
    """Plots the distribution of the 5 classes in the test set."""
    print("Generating class distribution plot...")
    inv_map = {v: k for k, v in class_mapping.items()}
    labels = [inv_map[val] for val in y_test_multi]
    
    df = pd.DataFrame({'Class': labels})
    counts = df['Class'].value_counts()
    
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    plt.title("Distribution of Network Traffic Classes in Test Set", fontsize=14, fontweight='bold')
    plt.xlabel("Traffic Class", fontsize=12)
    plt.ylabel("Number of Connections", fontsize=12)
    
    for i, v in enumerate(counts.values):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
        
    save_plot("class_distribution.png")

def evaluate_binary(X_test, y_test_bin, rf_bin, svm_bin):
    """Evaluates binary classifiers and creates confusion matrices and ROC curves."""
    print("\nEvaluating Binary Classifiers...")
    
    rf_preds = rf_bin.predict(X_test)
    svm_preds = svm_bin.predict(X_test)
    
    print("\n--- Random Forest Binary Classification Report ---")
    print(classification_report(y_test_bin, rf_preds, target_names=["Normal", "Anomaly"]))
    
    print("\n--- SVM Binary Classification Report ---")
    print(classification_report(y_test_bin, svm_preds, target_names=["Normal", "Anomaly"]))
    
    # 1. Confusion Matrix Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RF Confusion Matrix
    cm_rf = confusion_matrix(y_test_bin, rf_preds)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    axes[0].set_title("Random Forest Binary Confusion Matrix", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # SVM Confusion Matrix
    cm_svm = confusion_matrix(y_test_bin, svm_preds)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    axes[1].set_title("SVM Binary Confusion Matrix", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    
    plt.suptitle("Binary Classification Confusion Matrices (Normal vs Anomaly)", fontsize=14, fontweight='bold', y=1.02)
    save_plot("confusion_matrix_binary.png")
    
    # 2. ROC Curve Plot
    rf_probs = rf_bin.predict_proba(X_test)[:, 1]
    svm_probs = svm_bin.predict_proba(X_test)[:, 1]
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test_bin, rf_probs)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    fpr_svm, tpr_svm, _ = roc_curve(y_test_bin, svm_probs)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
    plt.plot(fpr_svm, tpr_svm, color='green', lw=2, label=f'SVM (AUC = {roc_auc_svm:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    
    save_plot("roc_curve_binary.png")

def evaluate_multiclass(X_test, y_test_multi, rf_multi, svm_multi, class_mapping):
    """Evaluates multi-class classifiers and plots confusion matrices."""
    print("\nEvaluating Multi-class Classifiers...")
    
    rf_preds = rf_multi.predict(X_test)
    svm_preds = svm_multi.predict(X_test)
    
    target_names = sorted(class_mapping, key=class_mapping.get)
    
    print("\n--- Random Forest Multi-class Classification Report ---")
    print(classification_report(y_test_multi, rf_preds, target_names=target_names))
    
    print("\n--- SVM Multi-class Classification Report ---")
    print(classification_report(y_test_multi, svm_preds, target_names=target_names))
    
    # Combined Confusion Matrix Plot for Multi-class
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # RF Confusion Matrix
    cm_rf = confusion_matrix(y_test_multi, rf_preds)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Purples', ax=axes[0],
                xticklabels=target_names, yticklabels=target_names)
    axes[0].set_title("Random Forest Multi-class Confusion Matrix", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # SVM Confusion Matrix
    cm_svm = confusion_matrix(y_test_multi, svm_preds)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=target_names, yticklabels=target_names)
    axes[1].set_title("SVM Multi-class Confusion Matrix", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    
    plt.suptitle("Multi-class Confusion Matrices (5 Categories)", fontsize=14, fontweight='bold', y=1.02)
    save_plot("confusion_matrix_multi.png")

def plot_feature_importance(rf_bin, feature_names):
    """Plots the top 15 feature importances from the Random Forest model."""
    print("Generating Feature Importance plot...")
    importances = rf_bin.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Top 15 features
    top_n = 15
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_names, hue=top_names, palette="rocket", legend=False)
    plt.title("Top 15 Most Important Traffic Features (Random Forest)", fontsize=14, fontweight='bold')
    plt.xlabel("Gini Importance Score", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    
    save_plot("feature_importance.png")

def main():
    X_test, y_test_bin, y_test_multi = load_data()
    
    # Load preprocessors (for feature names)
    preprocessors = joblib.load(os.path.join(MODELS_DIR, "preprocessors.joblib"))
    feature_names = preprocessors['feature_names']
    class_mapping = preprocessors['class_mapping']
    
    # Load Models
    print("Loading models...")
    rf_bin = joblib.load(os.path.join(MODELS_DIR, "rf_binary.joblib"))
    rf_multi = joblib.load(os.path.join(MODELS_DIR, "rf_multi.joblib"))
    svm_bin = joblib.load(os.path.join(MODELS_DIR, "svm_binary.joblib"))
    svm_multi = joblib.load(os.path.join(MODELS_DIR, "svm_multi.joblib"))
    
    # Generate Visualizations
    plot_class_distribution(y_test_multi, class_mapping)
    evaluate_binary(X_test, y_test_bin, rf_bin, svm_bin)
    evaluate_multiclass(X_test, y_test_multi, rf_multi, svm_multi, class_mapping)
    plot_feature_importance(rf_bin, feature_names)
    
    print("\nVisualizations and evaluations completed successfully.")

if __name__ == "__main__":
    main()