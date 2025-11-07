import kagglehub
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score
)
import os
import joblib
import pickle
from pathlib import Path
import time

#https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python

# ===== CHECKPOINT CONFIGURATION =====
CHECKPOINT_DIR = "checkpoints"
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

def save_checkpoint(data, filename):
    """Save checkpoint to disk"""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Checkpoint saved: {filename}")

def load_checkpoint(filename):
    """Load checkpoint from disk"""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def checkpoint_exists(filename):
    """Check if checkpoint exists"""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    return os.path.exists(filepath)

# ===== STEP 1: LOAD AND PREPROCESS DATA =====
print("="*60)
print("STEP 1: LOADING AND PREPROCESSING DATA")
print("="*60)

if checkpoint_exists('processed_data.pkl'):
    print("Found existing processed data checkpoint. Loading...")
    checkpoint = load_checkpoint('processed_data.pkl')
    X_train_scaled = checkpoint['X_train_scaled']
    X_test_scaled = checkpoint['X_test_scaled']
    y_train = checkpoint['y_train']
    y_test = checkpoint['y_test']
    scaler = checkpoint['scaler']
    feature_columns = checkpoint['feature_columns']
    print(f"✓ Loaded preprocessed data from checkpoint")
    print(f"  Training samples: {X_train_scaled.shape[0]:,}")
    print(f"  Test samples: {X_test_scaled.shape[0]:,}")
    print(f"  Features: {feature_columns}")
else:
    print("No checkpoint found. Processing data from scratch...")
    
    # Download the dataset
    path = kagglehub.dataset_download("ealaxi/paysim1")
    print(f"Dataset downloaded to: {path}")
    
    # Load the CSV directly with pandas
    csv_file = os.path.join(path, "PS_20174392719_1491204439457_log.csv")
    print("Loading CSV file...")
    df = pd.read_csv(csv_file)
    
    print(f"✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("\nFirst few rows:")
    print(df.head())
    
    # OPTIONAL: Sample for laptop-friendly training
    SAMPLE_SIZE = 100000  # Adjust based on your laptop capability
    print(f"\n⚠ Sampling {SAMPLE_SIZE:,} rows for laptop-friendly training...")
    print("(To use full dataset, comment out the sampling line)")
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    feature_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
            'oldbalanceDest', 'newbalanceDest']
    
    # SVM only works with numerical values, so we need to encode
    print("\nEncoding transaction types...")
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    feature_columns.append('type_encoded')
    
    X = df[feature_columns]
    y = df['isFraud']
    
    X = X.fillna(0)  # if anything is missing or loaded incorrectly
    
    print(f"\nFraud cases: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
    
    # Train-test split
    print("\nSplitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]:,}")
    print(f"Test set size: {X_test.shape[0]:,}")
    
    # Feature scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save checkpoint
    checkpoint_data = {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    save_checkpoint(checkpoint_data, 'processed_data.pkl')
    print("✓ Data preprocessing complete")

# ===== STEP 2: TRAIN SVM MODEL =====
print("\n" + "="*60)
print("STEP 2: TRAINING SVM MODEL")
print("="*60)

if checkpoint_exists('trained_model.pkl'):
    print("Found existing trained model. Loading...")
    svm_model = load_checkpoint('trained_model.pkl')
    print("✓ Model loaded from checkpoint")
else:
    print("No trained model found. Training from scratch...")
    print(f"Training samples: {X_train_scaled.shape[0]:,}")
    print(f"Features: {X_train_scaled.shape[1]}")
    print("\nTraining SVM with RBF kernel...")
    print("(This may take a while - progress will be shown below)")
    
    svm_model = SVC(
        kernel='rbf',           # Radial basis function kernel
        C=1.0,                  # Regularization parameter
        gamma='scale',          # Kernel coefficient
        class_weight='balanced', # Handle imbalanced dataset
        probability=True,       # Enable probability estimates for ROC-AUC
        random_state=42,
        verbose=True
    )
    
    start_time = time.time()
    svm_model.fit(X_train_scaled, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"\n✓ Model training complete in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save model checkpoint
    save_checkpoint(svm_model, 'trained_model.pkl')
    
    # Also save scaler and model for deployment
    joblib.dump(svm_model, os.path.join(CHECKPOINT_DIR, 'fraud_detection_svm.pkl'))
    joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, 'scaler.pkl'))
    print("✓ Model and scaler saved for deployment")

# ===== STEP 3: MAKE PREDICTIONS =====
print("\n" + "="*60)
print("STEP 3: MAKING PREDICTIONS")
print("="*60)

if checkpoint_exists('predictions.pkl'):
    print("Found existing predictions. Loading...")
    predictions = load_checkpoint('predictions.pkl')
    y_pred = predictions['y_pred']
    y_pred_proba = predictions['y_pred_proba']
    print("✓ Predictions loaded from checkpoint")
else:
    print("Running predictions on test set...")
    y_pred = svm_model.predict(X_test_scaled)
    y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
    
    predictions = {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    save_checkpoint(predictions, 'predictions.pkl')
    print("✓ Predictions complete")

# ===== STEP 4: EVALUATION METRICS =====
print("\n" + "="*60)
print("STEP 4: MODEL EVALUATION")
print("="*60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]:,}")
print(f"False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,}")
print(f"True Positives:  {cm[1,1]:,}")

# Classification Metrics
print("\n" + "-"*50)
print("METRICS:")
print("-"*50)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Precision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"F1-Score:     {f1:.4f}")
print(f"ROC-AUC:      {roc_auc:.4f}")

print("\n" + "-"*50)
print("DETAILED CLASSIFICATION REPORT")
print("-"*50)
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Save final results
final_results = {
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'roc_auc': roc_auc,
    'confusion_matrix': cm,
    'classification_report': classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'], output_dict=True)
}
save_checkpoint(final_results, 'final_results.pkl')

print("\n" + "="*60)
print("✓ COMPLETE! All checkpoints saved")
print("="*60)
print(f"\nCheckpoints saved in '{CHECKPOINT_DIR}/' directory:")
print("  - processed_data.pkl (preprocessed data)")
print("  - trained_model.pkl (trained SVM model)")
print("  - predictions.pkl (test predictions)")
print("  - final_results.pkl (evaluation metrics)")
print("  - fraud_detection_svm.pkl (model for deployment)")
print("  - scaler.pkl (scaler for deployment)")
print("\n📝 To resume from checkpoint: Run this script again")
print("🔄 To start fresh: Delete the 'checkpoints/' directory")