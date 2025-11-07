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
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import os
import joblib
import pickle
from pathlib import Path
import time
import matplotlib.pyplot as plt

# ===== CONFIGURATION =====
CHECKPOINT_DIR = "checkpoints"
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# IMBALANCE HANDLING STRATEGY
# Options: 'none', 'smote', 'undersample', 'smotetomek'
RESAMPLING_STRATEGY = 'smote'  # Change this to experiment

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

def print_class_distribution(y, label=""):
    """Print class distribution statistics"""
    fraud_count = y.sum()
    total = len(y)
    fraud_pct = (fraud_count / total) * 100
    print(f"\n{label} Class Distribution:")
    print(f"  Total samples: {total:,}")
    print(f"  Fraud cases: {fraud_count:,} ({fraud_pct:.2f}%)")
    print(f"  Legitimate cases: {total - fraud_count:,} ({100 - fraud_pct:.2f}%)")
    print(f"  Imbalance ratio: 1:{(total - fraud_count) / fraud_count:.1f}")

# ===== STEP 1: LOAD AND PREPROCESS DATA =====
print("="*60)
print("STEP 1: LOADING AND PREPROCESSING DATA")
print("="*60)

checkpoint_file = f'processed_data_{RESAMPLING_STRATEGY}.pkl'

if checkpoint_exists(checkpoint_file):
    print(f"Found existing processed data checkpoint for '{RESAMPLING_STRATEGY}'. Loading...")
    checkpoint = load_checkpoint(checkpoint_file)
    X_train_scaled = checkpoint['X_train_scaled']
    X_test_scaled = checkpoint['X_test_scaled']
    y_train = checkpoint['y_train']
    y_test = checkpoint['y_test']
    scaler = checkpoint['scaler']
    feature_columns = checkpoint['feature_columns']
    print(f"✓ Loaded preprocessed data from checkpoint")
    print(f"  Training samples: {X_train_scaled.shape[0]:,}")
    print(f"  Test samples: {X_test_scaled.shape[0]:,}")
    print_class_distribution(y_train, "Training")
else:
    print("No checkpoint found. Processing data from scratch...")
    
    # Download the dataset
    path = kagglehub.dataset_download("ealaxi/paysim1")
    print(f"Dataset downloaded to: {path}")
    
    # Load the CSV
    csv_file = os.path.join(path, "PS_20174392719_1491204439457_log.csv")
    print("Loading CSV file...")
    df = pd.read_csv(csv_file)
    
    print(f"✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # OPTIONAL: Sample for laptop-friendly training
    SAMPLE_SIZE = 100000
    print(f"\n⚠ Sampling {SAMPLE_SIZE:,} rows for laptop-friendly training...")
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    # Feature engineering
    feature_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                      'oldbalanceDest', 'newbalanceDest']
    
    print("\nEncoding transaction types...")
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    feature_columns.append('type_encoded')
    
    # Add engineered features for better fraud detection
    print("Creating engineered features...")
    df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)  # +1 to avoid division by zero
    
    feature_columns.extend(['balance_change_orig', 'balance_change_dest', 'amount_to_balance_ratio'])
    
    X = df[feature_columns]
    y = df['isFraud']
    
    X = X.fillna(0)
    
    print_class_distribution(y, "Original")
    
    # Train-test split (BEFORE resampling - important!)
    print("\nSplitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]:,}")
    print(f"Test set size: {X_test.shape[0]:,}")
    print_class_distribution(y_train, "Training (before resampling)")
    
    # Feature scaling BEFORE resampling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ===== HANDLE CLASS IMBALANCE =====
    print("\n" + "-"*60)
    print(f"APPLYING RESAMPLING STRATEGY: {RESAMPLING_STRATEGY.upper()}")
    print("-"*60)
    
    if RESAMPLING_STRATEGY == 'smote':
        # SMOTE: Synthetic Minority Over-sampling Technique
        # Creates synthetic fraud examples by interpolating between existing ones
        print("Applying SMOTE (Synthetic Minority Over-sampling)...")
        print("  → Creates synthetic fraud examples")
        print("  → Helps model learn fraud patterns better")
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        
    elif RESAMPLING_STRATEGY == 'undersample':
        # Random Under-sampling: Reduces majority class
        # Faster training but loses legitimate transaction data
        print("Applying Random Under-sampling...")
        print("  → Reduces legitimate transactions to match fraud count")
        print("  → Faster training but loses data")
        
        rus = RandomUnderSampler(random_state=42)
        X_train_scaled, y_train = rus.fit_resample(X_train_scaled, y_train)
        
    elif RESAMPLING_STRATEGY == 'smotetomek':
        # SMOTE + Tomek Links: Over-sample minorities, clean boundaries
        # Best of both worlds but slowest
        print("Applying SMOTE + Tomek Links...")
        print("  → Over-samples fraud cases with SMOTE")
        print("  → Cleans class boundaries with Tomek links")
        print("  → Best quality but slower")
        
        smt = SMOTETomek(random_state=42)
        X_train_scaled, y_train = smt.fit_resample(X_train_scaled, y_train)
        
    else:  # 'none'
        print("No resampling applied - using original imbalanced data")
        print("  → Relies only on class_weight='balanced' in SVM")
    
    if RESAMPLING_STRATEGY != 'none':
        print_class_distribution(y_train, "Training (after resampling)")
    
    # Save checkpoint
    checkpoint_data = {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    save_checkpoint(checkpoint_data, checkpoint_file)
    print("✓ Data preprocessing complete")

# ===== STEP 2: TRAIN SVM MODEL =====
print("\n" + "="*60)
print("STEP 2: TRAINING SVM MODEL")
print("="*60)

model_file = f'trained_model_{RESAMPLING_STRATEGY}.pkl'

if checkpoint_exists(model_file):
    print("Found existing trained model. Loading...")
    svm_model = load_checkpoint(model_file)
    print("✓ Model loaded from checkpoint")
else:
    print("No trained model found. Training from scratch...")
    print(f"Training samples: {X_train_scaled.shape[0]:,}")
    print(f"Features: {X_train_scaled.shape[1]}")
    print("\nTraining SVM with RBF kernel...")
    
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',  # Still helpful even with resampling
        probability=True,
        random_state=42,
        verbose=True
    )
    
    start_time = time.time()
    svm_model.fit(X_train_scaled, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"\n✓ Model training complete in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    save_checkpoint(svm_model, model_file)
    
    # Save for deployment
    joblib.dump(svm_model, os.path.join(CHECKPOINT_DIR, f'fraud_detection_svm_{RESAMPLING_STRATEGY}.pkl'))
    joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, f'scaler_{RESAMPLING_STRATEGY}.pkl'))
    print("✓ Model and scaler saved for deployment")

# ===== STEP 3: MAKE PREDICTIONS =====
print("\n" + "="*60)
print("STEP 3: MAKING PREDICTIONS")
print("="*60)

pred_file = f'predictions_{RESAMPLING_STRATEGY}.pkl'

if checkpoint_exists(pred_file):
    print("Found existing predictions. Loading...")
    predictions = load_checkpoint(pred_file)
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
    save_checkpoint(predictions, pred_file)
    print("✓ Predictions complete")

# ===== STEP 4: COMPREHENSIVE EVALUATION =====
print("\n" + "="*60)
print("STEP 4: MODEL EVALUATION (IMBALANCE-AWARE METRICS)")
print("="*60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives (TN):  {tn:,} - Correctly identified legitimate transactions")
print(f"False Positives (FP): {fp:,} - Legitimate flagged as fraud (Type I error)")
print(f"False Negatives (FN): {fn:,} - Fraud missed (Type II error) ⚠️ CRITICAL")
print(f"True Positives (TP):  {tp:,} - Correctly caught fraud ✓")

# Key metrics for imbalanced data
print("\n" + "-"*60)
print("PRIMARY METRICS FOR FRAUD DETECTION:")
print("-"*60)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print(f"Precision (PPV):  {precision:.4f} - Of flagged fraud, what % is actually fraud?")
print(f"Recall (TPR):     {recall:.4f} - Of all fraud, what % did we catch? ⭐")
print(f"F1-Score:         {f1:.4f} - Harmonic mean of precision & recall")
print(f"ROC-AUC:          {roc_auc:.4f} - Overall discriminative ability")
print(f"PR-AUC:           {avg_precision:.4f} - Better for imbalanced data ⭐")

# Business metrics
fraud_caught_pct = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
false_alarm_rate = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0

print("\n" + "-"*60)
print("BUSINESS IMPACT METRICS:")
print("-"*60)
print(f"Fraud Detection Rate: {fraud_caught_pct:.2f}% - Caught {tp} out of {tp+fn} frauds")
print(f"False Alarm Rate:     {false_alarm_rate:.2f}% - {fp} false alarms out of {fp+tn} legitimate")
print(f"Missed Fraud Cases:   {fn} - These slip through! ⚠️")

print("\n" + "-"*60)
print("DETAILED CLASSIFICATION REPORT")
print("-"*60)
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Cost-sensitive analysis
print("\n" + "-"*60)
print("COST ANALYSIS (Assuming: Fraud = $100 loss, False Alarm = $1 cost)")
print("-"*60)
cost_per_fraud = 100
cost_per_false_alarm = 1

fraud_prevented = tp * cost_per_fraud
fraud_missed = fn * cost_per_fraud
false_alarm_cost = fp * cost_per_false_alarm

net_benefit = fraud_prevented - fraud_missed - false_alarm_cost

print(f"Fraud Prevented:    ${fraud_prevented:,.2f} ({tp} cases)")
print(f"Fraud Missed:      -${fraud_missed:,.2f} ({fn} cases)")
print(f"False Alarm Cost:  -${false_alarm_cost:,.2f} ({fp} cases)")
print(f"Net Benefit:        ${net_benefit:,.2f}")

# Save final results
final_results = {
    'strategy': RESAMPLING_STRATEGY,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'roc_auc': roc_auc,
    'pr_auc': avg_precision,
    'confusion_matrix': cm,
    'fraud_detection_rate': fraud_caught_pct,
    'false_alarm_rate': false_alarm_rate,
    'classification_report': classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'], output_dict=True)
}
save_checkpoint(final_results, f'final_results_{RESAMPLING_STRATEGY}.pkl')

print("\n" + "="*60)
print("✓ COMPLETE! All checkpoints saved")
print("="*60)
print(f"\nStrategy used: {RESAMPLING_STRATEGY.upper()}")
print("\n📊 KEY TAKEAWAYS:")
print(f"  - Recall (fraud caught): {recall:.2%} ⭐ Most important for fraud detection")
print(f"  - Precision (accuracy when flagging): {precision:.2%}")
print(f"  - PR-AUC: {avg_precision:.4f} (better metric than ROC-AUC for imbalanced data)")
print("\n💡 TO EXPERIMENT:")
print("  1. Change RESAMPLING_STRATEGY at top of script:")
print("     - 'none': Original imbalanced data")
print("     - 'smote': Synthetic over-sampling (recommended)")
print("     - 'undersample': Reduce majority class")
print("     - 'smotetomek': Combination approach (best quality)")
print("  2. Compare recall and PR-AUC across strategies")
print("  3. Choose based on your business priorities (catch more fraud vs fewer false alarms)")