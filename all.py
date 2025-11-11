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
RESAMPLING_STRATEGY = 'smote'

# KERNELS TO COMPARE
KERNELS = ['rbf', 'linear', 'poly']

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

# ===== STEP 1: LOAD AND PREPROCESS DATA (ONCE FOR ALL KERNELS) =====
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
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    
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
        print("Applying SMOTE (Synthetic Minority Over-sampling)...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        
    elif RESAMPLING_STRATEGY == 'undersample':
        print("Applying Random Under-sampling...")
        rus = RandomUnderSampler(random_state=42)
        X_train_scaled, y_train = rus.fit_resample(X_train_scaled, y_train)
        
    elif RESAMPLING_STRATEGY == 'smotetomek':
        print("Applying SMOTE + Tomek Links...")
        smt = SMOTETomek(random_state=42)
        X_train_scaled, y_train = smt.fit_resample(X_train_scaled, y_train)
    
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

# ===== STEP 2: TRAIN AND EVALUATE ALL KERNELS =====
print("\n" + "="*60)
print("STEP 2: TRAINING AND EVALUATING ALL KERNELS")
print("="*60)

results_comparison = {}

for kernel in KERNELS:
    print("\n" + "="*60)
    print(f"TRAINING WITH KERNEL: {kernel.upper()}")
    print("="*60)
    
    model_file = f'trained_model_{RESAMPLING_STRATEGY}_{kernel}.pkl'
    pred_file = f'predictions_{RESAMPLING_STRATEGY}_{kernel}.pkl'
    
    # Train or load model
    if checkpoint_exists(model_file):
        print(f"Loading existing {kernel} model...")
        svm_model = load_checkpoint(model_file)
    else:
        print(f"Training new {kernel} model...")
        
        # Kernel-specific parameters
        if kernel == 'poly':
            svm_model = SVC(
                kernel='poly',
                degree=3,
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
                verbose=True
            )
        else:
            svm_model = SVC(
                kernel=kernel,
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
                verbose=True
            )
        
        start_time = time.time()
        svm_model.fit(X_train_scaled, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"\n✓ {kernel.upper()} model training complete in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        save_checkpoint(svm_model, model_file)
        joblib.dump(svm_model, os.path.join(CHECKPOINT_DIR, f'fraud_detection_svm_{RESAMPLING_STRATEGY}_{kernel}.pkl'))
    
    # Make predictions
    if checkpoint_exists(pred_file):
        print(f"Loading existing {kernel} predictions...")
        predictions = load_checkpoint(pred_file)
        y_pred = predictions['y_pred']
        y_pred_proba = predictions['y_pred_proba']
    else:
        print(f"Running {kernel} predictions on test set...")
        y_pred = svm_model.predict(X_test_scaled)
        y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
        
        predictions = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        save_checkpoint(predictions, pred_file)
    
    # Evaluate
    print(f"\nEvaluating {kernel.upper()} model...")
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    fraud_caught_pct = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    false_alarm_rate = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
    
    # Store results
    results_comparison[kernel] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': avg_precision,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'fraud_detection_rate': fraud_caught_pct,
        'false_alarm_rate': false_alarm_rate
    }
    
    print(f"\n{kernel.upper()} Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f} ⭐")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {avg_precision:.4f} ⭐")
    print(f"  Fraud Detection Rate: {fraud_caught_pct:.2f}%")
    print(f"  False Alarm Rate:     {false_alarm_rate:.2f}%")

# Save comparison results
save_checkpoint(results_comparison, f'kernel_comparison_{RESAMPLING_STRATEGY}.pkl')

# ===== STEP 3: VISUALIZE COMPARISON =====
print("\n" + "="*60)
print("STEP 3: VISUALIZING KERNEL COMPARISON")
print("="*60)

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'SVM Kernel Comparison - Fraud Detection\n(Resampling: {RESAMPLING_STRATEGY.upper()})', 
             fontsize=16, fontweight='bold')

kernels_display = [k.upper() for k in KERNELS]
colors = ['#3498db', '#e74c3c', '#2ecc71']

# 1. Recall (Most Important for Fraud Detection)
ax = axes[0, 0]
recall_values = [results_comparison[k]['recall'] for k in KERNELS]
bars = ax.bar(kernels_display, recall_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Recall (Fraud Detection Rate)', fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, recall_values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Precision
ax = axes[0, 1]
precision_values = [results_comparison[k]['precision'] for k in KERNELS]
bars = ax.bar(kernels_display, precision_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Precision (Accuracy When Flagging)', fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, precision_values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. F1-Score
ax = axes[0, 2]
f1_values = [results_comparison[k]['f1'] for k in KERNELS]
bars = ax.bar(kernels_display, f1_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('F1-Score (Balanced Metric)', fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, f1_values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# 4. PR-AUC (Best for Imbalanced Data)
ax = axes[1, 0]
pr_auc_values = [results_comparison[k]['pr_auc'] for k in KERNELS]
bars = ax.bar(kernels_display, pr_auc_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('PR-AUC (Best for Imbalanced Data) ⭐', fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, pr_auc_values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# 5. Fraud Detection Rate (%)
ax = axes[1, 1]
fraud_det_values = [results_comparison[k]['fraud_detection_rate'] for k in KERNELS]
bars = ax.bar(kernels_display, fraud_det_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Fraud Detection Rate', fontweight='bold')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, fraud_det_values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# 6. False Alarm Rate (%)
ax = axes[1, 2]
false_alarm_values = [results_comparison[k]['false_alarm_rate'] for k in KERNELS]
bars = ax.bar(kernels_display, false_alarm_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('False Alarm Rate (Lower is Better)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, false_alarm_values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, f'kernel_comparison_{RESAMPLING_STRATEGY}.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved to {CHECKPOINT_DIR}/kernel_comparison_{RESAMPLING_STRATEGY}.png")
plt.show()

# ===== FINAL SUMMARY =====
print("\n" + "="*60)
print("FINAL KERNEL COMPARISON SUMMARY")
print("="*60)

print(f"\nResampling Strategy: {RESAMPLING_STRATEGY.upper()}\n")

# Create comparison table
comparison_df = pd.DataFrame({
    'Kernel': kernels_display,
    'Recall ⭐': [f"{results_comparison[k]['recall']:.4f}" for k in KERNELS],
    'Precision': [f"{results_comparison[k]['precision']:.4f}" for k in KERNELS],
    'F1-Score': [f"{results_comparison[k]['f1']:.4f}" for k in KERNELS],
    'PR-AUC ⭐': [f"{results_comparison[k]['pr_auc']:.4f}" for k in KERNELS],
    'Fraud Caught (%)': [f"{results_comparison[k]['fraud_detection_rate']:.2f}%" for k in KERNELS],
    'False Alarms (%)': [f"{results_comparison[k]['false_alarm_rate']:.2f}%" for k in KERNELS]
})

print(comparison_df.to_string(index=False))

# Determine best kernel
best_recall_kernel = max(KERNELS, key=lambda k: results_comparison[k]['recall'])
best_pr_auc_kernel = max(KERNELS, key=lambda k: results_comparison[k]['pr_auc'])
best_f1_kernel = max(KERNELS, key=lambda k: results_comparison[k]['f1'])

print("\n" + "-"*60)
print("RECOMMENDATIONS:")
print("-"*60)
print(f"🏆 Best Recall (Catch Most Fraud):     {best_recall_kernel.upper()} - {results_comparison[best_recall_kernel]['recall']:.4f}")
print(f"🏆 Best PR-AUC (Overall Performance):  {best_pr_auc_kernel.upper()} - {results_comparison[best_pr_auc_kernel]['pr_auc']:.4f}")
print(f"🏆 Best F1-Score (Balanced):           {best_f1_kernel.upper()} - {results_comparison[best_f1_kernel]['f1']:.4f}")

print("\n💡 INTERPRETATION:")
print("  - RBF: Non-linear kernel, good for complex patterns (default choice)")
print("  - Linear: Fast, interpretable, good for linearly separable data")
print("  - Poly: Captures polynomial relationships, can overfit")
print("\n  → For fraud detection, prioritize RECALL and PR-AUC")
print(f"  → Recommended kernel: {best_pr_auc_kernel.upper()}")

print("\n✓ COMPLETE! All kernels trained, evaluated, and compared.")
print("="*60)