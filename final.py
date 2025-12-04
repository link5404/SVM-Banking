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

# ===== CONFIGURATION =====
CHECKPOINT_DIR = "checkpoints"
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# IMBALANCE HANDLING STRATEGY
# Options: 'none', 'smote', 'undersample', 'smotetomek'
#RESAMPLING_STRATEGY = 'none'  # Change this to experiment
RESAMPLING_STRATEGIES = ['none', 'smote', 'undersample', 'smotetomek']   

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

def create_enhanced_features(df):
    """
    Comprehensive feature engineering for PaySim1 fraud detection
    """
    # ===== 1. ORIGINAL NUMERIC FEATURES =====
    feature_columns = [
        'amount', 
        'oldbalanceOrg', 
        'newbalanceOrig',
        'oldbalanceDest', 
        'newbalanceDest'
    ]
    
    # ===== 2. TRANSACTION TYPE FEATURES =====
    print("Creating transaction type features...")
    
    # Encode transaction type
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    feature_columns.append('type_encoded')
    
    # High-risk transaction types (fraud mainly in TRANSFER and CASH_OUT)
    df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    df['is_cashout'] = (df['type'] == 'CASH_OUT').astype(int)
    df['is_high_risk_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
    feature_columns.extend(['is_transfer', 'is_cashout', 'is_high_risk_type'])
    
    # ===== 3. TEMPORAL FEATURES =====
    print("Creating temporal features...")
    
    # Hour of day (0-23) and day of month (0-30)
    df['hour_of_day'] = df['step'] % 24
    df['day_of_month'] = df['step'] // 24
    
    # Time-based risk indicators
    df['is_night_transaction'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
    df['is_weekend'] = (df['day_of_month'] % 7 >= 5).astype(int)
    
    feature_columns.extend(['hour_of_day', 'day_of_month', 
                           'is_night_transaction', 'is_weekend'])
    
    # ===== 4. ACCOUNT TYPE FEATURES =====
    print("Creating account type features...")
    
    # Identify merchants (start with 'M', have zero balances)
    df['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
    df['orig_is_merchant'] = df['nameOrig'].str.startswith('M').astype(int)
    feature_columns.extend(['dest_is_merchant', 'orig_is_merchant'])
    
    # ===== 5. BALANCE-BASED FEATURES =====
    print("Creating balance-based features...")
    
    # Balance changes
    df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Ratios and percentages
    df['amount_to_balance_ratio_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_balance_ratio_dest'] = df['amount'] / (df['oldbalanceDest'] + 1)
    
    # Percentage of balance transferred
    df['pct_balance_transferred'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    
    feature_columns.extend([
        'balance_change_orig', 
        'balance_change_dest',
        'amount_to_balance_ratio_orig',
        'amount_to_balance_ratio_dest',
        'pct_balance_transferred'
    ])
    
    # ===== 6. FRAUD PATTERN INDICATORS =====
    print("Creating fraud pattern indicators...")
    
    # Account draining (transaction empties origin account)
    df['drains_origin_account'] = (
        (df['newbalanceOrig'] == 0) & 
        (df['oldbalanceOrg'] > 0)
    ).astype(int)
    
    # Large transaction relative to typical behavior
    df['is_large_transaction'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    
    # Suspicious balance behavior
    df['balance_mismatch_orig'] = (
        abs(df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']) > 0.01
    ).astype(int)
    
    df['balance_mismatch_dest'] = (
        abs(df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']) > 0.01
    ).astype(int)
    
    # Zero balance after high-value transaction (classic fraud pattern)
    df['high_value_to_zero'] = (
        (df['amount'] > 200000) & 
        (df['newbalanceOrig'] == 0)
    ).astype(int)
    
    feature_columns.extend([
        'drains_origin_account',
        'is_large_transaction',
        'balance_mismatch_orig',
        'balance_mismatch_dest',
        'high_value_to_zero'
    ])
    
    # ===== 7. INTERACTION FEATURES =====
    print("Creating interaction features...")
    
    # High-risk type + large amount
    df['high_risk_large_amount'] = (
        df['is_high_risk_type'] * df['is_large_transaction']
    )
    
    # Night transaction + account draining
    df['night_drain_interaction'] = (
        df['is_night_transaction'] * df['drains_origin_account']
    )
    
    feature_columns.extend(['high_risk_large_amount', 'night_drain_interaction'])
    
    # ===== 8. HANDLE MISSING VALUES AND INFINITIES =====
    print("Handling missing values and infinities...")
    df[feature_columns] = df[feature_columns].fillna(0)
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], 0)
    
    print(f"\n✓ Created {len(feature_columns)} features total")
    print(f"  Original numeric features: 5")
    print(f"  Engineered features: {len(feature_columns) - 5}")
    
    return df, feature_columns

# ===== STEP 1: LOAD AND PREPROCESS DATA =====
for RESAMPLING_STRATEGY in RESAMPLING_STRATEGIES:
    print("="*60)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("="*60)

    checkpoint_file = f'processed_data_{RESAMPLING_STRATEGY}_enhanced.pkl'

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
        print(f"  Features: {len(feature_columns)}")
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
        
        # Apply enhanced feature engineering
        df, feature_columns = create_enhanced_features(df)
        
        X = df[feature_columns]
        y = df['isFraud']
        
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
            
        else:  # 'none'
            print("No resampling applied")
        
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

    # ===== STEP 2: TRAIN SVM MODEL =====

    model_file = f'trained_model_{RESAMPLING_STRATEGY}_enhanced.pkl'

    if checkpoint_exists(model_file):
        print("Found existing trained model. Loading...")
        svm_model = load_checkpoint(model_file)
    else:
        print("No trained model found. Training from scratch...")
        print(f"Training samples: {X_train_scaled.shape[0]:,}")
        print(f"Features: {X_train_scaled.shape[1]}")
        print("\nTraining SVM with RBF kernel...")
        
        svm_model = SVC(
            kernel='rbf',
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
        print(f"\n✓ Model training complete in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        save_checkpoint(svm_model, model_file)
        
        # Save for deployment
        joblib.dump(svm_model, os.path.join(CHECKPOINT_DIR, f'fraud_detection_svm_{RESAMPLING_STRATEGY}_enhanced.pkl'))
        joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, f'scaler_{RESAMPLING_STRATEGY}_enhanced.pkl'))
        print("✓ Model and scaler saved for deployment")

    # ===== STEP 3: MAKE PREDICTIONS =====
    print("\n" + "="*60)
    print("STEP 3: MAKING PREDICTIONS")
    print("="*60)

    pred_file = f'predictions_{RESAMPLING_STRATEGY}_enhanced.pkl'

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

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)


    # Business metrics
    fraud_caught_pct = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    false_alarm_rate = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0

    # Cost-sensitive analysis
    cost_per_fraud = 100
    cost_per_false_alarm = 1

    fraud_prevented = tp * cost_per_fraud
    fraud_missed = fn * cost_per_fraud
    false_alarm_cost = fp * cost_per_false_alarm

    net_benefit = fraud_prevented - fraud_missed - false_alarm_cost

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
        'num_features': len(feature_columns),
        'classification_report': classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'], output_dict=True)
    }
    save_checkpoint(final_results, f'final_results_{RESAMPLING_STRATEGY}_enhanced.pkl')
