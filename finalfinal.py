import kagglehub
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import os
import joblib
import pickle
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ===== CONFIGURATION =====
CHECKPOINT_DIR = "checkpoints"
MODELS_DIR = "trained_models"
PLOTS_DIR = "plots"
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
Path(MODELS_DIR).mkdir(exist_ok=True)
Path(PLOTS_DIR).mkdir(exist_ok=True)

RESAMPLING_STRATEGIES = ['none', 'smote', 'undersample', 'smotetomek']
KERNEL_TYPES = ['linear', 'rbf', 'poly', 'sigmoid']
ALL_RESULTS = []

# ===== UTILITY FUNCTIONS =====
def save_checkpoint(data, filename):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(filename):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def checkpoint_exists(filename):
    return os.path.exists(os.path.join(CHECKPOINT_DIR, filename))

def save_production_model(model, scaler, feature_columns, metadata, model_name):
    """Save complete model package for production inference"""
    model_path = os.path.join(MODELS_DIR, model_name)
    Path(model_path).mkdir(exist_ok=True)
    
    joblib.dump(model, os.path.join(model_path, "model.pkl"))
    joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))
    
    with open(os.path.join(model_path, "feature_columns.pkl"), 'wb') as f:
        pickle.dump(feature_columns, f)
    
    with open(os.path.join(model_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return model_path

def load_production_model(model_name):
    """Load complete model package for inference"""
    model_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(os.path.join(model_path, "model.pkl"))
    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
    
    with open(os.path.join(model_path, "feature_columns.pkl"), 'rb') as f:
        feature_columns = pickle.load(f)
    
    with open(os.path.join(model_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'metadata': metadata
    }

def create_enhanced_features(df):
    """Feature engineering for PaySim1 fraud detection"""
    feature_columns = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest'
    ]
    
    # Transaction type features
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    df['is_cashout'] = (df['type'] == 'CASH_OUT').astype(int)
    df['is_high_risk_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
    feature_columns.extend(['type_encoded', 'is_transfer', 'is_cashout', 'is_high_risk_type'])
    
    # Temporal features
    df['hour_of_day'] = df['step'] % 24
    df['day_of_month'] = df['step'] // 24
    df['is_night_transaction'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
    df['is_weekend'] = (df['day_of_month'] % 7 >= 5).astype(int)
    feature_columns.extend(['hour_of_day', 'day_of_month', 'is_night_transaction', 'is_weekend'])
    
    # Account type features
    df['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
    df['orig_is_merchant'] = df['nameOrig'].str.startswith('M').astype(int)
    feature_columns.extend(['dest_is_merchant', 'orig_is_merchant'])
    
    # Balance-based features
    df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['amount_to_balance_ratio_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_balance_ratio_dest'] = df['amount'] / (df['oldbalanceDest'] + 1)
    df['pct_balance_transferred'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    feature_columns.extend([
        'balance_change_orig', 'balance_change_dest',
        'amount_to_balance_ratio_orig', 'amount_to_balance_ratio_dest',
        'pct_balance_transferred'
    ])
    
    # Fraud pattern indicators
    df['drains_origin_account'] = (
        (df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)
    ).astype(int)
    df['is_large_transaction'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    df['balance_mismatch_orig'] = (
        abs(df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']) > 0.01
    ).astype(int)
    df['balance_mismatch_dest'] = (
        abs(df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']) > 0.01
    ).astype(int)
    df['high_value_to_zero'] = (
        (df['amount'] > 200000) & (df['newbalanceOrig'] == 0)
    ).astype(int)
    feature_columns.extend([
        'drains_origin_account', 'is_large_transaction',
        'balance_mismatch_orig', 'balance_mismatch_dest',
        'high_value_to_zero'
    ])
    
    # Interaction features
    df['high_risk_large_amount'] = df['is_high_risk_type'] * df['is_large_transaction']
    df['night_drain_interaction'] = df['is_night_transaction'] * df['drains_origin_account']
    feature_columns.extend(['high_risk_large_amount', 'night_drain_interaction'])
    
    # Handle missing values and infinities
    df[feature_columns] = df[feature_columns].fillna(0)
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], 0)
    
    return df, feature_columns

def plot_training_results(results_df):
    """Create comprehensive visualization of training results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SVM Fraud Detection: Training Results Comparison', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    plot_data = results_df.copy()
    plot_data['config'] = plot_data['resampling'] + '_' + plot_data['kernel']
    
    # 1. F1-Score comparison
    ax = axes[0, 0]
    pivot = plot_data.pivot(index='resampling', columns='kernel', values='f1')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'F1-Score'})
    ax.set_title('F1-Score by Resampling & Kernel', fontweight='bold')
    ax.set_xlabel('Kernel Type')
    ax.set_ylabel('Resampling Strategy')
    
    # 2. ROC-AUC comparison
    ax = axes[0, 1]
    pivot = plot_data.pivot(index='resampling', columns='kernel', values='roc_auc')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'ROC-AUC'})
    ax.set_title('ROC-AUC by Resampling & Kernel', fontweight='bold')
    ax.set_xlabel('Kernel Type')
    ax.set_ylabel('Resampling Strategy')
    
    # 3. Precision vs Recall
    ax = axes[0, 2]
    for resampling in RESAMPLING_STRATEGIES:
        subset = plot_data[plot_data['resampling'] == resampling]
        ax.scatter(subset['recall'], subset['precision'], label=resampling, s=100, alpha=0.7)
        for _, row in subset.iterrows():
            ax.annotate(row['kernel'], (row['recall'], row['precision']), 
                       fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision-Recall Trade-off', fontweight='bold')
    ax.legend(title='Resampling')
    ax.grid(True, alpha=0.3)
    
    # 4. Training time comparison
    ax = axes[1, 0]
    pivot = plot_data.pivot(index='resampling', columns='kernel', values='training_time')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', ax=ax, cbar_kws={'label': 'Time (s)'})
    ax.set_title('Training Time (seconds)', fontweight='bold')
    ax.set_xlabel('Kernel Type')
    ax.set_ylabel('Resampling Strategy')
    
    # 5. Top 10 configurations by F1
    ax = axes[1, 1]
    top_10 = plot_data.nlargest(10, 'f1')
    y_pos = np.arange(len(top_10))
    colors = plt.cm.RdYlGn(top_10['f1'] / top_10['f1'].max())
    ax.barh(y_pos, top_10['f1'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['resampling']}_{row['kernel']}" for _, row in top_10.iterrows()], fontsize=9)
    ax.set_xlabel('F1-Score', fontweight='bold')
    ax.set_title('Top 10 Configurations by F1-Score', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 6. Metrics comparison for best model
    ax = axes[1, 2]
    best = plot_data.loc[plot_data['f1'].idxmax()]
    metrics = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    values = [best[m] for m in metrics]
    colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics)))
    bars = ax.bar(metrics, values, color=colors_bar)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'Best Model Metrics\n({best["resampling"]}_{best["kernel"]})', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training results plot saved to: {plot_path}")
    plt.close()

# ===== MAIN TRAINING LOOP =====
print("="*80)
print("SVM FRAUD DETECTION TRAINING")
print(f"Experiments: {len(RESAMPLING_STRATEGIES)} resampling × {len(KERNEL_TYPES)} kernels = {len(RESAMPLING_STRATEGIES) * len(KERNEL_TYPES)} total")
print("="*80)

for RESAMPLING_STRATEGY in RESAMPLING_STRATEGIES:
    print(f"\n[{RESAMPLING_STRATEGY.upper()}]", end=" ")
    
    checkpoint_file = f'processed_data_{RESAMPLING_STRATEGY}_enhanced.pkl'
    
    if checkpoint_exists(checkpoint_file):
        checkpoint = load_checkpoint(checkpoint_file)
        X_train_scaled = checkpoint['X_train_scaled']
        X_test_scaled = checkpoint['X_test_scaled']
        y_train = checkpoint['y_train']
        y_test = checkpoint['y_test']
        scaler = checkpoint['scaler']
        feature_columns = checkpoint['feature_columns']
        print(f"Loaded preprocessed data ({X_train_scaled.shape[0]:,} train samples)")
    else:
        print("Processing data...")
        path = kagglehub.dataset_download("ealaxi/paysim1")
        csv_file = os.path.join(path, "PS_20174392719_1491204439457_log.csv")
        df = pd.read_csv(csv_file)
        
        SAMPLE_SIZE = 100000
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        
        df, feature_columns = create_enhanced_features(df)
        X = df[feature_columns]
        y = df['isFraud']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if RESAMPLING_STRATEGY == 'smote':
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        elif RESAMPLING_STRATEGY == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_train_scaled, y_train = rus.fit_resample(X_train_scaled, y_train)
        elif RESAMPLING_STRATEGY == 'smotetomek':
            smt = SMOTETomek(random_state=42)
            X_train_scaled, y_train = smt.fit_resample(X_train_scaled, y_train)
        
        checkpoint_data = {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_columns': feature_columns
        }
        save_checkpoint(checkpoint_data, checkpoint_file)
    
    for KERNEL in KERNEL_TYPES:
        print(f"  {KERNEL}:", end=" ")
        
        model_file = f'trained_model_{RESAMPLING_STRATEGY}_{KERNEL}_enhanced.pkl'
        
        if checkpoint_exists(model_file):
            svm_model = load_checkpoint(model_file)
            training_time = 0
            print("loaded", end=" ")
        else:
            if KERNEL == 'poly':
                svm_model = SVC(kernel='poly', degree=3, C=1.0, gamma='scale',
                               class_weight='balanced', probability=True, random_state=42)
            else:
                svm_model = SVC(kernel=KERNEL, C=1.0,
                               gamma='scale' if KERNEL in ['rbf', 'sigmoid'] else 'auto',
                               class_weight='balanced', probability=True, random_state=42)
            
            start_time = time.time()
            svm_model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            save_checkpoint(svm_model, model_file)
            print(f"trained ({training_time:.1f}s)", end=" ")
        
        pred_file = f'predictions_{RESAMPLING_STRATEGY}_{KERNEL}_enhanced.pkl'
        
        if checkpoint_exists(pred_file):
            predictions = load_checkpoint(pred_file)
            y_pred = predictions['y_pred']
            y_pred_proba = predictions['y_pred_proba']
        else:
            y_pred = svm_model.predict(X_test_scaled)
            y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
            save_checkpoint({'y_pred': y_pred, 'y_pred_proba': y_pred_proba}, pred_file)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save production model
        model_name = f"{RESAMPLING_STRATEGY}_{KERNEL}"
        metadata = {
            'resampling_strategy': RESAMPLING_STRATEGY,
            'kernel': KERNEL,
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc)
            },
            'confusion_matrix': {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            },
            'training_samples': int(X_train_scaled.shape[0]),
            'test_samples': int(X_test_scaled.shape[0]),
            'n_features': len(feature_columns)
        }
        save_production_model(svm_model, scaler, feature_columns, metadata, model_name)
        
        result = {
            'resampling': RESAMPLING_STRATEGY,
            'kernel': KERNEL,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'training_time': training_time
        }
        ALL_RESULTS.append(result)
        
        print(f"F1={f1:.3f}")

# ===== RESULTS & VISUALIZATION =====
print("\n" + "="*80)
print("TRAINING COMPLETE - GENERATING RESULTS")
print("="*80)

results_df = pd.DataFrame(ALL_RESULTS)
results_df = results_df.sort_values('f1', ascending=False)

save_checkpoint(results_df, 'all_experiments_results.pkl')
results_df.to_csv(os.path.join(CHECKPOINT_DIR, 'all_experiments_results.csv'), index=False)

print("\n" + results_df.to_string(index=False))

print("\n" + "="*80)
print("TOP 5 MODELS BY F1-SCORE")
print("="*80)
for idx, row in results_df.head(5).iterrows():
    print(f"{idx+1}. {row['resampling'].upper()}_{row['kernel'].upper()}: "
          f"F1={row['f1']:.4f} | Precision={row['precision']:.4f} | "
          f"Recall={row['recall']:.4f} | ROC-AUC={row['roc_auc']:.4f}")

# Generate plots
plot_training_results(results_df)

print("\n✓ All models saved to:", MODELS_DIR)
print("✓ Results saved to:", os.path.join(CHECKPOINT_DIR, 'all_experiments_results.csv'))
print("✓ Visualization saved to:", PLOTS_DIR)