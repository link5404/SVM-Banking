import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

CHECKPOINT_DIR = "checkpoints"

def load_checkpoint(filename):
    """Load checkpoint from disk"""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

# ===== 1. LOAD ALL RESULTS =====
print("="*80)
print("COMPREHENSIVE IMBALANCED DATA ANALYSIS")
print("="*80)

results_df = load_checkpoint('all_experiments_results.pkl')

if results_df is None:
    print("❌ Error: Results file not found. Run the training script first.")
    exit()

print(f"\n✓ Loaded results for {len(results_df)} experiments")
print(f"  Resampling strategies: {results_df['resampling'].unique().tolist()}")
print(f"  Kernel types: {results_df['kernel'].unique().tolist()}")

# ===== 2. CALCULATE ADDITIONAL IMBALANCED METRICS =====
print("\n" + "-"*80)
print("CALCULATING IMBALANCED-SPECIFIC METRICS")
print("-"*80)

# Add derived metrics
results_df['accuracy'] = (results_df['tp'] + results_df['tn']) / (
    results_df['tp'] + results_df['tn'] + results_df['fp'] + results_df['fn']
)
results_df['specificity'] = results_df['tn'] / (results_df['tn'] + results_df['fp'])
results_df['false_positive_rate'] = results_df['fp'] / (results_df['fp'] + results_df['tn'])
results_df['false_negative_rate'] = results_df['fn'] / (results_df['fn'] + results_df['tp'])
results_df['fraud_detection_rate'] = results_df['recall']  # Same as recall
results_df['false_alarm_rate'] = results_df['false_positive_rate']

# Balanced Accuracy (important for imbalanced data!)
results_df['balanced_accuracy'] = (results_df['recall'] + results_df['specificity']) / 2

# Geometric Mean (another important metric for imbalanced data)
results_df['geometric_mean'] = np.sqrt(results_df['recall'] * results_df['specificity'])

# Matthews Correlation Coefficient (best single metric for imbalanced data)
results_df['mcc'] = (
    (results_df['tp'] * results_df['tn'] - results_df['fp'] * results_df['fn']) /
    np.sqrt((results_df['tp'] + results_df['fp']) * 
            (results_df['tp'] + results_df['fn']) * 
            (results_df['tn'] + results_df['fp']) * 
            (results_df['tn'] + results_df['fn']))
)

# Cohen's Kappa
po = results_df['accuracy']  # Observed agreement
pe = (
    ((results_df['tp'] + results_df['fn']) * (results_df['tp'] + results_df['fp']) +
     (results_df['fp'] + results_df['tn']) * (results_df['fn'] + results_df['tn'])) /
    (results_df['tp'] + results_df['tn'] + results_df['fp'] + results_df['fn'])**2
)
results_df['cohens_kappa'] = (po - pe) / (1 - pe)

# Business metrics (assuming fraud cost = $100, false alarm cost = $1)
FRAUD_COST = 100
FALSE_ALARM_COST = 1

results_df['fraud_prevented_value'] = results_df['tp'] * FRAUD_COST
results_df['fraud_missed_value'] = results_df['fn'] * FRAUD_COST
results_df['false_alarm_cost'] = results_df['fp'] * FALSE_ALARM_COST
results_df['net_benefit'] = (
    results_df['fraud_prevented_value'] - 
    results_df['fraud_missed_value'] - 
    results_df['false_alarm_cost']
)

print("✓ Calculated additional metrics:")
print("  - Balanced Accuracy")
print("  - Geometric Mean")
print("  - Matthews Correlation Coefficient (MCC)")
print("  - Cohen's Kappa")
print("  - Business Value Metrics")

# ===== 3. DETAILED METRICS TABLE =====
print("\n" + "="*80)
print("DETAILED METRICS TABLE (Sorted by F1-Score)")
print("="*80)

display_cols = [
    'resampling', 'kernel', 
    'precision', 'recall', 'f1', 
    'balanced_accuracy', 'geometric_mean', 'mcc',
    'roc_auc', 'pr_auc',
    'specificity', 'false_alarm_rate'
]

results_sorted = results_df.sort_values('f1', ascending=False)
print("\n" + results_sorted[display_cols].to_string(index=False))

# ===== 4. BEST MODELS BY DIFFERENT CRITERIA =====
print("\n" + "="*80)
print("BEST MODELS BY DIFFERENT CRITERIA")
print("="*80)

criteria = {
    'F1-Score': 'f1',
    'Precision': 'precision',
    'Recall': 'recall',
    'ROC-AUC': 'roc_auc',
    'PR-AUC': 'pr_auc',
    'Balanced Accuracy': 'balanced_accuracy',
    'Geometric Mean': 'geometric_mean',
    'MCC': 'mcc',
    'Net Business Benefit': 'net_benefit'
}

for criterion_name, criterion_col in criteria.items():
    best = results_df.loc[results_df[criterion_col].idxmax()]
    print(f"\n🏆 Best {criterion_name}: {best['resampling'].upper()} + {best['kernel'].upper()}")
    print(f"   Value: {best[criterion_col]:.4f}")
    print(f"   Precision: {best['precision']:.4f} | Recall: {best['recall']:.4f} | F1: {best['f1']:.4f}")

# ===== 5. CONFUSION MATRIX ANALYSIS =====
print("\n" + "="*80)
print("CONFUSION MATRIX BREAKDOWN (Top 3 by F1)")
print("="*80)

top_3 = results_df.nlargest(3, 'f1')
for idx, row in top_3.iterrows():
    print(f"\n#{idx+1}: {row['resampling'].upper()} + {row['kernel'].upper()}")
    print(f"   F1-Score: {row['f1']:.4f}")
    print("\n   Confusion Matrix:")
    print(f"   ┌─────────────────┬──────────────┬──────────────┐")
    print(f"   │                 │  Predicted   │  Predicted   │")
    print(f"   │                 │  Legitimate  │    Fraud     │")
    print(f"   ├─────────────────┼──────────────┼──────────────┤")
    print(f"   │ Actual          │              │              │")
    print(f"   │ Legitimate      │  {row['tn']:>10,}  │  {row['fp']:>10,}  │")
    print(f"   ├─────────────────┼──────────────┼──────────────┤")
    print(f"   │ Actual          │              │              │")
    print(f"   │ Fraud           │  {row['fn']:>10,}  │  {row['tp']:>10,}  │")
    print(f"   └─────────────────┴──────────────┴──────────────┘")
    print(f"\n   Key Metrics:")
    print(f"   • Frauds Caught: {row['tp']:,} out of {row['tp'] + row['fn']:,} ({row['recall']*100:.1f}%)")
    print(f"   • False Alarms: {row['fp']:,} out of {row['fp'] + row['tn']:,} ({row['false_alarm_rate']*100:.2f}%)")
    print(f"   • Precision: {row['precision']:.4f} (When model says fraud, it's right {row['precision']*100:.1f}% of the time)")
    print(f"   • Net Benefit: ${row['net_benefit']:,.2f}")

# ===== 6. VISUALIZATIONS =====
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 24))

# 1. F1-Score Heatmap
ax1 = plt.subplot(4, 3, 1)
pivot_f1 = results_df.pivot(index='resampling', columns='kernel', values='f1')
sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax1, 
            vmin=0, vmax=1, cbar_kws={'label': 'F1-Score'})
ax1.set_title('F1-Score by Resampling & Kernel', fontsize=14, fontweight='bold')
ax1.set_xlabel('Kernel Type')
ax1.set_ylabel('Resampling Strategy')

# 2. Precision Heatmap
ax2 = plt.subplot(4, 3, 2)
pivot_prec = results_df.pivot(index='resampling', columns='kernel', values='precision')
sns.heatmap(pivot_prec, annot=True, fmt='.4f', cmap='Blues', ax=ax2,
            vmin=0, vmax=1, cbar_kws={'label': 'Precision'})
ax2.set_title('Precision by Resampling & Kernel', fontsize=14, fontweight='bold')
ax2.set_xlabel('Kernel Type')
ax2.set_ylabel('Resampling Strategy')

# 3. Recall Heatmap
ax3 = plt.subplot(4, 3, 3)
pivot_recall = results_df.pivot(index='resampling', columns='kernel', values='recall')
sns.heatmap(pivot_recall, annot=True, fmt='.4f', cmap='Oranges', ax=ax3,
            vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
ax3.set_title('Recall (Fraud Detection Rate)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Kernel Type')
ax3.set_ylabel('Resampling Strategy')

# 4. ROC-AUC Heatmap
ax4 = plt.subplot(4, 3, 4)
pivot_roc = results_df.pivot(index='resampling', columns='kernel', values='roc_auc')
sns.heatmap(pivot_roc, annot=True, fmt='.4f', cmap='Purples', ax=ax4,
            vmin=0.5, vmax=1, cbar_kws={'label': 'ROC-AUC'})
ax4.set_title('ROC-AUC by Resampling & Kernel', fontsize=14, fontweight='bold')
ax4.set_xlabel('Kernel Type')
ax4.set_ylabel('Resampling Strategy')

# 5. PR-AUC Heatmap
ax5 = plt.subplot(4, 3, 5)
pivot_pr = results_df.pivot(index='resampling', columns='kernel', values='pr_auc')
sns.heatmap(pivot_pr, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax5,
            vmin=0, vmax=1, cbar_kws={'label': 'PR-AUC'})
ax5.set_title('PR-AUC by Resampling & Kernel', fontsize=14, fontweight='bold')
ax5.set_xlabel('Kernel Type')
ax5.set_ylabel('Resampling Strategy')

# 6. Matthews Correlation Coefficient
ax6 = plt.subplot(4, 3, 6)
pivot_mcc = results_df.pivot(index='resampling', columns='kernel', values='mcc')
sns.heatmap(pivot_mcc, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax6,
            vmin=-1, vmax=1, cbar_kws={'label': 'MCC'})
ax6.set_title('Matthews Correlation Coefficient', fontsize=14, fontweight='bold')
ax6.set_xlabel('Kernel Type')
ax6.set_ylabel('Resampling Strategy')

# 7. Balanced Accuracy
ax7 = plt.subplot(4, 3, 7)
pivot_bacc = results_df.pivot(index='resampling', columns='kernel', values='balanced_accuracy')
sns.heatmap(pivot_bacc, annot=True, fmt='.4f', cmap='viridis', ax=ax7,
            vmin=0.5, vmax=1, cbar_kws={'label': 'Balanced Accuracy'})
ax7.set_title('Balanced Accuracy (Critical for Imbalanced Data)', fontsize=14, fontweight='bold')
ax7.set_xlabel('Kernel Type')
ax7.set_ylabel('Resampling Strategy')

# 8. Geometric Mean
ax8 = plt.subplot(4, 3, 8)
pivot_gm = results_df.pivot(index='resampling', columns='kernel', values='geometric_mean')
sns.heatmap(pivot_gm, annot=True, fmt='.4f', cmap='coolwarm', ax=ax8,
            vmin=0, vmax=1, cbar_kws={'label': 'Geometric Mean'})
ax8.set_title('Geometric Mean', fontsize=14, fontweight='bold')
ax8.set_xlabel('Kernel Type')
ax8.set_ylabel('Resampling Strategy')

# 9. False Alarm Rate
ax9 = plt.subplot(4, 3, 9)
pivot_far = results_df.pivot(index='resampling', columns='kernel', values='false_alarm_rate')
sns.heatmap(pivot_far, annot=True, fmt='.4f', cmap='Reds_r', ax=ax9,
            vmin=0, vmax=0.5, cbar_kws={'label': 'False Alarm Rate'})
ax9.set_title('False Alarm Rate (Lower is Better)', fontsize=14, fontweight='bold')
ax9.set_xlabel('Kernel Type')
ax9.set_ylabel('Resampling Strategy')

# 10. Bar chart: Top 10 configurations by F1
ax10 = plt.subplot(4, 3, 10)
top_10 = results_df.nlargest(10, 'f1').copy()
top_10['config'] = top_10['resampling'] + '\n' + top_10['kernel']
colors = sns.color_palette('RdYlGn', len(top_10))
bars = ax10.barh(range(len(top_10)), top_10['f1'], color=colors)
ax10.set_yticks(range(len(top_10)))
ax10.set_yticklabels(top_10['config'])
ax10.set_xlabel('F1-Score')
ax10.set_title('Top 10 Configurations by F1-Score', fontsize=14, fontweight='bold')
ax10.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, top_10['f1'])):
    ax10.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
              f'{val:.4f}', va='center', fontsize=9)

# 11. Precision-Recall Tradeoff
ax11 = plt.subplot(4, 3, 11)
for resampling in results_df['resampling'].unique():
    subset = results_df[results_df['resampling'] == resampling]
    ax11.scatter(subset['recall'], subset['precision'], 
                s=100, alpha=0.6, label=resampling)
ax11.set_xlabel('Recall (Fraud Detection Rate)')
ax11.set_ylabel('Precision')
ax11.set_title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
ax11.legend()
ax11.grid(True, alpha=0.3)

# 12. Net Business Benefit
ax12 = plt.subplot(4, 3, 12)
pivot_benefit = results_df.pivot(index='resampling', columns='kernel', values='net_benefit')
sns.heatmap(pivot_benefit, annot=True, fmt=',.0f', cmap='RdYlGn', ax=ax12,
            cbar_kws={'label': 'Net Benefit ($)'})
ax12.set_title('Net Business Benefit (Fraud=$100, FalseAlarm=$1)', 
               fontsize=14, fontweight='bold')
ax12.set_xlabel('Kernel Type')
ax12.set_ylabel('Resampling Strategy')

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'comprehensive_analysis.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Saved comprehensive_analysis.png")

# ===== 7. RESAMPLING STRATEGY COMPARISON =====
print("\n" + "="*80)
print("RESAMPLING STRATEGY COMPARISON (Averaged Across All Kernels)")
print("="*80)

resampling_avg = results_df.groupby('resampling').agg({
    'precision': 'mean',
    'recall': 'mean',
    'f1': 'mean',
    'balanced_accuracy': 'mean',
    'geometric_mean': 'mean',
    'mcc': 'mean',
    'roc_auc': 'mean',
    'pr_auc': 'mean',
    'false_alarm_rate': 'mean',
    'net_benefit': 'mean'
}).round(4)

print("\n" + resampling_avg.to_string())

# ===== 8. KERNEL COMPARISON =====
print("\n" + "="*80)
print("KERNEL COMPARISON (Averaged Across All Resampling Strategies)")
print("="*80)

kernel_avg = results_df.groupby('kernel').agg({
    'precision': 'mean',
    'recall': 'mean',
    'f1': 'mean',
    'balanced_accuracy': 'mean',
    'geometric_mean': 'mean',
    'mcc': 'mean',
    'roc_auc': 'mean',
    'pr_auc': 'mean',
    'false_alarm_rate': 'mean',
    'net_benefit': 'mean',
    'training_time': 'mean'
}).round(4)

print("\n" + kernel_avg.to_string())

# ===== 9. EXPORT DETAILED REPORT =====
print("\n" + "="*80)
print("EXPORTING DETAILED REPORTS")
print("="*80)

# Full report
results_df.to_csv(os.path.join(CHECKPOINT_DIR, 'full_analysis_report.csv'), index=False)
print("✓ Saved full_analysis_report.csv")

# Summary statistics
summary_stats = results_df.describe().T
summary_stats.to_csv(os.path.join(CHECKPOINT_DIR, 'summary_statistics.csv'))
print("✓ Saved summary_statistics.csv")

# Best models report
best_models = pd.DataFrame()
for criterion_name, criterion_col in criteria.items():
    best = results_df.loc[results_df[criterion_col].idxmax()]
    best['criterion'] = criterion_name
    best_models = pd.concat([best_models, best.to_frame().T])

best_models.to_csv(os.path.join(CHECKPOINT_DIR, 'best_models_by_criterion.csv'), index=False)
print("✓ Saved best_models_by_criterion.csv")

# ===== 10. RECOMMENDATIONS =====
print("\n" + "="*80)
print("RECOMMENDATIONS FOR IMBALANCED FRAUD DETECTION")
print("="*80)

best_f1 = results_df.loc[results_df['f1'].idxmax()]
best_balanced = results_df.loc[results_df['balanced_accuracy'].idxmax()]
best_business = results_df.loc[results_df['net_benefit'].idxmax()]

print(f"""
📊 ANALYSIS COMPLETE

Based on your fraud detection requirements:

1️⃣  BEST OVERALL MODEL (F1-Score):
   Configuration: {best_f1['resampling'].upper()} + {best_f1['kernel'].upper()}
   F1-Score: {best_f1['f1']:.4f}
   Precision: {best_f1['precision']:.4f} (When flagged, {best_f1['precision']*100:.1f}% are real fraud)
   Recall: {best_f1['recall']:.4f} (Catches {best_f1['recall']*100:.1f}% of all fraud)
   
2️⃣  MOST BALANCED MODEL (For Imbalanced Data):
   Configuration: {best_balanced['resampling'].upper()} + {best_balanced['kernel'].upper()}
   Balanced Accuracy: {best_balanced['balanced_accuracy']:.4f}
   MCC: {best_balanced['mcc']:.4f}
   Geometric Mean: {best_balanced['geometric_mean']:.4f}

3️⃣  BEST BUSINESS VALUE:
   Configuration: {best_business['resampling'].upper()} + {best_business['kernel'].upper()}
   Net Benefit: ${best_business['net_benefit']:,.2f}
   Fraud Prevented: ${best_business['fraud_prevented_value']:,.2f}
   False Alarm Cost: ${best_business['false_alarm_cost']:,.2f}

📈 KEY INSIGHTS:
   • Average Precision: {results_df['precision'].mean():.4f}
   • Average Recall: {results_df['recall'].mean():.4f}
   • Average False Alarm Rate: {results_df['false_alarm_rate'].mean():.4f}
   • Best Resampling: {resampling_avg['f1'].idxmax()}
   • Best Kernel: {kernel_avg['f1'].idxmax()}

⚠️  IMPORTANT FOR IMBALANCED DATA:
   - Use PR-AUC over ROC-AUC (better for rare positive class)
   - Focus on Balanced Accuracy, not regular Accuracy
   - Matthews Correlation Coefficient (MCC) is most reliable
   - Consider business costs when choosing threshold
""")

print("\n✅ Analysis complete! Check the 'checkpoints' folder for:")
print("   • comprehensive_analysis.png - Visual analysis")
print("   • full_analysis_report.csv - Complete metrics")
print("   • best_models_by_criterion.csv - Best performers")
print("   • summary_statistics.csv - Statistical summary")