"""
Script to display class distribution percentages as pie charts
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR = "plots"
Path(PLOTS_DIR).mkdir(exist_ok=True)

def get_class_distribution_percentage(y_data):
    """Calculate class distribution percentages"""
    total = len(y_data)
    fraud_count = np.sum(y_data == 1)
    non_fraud_count = np.sum(y_data == 0)
    fraud_pct = (fraud_count / total) * 100
    non_fraud_pct = (non_fraud_count / total) * 100
    
    return {
        'total': total,
        'non_fraud_count': non_fraud_count,
        'fraud_count': fraud_count,
        'non_fraud_pct': non_fraud_pct,
        'fraud_pct': fraud_pct
    }

def plot_pie_charts():
    """Create pie charts showing class distribution for all resampling strategies"""
    
    resampling_strategies = ['none', 'smote', 'undersample', 'smotetomek']
    distribution_data = []
    
    print("="*70)
    print("CLASS DISTRIBUTION PERCENTAGES")
    print("="*70)
    
    for strategy in resampling_strategies:
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f'processed_data_{strategy}_enhanced.pkl')
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                y_train = checkpoint['y_train']
                
            dist = get_class_distribution_percentage(y_train)
            distribution_data.append({
                'strategy': strategy,
                'data': dist
            })
            
            print(f"\n{strategy.upper()}")
            print(f"  Total Samples: {dist['total']:,}")
            print(f"  Non-Fraud: {dist['non_fraud_pct']:.2f}% ({dist['non_fraud_count']:,} samples)")
            print(f"  Fraud:     {dist['fraud_pct']:.2f}% ({dist['fraud_count']:,} samples)")
        else:
            print(f"\n⚠ Checkpoint not found for strategy: {strategy}")
    
    if not distribution_data:
        print("\n⚠ No checkpoint data found. Please run the main training script first.")
        return
    
    # Create pie charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Class Distribution Comparison Across Resampling Strategies', 
                 fontsize=32, fontweight='bold', y=0.98)
    
    colors = ['#2ecc71', '#e74c3c']  # Green for non-fraud, Red for fraud
    labels = ['Non-Fraud', 'Fraud']
    
    axes = axes.flatten()
    
    for idx, item in enumerate(distribution_data):
        ax = axes[idx]
        dist = item['data']
        strategy = item['strategy']
        
        sizes = [dist['non_fraud_pct'], dist['fraud_pct']]
        counts = [dist['non_fraud_count'], dist['fraud_count']]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels,
            colors=colors,
            autopct='%1.2f%%',
            startangle=90,
            explode=(0.05, 0.05),
            shadow=True,
            textprops={'fontsize': 16, 'weight': 'bold'}
        )
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(16)
            autotext.set_weight('bold')
        
        # Add title with sample counts
        ax.set_title(
            f'{strategy.upper()}\n'
            f'Total: {dist["total"]:,} samples\n'
            f'Non-Fraud: {counts[0]:,} | Fraud: {counts[1]:,}',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'class_distribution_pie_charts.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Pie charts saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_pie_charts()