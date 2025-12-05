# SVM Fraud Detection Training Pipeline

A comprehensive SVM-based fraud detection system with multiple resampling strategies and kernel types for handling imbalanced financial transaction data.

## Overview

This project trains and evaluates **16 different SVM model configurations** to detect fraudulent financial transactions using the PaySim synthetic dataset. It systematically explores the impact of:

- **4 Resampling Strategies** (handling class imbalance)
- **4 SVM Kernel Types** (capturing different decision boundaries)

**Total Experiments:** 4 resampling × 4 kernels = **16 model configurations**

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    STEP 1: main.py                          │
│                    (Training Pipeline)                       │
│                                                              │
│  1. Download PaySim Dataset                                 │
│  2. Feature Engineering (30+ features)                      │
│  3. Apply Resampling Strategies                             │
│  4. Train 16 SVM Models                                     │
│  5. Evaluate & Save Results                                 │
│                                                              │
│  Outputs: checkpoints/, trained_models/, plots/             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  STEP 2: visualize.py                       │
│               (Visualization Pipeline)                       │
│                                                              │
│  1. Load Checkpoint Data                                    │
│  2. Calculate Class Distributions                           │
│  3. Generate Pie Charts                                     │
│  4. Display Statistics                                      │
│                                                              │
│  Outputs: plots/class_distribution_pie_charts.png           │
└─────────────────────────────────────────────────────────────┘
```

> **Important:** You **MUST** run `main.py` before running `visualize.py`

---

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Install Dependencies

```bash
pip install kagglehub numpy pandas scikit-learn imbalanced-learn matplotlib seaborn joblib
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
kagglehub>=0.1.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

---

## Quick Start

### Run the Complete Pipeline

```bash
# Step 1: Train all models (required - first time may take 15-30 minutes)
python main.py

# Step 2: Generate visualization (optional - takes ~5 seconds)
python visualize.py
```

### Subsequent Runs

Thanks to the checkpoint system, subsequent runs of `main.py` will be **much faster** (~30 seconds) as it skips already-completed steps.

---

## Detailed Usage

### Step 1: Training Pipeline (`main.py`)

**Purpose:** Train and evaluate all SVM model configurations.

**What it does:**

1. **Data Acquisition**
   - Downloads PaySim synthetic financial transaction dataset from Kaggle
   - Samples 100,000 transactions (configurable)

2. **Feature Engineering**
   Creates 30+ features including:
   - **Transaction Type Features:** Encoded types, high-risk indicators
   - **Temporal Features:** Hour of day, day of month, night/weekend flags
   - **Merchant Features:** Merchant detection flags
   - **Balance Features:** Balance changes, ratios, mismatches
   - **Fraud Indicators:** Account draining, large transactions, anomalies
   - **Interaction Features:** Combined risk factors

3. **Resampling Strategies**
   Applies four approaches to handle class imbalance:
   
   | Strategy | Method | Effect |
   |----------|--------|--------|
   | `none` | Original data | ~99% non-fraud (baseline) |
   | `smote` | Synthetic oversampling | Balanced 50-50 split |
   | `undersample` | Random undersampling | Balanced but fewer samples |
   | `smotetomek` | SMOTE + Tomek links | Balanced with cleaner boundaries |

4. **Model Training**
   Trains SVM with four kernel types:
   - **Linear:** Fast, interpretable, works well for linearly separable data
   - **RBF:** Most popular, handles non-linear patterns well
   - **Polynomial:** Captures polynomial relationships (degree=3)
   - **Sigmoid:** Neural network-like decision boundaries

5. **Evaluation & Checkpointing**
   - Calculates comprehensive metrics
   - Saves models and results
   - Generates comparison visualizations

**Run Command:**
```bash
python main.py
```

**Outputs:**

```
checkpoints/
├── processed_data_none_enhanced.pkl
├── processed_data_smote_enhanced.pkl
├── processed_data_undersample_enhanced.pkl
├── processed_data_smotetomek_enhanced.pkl
├── trained_model_none_linear_enhanced.pkl
├── trained_model_none_rbf_enhanced.pkl
└── ... (16 model files + 16 prediction files)

trained_models/
├── none_linear/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── metadata.json
└── ... (16 model directories)

plots/
├── class_imbalance_comparison.png
└── training_results.png
```

---

### Step 2: Visualization Pipeline (`visualize.py`)

**Purpose:** Generate pie charts showing class distribution for each resampling strategy.

**What it does:**

1. Loads preprocessed data from `checkpoints/`
2. Calculates fraud vs. non-fraud percentages
3. Creates 2×2 grid of pie charts
4. Displays detailed statistics in console

**Prerequisites:**
- `main.py` must be run first
- Requires checkpoint files to exist

**Run Command:**
```bash
python visualize.py
```

**Output:**

```
plots/
└── class_distribution_pie_charts.png
```

---

## Configuration

### Modifying Sample Size

Edit `main.py` around line 280:

```python
# Default: 100,000 transactions
SAMPLE_SIZE = 100000

# For faster testing:
SAMPLE_SIZE = 10000

# For full dataset:
# Remove or comment out the sampling line:
# df = df.sample(n=SAMPLE_SIZE, random_state=42)
```

### Adding Custom Resampling Strategies

Edit `main.py` around line 22:

```python
RESAMPLING_STRATEGIES = ['none', 'smote', 'undersample', 'smotetomek']

# Add custom strategies:
# RESAMPLING_STRATEGIES = ['none', 'smote', 'undersample', 'smotetomek', 'adasyn']
```

### Adding Custom Kernels

Edit `main.py` around line 23:

```python
KERNEL_TYPES = ['linear', 'rbf', 'poly', 'sigmoid']

# Add custom kernels or remove unwanted ones:
# KERNEL_TYPES = ['linear', 'rbf']  # Train only 2 kernels
```

### Adjusting SVM Hyperparameters

Edit `main.py` around line 300:

```python
svm_model = SVC(
    kernel=KERNEL,
    C=1.0,              # Regularization (try: 0.1, 1.0, 10.0)
    gamma='scale',      # Kernel coefficient (try: 'scale', 'auto', 0.001)
    class_weight='balanced',
    probability=True,
    random_state=42
)
```

---

## Output Examples

### Console Output: `main.py`

```
================================================================================
SVM FRAUD DETECTION TRAINING
Experiments: 4 resampling × 4 kernels = 16 total
================================================================================

[NONE] Processing data...

============================================================
CLASS DISTRIBUTION - TRAINING SET (AFTER RESAMPLING)
Resampling Strategy: NONE
============================================================
Total Samples:       80,000
Non-Fraud (Class 0): 79,200 (99.00%)
Fraud (Class 1):     800 (1.00%)
Imbalance Ratio:     99.00:1 (non-fraud:fraud)
============================================================
  linear: trained (12.3s) F1=0.856
  rbf: trained (18.7s) F1=0.892
  poly: trained (22.1s) F1=0.873
  sigmoid: trained (15.4s) F1=0.841

[SMOTE] Loaded preprocessed data (158,400 train samples)

============================================================
CLASS DISTRIBUTION - TRAINING SET (AFTER RESAMPLING)
Resampling Strategy: SMOTE
============================================================
Total Samples:       158,400
Non-Fraud (Class 0): 79,200 (50.00%)
Fraud (Class 1):     79,200 (50.00%)
Imbalance Ratio:     1.00:1 (non-fraud:fraud)
============================================================
  linear: loaded F1=0.912
  rbf: loaded F1=0.945
  poly: loaded F1=0.928
  sigmoid: loaded F1=0.903

...

```

### Console Output: `visualize.py`

```
======================================================================
CLASS DISTRIBUTION PERCENTAGES
======================================================================

NONE
  Total Samples: 80,000
  Non-Fraud: 99.00% (79,200 samples)
  Fraud:     1.00% (800 samples)

SMOTE
  Total Samples: 158,400
  Non-Fraud: 50.00% (79,200 samples)
  Fraud:     50.00% (79,200 samples)

UNDERSAMPLE
  Total Samples: 1,600
  Non-Fraud: 50.00% (800 samples)
  Fraud:     50.00% (800 samples)

SMOTETOMEK
  Total Samples: 156,234
  Non-Fraud: 49.87% (77,893 samples)
  Fraud:     50.13% (78,341 samples)

✓ Pie charts saved to: plots/class_distribution_pie_charts.png
```

---

## Performance Metrics

Each model is evaluated using:

### Classification Metrics
- **Accuracy:** Overall correctness
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall

### Ranking Metrics
- **ROC-AUC:** Area under Receiver Operating Characteristic curve
- **PR-AUC:** Area under Precision-Recall curve (better for imbalanced data)

### Confusion Matrix Components
- **True Positives (TP):** Correctly identified frauds
- **False Positives (FP):** Non-frauds incorrectly flagged as fraud
- **True Negatives (TN):** Correctly identified non-frauds
- **False Negatives (FN):** Frauds missed by the model

### Metadata Saved for Each Model

```json
{
  "resampling_strategy": "smote",
  "kernel": "rbf",
  "training_date": "2024-12-04 10:30:45",
  "metrics": {
    "accuracy": 0.9876,
    "precision": 0.9234,
    "recall": 0.9456,
    "f1_score": 0.9344,
    "roc_auc": 0.9912,
    "pr_auc": 0.9823
  },
  "confusion_matrix": {
    "true_positives": 1520,
    "false_positives": 126,
    "true_negatives": 18234,
    "false_negatives": 120
  },
  "training_samples": 158400,
  "test_samples": 20000,
  "n_features": 32
}
```

---

---

## References

- **Dataset:** [PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **SMOTE Paper:** Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"
- **SVM:** Cortes & Vapnik (1995) - "Support-vector networks"

---

**Happy Training! 🚀**
