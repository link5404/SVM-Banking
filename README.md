# SVM-Banking

pip install kagglehub numpy pandas scikit-learn imbalanced-learn matplotlib seaborn joblib
```

---

## Output Examples

### Console Output (main.py)
```
SVM FRAUD DETECTION TRAINING
Experiments: 4 resampling × 4 kernels = 16 total
============================================================

[NONE] Processing data...
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
  ...
```

### Console Output (visualize.py)
```
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
...
