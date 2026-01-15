# ⚾ Baseball Project — Contact & Batted-Ball Outcome Modeling

This project builds a **data and machine learning pipeline** using Statcast data to model **batted-ball contact** and **ball-in-play outcomes**.  
The end goal is to predict:
1. Whether a hitter will **make contact** on a given pitch.  
2. If contact occurs, the **type and location** of the resulting ball in play.

---

## 🚀 Project Overview

The project transforms raw **Statcast** data (from [pybaseball](https://github.com/jldbc/pybaseball)) into model-ready datasets through several modular stages:

1. **Data Ingestion** — Collect and standardize Statcast data.
2. **Feature Engineering** — Extract numeric and categorical predictors (pre-contact only to avoid data leakage).
3. **Label Engineering**
   - **Contact labels:** Binary outcome (contact vs no-contact).
   - **Batted-ball labels:** Type of result (OUT, 1B, 2B, 3B, HR, ROE) and discretized spray location.
4. **Model Training** — ML models to predict contact and ball-in-play results.
   - Contact model: Logistic Regression
   - BIP outcome model: Random Forest multi-class classifier
5. **Evaluation & Visualization** — ROC curves, confusion matrices, feature importance, model metrics.

---

## 📁 Repository Structure

```
Baseball_project/
├── data_raw/                 # Unprocessed Statcast data
├── data_proc/                # Processed Parquet + metadata files
│   ├── features.parquet
│   ├── contact_labels.parquet
│   ├── labels.parquet
│   ├── training_contact.parquet
│   ├── training_bip.parquet
│   └── SxR_bins.json
│
├── models/                   # Trained models
│   ├── contact/
│   │   ├── baseline.pkl
│   │   ├── metrics.json
│   │   └── feature_info.json
│   └── bip/
│       ├── random_forest_outcome.pkl
│       ├── metrics_outcome.json
│       └── feature_info_outcome.json
│
├── reports/                  # Evaluation reports
│   ├── contact/
│   │   ├── eval_metrics.json
│   │   └── evaluation_plots.png
│   └── bip/
│       ├── eval_metrics.json
│       ├── evaluation_plots.png
│       ├── confusion_matrix_normalized.png
│       └── feature_importance.csv
│
├── src/
│   ├── data/
│   │   └── pull_statcast.py           # Statcast data ingestion
│   ├── features/
│   │   ├── make_features.py           # Feature engineering
│   │   ├── make_labels_contact.py     # Contact label generation
│   │   ├── make_labels_bip.py         # Ball-in-play outcome labeling
│   │   └── make_training_sets.py      # Feature-label fusion
│   ├── models/
│   │   ├── train_model_contact.py     # Contact model training
│   │   ├── train_model_bip.py         # BIP outcome model training
│   │   └── evaluate.py                # Model evaluation script
│   └── visualization/
│       └── plots.py                   # Utility plotting functions
│
├── notebooks/              # EDA and analysis notebooks
├── requirements.txt
└── README.md
```

---

## 🧩 Implemented Components

| Module | Status | Description |
|--------|---------|-------------|
| `src/data/make_dataset.py` | ✅ | Fetches Statcast data and saves raw CSV/Parquet files. |
| `src/features/make_features.py` | ✅ | Builds model features (pitch, batter, context variables). |
| `src/features/make_labels_contact.py` | ✅ | Generates `is_contact` binary label. |
| `src/features/make_labels_bip.py` | ✅ | Creates BIP outcome and spray location labels (`SxR_bins`). |
| `src/features/make_training_sets.py` | ✅ | Merges features + labels into training-ready datasets. |
| `src/models/train_model_contact.py` | ✅ | Trains baseline logistic regression for contact prediction. |
| `src/models/train_model_bip.py` | ✅ | Trains multi-class classifier for BIP outcome prediction. |
| `src/models/evaluate.py` | ✅ | Comprehensive model evaluation with metrics and visualizations. |
| `notebooks/` | ✅ | Contains exploratory plots, location heatmaps, and sanity checks. |

---

## 🔜 Future Enhancements

| Planned Enhancement | Goal |
|---------------------|------|
| Improved BIP models | Add launch angle/exit velocity features post-contact for better accuracy. |
| Spray location models | Train models to predict spray bins (sector_bin, ring_bin). |
| Pitch sequencing | Incorporate previous pitch context for better predictions. |
| Interactive dashboards | Web app for exploring predictions and visualizations. |
| `docs/` | Add project documentation and architecture diagram. |

---

## 📊 Example Outputs & Model Performance

**Contact label summary:**
```
is_contact
0    0.638
1    0.362
Name: frac, dtype: float
```

**Ball-in-play outcome distribution:**
```
OUT    0.664
1B     0.211
2B     0.067
HR     0.045
ROE    0.009
3B     0.005
```

**Contact Model Performance (Logistic Regression):**
```
Test Set:
  Accuracy:  0.60
  ROC-AUC:   0.63
  Precision: 0.46
  Recall:    0.58
  F1-Score:  0.51
```

**BIP Outcome Model Performance (Random Forest):**
```
Test Set:
  Accuracy:          0.25
  Precision (wtd):   0.55
  Recall (wtd):      0.25
  F1-Score (wtd):    0.30

Top Feature Importances:
  1. release_pos_z      (14.8%)
  2. release_pos_x      (14.6%)
  3. release_speed      (11.9%)
  4. pfx_x              (11.6%)
  5. pfx_z              (11.6%)
```

**Note:** BIP model performance is limited by using only pre-contact features. Adding post-contact features (launch angle, exit velocity) would significantly improve accuracy.

**Spray bin metadata (`S=10`, `R=5`):**
Stored in `data_proc/SxR_bins.json`.

---

## 🧠 Project Goals

This project implements a **two-stage predictive pipeline**:

1. **Stage 1: Contact Prediction**
   - Binary classifier predicting whether a pitch results in contact
   - Uses pre-contact features: pitch characteristics, count, game situation
   - Baseline: Logistic Regression (ROC-AUC: 0.63)

2. **Stage 2: Ball-in-Play Outcome Prediction**
   - Multi-class classifier predicting outcome type (OUT, 1B, 2B, 3B, HR, ROE)
   - Can also predict spray location (sector/ring bins)
   - Baseline: Random Forest (Accuracy: 0.25 with pre-contact features only)

**Use Cases:**
- Pitch probability analysis by type and location
- Hitter tendency modeling and spray patterns
- Defensive positioning optimization
- Pitch sequencing strategy

---

## ⚙️ Command Line Usage (Happy Path)

Run the full pipeline end-to-end:

```bash
# 1. Generate features (pre-contact only, no leakage)
python -m src.features.make_features \
  --input data_raw/statcast_full.parquet \
  --output data_proc/features.parquet

# 2. Generate contact labels (binary: contact vs no-contact)
python -m src.features.make_labels_contact \
  --input data_raw/statcast_full.parquet \
  --output data_proc/contact_labels.parquet

# 3. Generate BIP labels (outcome + spray grid)
python -m src.features.make_labels_bip \
  --input data_raw/statcast_full.parquet \
  --output data_proc/labels.parquet \
  --bins data_proc/SxR_bins.json \
  --S 10 --R 5

# 4. Create training datasets (merge features + labels)
python -m src.features.make_training_sets \
  --features data_proc/features.parquet \
  --contact_labels data_proc/contact_labels.parquet \
  --bip_labels data_proc/labels.parquet \
  --output_dir data_proc

# 5. Train baseline contact model
python -m src.models.train_model_contact \
  --input data_proc/training_contact.parquet \
  --output_dir models/contact

# 6. Train BIP outcome model
python -m src.models.train_model_bip \
  --input data_proc/training_bip.parquet \
  --output_dir models/bip \
  --target outcome \
  --model_type random_forest

# 7. Evaluate contact model
python -m src.models.evaluate \
  --model models/contact/baseline.pkl \
  --data data_proc/training_contact.parquet \
  --target is_contact \
  --task binary \
  --output_dir reports/contact

# 8. Evaluate BIP model
python -m src.models.evaluate \
  --model models/bip/random_forest_outcome.pkl \
  --data data_proc/training_bip.parquet \
  --target outcome \
  --task multiclass \
  --output_dir reports/bip \
  --feature_importance \
  --top_n 15
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main libraries used:
- `pybaseball`
- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn` (for future models)
- `pyarrow` (for Parquet I/O)

---

## 🧩 Next Milestone

- [ ] Merge datasets → `make_training_sets.py`
- [ ] Train first contact classifier (baseline logistic or random forest)
- [ ] Train BIP outcome model
- [ ] Add visual analytics (spray maps, feature importances)

---

## 📜 License

MIT License © 2025 [Anna Monso]
