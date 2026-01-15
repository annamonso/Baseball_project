# ⚾ Baseball Project — Contact & Batted-Ball Outcome Modeling

This project builds a **data and machine learning pipeline** using Statcast data to model **batted-ball contact** and **ball-in-play outcomes**.  
The end goal is to predict:
1. Whether a hitter will **make contact** on a given pitch.  
2. If contact occurs, the **type and location** of the resulting ball in play.

---

## 🚀 Project Overview

The project transforms raw **Statcast** data (from [pybaseball](https://github.com/jldbc/pybaseball)) into model-ready datasets through several modular stages:

1. **Data Ingestion** — Collect and standardize Statcast data with post-contact features (launch_speed, launch_angle, hit_distance_sc).
2. **Feature Engineering**
   - Pre-contact features for contact prediction (no data leakage)
   - Pre + post-contact features for BIP outcome prediction
3. **Label Engineering**
   - **Contact labels:** Binary outcome (contact vs no-contact).
   - **Batted-ball labels:** Type of result (OUT, 1B, 2B, 3B, HR, ROE) and discretized spray location.
4. **Model Training** — ML models to predict contact and ball-in-play results.
   - Contact model: Logistic Regression (pre-contact only)
   - BIP outcome model: Random Forest multi-class classifier (pre + post-contact)
5. **Evaluation & Visualization** — ROC curves, confusion matrices, feature importance, model metrics.

---

## 📁 Repository Structure

```
Baseball_project/
├── data_raw/                 # Unprocessed Statcast data
├── data_proc/                # Processed Parquet + metadata files
│   ├── features.parquet              # Pre-contact features (all pitches)
│   ├── features_bip.parquet          # Pre + post-contact features (BIP only)
│   ├── contact_labels.parquet
│   ├── labels.parquet
│   ├── training_contact.parquet
│   ├── training_bip.parquet          # Baseline (pre-contact only)
│   ├── training_bip_enhanced.parquet # Enhanced (with post-contact)
│   └── SxR_bins.json
│
├── models/                   # Trained models
│   ├── contact/
│   │   ├── baseline.pkl              # Logistic Regression
│   │   ├── metrics.json
│   │   └── feature_info.json
│   ├── bip/                          # Baseline (25% accuracy)
│   │   ├── random_forest_outcome.pkl
│   │   ├── metrics_outcome.json
│   │   └── feature_info_outcome.json
│   └── bip_enhanced/                 # Enhanced (50% accuracy)
│       ├── random_forest_outcome.pkl
│       ├── metrics_outcome.json
│       └── feature_info_outcome.json
│
├── reports/                  # Evaluation reports
│   ├── contact/
│   │   ├── eval_metrics.json
│   │   └── evaluation_plots.png
│   ├── bip/                          # Baseline model reports
│   │   ├── eval_metrics.json
│   │   ├── evaluation_plots.png
│   │   ├── confusion_matrix_normalized.png
│   │   └── feature_importance.csv
│   └── bip_enhanced/                 # Enhanced model reports
│       ├── eval_metrics.json
│       ├── evaluation_plots.png
│       ├── confusion_matrix_normalized.png
│       └── feature_importance.csv
│
├── src/
│   ├── data/
│   │   ├── pull_statcast.py           # Statcast data ingestion
│   │   └── columns.py                 # Column definitions (includes post-contact)
│   ├── features/
│   │   ├── make_features.py           # Pre-contact feature engineering
│   │   ├── make_features_bip.py       # BIP features (pre + post-contact)
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

## 🧠 Project Goals

This project implements a **two-stage predictive pipeline**:

1. **Stage 1: Contact Prediction**
   - Binary classifier predicting whether a pitch results in contact
   - Uses pre-contact features: pitch characteristics, count, game situation
   - Model: Logistic Regression (ROC-AUC: 0.63)

2. **Stage 2: Ball-in-Play Outcome Prediction**
   - Multi-class classifier predicting outcome type (OUT, 1B, 2B, 3B, HR, ROE)
   - Uses pre-contact + post-contact features (launch_speed, launch_angle, hit_distance_sc)
   - Can also predict spray location (sector/ring bins)
   - **Enhanced Model: Random Forest (Accuracy: 50.5%)**
   - Baseline (pre-contact only): 25.4% accuracy

**Use Cases:**
- Batted ball outcome prediction based on exit velocity and launch angle
- Hit probability analysis by pitch type and location
- Hitter tendency modeling and spray patterns
- Defensive positioning optimization
- Expected outcome analysis for player evaluation

---

## ⚙️ Command Line Usage (Happy Path)

### Option 1: Enhanced Pipeline (Recommended - 50% Accuracy)

Train BIP model with post-contact features for best performance:

```bash
# 1. Generate BIP features (pre + post-contact)
python -m src.features.make_features_bip \
  --input data_raw/statcast_full.parquet \
  --output data_proc/features_bip.parquet

# 2. Generate BIP labels
python -m src.features.make_labels_bip \
  --input data_raw/statcast_full.parquet \
  --output data_proc/labels.parquet \
  --bins data_proc/SxR_bins.json \
  --S 10 --R 5

# 3. Train enhanced BIP model
python -m src.models.train_model_bip \
  --input data_proc/training_bip_enhanced.parquet \
  --output_dir models/bip_enhanced \
  --target outcome \
  --model_type random_forest

# 4. Evaluate enhanced model
python -m src.models.evaluate \
  --model models/bip_enhanced/random_forest_outcome.pkl \
  --data data_proc/training_bip_enhanced.parquet \
  --target outcome \
  --task multiclass \
  --output_dir reports/bip_enhanced \
  --feature_importance \
  --top_n 15
```

### Option 2: Full Two-Stage Pipeline (Contact + BIP)

Complete end-to-end pipeline including contact prediction:

```bash
# 1. Generate pre-contact features
python -m src.features.make_features \
  --input data_raw/statcast_full.parquet \
  --output data_proc/features.parquet

# 2. Generate contact labels
python -m src.features.make_labels_contact \
  --input data_raw/statcast_full.parquet \
  --output data_proc/contact_labels.parquet

# 3. Create contact training set
python -m src.features.make_training_sets \
  --features data_proc/features.parquet \
  --contact_labels data_proc/contact_labels.parquet \
  --output_dir data_proc

# 4. Train contact model
python -m src.models.train_model_contact \
  --input data_proc/training_contact.parquet \
  --output_dir models/contact

# 5. Evaluate contact model
python -m src.models.evaluate \
  --model models/contact/baseline.pkl \
  --data data_proc/training_contact.parquet \
  --target is_contact \
  --task binary \
  --output_dir reports/contact

# 6-9. Then run Option 1 for BIP model
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

## 🔜 Future Enhancements

- [ ] Address class imbalance with SMOTE for rare outcomes (3B, ROE)
- [ ] Train spray location prediction models (sector_bin, ring_bin)
- [ ] Implement pitch sequencing features (previous 3 pitches)
- [ ] Add hyperparameter tuning with GridSearchCV
- [ ] Create model ensembles (Random Forest + XGBoost)
- [ ] Add player-specific historical statistics
- [ ] Build interactive dashboard for predictions
- [ ] Add comprehensive documentation

---

## 📜 License

MIT License © 2025 [Anna Monso]
