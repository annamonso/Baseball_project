# ⚾ Baseball Project — Contact & Batted-Ball Outcome Modeling

This project builds a **data and machine learning pipeline** using Statcast data to model **batted-ball contact** and **ball-in-play outcomes**.  
The end goal is to predict:
1. Whether a hitter will **make contact** on a given pitch.  
2. If contact occurs, the **type and location** of the resulting ball in play.

---

## 🚀 Project Overview

The project transforms raw **Statcast** data (from [pybaseball](https://github.com/jldbc/pybaseball)) into model-ready datasets through several modular stages:

1. **Data Ingestion** — Collect and standardize Statcast data.
2. **Feature Engineering** — Extract numeric and categorical predictors.
3. **Label Engineering**
   - **Contact labels:** Binary outcome (contact vs no-contact).
   - **Batted-ball labels:** Type of result (OUT, 1B, 2B, 3B, HR, ROE) and discretized spray location.
4. **Model Training** — (upcoming) ML models to predict contact and ball-in-play results.
5. **Evaluation & Visualization** — (upcoming) Spray charts, ROC curves, model metrics.

---

## 📁 Repository Structure

```
Baseball_project/
├── data_raw/                 # Unprocessed Statcast data
├── data_proc/                # Processed Parquet + metadata files
│   ├── features.parquet
│   ├── contact_labels.parquet
│   ├── labels.parquet
│   └── SxR_bins.json
│
├── src/
│   ├── data/
│   │   └── make_dataset.py            # Statcast data ingestion
│   ├── features/
│   │   ├── make_features.py           # Feature engineering
│   │   ├── make_labels_contact.py     # Contact label generation
│   │   ├── make_labels_bip.py         # Ball-in-play outcome labeling
│   │   └── make_training_sets.py      # (planned) Feature-label fusion
│   ├── models/
│   │   └── train_model.py             # (planned) ML training pipeline
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
| `notebooks/` | ✅ | Contains exploratory plots, location heatmaps, and sanity checks. |

---

## 🔜 Upcoming Components

| Planned Module | Goal |
|----------------|------|
| `src/features/make_training_sets.py` | Merge features + labels into training-ready datasets. |
| `src/models/train_model.py` | Train ML classifiers for contact and BIP prediction. |
| `src/models/evaluate.py` | Model evaluation and visualization (confusion matrix, ROC, spray maps). |
| `docs/` | Add project documentation and architecture diagram. |

---

## 📊 Example Outputs

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

**Spray bin metadata (`S=10`, `R=5`):**
Stored in `data_proc/SxR_bins.json`.

---

## 🧠 Project Goals

- Build a **two-stage predictive pipeline**:
  1. **Stage 1:** Contact probability model.
  2. **Stage 2:** Batted-ball outcome & spray prediction.

- Use these models to explore:
  - Hit probability by pitch type and location.
  - Hitter-specific spray tendencies.
  - Defensive alignment optimization.

---

## ⚙️ Example Command Line Usage

```bash
# Generate contact labels
python -m src.features.make_labels_contact   --input data_raw   --output data_proc/contact_labels.parquet

# Generate BIP labels (10x5 spray grid)
python -m src.features.make_labels_bip   --input data_raw   --output data_proc/labels.parquet   --bins data_proc/SxR_bins.json   --S 10 --R 5
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
