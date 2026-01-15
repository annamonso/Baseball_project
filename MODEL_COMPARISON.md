# BIP Outcome Model Comparison

## Performance Improvement Summary

| Metric | Baseline (Pre-contact only) | Enhanced (With Post-contact) | Improvement |
|--------|----------------------------|------------------------------|-------------|
| **Test Accuracy** | 25.4% | **50.5%** | **+99% (2x)** |
| **Weighted F1** | 0.301 | **0.546** | **+81%** |
| **Log Loss** | 1.626 | **1.436** | **-12% (better)** |

---

## Per-Class Performance (Test Set)

### Enhanced Model (With Post-Contact Features):

| Outcome | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **OUT** | 0.800 | 0.526 | 0.634 | 4,643 |
| **1B** | 0.377 | 0.512 | 0.434 | 1,485 |
| **HR** | 0.296 | 0.594 | 0.395 | 313 |
| **2B** | 0.186 | 0.305 | 0.231 | 472 |
| **3B** | 0.000 | 0.000 | 0.000 | 34 |
| **ROE** | 0.021 | 0.141 | 0.036 | 64 |

**Key Improvements:**
- **Home runs (HR)**: 59.4% recall (up from 39.1%)
- **Singles (1B)**: 51.2% recall (up from 29.9%)
- **Overall accuracy**: 50.5% (up from 25.4%)

---

## Feature Importance Comparison

### Baseline Model (Pre-Contact Only):
1. release_pos_z (14.8%)
2. release_pos_x (14.6%)
3. release_speed (11.9%)
4. pfx_x (11.6%)
5. pfx_z (11.6%)

### Enhanced Model (With Post-Contact):
1. **hit_distance_sc (22.6%)** 
2. **launch_angle (14.5%)** 
3. **launch_speed (12.6%)** 
4. release_pos_x (6.6%)
5. release_speed (6.4%)

**Key Finding:** The top 3 post-contact features account for ~50% of the model's predictive power!

---

## Confusion Matrix Analysis

### Baseline vs Enhanced - OUT Prediction:
- **Baseline**: 21.7% of outs correctly classified
- **Enhanced**: 52.6% of outs correctly classified (+143%)

### Home Run Prediction:
- **Baseline**: 39.1% recall
- **Enhanced**: 59.4% recall (+52%)

---

## Data Coverage

**Post-Contact Feature Availability (Balls in Play):**
- launch_speed: 99.6% coverage
- launch_angle: 99.7% coverage  
- hit_distance_sc: 99.3% coverage

**Training Set Size:**
- Baseline: 149,603 rows (includes duplicate joins)
- Enhanced: 35,055 rows (BIP only, cleaner dataset)

---

## Conclusion

Adding post-contact features (launch_speed, launch_angle, hit_distance_sc) **doubled** the BIP outcome prediction accuracy from 25% to 50%. The hit distance alone accounts for nearly 23% of the model's decision-making, making it the single most important feature.

This validates that:
1. **Pre-contact features alone** cannot accurately predict batted ball outcomes
2. **Exit velocity, launch angle, and distance** are critical for outcome classification
3. The model is now **practically useful** for analyzing BIP outcomes

---

## Next Steps for Further Improvement

1. **Address class imbalance**: Use SMOTE or class weighting for rare outcomes (3B, ROE)
2. **Feature engineering**: 
   - Spray angle bins as features
   - Batter/pitcher historical stats
   - Park factors
3. **Model ensembles**: Combine Random Forest with XGBoost
4. **Hyperparameter tuning**: GridSearchCV for optimal RF parameters
5. **Separate models**: Train specialized models for specific outcomes (HR vs non-HR)

---

Generated: 2026-01-15
Baseline Model: random_forest_outcome.pkl (pre-contact)
Enhanced Model: random_forest_outcome.pkl (pre + post-contact)
