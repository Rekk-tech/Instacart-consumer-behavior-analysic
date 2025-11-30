# Baseline Models Results - 20251128_195211

## 📊 Experiment Summary
- **Notebook**: 03_baseline_xgb.ipynb
- **Sample Size**: 10,000 users
- **Training Samples**: 27,518
- **Test Samples**: 6,880  
- **Features**: 71 features
- **Timestamp**: 20251128_195211

## 🤖 Model Performance

| Model | AUC | Precision | Recall | F1 | P@5 | P@10 |
|-------|-----|-----------|--------|----|----|------|
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## ⚠️ Critical Issues
- **Perfect AUC = 1.0000 detected**
- Likely data leakage or severe overfitting
- Requires immediate investigation before production use

## 🔝 Top 5 Important Features
1. **user_total_orders** - XGB: 0.0134, LGB: 936.0000
2. **ui_times_bought** - XGB: 0.0023, LGB: 452.0000
3. **ui_last_order_number** - XGB: 0.1861, LGB: 392.0000
4. **ui_orders_since_last_purchase** - XGB: 0.4678, LGB: 341.0000
5. **frequency** - XGB: 0.0014, LGB: 290.0000

## 📁 Artifacts Locations
- Models: `../models/archived/`
- Splits: `../data/splits/`
- Features: `../data/features/`
- Results: `../models/archived/baseline_results_20251128_195211.json`

## 🎯 Next Steps
1. Investigate data leakage in feature engineering
2. Implement proper temporal validation
3. Proceed to deep learning models (04_seq_models_tcn_lstm.ipynb)
