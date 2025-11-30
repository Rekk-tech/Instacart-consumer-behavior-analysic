# Sequential Models Training Report

**Date:** 2025-11-29 02:18:50  
**Notebook:** 04_seq_models_tcn_lstm.ipynb

##  Experiment Configuration

- **Training Samples:** 27,518
- **Test Samples:** 6,880
- **Sequence Length:** 20
- **Embedding Dimension:** 128
- **Vocabulary Size:** 10,001
- **Static Features:** 71

##  Model Performance

| Model | AUC | F1 | P@10 | R@10 | Improvement over Random |
|-------|-----|----|----- |------|------------------------|
| LSTM  | 0.7887 | 0.2906 | 0.1000 | 0.0018 | 57.7% |
| TCN   | 0.8007 | 0.2973 | 0.2000 | 0.0036 | 60.1% |

** Best Model:** TCN

##  Saved Artifacts

- **LSTM Model:** `lstm_final_20251129_021850.h5`
- **TCN Model:** `tcn_final_20251129_021850.h5`
- **Results Summary:** `seq_models_summary_20251129_021850.json`

##  Next Steps

1. Compare with baseline XGBoost/LightGBM models from notebook 03
2. Ensemble modeling: combine sequence models with tree-based models
3. Hyperparameter tuning: sequence length, embedding dimension, architecture depth
4. Advanced techniques: attention mechanisms, transformer models
