# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| XGBoost_standalone | 0.02845 | 0.04095 | 222.1 | 0.6497 | n/a |
| Hybrid_CNN_LSTM_Transformer | 0.02879 | 0.04184 | 203.8 | 0.6879 | 0.5000 |
| Vanilla_LSTM | 0.03058 | 0.04111 | 335.0 | 0.6063 | 0.5179 |
| Simplified_TFT | 0.03113 | 0.04187 | 332.2 | 0.5552 | 0.5121 |
| ARIMA | 0.03082 | 0.04620 | 104.7 | 0.6100 | n/a |
| Random_Walk_Drift | 0.03103 | 0.04382 | 171.7 | 0.5346 | n/a |

## Key observations

- Lowest overall MAE: XGBoost_standalone (0.02845).
- Highest directional accuracy: Hybrid_CNN_LSTM_Transformer (0.6879).
- Caution: the proposed Hybrid model does not outperform the simpler baselines on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.