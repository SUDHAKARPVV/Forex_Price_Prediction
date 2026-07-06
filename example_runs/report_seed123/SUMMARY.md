# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| XGBoost_standalone | 0.02498 | 0.03657 | 808.1 | 0.5624 | n/a |
| Hybrid_CNN_LSTM_Transformer | 0.02236 | 0.03363 | 470.7 | 0.5728 | 0.5016 |
| Vanilla_LSTM | 0.02217 | 0.03265 | 377.6 | 0.5810 | 0.5000 |
| Simplified_TFT | 0.02367 | 0.03359 | 918.8 | 0.5179 | 0.5143 |
| ARIMA | 0.01950 | 0.02963 | 134.3 | 0.5700 | n/a |
| Random_Walk_Drift | 0.02215 | 0.03268 | 118.1 | 0.4780 | n/a |

## Key observations

- Lowest overall MAE: ARIMA (0.01950).
- Highest directional accuracy: Vanilla_LSTM (0.5810).
- Caution: the proposed Hybrid model does not outperform Vanilla_LSTM on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Simplified_TFT's directional accuracy (0.5179) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Random_Walk_Drift's directional accuracy (0.4780) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.