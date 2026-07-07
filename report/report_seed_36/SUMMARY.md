# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.02047 | 0.02824 | 649.8 | 0.5316 | 0.5080 |
| Vanilla_LSTM | 0.02141 | 0.02982 | 852.5 | 0.5070 | 0.4698 |
| Simplified_TFT | 0.02086 | 0.02884 | 1214.3 | 0.5490 | 0.5000 |
| ARIMA | 0.01703 | 0.02470 | 101.9 | 0.5175 | n/a |
| Random_Walk_Drift | 0.01960 | 0.02742 | 262.3 | 0.5770 | n/a |

## Key observations

- Lowest overall MAE: ARIMA (0.01703).
- Highest directional accuracy: Random_Walk_Drift (0.5770).
- Caution: the proposed Hybrid model does not outperform Simplified_TFT, Random_Walk_Drift on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Vanilla_LSTM's directional accuracy (0.5070) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- ARIMA's directional accuracy (0.5175) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.