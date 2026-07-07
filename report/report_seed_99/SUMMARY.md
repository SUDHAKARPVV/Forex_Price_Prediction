# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.02093 | 0.02896 | 629.8 | 0.5219 | 0.4373 |
| Vanilla_LSTM | 0.01974 | 0.02758 | 480.9 | 0.5353 | 0.4615 |
| Simplified_TFT | 0.01985 | 0.02769 | 424.3 | 0.5292 | 0.5137 |
| ARIMA | 0.01703 | 0.02470 | 101.9 | 0.5175 | n/a |
| Random_Walk_Drift | 0.01960 | 0.02742 | 262.3 | 0.5770 | n/a |

## Key observations

- Lowest overall MAE: ARIMA (0.01703).
- Highest directional accuracy: Random_Walk_Drift (0.5770).
- Caution: the proposed Hybrid model does not outperform Vanilla_LSTM, Simplified_TFT, Random_Walk_Drift on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Hybrid_CNN_LSTM_Transformer's directional accuracy (0.5219) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Simplified_TFT's directional accuracy (0.5292) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- ARIMA's directional accuracy (0.5175) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.