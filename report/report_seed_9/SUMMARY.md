# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.02202 | 0.03049 | 480.1 | 0.5023 | 0.4728 |
| ARIMA | 0.01703 | 0.02470 | 101.9 | 0.5175 | n/a |
| GARCH | 0.01679 | 0.02451 | 120.6 | 0.5975 | n/a |

## Key observations

- Lowest overall MAE: GARCH (0.01679).
- Highest directional accuracy: GARCH (0.5975).
- Caution: the proposed Hybrid model does not outperform ARIMA, GARCH on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Hybrid_CNN_LSTM_Transformer's directional accuracy (0.5023) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- ARIMA's directional accuracy (0.5175) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.