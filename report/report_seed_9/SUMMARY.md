# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.02520 | 0.03778 | 433.8 | 0.5480 | 0.4982 |
| ARIMA | 0.02297 | 0.03513 | 123.3 | 0.4961 | n/a |
| GARCH | 0.02291 | 0.03509 | 127.8 | 0.5220 | n/a |

## Key observations

- Lowest overall MAE: GARCH (0.02291).
- Highest directional accuracy: Hybrid_CNN_LSTM_Transformer (0.5480).
- Caution: the proposed Hybrid model does not outperform the simpler baselines on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- ARIMA's directional accuracy (0.4961) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- GARCH's directional accuracy (0.5220) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.