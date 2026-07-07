# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.02253 | 0.03085 | 503.8 | 0.4692 | 0.5207 |
| ARIMA | 0.01988 | 0.02777 | 154.3 | 0.4858 | n/a |
| GARCH | 0.01961 | 0.02745 | 233.6 | 0.5762 | n/a |

## Key observations

- Lowest overall MAE: GARCH (0.01961).
- Highest directional accuracy: GARCH (0.5762).
- Caution: the proposed Hybrid model does not outperform ARIMA, GARCH on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- ARIMA's directional accuracy (0.4858) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.