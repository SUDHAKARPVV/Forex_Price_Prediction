# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| XGBoost_standalone | 0.03670 | 0.04867 | 206.2 | 0.6052 | n/a |
| Hybrid_CNN_LSTM_Transformer | 0.03953 | 0.05221 | 235.3 | 0.5184 | 0.4945 |
| Vanilla_LSTM | 0.03837 | 0.05054 | 217.9 | 0.5456 | 0.4984 |
| Simplified_TFT | 0.04417 | 0.05837 | 300.2 | 0.4956 | 0.5033 |
| ARIMA | 0.03768 | 0.04790 | 164.9 | 0.5850 | n/a |
| Random_Walk_Drift | 0.03873 | 0.05138 | 108.5 | 0.5102 | n/a |

## Key observations

- Lowest overall MAE: XGBoost_standalone (0.03670).
- Highest directional accuracy: XGBoost_standalone (0.6052).
- Caution: the proposed Hybrid model does not outperform XGBoost_standalone, Vanilla_LSTM, ARIMA on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Hybrid_CNN_LSTM_Transformer's directional accuracy (0.5184) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Simplified_TFT's directional accuracy (0.4956) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Random_Walk_Drift's directional accuracy (0.5102) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.