"""
XGBoost baseline, directly motivated by Dave et al. 2025 ("Predicting Forex
Prices: An Evaluation of LSTM, XGBoost and Transformer Architectures"),
which found XGBoost (and an LSTM+XGBoost ensemble) dramatically
outperforming both LSTM and Transformer models on RMSE for daily FX price
prediction using technical + fundamental indicators -- e.g. their
XGB_TM model reached an RMSE of 0.0017 on AUD/NZD versus 0.0037 for the
equivalent LSTM.

Gradient-boosted trees are a natural fit here: our technical + macro +
sentiment feature set is exactly the kind of engineered, tabular input
XGBoost excels at, and unlike the sequence models it isn't trying to learn
temporal structure from scratch -- it only needs a fixed-size feature
summary per forecast origin, which sidesteps a lot of the small-dataset
overfitting risk the deep models face.

Since XGBoost has no native notion of a sequence, each window is
summarised into a fixed-size feature vector (mean, std, and last value of
each of the 22 fused features over the lookback window, plus the 2 regime
features) rather than fed the raw (T, 22) tensor.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def summarize_window(x: np.ndarray) -> np.ndarray:
    """x: (T, F) -> (3*F,) fixed-size summary: [mean, std, last] per feature.
    This is the tabular-feature-engineering equivalent of what the LSTM/
    Transformer models extract sequentially -- a level, a variability
    measure, and the most recent value for every one of the 22 fused
    technical/macro/sentiment features.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    last = x[-1]
    return np.concatenate([mean, std, last])


def build_xgb_feature_matrix(dataset) -> "tuple[np.ndarray, np.ndarray]":
    """Iterate an FXWindowDataset (or an XGBAugmentedDataset -- the extra
    xgb_pred element is simply ignored here) and build (X, y) arrays
    suitable for scikit-learn / XGBoost: X is (N, 3*22 + 2), y is (N, horizon).
    """
    X, y = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        x_seq, target, regime_ctx = item[0], item[1], item[2]
        summary = summarize_window(x_seq.numpy())
        X.append(np.concatenate([summary, regime_ctx.numpy()]))
        y.append(target.numpy())
    return np.array(X), np.array(y)


class XGBAugmentedDataset:
    """Wraps an FXWindowDataset and a FITTED XGBoostForexModel, precomputing
    the XGBoost prediction for every window ONCE (XGBoost is frozen and
    doesn't change during neural-net training, so recomputing it inside
    every training step would be pure waste), and returning
    (x, y, regime_ctx, xgb_pred) 4-tuples instead of the base dataset's
    3-tuples.

    This is what makes the integration architectural rather than a
    post-hoc average: `xgb_pred` becomes a genuine input tensor to
    HybridCNNLSTMTransformer.forward(), fused with the deep context and
    skip connection inside the network (see models/hybrid_model.py), so
    the regime-aware decoder learns a data-dependent, per-sample way to
    weigh it -- not a single global scalar blend fitted after the fact.
    """

    def __init__(self, base_dataset, xgb_model: "XGBoostForexModel"):
        self.base_dataset = base_dataset
        X, _ = build_xgb_feature_matrix(base_dataset)
        self.xgb_preds = xgb_model.model.predict(X).astype("float32")  # (N, horizon), precomputed once
        assert len(self.xgb_preds) == len(base_dataset)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y, regime_ctx = self.base_dataset[idx]
        xgb_pred = torch.from_numpy(self.xgb_preds[idx])
        return x, y, regime_ctx, xgb_pred

    @property
    def indices(self):
        """Expose the underlying window dataset's origin indices, so
        downstream code (regime labelling, price reconstruction, ARIMA
        origin sampling) that expects `.indices` keeps working unchanged.
        """
        return self.base_dataset.indices

    @property
    def lookback(self):
        return self.base_dataset.lookback

    @property
    def panel(self):
        return self.base_dataset.panel


class XGBoostForexModel:
    """Multi-output XGBoost regressor (one tree ensemble per horizon step,
    via MultiOutputRegressor), matching Paper 1's XGB_TM/XGB_FM/XGB_TM_FM
    setup but adapted to our multi-step horizon.
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 4, learning_rate: float = 0.03, subsample: float = 0.8, colsample_bytree: float = 0.8):
        base = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",  # matches Paper 1's stated objective
            n_jobs=-1,
            random_state=42,
        )
        self.model = MultiOutputRegressor(base)
        self.is_fitted = False

    def fit(self, train_dataset, val_dataset=None):
        X_train, y_train = build_xgb_feature_matrix(train_dataset)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, dataset) -> np.ndarray:
        X, _ = build_xgb_feature_matrix(dataset)
        return self.model.predict(X)

    def predict_batch(self, x_seq: np.ndarray, regime_ctx: np.ndarray) -> np.ndarray:
        """x_seq: (B, T, F), regime_ctx: (B, 2) -> (B, horizon) predictions,
        for use in the Ensemble model where batches come from a DataLoader
        rather than a Dataset.
        """
        summaries = np.array([summarize_window(x) for x in x_seq])
        X = np.concatenate([summaries, regime_ctx], axis=1)
        return self.model.predict(X)
