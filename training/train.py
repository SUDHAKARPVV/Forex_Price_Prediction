"""
Training loop, shared by the Hybrid model and the deep-learning baselines
(VanillaLSTM, SimplifiedTFT), all three of which now return
{"forecast": ..., "direction_logits": ...}. ARIMA/Prophet are fit directly
at evaluation time in evaluate.py since they are not gradient-trained.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TRAIN_CFG


def combined_loss(pred: torch.Tensor, target: torch.Tensor, directional_weight: float = 0.0) -> torch.Tensor:
    """Huber (robust) regression loss plus an optional directional
    (sign-agreement) hinge penalty.

    Huber replaces plain MSE after the hourly-scale round showed the model
    chasing volatility spikes: with squared error, one 3-sigma bar
    outweighs dozens of ordinary bars, so training gradients are dominated
    by exactly the unpredictable jumps the model should NOT fit. Huber is
    quadratic near zero and linear beyond delta -- the industry-standard
    robust loss for financial return targets. Inputs arrive here already
    scaled by return_scale (see total_loss), so delta=1.0 means "one
    representative return magnitude": ordinary bars stay in the quadratic
    regime, spikes get a bounded, linear gradient.

    Pure MSE can be minimised by predictions that are small and correctly
    *scaled* but wrong in *sign* -- exactly the failure mode the initial
    evaluation run showed (low MAE, ~50% directional accuracy). The second
    term, relu(-pred * target), is zero whenever pred and target already
    agree in sign (regardless of magnitude, so it never rewards inflating
    predictions purely to "look more confident"), and grows linearly with
    the size of the disagreement when they don't. Because pred and target
    are both in log-return units, this term is naturally the same order of
    magnitude as the MSE term, so it nudges the model toward getting the
    *direction* right without needing any hand-tuned scale constant (an
    earlier tanh-saturation version of this loss did need one, and got it
    wrong -- it dominated the MSE term and caused prediction magnitudes to
    blow up while barely improving directional accuracy; kept as a cautionary
    note for anyone modifying this function).
    """
    reg = nn.functional.huber_loss(pred, target, delta=1.0)
    if directional_weight <= 0:
        return reg
    disagreement = torch.relu(-pred * target)
    return reg + directional_weight * disagreement.mean()


def directional_bce_loss(direction_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy between predicted P(return > 0) and the actual
    sign of the target return. This is what directly optimises directional
    accuracy -- the regression loss above only optimises it indirectly.
    """
    sign_target = (target > 0).float()
    return nn.functional.binary_cross_entropy_with_logits(direction_logits, sign_target)


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    direction_logits,
    directional_weight: float,
    classification_weight: float,
    return_scale: float = TRAIN_CFG.return_scale,
) -> torch.Tensor:
    """Combine the regression loss with the auxiliary classification loss
    on a comparable numeric scale.

    Log-return targets are tiny (~1e-2), so raw MSE sits around 1e-4 while
    the BCE classification loss sits around 0.1-0.7 -- a ~1000-4000x scale
    mismatch. An earlier version of this training loop combined them with a
    flat weight and the (much larger) BCE term completely swamped the MSE
    term, so the shared backbone was effectively only being trained for
    classification, which then overfit rapidly (train loss kept falling,
    val loss diverged after ~2 epochs) with no regression signal to
    regularise it. Dividing pred/target by `return_scale` (a fixed,
    representative log-return magnitude) before computing the regression
    loss brings both terms to a comparable O(1) scale before weighting --
    this does NOT change the model's actual predictions, only how the loss
    used for backpropagation is computed.
    """
    reg_loss = combined_loss(pred / return_scale, target / return_scale, directional_weight=directional_weight)
    if direction_logits is None or classification_weight <= 0:
        return reg_loss
    return reg_loss + classification_weight * directional_bce_loss(direction_logits, target)


def gaussian_nll_loss(pred: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor,
                      return_scale: float = TRAIN_CFG.return_scale) -> torch.Tensor:
    """Gaussian negative log-likelihood on return_scale-standardised
    residuals: 0.5 * (log sigma^2 + z^2 / sigma^2), z = (y - mu)/scale.

    This is the probabilistic-head training objective (GARCH emulation):
    the network must parameterise its own conditional variance per horizon
    step, so it is rewarded for tight bands when it is right and penalised
    for false confidence -- instead of MSE's pull toward a conservative
    zero mean on fat-tailed returns.
    """
    z2 = ((target - pred) / return_scale) ** 2
    return 0.5 * (log_var + z2 * torch.exp(-log_var)).mean()


def _forward(model, x_quant, x_text, regime_ctx, xgb_pred=None):
    """All models (hybrid + baselines) return {"forecast", "direction_logits"}.
    Baselines ignore xgb_pred (uniform calling convention); only the Hybrid
    returns "deep_forecast" (deep-supervision target) and "log_var"
    (probabilistic head -- switches the main loss to Gaussian NLL)."""
    out = model(x_quant, x_text, regime_ctx, xgb_pred)
    return out["forecast"], out.get("direction_logits"), out.get("deep_forecast"), out.get("log_var")


def _unpack_batch(batch):
    """data/dataset.py:FXWindowDataset yields 4-tuples
    (x_quant, x_text, y, regime_ctx); XGBAugmentedDataset yields 5-tuples
    with a precomputed xgb_pred appended. Handle both."""
    if len(batch) == 5:
        x_quant, x_text, y, regime_ctx, xgb_pred = batch
    else:
        x_quant, x_text, y, regime_ctx = batch
        xgb_pred = None
    return x_quant, x_text, y, regime_ctx, xgb_pred


def train_model(
    model,
    train_ds,
    val_ds,
    epochs: int = None,
    lr: float = None,
    batch_size: int = None,
    verbose: bool = True,
    device: str = "cpu",
    directional_weight: float = None,
    classification_weight: float = None,
    seed: int = None,
):
    epochs = epochs or TRAIN_CFG.epochs
    lr = lr or TRAIN_CFG.lr
    batch_size = batch_size or TRAIN_CFG.batch_size
    directional_weight = TRAIN_CFG.directional_loss_weight if directional_weight is None else directional_weight
    classification_weight = TRAIN_CFG.classification_loss_weight if classification_weight is None else classification_weight

    # Seed the training RNG (batch shuffling order) with the RUN's seed,
    # not a fixed constant. The old fixed TRAIN_CFG.seed here meant
    # "multi-seed" runs on real data were near-identical replicas: the
    # --seeds argument only varied synthetic DATA generation, so with real
    # prices/macro/news nothing varied at all (exposed by a 3-seed run
    # with std +/- 0.000 on every model). Model INITIALISATION is seeded
    # separately in main.py before each model is constructed.
    torch.manual_seed(seed if seed is not None else TRAIN_CFG.seed)
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Only optimise trainable params -- respects a frozen quant tower in
    # stage 2 of freeze-and-tune training (models/hybrid_model.py).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=TRAIN_CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    def loss_fn(pred, direction_logits, target, deep_forecast=None, log_var=None):
        if log_var is not None:
            # Probabilistic head: Gaussian NLL on the blended forecast with
            # the deep expert's predicted variance, plus the directional
            # hinge (on scale-standardised values, as before).
            loss = gaussian_nll_loss(pred, target, log_var)
            if directional_weight > 0:
                s = TRAIN_CFG.return_scale
                loss = loss + directional_weight * torch.relu(-(pred / s) * (target / s)).mean()
            if direction_logits is not None and classification_weight > 0:
                loss = loss + classification_weight * directional_bce_loss(direction_logits, target)
        else:
            loss = total_loss(
                pred, target, direction_logits,
                directional_weight=directional_weight,
                classification_weight=classification_weight,
            )
        if deep_forecast is not None and TRAIN_CFG.deep_supervision_weight > 0:
            # Deep supervision: the deep expert's MEAN is held to the robust
            # point objective (Huber + hinge), so it stays a complete
            # forecaster instead of hiding behind XGBoost.
            loss = loss + TRAIN_CFG.deep_supervision_weight * total_loss(
                deep_forecast, target, None,
                directional_weight=directional_weight,
                classification_weight=0.0,
            )
        return loss

    # Checkpoint selection: validation DIRECTIONAL ACCURACY first, val loss
    # as tiebreak. Selecting on val MSE alone made the Hybrid's residual
    # XGBoost fusion collapse to exactly XGBoost (correction -> 0, trust
    # -> 1, test DirAcc within 0.001 of the standalone trees): the epochs
    # where the deep pathway improves SIGN agreement rarely coincide with
    # the epochs of minimum squared error, because the directional signal
    # (mood/sentiment) barely moves the MSE needle on noisy FX returns.
    # DirectionalAccuracy is also the headline evaluation metric, so the
    # checkpoint criterion and the reported metric now agree. Applied
    # identically to every deep model (Hybrid AND baselines) -- this is a
    # training-procedure change, not a thumb on the comparison scale.
    best_dir_acc = -1.0
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_dir_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for batch in train_loader:
            x_quant, x_text, y, regime_ctx, xgb_pred = _unpack_batch(batch)
            x_quant, x_text, y, regime_ctx = x_quant.to(device), x_text.to(device), y.to(device), regime_ctx.to(device)
            if xgb_pred is not None:
                xgb_pred = xgb_pred.to(device)
            optimizer.zero_grad()
            pred, direction_logits, deep_forecast, log_var = _forward(model, x_quant, x_text, regime_ctx, xgb_pred)
            loss = loss_fn(pred, direction_logits, y, deep_forecast, log_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        sign_hits, sign_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x_quant, x_text, y, regime_ctx, xgb_pred = _unpack_batch(batch)
                x_quant, x_text, y, regime_ctx = x_quant.to(device), x_text.to(device), y.to(device), regime_ctx.to(device)
                if xgb_pred is not None:
                    xgb_pred = xgb_pred.to(device)
                pred, direction_logits, deep_forecast, log_var = _forward(model, x_quant, x_text, regime_ctx, xgb_pred)
                val_losses.append(loss_fn(pred, direction_logits, y, deep_forecast, log_var).item())
                sign_hits += (torch.sign(pred) == torch.sign(y)).sum().item()
                sign_total += y.numel()

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        val_dir_acc = sign_hits / sign_total if sign_total else 0.0
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dir_acc"].append(val_dir_acc)

        if verbose:
            print(f"epoch {epoch:02d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_dir_acc={val_dir_acc:.4f} | {time.time()-t0:.1f}s")

        improved = (val_dir_acc > best_dir_acc + 1e-6) or (
            abs(val_dir_acc - best_dir_acc) <= 1e-6 and val_loss < best_val - 1e-7
        )
        if improved:
            best_dir_acc = val_dir_acc
            best_val = min(best_val, val_loss)
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CFG.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val_dir_acc={best_dir_acc:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history


def train_two_stage(model, train_ds_full, val_ds_full, train_ds_text, val_ds_text,
                    epochs, lr, device="cpu", seed=None, **kw):
    """Freeze-and-tune (Fix 2): two-stage training.

    Stage 1 -- train the QUANTITATIVE pipeline text-free on the FULL
    dataset (model.text_enabled = False, so the text tower is bypassed and
    its 17 news-less historical years cannot dilute the quant weights).

    Stage 2 -- freeze the quant tower, enable the text tower, and fine-tune
    the text + fusion + decoder path at a reduced LR on the NEWS-DENSE
    recent subset (train_ds_text / val_ds_text), so the sentiment pathway
    is learned only where headlines actually exist.

    Falls back to single-stage train_model if the text subset is too small.
    """
    stage2_epochs = max(2, int(epochs * TRAIN_CFG.two_stage_stage2_frac))
    stage1_epochs = max(2, epochs - stage2_epochs)

    if train_ds_text is None or len(train_ds_text) < TRAIN_CFG.batch_size * 2:
        print("[two-stage] text-dense subset too small -- falling back to single-stage training")
        return train_model(model, train_ds_full, val_ds_full, epochs=epochs, lr=lr,
                           device=device, seed=seed, **kw)

    print(f"[two-stage] Stage 1: quant-only, text bypassed, {stage1_epochs} epochs, full history "
          f"(train={len(train_ds_full)})")
    model.set_text_enabled(False)
    model.freeze_quant_tower(False)
    model, hist1 = train_model(model, train_ds_full, val_ds_full, epochs=stage1_epochs,
                               lr=lr, device=device, seed=seed, **kw)

    frozen = model.freeze_quant_tower(True)
    model.set_text_enabled(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[two-stage] Stage 2: text+fusion+decoder fine-tune, {stage2_epochs} epochs, "
          f"news-dense subset (train={len(train_ds_text)}); froze {frozen:,} quant params, "
          f"{trainable:,} trainable")
    model, hist2 = train_model(model, train_ds_text, val_ds_text, epochs=stage2_epochs,
                               lr=lr * TRAIN_CFG.two_stage_stage2_lr_mult, device=device,
                               seed=(seed + 100) if seed is not None else None, **kw)

    model.freeze_quant_tower(False)  # leave everything trainable for downstream use
    history = {k: hist1.get(k, []) + hist2.get(k, []) for k in ("train_loss", "val_loss", "val_dir_acc")}
    history["stage1_epochs"] = len(hist1["train_loss"])
    return model, history
