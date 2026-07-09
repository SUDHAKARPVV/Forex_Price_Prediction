"""
Decoding Currency Dynamics — interactive project dashboard.

Run locally:   streamlit run dashboard/app.py
Pages: Overview · Architecture & Layer I/O · Data & Features · Live Prediction · Results.

The Live Prediction page needs a saved checkpoint (exports/dashboard/hybrid.pt);
create it once with:  python dashboard/save_model.py
Every other page works from the committed artifacts alone.
"""
import os
import sys
import json

# repo root on path + xgboost before torch (macOS/conda OpenMP)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import xgboost  # noqa: F401
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go

from config import DATA_CFG

NAVY = "#1F3759"; TEAL = "#0891B2"; TEAL_L = "#2DD4BF"; GREEN = "#059669"; AMBER = "#B45309"; SLATE = "#64748B"
CKPT = "exports/dashboard"

st.set_page_config(page_title="Decoding Currency Dynamics — Dashboard",
                   page_icon="📈", layout="wide")


# ----------------------------- cached loaders -----------------------------
def load_json(path):
    # NOT cached: JSON summaries are re-read every run so Results always
    # reflects the latest committed benchmark, not a stale cache.
    return json.load(open(path)) if os.path.exists(path) else None


def file_mtime(path):
    import datetime
    if os.path.exists(path):
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    return "—"


@st.cache_resource(show_spinner="Building feature panel + splits …")
def load_panel_and_splits():
    from data.dataset import build_fx_panel, time_split
    panel = build_fx_panel(pair="XAU/USD", n_days=10000, seed=9,
                           source="panel", real_interval="1d")
    train_ds, val_ds, test_ds = time_split(panel)
    return panel, train_ds, val_ds, test_ds


@st.cache_resource(show_spinner="Loading trained model + XGBoost expert …")
def load_model_and_xgb():
    """Returns (hybrid, xgb, test_x) or None if no checkpoint exists yet."""
    if not os.path.exists(os.path.join(CKPT, "hybrid.pt")):
        return None
    import joblib
    from baselines.xgboost_baseline import XGBoostForexModel, XGBAugmentedDataset
    from models.hybrid_model import HybridCNNLSTMTransformer
    panel, train_ds, val_ds, test_ds = load_panel_and_splits()
    xgb = XGBoostForexModel()
    xgb.model = joblib.load(os.path.join(CKPT, "xgb.pkl"))  # fitted MultiOutputRegressor
    test_x = XGBAugmentedDataset(test_ds, xgb)
    hybrid = HybridCNNLSTMTransformer()
    hybrid.load_state_dict(torch.load(os.path.join(CKPT, "hybrid.pt"), map_location="cpu"))
    hybrid.eval()
    return hybrid, xgb, test_x, panel, test_ds


def capture_layer_io(model, x_quant, x_text, regime_ctx, xgb_pred):
    """Register forward hooks on the model's top-level components and record the
    input/output tensor shape + parameter count of each, from one forward pass."""
    records, handles = [], []

    def shp(t):
        if isinstance(t, torch.Tensor):
            return "×".join(str(d) for d in t.shape)
        if isinstance(t, (tuple, list)) and t and isinstance(t[0], torch.Tensor):
            return "×".join(str(d) for d in t[0].shape)
        return "—"

    order = {}

    def mk(name):
        def hook(mod, inp, out):
            order.setdefault(name, len(order))
            records.append({
                "Layer / component": name,
                "Type": mod.__class__.__name__,
                "Input shape": shp(inp[0]) if inp else "—",
                "Output shape": shp(out),
                "Parameters": f"{sum(p.numel() for p in mod.parameters()):,}",
            })
        return hook

    for name, module in model.named_children():
        handles.append(module.register_forward_hook(mk(name)))
    model.eval()
    with torch.no_grad():
        model(x_quant, x_text, regime_ctx, xgb_pred)
    for h in handles:
        h.remove()
    # de-dup keeping first occurrence, in call order
    seen, uniq = set(), []
    for r in records:
        if r["Layer / component"] not in seen:
            seen.add(r["Layer / component"]); uniq.append(r)
    return uniq


def metric_card(col, label, value, color=NAVY, sub=""):
    col.markdown(
        f"<div style='background:#F1F5F9;border-radius:10px;padding:14px 16px'>"
        f"<div style='color:{SLATE};font-size:13px'>{label}</div>"
        f"<div style='color:{color};font-size:30px;font-weight:700;line-height:1.1'>{value}</div>"
        f"<div style='color:{SLATE};font-size:11px'>{sub}</div></div>", unsafe_allow_html=True)


# ----------------------------- sidebar -----------------------------
st.sidebar.title("📈 Decoding Currency Dynamics")
st.sidebar.caption("Hybrid CNN-LSTM-Transformer · XAU/USD multi-step forecasting")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🧱 Architecture & Layer I/O",
    "📊 Data & Features",
    "🔮 Live Prediction",
    "📈 Results & Baselines",
])
meta = load_json(os.path.join(CKPT, "meta.json"))
summ = load_json("multi_seed_summary.json")
st.sidebar.divider()
if meta:
    st.sidebar.success(f"Checkpoint loaded · seed {meta['seed']}\nsaved {meta['saved_at']}")
else:
    st.sidebar.warning("No checkpoint yet.\nRun `python dashboard/save_model.py`\nfor the Live Prediction page.")


# ============================= 1. OVERVIEW =============================
if page.startswith("🏠"):
    st.title("Decoding Currency Dynamics")
    st.markdown("##### AI-Driven Multi-Step Forecasting of Foreign Exchange Rates (XAU/USD)")
    st.caption("Student: PUPPALA V V SUDHAKAR · BITS ID 2024AA05488")
    st.divider()

    hyb = garch = arima = None
    if summ:
        hyb = summ["Hybrid_CNN_LSTM_Transformer"]["DirectionalAccuracy"]["mean"]
        garch = summ.get("GARCH", {}).get("DirectionalAccuracy", {}).get("mean")
        arima = summ.get("ARIMA", {}).get("DirectionalAccuracy", {}).get("mean")
    c = st.columns(4)
    metric_card(c[0], "Hybrid Directional Acc.", f"{hyb:.3f}" if hyb else "—", TEAL, "3-seed mean, 962 test windows")
    metric_card(c[1], "GARCH baseline", f"{garch:.3f}" if garch else "—", NAVY, "econometric benchmark")
    metric_card(c[2], "Model parameters", f"{meta['n_params']/1e6:.2f}M" if meta else "4.39M", GREEN, "dual-tower hybrid")
    metric_card(c[3], "Input features", f"{DATA_CFG.n_total_features}", AMBER, "technical + macro + sentiment")

    st.markdown("")
    st.subheader("Abstract")
    st.markdown(
        "Foreign-exchange markets are among the most liquid yet hardest to forecast — prices are driven at once by "
        "**price action, macroeconomic fundamentals, and market sentiment**, and classical models (ARIMA, GARCH) "
        "assume linearity and stationarity that currency data routinely violates. This project builds an "
        "**AI-driven framework for multi-step forecasting of the XAU/USD (gold) exchange rate**. Its core is a "
        "**Hybrid CNN-LSTM-Transformer** that fuses three real data streams into one model and forecasts the next "
        "**10 trading days** together with a calibrated uncertainty band — so it predicts not just the move, but "
        "how confident it is. Every result is measured honestly against classical baselines under a leakage-free, "
        "regime-aware protocol.")

    st.subheader("What the model does, in one line")
    st.markdown(
        f"<div style='background:#0F172A;color:#CBD5E1;border-radius:10px;padding:14px 16px;font-size:14px'>"
        f"<b style='color:{TEAL_L}'>Price + Macro + News-sentiment</b> &nbsp;→&nbsp; "
        f"CNN (local patterns) → cross-attention fusion → Transformer (global context) → Bi-LSTM/GRU (memory) "
        f"→ &nbsp;<b style='color:{TEAL_L}'>10-day forecast + confidence band</b></div>",
        unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("🎯 Goals & objectives")
        st.markdown(
            "- Design a **Hybrid CNN-LSTM-Transformer** for multi-step FX forecasting\n"
            "- Build a **multi-modal fusion pipeline** (technical + macro + FinBERT news sentiment)\n"
            "- Quantify each component's contribution via **ablation studies**\n"
            "- Evaluate across **horizons and volatility regimes**, honestly\n"
            "- Benchmark against **ARIMA / GARCH** with full walk-forward\n"
            "- Provide a **regime-aware, uncertainty-calibrated** forecast, not just a point estimate")
    with g2:
        st.subheader("🔭 Scope & approach")
        st.markdown(
            "- Instrument: **XAU/USD (gold)**, daily bars, ~26 years of history\n"
            "- **31 engineered features** across 3 streams, 60-bar lookback\n"
            "- **Two-pipeline** design: data extraction/verification, then train/test\n"
            "- Training: **freeze-and-tune**, Gaussian-NLL heads, modality masking, deep supervision\n"
            "- Decision layer: **conviction filtering** + costed backtest\n"
            "- Reproducible, open-source Python stack (PyTorch, FinBERT, XGBoost)")

    st.info(
        "**Honest status.** GARCH's momentum drift still leads unfiltered directional accuracy; the Hybrid "
        "(~0.53) narrows the gap with much lower variance, best MAE among deep configs, and adds a probabilistic "
        "conviction layer. The evaluator's target of **0.60** is the current work item — the main lever is denser "
        "news coverage (currently ~18% of test bars). Use the sidebar to explore the architecture, data, live "
        "predictions and results.", icon="ℹ️")


# ================== 2. ARCHITECTURE & LAYER I/O ==================
elif page.startswith("🧱"):
    st.title("🧱 Architecture & Layer Input/Output")
    st.markdown(
        "The Hybrid is a **dual-tower** network. Below is the pictorial data-flow with the **tensor shape at every "
        "hand-off**, then a **live table of every component's input → output shape and parameter count**, captured "
        "from a real forward pass (batch size B = 1).")

    st.subheader("Pictorial architecture (data flow with shapes)")
    dot = f"""
    digraph G {{
      rankdir=LR; bgcolor="transparent"; splines=ortho;
      node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#CBD5E1" fontcolor="white"];
      edge [color="#64748B" fontname="Helvetica" fontsize=9 fontcolor="#334155"];

      qin  [label="Quant input\\n(B,60,18)" fillcolor="#94A3B8" fontcolor="#0F172A"];
      cnn  [label="Dilated Causal CNN\\n(B,60,128)" fillcolor="#1E2738"];
      tin  [label="Text / sentiment input\\n(B,60,13)" fillcolor="#94A3B8" fontcolor="#0F172A"];
      gru  [label="Sentiment GRU\\n(B,60,128)" fillcolor="#1E2738"];
      fuse [label="Cross-Attention Fusion\\n+ presence gate\\n(B,60,128)" fillcolor="#0891B2"];
      trf  [label="Transformer Encoder\\n(B,60,256)" fillcolor="#1E2738"];
      rec  [label="Bi-LSTM ∥ Bi-GRU\\n+ attention pool\\n(B,256)" fillcolor="#1E2738"];
      head [label="Regime-aware heads\\n(μ, σ²) × 10" fillcolor="#059669"];
      xgb  [label="XGBoost expert\\n(B,10)" fillcolor="#B45309"];
      out  [label="Forecast + band\\n(B,10)" fillcolor="#1F3759"];

      qin -> cnn [label="Tower A"];
      tin -> gru [label="Tower B"];
      cnn -> fuse [label="Query"];
      gru -> fuse [label="Key/Value"];
      fuse -> trf; trf -> rec; rec -> head;
      head -> out [label="deep"];
      xgb -> out [label="trust gate"];
    }}"""
    st.graphviz_chart(dot, use_container_width=True)

    st.subheader("Live per-component input / output shapes")
    try:
        from models.hybrid_model import HybridCNNLSTMTransformer
        m = HybridCNNLSTMTransformer()
        B, T = 1, DATA_CFG.lookback
        xq = torch.zeros(B, T, DATA_CFG.n_technical_features + DATA_CFG.n_macro_features)
        xt = torch.zeros(B, T, DATA_CFG.n_sentiment_features)
        rc = torch.zeros(B, 2)
        xg = torch.zeros(B, DATA_CFG.horizon)
        rows = capture_layer_io(m, xq, xt, rc, xg)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Total trainable parameters: {m.count_parameters():,}  "
                   f"({m.count_parameters()/1e6:.2f}M). Shapes are exact; weights are irrelevant to shape.")
    except Exception as e:
        st.error(f"Could not introspect the model: {e}")


# ===================== 3. DATA & FEATURES =====================
elif page.startswith("📊"):
    st.title("📊 Data & Feature Engineering")
    st.markdown("Three real, incrementally-cached streams are aligned to a common daily grid, giving "
                f"**{DATA_CFG.n_total_features} features** per bar over a 60-bar lookback window.")
    names = meta["feature_names"] if meta else None
    if not os.path.exists("exports/feature_panel.csv"):
        st.error("exports/feature_panel.csv not found — run `python build_dataset.py` first.")
    else:
        dfp = pd.read_csv("exports/feature_panel.csv")
        feat_cols = [c for c in dfp.columns if c not in ("date", "close", "realized_vol", "atr")]
        nt, nm = DATA_CFG.n_technical_features, DATA_CFG.n_macro_features
        tech, macro, sent = feat_cols[:nt], feat_cols[nt:nt+nm], feat_cols[nt+nm:]
        c = st.columns(3)
        c[0].markdown(f"**🟦 Technical ({len(tech)})**"); c[0].caption(", ".join(tech))
        c[1].markdown(f"**🟩 Macro ({len(macro)})**"); c[1].caption(", ".join(macro))
        c[2].markdown(f"**🟪 Sentiment ({len(sent)})**"); c[2].caption(", ".join(sent))
        st.divider()
        colA, colB = st.columns([2, 1])
        with colA:
            st.subheader("Gold price (XAU/USD proxy, GC=F)")
            dts = pd.to_datetime(dfp["date"], utc=True, errors="coerce")
            fig = go.Figure(go.Scatter(x=dts, y=dfp["close"], line=dict(color=TEAL, width=1)))
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title="close", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            st.subheader("At a glance")
            st.metric("Bars", f"{len(dfp):,}")
            if "sig_none" in dfp.columns:
                test = dfp.iloc[int(len(dfp)*0.85):]
                st.metric("Test-set news coverage", f"{(test['sig_none']==0).mean()*100:.1f}%")
            st.metric("Date range", f"{str(dfp['date'].iloc[0])[:10]} → {str(dfp['date'].iloc[-1])[:10]}")
        st.divider()
        # ---- Macro indicators ----
        st.subheader("📉 Macroeconomic indicators (stationary, real feeds)")
        st.caption("Yahoo rates/dollar-index + BLS CPI, transformed to stationary form and forward-filled onto "
                   "the daily grid. Shown over the recent window for readability.")
        macro_present = [m_ for m_ in macro if m_ in dfp.columns]
        recent = dfp.tail(750)
        rdts = pd.to_datetime(recent["date"], utc=True, errors="coerce")
        mfig = go.Figure()
        palette = [TEAL, NAVY, GREEN, AMBER, "#7C3AED", SLATE]
        for i, mcol in enumerate(macro_present):
            mfig.add_trace(go.Scatter(x=rdts, y=recent[mcol], name=mcol,
                                      line=dict(color=palette[i % len(palette)], width=1.4)))
        mfig.update_layout(height=300, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                           legend=dict(orientation="h", y=1.12), yaxis_title="stationary value")
        st.plotly_chart(mfig, use_container_width=True)

        st.divider()
        # ---- FinBERT sentiment scoring ----
        st.subheader("🗞️ FinBERT news-sentiment scoring")
        s1, s2 = st.columns([3, 2])
        with s1:
            st.markdown("**Per-bar sentiment signal** (decayed score + diffusion breadth), recent window")
            sfig = go.Figure()
            if "sent_decay" in dfp.columns:
                sfig.add_trace(go.Scatter(x=rdts, y=recent["sent_decay"], name="sent_decay (EWMA)",
                                          line=dict(color=TEAL, width=1.6)))
            if "sent_diffusion" in dfp.columns:
                sfig.add_trace(go.Scatter(x=rdts, y=recent["sent_diffusion"], name="diffusion breadth",
                                          line=dict(color=AMBER, width=1.4)))
            # buy / sell markers
            for col, nm, col_c, sym in (("sig_buy", "BUY", GREEN, "triangle-up"),
                                        ("sig_sell", "SELL", "#DC2626", "triangle-down")):
                if col in recent.columns:
                    mk = recent[col] == 1
                    if mk.any():
                        sfig.add_trace(go.Scatter(x=rdts[mk.values], y=recent.loc[mk, "sent_decay"] if "sent_decay" in recent else recent.loc[mk, col]*0,
                                                  mode="markers", name=nm,
                                                  marker=dict(color=col_c, size=8, symbol=sym)))
            sfig.add_hline(y=0, line_dash="dot", line_color=SLATE)
            sfig.update_layout(height=300, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", y=1.12), yaxis_title="sentiment")
            st.plotly_chart(sfig, use_container_width=True)
        with s2:
            st.markdown("**FinBERT per-headline polarity** (whole news archive)")
            arch = "exports/archive/news_GCF.csv"
            if os.path.exists(arch):
                a = pd.read_csv(arch)
                if "polarity" in a.columns:
                    hfig = go.Figure(go.Histogram(x=a["polarity"].dropna(), nbinsx=30, marker_color=TEAL))
                    hfig.update_layout(height=300, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                                       xaxis_title="polarity  (−1 bearish → +1 bullish)", yaxis_title="headlines")
                    st.plotly_chart(hfig, use_container_width=True)
                    npos = int((a["polarity"] >= 0.15).sum()); nneg = int((a["polarity"] <= -0.15).sum())
                    st.caption(f"{len(a):,} scored headlines · {npos:,} bullish · {nneg:,} bearish · "
                               f"{len(a)-npos-nneg:,} neutral")
            else:
                st.caption("News archive not present in this deployment.")

        st.divider()
        st.subheader("Most recent engineered features")
        st.dataframe(dfp[["date"] + feat_cols].tail(8), use_container_width=True, hide_index=True)


# ===================== 4. LIVE PREDICTION =====================
elif page.startswith("🔮"):
    st.title("🔮 Live Prediction")
    bundle = load_model_and_xgb()
    if bundle is None:
        st.warning("No trained checkpoint found. Generate one first:")
        st.code("python dashboard/save_model.py", language="bash")
        st.stop()
    hybrid, xgb, test_x, panel, test_ds = bundle
    n = len(test_x)
    origins = test_ds.indices
    dates = [str(panel.dates[t])[:10] for t in origins]

    st.markdown("Pick a forecast origin from the **test set** (unseen data), then press **FX Price Predict** — "
                "the trained model runs a live forward pass and predicts the next 10 daily log-return steps with an "
                "uncertainty band, which we compare to what actually happened.")
    cpick, cbtn = st.columns([3, 1])
    with cpick:
        idx = st.slider("Test-set forecast origin", 0, n - 1, n - 1,
                        format="%d", help="Rightmost = most recent test bar")
        st.caption(f"Origin date: **{dates[idx]}**  ·  test window {idx+1} of {n}")
    with cbtn:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        go_pred = st.button("🔮 FX Price Predict", type="primary", use_container_width=True)

    # Run inference only on button click; persist the result across reruns.
    if go_pred:
        x_quant, x_text, y, regime_ctx, xgb_pred = test_x[idx]
        xb = {k: v.unsqueeze(0) for k, v in
              dict(x_quant=x_quant, x_text=x_text, regime_ctx=regime_ctx, xgb_pred=xgb_pred).items()}
        hybrid.eval()
        with torch.no_grad():
            out = hybrid(xb["x_quant"], xb["x_text"], xb["regime_ctx"], xb["xgb_pred"])
        fc = out["forecast"][0].numpy() if isinstance(out, dict) else out[0].numpy()
        bd = out["band"][0].numpy() if isinstance(out, dict) and out.get("band") is not None else None
        st.session_state["live_pred"] = {
            "idx": idx, "date": dates[idx], "forecast": fc, "band": bd,
            "actual": y.numpy(), "xgb1": float(xgb_pred[0].item()),
            "layers": capture_layer_io(hybrid, xb["x_quant"], xb["x_text"], xb["regime_ctx"], xb["xgb_pred"]),
        }

    if "live_pred" not in st.session_state:
        st.info("Press **🔮 FX Price Predict** to run the model on the selected date.", icon="👆")
        st.stop()

    P = st.session_state["live_pred"]
    forecast, band, actual = P["forecast"], P["band"], P["actual"]
    st.success(f"Prediction for origin **{P['date']}** (test window {P['idx']+1} of {n})")
    h = np.arange(1, DATA_CFG.horizon + 1)
    fig = go.Figure()
    if band is not None:
        fig.add_trace(go.Scatter(x=np.r_[h, h[::-1]],
                                 y=np.r_[forecast + band, (forecast - band)[::-1]],
                                 fill="toself", fillcolor="rgba(8,145,178,0.15)",
                                 line=dict(width=0), name="uncertainty band"))
    fig.add_trace(go.Scatter(x=h, y=forecast, name="forecast", line=dict(color=TEAL, width=3)))
    fig.add_trace(go.Scatter(x=h, y=actual, name="actual", line=dict(color=NAVY, width=2, dash="dot")))
    fig.update_layout(height=340, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="forecast horizon (days ahead)", yaxis_title="cumulative log-return")
    st.plotly_chart(fig, use_container_width=True)

    dir_hit = (np.sign(forecast) == np.sign(actual)).mean()
    conv = float(np.abs(forecast[0]) / (band[0] + 1e-9)) if band is not None else float(abs(forecast[0]))
    sig = "BUY" if forecast[0] > 0 else "SELL"
    c = st.columns(4)
    metric_card(c[0], "1-step direction", sig, GREEN if sig == "BUY" else AMBER)
    metric_card(c[1], "Directional hit-rate", f"{dir_hit*100:.0f}%", TEAL, "this window, 10 horizons")
    metric_card(c[2], "Conviction |μ|/σ", f"{conv:.2f}", NAVY, "t-statistic of the 1-step move")
    metric_card(c[3], "XGBoost expert (1-step)", f"{P['xgb1']:+.4f}", SLATE, "fused internal expert")

    with st.expander("🔬 Per-layer output shapes for THIS prediction"):
        st.dataframe(pd.DataFrame(P["layers"]), use_container_width=True, hide_index=True)


# ===================== 5. RESULTS & BASELINES =====================
elif page.startswith("📈"):
    st.title("📈 Results & Baselines")
    hc, rc = st.columns([4, 1])
    hc.caption(f"Live from the latest committed benchmark · `multi_seed_summary.json` updated "
               f"**{file_mtime('multi_seed_summary.json')}**. Re-run `python run_multi_seed.py --source panel` "
               f"to refresh the numbers, then reload.")
    if rc.button("🔄 Refresh", use_container_width=True):
        st.cache_resource.clear(); st.rerun()
    if not summ:
        st.error("multi_seed_summary.json not found — run `python run_multi_seed.py --source panel`.")
    else:
        n_test = (meta.get("split", {}).get("test") if meta else None) or 962
        nice = {"Hybrid_CNN_LSTM_Transformer": "Hybrid CNN-LSTM-Transformer",
                "GARCH": "GARCH (AR1-GARCH1,1)", "ARIMA": "ARIMA (walk-forward)"}
        rows = []
        for k, label in nice.items():
            if k in summ:
                rows.append({"Model": label,
                             "DirAcc (mean)": round(summ[k]["DirectionalAccuracy"]["mean"], 4),
                             "DirAcc (std)": round(summ[k]["DirectionalAccuracy"]["std"], 4),
                             "MAE": round(summ[k]["MAE"]["mean"], 5),
                             "RMSE": round(summ[k]["RMSE"]["mean"], 5)})
        df = pd.DataFrame(rows).sort_values("DirAcc (mean)", ascending=False)
        st.subheader(f"Walk-forward comparison ({n_test} test windows, 3 seeds)")
        st.dataframe(df, use_container_width=True, hide_index=True)

        vals = {nice[k]: summ[k]["DirectionalAccuracy"]["values"] for k in nice if k in summ}
        fig = go.Figure()
        for name, v in vals.items():
            fig.add_trace(go.Bar(name=name, x=["seed 9", "seed 36", "seed 99"], y=v))
        fig.update_layout(barmode="group", height=320, template="plotly_white",
                          yaxis_title="Directional accuracy", yaxis_range=[0.45, 0.6],
                          margin=dict(l=0, r=0, t=10, b=0))
        fig.add_hline(y=0.5, line_dash="dot", line_color=SLATE, annotation_text="coin flip")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Ablation — sentiment diffusion feature")
        st.table(pd.DataFrame([
            {"Configuration": "Without sent_diffusion (30 feat)", "DirAcc": 0.5006},
            {"Configuration": "Placebo — shuffled diffusion (31 feat)", "DirAcc": 0.5209},
            {"Configuration": "With real sent_diffusion (31 feat)", "DirAcc": 0.5345},
        ]))
        st.caption("The +3.4pp gain decomposes into ~2.0pp added-channel effect (a noise column achieves it) "
                   "and ~1.4pp genuine diffusion signal. See report Section 6a.")
