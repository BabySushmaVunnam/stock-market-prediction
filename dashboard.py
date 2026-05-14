"""
Stock Market Prediction — Streamlit Dashboard

Provides a visual UI to run each pipeline step and view predictions.
Run with:  streamlit run dashboard.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PYTHON = sys.executable
ROOT   = Path(__file__).parent

# Track which steps were actually run this session
if "steps_run" not in st.session_state:
    st.session_state.steps_run = set()   # values: 1, 2, 3, 4

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.step-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 8px;
    min-height: 160px;
}
.step-number {
    font-size: 11px;
    color: #6c7086;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.step-title {
    font-size: 18px;
    font-weight: 700;
    color: #cdd6f4;
    margin: 4px 0 6px 0;
}
.step-desc {
    font-size: 13px;
    color: #6c7086;
    line-height: 1.5;
}
.status-done  { color: #a6e3a1; font-size: 13px; font-weight: 600; }
.status-ready { color: #89b4fa; font-size: 13px; }
.status-wait  { color: #6c7086; font-size: 13px; }
.pred-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #313244;
    text-align: center;
}
.pred-ticker { font-size: 22px; font-weight: 800; color: #cdd6f4; }
.pred-price  { font-size: 28px; font-weight: 700; margin: 6px 0; }
.pred-up     { color: #a6e3a1; }
.pred-down   { color: #f38ba8; }
.pred-meta   { font-size: 12px; color: #6c7086; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"]

def raw_data_exists() -> bool:
    return any((ROOT / "data" / "raw").glob("*/*.parquet") if (ROOT / "data" / "raw").exists() else [])

def processed_data_exists() -> bool:
    return any((ROOT / "data" / "processed").glob("*.parquet")) if (ROOT / "data" / "processed").exists() else False

def models_exist() -> bool:
    return any((ROOT / "models" / "artifacts").glob("*_xgboost.pkl")) if (ROOT / "models" / "artifacts").exists() else False

def run_step(cmd: list[str], log_area) -> bool:
    """Run a subprocess and stream output to a Streamlit text area."""
    proc = subprocess.Popen(
        [PYTHON] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
    )
    output_lines = []
    for line in proc.stdout:
        output_lines.append(line.rstrip())
        log_area.code("\n".join(output_lines[-40:]), language="")
    proc.wait()
    return proc.returncode == 0

def load_predictions() -> list[dict]:
    try:
        from serving.predict import available_tickers, predict_next_close
        results = []
        for t in available_tickers():
            try:
                results.append(predict_next_close(t))
            except Exception:
                pass
        return results
    except Exception:
        return []

def load_processed(ticker: str) -> pd.DataFrame | None:
    path = ROOT / "data" / "processed" / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

# ── Header ────────────────────────────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## 📈 Stock Market Prediction Pipeline")
    st.markdown("Run each step in order, then explore live predictions and charts below.")
with col_h2:
    st.markdown("<br/>", unsafe_allow_html=True)
    st.link_button("🔗 Open API Docs (Swagger)", "http://localhost:8000/docs", use_container_width=True)
    st.link_button("⚡ Raw API — /predict/all",  "http://localhost:8000/predict/all", use_container_width=True)
st.divider()

# ── Pipeline tiles ────────────────────────────────────────────────────────────

st.markdown("### Pipeline Steps")

col1, col2, col3, col4 = st.columns(4)

STEPS = [
    {
        "col":   col1,
        "num":   "Step 1",
        "title": "Data Ingestion",
        "desc":  "Fetch 2 years of OHLCV data from Yahoo Finance and save as partitioned Parquet.",
        "cmd":   ["-m", "ingestion.fetch_stocks", "--days", "730"],
        "ready": lambda: True,
        "done":  raw_data_exists,
    },
    {
        "col":   col2,
        "num":   "Step 2",
        "title": "Feature Engineering",
        "desc":  "Compute 25 technical indicators: MA, EMA, MACD, RSI, Bollinger Bands, lag features.",
        "cmd":   ["-m", "pipeline.features"],
        "ready": raw_data_exists,
        "done":  processed_data_exists,
    },
    {
        "col":   col3,
        "num":   "Step 3",
        "title": "Train Models",
        "desc":  "Train XGBoost on each ticker using a time-series split. Saves model + metadata.",
        "cmd":   ["-m", "pipeline.features"],   # placeholder; handled specially below
        "ready": processed_data_exists,
        "done":  models_exist,
    },
    {
        "col":   col4,
        "num":   "Step 4",
        "title": "Predictions",
        "desc":  "Load trained models and predict next trading day's close price for all tickers.",
        "cmd":   [],
        "ready": models_exist,
        "done":  models_exist,
    },
]

for step in STEPS:
    with step["col"]:
        step_num = int(step["num"].split()[1])
        ran_this_session = step_num in st.session_state.steps_run
        is_ready = step["ready"]()

        if ran_this_session:
            status  = "✅ Complete"
            css_cls = "status-done"
        elif not is_ready:
            status  = "⏳ Waiting for previous step"
            css_cls = "status-wait"
        else:
            status  = "🔵 Ready to run"
            css_cls = "status-ready"

        st.markdown(f"""
        <div class="step-card">
          <div class="step-number">{step['num']}</div>
          <div class="step-title">{step['title']}</div>
          <div class="step-desc">{step['desc']}</div>
          <br/>
          <span class="{css_cls}">{status}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Run buttons ───────────────────────────────────────────────────────────────

b1, b2, b3, b4 = st.columns(4)

with b1:
    run1 = st.button("▶ Run Ingestion",         use_container_width=True, type="primary")
with b2:
    run2 = st.button("▶ Run Feature Engineering", use_container_width=True,
                     type="primary", disabled=not raw_data_exists())
with b3:
    run3 = st.button("▶ Train All Models",       use_container_width=True,
                     type="primary", disabled=not processed_data_exists())
with b4:
    run4 = st.button("🔄 Refresh Predictions",   use_container_width=True,
                     type="secondary", disabled=not models_exist())

# ── Step execution ────────────────────────────────────────────────────────────

if run1:
    st.markdown("#### Step 1 — Ingestion Log")
    log = st.empty()
    with st.spinner("Fetching data from Yahoo Finance..."):
        ok = run_step(["-m", "ingestion.fetch_stocks", "--days", "730"], log)
    if ok:
        st.session_state.steps_run.add(1)
        st.success("Ingestion complete!")
    else:
        st.error("Ingestion failed. Check log above.")

if run2:
    st.markdown("#### Step 2 — Feature Engineering Log")
    log = st.empty()
    with st.spinner("Computing technical indicators..."):
        ok = run_step(["-m", "pipeline.features"], log)
    if ok:
        st.session_state.steps_run.add(2)
        st.success("Features computed!")
    else:
        st.error("Feature engineering failed.")

if run3:
    st.markdown("#### Step 3 — Training Log")
    all_ok = True
    for ticker in TICKERS:
        st.markdown(f"**Training {ticker}...**")
        log = st.empty()
        ok = run_step(["-m", "models.train", "--ticker", ticker], log)
        if not ok:
            all_ok = False
    if all_ok:
        st.session_state.steps_run.add(3)
        st.success("All models trained!")
    else:
        st.error("Some models failed.")

if run4:
    st.session_state.steps_run.add(4)
    st.rerun()

st.divider()

# ── Prediction cards ──────────────────────────────────────────────────────────

st.markdown("### Live Predictions — Next Trading Day")

predictions = load_predictions()

if not predictions:
    st.info("No predictions yet. Complete Steps 1–3 above, then click 'Refresh Predictions'.")
else:
    cols = st.columns(len(predictions))
    for col, pred in zip(cols, predictions):
        arrow    = "▲" if pred["direction"] == "UP" else "▼"
        css_dir  = "pred-up" if pred["direction"] == "UP" else "pred-down"
        chg_sign = "+" if pred["change_pct"] > 0 else ""
        with col:
            st.markdown(f"""
            <div class="pred-card">
              <div class="pred-ticker">{pred['ticker']}</div>
              <div class="pred-meta">Last close: ${pred['last_close']:,.2f}</div>
              <div class="pred-price {css_dir}">{arrow} ${pred['predicted_close']:,.2f}</div>
              <div class="pred-meta">{chg_sign}{pred['change_pct']:.2f}% predicted change</div>
              <br/>
              <div class="pred-meta">MAE: ${pred['test_mae']:.2f} &nbsp;|&nbsp; MAPE: {pred['test_mape']:.1f}%</div>
              <div class="pred-meta">Trained on data through {pred['train_end']}</div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ── Chart section ─────────────────────────────────────────────────────────────

st.markdown("### Charts & Analysis")

available = sorted([p.stem for p in (ROOT / "data" / "processed").glob("*.parquet")]) if (ROOT / "data" / "processed").exists() else []

CHART_BG  = "#11111b"
GRID_CLR  = "#313244"
FONT_CLR  = "#cdd6f4"
LAYOUT    = dict(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                 font=dict(color=FONT_CLR, size=12),
                 legend=dict(bgcolor="#1e1e2e", bordercolor=GRID_CLR, borderwidth=1),
                 margin=dict(l=0, r=0, t=40, b=0))

def apply_grid(fig, nrows=1):
    for i in range(1, nrows + 1):
        fig.update_xaxes(gridcolor=GRID_CLR, row=i, col=1)
        fig.update_yaxes(gridcolor=GRID_CLR, row=i, col=1)

if not available:
    st.info("Run Steps 1 & 2 to generate chart data.")
else:
    # ── Controls ──────────────────────────────────────────────────────────────
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        selected = st.selectbox("Select ticker", available, index=0)
    with cc2:
        window = st.selectbox("Date range", ["3 months", "6 months", "1 year", "All"], index=2)

    df = load_processed(selected)

    if df is not None:
        cutoff = {"3 months": 63, "6 months": 126, "1 year": 252, "All": len(df)}[window]
        df_view = df.tail(cutoff).copy()
        pred_row = next((p for p in predictions if p["ticker"] == selected), None)

        # ── Stats row ─────────────────────────────────────────────────────────
        latest = df_view.iloc[-1]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Last Close", f"${latest['close']:.2f}")
        m2.metric("RSI (14)",   f"{latest['rsi_14']:.1f}",
                  delta="Overbought" if latest["rsi_14"] > 70 else ("Oversold" if latest["rsi_14"] < 30 else "Neutral"),
                  delta_color="inverse")
        m3.metric("MA 7",  f"${latest['ma_7']:.2f}")
        m4.metric("MA 50", f"${latest['ma_50']:.2f}")
        m5.metric("MACD",  f"{latest['macd']:.2f}", delta=f"Signal {latest['macd_signal']:.2f}")

        # ── Tabs ──────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Price & Indicators",
            "📦 Volume",
            "🎯 Backtest: Predicted vs Actual",
            "🔑 Feature Importance",
            "🌐 All Tickers Comparison",
        ])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — Candlestick + MA + Bollinger + RSI + MACD
        # ══════════════════════════════════════════════════════════════════════
        with tab1:
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.55, 0.25, 0.20],
                vertical_spacing=0.03,
                subplot_titles=["Price & Moving Averages", "RSI (14)", "MACD"],
            )

            fig.add_trace(go.Candlestick(
                x=df_view["date"], open=df_view["open"], high=df_view["high"],
                low=df_view["low"], close=df_view["close"], name="OHLC",
                increasing_line_color="#a6e3a1", decreasing_line_color="#f38ba8",
            ), row=1, col=1)

            for col_name, color, dash in [
                ("ma_7", "#89b4fa", "solid"), ("ma_21", "#fab387", "dot"), ("ma_50", "#cba6f7", "dash"),
            ]:
                fig.add_trace(go.Scatter(x=df_view["date"], y=df_view[col_name],
                    name=col_name.upper(), line=dict(color=color, width=1.5, dash=dash)), row=1, col=1)

            fig.add_trace(go.Scatter(x=df_view["date"], y=df_view["bb_upper"], name="BB Upper",
                line=dict(color="#6c7086", width=1, dash="dot"), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_view["date"], y=df_view["bb_lower"], name="BB Lower",
                fill="tonexty", fillcolor="rgba(108,112,134,0.08)",
                line=dict(color="#6c7086", width=1, dash="dot"), showlegend=False), row=1, col=1)

            if pred_row:
                fig.add_trace(go.Scatter(
                    x=[df_view["date"].iloc[-1]], y=[pred_row["predicted_close"]],
                    mode="markers+text",
                    marker=dict(symbol="star", size=18,
                                color="#a6e3a1" if pred_row["direction"] == "UP" else "#f38ba8"),
                    text=[f"  Next: ${pred_row['predicted_close']:.2f}"],
                    textposition="middle right", name="Prediction",
                ), row=1, col=1)

            fig.add_trace(go.Scatter(x=df_view["date"], y=df_view["rsi_14"],
                name="RSI 14", line=dict(color="#89dceb", width=1.5)), row=2, col=1)
            for level, clr in [(70, "#f38ba8"), (30, "#a6e3a1")]:
                fig.add_hline(y=level, line_dash="dot", line_color=clr, line_width=1, row=2, col=1)

            bar_colors = ["#a6e3a1" if v >= 0 else "#f38ba8" for v in df_view["macd_diff"]]
            fig.add_trace(go.Bar(x=df_view["date"], y=df_view["macd_diff"],
                marker_color=bar_colors, showlegend=False, name="Hist"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_view["date"], y=df_view["macd"],
                name="MACD", line=dict(color="#89b4fa", width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_view["date"], y=df_view["macd_signal"],
                name="Signal", line=dict(color="#fab387", width=1.5)), row=3, col=1)

            fig.update_layout(height=700, xaxis_rangeslider_visible=False, **LAYOUT)
            apply_grid(fig, 3)
            st.plotly_chart(fig, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — Volume analysis
        # ══════════════════════════════════════════════════════════════════════
        with tab2:
            fig2 = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.6, 0.4],
                vertical_spacing=0.04,
                subplot_titles=["Close Price", "Volume vs 5-Day Average"],
            )

            fig2.add_trace(go.Scatter(x=df_view["date"], y=df_view["close"],
                name="Close", line=dict(color="#89b4fa", width=2)), row=1, col=1)

            vol_colors = ["#a6e3a1" if c >= 0 else "#f38ba8" for c in df_view["vol_change"]]
            fig2.add_trace(go.Bar(x=df_view["date"], y=df_view["volume"],
                name="Volume", marker_color=vol_colors, opacity=0.7), row=2, col=1)

            vol_ma5 = df_view["volume"].rolling(5).mean()
            fig2.add_trace(go.Scatter(x=df_view["date"], y=vol_ma5,
                name="5-Day Avg Volume", line=dict(color="#fab387", width=1.5, dash="dot")), row=2, col=1)

            fig2.update_layout(height=550, **LAYOUT)
            apply_grid(fig2, 2)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Green bars = volume above 5-day average (strong move). Red = below average (weak move).")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 — Backtest: Predicted vs Actual on test set
        # ══════════════════════════════════════════════════════════════════════
        with tab3:
            meta_path  = ROOT / "models" / "artifacts" / f"{selected}_xgboost_meta.json"
            model_path = ROOT / "models" / "artifacts" / f"{selected}_xgboost.pkl"

            if not model_path.exists():
                st.info(f"No trained model for {selected}. Complete Step 3 first.")
            else:
                import json, joblib
                meta  = json.loads(meta_path.read_text())
                model = joblib.load(model_path)
                feature_cols = meta["features"]

                test_df = df[df["date"].astype(str) >= meta["test_start"]].copy()
                preds   = model.predict(test_df[feature_cols])
                actual  = test_df["target"].values

                errors     = abs(actual - preds)
                avg_error  = errors.mean()

                fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.05,
                    subplot_titles=["Predicted vs Actual Close Price", "Absolute Error per Day"])

                fig3.add_trace(go.Scatter(x=test_df["date"], y=actual,
                    name="Actual", line=dict(color="#a6e3a1", width=2)), row=1, col=1)
                fig3.add_trace(go.Scatter(x=test_df["date"], y=preds,
                    name="Predicted", line=dict(color="#89b4fa", width=2, dash="dot")), row=1, col=1)
                fig3.add_trace(go.Scatter(
                    x=list(test_df["date"]) + list(test_df["date"])[::-1],
                    y=list(preds + errors) + list(preds - errors)[::-1],
                    fill="toself", fillcolor="rgba(137,180,250,0.08)",
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="Error band",
                ), row=1, col=1)

                fig3.add_trace(go.Bar(x=test_df["date"], y=errors,
                    name="Abs Error", marker_color="#cba6f7", opacity=0.7), row=2, col=1)
                fig3.add_hline(y=avg_error, line_dash="dot", line_color="#fab387",
                               line_width=1.5, row=2, col=1,
                               annotation_text=f" Avg ${avg_error:.2f}",
                               annotation_font_color="#fab387")

                fig3.update_layout(height=580, **LAYOUT)
                apply_grid(fig3, 2)
                st.plotly_chart(fig3, use_container_width=True)

                e1, e2, e3 = st.columns(3)
                e1.metric("Test MAE",  f"${meta['metrics']['mae']:.2f}", help="Avg dollar error on test set")
                e2.metric("Test MAPE", f"{meta['metrics']['mape']:.2f}%", help="Avg % error on test set")
                e3.metric("Directional Accuracy", f"{meta['metrics']['directional_accuracy']:.1f}%",
                          help="How often the model predicted UP/DOWN correctly")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4 — Feature Importance
        # ══════════════════════════════════════════════════════════════════════
        with tab4:
            model_path = ROOT / "models" / "artifacts" / f"{selected}_xgboost.pkl"
            meta_path  = ROOT / "models" / "artifacts" / f"{selected}_xgboost_meta.json"

            if not model_path.exists():
                st.info(f"No trained model for {selected}. Complete Step 3 first.")
            else:
                import json, joblib
                model        = joblib.load(model_path)
                meta         = json.loads(meta_path.read_text())
                feature_cols = meta["features"]

                importances = pd.Series(model.feature_importances_, index=feature_cols)
                importances = importances.sort_values(ascending=True)

                bar_clrs = ["#89b4fa" if v < 0.05 else "#a6e3a1" if v < 0.15 else "#fab387"
                            for v in importances.values]

                fig4 = go.Figure(go.Bar(
                    x=importances.values,
                    y=importances.index,
                    orientation="h",
                    marker_color=bar_clrs,
                    text=[f"{v*100:.1f}%" for v in importances.values],
                    textposition="outside",
                ))
                fig4.update_layout(
                    height=500,
                    title=f"Feature Importance — {selected} XGBoost Model",
                    xaxis_title="Importance Score",
                    **LAYOUT,
                )
                fig4.update_xaxes(gridcolor=GRID_CLR)
                fig4.update_yaxes(gridcolor=GRID_CLR)
                st.plotly_chart(fig4, use_container_width=True)
                st.caption("🟠 High importance  🟢 Medium  🔵 Low — the model relies most on the top features to make predictions.")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5 — All Tickers Comparison
        # ══════════════════════════════════════════════════════════════════════
        with tab5:
            all_dfs = {}
            for t in available:
                d = load_processed(t)
                if d is not None:
                    all_dfs[t] = d.tail(cutoff).set_index("date")["close"]

            if all_dfs:
                combined = pd.DataFrame(all_dfs).dropna()

                # ── Normalised performance (rebased to 100) ───────────────────
                normed = combined / combined.iloc[0] * 100

                fig5a = go.Figure()
                colors_map = {"AAPL": "#89b4fa", "MSFT": "#a6e3a1", "GOOGL": "#fab387",
                              "NVDA": "#cba6f7", "SPY": "#f38ba8"}
                for ticker_name, series in normed.items():
                    fig5a.add_trace(go.Scatter(
                        x=series.index, y=series.values,
                        name=ticker_name,
                        line=dict(color=colors_map.get(ticker_name, "#cdd6f4"), width=2),
                    ))
                fig5a.add_hline(y=100, line_dash="dot", line_color="#6c7086", line_width=1)
                fig5a.update_layout(
                    height=380,
                    title="Normalised Performance (100 = start of period)",
                    yaxis_title="Indexed Price",
                    **LAYOUT,
                )
                fig5a.update_xaxes(gridcolor=GRID_CLR)
                fig5a.update_yaxes(gridcolor=GRID_CLR)
                st.plotly_chart(fig5a, use_container_width=True)

                # ── Correlation heatmap ────────────────────────────────────────
                daily_returns = combined.pct_change().dropna()
                corr = daily_returns.corr().round(2)

                fig5b = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale="RdBu",
                    zmid=0,
                    text=corr.values,
                    texttemplate="%{text}",
                    hovertemplate="%{x} vs %{y}: %{z}<extra></extra>",
                    showscale=True,
                ))
                fig5b.update_layout(
                    height=380,
                    title="Daily Returns Correlation — How stocks move together",
                    **LAYOUT,
                )
                st.plotly_chart(fig5b, use_container_width=True)
                st.caption("Values near +1 = stocks move together. Near 0 = independent. Near -1 = move opposite.")

st.divider()
st.markdown("<div style='text-align:center; color:#6c7086; font-size:12px;'>Stock Market Prediction Pipeline · yfinance · XGBoost · FastAPI · Streamlit</div>", unsafe_allow_html=True)
