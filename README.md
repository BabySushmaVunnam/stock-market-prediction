# Stock Market Prediction Pipeline

A production-grade **data engineering + ML pipeline** that ingests raw stock market data, transforms it into predictive features, trains a forecasting model, and serves predictions via a REST API.

> The goal is not to beat the market — it's to demonstrate end-to-end data engineering: ingestion, transformation, orchestration, modeling, and serving.

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Data Sources  │────▶│   Ingestion Layer │────▶│   Raw Storage       │
│  (yfinance API) │     │  (Python scripts) │     │  (Parquet / S3)     │
└─────────────────┘     └──────────────────┘     └────────┬────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Prediction API │◀────│   ML Model        │◀────│  Feature Store      │
│  (FastAPI)      │     │  (XGBoost/LSTM)   │     │  (Processed Parquet)│
└─────────────────┘     └──────────────────┘     └────────▲────────────┘
                                                           │
                                              ┌────────────┴────────────┐
                                              │  Transformation Layer    │
                                              │  (Feature Engineering)   │
                                              └────────────▲────────────┘
                                                           │
                                              ┌────────────┴────────────┐
                                              │  Orchestration (Airflow) │
                                              │  Daily scheduled DAG     │
                                              └─────────────────────────┘
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data Ingestion | `yfinance`, Python |
| Storage | Parquet (local) / AWS S3 |
| Transformation | Pandas, dbt (optional) |
| Orchestration | Apache Airflow |
| Modeling | XGBoost, LSTM (TensorFlow/Keras) |
| Serving | FastAPI |
| Containerization | Docker, Docker Compose |
| Testing | Pytest, Great Expectations |
| CI/CD | GitHub Actions |

---

## Project Structure

```
stock-market-prediction/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env.example
│
├── data/
│   ├── raw/                  # Raw OHLCV data per ticker
│   └── processed/            # Feature-engineered Parquet files
│
├── ingestion/
│   ├── fetch_stocks.py       # Pull OHLCV data from yfinance
│   ├── tickers.txt           # List of stock tickers to track
│   └── schema.py             # Data schema / type definitions
│
├── pipeline/
│   ├── clean.py              # Null handling, outlier detection
│   ├── features.py           # Feature engineering (MA, RSI, MACD)
│   └── validate.py           # Data quality checks
│
├── models/
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Backtesting + metrics
│   └── artifacts/            # Saved model files (.pkl, .h5)
│
├── orchestration/
│   └── dags/
│       └── stock_pipeline.py # Airflow DAG — daily pipeline
│
├── serving/
│   ├── main.py               # FastAPI app
│   └── predict.py            # Prediction logic
│
└── tests/
    ├── test_ingestion.py
    ├── test_features.py
    └── test_api.py
```

---

## Pipeline Phases

### Phase 1 — Data Ingestion
- Fetch daily OHLCV (Open, High, Low, Close, Volume) data for a list of tickers
- Source: `yfinance` (free, no API key required)
- Store as Parquet partitioned by `ticker/year/month`
- Tickers to start: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, SPY

### Phase 2 — Data Transformation & Feature Engineering
Technical indicators computed per ticker:

| Feature | Description |
|---------|-------------|
| `ma_7`, `ma_21`, `ma_50` | Simple moving averages (7, 21, 50 day) |
| `ema_12`, `ema_26` | Exponential moving averages |
| `macd`, `macd_signal` | MACD line and signal line |
| `rsi_14` | Relative Strength Index (14-day) |
| `bb_upper`, `bb_lower` | Bollinger Bands |
| `vol_change` | Volume change % vs 5-day avg |
| `lag_1`, `lag_5`, `lag_10` | Lagged close prices |
| `target` | Next-day close price (prediction target) |

### Phase 3 — Orchestration
- Airflow DAG runs daily at market close (5PM ET)
- Tasks: `ingest → clean → feature_engineer → validate → retrain (weekly)`
- Failed tasks trigger alerts

### Phase 4 — Modeling
Two model approaches:

**XGBoost (baseline)**
- Fast to train, interpretable feature importance
- Input: engineered features → Output: next-day close price

**LSTM (deep learning)**
- Captures sequential patterns in time-series
- Input: 30-day rolling window → Output: next-day close price

Evaluation metrics: MAE, RMSE, MAPE, directional accuracy

### Phase 5 — Serving
FastAPI REST endpoint:
```
GET /predict?ticker=AAPL&model=xgboost
→ { "ticker": "AAPL", "predicted_close": 189.45, "confidence": 0.72, "date": "2026-05-01" }
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Docker + Docker Compose
- Git

### Setup

```bash
# Clone the repo
git clone https://github.com/BabySushmaVunnam/stock-market-prediction.git
cd stock-market-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env file
cp .env.example .env
```

### Run ingestion manually
```bash
python ingestion/fetch_stocks.py --tickers AAPL MSFT GOOGL --days 365
```

### Run feature engineering
```bash
python pipeline/features.py --input data/raw --output data/processed
```

### Train model
```bash
python models/train.py --model xgboost --ticker AAPL
```

### Start prediction API
```bash
uvicorn serving.main:app --reload
# API docs: http://localhost:8000/docs
```

### Run with Docker
```bash
docker-compose up --build
```

---

## Roadmap

- [x] Project structure and README
- [x] Phase 1: Data ingestion script (yfinance → partitioned Parquet)
- [x] Phase 2: Feature engineering pipeline (25 indicators: MA, EMA, MACD, RSI, Bollinger Bands)
- [x] Phase 2: Data quality validation (6 checks per ticker)
- [ ] Phase 3: Airflow DAG setup
- [x] Phase 4: XGBoost baseline model (MAE ~$6–21 across tickers)
- [ ] Phase 4: LSTM model
- [x] Phase 5: FastAPI serving layer (`/predict`, `/predict/all`, `/health`)
- [ ] Docker + Docker Compose setup
- [x] GitHub Actions CI/CD
- [ ] Data quality tests with Great Expectations
- [ ] Streamlit dashboard (optional)

---

## Data Engineering Concepts Demonstrated

- **Batch ingestion** with incremental loading (only fetch new dates)
- **Schema enforcement** and data type validation at ingestion
- **Partitioned storage** strategy for time-series data
- **Idempotent pipeline** — safe to re-run without duplicating data
- **Orchestration** with dependency management and failure alerting
- **Feature store pattern** — separation of raw vs processed data
- **Model versioning** — artifacts tracked with metadata
- **API serving** of ML predictions with request validation

---

## Author

**Sushma Vunnam** — Data Engineer  
[Portfolio](https://babysushmavunnam.github.io) | [GitHub](https://github.com/BabySushmaVunnam) | [LinkedIn](https://linkedin.com/in/sushma-vunnam)
in on