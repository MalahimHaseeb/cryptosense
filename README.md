# CryptoSense

BTC/USD next-hour price prediction using a lightweight LSTM neural network trained on 13 years of historical data.

## Overview

- 7.6M+ minute-level BTC/USD data resampled to hourly candles (126k rows)
- 2-layer LSTM model with ~29k parameters
- Predicts next hour close price from last 60 hours
- Served via FastAPI with Swagger UI
- MAE: ~$4,908 | RMSE: ~$7,192

## Tech Stack

- TensorFlow / Keras
- FastAPI
- NumPy, Pandas, Scikit-learn
- Google Colab (T4 GPU)
- Matplotlib / Seaborn

## Dataset

Download from Kaggle: [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

1. Download `btcusd_1-min_data.csv`
2. Upload to your Google Drive at `MyDrive/cryptosense/btcusd_1-min_data.csv`
3. Open the notebook in Colab and run all cells

## Model Architecture

LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)

## API Usage

```bash
POST /predict
{
  "prices": [60000, 60100, ...]  # exactly 60 close prices
}
```

Response:
```json
{
  "predicted_next_hour_price": 61635.61
}
```

## Run Locally

```bash
pip install tensorflow fastapi uvicorn scikit-learn pandas numpy
uvicorn app.main:app --reload
```

## Author

**Malahim Haseeb** — [malahim.dev](https://malahim.dev) · [GitHub](https://github.com/MalahimHaseeb) · [LinkedIn](https://linkedin.com/in/malahimhaseeb)