# Rainfall Prediction using LSTM + DeepSpeed

This script trains an LSTM model for rainfall prediction using the DeepSpeed optimization library to improve training efficiency.

## Features
- LSTM model for time series forecasting
- DeepSpeed integration for training speed-up
- Tracks MSE, MAE, and R² metrics
- Predicts rainfall for the next 7 days

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements_deepspeed.txt
   ```

2. Place your dataset as `sample.xlsx` with columns: Date, Rainfall.

3. Run the script:
   ```bash
   python deepspeed.py
   ```

## Input Format
Excel file with two columns:
- Column 1: Date/Day
- Column 2: Rainfall values

## Output
- Printed losses and metrics per epoch
- Predicted rainfall values for 7 days

---