# Rainfall Prediction using LSTM + PyTorch Ignite

This script leverages PyTorch Ignite to train an LSTM model for time series rainfall forecasting.

## Features
- Ignite training and evaluation loops
- Custom MAE metric included
- Tracks MSE, MAE, and R² scores
- Predicts rainfall for the next 7 days

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements_ignite.txt
   ```

2. Prepare your Excel file (`sample.xlsx`) with columns: Date, Rainfall.

3. Run the script:
   ```bash
   python pytorch_ignite.py
   ```

## Input Format
Excel file with two columns:
- Column 1: Date/Day
- Column 2: Rainfall values

## Output
- Console logs for training, validation, testing
- 7-day rainfall forecast

---