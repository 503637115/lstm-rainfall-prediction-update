# Rainfall Prediction using LSTM + PyTorch Lightning

This script uses PyTorch Lightning to build and train an LSTM model for rainfall prediction.

## Features
- Structured training with PyTorch Lightning
- Logs MSE, MAE, and R² scores
- Uses Lightning callbacks for test evaluation
- Predicts rainfall for the next 7 days

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements_lightning.txt
   ```

2. Prepare `sample.xlsx` with columns: Date, Rainfall.

3. Run the script:
   ```bash
   python pytorch_lightning.py
   ```

## Input Format
Excel file with two columns:
- Column 1: Date/Day
- Column 2: Rainfall values

## Output
- Epoch-wise loss metrics
- Final predictions printed to console

---