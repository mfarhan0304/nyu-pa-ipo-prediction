import yfinance as yf
import pandas as pd

# Define the ticker symbol
vix = yf.Ticker("^VIX")

# Fetch historical data from 2000-01-01 to 2025-06-01
vix_data = vix.history(start="2000-01-01", end="2025-06-01")

# Display the first few rows
print(vix_data.head())

# Optional: Save to CSV for later use
vix_data.to_csv("vix_2000_to_2025.csv")