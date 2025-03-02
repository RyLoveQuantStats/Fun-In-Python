"""
PROJECT: S&P 500 Options Volatility Smile Analysis

This script demonstrates how to:
1. Load and parse raw CBOE option price data from a CSV.
2. Clean the data by removing rows with no meaningful bid/ask information.
3. Filter options by specific maturities (in this case, March 11, 2025 and February 20, 2026).
4. Document market parameters such as the underlying price, risk-free rate, and dividend yield.
5. Generate a final spreadsheet of the filtered data.
6. Plot the volatility smile for the chosen maturities.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import yfinance as yf

def debug_info(df, name="DataFrame"):
    print(f"\n--- Debug Info: {name} ---")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())
    print("Head:\n", df.head())

# 1. Load Data with try/except from CSV 
file_path = r"C:\\Users\\ryanl\\OneDrive\\Desktop\\MS Finance Courses\\Applied Derivatives\\Implied Volatility\\SPX_Data.csv"
try:
    df = pd.read_csv(file_path, delimiter=";", skiprows=2, header=None)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    sys.exit(1)
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    sys.exit(1)

# 2. Split columns into two
df = df[0].str.split(",", expand=True)

# 3. Rename columns for ease of access later
df.columns = [
    "Expiration", "Calls", "Last Sale Call", "Net Call", "Bid Call", "Ask Call", "Volume Call", 
    "IV Call", "Delta Call", "Gamma Call", "Open Interest Call", "Strike", 
    "Puts", "Last Sale Put", "Net Put", "Bid Put", "Ask Put", "Volume Put",
    "IV Put", "Delta Put", "Gamma Put", "Open Interest Put"
]

numeric_cols = [
    "Last Sale Call", "Net Call", "Bid Call", "Ask Call", "Volume Call", 
    "IV Call", "Delta Call", "Gamma Call", "Open Interest Call", "Strike", 
    "Last Sale Put", "Net Put", "Bid Put", "Ask Put", "Volume Put",
    "IV Put", "Delta Put", "Gamma Put", "Open Interest Put"
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

debug_info(df, "Raw Data (after splitting columns)")

# 4. Remove rows with no info
initial_rows = df.shape[0]
df = df.dropna(subset=["Bid Call", "Ask Call", "Bid Put", "Ask Put"])
df = df[(df["Bid Call"] > 0) & (df["Ask Call"] > df["Bid Call"]) &
        (df["Bid Put"] > 0)  & (df["Ask Put"]  > df["Bid Put"])]
print(f"\nRemoved {initial_rows - df.shape[0]} rows with invalid bids/asks.")

# Remove extreme IV outliers with extreme low or high volatility
df = df[(df["IV Call"] > 0.01) & (df["IV Call"] < 2)]
df = df[(df["IV Put"]  > 0.01) & (df["IV Put"]  < 2)]

# 5. Filter for desired maturities
df["Expiration"] = df["Expiration"].astype(str)
required_maturities = ["Tue Mar 11 2025", "Fri Feb 20 2026"]
df_filtered = df[df["Expiration"].isin(required_maturities)]

if df_filtered.empty:
    print("Warning: No data found for the specified maturities!")
else:
    debug_info(df_filtered, "Filtered Data")

# 6. Market Data
underlying_price = 6108.75 # Most recent price given 
one_year_treasury = 0.0415 # Most recent 1 year treasury rate (2/25/2025)
two_year_treasury = 0.0419 # Most recent 2 year treasury rate (2/25/2025)
dividend_yield = 0.0132 # Most recent SPY dividend yield (02/25/2025) 

# 7. Save final data
df_filtered.to_excel("filtered_spx_options.xlsx", index=False)
print("Data saved as filtered_spx_options.csv")

# 8. Plot the Volatility Smile
plt.figure(figsize=(10, 6))

# Define custom markers for calls and puts
markers = {"Call": "o", "Put": "s"}
colors = {"Call": "blue", "Put": "red"}

for maturity in required_maturities:
    subset = df_filtered[df_filtered["Expiration"] == maturity]

    # Plot Call IV
    plt.scatter(
        subset["Strike"], 
        subset["IV Call"], 
        label=f"Call IV - {maturity}", 
        alpha=0.6, 
        marker=markers["Call"], 
        color=colors["Call"]
    )

    # Plot Put IV
    plt.scatter(
        subset["Strike"], 
        subset["IV Put"], 
        label=f"Put IV - {maturity}", 
        alpha=0.6, 
        marker=markers["Put"], 
        color=colors["Put"]
    )

plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.title("Volatility Smile for Selected Maturities")
plt.legend()
plt.grid(True)  # Add grid for clarity
plt.show()

