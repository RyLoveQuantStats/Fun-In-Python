# ======================================================================
# Advanced Crypto Analysis: Integrating Financial Metrics & Machine Learning
# ======================================================================
"""
This script:
- Fetches historical data for the top 30 coins by market cap using the CoinGecko API,
  then filters out stable coins and duplicate tokens.
- Merges the data and computes daily returns.
"""

from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from datetime import datetime, timedelta
import time

from constants import cg_demo_api_key
# ----------------------------------------------------------------------
# 1) Get the Top 30 Coins by Market Cap and Filter Unwanted Tokens
# ----------------------------------------------------------------------
cg = CoinGeckoAPI(cg_demo_api_key)

# Retrieve top 30 coins by market cap
top_coins = cg.get_coins_markets(
    vs_currency='usd',
    order='market_cap_desc',
    per_page=30,
    page=1,
    sparkline=False
)

# Define exclusion sets for stable coins and duplicate tokens
stable_coin_symbols = {"USDT", "USDC", "BUSD", "DAI", "USDS", "TUSD", "PAX", "GUSD", "USDE"}
duplicate_symbols   = {"WBTC", "STETH", "WSTETH", "WETH"}  
exclusion_list = stable_coin_symbols.union(duplicate_symbols)

# Build a mapping from coin id to ticker for coins not in the exclusion list.
# (Optionally, you could also filter based on trading volume if desired.)
coin_mapping = {}
for coin in top_coins:
    symbol = coin["symbol"].upper()
    if symbol in exclusion_list:
        continue
    coin_id = coin["id"]
    ticker = f"{symbol}-USD"
    coin_mapping[coin_id] = ticker

print("Filtered Coins (after excluding stable coins and duplicates):")
for coin_id, ticker in coin_mapping.items():
    print(f"{coin_id} -> {ticker}")

# ----------------------------------------------------------------------
# 2) Define a Function to Fetch Historical Data in Segments
# ----------------------------------------------------------------------
def fetch_coin_data_segmented(coin_id, start_date, end_date, vs_currency="usd", segment_days=365):
    """
    Fetch historical data for a coin between start_date and end_date by splitting the query
    into segments no longer than segment_days. Returns a DataFrame with a datetime index.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    all_segments = []
    current_start = start_dt
    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=segment_days), end_dt)
        start_ts = int(current_start.timestamp())
        end_ts = int(current_end.timestamp())
        try:
            data = cg.get_coin_market_chart_range_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                from_timestamp=start_ts,
                to_timestamp=end_ts
            )
        except Exception as e:
            print(f"Error fetching data for {coin_id} from {current_start.date()} to {current_end.date()}: {e}")
            current_start = current_end
            continue

        prices = data.get("prices", [])
        if prices:
            df_segment = pd.DataFrame(prices, columns=["timestamp", "price"])
            df_segment["timestamp"] = pd.to_datetime(df_segment["timestamp"], unit='ms')
            df_segment.set_index("timestamp", inplace=True)
            all_segments.append(df_segment)
        else:
            print(f"No price data for {coin_id} from {current_start.date()} to {current_end.date()}")
        time.sleep(1)  # Respect API rate limits
        current_start = current_end

    if all_segments:
        df_full = pd.concat(all_segments)
        df_full = df_full[~df_full.index.duplicated(keep='first')]
        df_full.sort_index(inplace=True)
        return df_full
    else:
        return None

# ----------------------------------------------------------------------
# 3) Fetch Historical Data for Each Filtered Coin and Merge into a Single DataFrame
# ----------------------------------------------------------------------
end_date = datetime.today()
start_date = end_date - timedelta(days=365)
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")
print(f"\nFetching data from {start_date_str} to {end_date_str}")

all_data = []
for coin_id, ticker in coin_mapping.items():
    print(f"Fetching data for {ticker} ({coin_id})...")
    df = fetch_coin_data_segmented(coin_id, start_date_str, end_date_str)
    if df is not None and not df.empty:
        # Keep only the price column and rename it to the ticker symbol.
        df = df[['price']].copy()
        df.rename(columns={"price": ticker}, inplace=True)
        all_data.append(df)
    else:
        print(f"Warning: No data for {ticker}")

if not all_data:
    raise ValueError("No valid data retrieved. Please check your CoinGecko API setup.")

crypto_data = pd.concat(all_data, axis=1, join="outer").sort_index().dropna(how="all")
print("\nMerged crypto data (first few rows):")
print(crypto_data.head())

# ----------------------------------------------------------------------
# 4) Calculate Daily Returns
# ----------------------------------------------------------------------
returns = crypto_data.pct_change().dropna()
print("\nDaily returns (first few rows):")
print(returns.head())


# ======================================================================
# Advanced Financial Analysis: Correlations, Rolling Beta & OLS Regression
# ======================================================================

# A) Compute Correlation Matrices (Pearson, Kendall, Spearman)
corr_pearson = returns.corr()
corr_kendall = returns.corr(method='kendall')
corr_spearman = returns.corr(method='spearman')

print("\nPearson Correlation Matrix:\n", corr_pearson)
print("\nKendall Correlation Matrix:\n", corr_kendall)
print("\nSpearman Correlation Matrix:\n", corr_spearman)

# B) Compute Rolling Beta for BTC-USD and ETH-USD (60-day window)
def rolling_beta(target, market, window=60):
    rolling_cov = returns[target].rolling(window).cov(returns[market])
    rolling_var = returns[market].rolling(window).var()
    return rolling_cov / rolling_var

if 'BTC-USD' in returns.columns and 'ETH-USD' in returns.columns:
    rolling_beta_btc_eth = rolling_beta('BTC-USD', 'ETH-USD')
    rolling_beta_eth_btc = rolling_beta('ETH-USD', 'BTC-USD')
    
    plt.figure(figsize=(10,5))
    rolling_beta_btc_eth.plot(label="BTC Beta vs ETH")
    rolling_beta_eth_btc.plot(label="ETH Beta vs BTC")
    plt.legend()
    plt.title("Rolling Beta of BTC and ETH (60-day window)")
    plt.xlabel("Date")
    plt.ylabel("Beta")
    plt.tight_layout()
    plt.show()
else:
    print("BTC-USD or ETH-USD data missing; skipping rolling beta calculation.")

# C) OLS Regression to Estimate BTC-USD and ETH-USD Returns
if 'BTC-USD' in returns.columns:
    X_btc = returns.drop(columns=['BTC-USD'])
    y_btc = returns['BTC-USD']
    X_btc = sm.add_constant(X_btc)
    btc_model = sm.OLS(y_btc, X_btc, missing='drop').fit()
    print("\nOLS Regression for BTC-USD Returns:\n", btc_model.summary())
else:
    print("BTC-USD data missing; skipping OLS regression for BTC.")

if 'ETH-USD' in returns.columns:
    X_eth = returns.drop(columns=['ETH-USD'])
    y_eth = returns['ETH-USD']
    X_eth = sm.add_constant(X_eth)
    eth_model = sm.OLS(y_eth, X_eth, missing='drop').fit()
    print("\nOLS Regression for ETH-USD Returns:\n", eth_model.summary())
else:
    print("ETH-USD data missing; skipping OLS regression for ETH.")

# ======================================================================
# Improved Machine Learning: Optimized Random Forest Regression
# ======================================================================

# Function to add lagged features for a given asset (e.g., lag1)
def add_lagged_features(df, asset, lags=1):
    df_lagged = df.copy()
    for lag in range(1, lags+1):
        df_lagged[f"{asset}_lag{lag}"] = df_lagged[asset].shift(lag)
    return df_lagged

# Add lagged returns for BTC-USD and ETH-USD if present
if 'BTC-USD' in returns.columns:
    returns = add_lagged_features(returns, 'BTC-USD', lags=1)
if 'ETH-USD' in returns.columns:
    returns = add_lagged_features(returns, 'ETH-USD', lags=1)

# Function for optimized Random Forest regression
def run_rf_regression(target):
    if target not in returns.columns:
        print(f"{target} not in returns data, skipping regression.")
        return
    
    # Select top 4 predictors based on absolute Pearson correlation with the target (excluding the target itself)
    correlations = returns.corr()[target].abs().drop(target)
    top_assets = correlations.sort_values(ascending=False).head(4).index.tolist()
    print(f"\nUsing top correlated assets {top_assets} to predict {target} returns.")
    
    # Prepare predictors; if the target is BTC or ETH, include its lagged return
    predictors = top_assets.copy()
    if target in ['BTC-USD', 'ETH-USD']:
        lag_feature = f"{target}_lag1"
        if lag_feature in returns.columns:
            predictors.append(lag_feature)
    
    X = returns[predictors]
    y = returns[target]
    
    df_model = pd.concat([X, y], axis=1).dropna()
    X = df_model[predictors]
    y = df_model[target]
    
    if X.empty:
        print(f"Not enough data for Random Forest Model to predict {target}.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Random Forest RMSE for predicting {target}: {rmse:.4f}")

if 'BTC-USD' in returns.columns:
    run_rf_regression('BTC-USD')
if 'ETH-USD' in returns.columns:
    run_rf_regression('ETH-USD')

# ======================================================================
# Additional Analyses: Hierarchical Clustering & PCA
# ======================================================================

# Hierarchical Clustering on Pearson Correlations
link_mat = linkage(corr_pearson, method='ward')

# PCA for dimensionality reduction (using all available return features)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(returns.dropna(axis=1))

# ======================================================================
# Visualizations
# ======================================================================

# (A) Pearson Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Matrix (Crypto Assets)')
plt.tight_layout()
plt.show()

# (B) Hierarchical Clustering Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(link_mat, labels=corr_pearson.columns, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.tight_layout()
plt.show()

# (C) Rolling Correlation between BTC-USD and ETH-USD over a 30-day window
window_size = 30
if 'BTC-USD' in returns.columns and 'ETH-USD' in returns.columns:
    rolling_corr = returns['BTC-USD'].rolling(window=window_size).corr(returns['ETH-USD'])
    plt.figure(figsize=(10, 6))
    rolling_corr.plot()
    plt.title(f'Rolling {window_size}-Day Correlation: BTC-USD vs ETH-USD')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.tight_layout()
    plt.show()
else:
    print("BTC-USD or ETH-USD data missing; skipping rolling correlation plot.")

# (D) PCA Scatter Plot (2 Components)
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA (2 Components) of Returns')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()
