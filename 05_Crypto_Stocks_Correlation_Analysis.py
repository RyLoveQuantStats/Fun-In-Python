# ======================================================================
# Advanced Correlation Analysis of Bitcoin vs. Crypto-related Stocks
# Using the OpenBB platform
# ======================================================================

"""
This script fetches data for:
- Multiple equity tickers (e.g., COIN, MSTR, etc.) using obb.equity.price.historical()
- BTC-USD using obb.crypto.price.historical()

Then it merges them, calculates returns, and performs:
- Correlation calculations (Pearson, Kendall, Spearman)
- Rolling correlation (BTC-USD vs COIN)
- Hierarchical clustering
- PCA (2D)
- Granger causality test (COIN -> BTC-USD)
- Random Forest regression to predict BTC-USD from other tickers
- Visualizations (correlation heatmap, dendrogram, rolling correlation, PCA)
"""

from openbb import obb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ======================================================================
# 1) Fetch historical data for multiple assets
# ======================================================================

# Stock tickers that are often considered crypto-exposed
stock_assets = [
    'COIN',     # Coinbase (crypto exchange)
    'MSTR',     # MicroStrategy (major Bitcoin treasury)
    'RIOT',     # Riot Platforms (Bitcoin mining)
    'MARA',     # Marathon Digital (Bitcoin mining)
    'NVDA',     # NVIDIA (GPUs used for mining/AI)
    'TSLA',     # Tesla (held Bitcoin in the past)
    'HUT',      # Hut 8 Mining (Bitcoin mining)
    'HIVE',     # Hive Blockchain (Bitcoin/Ethereum mining)
    'ARBK',     # Argo Blockchain (Bitcoin mining)
    'CAN',      # Canaan (Bitcoin mining hardware)
    'SOS',      # SOS Limited (crypto mining/blockchain services)
    'BKKT',     # Bakkt (digital asset platform and custodian)
    'HOOD',     # Robinhood (retail trading, including crypto)
    'SQ',       # Block (formerly Square, Bitcoin-friendly)
    'PYPL',     # PayPal (crypto buying/selling platform)
    'MA',       # Mastercard (crypto integration/payments)
    'AMD',      # AMD (CPUs/GPUs for mining/AI)
    'IBM',      # IBM (enterprise blockchain solutions)
    'BTCS',     # BTCS Inc. (blockchain infrastructure, staking)
]

# BTC-USD will be fetched from the crypto module
crypto_assets = [
    'BTC-USD','ETH-USD','SOL-USD'
]

start_date = '2022-01-01'
end_date = '2024-01-01'

all_data = []

# Fetch equity data
for asset in stock_assets:
    try:
        # The historical() method fetches a custom OBBject. We'll convert to DataFrame.
        hist_data = obb.equity.price.historical(
            symbol=asset,
            start_date=start_date,
            end_date=end_date
        )
        # Convert the OBBject to a pandas DataFrame
        hist_df = hist_data.to_df()
        # Check if 'Close' is available
        if 'close' not in hist_df.columns:
            print(f"Warning: 'Close' not found for {asset}. Columns: {list(hist_df.columns)}")
            continue
        # Keep only the Close column, rename it
        hist_df = hist_df[['close']].copy()
        hist_df.rename(columns={'close': asset}, inplace=True)
        # Append to the list for merging
        all_data.append(hist_df)
    except Exception as e:
        print(f"Error retrieving data for {asset}: {e}")

# Fetch crypto data (for BTC-USD)
for asset in crypto_assets:
    try:
        hist_data = obb.crypto.price.historical(
            symbol=asset,
            start_date=start_date,
            end_date=end_date
        )
        # Convert the OBBject to a pandas DataFrame
        hist_df = hist_data.to_df()
        # Check if 'Close' is available
        if 'close' not in hist_df.columns:
            print(f"Warning: 'close' not found for {asset}. Columns: {list(hist_df.columns)}")
            continue
        # Keep only the Close column, rename it
        hist_df = hist_df[['close']].copy()
        hist_df.rename(columns={'close': asset}, inplace=True)
        # Append to the list
        all_data.append(hist_df)
    except Exception as e:
        print(f"Error retrieving data for {asset}: {e}")

# Combine into a single DataFrame
if not all_data:
    raise ValueError("No valid data retrieved. Please check your tickers and OpenBB setup.")

# Merge on index, using an outer join to keep all dates
data_combined = pd.concat(all_data, axis=1, join='outer')

# Sort by date just in case
data_combined.sort_index(inplace=True)

# Drop rows that are completely empty
data_combined.dropna(how='all', inplace=True)

# ======================================================================
# 2) Calculate daily returns
# ======================================================================
returns = data_combined.pct_change().dropna()

# ======================================================================
# 3) Compute different correlation measures
# ======================================================================
corr_pearson = returns.corr()
corr_kendall = returns.corr(method='kendall')
corr_spearman = returns.corr(method='spearman')

print("\nPearson Correlation Matrix:\n", corr_pearson)
print("\nKendall Correlation Matrix:\n", corr_kendall)
print("\nSpearman Correlation Matrix:\n", corr_spearman)

# ======================================================================
# 4) Rolling correlation example
# ======================================================================

window_size = 30
if 'BTC-USD' in returns.columns and 'COIN' in returns.columns:
    rolling_corr = returns['BTC-USD'].rolling(window=window_size).corr(returns['COIN'])
else:
    rolling_corr = None

# ======================================================================
# 5) Perform hierarchical clustering on Pearson correlations
# ======================================================================
from scipy.cluster.hierarchy import linkage, dendrogram
link_mat = linkage(corr_pearson, method='ward')

# ======================================================================
# 6) PCA for dimensionality reduction
# ======================================================================
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(returns.dropna(axis=1))

# ======================================================================
# 7) Granger Causality Test (example with BTC-USD and COIN)
# ======================================================================
from statsmodels.tsa.stattools import grangercausalitytests
gc_test = None
if 'BTC-USD' in returns.columns and 'COIN' in returns.columns:
    pairs_df = returns[["COIN", "BTC-USD"]].dropna()
    gc_test = grangercausalitytests(pairs_df, maxlag=5, verbose=False)

# ======================================================================
# 8) Random Forest Regression to predict BTC-USD from others
# ======================================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if 'BTC-USD' in returns.columns:
    X = returns.drop('BTC-USD', axis=1).copy()
    y = returns['BTC-USD'].copy()

    # Handle any potential NA values by dropping them or filling them
    df_model = pd.concat([X, y], axis=1).dropna()
    X = df_model.drop('BTC-USD', axis=1)
    y = df_model['BTC-USD']

    if not X.empty:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("\nRandom Forest RMSE (Predicting BTC-USD):", rmse)
    else:
        print("\nNot enough data for Random Forest Model.")
else:
    print("\nBTC-USD not in columns, skipping Random Forest Model.")

# ======================================================================
# 9) Visualizations
# ======================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# (A) Pearson Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Matrix (Crypto & Related Stocks)')
plt.tight_layout()
plt.show()

# (B) Hierarchical Clustering Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(link_mat, labels=corr_pearson.columns, leaf_rotation=90)
plt.title('Hierarchical Clustering')
plt.tight_layout()
plt.show()

# (C) Rolling Correlation Chart
if rolling_corr is not None:
    plt.figure(figsize=(10, 6))
    rolling_corr.plot()
    plt.title(f'Rolling {window_size}-Day Correlation: BTC-USD vs. COIN')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.tight_layout()
    plt.show()
else:
    print("\nSkipping rolling correlation plot because data is missing for BTC-USD or COIN.")

# (D) Optional PCA Plot (2D)
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA (2 Components) of Returns')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

# ======================================================================
# 10) Print optional Granger Causality results
# ======================================================================
if gc_test:
    print("\nGranger Causality Test Results (COIN -> BTC-USD, up to 5 lags):")
    for lag, results in gc_test.items():
        test_stat = results[0]['ssr_ftest'][0]
        p_value = results[0]['ssr_ftest'][1]
        print(f"Lag {lag}: F-stat={test_stat:.4f}, p-value={p_value:.4g}")
