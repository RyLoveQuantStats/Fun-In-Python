### Payout Diagram + PnL ###
### Author: Ryan Loveless ###
### Date: 2022-02-22 ###

# For computing a payout diagram, pnl and other metrics for options trading. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, skew

# Input Data
spot = 100.0  # Spot price
volatility = 0.2  # Volatility
risk_free_rate = 0.05  # Risk-free rate
expected_drift = 0.05  # Expected drift
maturity = 0.02  # Maturity (in years)
cash_balance = 0.0

# Portfolio Data
def portfolio_payout(asset_price, portfolio):
    """Calculate portfolio payout based on asset price."""
    payout = 0.0
    for position in portfolio:
        type_, number, strike, value = position
        if type_ == 'call':
            payout += number * max(asset_price - strike, 0)
        elif type_ == 'put':
            payout += number * max(strike - asset_price, 0)
        elif type_ == 'forward':
            payout += number * (asset_price - strike)
    return payout

# Example Portfolio
portfolio = [
    ('call', 0.0, 95.4815, 4.66),
    ('forward', 100.0, 100.0, 0.0),
]

# Simulation Data
num_simulations = 100000
low_payout_level = 90
high_payout_level = 110

# Simulate Asset Prices at Maturity
np.random.seed(42)  # For reproducibility
z = np.random.normal(0, 1, num_simulations)
asset_prices = spot * np.exp((risk_free_rate - 0.5 * volatility ** 2) * maturity + volatility * np.sqrt(maturity) * z)

# Calculate Portfolio Values and PnL
initial_value = sum(position[3] for position in portfolio) + cash_balance
portfolio_values = np.array([portfolio_payout(price, portfolio) + cash_balance for price in asset_prices])
pnl = portfolio_values - initial_value

# Statistics
mean_pnl = np.mean(pnl)
std_pnl = np.std(pnl)
skew_pnl = skew(pnl)
kurt_pnl = kurtosis(pnl)

# Visualization
plt.figure(figsize=(10, 6))
plt.hist(pnl, bins=100, range=(-1000, 1000), color='blue', alpha=0.7, label='PnL Distribution')
plt.axvline(mean_pnl, color='red', linestyle='--', label=f'Mean PnL: {mean_pnl:.2f}')
plt.title('Portfolio PnL Histogram')
plt.xlabel('PnL')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()

# Payout Diagram
asset_levels = np.linspace(low_payout_level, high_payout_level, 100)
payouts = [portfolio_payout(price, portfolio) for price in asset_levels]

plt.figure(figsize=(10, 6))
plt.plot(asset_levels, payouts, label='Payout Diagram', color='green')
plt.title('Payout Diagram')
plt.xlabel('Asset Level')
plt.ylabel('Payout')
plt.grid()
plt.legend()
plt.show()

# Output Results
print("Portfolio Statistics:")
print(f"Initial Portfolio Value: {initial_value:.2f}")
print(f"Mean PnL: {mean_pnl:.2f}")
print(f"Standard Deviation: {std_pnl:.2f}")
print(f"Skew: {skew_pnl:.4f}")
print(f"Kurtosis: {kurt_pnl:.4f}")
