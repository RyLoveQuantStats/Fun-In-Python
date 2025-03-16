import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Helper Functions
# ----------------------------
def monthly_payment_amortization(principal, annual_interest_rate, n_months=60):
    """
    Calculates the fixed monthly payment for an amortizing loan.
    
    Formula:
      Payment = principal * (r / (1 - (1 + r)^(-n)))
    where r is the monthly interest rate.
    """
    r = annual_interest_rate / 12.0
    payment = principal * (r / (1 - (1 + r) ** (-n_months)))
    return payment

def generate_linear_path(start_price, end_price, n_months=60):
    """
    Generates a linear price path (numpy array) from start_price to end_price over n_months.
    """
    return np.linspace(start_price, end_price, n_months + 1)

# ----------------------------
# Simulation Function
# ----------------------------
def simulate_btc_btci_path(
    btc_initial, 
    btc_final, 
    months=60,
    correlation=0.92,
    dividend_yield=0.28,        # 28% dividend per month on BTCI holdings
    loan_apr=0.105,             # 10.5% APR for the leveraged case
    initial_investment=10000,   # Your own funds for BTCI (non-leveraged)
    loan_amount=20000,          # Borrowed funds for leveraged case
    btci_initial_price=4.0      # Baseline: when BTC = 80K, assume BTCI = 4.0
):
    """
    Simulate month-by-month evolution over 5 years (60 months) given a linear BTC price path.
    
    For BTC:
      - We generate a linear price path from btc_initial to btc_final.
    
    For BTCI:
      - The initial price is given (btci_initial_price).
      - Each month, BTCI's price changes by:
            btci_return = correlation * ( (BTC_t - BTC_t-1) / BTC_t-1 )
      - We update BTCI's price accordingly.
    
    We simulate two strategies:
      1. Non-Leveraged BTCI: 
         - Buy with initial_investment, compute initial shares = investment / btci_initial_price.
         - Each month, receive a dividend = dividend_yield * (shares * previous month's BTCI price)
         - Reinvest the dividend (i.e. buy additional shares at the current BTCI price).
      2. Leveraged BTCI:
         - Buy with (initial_investment + loan_amount) using the initial BTCI price.
         - Compute the fixed monthly amortizing loan payment for loan_amount at loan_apr over 60 months.
         - Each month, receive a dividend from your BTCI holdings, then use that dividend to help pay the monthly loan payment.
         - If the dividend exceeds the payment, reinvest the net cash to buy more BTCI shares.
         - Also update the outstanding loan principal based on the amortizing schedule.
    
    The function returns a DataFrame with monthly values including BTC price, BTCI price, 
    portfolio values for both strategies, loan details, and dividend information.
    """
    # Generate BTC price path
    btc_prices = generate_linear_path(btc_initial, btc_final, months)
    
    # Initialize arrays for BTCI price path
    btci_prices = np.zeros(months + 1)
    btci_prices[0] = btci_initial_price
    
    # For non-leveraged strategy: initial shares purchased with $10K
    shares_nonlev = initial_investment / btci_initial_price
    
    # For leveraged strategy: initial shares purchased with ($10K + $20K)
    shares_lev = (initial_investment + loan_amount) / btci_initial_price
    
    # Set up loan details for leveraged strategy
    outstanding_principal = loan_amount
    monthly_payment = monthly_payment_amortization(loan_amount, loan_apr, n_months=months)
    monthly_interest_rate = loan_apr / 12.0
    
    # Create lists to store monthly simulation data
    months_list = []
    btc_price_list = []
    btci_price_list = []
    nonlev_shares_list = []
    nonlev_value_list = []
    lev_shares_list = []
    lev_value_list = []
    dividend_nonlev_list = []
    dividend_lev_list = []
    loan_outstanding_list = []
    loan_interest_list = []
    loan_principal_list = []
    net_dividend_list = []
    shortfall_list = []  # if dividend doesn't cover loan payment
    
    # Loop over each month (0 to months)
    for t in range(months + 1):
        # Record month index
        months_list.append(t)
        btc_current = btc_prices[t]
        btc_price_list.append(btc_current)
        
        # For t=0, BTCI price is already set. For t>=1, update BTCI price:
        if t > 0:
            btc_return = (btc_prices[t] - btc_prices[t-1]) / btc_prices[t-1]
            btci_return = correlation * btc_return
            btci_prices[t] = btci_prices[t-1] * (1 + btci_return)
        btci_current = btci_prices[t]
        btci_price_list.append(btci_current)
        
        # --- Non-Leveraged BTCI Portfolio (Reinvesting Dividends) ---
        # Compute dividend based on the previous month's BTCI price
        if t == 0:
            dividend_nonlev = 0.0
        else:
            # Use previous month's BTCI value for dividend calculation:
            dividend_nonlev = dividend_yield * (shares_nonlev * btci_prices[t-1])
            # Reinvest dividend: buy additional shares at current BTCI price
            shares_nonlev += dividend_nonlev / btci_current
        nonlev_shares_list.append(shares_nonlev)
        nonlev_value = shares_nonlev * btci_current
        nonlev_value_list.append(nonlev_value)
        dividend_nonlev_list.append(dividend_nonlev)
        
        # --- Leveraged BTCI Portfolio ---
        # Calculate dividend on the leveraged position (using previous month's BTCI price)
        if t == 0:
            dividend_lev = 0.0
        else:
            dividend_lev = dividend_yield * (shares_lev * btci_prices[t-1])
        
        # For the loan, if t>0, update the outstanding principal using the amortization formula:
        if t > 0:
            interest_payment = outstanding_principal * monthly_interest_rate
            principal_payment = monthly_payment - interest_payment
            # Reduce outstanding principal
            outstanding_principal = max(0, outstanding_principal - principal_payment)
        else:
            interest_payment = 0.0
            principal_payment = 0.0
        
        # For the leveraged portfolio, assume you pay the full monthly loan payment using the dividend.
        # Then, if dividend exceeds the monthly payment, the excess is reinvested to buy additional BTCI shares.
        net_dividend = dividend_lev - monthly_payment
        if net_dividend > 0:
            shares_lev += net_dividend / btci_current
        # If dividend is lower than monthly payment, record the shortfall (this might have to be paid out-of-pocket)
        shortfall = monthly_payment - dividend_lev if dividend_lev < monthly_payment else 0.0
        
        lev_shares_list.append(shares_lev)
        lev_value = shares_lev * btci_current
        lev_value_list.append(lev_value)
        dividend_lev_list.append(dividend_lev)
        net_dividend_list.append(net_dividend)
        shortfall_list.append(shortfall)
        loan_outstanding_list.append(outstanding_principal)
        loan_interest_list.append(interest_payment)
        loan_principal_list.append(principal_payment)
    
    # Build a DataFrame of the simulation results for this scenario
    df = pd.DataFrame({
        "Month": months_list,
        "BTC Price": btc_price_list,
        "BTCI Price": btci_price_list,
        "NonLev Shares": nonlev_shares_list,
        "NonLev Portfolio Value": nonlev_value_list,
        "NonLev Dividend": dividend_nonlev_list,
        "Lev Shares": lev_shares_list,
        "Lev Portfolio Value": lev_value_list,
        "Lev Dividend": dividend_lev_list,
        "Net Dividend (Lev)": net_dividend_list,
        "Loan Outstanding": loan_outstanding_list,
        "Loan Interest Payment": loan_interest_list,
        "Loan Principal Payment": loan_principal_list,
        "Dividend Shortfall (Lev)": shortfall_list,
        "Monthly Loan Payment": [monthly_payment]*(months+1)
    })
    
    return df

# ----------------------------
# Main Simulation for Multiple Scenarios
# ----------------------------
# Define our final BTC price scenarios.
# We assume the simulation always starts at 80K and goes linearly to the target final price.
btc_final_scenarios = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000]
simulation_results = {}  # to store each scenario's DataFrame
final_summary = []       # to store final portfolio values

# Loop over each final BTC price scenario
for final_price in btc_final_scenarios:
    scenario_name = f"BTC 80K -> {final_price/1000:.0f}K"
    df_scenario = simulate_btc_btci_path(
        btc_initial=80000,
        btc_final=final_price,
        months=60,
        correlation=0.92,
        dividend_yield=0.28,
        loan_apr=0.105,
        initial_investment=10000,
        loan_amount=20000,
        btci_initial_price=4.0   # when BTC is 80K
    )
    simulation_results[scenario_name] = df_scenario
    final_row = df_scenario.iloc[-1]
    # For comparison, compute a Bitcoin holding value:
    # If you invest $10K in BTC at 80K, you get (10000/80000) BTC.
    btc_holding_value = (10000 / 80000) * final_price
    final_summary.append({
        "Scenario": scenario_name,
        "BTC Final Price": final_price,
        "Bitcoin Holding Value": btc_holding_value,
        "BTCI Non-Lev Portfolio": final_row["NonLev Portfolio Value"],
        "BTCI Lev Portfolio": final_row["Lev Portfolio Value"],
        "Total Loan Paid": final_row["Monthly Loan Payment"] * 60  # cumulative payment over 5 years
    })

summary_df = pd.DataFrame(final_summary)
print("Final Portfolio Summary over 5 Years for Each Scenario:")
print(summary_df)

# ----------------------------
# Visualization
# ----------------------------
plt.figure(figsize=(12, 6))
x = summary_df["BTC Final Price"]

plt.plot(x, summary_df["Bitcoin Holding Value"], marker='o', label="Bitcoin Holding (No Dividend)")
plt.plot(x, summary_df["BTCI Non-Lev Portfolio"], marker='o', label="Non-Leveraged BTCI (Reinvested Dividends)")
plt.plot(x, summary_df["BTCI Lev Portfolio"], marker='o', label="Leveraged BTCI (with $20K Loan)")
plt.xlabel("Final BTC Price (Constant for 5 Years)")
plt.ylabel("Final Portfolio Value ($)")
plt.title("5-Year Final Portfolio Values vs. BTC Final Price")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# (Optional) Plot a sample time series from one scenario
# ----------------------------
sample_scenario = simulation_results["BTC 80K -> 20K"]
plt.figure(figsize=(12, 6))
plt.plot(sample_scenario["Month"], sample_scenario["BTC Price"], label="BTC Price")
plt.plot(sample_scenario["Month"], sample_scenario["BTCI Price"], label="BTCI Price")
plt.xlabel("Month")
plt.ylabel("Price")
plt.title("Sample Price Paths for Scenario: BTC 80K -> 20K")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(sample_scenario["Month"], sample_scenario["NonLev Portfolio Value"], label="Non-Leveraged BTCI Value")
plt.plot(sample_scenario["Month"], sample_scenario["Lev Portfolio Value"], label="Leveraged BTCI Value")
plt.xlabel("Month")
plt.ylabel("Portfolio Value ($)")
plt.title("Portfolio Value Evolution for Scenario: BTC 80K -> 20K")
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1) Helper Functions
# ----------------------------

def monthly_payment_amortization(principal, annual_interest_rate, n_months=60):
    """
    Calculates the fixed monthly payment for an amortizing loan using:
      Payment = principal * [r / (1 - (1 + r)^(-n))]
    where r is monthly_interest_rate = annual_interest_rate/12.
    """
    r = annual_interest_rate / 12.0
    payment = principal * (r / (1 - (1 + r) ** (-n_months)))
    return payment

def generate_linear_path(start_price, end_price, n_months=60):
    """
    Generates a linear price path (numpy array) from start_price to end_price over n_months steps.
    """
    return np.linspace(start_price, end_price, n_months + 1)

def pick_random_dividend_yield(
    p_high=0.5,
    range_low=(0.15, 0.25),
    range_high=(0.28, 0.33)
):
    """
    With probability p_high, pick a random yield from range_high.
    Otherwise pick from range_low.
    """
    if np.random.rand() < p_high:
        return np.random.uniform(*range_high)
    else:
        return np.random.uniform(*range_low)

# ----------------------------
# 2) Single Simulation Function
# ----------------------------
def run_stress_test_simulation(
    months=60,
    btc_start=80000,
    btc_end=20000,
    correlation=0.92,
    loan_apr=0.105,
    loan_amount=20000,
    personal_investment=10000,
    btci_start_price=4.0,
    p_high=0.5,
    range_low=(0.15, 0.25),
    range_high=(0.28, 0.33)
):
    """
    Runs a single 5-year simulation where:
      - BTC goes linearly from btc_start to btc_end over 'months'.
      - BTCI price changes each month by 0.92 * (BTC % change).
      - Dividend yield each month is chosen randomly:
          * with probability p_high from range_high
          * otherwise from range_low
      - We track both:
          (1) Non-leveraged BTCI (reinvest dividends)
          (2) Leveraged BTCI ($20k loan at 10.5% APR, 5 years)
            - pay monthly loan payment using dividend
            - if leftover dividend, reinvest
      Returns final values and shortfall info.
    """

    # 1) BTC Price Path
    btc_prices = generate_linear_path(btc_start, btc_end, months)

    # 2) BTCI Price Path (initialized)
    btci_prices = np.zeros(months + 1)
    btci_prices[0] = btci_start_price

    # 3) Non-Leveraged Setup
    #    Start with personal_investment => initial shares
    shares_nonlev = personal_investment / btci_start_price

    # 4) Leveraged Setup
    #    Start with personal_investment + loan_amount
    shares_lev = (personal_investment + loan_amount) / btci_start_price
    outstanding_principal = loan_amount
    monthly_payment = monthly_payment_amortization(loan_amount, loan_apr, n_months=months)
    monthly_interest_rate = loan_apr / 12.0

    # Track shortfall months in leveraged scenario
    shortfall_months = 0

    for t in range(months + 1):
        if t > 0:
            # Update BTCI price based on correlation with BTC monthly return
            btc_return = (btc_prices[t] - btc_prices[t-1]) / btc_prices[t-1]
            btci_return = correlation * btc_return
            btci_prices[t] = btci_prices[t-1] * (1 + btci_return)
        # Current BTCI price
        current_btci_price = btci_prices[t]

        # --------------- Non-Leveraged (Reinvest) ---------------
        if t > 0:
            # Dividend yield is random each month
            div_yield = pick_random_dividend_yield(p_high, range_low, range_high)
            # Dividend based on last month's BTCI price * shares
            prev_btci_price = btci_prices[t-1]
            dividend_nonlev = div_yield * (shares_nonlev * prev_btci_price)
            # Reinvest
            shares_nonlev += dividend_nonlev / current_btci_price

        # --------------- Leveraged ---------------
        if t > 0:
            # Random monthly dividend yield
            div_yield_lev = pick_random_dividend_yield(p_high, range_low, range_high)
            prev_btci_price_lev = btci_prices[t-1]
            dividend_lev = div_yield_lev * (shares_lev * prev_btci_price_lev)

            # Loan amortization
            interest_payment = outstanding_principal * monthly_interest_rate
            principal_payment = monthly_payment - interest_payment
            outstanding_principal = max(0, outstanding_principal - principal_payment)

            # Net dividend after paying the loan
            net_dividend = dividend_lev - monthly_payment
            if net_dividend < 0:
                # shortfall
                shortfall_months += 1
                # (You could model paying out-of-pocket or selling shares_lev, but we won't do that here.)
            else:
                # Reinvest leftover
                shares_lev += net_dividend / current_btci_price

    # At the end, compute final portfolio values
    final_btci_price = btci_prices[-1]
    final_value_nonlev = shares_nonlev * final_btci_price
    final_value_lev = shares_lev * final_btci_price

    return final_value_nonlev, final_value_lev, shortfall_months

# ----------------------------
# 3) Monte Carlo Simulation
# ----------------------------
def run_monte_carlo_stress_test(
    n_runs=1000,
    months=60,
    btc_start=80000,
    btc_end=5000,
    correlation=0.92,
    loan_apr=0.105,
    loan_amount=20000,
    personal_investment=10000,
    btci_start_price=4.0,
    p_high=0.5,
    range_low=(0.15, 0.25),
    range_high=(0.28, 0.33),
    random_seed=42
):
    """
    Runs n_runs Monte Carlo simulations of the scenario:
      - BTC linearly from btc_start -> btc_end
      - Each month, dividend yield is random from range_low or range_high
      - correlation for BTCI price movement
      - 10.5% APR loan, 5 years
      - Tracks final values (non-lev, lev) and shortfall months for each run
    Returns a DataFrame with results of all runs.
    """
    np.random.seed(random_seed)

    results = {
        "Run": [],
        "Final Value (NonLev)": [],
        "Final Value (Lev)": [],
        "Shortfall Months (Lev)": []
    }

    for i in range(n_runs):
        fv_nonlev, fv_lev, shortfall = run_stress_test_simulation(
            months=months,
            btc_start=btc_start,
            btc_end=btc_end,
            correlation=correlation,
            loan_apr=loan_apr,
            loan_amount=loan_amount,
            personal_investment=personal_investment,
            btci_start_price=btci_start_price,
            p_high=p_high,
            range_low=range_low,
            range_high=range_high
        )
        results["Run"].append(i+1)
        results["Final Value (NonLev)"].append(fv_nonlev)
        results["Final Value (Lev)"].append(fv_lev)
        results["Shortfall Months (Lev)"].append(shortfall)

    return pd.DataFrame(results)

# ----------------------------
# 4) Run the Stress Test
# ----------------------------
if __name__ == "__main__":
    # Example usage:
    n_runs = 1000
    df_monte_carlo = run_monte_carlo_stress_test(
        n_runs=n_runs,
        months=60,
        btc_start=80000,
        btc_end=20000,
        correlation=0.92,
        loan_apr=0.105,
        loan_amount=20000,
        personal_investment=10000,
        btci_start_price=4.0,
        p_high=0.5,             # 50% chance of picking from high range
        range_low=(0.15, 0.25), # 15% - 25%
        range_high=(0.28, 0.33),# 28% - 33%
        random_seed=42
    )

    # Print summary stats
    print("\nMonte Carlo Stress Test Results (BTC from 80k -> 20k, Dividend Range 15-25% or 28-33%):")
    print(df_monte_carlo.describe())

    # Count how many runs had zero shortfall months, or at least one shortfall
    no_shortfall = (df_monte_carlo["Shortfall Months (Lev)"] == 0).sum()
    any_shortfall = (df_monte_carlo["Shortfall Months (Lev)"] > 0).sum()
    print(f"\nOut of {n_runs} runs:")
    print(f" - {no_shortfall} runs had NO shortfall months.")
    print(f" - {any_shortfall} runs had at least one shortfall month.")

    # Plot histograms of final values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(df_monte_carlo["Final Value (NonLev)"], bins=30, color='orange', alpha=0.7)
    plt.title("Non-Leveraged BTCI Final Value Distribution")
    plt.xlabel("Final Portfolio Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(df_monte_carlo["Final Value (Lev)"], bins=30, color='green', alpha=0.7)
    plt.title("Leveraged BTCI Final Value Distribution")
    plt.xlabel("Final Portfolio Value")

    plt.tight_layout()
    plt.show()