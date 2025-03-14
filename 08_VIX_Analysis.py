import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

def main():
    """
    Script to reproduce key statistics for the pair:
      - "Shorter-Dated Contract" (UX3) as our stand-in for the VIX
      - "Longer-Dated Contract"  (UX8) as our futures contract
    """
    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    file_path = r"C:\Users\ryanl\OneDrive\Desktop\MS Finance Courses\Applied Derivatives\Assignments\VIX\VIX_Futures.xlsx"
    # Use header=2 so that row index 2 (third row) is used as the header.
    df = pd.read_excel(file_path, header=3, engine='openpyxl')
    
    # Optional: Remove any extra whitespace from column names
    df.columns = df.columns.str.strip()
    
    print("Columns:", df.columns)
    print(df.head())
    
    # -------------------------------------------------------------------------
    # 2. Define Our "VIX" and "Futures" for This Assignment
    #    "Initial 3 Contract 8" means use UX3 vs UX8.
    # -------------------------------------------------------------------------
    # Convert the "Date" column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    
    myVIX_col = 'UX3'       # Shorter-dated contract (treated like VIX)
    myFut_col = 'UX8'       # Longer-dated futures contract
    basis_col = 'Basis_3_8' # We'll store the basis (UX8 - UX3) here
    df[basis_col] = df[myFut_col] - df[myVIX_col]
    
    # -------------------------------------------------------------------------
    # 3. Compute Descriptive Statistics (Exhibit 1 style)
    # -------------------------------------------------------------------------
    def descriptive_stats(series):
        s = series.dropna()
        return {
            'Count': len(s),
            'Mean': s.mean(),
            'Std Dev': s.std(),
            'Min': s.min(),
            '10th Pctl': s.quantile(0.10),
            'Median': s.median(),
            '90th Pctl': s.quantile(0.90),
            'Max': s.max(),
            'Skewness': s.skew(),
            'Kurtosis': s.kurtosis()
        }
    
    def first_diff_stats(series):
        diff_s = series.dropna().diff().dropna()
        return {
            'Mean(1stDiff)': diff_s.mean(),
            'Std Dev(1stDiff)': diff_s.std()
        }
    
    columns_of_interest = [myVIX_col, myFut_col, basis_col]
    stats_exhibit1_levels = {col: descriptive_stats(df[col]) for col in columns_of_interest}
    stats_exhibit1_diffs = {col: first_diff_stats(df[col]) for col in columns_of_interest}
    
    stats_exhibit1_levels_df = pd.DataFrame(stats_exhibit1_levels).T
    stats_exhibit1_diffs_df = pd.DataFrame(stats_exhibit1_diffs).T
    
    # -------------------------------------------------------------------------
    # 4. Reproduce the First 5 Columns from Exhibit 2
    # -------------------------------------------------------------------------
    # (A) Correlation (UX3 vs UX8)
    valid_df = df[[myVIX_col, myFut_col]].dropna()
    corr_value = valid_df[myVIX_col].corr(valid_df[myFut_col])
    
    # (B) Regression: UX8 ~ UX3
    X = sm.add_constant(valid_df[myVIX_col])
    y = valid_df[myFut_col]
    model = sm.OLS(y, X).fit()
    beta_value = model.params[myVIX_col]
    r_squared = model.rsquared
    t_stat_beta = model.tvalues[myVIX_col]
    
    # (C) Std Dev of first differences for UX8
    fut_first_diff_std = valid_df[myFut_col].diff().dropna().std()
    
    # (D) t-stat for difference in means: Test if the mean of the basis (UX8 - UX3) is zero.
    basis_series = df[basis_col].dropna()
    t_stat_basis, p_value_basis = stats.ttest_1samp(basis_series, popmean=0)
    
    exhibit2_stats = {
        "Correlation (UX3 vs UX8)": corr_value,
        "Beta (UX8 ~ UX3)": beta_value,
        "Std Dev(1stDiff UX8)": fut_first_diff_std,
        "t-Stat (Basis=0)": t_stat_basis,
        "R^2 (UX8 ~ UX3)": r_squared
    }
    
    # -------------------------------------------------------------------------
    # 5. Display/Print Results
    # -------------------------------------------------------------------------
    print("\n=== Descriptive Statistics (Levels) ===")
    print(stats_exhibit1_levels_df.round(4))
    print("\n=== Descriptive Statistics (First Differences) ===")
    print(stats_exhibit1_diffs_df.round(4))
    print("\n=== Regression & Correlation Stats (Exhibit 2) ===")
    for k, v in exhibit2_stats.items():
        print(f"{k}: {v:.4f}")
    print("\n=== Additional Notes ===")
    print(f"t-Stat for (Basis=0): {t_stat_basis:.4f}, p-value={p_value_basis:.4f}")
    print("Interpretation: If p-value < 0.05, the basis mean is significantly different from 0.")

if __name__ == "__main__":
    main()
