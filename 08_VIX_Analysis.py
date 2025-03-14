import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from tabulate import tabulate
import matplotlib.pyplot as plt
import os
import datetime

def main():
    """
    This script reproduces key statistics and visualizations for the VIX.
    
    Analysis Overview:
      - UX3 is used as a proxy for the VIX (shorter-dated contract).
      - UX8 is used as the actively traded futures contract.
      - The Futures Basis is computed as: UX8 - UX3.
      
    The script:
      1. Loads data from an Excel file and filters it to the period Jan 2018 - Dec 2023.
      2. Computes descriptive statistics for levels and first differences (Exhibit 1).
      3. Reproduces key regression/correlation metrics (Exhibit 2).
      4. Displays the results in formatted tables using tabulate.
      5. Generates visualization plots (time series, scatter with regression, and histogram).
      6. Exports all computed tables into one CSV file and saves each plot separately.
    """
    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    file_path = r"C:\Users\ryanl\OneDrive\Desktop\MS Finance Courses\Applied Derivatives\Assignments\VIX\VIX_Futures.xlsx"
    # Use header=3 because the header is on row 4 (0-indexed 3)
    df = pd.read_excel(file_path, header=3, engine='openpyxl')
    
    # Remove any extra whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Display initial columns and sample rows
    print("Columns:", df.columns.tolist())
    print(df.head())
    
    # -------------------------------------------------------------------------
    # 2. Filter Data by Date Range (January 2018 - December 2023)
    # -------------------------------------------------------------------------
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    
    start_date = pd.to_datetime("2018-01-01")
    end_date = pd.to_datetime("2023-12-31")
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # -------------------------------------------------------------------------
    # 3. Define the Analysis Pair and Compute the Futures Basis
    # -------------------------------------------------------------------------
    # "Initial 3 Contract 8" means:
    #    - UX3 is used as the VIX proxy (shorter-dated)
    #    - UX8 is used as the futures contract (longer-dated)
    myVIX_col = 'UX3'
    myFut_col = 'UX8'
    basis_col = 'Basis_3_8'
    df[basis_col] = df[myFut_col] - df[myVIX_col]
    
    # -------------------------------------------------------------------------
    # 4. Compute Descriptive Statistics (Exhibit 1)
    #    For Levels and First Differences for:
    #      - VIX (UX3)
    #      - Futures Contract (UX8)
    #      - Futures Basis (UX8 - UX3)
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
    
    # Compute stats for UX3, UX8, and the Futures Basis.
    columns_of_interest = [myVIX_col, myFut_col, basis_col]
    stats_levels = {col: descriptive_stats(df[col]) for col in columns_of_interest}
    stats_diffs = {col: first_diff_stats(df[col]) for col in columns_of_interest}
    
    stats_levels_df = pd.DataFrame(stats_levels).T
    stats_diffs_df = pd.DataFrame(stats_diffs).T
    
    # Print Exhibit 1 tables using tabulate
    print("\n=== Exhibit 1: Descriptive Statistics (Levels) ===")
    print(tabulate(stats_levels_df, headers="keys", tablefmt="pretty", floatfmt=".4f"))
    
    print("\n=== Exhibit 1: Descriptive Statistics (First Differences) ===")
    print(tabulate(stats_diffs_df, headers="keys", tablefmt="pretty", floatfmt=".4f"))
    
    # -------------------------------------------------------------------------
    # 5. Regression & Correlation Analysis (Exhibit 2)
    #    Compute:
    #      1. Correlation (UX3 vs UX8)
    #      2. Beta from regression: UX8 ~ UX3
    #      3. Std Dev of first differences for UX8
    #      4. t-Stat for testing if the mean of the Futures Basis is zero
    #      5. R^2 from the regression of UX8 on UX3
    # -------------------------------------------------------------------------
    valid_df = df[[myVIX_col, myFut_col]].dropna()
    
    # (A) Correlation between UX3 and UX8
    corr_value = valid_df[myVIX_col].corr(valid_df[myFut_col])
    
    # (B) Regression: UX8 ~ UX3
    X = sm.add_constant(valid_df[myVIX_col])
    y = valid_df[myFut_col]
    model = sm.OLS(y, X).fit()
    beta_value = model.params[myVIX_col]
    r_squared = model.rsquared
    t_stat_beta = model.tvalues[myVIX_col]
    
    # (C) Standard deviation of first differences for UX8
    fut_first_diff_std = valid_df[myFut_col].diff().dropna().std()
    
    # (D) t-Stat: Test if the mean of the Futures Basis is zero
    basis_series = df[basis_col].dropna()
    t_stat_basis, p_value_basis = stats.ttest_1samp(basis_series, popmean=0)
    
    # Compile the Exhibit 2 statistics into a dictionary.
    exhibit2_stats = {
        "Correlation (UX3 vs UX8)": corr_value,
        "Beta (UX8 ~ UX3)": beta_value,
        "Std Dev(1stDiff UX8)": fut_first_diff_std,
        "t-Stat (Basis=0)": t_stat_basis,
        "R^2 (UX8 ~ UX3)": r_squared
    }
    
    exhibit2_stats_df = pd.DataFrame(exhibit2_stats, index=["Value"]).T
    print("\n=== Exhibit 2: Regression & Correlation Statistics ===")
    print(tabulate(exhibit2_stats_df, headers="keys", tablefmt="pretty", floatfmt=".4f"))
    
    # -------------------------------------------------------------------------
    # 6. Visualization Plots
    # -------------------------------------------------------------------------
    output_dir = r"C:\Users\ryanl\OneDrive\Desktop\MS Finance Courses\Applied Derivatives\Assignments\VIX\Outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # a) Time Series Plot: UX3, UX8, and Futures Basis over time.
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df[myVIX_col], label='UX3 (VIX proxy)', linestyle='-', marker='.', markersize=2)
    plt.plot(df['Date'], df[myFut_col], label='UX8 (Futures)', linestyle='-', marker='.', markersize=2)
    plt.plot(df['Date'], df[basis_col], label='Futures Basis (UX8 - UX3)', linestyle='--', marker='.', markersize=2)
    plt.xlabel("Date")
    plt.ylabel("Price / Index Level")
    plt.title("Time Series of VIX Proxy, Futures, and Futures Basis")
    plt.legend()
    plt.tight_layout()
    ts_plot_path = os.path.join(output_dir, "Time_Series_Plot.png")
    plt.savefig(ts_plot_path)
    plt.show()
    
    # b) Scatter Plot with Regression Line: UX3 vs UX8
    plt.figure(figsize=(8, 6))
    plt.scatter(valid_df[myVIX_col], valid_df[myFut_col], alpha=0.5, label='Data Points')
    x_vals = np.linspace(valid_df[myVIX_col].min(), valid_df[myVIX_col].max(), 100)
    y_vals = model.params[0] + model.params[myVIX_col] * x_vals
    plt.plot(x_vals, y_vals, color='red', label='Regression Line')
    plt.xlabel("UX3 (VIX proxy)")
    plt.ylabel("UX8 (Futures)")
    plt.title("Scatter Plot with Regression Line: UX3 vs UX8")
    plt.legend()
    plt.tight_layout()
    scatter_plot_path = os.path.join(output_dir, "Scatter_Plot.png")
    plt.savefig(scatter_plot_path)
    plt.show()
    
    # c) Histogram: Distribution of the Futures Basis
    plt.figure(figsize=(8, 6))
    plt.hist(df[basis_col].dropna(), bins=30, alpha=0.75, color='blue')
    plt.xlabel("Futures Basis (UX8 - UX3)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Futures Basis")
    plt.tight_layout()
    hist_plot_path = os.path.join(output_dir, "Histogram_Basis.png")
    plt.savefig(hist_plot_path)
    plt.show()
    
    # -------------------------------------------------------------------------
    # 7. Combine All Results into One CSV File
    # -------------------------------------------------------------------------
    # For Exhibit 1 (Levels)
    levels_export = stats_levels_df.copy()
    levels_export.insert(0, "Statistic", levels_export.index)
    levels_export["Exhibit"] = "Exhibit 1 Levels"
    levels_export.reset_index(drop=True, inplace=True)
    
    # For Exhibit 1 (First Differences)
    diffs_export = stats_diffs_df.copy()
    diffs_export.insert(0, "Statistic", diffs_export.index)
    diffs_export["Exhibit"] = "Exhibit 1 First Differences"
    diffs_export.reset_index(drop=True, inplace=True)
    
    # For Exhibit 2 (Regression & Correlation)
    exhibit2_export = exhibit2_stats_df.copy()
    exhibit2_export.insert(0, "Metric", exhibit2_export.index)
    exhibit2_export["Exhibit"] = "Exhibit 2 Regression Stats"
    exhibit2_export.reset_index(drop=True, inplace=True)
    
    # Combine all the tables vertically.
    combined_df = pd.concat([levels_export, diffs_export, exhibit2_export], ignore_index=True)
    
    # Append a timestamp to the CSV file name for uniqueness.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_csv_path = os.path.join(output_dir, f"Combined_Results_{timestamp}.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    
    print("\nCSV file saved with all results at:")
    print(f" - {combined_csv_path}")
    
    # -------------------------------------------------------------------------
    # 8. Additional Notes
    # -------------------------------------------------------------------------
    print("\n=== Additional Notes ===")
    print(f"t-Stat for (Basis=0): {t_stat_basis:.4f}, p-value: {p_value_basis:.4f}")
    if p_value_basis < 0.05:
        print("=> The mean of the futures basis is significantly different from 0 at the 5% level.")
    else:
        print("=> The mean of the futures basis is NOT significantly different from 0 at the 5% level.")
    print(f"Time Series Plot saved at: {ts_plot_path}")
    print(f"Scatter Plot saved at: {scatter_plot_path}")
    print(f"Histogram saved at: {hist_plot_path}")

if __name__ == "__main__":
    main()


"""
===========================================================================
RESULT INTERPRETATION & CONCLUSIONS
===========================================================================

1. Data Overview:
   - The dataset covers observations from January 2018 to December 2023.
   - Key columns used in the analysis:
       • UX3 is used as a proxy for the VIX (shorter-dated contract).
       • UX8 is used as the actively traded futures contract.
       • The Futures Basis is computed as: UX8 – UX3.
   - After filtering, there are 1511 observations for each series.

2. Descriptive Statistics (Exhibit 1):
   a) Levels (Raw Data):
       - UX3 (VIX proxy):
           • Mean: ~22.36
           • Std Dev: ~5.73
           • Range: 12.325 (min) to 59.925 (max)
           • Skewness: 0.84 (moderate right skew)
           • Kurtosis: 1.39 (close to normal but slightly flatter)
       - UX8 (Futures):
           • Mean: ~22.89
           • Std Dev: ~4.48
           • Range: 14.9 (min) to 34.125 (max)
           • Skewness: 0.02 (almost symmetric)
           • Kurtosis: -1.46 (flatter, platykurtic)
       - Futures Basis (UX8 – UX3):
           • Mean: ~0.53 (on average, futures are slightly higher than the VIX proxy)
           • Std Dev: ~2.20 (high variability)
           • Range: approximately -25.80 to 4.00
           • Skewness: -3.71 (strong left skew)
           • Kurtosis: 26.82 (very heavy tails, indicating potential outliers)
   
   b) First Differences (Daily Changes):
       - UX3 and UX8 both have mean first differences near zero, indicating no consistent daily drift.
       - The standard deviation of daily changes is higher for UX3 (~0.97) compared to UX8 (~0.46), showing that the VIX proxy is more volatile on a day-to-day basis.
       - The first differences of the Futures Basis also center around zero with a moderate variability (~0.62).

3. Regression & Correlation Analysis (Exhibit 2):
   - The Pearson correlation between UX3 and UX8 is very high (~0.936), indicating a strong linear relationship.
   - The regression of UX8 on UX3 yields a beta of ~0.733, which implies that for each 1-point change in UX3, UX8 changes by about 0.73 points.
   - The R² value of ~0.876 indicates that approximately 87.6% of the variation in UX8 can be explained by variations in UX3.
   - The t-statistic for testing whether the mean of the Futures Basis is zero is ~9.394 with a p-value essentially 0.0000,
     meaning that the average basis is statistically significantly different from zero.

4. Visualizations:
   - **Time Series Plot:**  
     Illustrates how UX3, UX8, and the Futures Basis evolve over time.
   - **Scatter Plot with Regression Line:**  
     Confirms the strong linear relationship between UX3 and UX8 and displays the fitted regression line.
   - **Histogram of the Futures Basis:**  
     Highlights the highly skewed distribution of the basis with heavy tails, supporting the descriptive statistics.

5. Overall Conclusions:
   - Although UX3 (the VIX proxy) and UX8 (the futures contract) are strongly correlated, 
     there exists a statistically significant futures basis (UX8 – UX3) that is both volatile and non-normally distributed.
   - The regression analysis shows that UX8 is less sensitive to changes in UX3 (beta < 1), and a high R² indicates a tight linear relationship.
   - The extreme skewness and kurtosis in the basis suggest that market pricing of the futures relative to the VIX proxy has episodes of large deviations,
     which may be important when considering trading strategies based on the futures basis.

===========================================================================
"""
