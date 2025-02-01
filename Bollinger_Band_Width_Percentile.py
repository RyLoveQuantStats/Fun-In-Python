### Bollinger Band Width Percentile ###
### Author: Ryan Loveless ###
### Date: 2022-02-22 ###

# For computing the Bollinger Band Width Percentile in Python for volatility analysis and trading strategies.

import numpy as np
import pandas as pd

###############################################################################
# Constants (similar to Pine's 'var string' or top-level variables)

S_HMMML = 'High - Mid Hi - Mid - Mid Low - Low'
S_HML   = 'High - Mid - Low'
S_HL    = 'High - Low'

###############################################################################
# Example user "inputs" from your Pine script
# In Pine, these are script inputs (e.g. input.int, input.string, etc.).
# Here, we store them in Python variables or read them from a config.

i_priceSrc      = None  # e.g., data["close"]
i_basisType     = 'SMA'   # e.g., 'SMA', 'EMA', 'WMA', 'RMA', 'HMA', 'VWMA'
i_bbwpLen       = 13
i_bbwpLkbk      = 252

i_c_typ_line    = 'Spectrum'   # or 'Solid'
i_c_so_line     = '#FFFF00'
i_c_typ_sp_line = S_HMMML
i_c_sp_hi_line  = '#FF0000'
i_c_sp_mhi_line = '#FFFF00'
i_c_sp_mid_line = '#00FF00'
i_c_sp_mlo_line = '#00FFFF'
i_c_sp_lo_line  = '#0000FF'
i_p_width_line  = 2

i_ma1On         = True
i_ma1Type       = 'SMA'
i_c_ma1         = '#FFFFFF'
i_ma1Len        = 5
i_ma2On         = False
i_ma2Type       = 'SMA'
i_c_ma2         = '#00FFFF'
i_ma2Len        = 8

i_alrtsOn       = True
i_upperLevel    = 98
i_lowerLevel    = 2

###############################################################################
# 1) f_maType: Replicates Pine's f_maType for different MAs
###############################################################################
def f_maType(price_series: pd.Series, length: int, ma_type: str) -> pd.Series:
    """
    Returns the chosen moving average (SMA, EMA, WMA, RMA, HMA, or VWMA)
    over 'price_series' using a window of 'length'.
    """
    import pandas_ta as ta  # or your chosen library

    if ma_type == "SMA":
        return ta.sma(price_series, length=length)
    elif ma_type == "EMA":
        return ta.ema(price_series, length=length)
    elif ma_type == "WMA":
        return ta.wma(price_series, length=length)
    elif ma_type == "RMA":
        return ta.rma(price_series, length=length)
    elif ma_type == "HMA":
        return ta.hma(price_series, length=length)
    else:  # "VWMA"
        return ta.vwma(price_series, length=length)

###############################################################################
# 2) f_bbwp: Bollinger Band Width Percentile
###############################################################################
def f_bbwp(
    price_series: pd.Series,
    bbw_len: int,
    bbwp_len: int,
    basis_type: str,
    bar_index_series: pd.Series
) -> pd.Series:
    """
    Replicates the Pine function f_bbwp which does:
      1) compute Bollinger Band Width = ((basis+dev) - (basis - dev)) / basis
      2) find how many of the previous 'bbwp_len' bars had a smaller BBW
      3) express that as a percentile * 100
    'bar_index_series' is a Series or array for 'bar_index' in Pine. 
    Returns a pd.Series of BBWP values.
    """
    # 1) get basis from f_maType
    basis = f_maType(price_series, bbw_len, basis_type)
    # standard dev
    dev = price_series.rolling(bbw_len).std()
    # Bollinger Band Width
    bbw = ((basis + dev) - (basis - dev)) / basis  # = (2*dev)/basis

    result = np.full(len(price_series), np.nan, dtype=float)
    arr_bbwp = bbw.to_numpy()
    arr_bar_index = bar_index_series.to_numpy() if isinstance(bar_index_series, pd.Series) else bar_index_series

    for idx in range(len(price_series)):
        # if idx < (bbw_len - 1), can't compute a full BBW yet
        if idx < (bbw_len - 1):
            continue

        # In Pine, "bar_index >= bbwp_len" => if idx < bbwp_len, partial
        if idx < bbwp_len:
            length_to_check = idx
        else:
            length_to_check = bbwp_len

        currentBBW = arr_bbwp[idx]
        if pd.isna(currentBBW):
            continue

        bbwSum = 0.0
        # Count how many olderBBW <= currentBBW
        for iBack in range(1, length_to_check + 1):
            if idx - iBack < 0:
                break
            olderBBW = arr_bbwp[idx - iBack]
            bbwSum += 0 if olderBBW > currentBBW else 1

        res_val = (bbwSum / length_to_check) * 100.0
        result[idx] = res_val

    return pd.Series(result, index=price_series.index, name="BBWP")

###############################################################################
# 3) f_5Col, f_3Col, f_clrSlct: color logic from Pine
###############################################################################
def f_5Col(val, lowV, lmV, midV, hmV, hiV, lowC, lmC, midC, mhC, hiC):
    """
    Pine uses color.from_gradient. We'll approximate banding logic:
    - If val <= lmV => lowC
    - elif val <= midV => lmC
    - elif val <= hmV => midC
    - elif val <= hiV => mhC
    - else => hiC
    """
    if val <= lmV:
        return lowC
    elif val <= midV:
        return lmC
    elif val <= hmV:
        return midC
    elif val <= hiV:
        return mhC
    else:
        return hiC

def f_3Col(val, lowV, midV, hiV, lowC, midC, hiC):
    """
    Similar banding logic for 3 steps
    """
    if val <= midV:
        return lowC
    elif val <= hiV:
        return midC
    else:
        return hiC

def f_clrSlct(val, c_type, solid_color, grad_type,
              lowV, lmV, midV, hmV, hiV,
              lowC, lmC, midC, mhC, hiC):
    """
    Pine's c_type = 'Solid' or 'Spectrum'
    grad_type can be S_HL, S_HML, S_HMMML
    """
    if pd.isna(val):
        return None

    if c_type == 'Solid':
        return solid_color
    else:
        # use gradient approach
        if grad_type == S_HL:
            # simplified 2-stop approach
            mid_pt = (lowV + hiV) / 2.0
            return lowC if val < mid_pt else hiC
        elif grad_type == S_HML:
            # 3-step
            return f_3Col(val, lowV, midV, hiV, lowC, midC, hiC)
        else:
            # S_HMMML => 5-step
            return f_5Col(val, lowV, lmV, midV, hmV, hiV, lowC, lmC, midC, mhC, hiC)

###############################################################################
# 4) Main logic from the script
###############################################################################
def compute_bbwp_dataframe(df: pd.DataFrame):
    """
    1. Compute BBWP via f_bbwp
    2. Compute color for each bar
    3. Compute MAs of BBWP if needed
    4. hiAlrtBar, loAlrtBar
    """
    # i_priceSrc -> df["Close"], for instance
    df["BBWP"] = f_bbwp(df["Close"], i_bbwpLen, i_bbwpLkbk, i_basisType, pd.Series(range(len(df))))

    def row_color(row):
        val = row["BBWP"]
        return f_clrSlct(
            val,
            i_c_typ_line,
            i_c_so_line,
            i_c_typ_sp_line,
            0, 25, 50, 75, 100,
            i_c_sp_lo_line, i_c_sp_mlo_line, i_c_sp_mid_line, i_c_sp_mhi_line, i_c_sp_hi_line
        )
    df["BBWP_Color"] = df.apply(row_color, axis=1)

    # MA1
    if i_ma1On:
        df["bbwpMA1"] = f_maType(df["BBWP"], i_ma1Len, i_ma1Type)
    else:
        df["bbwpMA1"] = np.nan

    # MA2
    if i_ma2On:
        df["bbwpMA2"] = f_maType(df["BBWP"], i_ma2Len, i_ma2Type)
    else:
        df["bbwpMA2"] = np.nan

    # Alerts
    def hi_alert(row):
        return row["BBWP"] if (i_alrtsOn and row["BBWP"] >= i_upperLevel) else np.nan
    def lo_alert(row):
        return row["BBWP"] if (i_alrtsOn and row["BBWP"] <= i_lowerLevel) else np.nan

    df["hiAlrtBar"] = df.apply(hi_alert, axis=1)
    df["loAlrtBar"] = df.apply(lo_alert, axis=1)

    return df

###############################################################################
# Example usage + optional colored background plotting
###############################################################################
def example_plot_with_background(df: pd.DataFrame):
    """
    Example of how you might replicate a color-gradient background in matplotlib
    for BBWP, similar to the Pine gradient logic.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # We'll define a custom colormap with 5 stops: 
    # lo -> mlo -> mid -> mhi -> hi
    # e.g. positions at 0.00, 0.25, 0.50, 0.75, 1.00 in a colormap
    stops = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [i_c_sp_lo_line, i_c_sp_mlo_line, i_c_sp_mid_line, i_c_sp_mhi_line, i_c_sp_hi_line]
    cmap = mcolors.LinearSegmentedColormap.from_list("BBWPcmap", list(zip(stops, colors)))

    # We'll create a 2D mesh so we can color the background from BBWP=0..100
    x_vals = np.arange(len(df))
    y_vals = np.arange(101)  # 0..100 inclusive
    XX, YY = np.meshgrid(x_vals, y_vals)

    # We'll create a pcolormesh where "color" = YY, so as Y goes from 0..100, we pick from the colormap
    # Then overlay the actual BBWP line
    fig, ax = plt.subplots(figsize=(10, 5))

    # pcolormesh for background
    # shading='auto' or 'nearest' depending on your mpl version
    c = ax.pcolormesh(XX, YY, YY, cmap=cmap, vmin=0, vmax=100, shading='auto')

    # Plot the actual BBWP
    ax.plot(x_vals, df["BBWP"], color='k', label="BBWP")

    # Mark hi alerts / lo alerts
    ax.scatter(x_vals, df["hiAlrtBar"], color='red', label="High Alerts")
    ax.scatter(x_vals, df["loAlrtBar"], color='blue', label="Low Alerts")

    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(df))
    ax.set_title("BBWP with gradient background")
    ax.legend()
    plt.colorbar(c, ax=ax, label="BBWP color scale")
    plt.show()

def main():
    # Create dummy DataFrame for demonstration
    # In real use, fetch data from yfinance or your data source
    dates = pd.date_range("2022-01-01", periods=300, freq="D")
    close_prices = np.random.rand(300)*100 + 100
    df = pd.DataFrame({
        "Close": close_prices,
        "High": close_prices+1,
        "Low":  close_prices-1,
        "Volume": np.random.randint(1000, 5000, size=300)
    }, index=dates)

    # Compute the BBWP logic
    result = compute_bbwp_dataframe(df)

    # Now demonstrate colorful background
    example_plot_with_background(result)

if __name__ == "__main__":
    main()
