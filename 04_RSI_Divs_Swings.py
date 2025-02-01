### Relative Strength Index and High/Low Swings ###
### Author: Ryan Loveless ###
### Date: 2025-01-12 ###

# For detecting RSI divergences (regular and hidden) and high/low swings to identify potential trade setups. 


import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np

class TradingStrategy:
    def __init__(
        self,
        account_balance=2500.0,
        risk_per_trade=2.0,         # % of account balance to risk per trade
        max_trades_per_day=5,
        daily_max_profit=200.0,     # Stop trading if daily PnL >= +$200
        daily_max_loss=-200.0,      # Stop trading if daily PnL <= -$200
        rr_ratio=2.0                # Risk:Reward ratio (1:2 by default)
    ):
        self.initial_balance = account_balance
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.max_trades_per_day = max_trades_per_day
        self.daily_max_profit = daily_max_profit
        self.daily_max_loss = daily_max_loss
        self.rr_ratio = rr_ratio

        # For tracking daily PnL, trades, etc.
        self.current_date = None
        self.daily_pnl = 0.0
        self.daily_trades_count = 0

        # List to store all completed trades
        self.all_trades = []

    # --------------------------------------------------------------------
    # Data Fetch
    # --------------------------------------------------------------------
    def fetch_data(self, symbol, period='1mo', interval='30m'):
        """
        Fetch market data from Yahoo Finance. Handles any MultiIndex columns 
        by flattening them, then renaming to single-level columns 
        like 'Close', 'Open', 'High', 'Low', 'Volume'.
        """
        print(f"[{symbol}] Fetching data (period={period}, interval={interval})...")
        data = yf.download(symbol, period=period, interval=interval)

        if data.empty:
            print(f"[{symbol}] No data returned. Empty DataFrame.")
            return pd.DataFrame()

        data.reset_index(inplace=True)

        # Flatten potential MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            new_cols = []
            for col_tuple in data.columns:
                if isinstance(col_tuple, tuple):
                    col_name = "_".join([x for x in col_tuple if x])
                else:
                    col_name = col_tuple
                new_cols.append(col_name)
            data.columns = new_cols

            # Rename if needed (single symbol assumption)
            possible_renames = {
                f"Close_{symbol}": "Close",
                f"Open_{symbol}": "Open",
                f"High_{symbol}": "High",
                f"Low_{symbol}": "Low",
                f"Volume_{symbol}": "Volume",
                f"Adj Close_{symbol}": "Adj Close",
            }
            for old_col, new_col in possible_renames.items():
                if old_col in data.columns:
                    data.rename(columns={old_col: new_col}, inplace=True)

        # Ensure we have at least 'Close', 'Open', 'High', 'Low' columns
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(data.columns):
            print(f"[{symbol}] Warning: Missing one or more required columns: {required_cols}")

        print(f"[{symbol}] Rows: {len(data)}  Columns: {list(data.columns)}")
        return data

    # --------------------------------------------------------------------
    # Indicators: RSI + Pivot Points
    # --------------------------------------------------------------------
    def calculate_rsi(self, data, length=14):
        """
        Calculate RSI and drop rows with NaN.
        """
        if 'Close' not in data.columns:
            # Fallback rename if only 'Adj Close'
            if 'Adj Close' in data.columns:
                data.rename(columns={'Adj Close': 'Close'}, inplace=True)
            else:
                print("No 'Close' column. Cannot compute RSI.")
                return data

        data['RSI'] = ta.rsi(data['Close'], length=length)

        if data['RSI'].isna().all():
            print("RSI returned all NaN. Possibly not enough bars.")
            return data

        before = len(data)
        data.dropna(subset=['RSI'], inplace=True)
        after = len(data)
        print(f"[RSI] Dropped {before - after} rows. Remaining: {after}")
        data.reset_index(drop=True, inplace=True)
        return data

    def pivot_points(self, series, left=2, right=2, direction='high'):
        """
        Identify pivot highs or lows using "left" and "right" bars on each side.
        Returns a boolean Series (True where pivot found).
        """
        if direction not in ['high', 'low']:
            raise ValueError("direction must be 'high' or 'low'.")
        # For each row, check if it's a local max or min
        shifted_cols = {}
        for i in range(1, left + 1):
            shifted_cols[f'left_{i}'] = series.shift(i)
        for j in range(1, right + 1):
            shifted_cols[f'right_{j}'] = series.shift(-j)

        df_compare = pd.DataFrame(shifted_cols)
        if direction == 'high':
            cond = True
            for col in df_compare.columns:
                cond &= (series >= df_compare[col])
        else:  # 'low'
            cond = True
            for col in df_compare.columns:
                cond &= (series <= df_compare[col])
        return cond

    def detect_divergences(self, data, left=2, right=2):
        """
        Pivot-based detection of REGULAR RSI divergences:
        - Bullish: Price makes lower low, RSI makes higher low
        - Bearish: Price makes higher high, RSI makes lower high
        """
        if 'RSI' not in data.columns:
            data['Bullish_Divergence'] = 0
            data['Bearish_Divergence'] = 0
            return data

        data['PricePivotHigh'] = self.pivot_points(data['Close'], left, right, direction='high')
        data['PricePivotLow']  = self.pivot_points(data['Close'], left, right, direction='low')
        data['RSIPivotHigh']   = self.pivot_points(data['RSI'],   left, right, direction='high')
        data['RSIPivotLow']    = self.pivot_points(data['RSI'],   left, right, direction='low')

        data['Bullish_Divergence'] = 0
        data['Bearish_Divergence'] = 0

        # Bullish Divergence
        price_pivots_low = data.index[data['PricePivotLow']]
        rsi_pivots_low   = data.index[data['RSIPivotLow']]
        for i in range(1, len(price_pivots_low)):
            cur_idx = price_pivots_low[i]
            prev_idx = price_pivots_low[i - 1]
            # Price lower low?
            if data['Close'].iloc[cur_idx] < data['Close'].iloc[prev_idx]:
                if (prev_idx in rsi_pivots_low) and (cur_idx in rsi_pivots_low):
                    # RSI higher low?
                    if data['RSI'].iloc[cur_idx] > data['RSI'].iloc[prev_idx]:
                        data.loc[cur_idx, 'Bullish_Divergence'] = 1

        # Bearish Divergence
        price_pivots_high = data.index[data['PricePivotHigh']]
        rsi_pivots_high   = data.index[data['RSIPivotHigh']]
        for i in range(1, len(price_pivots_high)):
            cur_idx = price_pivots_high[i]
            prev_idx = price_pivots_high[i - 1]
            # Price higher high?
            if data['Close'].iloc[cur_idx] > data['Close'].iloc[prev_idx]:
                if (prev_idx in rsi_pivots_high) and (cur_idx in rsi_pivots_high):
                    # RSI lower high?
                    if data['RSI'].iloc[cur_idx] < data['RSI'].iloc[prev_idx]:
                        data.loc[cur_idx, 'Bearish_Divergence'] = 1

        return data

    # --------------------------------------------------------------------
    # Position Sizing & PnL Tracking
    # --------------------------------------------------------------------
    def calculate_position_size(self, entry_price, stop_loss):
        """
        Return the position size (quantity) based on:
          - current account_balance
          - risk_per_trade% (e.g., 2%)
          - difference between entry_price and stop_loss
        """
        risk_dollars = self.account_balance * (self.risk_per_trade / 100.0)
        stop_loss_distance = abs(entry_price - stop_loss)
        if stop_loss_distance == 0:
            return 0

        raw_qty = risk_dollars / stop_loss_distance
        # You may have other constraints, e.g., min or max contract size
        return np.floor(min(raw_qty, 100))  # example: cap at 100, round down

    def check_daily_limits(self):
        """
        If daily PnL >= daily_max_profit or daily PnL <= daily_max_loss,
        or we've hit max_trades_per_day, we skip new trades.
        """
        if self.daily_trades_count >= self.max_trades_per_day:
            return False
        if self.daily_pnl >= self.daily_max_profit:
            return False
        if self.daily_pnl <= self.daily_max_loss:
            return False
        return True

    def update_daily_counters(self, current_bar_date):
        """
        If the date has changed from the last bar's date, reset daily counters.
        """
        if self.current_date is None or current_bar_date != self.current_date:
            self.current_date = current_bar_date
            self.daily_pnl = 0.0
            self.daily_trades_count = 0

    def simulate_trade_exit(self, side, entry_price, stop_loss, take_profit, high, low, close):
        """
        VERY SIMPLIFIED exit simulation: 
        - If side='buy', we check if 'low' <= stop_loss => exit at stop_loss
          else if 'high' >= take_profit => exit at take_profit
          else exit on the close of this bar
        - If side='sell', do the reverse logic
        This is just one simplistic approach to see if SL/TP gets hit intrabar.
        """
        exit_price = close  # default exit at bar's close

        if side == 'buy':
            # Stop hit?
            if low <= stop_loss:
                exit_price = stop_loss
            # Or target hit?
            elif high >= take_profit:
                exit_price = take_profit
        else:  # side == 'sell'
            if high >= stop_loss:
                exit_price = stop_loss
            elif low <= take_profit:
                exit_price = take_profit

        # PnL is (exit_price - entry_price)*qty for a BUY (negative if exit < entry)
        # For a SELL, PnL is (entry_price - exit_price)*qty
        return exit_price

    # --------------------------------------------------------------------
    # Core Strategy Loop
    # --------------------------------------------------------------------
    def run_strategy(self, symbol):
        """
        1. Fetch data
        2. Calculate RSI
        3. Detect divergences
        4. For each bar, check daily limits, see if divergence => open trade
        5. Simulate trade exit on next bar (for a simplistic approach)
        """
        data = self.fetch_data(symbol)
        if data.empty:
            print(f"[{symbol}] No data; exiting.")
            return

        data = self.calculate_rsi(data, length=14)
        if 'RSI' not in data.columns:
            print(f"[{symbol}] No valid RSI; skipping.")
            return

        data = self.detect_divergences(data, left=2, right=2)

        # We'll iterate up to the second-to-last bar and simulate exit on the next bar
        for i in range(len(data) - 1):
            row = data.iloc[i]
            next_row = data.iloc[i+1]

            # 1) Check if new day => reset daily counters
            self.update_daily_counters(row['Datetime'].date())

            # 2) Check daily limits
            if not self.check_daily_limits():
                continue

            # 3) Determine if there's a signal
            if row['Bullish_Divergence'] == 1:
                side = 'buy'
                # Stop is 1% below the current close (example)
                stop_loss = row['Close'] * 0.99
                # Take-profit based on R:R ratio, e.g., if we risk 1%, we target rr_ratio * 1%
                risk_perc = 0.01
                reward_perc = self.rr_ratio * risk_perc
                take_profit = row['Close'] * (1 + reward_perc)

            elif row['Bearish_Divergence'] == 1:
                side = 'sell'
                stop_loss = row['Close'] * 1.01
                risk_perc = 0.01
                reward_perc = self.rr_ratio * risk_perc
                take_profit = row['Close'] * (1 - reward_perc)
            else:
                continue  # No signal

            # 4) Calculate position size
            entry_price = row['Close']
            qty = self.calculate_position_size(entry_price, stop_loss)
            if qty <= 0:
                continue

            # 5) "Open" the trade, then simulate exit on the NEXT bar
            #    We'll see if next bar's high/low triggers SL or TP, else we exit at next bar's close
            exit_price = self.simulate_trade_exit(
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                high=next_row['High'],
                low=next_row['Low'],
                close=next_row['Close']
            )

            # 6) Calculate PnL
            if side == 'buy':
                pnl = (exit_price - entry_price) * qty
            else:  # sell
                pnl = (entry_price - exit_price) * qty

            # 7) Update account balance & daily PnL
            self.account_balance += pnl
            self.daily_pnl += pnl
            self.daily_trades_count += 1

            # 8) Record the trade
            trade_dict = {
                'symbol': symbol,
                'datetime_entry': row['Datetime'],
                'datetime_exit': next_row['Datetime'],
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'qty': qty,
                'pnl': pnl,
                'balance_after': self.account_balance
            }
            self.all_trades.append(trade_dict)

            print(f"[{symbol}] {side.upper()} {qty} @ {entry_price:.2f} / Exit {exit_price:.2f} => PnL: {pnl:.2f}")

        # Summarize
        total_pnl = sum(t['pnl'] for t in self.all_trades if t['symbol'] == symbol)
        print(f"\n[{symbol}] Strategy complete. Final account balance: {self.account_balance:.2f}")
        print(f"[{symbol}] Total PnL from trades: {total_pnl:.2f}\n")

# -------------------------------------------------------------------------
# Example usage on multiple futures symbols
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # You can add more futures tickers, e.g. "ES=F", "NQ=F", "CL=F", "GC=F", "YM=F", etc.
    futures_symbols = ["ES=F", "NQ=F"]

    strategy = TradingStrategy(
        account_balance=10000.0,
        risk_per_trade=2.0,       # 2% risk per trade
        max_trades_per_day=3,
        daily_max_profit=500.0,
        daily_max_loss=-300.0,
        rr_ratio=2.0              # 1:2 risk-to-reward
    )

    for fut in futures_symbols:
        strategy.run_strategy(fut)

    # Print all trades summary
    df_trades = pd.DataFrame(strategy.all_trades)
    if not df_trades.empty:
        print("All Trades:\n", df_trades)
    else:
        print("No trades were placed.")
