### Options Trading Calculcator ###
### Author: Ryan Loveless ###
### Date: 2022-02-22 ###

# For computing pnl and other metrics for options trading. 


import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Options Portfolio Tracker"

# Initialize trade log (data storage)
trade_log = []

# Layout
app.layout = html.Div([
    html.H1("Options Portfolio Tracker", style={"text-align": "center"}),

    # Input fields for trade details
    html.Div([
        html.Label("Symbol:"),
        dcc.Input(id="symbol-input", type="text", placeholder="Enter symbol...", debounce=True),
        html.Br(),

        html.Label("Side (Long/Short):"),
        dcc.Input(id="side-input", type="text", placeholder="Enter side..."),
        html.Br(),

        html.Label("Quantity:"),
        dcc.Input(id="quantity-input", type="number", placeholder="Enter quantity..."),
        html.Br(),

        html.Label("Average Fill Price:"),
        dcc.Input(id="avg-fill-input", type="number", placeholder="Enter average fill price..."),
        html.Br(),

        html.Label("Last Price:"),
        dcc.Input(id="last-price-input", type="number", placeholder="Enter last price..."),
        html.Br(),

        html.Label("Leverage (e.g., 500:1):"),
        dcc.Input(id="leverage-input", type="text", placeholder="Enter leverage..."),
        html.Br(),
    ], style={"margin-bottom": "20px", "max-width": "400px", "margin": "auto"}),

    # Buttons
    html.Div([
        html.Button("Calculate", id="calculate-button", n_clicks=0),
        html.Button("Add to Portfolio", id="add-to-portfolio-button", n_clicks=0, style={"margin-left": "10px"}),
        html.Button("Clear Portfolio", id="clear-portfolio-button", n_clicks=0, style={"margin-left": "10px"})
    ], style={"text-align": "center", "margin-bottom": "20px"}),

    # Portfolio summary
    html.Div(id="portfolio-summary", style={"border": "1px solid #ccc", "padding": "10px", "border-radius": "5px", "margin-top": "20px"}),

    # Output section for calculated results
    html.Div(id="output-section", style={"margin-top": "20px"}),

    # Trade log (Portfolio display)
    html.Div([
        html.H4("Portfolio", style={"text-align": "center"}),
        html.Div(id="portfolio-table", style={"margin-top": "20px"})
    ])
], style={"font-family": "Arial, sans-serif", "padding": "20px"})


# Callback to calculate and update metrics or add to portfolio
@app.callback(
    [
        Output("output-section", "children"),
        Output("portfolio-table", "children"),
        Output("portfolio-summary", "children")
    ],
    [
        Input("calculate-button", "n_clicks"),
        Input("add-to-portfolio-button", "n_clicks"),
        Input("clear-portfolio-button", "n_clicks")
    ],
    [
        State("symbol-input", "value"),
        State("side-input", "value"),
        State("quantity-input", "value"),
        State("avg-fill-input", "value"),
        State("last-price-input", "value"),
        State("leverage-input", "value")
    ]
)
def update_portfolio(calculate_clicks, add_to_portfolio_clicks, clear_portfolio_clicks,
                     symbol, side, quantity, avg_fill_price, last_price, leverage):
    global trade_log

    # Clear portfolio
    if clear_portfolio_clicks > 0:
        trade_log = []
        return "", "", html.Div("Portfolio cleared!", style={"color": "red", "text-align": "center"})

    # Ensure all fields are filled
    if not all([symbol, side, quantity, avg_fill_price, last_price, leverage]):
        return html.Div("Please fill in all fields!", style={"color": "red"}), "", ""

    # Parse leverage
    try:
        leverage_ratio = int(leverage.split(":")[0]) if ":" in leverage else 1
    except ValueError:
        return html.Div("Invalid leverage format. Use '500:1' or similar.", style={"color": "red"}), "", ""

    # Calculate metrics
    multiplier = 100  # Standard option multiplier
    trade_value = avg_fill_price * quantity * multiplier
    market_value = last_price * quantity * multiplier
    unrealized_pnl = (last_price - avg_fill_price) * quantity * multiplier
    margin = trade_value / leverage_ratio

    # Calculate button action
    if calculate_clicks > 0:
        output_metrics = html.Div([
            html.H4(f"Trade Metrics for {symbol}"),
            html.P(f"Side: {side}"),
            html.P(f"Quantity: {quantity}"),
            html.P(f"Average Fill Price: ${avg_fill_price:.2f}"),
            html.P(f"Last Price: ${last_price:.2f}"),
            html.P(f"Trade Value: ${trade_value:,.2f}"),
            html.P(f"Market Value: ${market_value:,.2f}"),
            html.P(f"Unrealized P&L: ${unrealized_pnl:,.2f}"),
            html.P(f"Margin Requirement: ${margin:,.2f}")
        ], style={"border": "1px solid #ccc", "padding": "10px", "border-radius": "5px"})
    else:
        output_metrics = ""

    # Add to Portfolio button action
    if add_to_portfolio_clicks > 0:
        trade_log.append({
            "Symbol": symbol,
            "Side": side,
            "Quantity": quantity,
            "Avg Fill Price": avg_fill_price,
            "Last Price": last_price,
            "Trade Value": trade_value,
            "Market Value": market_value,
            "Unrealized P&L": unrealized_pnl,
            "Margin": margin
        })

    # Display portfolio
    if trade_log:
        trade_log_df = pd.DataFrame(trade_log)
        portfolio_table = html.Table([
            html.Thead(html.Tr([html.Th(col) for col in trade_log_df.columns])),
            html.Tbody([
                html.Tr([html.Td(row[col]) for col in trade_log_df.columns]) for _, row in trade_log_df.iterrows()
            ])
        ], style={"width": "100%", "border": "1px solid black", "border-collapse": "collapse"})
    else:
        portfolio_table = html.Div("Portfolio is empty!", style={"text-align": "center", "color": "gray"})

    # Portfolio summary
    if trade_log:
        total_trade_value = sum(trade["Trade Value"] for trade in trade_log)
        total_market_value = sum(trade["Market Value"] for trade in trade_log)
        total_unrealized_pnl = sum(trade["Unrealized P&L"] for trade in trade_log)
        total_margin = sum(trade["Margin"] for trade in trade_log)

        portfolio_summary = html.Div([
            html.H4("Portfolio Summary"),
            html.P(f"Total Trade Value: ${total_trade_value:,.2f}"),
            html.P(f"Total Market Value: ${total_market_value:,.2f}"),
            html.P(f"Total Unrealized P&L: ${total_unrealized_pnl:,.2f}"),
            html.P(f"Total Margin Requirement: ${total_margin:,.2f}")
        ])
    else:
        portfolio_summary = ""

    return output_metrics, portfolio_table, portfolio_summary


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
