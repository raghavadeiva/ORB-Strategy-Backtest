# ORB Strategy Backtester

## 1. Functional and Logical Overview

The `backtest.py` script is a sophisticated intraday algorithmic backtesting engine designed specifically for trading 3x leveraged Exchange Traded Funds (ETFs). It models a momentum-based breakout and breakdown strategy, utilizing the 15-minute Opening Range Breakout (ORB) as its primary timing mechanism. The system dynamically scales risk, applies market internals as filters, and simulates realistic trade management to evaluate historical performance.

### Core Trading Logic & Data Pipeline

* **Asset Mapping:** The script categorizes foundational market sectors and indices (e.g., SPY, SOXX, XLK) by volatility (low, medium, high, extreme) and automatically maps them to their 3x leveraged Bull and Bear equivalents (e.g., SOXX maps to SOXL for longs and SOXS for shorts).


* **Data Aggregation:** It queries 1-minute historical pricing data directly from the Alpaca API using the full market SIP data feed (`DataFeed.SIP`) and caches this data locally in `.parquet` files to optimize subsequent runs.


* **Regime Warm-Up:** When processing custom date ranges, the engine automatically fetches an additional 60 days of preceding data to ensure lagging indicators (like EMA and ADX) are fully calculated and "warmed up" before the first simulated trading day.



### Indicator Suite

The engine calculates a confluence of lagging, leading, and volume-based metrics to qualify trades:

* **Volume Weighted Average Price (VWAP):** Calculated dynamically from the daily open using typical price and cumulative volume.


* **Opening Range Breakout (ORB):** Establishes the high and low price levels formed between 09:30 and 09:44 EST.


* **Relative Strength (ORB_RS):** Measures the percentage change of the asset specifically during the 15-minute ORB window.


* **Average Directional Index (ADX):** Uses a 14-period smoothing window to measure trend strength.


* **Synthetic Market Internals (ADD):** Constructs a synthetic intraday Advance/Decline line by evaluating the real-time breadth of 11 distinct SPDR sector ETFs. It also measures "ADD Delta," tracking the 15-minute rate of change in market breadth.


* **Volatility (VIX) Surge:** Uses VIXY as a proxy to calculate the intraday percentage surge in market fear relative to the daily open.


* **Value Area:** Calculates a volume profile based on typical price, identifying the upper (`VA_High`) and lower (`VA_Low`) bounds where 70% of the previous day's volume occurred.



### Trade Execution & Management

* **Entry Triggers:** A trade is triggered before a user-defined time cutoff if the asset's price breaks the ORB (above `ORB_High` for bulls, below `ORB_Low` for bears) while simultaneously aligning with VWAP direction. In "macro" mode, entries must also pass structural breadth tests (ADD), trend strength tests (ADX > threshold), and Value Area breakouts.


* **Take Profit (TP) & Stop Loss (SL):** Trades are managed via fixed percentage risk parameters, which can optionally be converted into a dynamic trailing stop once the trade achieves 50% of its target.


* **Scale-Out Mechanism:** If activated, the engine simulates selling 33% of the position at breakeven (1R) to secure partial profits while letting the remainder run.


* **Time-Based Exits:** Any position remaining open at 15:58 EST is automatically liquidated at the current market close price to avoid overnight gap risk.


* **Circuit Breakers:** An optional risk management feature tracks consecutive losing trades per asset. If the maximum loss streak is hit, the engine imposes a multi-day trading ban on that specific asset.



---

## 2. Technical Execution and Parameters

### Prerequisites and Setup

To execute the engine, Python must be installed along with the required libraries (`pandas`, `numpy`, `alpaca-py`, and a parquet engine like `pyarrow`). You must also have valid Alpaca API credentials exported as system environment variables before running the script:

* `ALPACA_API_KEY`

* `ALPACA_SECRET_KEY`


**Basic Execution Command:**

```bash
python backtest.py --group extreme --days 30 --bull_tp 8.0 --bull_sl 3.0 --bear_tp 5.0 --bear_sl 2.0

```

### Parameter Breakdown

The script accepts extensive command-line arguments to modify the strategy's behavior without editing the source code.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--group` | String | `extreme` | Target volatility group to backtest (`high`, `medium`, `low`, `extreme`) or specific comma-separated tickers.

 |
| `--mode` | String | `macro` | Trading mode (`macro` uses full indicators, `micro` ignores higher-timeframe context).

 |
| `--capital` | Float | `10000.0` | Initial starting capital used to calculate dollar-based PnL.

 |
| `--start_date` / `--end_date` | String | `None` | Custom date range in `YYYY-MM-DD` format. Overrides the `--days` parameter.

 |
| `--days` | Integer | `2500` | Lookback period in days from the hardcoded end date if custom dates are not provided.

 |
| `--bull_tp` / `--bear_tp` | String | `8.0,XLK:6` | Take profit percentages for long/short trades. Supports asset-specific overrides (e.g., standard 8.0%, but XLK uses 6.0%).

 |
| `--bull_sl` / `--bear_sl` | String | `3.0,XLK:1.5` | Stop loss percentages for long/short trades. Supports asset-specific overrides.

 |
| `--bull_adx` / `--bear_adx` | Float | `11.0` / `13.0` | Minimum ADX value required to authorize a long or short trade.

 |
| `--cutoff` | String | `10:30` | The time after which no new trades will be opened (Format: HH:MM).

 |
| `--orb_min` / `--orb_max` | Float | `0.0066` / `0.0166` | The minimum and maximum allowable ORB width percentage. Rejects trades if the opening range is too tight or excessively wide.

 |
| `--max_concurrent` | Integer | `1` | The maximum number of overlapping trades allowed across the portfolio at any given time.

 |

**Boolean Toggle Flags**
Append these flags to the command line to activate specific behaviors:

* `--use_cb`: Enables the consecutive loss circuit breaker logic.


* Requires `--max_streak` (default: 3) and `--cooldown` (default: 5) to dictate the rules.




* `--use_rs_ranking`: Ranks assets by ORB momentum in a priority queue to decide which trades to take if multiple assets trigger simultaneously.


* `--slippage`: Applies a simulated slippage penalty to stop-out prices based on the entry constraints.


* `--scale_out`: Sells 33% of the position when the trade reaches a 1:1 risk-to-reward ratio.


* `--trail`: Replaces the fixed Take Profit with a dynamic trailing stop that activates at 50% of the target.


* `--no_lagging` / `--no_leading`: Disables specific indicator checks (ADX/9EMA or VIX/ADD) to isolate logic.


* `--clear_cache`: Wipes the `./data_cache` directory, forcing a fresh download of Alpaca `.parquet` files.



### Outputs and Logging

Upon completion, the engine prints a portfolio summary to the console containing Total Trades, Win Rate, Net PnL, Final Capital, Average Drawdown (MAE), and Missed Profit.

It automatically generates a detailed trade-by-trade CSV file located in a dynamically generated `exports/` folder structure based on the strategy group, mode, and parameter configuration (e.g., `exports/extreme/macro/bull_tp.../backtest_...csv`). This CSV includes entry/exit timestamps, reasons for exit, slippage accounting, and indicator values at the time of entry.
