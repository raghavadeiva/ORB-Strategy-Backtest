"""
Compute Sharpe Ratio from Backtester Export CSV

The backtester outputs a Running_Capital column at each trade's entry.
We reconstruct a daily (or per-trade) equity curve from this.

Approach:
1. Load trade log CSVs
2. Reconstruct equity curve at each trade entry timestamp
3. Compute returns (per-trade or daily)
4. Apply Sharpe ratio formula: (mean_return - risk_free_rate) / std_return

Sharpe Ratio = (mean_return - rf_rate) / std_return
- mean_return: average of returns per period (daily, here)
- rf_rate: risk-free rate (assume 0 for simplicity, or use T-bill rate)
- std_return: standard deviation of returns (sample std)

We'll compute both per-trade and daily versions.
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

BACKTESTER_DIR = "/Users/akdeiva/backtest"
EXPORTS_DIR = os.path.join(BACKTESTER_DIR, "exports")

# Risk-free rate (annual); we'll convert to daily
# Using 4.5% annually (~10-year Treasury yield as of late 2024)
RISK_FREE_RATE_ANNUAL = 0.045
RISK_FREE_RATE_DAILY = (1 + RISK_FREE_RATE_ANNUAL) ** (1/252) - 1

def find_best_exports(num_files=5):
    """Find the top export files from grid search results."""
    # Read the grid search summary to find top performers
    summary_path = os.path.join(BACKTESTER_DIR, "grid_search_summary.csv")
    summary = pd.read_csv(summary_path)
    
    # Sort by Net PnL descending
    summary_sorted = summary.sort_values('Net_PnL', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"TOP 10 CONFIGURATIONS BY NET P&L")
    print(f"{'='*80}")
    print(summary_sorted[['Bull_TP', 'Bull_SL', 'Bear_TP', 'Bear_SL', 'Cutoff', 
                          'Bull_ADX', 'Bear_ADX', 'Use_CB', 'Use_RS', 
                          'Total_Trades', 'Win_Rate_Pct', 'Net_PnL', 'Final_Capital']].head(10).to_string())
    
    return summary_sorted.head(num_files)


def reconstruct_equity_curve(csv_path):
    """
    Load a single export CSV and reconstruct the equity curve.
    
    The CSV has:
    - Date: trade date (e.g., "2020-11-04")
    - Entry_Time: entry time (e.g., "10:04:00")
    - Exit_Time: exit time (e.g., "15:58:00")
    - Running_Capital: capital at entry time
    - PnL_USD: dollar P&L for this trade
    - PnL_Pct: percentage return for this trade
    
    We reconstruct:
    1. Per-trade returns (PnL_Pct)
    2. Daily returns (sum of PnL_USD / starting capital of day)
    """
    df = pd.read_csv(csv_path)
    
    # Parse datetime
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Entry_Time'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    # ---- Per-trade returns ----
    per_trade_returns = df['PnL_Pct'].values / 100.0  # convert % to decimal
    
    # ---- Daily returns ----
    # Group by date, sum PnL_USD, divide by starting capital for that day
    df['TradingDate'] = pd.to_datetime(df['Date'])
    daily_groups = df.groupby('TradingDate').agg({
        'PnL_USD': 'sum',
        'Running_Capital': 'min'  # capital at start of day (first trade's entry capital)
    }).reset_index()
    
    # Daily return = daily PnL / day starting capital
    daily_returns = (daily_groups['PnL_USD'] / daily_groups['Running_Capital']).values
    
    return df, per_trade_returns, daily_returns


def compute_sharpe(returns, rf_rate=0.0, periods_per_year=252):
    """
    Compute Sharpe ratio.
    
    Sharpe = (mean_return - rf_rate_per_period) / std_return
    Annualized Sharpe = Sharpe * sqrt(periods_per_year)
    
    Here we compute both per-period and annualized.
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    
    if len(returns) < 2:
        return None, None, None
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # sample std
    
    if std_return == 0:
        return None, None, None
    
    sharpe_raw = (mean_return - rf_rate) / std_return
    sharpe_annualized = sharpe_raw * np.sqrt(periods_per_year)
    
    return sharpe_annualized, mean_return * periods_per_year * 100, std_return * np.sqrt(periods_per_year) * 100


def compute_sortino(returns, rf_rate=0.0, periods_per_year=252):
    """
    Compute Sortino ratio (downside deviation only).
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    
    if len(returns) < 2:
        return None
    
    mean_return = np.mean(returns)
    
    # Downside deviation: std of returns below 0 (or below target)
    downside_returns = returns[returns < rf_rate]
    if len(downside_returns) == 0:
        return None
    
    downside_std = np.std(downside_returns, ddof=1)
    if downside_std == 0:
        return None
    
    sortino = (mean_return - rf_rate) / downside_std
    sortino_annualized = sortino * np.sqrt(periods_per_year)
    
    return sortino_annualized


def compute_max_drawdown(df):
    """Compute max drawdown from the Running_Capital column."""
    capital = df['Running_Capital'].values
    if len(capital) == 0:
        return 0.0
    
    # Running peak
    peaks = np.maximum.accumulate(capital)
    drawdowns = (capital - peaks) / peaks
    max_drawdown = np.min(drawdowns)  # most negative = max drawdown
    
    return max_drawdown


def compute_calmar(returns, df, periods_per_year=252):
    """Calmar ratio = annual return / max drawdown."""
    sharpe_annual, annual_return_pct, annual_vol_pct = compute_sharpe(
        returns, rf_rate=RISK_FREE_RATE_DAILY, periods_per_year=periods_per_year
    )
    
    max_dd = compute_max_drawdown(df)
    if max_dd == 0 or max_dd is None or abs(max_dd) < 1e-10:
        return None
    
    # Annual return as decimal
    annual_return = annual_return_pct / 100.0
    calmar = annual_return / abs(max_dd)
    return calmar


def analyze_export(csv_path, label=""):
    """Full analysis of a single export file."""
    df, per_trade_returns, daily_returns = reconstruct_equity_curve(csv_path)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {label or os.path.basename(csv_path)}")
    print(f"{'='*80}")
    print(f"Starting Capital: ${df['Running_Capital'].iloc[0]:,.2f}")
    print(f"Ending Capital:   ${df['Running_Capital'].iloc[-1]:,.2f}")
    print(f"Total Trades:     {len(df)}")
    print(f"Net PnL:          ${df['PnL_USD'].sum():,.2f}")
    print(f"Return (total):   {((df['Running_Capital'].iloc[-1] / df['Running_Capital'].iloc[0]) - 1) * 100:.1f}%")
    print(f"Trading Days:     {df['TradingDate'].nunique()}")
    print(f"Date Range:       {df['TradingDate'].min().date()} to {df['TradingDate'].max().date()}")
    
    # Per-trade Sharpe
    sharpe_trade, annual_ret_trade, annual_vol_trade = compute_sharpe(
        per_trade_returns, rf_rate=RISK_FREE_RATE_DAILY, periods_per_year=len(per_trade_returns)
    )
    sortino_trade = compute_sortino(per_trade_returns, rf_rate=RISK_FREE_RATE_DAILY, periods_per_year=len(per_trade_returns))
    
    print(f"\n--- PER-TRADE METRICS ---")
    print(f"Win Rate:         {len(df[df['PnL_Pct'] > 0]) / len(df) * 100:.1f}%")
    print(f"Avg Win:          ${df[df['PnL_USD'] > 0]['PnL_USD'].mean():,.2f}")
    print(f"Avg Loss:         ${df[df['PnL_USD'] <= 0]['PnL_USD'].mean():,.2f}")
    print(f"Profit Factor:    {abs(df[df['PnL_USD'] > 0]['PnL_USD'].sum() / df[df['PnL_USD'] <= 0]['PnL_USD'].sum()):.2f}")
    if sharpe_trade:
        print(f"Sharpe (per-trade, annualized): {sharpe_trade:.2f}")
    if sortino_trade:
        print(f"Sortino (per-trade, annualized): {sortino_trade:.2f}")
    
    # Daily Sharpe
    sharpe_daily, annual_ret_daily, annual_vol_daily = compute_sharpe(
        daily_returns, rf_rate=RISK_FREE_RATE_DAILY, periods_per_year=252
    )
    sortino_daily = compute_sortino(daily_returns, rf_rate=RISK_FREE_RATE_DAILY, periods_per_year=252)
    calmar_daily = compute_calmar(daily_returns, df, periods_per_year=252)
    
    print(f"\n--- DAILY METRICS ---")
    print(f"Annualized Return: {annual_ret_daily:.1f}%/yr")
    print(f"Annualized Vol:    {annual_vol_daily:.1f}%/yr")
    if sharpe_daily:
        print(f"Sharpe (daily, annualized):    {sharpe_daily:.2f}")
    if sortino_daily:
        print(f"Sortino (daily, annualized):   {sortino_daily:.2f}")
    if calmar_daily:
        print(f"Calmar (daily, annualized):    {calmar_daily:.2f}")
    
    max_dd = compute_max_drawdown(df)
    print(f"Max Drawdown:     {max_dd * 100:.1f}%")
    
    win_rate = len(df[df['PnL_Pct'] > 0]) / len(df) * 100 if len(df) > 0 else 0
    print(f"\n--- KEY RATIOS SUMMARY ---")
    print(f"  Sharpe (daily, annualized):    {sharpe_daily:.2f}" if sharpe_daily else "  Sharpe (daily): N/A")
    print(f"  Sortino (daily, annualized):   {sortino_daily:.2f}" if sortino_daily else "  Sortino (daily): N/A")
    print(f"  Calmar (daily, annualized):    {calmar_daily:.2f}" if calmar_daily else "  Calmar (daily): N/A")
    print(f"  Max Drawdown:                  {max_dd*100:.1f}%")
    print(f"  Win Rate:                      {win_rate:.1f}%")
    
    return {
        'sharpe_daily': sharpe_daily,
        'sharpe_trade': sharpe_trade,
        'sortino_daily': sortino_daily,
        'calmar_daily': calmar_daily,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'total_trades': len(df),
        'net_pnl': df['PnL_USD'].sum(),
        'final_capital': df['Running_Capital'].iloc[-1],
        'starting_capital': df['Running_Capital'].iloc[0],
        'annual_return': annual_ret_daily if annual_ret_daily else 0,
        'annual_volatility': annual_vol_daily if annual_vol_daily else 0,
    }


def main():
    """Main execution: analyze top-performing and representative configs."""
    
    # Step 1: Read grid search summary
    summary_df = find_best_exports(num_files=5)
    
    if summary_df.empty:
        print("No grid search summary found!")
        return
    
    # Step 2: Find and analyze export files for the top configurations
    results = []
    
    # Top performer
    top_row = summary_df.iloc[0]
    print(f"\n\n>>> TOP PERFORMER CONFIGURATION <<<")
    print(f"  Bull TP: {top_row['Bull_TP']}, Bull SL: {top_row['Bull_SL']}")
    print(f"  Bear TP: {top_row['Bear_TP']}, Bear SL: {top_row['Bear_SL']}")
    print(f"  Cutoff: {top_row['Cutoff']}, Use_CB: {top_row['Use_CB']}, Use_RS: {top_row['Use_RS']}")
    print(f"  Total Trades: {top_row['Total_Trades']}, Win Rate: {top_row['Win_Rate_Pct']}%")
    print(f"  Net PnL: ${top_row['Net_PnL']:,.2f}, Final Capital: ${top_row['Final_Capital']:,.2f}")
    
    # Search for export files matching the top config
    # The directory name encodes: bull_tp{tp}_sl{sl}_adx{adx}_orb{orb}_co{cutoff}
    bull_tp_str = str(top_row['Bull_TP']).replace(',', '_').replace('"', '')
    bull_sl_str = str(top_row['Bull_SL']).replace(',', '_').replace('"', '')
    
    # Try to find matching export directory
    pattern = f"*/bull_tp*{bull_tp_str}*sl*{bull_sl_str}*co{str(top_row['Cutoff']).replace(':', '')}/*"
    search_path = os.path.join(EXPORTS_DIR, "extreme/macro", pattern)
    matching_files = glob.glob(search_path, recursive=True)
    
    if matching_files:
        best_result = analyze_export(matching_files[0], "TOP PERFORMER")
        results.append(best_result)
    else:
        print(f"\n  (Could not find export file for exact top config, searching alternatives...)")
        
        # Try a different approach: find the highest-performing export file directly
        all_csvs = glob.glob(os.path.join(EXPORTS_DIR, "extreme/macro/**/*.csv"), recursive=True)
        if all_csvs:
            # Pick a few representative files to analyze
            for csv_file in all_csvs[:3]:
                result = analyze_export(csv_file, csv_file.split('/')[-2])
                results.append(result)
    
    # Step 3: Analyze a representative sample from grid search
    print(f"\n\n>>> ANALYZING ADDITIONAL CONFIGS FROM GRID SEARCH <<<")
    
    # Find files for different configurations
    all_config_dirs = sorted(glob.glob(os.path.join(EXPORTS_DIR, "extreme/macro/*")))
    print(f"Found {len(all_config_dirs)} configuration directories in exports/extreme/macro/")
    
    # Sample 3 more for comparison
    additional_results = []
    for dir_path in all_config_dirs[:5]:
        csvs = glob.glob(os.path.join(dir_path, "*.csv"))
        if csvs:
            result = analyze_export(csvs[0], os.path.basename(dir_path))
            additional_results.append(result)
    
    # Summary table
    print(f"\n\n{'='*80}")
    print(f"SHARPE RATIO COMPARISON ACROSS CONFIGS")
    print(f"{'='*80}")
    
    all_results = results + additional_results
    for r in all_results:
        if r:
            print(f"  Config: {r.get('config_name', 'N/A')}")
            print(f"    Sharpe (daily):   {r['sharpe_daily']:.2f}" if r['sharpe_daily'] else "    Sharpe (daily): N/A")
            print(f"    Sortino (daily):  {r['sortino_daily']:.2f}" if r['sortino_daily'] else "    Sortino (daily): N/A")
            print(f"    Calmar (daily):   {r['calmar_daily']:.2f}" if r['calmar_daily'] else "    Calmar (daily): N/A")
            print(f"    Max Drawdown:    {r['max_drawdown']*100:.1f}%")
            print(f"    Win Rate:        {r['win_rate']:.1f}%")
            print(f"    Trades:          {r['total_trades']}")
            print(f"    Net PnL:         ${r['net_pnl']:,.2f}")
            print(f"    Final Capital:   ${r['final_capital']:,.2f}")
            print()


if __name__ == "__main__":
    main()
