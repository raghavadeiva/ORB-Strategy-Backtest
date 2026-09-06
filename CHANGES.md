# CHANGES.md

## Overview
This file documents the commit history and preparation of the Backtest_with_data_trial_run copy repository for GitHub.

## Repository Purpose
This is a comprehensive backtesting system for leveraged ETFs using a momentum-based breakout strategy with a 15-minute Opening Range Breakout (ORB) timing mechanism.

## Files Included
- **backtest.py**: The main backtesting engine for 3x leveraged ETFs
- **compute_sharpe.py**: Sharpe ratio analysis for backtester export CSVs
- **direction_analyzer.py**: Analyze bull vs bear trades in CSV logs
- **log_parser.py**: Deep performance metrics for trade logs
- **master_strategist_runner.py**: Asynchronous grid search runner for backtesting
- **sharpe_analysis.py**: Comprehensive Sharpe ratio analysis for backtester strategy
- **trade_analyzer.py**: Batch analysis of detailed trade logs
- **README.md**: Project documentation and setup instructions

## File Organization
- **data_cache/**: 1-minute historical pricing data from Alpaca API (311MB total, excluded from git)
- **exports/**: Generated backtest results and grid search artifacts (102MB total, excluded from git)
- **.gitignore**: Git ignore file to exclude large files and generated artifacts

## Commit History

### Initial Commit (main branch)
- **Commit ID**: f6c3351
- **Date**: 2026-09-06
- **Author**: openhands <openhands@all-hands.dev>

**What Changed:**
- Created initial repository structure with all core backtesting files
- Added comprehensive .gitignore to exclude large data files and generated artifacts
- Added detailed README.md with project documentation

**Key Features Added:**
1. **Master Leveraged Trading Engine** (`backtest.py`): Sophisticated intraday algorithmic backtesting engine for trading 3x leveraged ETFs using momentum-based breakout strategy

2. **Sharpe Ratio Analysis Tools** (`compute_sharpe.py`, `sharpe_analysis.py`): Comprehensive analysis tools for evaluating backtest performance metrics

3. **Trade Analysis Tools** (`direction_analyzer.py`, `log_parser.py`, `trade_analyzer.py`): Advanced tools for analyzing trade performance and directional bias

4. **Grid Search Runner** (`master_strategist_runner.py`): Asynchronous runner for systematic parameter optimization

## Setup Instructions

### Prerequisites
- Python 3.x
- Required packages: pandas, numpy, alpaca-py, pyarrow
- Valid Alpaca API credentials (environment variables: ALPACA_API_KEY, ALPACA_SECRET_KEY)

### Running Backtests
```bash
python backtest.py --group extreme --days 30 --bull_tp 8.0 --bull_sl 3.0 --bear_tp 5.0 --bear_sl 2.0
```

### Running Analysis Tools
Analysis tools can be run with the following general patterns:
- `python compute_sharpe.py` (requires grid_search_summary.csv in /Users/akdeiva/backtest)
- `python direction_analyzer.py [path_to_csv]`
- `python log_parser.py [path_to_csv]`
- `python master_strategist_runner.py --workers 4`
- `python sharpe_analysis.py`
- `python trade_analyzer.py [folder_path]`

## Notes
- **Large Files Excluded**: data_cache/*.parquet and exports/**/*.csv are excluded to keep repository size manageable
- **Environment Variables**: API credentials should be stored in .env file or environment variables
- **Generated Artifacts**: All generated output files (CSVs, logs) are excluded from version control
- **Continuous Integration**: Repository ready for CI/CD pipeline integration

## Next Steps
1. Configure GitHub repository and push to remote
2. Set up CI/CD pipeline for automated testing and deployment
3. Configure monitoring for backtest runs and performance metrics
4. Document additional analysis scripts and usage patterns

## Repository Statistics
- **Total Files**: 7 source files + README.md + .gitignore
- **Excluded Data**: ~413MB (data_cache: 311MB, exports: 102MB)
- **Git Size**: ~2.4KB (source code only)
- **Commit Count**: 1 initial commit

## Version Control Notes
- Main branch: main
- Git LFS not configured (data files excluded via .gitignore)
- Ready for GitHub Actions CI/CD integration