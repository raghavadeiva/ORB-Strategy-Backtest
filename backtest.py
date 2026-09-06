import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Expanded Master Mapping of Base Indices to their Leveraged Equivalents
ETF_MAPPING = {
    # --- BROAD MARKET ---
    'SPY':  {'bull': 'SPXL', 'bear': 'SPXS', 'leverage': 3, 'volatility': 'low'},
    'DIA':  {'bull': 'UDOW', 'bear': 'SDOW', 'leverage': 3, 'volatility': 'low'},

    # --- HIGH BETA SECTORS (The Best Targets) ---
    'SOXX': {'bull': 'SOXL', 'bear': 'SOXS', 'leverage': 3, 'volatility': 'extreme'},
    'XLK':  {'bull': 'TECL', 'bear': 'TECS', 'leverage': 3, 'volatility': 'extreme'},

    'XBI':  {'bull': 'LABU', 'bear': 'LABD', 'leverage': 3, 'volatility': 'high'},
    'IWM':  {'bull': 'TNA',  'bear': 'TZA',  'leverage': 3, 'volatility': 'high'},
    'XRT':  {'bull': 'RETL', 'bear': 'RETS', 'leverage': 3, 'volatility': 'high'},

    # --- MEDIUM BETA SECTORS ---
    'QQQ':  {'bull': 'TQQQ', 'bear': 'SQQQ', 'leverage': 3, 'volatility': 'medium'},
    'XLF':  {'bull': 'FAS',  'bear': 'FAZ',  'leverage': 3, 'volatility': 'medium'},
    'XLRE': {'bull': 'DRN',  'bear': 'DRV',  'leverage': 3, 'volatility': 'medium'}
}

SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLY', 'XLC', 'XLI', 'XLP', 'XLU', 'XLE', 'XLB', 'XLRE']
DATA_DIR = "./data_cache"

def parse_ticker_param(param_val, target_ticker):
    """Safely parses strings like '8.0,XLK:6' to return the correct float for the ticker."""
    parts = str(param_val).split(',')
    try:
        val = float(parts[0]) 
    except ValueError:
        val = 0.0
    for part in parts[1:]:
        if ':' in part:
            t, v = part.split(':')
            if t.strip().upper() == target_ticker.strip().upper():
                try:
                    val = float(v)
                except ValueError:
                    pass
    return val

def fetch_alpaca_data(ticker, client, start_date, end_date):
    """Fetches 1-minute data from Alpaca, caches it locally, and stitches missing data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{ticker}_1min.parquet")
    
    start_ts = pd.Timestamp(start_date)
    if start_ts.tz is None: start_ts = start_ts.tz_localize('America/New_York')
    end_ts = pd.Timestamp(end_date)
    if end_ts.tz is None: end_ts = end_ts.tz_localize('America/New_York')
    
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
    
        if not df.empty:
            cache_start = df.index.min()
            cache_end = df.index.max()

            if cache_end < end_ts:
                try:
                    request = StockBarsRequest(
                        symbol_or_symbols=[ticker], timeframe=TimeFrame.Minute,
                        start=cache_end.to_pydatetime(), end=end_date,
                        feed=DataFeed.SIP  # <--- ADDED FEED PARAMETER
                    )
                    bars = client.get_stock_bars(request)
                    if not bars.df.empty:
                        delta_df = bars.df.loc[ticker].copy()
                        delta_df.index = delta_df.index.tz_convert('America/New_York')
                        delta_df = delta_df.between_time('09:30', '15:59')
                        df = pd.concat([df, delta_df])
                        df = df[~df.index.duplicated(keep='last')]
                        df.sort_index(inplace=True)
                        df.to_parquet(file_path)
                except Exception as e:
                    print(f"[!] Alpaca API Error stitching forward data for {ticker}: {e}")
                    pass
            
            if cache_start > start_ts + pd.Timedelta(days=1):
                try:
                    request = StockBarsRequest(
                        symbol_or_symbols=[ticker], timeframe=TimeFrame.Minute,
                        start=start_date, end=cache_start.to_pydatetime(),
                        feed=DataFeed.SIP  # <--- ADDED FEED PARAMETER
                    )
                    bars = client.get_stock_bars(request)
                    if not bars.df.empty:
                        hist_df = bars.df.loc[ticker].copy()
                        hist_df.index = hist_df.index.tz_convert('America/New_York')
                        hist_df = hist_df.between_time('09:30', '15:59')
                        df = pd.concat([hist_df, df])
                        df = df[~df.index.duplicated(keep='last')]
                        df.sort_index(inplace=True)
                        df.to_parquet(file_path)
                except Exception as e:
                    print(f"[!] Alpaca API Error stitching backward data for {ticker}: {e}")
                    pass

            return df[(df.index >= start_ts) & (df.index <= end_ts)]


    try:
        request = StockBarsRequest(
            symbol_or_symbols=[ticker], 
            timeframe=TimeFrame.Minute, 
            start=start_date, 
            end=end_date,
            feed=DataFeed.SIP  # <--- ADDED FEED PARAMETER
        )
        bars = client.get_stock_bars(request)
        if bars.df.empty: return pd.DataFrame()
            
        df = bars.df.loc[ticker].copy()
        df.index = df.index.tz_convert('America/New_York')
        df = df.between_time('09:30', '15:59')
        df.to_parquet(file_path)
        return df[(df.index >= start_ts) & (df.index <= end_ts)]
    except Exception as e:
        print(f"[!] Alpaca API Error fetching full data for {ticker}: {e}")
        return pd.DataFrame()

def build_synthetic_internals(client, start_date, end_date):
    """Builds a Synthetic Intraday Advance/Decline Line using 11 SPDR sectors."""
    sector_dfs = {}
    for sector in SECTOR_ETFS:
        df = fetch_alpaca_data(sector, client, start_date, end_date)
        if not df.empty: sector_dfs[sector] = df['close']
            
    if not sector_dfs: return pd.Series(dtype=float)
        
    sectors_close = pd.DataFrame(sector_dfs)
    changes = sectors_close.pct_change(fill_method=None)
    advances = (changes > 0).sum(axis=1)
    declines = (changes < 0).sum(axis=1)
    net_advances = advances - declines
    return net_advances.groupby(net_advances.index.date).cumsum()

def build_and_calculate_indicators(signal, bull, bear, client, fetch_start_date, target_start_date, end_date, is_macro):
    """Calculates all underlying metrics, utilizing the fetch_start_date to warm up the EMA and ADX."""
    print(f"[{signal}] Downloading data and mapping indicators...")
    sig_df = fetch_alpaca_data(signal, client, fetch_start_date, end_date)
    bull_df = fetch_alpaca_data(bull, client, fetch_start_date, end_date)
    bear_df = fetch_alpaca_data(bear, client, fetch_start_date, end_date)
    
    # --- FETCH VIX PROXY FOR FEAR INDEX ---
    vix_df = fetch_alpaca_data('VIXY', client, fetch_start_date, end_date)
    
    if sig_df.empty or bull_df.empty or bear_df.empty:
        raise ValueError(f"Missing core ETF data for {signal}.")
        
    df = pd.DataFrame(index=sig_df.index)
    df['Date'] = df.index.date
    
    df[f'{signal}_Open'], df[f'{signal}_High'], df[f'{signal}_Low'], df[f'{signal}_Close'], df[f'{signal}_Volume'] = sig_df['open'], sig_df['high'], sig_df['low'], sig_df['close'], sig_df['volume']
    df[f'{bull}_Open'], df[f'{bull}_High'], df[f'{bull}_Low'], df[f'{bull}_Close'], df[f'{bull}_Volume'] = bull_df['open'], bull_df['high'], bull_df['low'], bull_df['close'], bull_df['volume']
    df[f'{bear}_Open'], df[f'{bear}_High'], df[f'{bear}_Low'], df[f'{bear}_Close'], df[f'{bear}_Volume'] = bear_df['open'], bear_df['high'], bear_df['low'], bear_df['close'], bear_df['volume']
    
    if not vix_df.empty:
        df['VIX_Open'] = vix_df['open']
        df['VIX_Close'] = vix_df['close']
    else:
        df['VIX_Open'] = np.nan
        df['VIX_Close'] = np.nan
        
    df = df.dropna(subset=[f'{signal}_Close', f'{bull}_Close', f'{bear}_Close'])
    
    # 1. VWAP
    df['Typical_Price'] = (df[f'{signal}_High'] + df[f'{signal}_Low'] + df[f'{signal}_Close']) / 3
    df['Cum_VP'] = (df['Typical_Price'] * df[f'{signal}_Volume']).groupby(df['Date']).cumsum()
    df['Cum_Vol'] = df.groupby('Date')[f'{signal}_Volume'].cumsum()
    df[f'{signal}_VWAP'] = np.where(df['Cum_Vol'] == 0, df['Typical_Price'], df['Cum_VP'] / df['Cum_Vol'])
    
    for muscle in [bull, bear]:
        df[f'{muscle}_Typical_Price'] = (df[f'{muscle}_High'] + df[f'{muscle}_Low'] + df[f'{muscle}_Close']) / 3
        df[f'{muscle}_Cum_VP'] = (df[f'{muscle}_Typical_Price'] * df[f'{muscle}_Volume']).groupby(df['Date']).cumsum()
        df[f'{muscle}_Cum_Vol'] = df.groupby('Date')[f'{muscle}_Volume'].cumsum()
        df[f'{muscle}_VWAP'] = np.where(df[f'{muscle}_Cum_Vol'] == 0, df[f'{muscle}_Typical_Price'], df[f'{muscle}_Cum_VP'] / df[f'{muscle}_Cum_Vol'])

    # 2. ORB (9:30 - 9:44)
    orb_window = df.between_time('09:30', '09:44')
    df = df.join(orb_window.groupby('Date')[f'{signal}_High'].max().rename(f'{signal}_ORB_High'), on='Date')
    df = df.join(orb_window.groupby('Date')[f'{signal}_Low'].min().rename(f'{signal}_ORB_Low'), on='Date')
    df[f'{signal}_ORB_High'] = df.groupby('Date')[f'{signal}_ORB_High'].ffill()
    df[f'{signal}_ORB_Low'] = df.groupby('Date')[f'{signal}_ORB_Low'].ffill()
    
    # Relative Strength
    orb_opens = orb_window.groupby('Date')[f'{signal}_Open'].first().rename('ORB_Open')
    orb_closes = orb_window.groupby('Date')[f'{signal}_Close'].last().rename('ORB_Close')
    df = df.join(orb_opens, on='Date').join(orb_closes, on='Date')
    df['ORB_RS'] = (df['ORB_Close'] - df['ORB_Open']) / df['ORB_Open']
    
    # 3. Macro Context & Leading Indicators
    if is_macro:
        ad_line = build_synthetic_internals(client, fetch_start_date, end_date)
        df['ADD_Close'] = ad_line if not ad_line.empty else np.nan
        
        # --- LEADING INDICATORS: DELTA & SURGE ---
        df['ADD_Delta_15m'] = df['ADD_Close'].diff(15)
        
        df['VIX_Day_Open'] = df.groupby('Date')['VIX_Open'].transform('first')
        df['VIX_Surge_Pct'] = ((df['VIX_Close'] - df['VIX_Day_Open']) / df['VIX_Day_Open']) * 100
        
        daily_df = df.groupby('Date').agg({f'{signal}_High': 'max', f'{signal}_Low': 'min', f'{signal}_Close': 'last'}).dropna()
        daily_df['9EMA'] = daily_df[f'{signal}_Close'].ewm(span=9, adjust=False).mean()
        
        daily_df['Prior_High'] = daily_df[f'{signal}_High'].shift(1)
        daily_df['Prior_Low'] = daily_df[f'{signal}_Low'].shift(1)
        daily_df['Prior_9EMA'] = daily_df['9EMA'].shift(1)
        
        # ADX Calculation
        prior_close = daily_df[f'{signal}_Close'].shift(1)
        tr1 = daily_df[f'{signal}_High'] - daily_df[f'{signal}_Low']
        tr2 = (daily_df[f'{signal}_High'] - prior_close).abs()
        tr3 = (daily_df[f'{signal}_Low'] - prior_close).abs()
        daily_df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_m = daily_df[f'{signal}_High'] - daily_df['Prior_High']
        dn_m = daily_df['Prior_Low'] - daily_df[f'{signal}_Low']
        
        daily_df['+DM'] = np.where((up_m > dn_m) & (up_m > 0), up_m, 0)
        daily_df['-DM'] = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0)
        
        daily_df['TR_smooth'] = daily_df['TR'].ewm(alpha=1/14, adjust=False).mean()
        daily_df['+DM_smooth'] = daily_df['+DM'].ewm(alpha=1/14, adjust=False).mean()
        daily_df['-DM_smooth'] = daily_df['-DM'].ewm(alpha=1/14, adjust=False).mean()
        
        daily_df['+DI'] = 100 * (daily_df['+DM_smooth'] / daily_df['TR_smooth'])
        daily_df['-DI'] = 100 * (daily_df['-DM_smooth'] / daily_df['TR_smooth'])
        daily_df['DX'] = 100 * abs(daily_df['+DI'] - daily_df['-DI']) / (daily_df['+DI'] + daily_df['-DI'])
        daily_df['ADX'] = daily_df['DX'].ewm(alpha=1/14, adjust=False).mean()
        
        daily_df['Prior_ADX'] = daily_df['ADX'].shift(1)
        
        va_highs, va_lows = {}, {}
        for date, day_data in df.groupby('Date'):
            bins = np.round(day_data['Typical_Price'], 1)
            profile = day_data.groupby(bins)[f'{signal}_Volume'].sum().sort_values(ascending=False)
            if not profile.empty:
                va_bins = profile[profile.cumsum() <= profile.sum() * 0.70].index
                if len(va_bins) > 0:
                    va_highs[date] = va_bins.max()
                    va_lows[date] = va_bins.min()
                
        daily_df['VA_High'] = pd.Series(va_highs).shift(1)
        daily_df['VA_Low'] = pd.Series(va_lows).shift(1)
        df = df.join(daily_df[['Prior_High', 'Prior_Low', 'Prior_9EMA', 'Prior_ADX', 'VA_High', 'VA_Low']], on='Date')
    
    # --- SLICE TO TARGET START DATE TO EXCLUDE WARM-UP ---
    target_ts = pd.Timestamp(target_start_date)
    if target_ts.tz is None: target_ts = target_ts.tz_localize('America/New_York')
    df = df[df.index >= target_ts]
    
    return df

def execute_engine(df, signal, bull, bear, bull_tp, bull_sl, bear_tp, bear_sl, mode, cutoff_time, use_slippage, use_scale_out, bull_adx, bear_adx, bull_add, bear_add, bull_add_delta, bear_add_delta, bull_vix_limit, bear_vix_limit, orb_min, orb_max, use_trail, no_lagging, no_leading):
    print(f"[{signal}] Generating Theoretical Trades...")
    trade_log = []
    is_macro = (mode == 'macro')
    
    for date, day_data in df.groupby('Date'):
        if is_macro and pd.isna(day_data['Prior_High'].iloc[0]): continue 
        
        orb_rs_val = day_data['ORB_RS'].iloc[0] if not pd.isna(day_data['ORB_RS'].iloc[0]) else 0.0

        states = {
            'BULL': {'in_trade': False, 'ticker': bull, 'entry': 0, 'mfe': 0, 'mae': 0, 'time': None, 'scaled_out': False, 'trail_active': False, 'traded': False, 'vwap': 0, 'orb_w': 0, 'adx': 0, 'add': 0, 'add_delta': 0, 'vix': 0, 'entry_open': 0, 'entry_high': 0, 'entry_low': 0, 'entry_close': 0, 'orb_high': 0, 'orb_low': 0},
            'BEAR': {'in_trade': False, 'ticker': bear, 'entry': 0, 'mfe': 0, 'mae': 0, 'time': None, 'scaled_out': False, 'trail_active': False, 'traded': False, 'vwap': 0, 'orb_w': 0, 'adx': 0, 'add': 0, 'add_delta': 0, 'vix': 0, 'entry_open': 0, 'entry_high': 0, 'entry_low': 0, 'entry_close': 0, 'orb_high': 0, 'orb_low': 0}
        }
        
        trading_session = day_data.between_time('09:45', '15:59')
        
        for timestamp, row in trading_session.iterrows():
            
            # --- ENTRY LOGIC ---
            if timestamp.time() <= pd.to_datetime(cutoff_time).time():
                orb_width = (row[f'{signal}_ORB_High'] - row[f'{signal}_ORB_Low']) / row[f'{signal}_ORB_Low']
                adx_val = row.get('Prior_ADX', np.nan)
                
                add_val = row.get('ADD_Close', np.nan)
                add_delta = row.get('ADD_Delta_15m', 0.0)
                vix_surge = row.get('VIX_Surge_Pct', 0.0)
                vwap_val = row[f'{signal}_VWAP']
                
                # Bullish Breakout
                if not states['BULL']['in_trade'] and not states['BULL']['traded']:
                    trigger = (row[f'{signal}_Close'] > row[f'{signal}_ORB_High']) and (row[f'{signal}_Close'] > vwap_val)
                    if is_macro:
                        trigger = trigger and (orb_min <= orb_width <= orb_max)
                        trigger = trigger and (row[f'{signal}_Close'] > row['VA_Low'])
                        trigger = trigger and not (0 < (row['Prior_High'] - row[f'{signal}_ORB_High']) / row[f'{signal}_ORB_High'] < 0.0015)
                        
                        # STRUCTURAL BREADTH (Always Active)
                        trigger = trigger and (add_val > bull_add if not pd.isna(add_val) else True)
                        
                        # --- SHORT CIRCUIT TOGGLES ---
                        if not no_lagging:
                            trigger = trigger and (adx_val >= bull_adx if not pd.isna(adx_val) else True)
                            trigger = trigger and (row[f'{signal}_Close'] > row['Prior_9EMA'])
                            
                        if not no_leading:
                            trigger = trigger and (add_delta >= bull_add_delta if not pd.isna(add_delta) else True)
                            trigger = trigger and (vix_surge < bull_vix_limit if not pd.isna(vix_surge) else True)
                        
                    if trigger:
                        states['BULL'].update({'in_trade': True, 'traded': True, 'entry': row[f'{bull}_Close'], 'mfe': row[f'{bull}_Close'], 'mae': row[f'{bull}_Close'], 'time': timestamp, 'vwap': vwap_val, 'orb_w': orb_width, 'adx': adx_val, 'add': add_val, 'add_delta': add_delta, 'vix': vix_surge, 'entry_open': row[f'{bull}_Open'], 'entry_high': row[f'{bull}_High'], 'entry_low': row[f'{bull}_Low'], 'entry_close': row[f'{bull}_Close'], 'orb_high': row[f'{signal}_ORB_High'], 'orb_low': row[f'{signal}_ORB_Low']})

                # Bearish Breakdown
                if not states['BEAR']['in_trade'] and not states['BEAR']['traded']:
                    trigger = (row[f'{signal}_Close'] < row[f'{signal}_ORB_Low']) and (row[f'{signal}_Close'] < vwap_val)
                    if is_macro:
                        trigger = trigger and (orb_min <= orb_width <= orb_max)
                        trigger = trigger and (row[f'{signal}_Close'] < row['VA_High'])
                        trigger = trigger and not (0 < (row[f'{signal}_ORB_Low'] - row['Prior_Low']) / row[f'{signal}_ORB_Low'] < 0.0015)
                        
                        # STRUCTURAL BREADTH (Always Active)
                        trigger = trigger and (add_val < bear_add if not pd.isna(add_val) else True)
                        
                        # --- SHORT CIRCUIT TOGGLES ---
                        if not no_lagging:
                            trigger = trigger and (adx_val >= bear_adx if not pd.isna(adx_val) else True)
                            trigger = trigger and (row[f'{signal}_Close'] < row['Prior_9EMA'])

                        if not no_leading:
                            trigger = trigger and (add_delta <= bear_add_delta if not pd.isna(add_delta) else True)
                            trigger = trigger and (vix_surge >= bear_vix_limit if not pd.isna(vix_surge) else True)

                    if trigger:
                        states['BEAR'].update({'in_trade': True, 'traded': True, 'entry': row[f'{bear}_Close'], 'mfe': row[f'{bear}_Close'], 'mae': row[f'{bear}_Close'], 'time': timestamp, 'vwap': vwap_val, 'orb_w': orb_width, 'adx': adx_val, 'add': add_val, 'add_delta': add_delta, 'vix': vix_surge, 'entry_open': row[f'{bear}_Open'], 'entry_high': row[f'{bear}_High'], 'entry_low': row[f'{bear}_Low'], 'entry_close': row[f'{bear}_Close'], 'orb_high': row[f'{signal}_ORB_High'], 'orb_low': row[f'{signal}_ORB_Low']})

            # --- MANAGEMENT LOGIC ---
            for side, st in states.items():
                if st['in_trade']:
                    ticker = st['ticker']
                    active_tp_pct = bull_tp if side == 'BULL' else bear_tp
                    active_sl_pct = bull_sl if side == 'BULL' else bear_sl
                    
                    curr_open, curr_high, curr_low, curr_close = row[f'{ticker}_Open'], row[f'{ticker}_High'], row[f'{ticker}_Low'], row[f'{ticker}_Close']
                    
                    st['mfe'] = max(st['mfe'], curr_high)
                    st['mae'] = min(st['mae'], curr_low)
                    
                    runup = (curr_high - st['entry']) / st['entry']
                    drawdown = (curr_low - st['entry']) / st['entry']
                    mfe_runup = (st['mfe'] - st['entry']) / st['entry']
                    
                    if use_scale_out and not st['scaled_out'] and mfe_runup >= active_sl_pct:
                        st['scaled_out'] = True
                        
                    if use_trail and not st['trail_active'] and mfe_runup >= (active_tp_pct * 0.50):
                        st['trail_active'] = True
                        
                    if use_trail and st['trail_active']:
                        trailed_stop = mfe_runup - active_sl_pct
                        effective_stop = max(0.001, trailed_stop)
                    elif use_scale_out and st['scaled_out']:
                        effective_stop = 0.001
                    else:
                        effective_stop = -active_sl_pct
                        
                    hit_target = False if use_trail else (runup >= active_tp_pct)
                    hit_stop = drawdown <= effective_stop
                    is_eod = timestamp.time() >= pd.to_datetime('15:58').time()
                    
                    if hit_target and hit_stop:
                        hit_target = False
                    
                    if hit_target or hit_stop or is_eod:
                        if hit_target:
                            reason, exit_price = "Target Hit", st['entry'] * (1 + active_tp_pct)
                            base_exit_pct = active_tp_pct
                        elif hit_stop:
                            if use_trail and st['trail_active']:
                                reason = "Trailing Stop (Slippage)" if use_slippage else "Trailing Stop"
                                exit_price = curr_low if use_slippage else st['entry'] * (1 + effective_stop)
                                base_exit_pct = (curr_low - st['entry']) / st['entry'] if use_slippage else effective_stop
                            elif st['scaled_out']:
                                reason = "BE Stop (Slippage)" if use_slippage else "BE Stop"
                                exit_price = curr_low if use_slippage else st['entry'] * 1.001
                                base_exit_pct = (curr_low - st['entry']) / st['entry'] if use_slippage else 0.001
                            else:
                                reason = "Stop Loss (Slippage)" if use_slippage else "Stop Loss"
                                exit_price = curr_low if use_slippage else st['entry'] * (1 - active_sl_pct)
                                base_exit_pct = (curr_low - st['entry']) / st['entry'] if use_slippage else -active_sl_pct
                        else:
                            reason, exit_price = "EOD Close", curr_close
                            base_exit_pct = (curr_close - st['entry']) / st['entry']
                        
                        if use_scale_out and st['scaled_out']:
                            final_pct = (0.33 * active_sl_pct) + (0.67 * base_exit_pct)
                            reason += " [Scaled 33%]"
                        else:
                            final_pct = base_exit_pct
                        
                        mfe_pct = (st['mfe'] - st['entry']) / st['entry'] * 100
                        mae_pct = (st['mae'] - st['entry']) / st['entry'] * 100
                        left_on_table = max(0.0, mfe_pct - (final_pct * 100)) if final_pct > 0 else 0.0
                        
                        trade_log.append({
                            'Date': date.strftime('%Y-%m-%d'),
                            'Underlying': signal,
                            'Signal': side,
                            'Traded': ticker,
                            'Entry_Time': st['time'].strftime('%H:%M:%S'),
                            'Exit_Time': timestamp.strftime('%H:%M:%S'),
                            'Reason': reason,
                            'Entry_Open': round(st['entry_open'], 2),
                            'Entry_High': round(st['entry_high'], 2),
                            'Entry_Low': round(st['entry_low'], 2),
                            'Entry_Close': round(st['entry_close'], 2),
                            'Exit_Open': round(curr_open, 2),
                            'Exit_High': round(curr_high, 2),
                            'Exit_Low': round(curr_low, 2),
                            'Exit_Close': round(curr_close, 2),
                            'Signal_ORB_High': round(st['orb_high'], 2),
                            'Signal_ORB_Low': round(st['orb_low'], 2),
                            'VWAP_Entry': round(st['vwap'], 2),
                            'VWAP_Exit': round(row[f'{signal}_VWAP'], 2),
                            'Entry_Price': round(st['entry'], 2),
                            'Exit_Price': round(exit_price, 2),
                            'PnL_Pct': round(final_pct * 100, 2),
                            'ORB_W_Pct': round(st['orb_w'] * 100, 2),
                            'ORB_RS': round(orb_rs_val * 100, 4),
                            'ADX_Entry': round(st['adx'], 1) if not pd.isna(st['adx']) else 0.0,
                            'ADD_Entry': round(st['add'], 0) if not pd.isna(st.get('add', 0)) else 0.0,
                            'ADD_Delta': round(st['add_delta'], 0) if not pd.isna(st['add_delta']) else 0.0,
                            'VIX_Surge': round(st['vix'], 2) if not pd.isna(st['vix']) else 0.0,
                            'MFE_Pct': round(mfe_pct, 2),
                            'MAE_Pct': round(mae_pct, 2),
                            'Missed_Profit_%': round(left_on_table, 2)
                        })
                        st['in_trade'] = False

    return pd.DataFrame(trade_log)

def main():
    parser = argparse.ArgumentParser(description="Master Leveraged Engine with Leading Indicators")
    parser.add_argument('--group', type=str, default='extreme', help='Volatility group (high, medium, low, extreme)')
    parser.add_argument('--mode', type=str, default='macro', choices=['macro', 'micro'], help='Trading Mode')
    parser.add_argument('--capital', type=float, default=10000.0, help='Starting Capital')
    
    # --- Date Range Arguments ---
    parser.add_argument('--start_date', type=str, default=None, help='Specific start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='Specific end date (YYYY-MM-DD)')
    
    parser.add_argument('--bull_tp', type=str, default='8.0,XLK:6', help='Bull Take Profit percent')
    parser.add_argument('--bull_sl', type=str, default='3.0,XLK:1.5', help='Bull Stop Loss percent')
    parser.add_argument('--bear_tp', type=str, default='5', help='Bear Take Profit percent')
    parser.add_argument('--bear_sl', type=str, default='2.0,XLK:1.5', help='Bear Stop Loss percent')
    parser.add_argument('--bull_adx', type=float, default=11.0, help='Minimum ADX threshold for Bull')
    parser.add_argument('--bear_adx', type=float, default=13.0, help='Minimum ADX threshold for Bear')
    
    parser.add_argument('--bull_add', type=float, default=0.0, help='Legacy static ADD threshold')
    parser.add_argument('--bear_add', type=float, default=0.0, help='Legacy static ADD threshold')
    
    # --- NEW: LEADING INDICATOR ARGUMENTS ---
    parser.add_argument('--bull_add_delta', type=float, default=9999.0, help='ADD must surge by +X stocks in 15 mins to go long')
    parser.add_argument('--bear_add_delta', type=float, default=-9999.0, help='ADD must plunge by -X stocks in 15 mins to short')
    parser.add_argument('--bull_vix_limit', type=float, default=9999.0, help='Max VIX surge allowed for Bull trades')
    parser.add_argument('--bear_vix_limit', type=float, default=-9999.0, help='Min VIX surge required for Bear trades')
    
    # --- SHORT CIRCUIT TOGGLES ---
    parser.add_argument('--no_lagging', action='store_true', help='Disable ADX and 9-EMA Checks')
    parser.add_argument('--no_leading', action='store_true', help='Disable VIX and ADD Delta Checks')
    
    parser.add_argument('--use_cb', action='store_true', help='Enable the consecutive loss Circuit Breaker logic')
    parser.add_argument('--max_streak', type=int, default=3, help='Max consecutive losses before pausing trading')
    parser.add_argument('--cooldown', type=int, default=5, help='Number of days to pause trading after circuit breaker trips')
    
    parser.add_argument('--use_rs_ranking', action='store_true', help='Rank assets by ORB momentum as a Priority Queue')
    
    parser.add_argument('--days', type=int, default=2500, help='Days to backtest (overridden by start_date/end_date)')
    parser.add_argument('--cutoff', type=str, default='10:30', help='Time cutoff')
    parser.add_argument('--max_concurrent', type=int, default=1, help='Max concurrent trades')
    parser.add_argument('--slippage', action='store_true', help='Apply real-world slippage')
    parser.add_argument('--scale_out', action='store_true', help='Apply 33 percent scale-out at 1R')
    parser.add_argument('--orb_min', type=float, default=0.0066, help='Minimum ORB width percentage')
    parser.add_argument('--orb_max', type=float, default=0.0166, help='Maximum ORB width percentage')
    parser.add_argument('--trail', action='store_true', help='Use dynamic trailing stop instead of fixed TP')
    parser.add_argument('--clear_cache', action='store_true', help='Wipe cache')
    args = parser.parse_args()
    
    if args.clear_cache and os.path.exists(DATA_DIR):
        import shutil
        shutil.rmtree(DATA_DIR)
            
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    client = StockHistoricalDataClient(api_key, api_secret) if api_key else None
    if not client: return print("Error: Alpaca API keys missing.")

    group_val = args.group.strip().lower()
    if group_val == 'high':
        signals = [k for k, v in ETF_MAPPING.items() if v['volatility'] == 'high']
    elif group_val == 'extreme':
        signals = [k for k, v in ETF_MAPPING.items() if v['volatility'] == 'extreme']
    elif group_val == 'medium':
        signals = [k for k, v in ETF_MAPPING.items() if v['volatility'] == 'medium']
    elif group_val == 'low':
        signals = [k for k, v in ETF_MAPPING.items() if v['volatility'] == 'low']
    else:
        signals = [t.strip().upper() for t in args.group.split(',')]

    # --- REGIME WARM-UP LOGIC ---
    if args.start_date and args.end_date:
        target_start_date = pd.to_datetime(args.start_date + ' 09:30:00').tz_localize('America/New_York')
        end_date = pd.to_datetime(args.end_date + ' 16:00:00').tz_localize('America/New_York')
        date_label = f"{args.start_date}_to_{args.end_date}"
        
        # Download 60 extra days to "warm up" the EMA and ADX calculations!
        fetch_start_date = target_start_date - timedelta(days=60)
        print(f"[*] Custom Regime: {target_start_date.date()} to {end_date.date()} (Pre-fetching 60 days to warm up indicators)")
    else:
        end_date = pd.to_datetime('2026-09-04 16:00:00').tz_localize('America/New_York')
        target_start_date = end_date - timedelta(days=args.days)
        # REVERT TO COLD START FOR STANDARD RUNS TO MATCH BASELINE
        fetch_start_date = target_start_date 
        date_label = f"{args.days}d"
    
    print("==================================================")
    print(f" MASTER LEVERAGED ENGINE ({args.mode.upper()} MODE)")
    print(f" Target Group : {group_val.upper()} ({', '.join(signals)})")
    print(f" Bull Params  : TP {args.bull_tp}% | SL {args.bull_sl}% | ADX {args.bull_adx} | ADD > {args.bull_add}")
    print(f" Bear Params  : TP {args.bear_tp}% | SL {args.bear_sl}% | ADX {args.bear_adx} | ADD < {args.bear_add}")
    print(f" Indicators   : Lagging={'DISABLED' if args.no_lagging else 'ACTIVE'} | Leading={'DISABLED' if args.no_leading else 'ACTIVE'}")
    print("==================================================")
    
    all_results = []
    daily_rs_records = []
    
    for signal in signals:
        if signal in ETF_MAPPING:
            bull, bear = ETF_MAPPING[signal]['bull'], ETF_MAPPING[signal]['bear']
            multiplier = 1.0
            
            active_bull_tp = (parse_ticker_param(args.bull_tp, signal) * multiplier) / 100.0
            active_bull_sl = (parse_ticker_param(args.bull_sl, signal) * multiplier) / 100.0
            active_bear_tp = (parse_ticker_param(args.bear_tp, signal) * multiplier) / 100.0
            active_bear_sl = (parse_ticker_param(args.bear_sl, signal) * multiplier) / 100.0
            
            try:
                df = build_and_calculate_indicators(signal, bull, bear, client, fetch_start_date, target_start_date, end_date, args.mode == 'macro')
                
                if args.use_rs_ranking:
                    rs_df = df[['Date', 'ORB_RS']].drop_duplicates().copy()
                    rs_df['Signal'] = signal
                    daily_rs_records.append(rs_df)
                
                res = execute_engine(df, signal, bull, bear, active_bull_tp, active_bull_sl, active_bear_tp, active_bear_sl, args.mode, args.cutoff, args.slippage, args.scale_out, args.bull_adx, args.bear_adx, args.bull_add, args.bear_add, args.bull_add_delta, args.bear_add_delta, args.bull_vix_limit, args.bear_vix_limit, args.orb_min, args.orb_max, args.trail, args.no_lagging, args.no_leading)
                if not res.empty: all_results.append(res)
            except Exception as e:
                print(f"[*] Error processing {signal}: {e}")
            
    if not all_results: return print("No trades triggered.")
        
    portfolio = pd.concat(all_results, ignore_index=True)
    
    if args.use_rs_ranking and daily_rs_records:
        master_rs = pd.concat(daily_rs_records)
        master_rs['Date_Str'] = master_rs['Date'].astype(str)
        master_rs['Abs_RS'] = master_rs['ORB_RS'].abs()
        portfolio['Date_Str'] = portfolio['Date'].astype(str)
        portfolio = pd.merge(portfolio, master_rs[['Date_Str', 'Signal', 'Abs_RS']], on=['Date_Str', 'Signal'], how='left')
        portfolio['Entry_DT'] = pd.to_datetime(portfolio['Date'].astype(str) + ' ' + portfolio['Entry_Time'])
        portfolio.sort_values(by=['Entry_DT', 'Abs_RS'], ascending=[True, False], inplace=True)
        portfolio.reset_index(drop=True, inplace=True)
    else:
        portfolio['Entry_DT'] = pd.to_datetime(portfolio['Date'].astype(str) + ' ' + portfolio['Entry_Time'])
        portfolio.sort_values('Entry_DT', inplace=True)
        portfolio.reset_index(drop=True, inplace=True)
    
    filtered_indices, active_exits, running_capital = [], [], args.capital
    cb_tracker, traded_assets_by_date = {}, {}
    
    for idx, row in portfolio.iterrows():
        trade_date = pd.to_datetime(row['Date']).date()
        trade_date_str = str(trade_date)
        traded_muscle = row['Traded']     
        underlying_signal = row['Underlying'] 
        
        if traded_muscle not in cb_tracker: cb_tracker[traded_muscle] = {'streak': 0, 'banned_until': pd.Timestamp.min.date()}
        if trade_date_str not in traded_assets_by_date: traded_assets_by_date[trade_date_str] = set()
            
        if underlying_signal in traded_assets_by_date[trade_date_str] or (args.use_cb and trade_date <= cb_tracker[traded_muscle]['banned_until']):
            continue
            
        active_exits = [exit_dt for exit_dt in active_exits if exit_dt > row['Entry_DT']]
        if len(active_exits) < args.max_concurrent:
            filtered_indices.append(idx)
            active_exits.append(pd.to_datetime(row['Date'] + ' ' + row['Exit_Time']))
            traded_assets_by_date[trade_date_str].add(underlying_signal)
            
            pnl_usd = (running_capital / args.max_concurrent) * (row['PnL_Pct'] / 100.0)
            running_capital += pnl_usd
            portfolio.at[idx, 'PnL_USD'] = round(pnl_usd, 2)
            portfolio.at[idx, 'Running_Capital'] = round(running_capital, 2)
            
            if args.use_cb:
                if 'Target Hit' in row['Reason']: cb_tracker[traded_muscle]['streak'] = 0
                elif 'Stop Loss' in row['Reason']:
                    cb_tracker[traded_muscle]['streak'] += 1
                    if cb_tracker[traded_muscle]['streak'] >= args.max_streak:
                        cb_tracker[traded_muscle]['banned_until'] = trade_date + timedelta(days=args.cooldown)
                        cb_tracker[traded_muscle]['streak'] = 0
                        portfolio.at[idx, 'Reason'] += ' [CB TRIPPED]'
            
    portfolio = portfolio.loc[filtered_indices].copy()
    
    print(f"\n=== PORTFOLIO RESULTS ({date_label}) ===")
    print(f"Total Trades Taken  : {len(portfolio)}")
    print(f"Win Rate            : {(len(portfolio[portfolio['PnL_USD'] > 0]) / len(portfolio)) * 100:.1f}%")
    print(f"Total Net PnL       : ${portfolio['PnL_USD'].sum():,.2f}")
    print(f"Final Capital       : ${portfolio['Running_Capital'].iloc[-1]:,.2f}")

    safe_group_name = args.group.replace(',', '_').lower()

    # --- INJECT FULL CONFIG METADATA FOR THE ANALYZER SCRIPTS ---
    portfolio['Config_Mode'] = args.mode
    portfolio['Config_Group'] = safe_group_name
    portfolio['Config_Bull_TP'] = args.bull_tp
    portfolio['Config_Bull_SL'] = args.bull_sl
    portfolio['Config_Bear_TP'] = args.bear_tp
    portfolio['Config_Bear_SL'] = args.bear_sl
    portfolio['Config_Bull_ADX'] = args.bull_adx
    portfolio['Config_Bear_ADX'] = args.bear_adx
    portfolio['Config_Bull_ADD'] = args.bull_add
    portfolio['Config_Bear_ADD'] = args.bear_add
    portfolio['Config_ORB_Min'] = args.orb_min
    portfolio['Config_ORB_Max'] = args.orb_max
    portfolio['Config_Slippage'] = args.slippage
    portfolio['Config_ScaleOut'] = args.scale_out
    portfolio['Config_Trail'] = args.trail
    portfolio['Config_MaxConcurrent'] = args.max_concurrent
    portfolio['Config_Cutoff'] = args.cutoff
    portfolio['Config_UseCB'] = args.use_cb
    portfolio['Config_UseRS'] = args.use_rs_ranking
    portfolio['Config_NoLagging'] = args.no_lagging
    portfolio['Config_NoLeading'] = args.no_leading

    print_cols = ['Date', 'Underlying', 'Signal', 'Traded', 'Entry_Time', 'Exit_Time', 'Reason', 'Entry_Price', 'Exit_Price', 'PnL_USD', 'PnL_Pct']
    csv_cols = ['Date', 'Underlying', 'Signal', 'Traded', 'Entry_Time', 'Exit_Time', 'Reason', 'Entry_Open', 'Entry_High', 'Entry_Low', 'Entry_Close', 'Exit_Open', 'Exit_High', 'Exit_Low', 'Exit_Close', 'Signal_ORB_High', 'Signal_ORB_Low', 'VWAP_Entry', 'VWAP_Exit', 'Entry_Price', 'Exit_Price', 'PnL_USD', 'PnL_Pct', 'Running_Capital', 'ORB_W_Pct', 'ORB_RS', 'ADX_Entry', 'ADD_Entry', 'ADD_Delta', 'VIX_Surge', 'MFE_Pct', 'MAE_Pct', 'Missed_Profit_%', 'Config_Mode', 'Config_Group', 'Config_Bull_TP', 'Config_Bull_SL', 'Config_Bear_TP', 'Config_Bear_SL', 'Config_Bull_ADX', 'Config_Bear_ADX', 'Config_Bull_ADD', 'Config_Bear_ADD', 'Config_ORB_Min', 'Config_ORB_Max', 'Config_Slippage', 'Config_ScaleOut', 'Config_Trail', 'Config_MaxConcurrent', 'Config_Cutoff', 'Config_UseCB', 'Config_UseRS', 'Config_NoLagging', 'Config_NoLeading']
    
    flags = []
    if args.slippage: flags.append("slip")
    if args.use_cb: flags.append("cb")
    if args.use_rs_ranking: flags.append("rs")
    if args.no_lagging: flags.append("noLag")
    if args.no_leading: flags.append("noLead")
    flag_str = "_" + "-".join(flags) if flags else ""
    
    safe_tp = str(args.bull_tp).replace(':', '_').replace(',', '-')
    safe_sl = str(args.bull_sl).replace(':', '_').replace(',', '-')
    safe_bear_tp = str(args.bear_tp).replace(':', '_').replace(',', '-')
    safe_bear_sl = str(args.bear_sl).replace(':', '_').replace(',', '-')
    
    export_dir = os.path.join(
        "exports", 
        safe_group_name, 
        args.mode, 
        f"bull_tp{safe_tp}_sl{safe_sl}_bear_tp{safe_bear_tp}_sl{safe_bear_sl}_adx{args.bull_adx}-{args.bear_adx}_orb{args.orb_min}-{args.orb_max}_co{args.cutoff.replace(':', '')}"
    )
    
    os.makedirs(export_dir, exist_ok=True)
    
    csv_filename = f"backtest_bull_tp{safe_tp}_sl{safe_sl}_bear_tp{safe_bear_tp}_sl{safe_bear_sl}_bullADX{args.bull_adx}_bearADX{args.bear_adx}_orb{args.orb_min}-{args.orb_max}{flag_str}_{date_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_filepath = os.path.join(export_dir, csv_filename)
    
    portfolio[csv_cols].to_csv(csv_filepath, index=False)
    print(f"\n[+] Full trade log successfully exported to: {csv_filepath}")

    print(f"Average MAE (Drawdown): {portfolio['MAE_Pct'].mean():.2f}%")
    print(f"Average Missed Profit : {portfolio['Missed_Profit_%'].mean():.2f}%")

    print("\n--- Trade Log (Last 20 Trades) ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(portfolio[print_cols].tail(20).to_string(index=False))

if __name__ == "__main__":
    main()