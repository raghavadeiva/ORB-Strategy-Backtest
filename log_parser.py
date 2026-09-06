import re
import pandas as pd
import argparse
import os
import glob

def analyze_csv_deep_dive(filepath):
    """Reads the actual trade log CSV and extracts deep performance metrics."""
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
            
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        reasons = df['Reason'].fillna('')
        signals = df['Signal'].astype(str)
            
        # Basic Counts & Breakdown
        total_trades = len(df)
        wins = len(df[df['PnL_USD'] > 0])
        losses = len(df[df['PnL_USD'] <= 0])
        
        bull_df = df[df['Signal'].str.contains('BULL', case=False, na=False)]
        bear_df = df[df['Signal'].str.contains('BEAR', case=False, na=False)]
        
        bull_trades = len(bull_df)
        bull_wins = len(bull_df[bull_df['PnL_USD'] > 0])
        bull_wr = (bull_wins / bull_trades * 100) if bull_trades else 0.0
        
        bear_trades = len(bear_df)
        bear_wins = len(bear_df[bear_df['PnL_USD'] > 0])
        bear_wr = (bear_wins / bear_trades * 100) if bear_trades else 0.0
        
        # Exit Reasons Breakdown
        target_hits = reasons.str.contains('Target Hit', case=False).sum()
        stop_losses = reasons.str.contains('Stop Loss|Trailing Stop|BE Stop', case=False).sum()
        eod_closes = reasons.str.contains('EOD Close', case=False).sum()
        
        # EOD Performance
        eod_df = df[reasons.str.contains('EOD Close', case=False)]
        eod_pos = (eod_df['PnL_USD'] > 0).sum()
        eod_neg = (eod_df['PnL_USD'] <= 0).sum()
        
        # Peak and Drawdown
        df['Peak'] = df['Running_Capital'].cummax()
        df['Drawdown_Pct'] = (df['Running_Capital'] - df['Peak']) / df['Peak'] * 100
        max_drawdown = df['Drawdown_Pct'].min()
        
        peak_idx = df['Running_Capital'].idxmax()
        peak_capital = df['Running_Capital'].max()
        peak_date = df.loc[peak_idx, 'Date']
        final_capital = df['Running_Capital'].iloc[-1]
        
        # Losing/Winning Streaks with Dates and Capital Context
        max_win_streak = {'count': 0, 'start_date': 'N/A', 'end_date': 'N/A', 'start_cap': 0.0, 'end_cap': 0.0}
        max_loss_streak = {'count': 0, 'start_date': 'N/A', 'end_date': 'N/A', 'start_cap': 0.0, 'end_cap': 0.0}
        
        cur_win = {'count': 0, 'start_date': None, 'start_cap': 10000.0}
        cur_loss = {'count': 0, 'start_date': None, 'start_cap': 10000.0}
        
        prev_cap = 10000.0
        prev_date = "N/A"
        
        for idx, row in df.iterrows():
            is_win = row['PnL_USD'] > 0
            date = row['Date']
            cap = row['Running_Capital']
            
            if is_win:
                if cur_win['count'] == 0:
                    cur_win['start_date'] = date
                    cur_win['start_cap'] = prev_cap
                cur_win['count'] += 1
                
                # Check and reset loss streak
                if cur_loss['count'] > max_loss_streak['count']:
                    max_loss_streak = {
                        'count': cur_loss['count'], 'start_date': cur_loss['start_date'], 
                        'end_date': prev_date, 'start_cap': cur_loss['start_cap'], 'end_cap': prev_cap
                    }
                cur_loss = {'count': 0, 'start_date': None, 'start_cap': cap}
            else:
                if cur_loss['count'] == 0:
                    cur_loss['start_date'] = date
                    cur_loss['start_cap'] = prev_cap
                cur_loss['count'] += 1
                
                # Check and reset win streak
                if cur_win['count'] > max_win_streak['count']:
                    max_win_streak = {
                        'count': cur_win['count'], 'start_date': cur_win['start_date'], 
                        'end_date': prev_date, 'start_cap': cur_win['start_cap'], 'end_cap': prev_cap
                    }
                cur_win = {'count': 0, 'start_date': None, 'start_cap': cap}
                
            prev_cap = cap
            prev_date = date
            
        # Final loop checks
        if cur_win['count'] > max_win_streak['count']:
            max_win_streak = {'count': cur_win['count'], 'start_date': cur_win['start_date'], 'end_date': prev_date, 'start_cap': cur_win['start_cap'], 'end_cap': prev_cap}
        if cur_loss['count'] > max_loss_streak['count']:
            max_loss_streak = {'count': cur_loss['count'], 'start_date': cur_loss['start_date'], 'end_date': prev_date, 'start_cap': cur_loss['start_cap'], 'end_cap': prev_cap}
                
        # Largest Single Trades
        largest_win = df['PnL_USD'].max()
        largest_loss = df['PnL_USD'].min()
        
        # --- TICKER-SPECIFIC BREAKDOWN ---
        ticker_stats = {}
        for ticker in df['Traded'].dropna().unique():
            t_df = df[df['Traded'] == ticker].copy()
            t_trades = len(t_df)
            t_wins = len(t_df[t_df['PnL_USD'] > 0])
            t_losses = t_trades - t_wins
            t_wr = (t_wins / t_trades * 100) if t_trades else 0.0
            t_pnl = t_df['PnL_USD'].sum()
            
            t_reasons = t_df['Reason'].fillna('')
            t_targets = t_reasons.str.contains('Target Hit', case=False).sum()
            t_stops = t_reasons.str.contains('Stop Loss|Trailing Stop|BE Stop', case=False).sum()
            t_eods = t_reasons.str.contains('EOD Close', case=False).sum()
            
            t_df['Cum_PnL'] = t_df['PnL_USD'].cumsum()
            t_df['Peak'] = t_df['Cum_PnL'].cummax()
            t_max_dd_usd = (t_df['Cum_PnL'] - t_df['Peak']).min()
            
            t_max_ws, t_max_ls = 0, 0
            c_w, c_l = 0, 0
            for _, r in t_df.iterrows():
                if r['PnL_USD'] > 0:
                    c_w += 1
                    t_max_ls = max(t_max_ls, c_l)
                    c_l = 0
                else:
                    c_l += 1
                    t_max_ws = max(t_max_ws, c_w)
                    c_w = 0
            t_max_ws = max(t_max_ws, c_w)
            t_max_ls = max(t_max_ls, c_l)
            
            ticker_stats[ticker] = {
                'Trades': t_trades, 'Wins': t_wins, 'Losses': t_losses, 'Win_Rate': t_wr,
                'Net_PnL': t_pnl, 'Max_DD_USD': t_max_dd_usd,
                'Max_Win_Streak': t_max_ws, 'Max_Loss_Streak': t_max_ls,
                'Targets': t_targets, 'Stops': t_stops, 'EODs': t_eods
            }
        
        return {
            'File_Name': os.path.basename(filepath),
            'Trades': total_trades,
            'Wins': wins,
            'Losses': losses,
            'Bull_Trades': bull_trades,
            'Bull_WR': bull_wr,
            'Bear_Trades': bear_trades,
            'Bear_WR': bear_wr,
            'Target_Hits': target_hits,
            'Stop_Losses': stop_losses,
            'EOD_Closes': eod_closes,
            'EOD_Pos': eod_pos,
            'EOD_Neg': eod_neg,
            'Peak': peak_capital,
            'Peak_Date': peak_date,
            'Final': final_capital,
            'Max_DD': max_drawdown,
            'Max_Win_Streak': max_win_streak,
            'Max_Loss_Streak': max_loss_streak,
            'Largest_Win': largest_win,
            'Largest_Loss': largest_loss,
            'Ticker_Stats': ticker_stats
        }
    except Exception as e:
        print(f"Error parsing CSV {filepath}: {e}")
        return None

def analyze_text_log(filepath):
    """
    Parses a raw terminal log file, extracts the PnL and parameters, 
    and generates a statistical breakdown with a deep-dive CSV analysis.
    """
    if not os.path.exists(filepath):
        print(f"[!] Error: Could not find '{filepath}'.")
        return

    print(f"[*] Ingesting and parsing raw text data from '{filepath}'...")
    
    data = []
    pnl_pattern = re.compile(r'PnL:\s*\$([-\d,.]+)')

    with open(filepath, 'r') as f:
        for line in f:
            if '[+] Completed' not in line:
                continue
                
            pnl_match = pnl_pattern.search(line)
            if not pnl_match:
                continue
                
            pnl_val = float(pnl_match.group(1).replace(',', ''))
            run_data = {'PnL': pnl_val}
            
            cmd_start = line.find('--group')
            if cmd_start == -1:
                continue
                
            cmd_part = line[cmd_start:].strip()
            run_data['Raw_Command'] = cmd_part
            tokens = cmd_part.split()
            
            i = 0
            while i < len(tokens):
                if tokens[i].startswith('--'):
                    key = tokens[i][2:]
                    if i + 1 < len(tokens) and not tokens[i+1].startswith('--'):
                        val = tokens[i+1]
                        try:
                            run_data[key] = float(val)
                        except ValueError:
                            run_data[key] = val
                        i += 2
                    else:
                        run_data[key] = True
                        i += 1
                else:
                    i += 1
                    
            data.append(run_data)

    df = pd.DataFrame(data)
    
    if df.empty:
        print("[!] No valid completed runs found in the log.")
        return

    bool_flags = ['slippage', 'use_cb', 'use_rs_ranking', 'scale_out', 'trail']
    for flag in bool_flags:
        if flag in df.columns:
            df[flag] = df[flag].fillna(False)

    print("\n=======================================================================================")
    print(f" 📊 RAW TEXT LOG ANALYSIS: {os.path.basename(filepath)}")
    print("=======================================================================================\n")

    # --- 1. DUPLICATE EXECUTIONS REPORT ---
    duplicates = df['Raw_Command'].value_counts()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        print("⚠️ DUPLICATE EXECUTIONS DETECTED ⚠️")
        print("-" * 120)
        print(f"Found {len(duplicates)} unique configurations that were executed more than once:\n")
        for cmd, count in duplicates.head(5).items():
            print(f"[{count}x] {cmd}")
        if len(duplicates) > 5:
            print(f"... and {len(duplicates) - 5} more duplicated configurations.\n")
        else:
            print("\n")

    # --- 2. DEDUPLICATE ---
    df_unique = df.drop_duplicates(subset=['Raw_Command'], keep='last').copy()
    
    total_runs = len(df_unique)
    profitable_runs = len(df_unique[df_unique['PnL'] > 0])
    profit_pct = (profitable_runs / total_runs) * 100 if total_runs else 0
    avg_pnl = df_unique['PnL'].mean()

    print("--- HIGH-LEVEL SUMMARY (UNIQUE RUNS ONLY) ---")
    print(f"Total Unique Configs Parsed : {total_runs:,}")
    print(f"Profitable Configurations   : {profitable_runs:,} ({profit_pct:.1f}%)")
    print(f"Average Net PnL Achieved    : ${avg_pnl:,.2f}\n")

    # --- 3. TOP 10 CONFIGURATIONS ---
    top_10 = df_unique.sort_values(by='PnL', ascending=False).head(10)
    print("🏆 TOP 10 PERFORMING CONFIGURATIONS 🏆")
    print("-" * 120)
    cols = ['PnL'] + [c for c in df_unique.columns if c not in ['PnL', 'Raw_Command']]
    print(top_10[cols].to_string(index=False))
    print("\n")

    # --- 4. CSV DEEP DIVE (THE WATERFALL METRICS) ---
    print("=======================================================================================")
    print(" 🔬 DEEP DIVE: TOP 10 FORENSIC ANALYSIS (Reading from CSV Logs) 🔬")
    print("=======================================================================================\n")

    for index, (idx, row) in enumerate(top_10.iterrows()):
        
        # Exact Extraction
        safe_group = str(row.get('group', 'extreme')).replace(',', '_').lower()
        safe_tp = str(row.get('bull_tp', '')).replace(':', '_').replace(',', '-')
        safe_sl = str(row.get('bull_sl', '')).replace(':', '_').replace(',', '-')
        
        # FIX: Align parser defaults exactly with the backtester's argument defaults
        bull_adx = row.get('bull_adx', 13.0)
        bear_adx = row.get('bear_adx', 11.0)
        orb_min = row.get('orb_min', 0.0066)
        orb_max = row.get('orb_max', 0.0166)
        cutoff = str(row.get('cutoff', '11:30')).replace(':', '')
        
        # Exact Flags
        flags = []
        if row.get('slippage'): flags.append("slip")
        if row.get('scale_out'): flags.append("scale")
        if row.get('trail'): flags.append("trail")
        if row.get('use_cb'): flags.append("cb")
        if row.get('use_rs_ranking'): flags.append("rs")
        flag_str = "_" + "-".join(flags) if flags else ""
        
        # Exact Folder Path Reconstruction
        folder_name = f"bull_tp{safe_tp}_sl{safe_sl}_adx{bull_adx}-{bear_adx}_orb{orb_min}-{orb_max}_co{cutoff}"
        export_dir = os.path.join("exports", safe_group, "macro", folder_name)
        
        # Exact File Prefix Match
        csv_prefix = f"backtest_bullADX{bull_adx}_bearADX{bear_adx}_orb{orb_min}-{orb_max}{flag_str}_*.csv"
        search_path = os.path.join(export_dir, csv_prefix)
        
        matched_files = glob.glob(search_path)
        
        print(f"Rank #{index + 1} | Net PnL: ${row['PnL']:,.2f}")
        print(f"Config : TP {row.get('bull_tp')} | SL {row.get('bull_sl')} | ADX {bull_adx}/{bear_adx} | CB: {row.get('use_cb', False)} | RS: {row.get('use_rs_ranking', False)}")
        
        if not matched_files:
            print(f"   [!] Could not locate corresponding CSV. Checked path: {search_path}\n")
            continue
            
        correct_file = None
        for fpath in matched_files:
            try:
                # Fast check: read only the PnL column to find the match
                df_temp = pd.read_csv(fpath, usecols=['PnL_USD'])
                file_pnl = df_temp['PnL_USD'].sum()
                
                # Check if it matches within $1.00 to account for floating point rounding
                if abs(file_pnl - row['PnL']) < 1.0:
                    correct_file = fpath
                    break
            except Exception:
                continue
                
        if not correct_file:
            print(f"   [!] Found {len(matched_files)} files, but none matched the exact PnL of ${row['PnL']:,.2f}\n")
            continue
            
        metrics = analyze_csv_deep_dive(correct_file)
        
        if metrics:
            ws = metrics['Max_Win_Streak']
            ls = metrics['Max_Loss_Streak']
            
            print(f"   ► File Analyzed     : {metrics['File_Name']}")
            print(f"   ► Portfolio Peak    : ${metrics['Peak']:,.2f} on {metrics['Peak_Date']} (Final: ${metrics['Final']:,.2f})")
            print(f"   ► Max Drawdown      : {metrics['Max_DD']:.2f}% from peak")
            print(f"   ► Overall Trades    : {metrics['Wins']} Wins / {metrics['Losses']} Losses ({metrics['Trades']} Total)")
            print(f"   ► Directional       : Bull {metrics['Bull_Trades']} trades ({metrics['Bull_WR']:.1f}% WR) | Bear {metrics['Bear_Trades']} trades ({metrics['Bear_WR']:.1f}% WR)")
            print(f"   ► Exit Reasons      : {metrics['Target_Hits']} Targets | {metrics['Stop_Losses']} Stops | {metrics['EOD_Closes']} EODs ({metrics['EOD_Pos']} Pos / {metrics['EOD_Neg']} Neg)")
            print(f"   ► Max Win Streak    : {ws['count']} trades ({ws['start_date']} to {ws['end_date']}) | Capital: ${ws['start_cap']:,.2f} -> ${ws['end_cap']:,.2f}")
            print(f"   ► Max Loss Streak   : {ls['count']} trades ({ls['start_date']} to {ls['end_date']}) | Capital: ${ls['start_cap']:,.2f} -> ${ls['end_cap']:,.2f}")
            print(f"   ► Extremes          : Largest Win +${metrics['Largest_Win']:,.2f} | Largest Loss -${abs(metrics['Largest_Loss']):,.2f}")
            
            print(f"   ► TICKER BREAKDOWN  :")
            for tkr, t_stats in metrics['Ticker_Stats'].items():
                print(f"       [{tkr}] {t_stats['Trades']} Trades ({t_stats['Win_Rate']:.1f}% WR) | Net: ${t_stats['Net_PnL']:,.2f} | Max DD: ${t_stats['Max_DD_USD']:,.2f} | W/L Streak: {t_stats['Max_Win_Streak']}/{t_stats['Max_Loss_Streak']} | Exits: {t_stats['Targets']}T/{t_stats['Stops']}S/{t_stats['EODs']}E")
        else:
            print("   [!] Error parsing CSV metrics.")
        print("-" * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Raw Terminal Logs & Deep Dive CSVs")
    parser.add_argument('--file', type=str, required=True, help="Path to your raw text log file")
    args = parser.parse_args()
    
    analyze_text_log(args.file)