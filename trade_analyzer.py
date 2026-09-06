import os
import glob
import pandas as pd
import argparse
import warnings

# Suppress pandas future warnings for cleaner terminal output
warnings.simplefilter(action='ignore', category=FutureWarning)

def batch_analyze(folder_path):
    if not os.path.exists(folder_path):
        print(f"[!] Directory not found: {folder_path}")
        return

    # Find all CSV files in the folder (and subfolders)
    csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    
    if not csv_files:
        print(f"[!] No CSV files found in {folder_path}")
        return

    print(f"[*] Found {len(csv_files)} trade logs. Ingesting data...\n")

    # Load all CSVs into a single master DataFrame
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Ensure it's a detailed trade log by checking for our specific columns
            if 'MFE_Pct' in df.columns and 'Config_TP' in df.columns:
                df_list.append(df)
        except Exception as e:
            print(f"[!] Could not read {os.path.basename(file)}: {e}")

    if not df_list:
        print("[!] No valid detailed trade logs found.")
        return

    master_df = pd.concat(df_list, ignore_index=True)

    # The columns we want to group by to define a "Configuration"
    group_cols = [
        'Config_TP', 'Config_SL', 'Config_Cutoff', 'Config_ADX', 
        'Config_ORB_Min', 'Config_ORB_Max', 'Config_Trail'
    ]

    # Fill NaNs in config columns just in case some runs lacked them
    for col in group_cols:
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna('N/A')

    # Ensure all grouping columns exist in the dataframe
    existing_group_cols = [col for col in group_cols if col in master_df.columns]

    print("=========================================================================================")
    print(" BATCH TRADE EXCURSION ANALYSIS (RANKED BY NET PNL)")
    print("=========================================================================================\n")

    results = []

    # Group the massive dataframe by unique configurations
    for config, group in master_df.groupby(existing_group_cols):
        # Handle single-column vs multi-column groupby tuples
        if type(config) is not tuple:
            config = (config,)
            
        config_dict = dict(zip(existing_group_cols, config))
        
        total_trades = len(group)
        wins = group[group['PnL_USD'] > 0]
        losses = group[group['PnL_USD'] <= 0]
        
        win_rate = (len(wins) / total_trades) * 100
        net_pnl = group['PnL_USD'].sum()

        # Excursion calculations
        avg_mae_wins = wins['MAE_Pct'].mean() if not wins.empty else 0.0
        avg_mfe_losses = losses['MFE_Pct'].mean() if not losses.empty else 0.0

        results.append({
            'Config': f"TP:{config_dict.get('Config_TP')} | SL:{config_dict.get('Config_SL')} | ADX:{config_dict.get('Config_ADX')} | ORB:{config_dict.get('Config_ORB_Min')}-{config_dict.get('Config_ORB_Max')} | Trail:{config_dict.get('Config_Trail')}",
            'Trades': total_trades,
            'WinRate': win_rate,
            'NetPnL': net_pnl,
            'AvgMAE_Wins': avg_mae_wins,
            'AvgMFE_Losses': avg_mfe_losses
        })

    # Convert results to DataFrame and sort by best Net PnL
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='NetPnL', ascending=False)

    # Print top 15 configurations
    print("--- TOP 15 CONFIGURATIONS BY PROFIT ---")
    for idx, row in results_df.head(15).iterrows():
        print(f"\n{row['Config']}")
        print(f"  Trades: {row['Trades']:<5} | Win Rate: {row['WinRate']:>5.1f}% | Net PnL: ${row['NetPnL']:>9,.2f}")
        print(f"  -> Winners avg dip (MAE)    : {row['AvgMAE_Wins']:.2f}%  <- (Stop-loss buffer limit)")
        print(f"  -> Losers avg run-up (MFE)  : +{row['AvgMFE_Losses']:.2f}%  <- (Trailing stop activation limit)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Analyze Detailed Trade Logs")
    parser.add_argument("folder", type=str, help="Path to the folder containing the CSV trade logs")
    args = parser.parse_args()
    
    batch_analyze(args.folder)