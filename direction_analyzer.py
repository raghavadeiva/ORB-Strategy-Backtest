import pandas as pd
import argparse
import os
import glob
import warnings

# Suppress pandas FutureWarnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

def analyze_directional_bias(filepath):
    # Load the CSV
    df = pd.read_csv(filepath)
    
    if df.empty:
        print("Trade log is empty.")
        return

    print(f"\n=======================================================")
    print(f" DIRECTIONAL BIAS ANALYSIS: {os.path.basename(filepath)}")
    print(f"=======================================================")

    bull_trades = df[df['Signal'].isin(['BULL', 'BULLISH', 'LONG'])]
    bear_trades = df[df['Signal'].isin(['BEAR', 'BEARISH', 'SHORT'])]

    def print_stats(side_df, side_name):
        if side_df.empty:
            print(f"\nNo {side_name} trades found.")
            return
            
        side_df = side_df.copy() # Prevents pandas SettingWithCopyWarning
            
        total = len(side_df)
        wins = side_df[side_df['PnL_USD'] > 0]
        losses = side_df[side_df['PnL_USD'] <= 0]
        
        win_rate = (len(wins) / total) * 100
        net_pnl = side_df['PnL_USD'].sum()
        
        avg_mfe = side_df['MFE_Pct'].mean()
        avg_mae = side_df['MAE_Pct'].mean()
        
        # CORRECT TIME MATH: Convert to total minutes since midnight to average correctly
        times = pd.to_datetime(side_df['Entry_Time'], format='%H:%M:%S')
        total_minutes = times.dt.hour * 60 + times.dt.minute
        avg_total_minutes = total_minutes.mean()
        
        avg_hour = int(avg_total_minutes // 60)
        avg_min = int(avg_total_minutes % 60)

        print(f"\n--- {side_name} TRADES ---")
        print(f"Total Trades   : {total}")
        print(f"Win Rate       : {win_rate:.2f}%")
        print(f"Net PnL        : ${net_pnl:,.2f}")
        print(f"Average MFE    : +{avg_mfe:.2f}% (How far it runs on average)")
        print(f"Average MAE    : {avg_mae:.2f}% (How far it dips on average)")
        print(f"Avg Entry Time : {avg_hour:02d}:{avg_min:02d}")
        
        print("\nExit Reasons:")
        print(side_df['Reason'].value_counts().to_string())

    print_stats(bull_trades, "BULLISH (LONG)")
    print_stats(bear_trades, "BEARISH (SHORT)")
    print("\n=======================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze Bull vs Bear trades in CSV logs")
    parser.add_argument("path", type=str, help="Path to a specific CSV file or a folder containing CSVs")
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        analyze_directional_bias(args.path)
    elif os.path.isdir(args.path):
        csv_files = glob.glob(os.path.join(args.path, "*.csv"))
        for file in csv_files:
            analyze_directional_bias(file)
    else:
        print("Invalid path provided.")

if __name__ == "__main__":
    main()