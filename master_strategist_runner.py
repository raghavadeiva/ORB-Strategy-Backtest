import argparse
import itertools
import subprocess
import os
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

PROGRESS_FILE = "completed_jobs.log"
RESULTS_FILE = "grid_search_summary.csv"
write_lock = threading.Lock()

def float_range(start: float, stop: float, step: float) -> List[float]:
    """Generates a list of floats from start to stop (inclusive) by step."""
    vals = []
    curr = start
    while curr <= stop + 1e-9:
        vals.append(round(curr, 4))
        curr += step
    return vals

def get_base_val(param_val) -> float:
    """Safely extracts the default float value if a string contains ticker-overrides."""
    return float(str(param_val).split(',')[0])

def build_commands() -> List[Tuple[List[str], dict, str]]:
    """Builds the list of subprocess commands for every permutation."""
    
    # -------------------------------------------------------------------
    # PARAMETER GRIDS: Adjust these arrays to widen or narrow your search
    # -------------------------------------------------------------------
    groups = ['extreme']

    # Take Profit & Stop Loss
    bull_tps = ['8.0,XLK:4.5', '8.0,XLK:6']
    bull_sls = ['3.0,XLK:2.0', '3.0,XLK:1.5']
    bear_tps = [5, '8.0,XLK:6']           
    bear_sls = [2, '2.0,XLK:1.5']         
    
    # Filters
    bull_adxs = [11.0, 13.0]
    bear_adxs = [13.0, 15.0]

    bull_add = [0.0]
    bear_add = [0.0]
    
    # Timings & Ranges
    cutoffs = ['10:30']
    orb_maxs = [0.0133, 0.0166]
    orb_min = [0.0066]
        
    # --- NEW: Protection & Routing Features ---
    use_cbs = [True, False]
    max_streaks = [3]
    cooldowns = [3]
    use_rs_rankings = [True, False]

    keys = [
        'group', 'bull_tp', 'bull_sl', 'bear_tp', 'bear_sl', 
        'cutoff', 'bull_adx', 'bear_adx', 'orb_max', 'orb_min', 
        "bull_add", "bear_add", 
        'use_cb', 'max_streak', 'cooldown', 'use_rs_ranking'
    ]
    
    # Generate the cartesian product of all combinations
    combinations = list(itertools.product(
        groups, bull_tps, bull_sls, bear_tps, bear_sls,
        cutoffs, bull_adxs, bear_adxs, orb_maxs, orb_min, bull_add, bear_add, 
        use_cbs, max_streaks, cooldowns, use_rs_rankings
    ))
    
    jobs = []
    for combo in combinations:
        params = dict(zip(keys, combo))

        params['slippage'] = True
        params['noleading'] = True
        params['trail'] = True
        # Construct the base terminal command
        cmd = [
            "python", "./backtest.py",
            "--group", str(params['group']),
            "--bull_tp", str(params['bull_tp']),
            "--bull_sl", str(params['bull_sl']),
            "--bear_tp", str(params['bear_tp']),
            "--bear_sl", str(params['bear_sl']),
            "--bull_adx", str(params['bull_adx']),
            "--bear_adx", str(params['bear_adx']),
            "--cutoff", str(params['cutoff']),
            "--orb_max", str(params['orb_max'])
        ]
        
        # Append boolean and conditional flags
        if params['use_cb']:
            cmd.append("--use_cb")
            cmd.extend(["--max_streak", str(params['max_streak'])])
            cmd.extend(["--cooldown", str(params['cooldown'])])
        if params['use_rs_ranking']:
            cmd.append("--use_rs_ranking")

        cmd.append("--slippage")
        cmd.append("--no_leading")

        cmd_str = " ".join(cmd)
        jobs.append((cmd, params, cmd_str))
        
    return jobs

def run_command(job_data: Tuple[List[str], dict, str]):
    """Executes a single command using subprocess and handles the output."""
    cmd, params, cmd_str = job_data
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[!] Error running: {cmd_str}\n{result.stderr}")
        else:
            trades, win_rate, pnl, capital = 0, 0.0, 0.0, 10000.0
            avg_mae, avg_missed = 0.0, 0.0
            
            if "No trades triggered" not in result.stdout:
                # Capture Core Metrics
                trades_m = re.search(r'Total Trades Taken\s*:\s*([\d,]+)', result.stdout)
                if trades_m: trades = int(trades_m.group(1).replace(',', ''))
                
                wr_m = re.search(r'Win Rate\s*:\s*([\d.]+)%', result.stdout)
                if wr_m: win_rate = float(wr_m.group(1))
                
                pnl_m = re.search(r'Total Net PnL\s*:\s*\$([-\d.,]+)', result.stdout)
                if pnl_m: pnl = float(pnl_m.group(1).replace(',', ''))
                
                cap_m = re.search(r'Final Capital\s*:\s*\$([-\d.,]+)', result.stdout)
                if cap_m: capital = float(cap_m.group(1).replace(',', ''))
                
                # --- NEW: Capture Additional Context Metrics ---
                mae_m = re.search(r'Average MAE.*?:\s*([-\d.]+)%', result.stdout)
                if mae_m: avg_mae = float(mae_m.group(1))
                
                missed_m = re.search(r'Average Missed Profit.*?:\s*([-\d.]+)%', result.stdout)
                if missed_m: avg_missed = float(missed_m.group(1))
                
            # Wraps string parameters in quotes to prevent CSV breaking
            safe_bull_tp = f'"{params["bull_tp"]}"' if ',' in str(params['bull_tp']) else params['bull_tp']
            safe_bull_sl = f'"{params["bull_sl"]}"' if ',' in str(params['bull_sl']) else params['bull_sl']
            safe_bear_tp = f'"{params["bear_tp"]}"' if ',' in str(params['bear_tp']) else params['bear_tp']
            safe_bear_sl = f'"{params["bear_sl"]}"' if ',' in str(params['bear_sl']) else params['bear_sl']
            
            csv_row = (f"{params['group']},{safe_bull_tp},{safe_bull_sl},"
                       f"{safe_bear_tp},{safe_bear_sl},{params['cutoff']},"
                       f"{params['slippage']},{params['noleading']},{params['bull_adx']},{params['bear_adx']},"
                       f"{params['bull_add']},{params['bear_add']},{params['orb_min']},"
                       f"{params['orb_max']},{params['trail']},"
                       f"{params['use_cb']},{params['max_streak']},{params['cooldown']},{params['use_rs_ranking']},"
                       f"{trades},{win_rate},{pnl},{capital},{avg_mae},{avg_missed}\n")

            print(f"[+] Completed (PnL: ${pnl:,.2f}): {cmd_str}")
            
            with write_lock:
                with open(PROGRESS_FILE, "a") as f:
                    f.write(cmd_str + "\n")
                with open(RESULTS_FILE, "a") as f:
                    f.write(csv_row)
    except Exception as e:
        print(f"[!] Exception running {cmd_str}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Asymmetric Grid Search Runner")
    parser.add_argument('--workers', type=int, default=4, help='Number of concurrent backtests to run')
    parser.add_argument('--dry_run', action='store_true', help='Only print combinations, do not execute')
    parser.add_argument('--sample', type=int, help='Run a random subset sample of the combinations')
    parser.add_argument('--reset', action='store_true', help='Wipe progress tracking file and start fresh')
    args = parser.parse_args()

    if args.reset and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
        print(f"[*] Progress and Summary files wiped. Starting fresh.")

    # Write expanded header for the new parameters
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("Group,Bull_TP,Bull_SL,Bear_TP,Bear_SL,Cutoff,Slippage,Bull_ADX,Bear_ADX,Bull_ADD,Bear_ADD,ORB_Min,ORB_Max,Trail,Use_CB,Max_Streak,Cooldown,Use_RS,Total_Trades,Win_Rate_Pct,Net_PnL,Final_Capital,Avg_MAE,Avg_Missed_Profit\n")

    jobs = build_commands()
    total_commands = len(jobs)

    completed_jobs = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            completed_jobs = set(line.strip() for line in f if line.strip())

    jobs = [job for job in jobs if job[2] not in completed_jobs]
    
    print(f"Generated {total_commands:,} total configuration combinations after constraints.")
    if completed_jobs:
        print(f"Found {len(completed_jobs):,} previously completed jobs. {len(jobs):,} remaining to execute.")

    if not jobs:
        print("[*] All combinations have already been executed!")
        return

    if args.sample:
        import random
        jobs = random.sample(jobs, min(args.sample, len(jobs)))
        print(f"Sampling down to {len(jobs)} random configurations for testing.")

    if args.dry_run:
        print("\n[Dry Run] Displaying the first 10 commands:")
        for job in jobs[:10]:
            print(job[2])
        return

    print(f"\nExecuting with {args.workers} concurrent threads...")
    print("This will take a significant amount of time. Logs will output as jobs complete.\n")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        executor.map(run_command, jobs)

if __name__ == "__main__":
    main()