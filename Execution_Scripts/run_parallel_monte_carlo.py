import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Simulation parameters
node_counts = [500, 1000, 2000, 3000, 4000, 5000]
avg_send = 1000000       # average sending interval (ms)
simtime = 86400000        # 24 hours in ms
collision = 0             # Simplified collision check
k_intervals = 3           # K=3 feature window
iterations = 10           # Number of Monte Carlo runs per config

scripts = {
    "ALOHA": "../Core_Code/loraDir - ALOHA.py",
    "Plain_LBT": "../Core_Code/loraDir - Plain LBT.py",
    "Classifier_LBT": "../Core_Code/loraDir - Classifier LBT.py"
}

# Physical Configurations
# SN1: fixed SF12 (Exp 4)
# SN3: adaptive SF (Exp 3)
configs = {
    "SN1": 4,
    "SN3": 3
}

def run_single_sim(task):
    script_name, nodes, experiment_id, scenario_id, iter_idx = task
    script_file = scripts[script_name]
    
    cmd = [
        "python", script_file,
        str(nodes), str(avg_send), str(experiment_id), str(simtime),
        str(collision), str(scenario_id), str(k_intervals)
    ]
    
    try:
        # Run subprocess with timeout to prevent hang
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        stdout = process.stdout
        
        # Parse DER method 2
        der_match = re.search(r"DER method 2:\s*([\d\.]+)", stdout)
        if der_match:
            return float(der_match.group(1))
        else:
            der_match_1 = re.search(r"DER:\s*([\d\.]+)", stdout)
            if der_match_1:
                return float(der_match_1.group(1))
            return None
    except Exception as e:
        print(f"Error running {script_file} (nodes={nodes}, exp={experiment_id}, scen={scenario_id}, iter={iter_idx}): {e}")
        return None

def main():
    print("=====================================================================")
    print("   LAUNCHING MONTE CARLO SIMULATIONS Sweep (10 Iterations per Config)")
    print("=====================================================================")
    
    tasks = []
    # Build list of all runs
    for scenario in [1, 2, 3]:
        for config_name, exp_id in configs.items():
            for script_name in scripts.keys():
                for nodes in node_counts:
                    for i in range(iterations):
                        tasks.append((script_name, nodes, exp_id, scenario, i))
                        
    total_tasks = len(tasks)
    print(f"Total simulation runs to execute: {total_tasks}")
    
    # Run in parallel using ProcessPoolExecutor
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count - 1)  # Leave 1 core free for OS responsiveness
    print(f"Running in parallel using {max_workers} CPU workers...")
    
    results_raw = []
    completed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(run_single_sim, task): task for task in tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            script_name, nodes, exp_id, scenario, iter_idx = task
            der = future.result()
            
            completed += 1
            if der is not None:
                results_raw.append({
                    "Scenario": scenario,
                    "Config": "SN1" if exp_id == 4 else "SN3",
                    "Approach": script_name,
                    "Nodes": nodes,
                    "DER": der
                })
            
            # Progress reporting
            if completed % 20 == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                est_total = (elapsed / completed) * total_tasks
                est_rem = est_total - elapsed
                print(f"Progress: {completed}/{total_tasks} runs completed ({completed/total_tasks*100:.1f}%). "
                      f"Elapsed: {elapsed/60:.1f}m. Est. Remaining: {est_rem/60:.1f}m.")
                      
    print(f"\nAll simulations completed in {(time.time() - start_time)/60:.2f} minutes.")
    
    # Save raw data
    df_raw = pd.DataFrame(results_raw)
    df_raw.to_csv("../Results_Data/monte_carlo_raw_results.csv", index=False)
    
    # Calculate Averages (group by Scenario, Config, Approach, Nodes)
    df_grouped = df_raw.groupby(["Scenario", "Config", "Approach", "Nodes"])["DER"].mean().reset_index()
    df_grouped.to_csv("../Results_Data/monte_carlo_averages.csv", index=False)
    print("Averages calculated and saved to monte_carlo_averages.csv")
    
    # Pivot to make it easy to plot
    # Columns: Scenario, Config, Nodes, ALOHA_DER, Plain_LBT_DER, Classifier_LBT_DER
    df_pivot = df_grouped.pivot(index=["Scenario", "Config", "Nodes"], columns="Approach", values="DER").reset_index()
    df_pivot.to_csv("../Results_Data/monte_carlo_pivoted.csv", index=False)
    
    # Generate Plots for each Scenario
    # Each plot will show SN1 (Fixed SF12, dashed) vs SN3 (Adaptive SF, solid) for the three approaches
    colors = {
        "ALOHA": "#d9534f",        # Crimson Red
        "Plain_LBT": "#0275d8",    # Royal Blue
        "Classifier_LBT": "#5cb85c" # Emerald Green
    }
    
    labels = {
        "ALOHA": "ALOHA",
        "Plain_LBT": "Plain LBT",
        "Classifier_LBT": "Classifier-Driven LBT (Proposed)"
    }
    
    for scenario_id in [1, 2, 3]:
        plt.figure(figsize=(9.5, 7))
        sc_df = df_pivot[df_pivot["Scenario"] == scenario_id]
        
        # Plot SN1 (Fixed SF12, dashed lines)
        sn1_df = sc_df[sc_df["Config"] == "SN1"]
        for approach in ["ALOHA", "Plain_LBT", "Classifier_LBT"]:
            plt.plot(
                sn1_df["Nodes"], sn1_df[approach], 
                marker='o', linestyle='--', color=colors[approach], linewidth=1.8, markersize=7,
                label=f"{labels[approach]} - SN1 (Fixed SF12)"
            )
            
        # Plot SN3 (Adaptive SF, solid lines)
        sn3_df = sc_df[sc_df["Config"] == "SN3"]
        for approach in ["ALOHA", "Plain_LBT", "Classifier_LBT"]:
            plt.plot(
                sn3_df["Nodes"], sn3_df[approach], 
                marker='o', linestyle='-', color=colors[approach], linewidth=2.2, markersize=8,
                label=f"{labels[approach]} - SN3 (Adaptive SF)"
            )
            
        plt.title(f'Performance Comparison: SN1 (Fixed SF12) vs. SN3 (Adaptive SF)\nScenario {scenario_id} (Average of 10 runs)', fontsize=13, fontweight='bold', pad=15)
        plt.xlabel('Number of Nodes', fontsize=11, labelpad=10)
        plt.ylabel('Data Extraction Rate (DER Fraction)', fontsize=11, labelpad=10)
        
        plt.grid(True, linestyle=':', alpha=0.6, color='#999999')
        plt.legend(fontsize=9.5, loc='lower left')
        
        plt.ylim(0.5, 1.05)
        plt.xlim(0, 5200)
        
        # Styling details
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plot_filename = f"fig_mc_scenario{scenario_id}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_filename}")
        plt.close()

if __name__ == "__main__":
    main()
