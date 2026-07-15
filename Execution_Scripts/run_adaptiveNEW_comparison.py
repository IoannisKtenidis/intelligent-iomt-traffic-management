import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import shutil

# Simulation parameters
node_counts = [500, 1000, 2000, 3000, 4000, 5000]
avg_send = 1000000       # average sending interval (ms)
simtime = 86400000        # 24 hours in ms
collision = 0             # Simplified collision check
k_intervals = 3           # K=3 feature window
iterations = 100          # 100 Monte Carlo runs per configuration
scenario_id = 1           # Scenario 1 (p=0.018, q=0.764)
experiment_id = 3         # Experiment 3 = Adaptive SF (SN3 configuration)

scripts = {
    "ALOHA": "../Core_Code/loraDir - ALOHA.py",
    "Plain_LBT": "../Core_Code/loraDir - Plain LBT.py",
    "Classifier_LBT": "../Core_Code/loraDir - Classifier LBT.py"
}

output_dir = "../Results_Data/adaptiveNEW"
os.makedirs(output_dir, exist_ok=True)

def run_single_sim(task):
    script_name, nodes, iter_idx = task
    script_file = scripts[script_name]
    
    # CommandLine format: ./loraDir <nodes> <avgsend> <experiment> <simtime> [collision] [scenario] [k_intervals]
    cmd = [
        "python", script_file,
        str(nodes), str(avg_send), str(experiment_id), str(simtime),
        str(collision), str(scenario_id), str(k_intervals)
    ]
    
    try:
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
        print(f"Error running {script_file} (nodes={nodes}, iter={iter_idx}): {e}")
        return None

def main():
    print("=====================================================================")
    print(f"  LAUNCHING ADAPTIVE SF COMPARATIVE Sweep (100 Iterations per Config)")
    print("=====================================================================")
    
    tasks = []
    # Build list of all runs
    for script_name in scripts.keys():
        for nodes in node_counts:
            for i in range(iterations):
                tasks.append((script_name, nodes, i))
                        
    total_tasks = len(tasks)
    print(f"Total simulation runs to execute: {total_tasks}")
    
    # Run in parallel using ProcessPoolExecutor
    cpu_count = os.cpu_count() or 4
    # Cap workers at 3 to prevent virtual memory paging/thrashing
    max_workers = min(3, max(1, cpu_count - 1))
    print(f"Running in parallel using {max_workers} CPU workers (capped at 3)...")
    
    results_raw = []
    completed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_sim, task): task for task in tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            script_name, nodes, iter_idx = task
            der = future.result()
            
            completed += 1
            if der is not None:
                results_raw.append({
                    "Approach": script_name,
                    "Nodes": nodes,
                    "DER": der
                })
            
            # Progress reporting
            if completed % 50 == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                est_total = (elapsed / completed) * total_tasks
                est_rem = est_total - elapsed
                print(f"Progress: {completed}/{total_tasks} runs completed ({completed/total_tasks*100:.1f}%). "
                      f"Elapsed: {elapsed/60:.1f}m. Est. Remaining: {est_rem/60:.1f}m.")
                      
    print(f"\nAll simulations completed in {(time.time() - start_time)/60:.2f} minutes.")
    
    # Save raw data
    df_raw = pd.DataFrame(results_raw)
    raw_path = os.path.join(output_dir, "adaptive_comparison_raw.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"Raw results saved to {raw_path}")
    
    # Calculate Averages (group by Approach, Nodes)
    df_grouped = df_raw.groupby(["Approach", "Nodes"])["DER"].mean().reset_index()
    averages_path = os.path.join(output_dir, "adaptive_comparison_averages.csv")
    df_grouped.to_csv(averages_path, index=False)
    print(f"Averages saved to {averages_path}")
    
    # Pivot to make it easy to plot / read
    df_pivot = df_grouped.pivot(index="Nodes", columns="Approach", values="DER").reset_index()
    pivoted_path = os.path.join(output_dir, "adaptive_comparison_pivoted.csv")
    df_pivot.to_csv(pivoted_path, index=False)
    print(f"Pivoted averages saved to {pivoted_path}")
    print(df_pivot)
    
    # Generate Plots
    colors = {
        "ALOHA": "#d9534f",        # Crimson Red
        "Plain_LBT": "#0275d8",    # Royal Blue
        "Classifier_LBT": "#5cb85c" # Emerald Green
    }
    
    labels = {
        "ALOHA": "Standard ALOHA",
        "Plain_LBT": "Plain LBT",
        "Classifier_LBT": "Classifier-Driven LBT (Proposed)"
    }
    
    plt.figure(figsize=(9.5, 7))
    for approach in ["ALOHA", "Plain_LBT", "Classifier_LBT"]:
        if approach in df_pivot.columns:
            plt.plot(
                df_pivot["Nodes"], df_pivot[approach], 
                marker='o' if approach == "Classifier_LBT" else 's' if approach == "Plain_LBT" else '^',
                linestyle='-' if approach == "Classifier_LBT" else '-.' if approach == "Plain_LBT" else '--',
                color=colors[approach], linewidth=2.2, markersize=8,
                label=labels[approach]
            )
            
    plt.title('Adaptive SF Configuration: ALOHA vs. Plain LBT vs. Classifier LBT\nHome Scenario (Scenario 1, p=0.018, q=0.764)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Number of End-Nodes', fontsize=11, labelpad=10)
    plt.ylabel('Data Extraction Rate (DER Fraction)', fontsize=11, labelpad=10)
    
    plt.grid(True, linestyle=':', alpha=0.6, color='#999999')
    plt.legend(fontsize=10.5, loc='lower left')
    
    plt.ylim(0.5, 1.02)
    plt.xlim(0, 5200)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plot_filename = os.path.join(output_dir, "fig_adaptive_comparison.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_filename}")
    
    # Also save a copy in the root folder
    root_plot_path = "../Figures/fig_"
    shutil.copy(plot_filename, root_plot_path)
    print(f"Saved copy in root folder: {root_plot_path}")
    
    # Copy to brain directory for artifact viewing
    brain_dir = r"C:\Users\jokte\.gemini\antigravity\brain\3dc26943-42c0-4305-8918-827f1ca989c4"
    if os.path.exists(brain_dir):
        brain_plot_path = os.path.join(brain_dir, "fig_adaptive_comparison.png")
        shutil.copy(plot_filename, brain_plot_path)
        print(f"Copied plot to brain directory: {brain_plot_path}")
        
    plt.close()

if __name__ == "__main__":
    main()
