import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# Experiment configurations
node_counts = [500, 1000, 2000, 3000, 4000, 5000]
avg_send = 1000000       # average sending interval (ms)
simtime = 86400000        # 24 hours in ms
collision = 0             # Simplified collision check
scenario_id = 1           # Scenario 1 (p=0.018, q=0.764)
k_intervals = 3           # K=3 feature window

scripts = {
    "ALOHA": "../src/core/lora_dir_aloha.py",
    "Plain_LBT": "../src/core/lora_dir_plain_lbt.py",
    "Classifier_LBT": "../src/core/lora_dir_classifier_lbt.py"
}

def run_sim(script_name, nodes, experiment_id):
    cmd = [
        "python", script_name,
        str(nodes), str(avg_send), str(experiment_id), str(simtime),
        str(collision), str(scenario_id), str(k_intervals)
    ]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Parse DER method 2 from stdout
        der_match = re.search(r"DER method 2:\s*([\d\.]+)", stdout)
        if der_match:
            der = float(der_match.group(1)) * 100.0  # Convert to percentage
            print(f"[{script_name}] Nodes: {nodes}, Exp: {experiment_id} -> DER: {der:.2f}%")
            return der
        else:
            der_match_1 = re.search(r"DER:\s*([\d\.]+)", stdout)
            if der_match_1:
                der = float(der_match_1.group(1)) * 100.0
                print(f"[{script_name}] Nodes: {nodes}, Exp: {experiment_id} -> DER: {der:.2f}% (method 1 fallback)")
                return der
            print(f"[{script_name}] Nodes: {nodes}, Exp: {experiment_id} -> Failed to parse DER")
            return None
    except Exception as e:
        print(f"Failed to run simulation for {script_name}: {e}")
        return None

def execute_and_plot(experiment_id, filename_suffix, title_prefix, plot_filename):
    results = []
    
    for nodes in node_counts:
        row = {"Nodes": nodes}
        for name, script_file in scripts.items():
            der = run_sim(script_file, nodes, experiment_id)
            row[f"{name}_DER"] = der
        results.append(row)
        
    df = pd.DataFrame(results)
    csv_filename = f"results_{filename_suffix}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved results to {csv_filename}")
    print(df)
    
    # Plot results
    plt.figure(figsize=(9, 6.5))
    
    # Premium color palette
    # ALOHA: Crimson/Tomato Red, Plain LBT: Royal Blue, Classifier LBT: Emerald Green
    plt.plot(df["Nodes"], df["ALOHA_DER"], marker='o', linestyle='--', color='#d9534f', linewidth=2, markersize=8, label='ALOHA (Baseline)')
    plt.plot(df["Nodes"], df["Plain_LBT_DER"], marker='s', linestyle='-.', color='#0275d8', linewidth=2, markersize=8, label='Plain LBT (Without Classifier)')
    plt.plot(df["Nodes"], df["Classifier_LBT_DER"], marker='^', linestyle='-', color='#5cb85c', linewidth=2.5, markersize=9, label='Classifier-Driven LBT (Proposed)')
    
    plt.title(f'Performance Comparison under {title_prefix} Configuration\n(Scenario 1: p=0.018, q=0.764)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Number of Nodes (Network Density)', fontsize=11, labelpad=10)
    plt.ylabel('Data Extraction Rate (DER %)', fontsize=11, labelpad=10)
    
    plt.grid(True, linestyle=':', alpha=0.6, color='#999999')
    plt.legend(fontsize=10.5, loc='lower left')
    
    plt.ylim(50, 102)
    plt.xlim(0, 5200)
    
    # Styling details
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}\n")
    plt.close()

if __name__ == "__main__":
    # 1. Run for SN1 (Fixed SF12, Experiment 4)
    print("=================== RUNNING FOR SN1 (FIXED SF12) ===================")
    execute_and_plot(experiment_id=4, filename_suffix="sn1", title_prefix="SN1 (Fixed SF12)", plot_filename = "../Figures/fig_8_sn1.png")
    
    # 2. Run for SN3 (Adaptive SF, Experiment 3)
    print("=================== RUNNING FOR SN3 (ADAPTIVE SF) ===================")
    execute_and_plot(experiment_id=3, filename_suffix="sn3", title_prefix="SN3 (Adaptive SF)", plot_filename = "../Figures/fig_8_sn3.png")
    
    print("Comparative simulations completed successfully!")
