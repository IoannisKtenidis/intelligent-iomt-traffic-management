import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# Experiment configurations
node_counts = [200, 1000, 2000, 3000, 4000]
avg_send = 1000000       # average sending interval (ms)
simtime = 86400000        # 24 hours in ms
collision = 0             # Simplified collision check
scenario_id = 1           # Scenario 1 (p=0.018, q=0.764)
k_intervals = 3           # K=3 feature window

script_name = "../Core_Code/loraDir - Classifier LBT.py"

def run_sim(nodes, experiment_id):
    cmd = [
        "python", script_name,
        str(nodes), str(avg_send), str(experiment_id), str(simtime),
        str(collision), str(scenario_id), str(k_intervals)
    ]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Parse DER method 2 (e.g. "DER method 2: 0.9517...")
        der_match = re.search(r"DER method 2:\s*([\d\.]+)", stdout)
        if der_match:
            der = float(der_match.group(1))
            print(f"Nodes: {nodes}, Exp: {experiment_id} -> DER: {der:.4f}")
            return der
        else:
            der_match_1 = re.search(r"DER:\s*([\d\.]+)", stdout)
            if der_match_1:
                der = float(der_match_1.group(1))
                print(f"Nodes: {nodes}, Exp: {experiment_id} -> DER: {der:.4f} (method 1 fallback)")
                return der
            print(f"Nodes: {nodes}, Exp: {experiment_id} -> Failed to parse DER")
            return None
    except Exception as e:
        print(f"Failed to run simulation for Exp {experiment_id}: {e}")
        return None

if __name__ == "__main__":
    results = []
    
    for nodes in node_counts:
        print(f"\n--- Running node count: {nodes} ---")
        # SN1: fixed SF12 (Exp 4)
        der_sn1 = run_sim(nodes, experiment_id=4)
        # SN2: fixed SF7 (Exp 2)
        der_sn2 = run_sim(nodes, experiment_id=2)
        # SN3: adaptive SF (Exp 3)
        der_sn3 = run_sim(nodes, experiment_id=3)
        
        results.append({
            "Nodes": nodes,
            "SN1_DER": der_sn1,
            "SN2_DER": der_sn2,
            "SN3_DER": der_sn3
        })
        
    df = pd.DataFrame(results)
    df.to_csv("../Results_Data/recreate_fig8_results.csv", index=False)
    print("\nResults saved to recreate_fig8_results.csv")
    print(df)
    
    # Generate Plot exactly like the user's reference figure
    plt.figure(figsize=(9, 6.5))
    
    # Standard styles & colors matching the reference:
    # SN1: blue line, circular markers
    # SN2: orange line, circular markers
    # SN3: green line, circular markers
    plt.plot(df["Nodes"], df["SN1_DER"], marker='o', linestyle='-', color='blue', linewidth=1.5, markersize=6, label='SN1 (SF12) - AI')
    plt.plot(df["Nodes"], df["SN2_DER"], marker='o', linestyle='-', color='orange', linewidth=1.5, markersize=6, label='SN2 (SF7) - AI')
    plt.plot(df["Nodes"], df["SN3_DER"], marker='o', linestyle='-', color='green', linewidth=1.5, markersize=6, label='SN3 (Adaptive) - AI')
    
    plt.title('DER vs. Total Nodes with AI (Remote Home Monitoring)', fontsize=12, pad=12)
    plt.xlabel('Number of End-Nodes', fontsize=10)
    plt.ylabel('Data Extraction Rate (DER)', fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9, loc='center right')
    
    plt.ylim(0.3, 1.05)
    plt.xlim(0, 4200)
    
    plot_filename = "../Figures/fig_8_recreated.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {plot_filename}")
