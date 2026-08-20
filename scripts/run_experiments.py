import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# Experiment configurations
node_counts = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
avg_send = 1000000       # average sending interval
experiment = 4            # LoRaWAN configuration (SF12, BW125)
simtime = 86400000        # 24 hours in ms
collision = 0             # Simplified collision check
scenario = 1              # Scenario 1 (p=0.018, q=0.764)
k_intervals = 3           # K=3 feature window

results = []

def run_sim(nodes, lbt_enabled):
    lbt_flag = 1 if lbt_enabled else 0
    cmd = [
        "python", "../src/core/lora_dir_hospital_scenario.py",
        str(nodes), str(avg_send), str(experiment), str(simtime),
        str(collision), str(scenario), str(k_intervals), str(lbt_flag)
    ]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Parse DER method 2 from stdout
        # e.g., "DER method 2: 0.999"
        der_match = re.search(r"DER method 2:\s*([\d\.]+)", stdout)
        if der_match:
            der = float(der_match.group(1)) * 100.0  # Convert to percentage
            print(f"Nodes: {nodes}, LBT: {lbt_enabled} -> DER: {der:.2f}%")
            return der
        else:
            print(f"Error parsing output for nodes={nodes}, lbt={lbt_enabled}")
            # Try parsing the first DER method
            der_match_1 = re.search(r"DER:\s*([\d\.]+)", stdout)
            if der_match_1:
                return float(der_match_1.group(1)) * 100.0
            return None
    except Exception as e:
        print(f"Failed to run simulation: {e}")
        return None

# Run all combinations
for nodes in node_counts:
    # 1. Run Baseline (without LBT)
    der_baseline = run_sim(nodes, lbt_enabled=False)
    
    # 2. Run Proposed LBT
    der_lbt = run_sim(nodes, lbt_enabled=True)
    
    if der_baseline is not None or der_lbt is not None:
        results.append({
            "Nodes": nodes,
            "Baseline_DER": der_baseline,
            "Proposed_LBT_DER": der_lbt
        })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("../Results_Data/scalability_results.csv", index=False)
print("\nResults saved to scalability_results.csv")
print(df)

# Generate Plot
plt.figure(figsize=(8, 6))
if "Baseline_DER" in df.columns and not df["Baseline_DER"].isna().all():
    plt.plot(df["Nodes"], df["Baseline_DER"], marker='o', linestyle='--', color='blue', label='Baseline (ALOHA)')
if "Proposed_LBT_DER" in df.columns and not df["Proposed_LBT_DER"].isna().all():
    plt.plot(df["Nodes"], df["Proposed_LBT_DER"], marker='s', linestyle='-', color='green', label='Proposed LBT (Classifier-driven)')

plt.title('Scalability Comparison in Large-Scale Scenario')
plt.xlabel('Number of Nodes')
plt.ylabel('Data Extraction Rate (DER %)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.ylim(50, 105)
plt.xlim(0, 5200)

plot_filename = "../Figures/fig_8_scalability.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as {plot_filename}")
