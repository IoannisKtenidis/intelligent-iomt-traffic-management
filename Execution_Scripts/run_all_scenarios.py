import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# Experiment configurations
node_counts = [500, 1000, 2000, 3000, 4000, 5000]
avg_send = 1000000       # average sending interval
experiment = 4            # LoRaWAN configuration (SF12, BW125)
simtime = 86400000        # 24 hours in ms
collision = 0             # Simplified collision check
k_intervals = 3           # K=3 feature window

results = []

def run_sim(nodes, scenario_id, lbt_enabled):
    lbt_flag = 1 if lbt_enabled else 0
    cmd = [
        "python", "../Core_Code/loraDir - Hospital Scenario.py",
        str(nodes), str(avg_send), str(experiment), str(simtime),
        str(collision), str(scenario_id), str(k_intervals), str(lbt_flag)
    ]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Parse DER method 2 from stdout
        der_match = re.search(r"DER method 2:\s*([\d\.]+)", stdout)
        if der_match:
            der = float(der_match.group(1)) * 100.0  # Convert to percentage
            print(f"Nodes: {nodes}, Scenario: {scenario_id}, LBT: {lbt_enabled} -> DER: {der:.2f}%")
            return der
        else:
            # Fallback to DER
            der_match_1 = re.search(r"DER:\s*([\d\.]+)", stdout)
            if der_match_1:
                return float(der_match_1.group(1)) * 100.0
            return None
    except Exception as e:
        print(f"Failed to run simulation: {e}")
        return None

# Run all combinations
for scenario_id in [1, 2, 3]:
    for nodes in node_counts:
        # 1. Run Baseline (without LBT)
        der_baseline = run_sim(nodes, scenario_id, lbt_enabled=False)
        
        # 2. Run Proposed LBT
        der_lbt = run_sim(nodes, scenario_id, lbt_enabled=True)
        
        if der_baseline is not None or der_lbt is not None:
            results.append({
                "Scenario": scenario_id,
                "Nodes": nodes,
                "Baseline_DER": der_baseline,
                "Proposed_LBT_DER": der_lbt
            })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("../Results_Data/all_scenarios_results.csv", index=False)
print("\nAll results saved to all_scenarios_results.csv")
print(df)

# Generate Plot
plt.figure(figsize=(10, 7))
colors = {1: 'blue', 2: 'green', 3: 'red'}
markers = {1: 'o', 2: 's', 3: '^'}

for scenario_id in [1, 2, 3]:
    scenario_df = df[df["Scenario"] == scenario_id]
    
    # Plot Baseline
    plt.plot(
        scenario_df["Nodes"], scenario_df["Baseline_DER"],
        marker=markers[scenario_id], linestyle='--', color=colors[scenario_id],
        label=f'Scenario {scenario_id} - Baseline (ALOHA)'
    )
    # Plot Proposed
    plt.plot(
        scenario_df["Nodes"], scenario_df["Proposed_LBT_DER"],
        marker=markers[scenario_id], linestyle='-', color=colors[scenario_id],
        label=f'Scenario {scenario_id} - Proposed LBT'
    )

plt.title('Scalability Comparison across Transition Scenarios (Scenario 1, 2, and 3)')
plt.xlabel('Number of Nodes')
plt.ylabel('Data Extraction Rate (DER %)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.ylim(50, 105)
plt.xlim(0, 5200)

plot_filename = "../Figures/fig_8_all_scenarios.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as {plot_filename}")
