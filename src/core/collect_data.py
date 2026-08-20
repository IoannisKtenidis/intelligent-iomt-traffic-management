import random
import pandas as pd
import os

# Configured scenarios
scenarios = {
    1: {"p": 0.018, "q": 0.764},
    2: {"p": 0.020, "q": 0.849},
    3: {"p": 0.022, "q": 0.934}
}

# Number of transmissions to simulate per scenario
num_samples = 500000

for scenario_id, params in scenarios.items():
    p = params["p"]
    q = params["q"]
    
    print(f"Collecting data for Scenario {scenario_id} (p={p}, q={q})...")
    
    # Initialize state based on stationary probability
    prob_healthy = q / (p + q)
    state = 'Healthy' if random.random() < prob_healthy else 'Not-Healthy'
    
    intervals = []
    states = []
    
    for _ in range(num_samples):
        # 1. State transition
        if state == 'Healthy':
            if random.random() < p:
                state = 'Not-Healthy'
        else:
            if random.random() < q:
                state = 'Healthy'
        
        # 2. Get lambda based on state
        if state == 'Healthy':
            lam = 1.96
        else:
            if random.random() < 0.257:
                lam = 2.72
            else:
                lam = 12.0
        
        # Generate transmission interval in ms
        interval = random.expovariate(lam / 86400000.0)
        
        intervals.append(interval)
        states.append(1 if state == 'Not-Healthy' else 0)
        
    # Build dataset with sliding window of 4 intervals as features
    # dt_0: current interval
    # dt_1: previous interval
    # dt_2: 2 intervals ago
    # dt_3: 3 intervals ago
    # label: true state at current transmission
    data = []
    for i in range(3, num_samples):
        data.append({
            "dt_0": intervals[i],
            "dt_1": intervals[i-1],
            "dt_2": intervals[i-2],
            "dt_3": intervals[i-3],
            "label": states[i]
        })
        
    df = pd.DataFrame(data)
    filename = os.path.join(os.path.dirname(__file__), "..", "..", "data", f"dataset_scenario{scenario_id}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} samples to {filename}")
print("Data collection complete!")
