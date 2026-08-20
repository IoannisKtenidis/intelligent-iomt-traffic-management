import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import os

# Scenarios to process
scenarios = [1, 2, 3]
feature_counts = [1, 2, 3, 4]

# A dictionary to store comparison results
results = []

for scenario_id in scenarios:
    filename = os.path.join(os.path.dirname(__file__), "..", "..", "data", f"dataset_scenario{scenario_id}.csv")
    if not os.path.exists(filename):
        print(f"Error: Dataset {filename} not found. Please run collect_data.py first.")
        continue
        
    print(f"\n==================================================")
    print(f"Training and Evaluating Models for Scenario {scenario_id}")
    print(f"==================================================")
    
    df = pd.read_csv(filename)
    
    # Check class distribution
    class_counts = df['label'].value_counts()
    print("Class distribution:")
    print(class_counts)
    
    for k in feature_counts:
        # Select features based on K
        features = [f"dt_{i}" for i in range(k)]
        X = df[features]
        y = df['label']
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train XGBoost Classifier
        # Use scale_pos_weight to handle class imbalance (Healthy is ~97.7%, Not-Healthy is ~2.3%)
        scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        
        model = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred) # Recall for class 1 (Not-Healthy)
        
        # Specificity: recall for class 0 (Healthy)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nFeatures (K={k}): {features}")
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Sensitivity: {sensitivity:.4f} (TP={tp}, FN={fn})")
        print(f"Specificity: {specificity:.4f} (TN={tn}, FP={fp})")
        print(f"F1 Score:    {f1:.4f}")
        
        results.append({
            "Scenario": scenario_id,
            "K_intervals": k,
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1_Score": f1
        })
        
        # Save model
        model_name = f"xgb_model_scenario{scenario_id}_k{k}.json"
        model.save_model(os.path.join(os.path.dirname(__file__), "..", "..", "models", model_name))
        print(f"Saved model to {model_name}")

# Print a final summary table
print("\n\n==================================================================")
print("FINAL SUMMARY COMPARISON TABLE")
print("==================================================================")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(os.path.dirname(__file__), "..", "..", "Results_Data", "xgboost_metrics_summary.csv"), index=False)
print("Summary saved to xgboost_metrics_summary.csv")
