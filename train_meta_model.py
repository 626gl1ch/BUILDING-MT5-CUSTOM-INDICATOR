import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def train_meta_model(db_path="strategy_learning_db.csv", model_path="meta_model.pkl"):
    print(f"Loading knowledge base from {db_path}...")
    if not os.path.exists(db_path):
        print("Database not found. Please run mass_search_sequential.py first to generate data.")
        return
        
    df = pd.read_csv(db_path)
    if len(df) < 100:
        print(f"Not enough data to train. Need at least 100 samples, found {len(df)}.")
        return
        
    print(f"Found {len(df)} strategy records. Preparing data for ML...")
    
    # We want to predict 'passed_synthetic'
    # Features: entry, regime, sl_atr, tp_atr, risk_pct, trailing
    
    # Convert categorical strings to numbers
    le_entry = LabelEncoder()
    le_regime = LabelEncoder()
    
    df['entry_encoded'] = le_entry.fit_transform(df['entry'].astype(str))
    df['regime_encoded'] = le_regime.fit_transform(df['regime'].astype(str))
    df['trailing_encoded'] = df['trailing'].astype(int)
    
    features = ['entry_encoded', 'regime_encoded', 'sl_atr', 'tp_atr', 'risk_pct', 'trailing_encoded']
    X = df[features]
    y = df['passed_synthetic'].astype(int)
    
    # Check class balance
    pass_count = y.sum()
    fail_count = len(y) - pass_count
    print(f"Class Balance -> Passed: {pass_count}, Failed: {fail_count}")
    
    if pass_count == 0 or fail_count == 0:
        print("Model requires both passed and failed strategies to learn the boundary. Keep running the search.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Meta-Labeler...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\n--- Model Performance Report ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))
    
    # Save the model and encoders
    print(f"Saving Meta-Model to {model_path}...")
    joblib.dump({
        'model': clf,
        'le_entry': le_entry,
        'le_regime': le_regime,
        'features': features
    }, model_path)
    print("Done! The search engine can now load this model to pre-filter bad parameters.")

if __name__ == "__main__":
    train_meta_model()
