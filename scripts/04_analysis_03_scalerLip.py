# -*- coding: utf-8 -*-
"""
scripts/04_analysis_03_scalerLip.py

[Task 3: Scaler Lipschitz Constant Estimation]

This script empirically estimates the Lipschitz constant of the data preprocessing 
pipeline (the Scikit-learn scaler). Since the total system Lipschitz constant is 
the product of the scaler's constant and the neural network's constant, determining 
this value is crucial for the theoretical error bounds in Chapter 4.

Method:
1. Loads the trained `scaler.joblib` and the original unscaled training data.
2. Randomly samples a large number of pairs of points (x1, x2) from the data.
3. Computes the expansion ratio: ||Scaler(x1) - Scaler(x2)|| / ||x1 - x2||.
4. The maximum observed ratio serves as an empirical lower bound for the Lipschitz constant.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys, traceback
import json

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.resolve()))
try:
    from src.config import *
    from t_train_Transformer import InvertableColumnTransformer
except ImportError as e:
    traceback.print_exc()
    raise ImportError(
        f"Could not import `InvertableColumnTransformer` from `t_train_Transformer.py`. "
        f"Please ensure both scripts are in the same directory or the path is correctly set. Error: {e}"
    )


def estimate_lipschitz(args):
    """
    Main function to load data and scaler, then estimate the constant.
    """
    # Construct paths
    exp_dir = Path(args.experiment_dir)
    scaler_path = exp_dir / "scaler.joblib"
    config_path = exp_dir / "config.json"
    data_path = Path(args.data_path)

    print("--- Starting Scaler Lipschitz Constant Estimation ---")

    # --- 1. Load Components ---
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found in experiment directory: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading scaler from: {scaler_path}")
    # Load the fitted scaler pipeline
    scaler = joblib.load(scaler_path)
    
    # Retrieve the exact column names used during training from config.json.
    # This ensures the dataframe passed to the scaler has the correct schema.
    print(f"Loading state column names from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        state_columns = cfg.get('STATE_COLS_DEFAULT')
        if not state_columns:
            raise ValueError("`STATE_COLS_DEFAULT` not found or is empty in config.json.")

    # Load the raw (unscaled) training data.
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, usecols=state_columns)
    # Reorder columns to match scaler expectation
    df = df[state_columns]
    
    print(f"Loaded {len(df)} samples with {len(state_columns)} state features.")

    # --- 2. Sample Point Pairs ---
    print(f"Sampling {args.num_pairs} point pairs...")
    n_samples = len(df)
    if n_samples < 2:
        raise ValueError("Not enough data to sample pairs.")

    # Randomly select indices for pairs (x1, x2).
    indices1 = np.random.choice(n_samples, args.num_pairs, replace=True)
    indices2 = np.random.choice(n_samples, args.num_pairs, replace=True)
    
    # Resample to ensure x1 != x2 to avoid division by zero (0/0).
    mask = indices1 == indices2
    while np.any(mask):
        indices2[mask] = np.random.choice(n_samples, np.sum(mask), replace=True)
        mask = indices1 == indices2

    # Extract the actual data points.
    x1_samples = df.iloc[indices1].values
    x2_samples = df.iloc[indices2].values
    
    # --- 3. Transform and Compute Ratios ---
    print("Transforming samples and computing distance ratios...")
    # Apply the scaler transformation to get y = Scaler(x)
    y1_samples = scaler.transform(pd.DataFrame(x1_samples, columns=state_columns))
    y2_samples = scaler.transform(pd.DataFrame(x2_samples, columns=state_columns))
    
    max_ratio = 0.0
    
    # Process in batches to manage memory usage for large N.
    batch_size = 10000
    for i in tqdm(range(0, args.num_pairs, batch_size), desc="Calculating Ratios"):
        batch_x1 = x1_samples[i:i+batch_size]
        batch_x2 = x2_samples[i:i+batch_size]
        batch_y1 = y1_samples[i:i+batch_size]
        batch_y2 = y2_samples[i:i+batch_size]
        
        # Calculate Euclidean (L2) distance in input space (X) and output space (Y).
        norm_x_diff = np.linalg.norm(batch_x1 - batch_x2, axis=1)
        norm_y_diff = np.linalg.norm(batch_y1 - batch_y2, axis=1)
        
        # Filter out extremely close points to avoid numerical instability, 
        # though x1!=x2 check earlier handles exact duplicates.
        valid_mask = norm_x_diff > 1e-9
        if not np.any(valid_mask):
            continue
            
        # Lipschitz ratio = ||y1 - y2|| / ||x1 - x2||
        ratios = norm_y_diff[valid_mask] / norm_x_diff[valid_mask]
        
        # Update the maximum observed ratio.
        batch_max = np.max(ratios)
        if batch_max > max_ratio:
            max_ratio = batch_max

    # --- 4. Report Result ---
    print("\n--- Scaler Lipschitz Constant Estimation Summary ---")
    print(f"Number of pairs sampled: {args.num_pairs}")
    print(f"[To replace the value in Chapter 4]")
    print(f"Empirically Estimated Lipschitz Constant: {max_ratio:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate the Lipschitz constant of a scikit-learn scaler.")
    parser.add_argument("--experiment_dir", type=str, required=True, default=str(PATHS["dynamics_model_dir"]), help="Path to the experiment directory containing scaler.joblib.")
    parser.add_argument("--data_path", type=str, default=str(PATHS["processed_data_dir"] / "training_data_stepwise" / "strategy_comparison_stepwise_1000.csv"), help="Path to the original (unscaled) training data CSV file.")
    parser.add_argument("--num_pairs", type=int, default=100000, help="Number of random point pairs to sample for the estimation.")
    
    args = parser.parse_args()
    estimate_lipschitz(args)
