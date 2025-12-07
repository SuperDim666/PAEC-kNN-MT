# -*- coding: utf-8 -*-
"""
scripts/04_analysis_04_optim_gap.py

[Task 4: Optimization Gap Analysis]

This script empirically verifies the "sub-optimality gap" (epsilon_sub) assumption
used in Theorem 4.2.4. It compares the optimal Lyapunov value V(S_next) found by two methods:
1. The Hybrid Strategy (Our Method): Uses discrete enumeration + gradient descent (Teacher strategy).
2. Brute-force Grid Search (Approximation of Global Optimum): Exhaustively searches the action space.

The resulting 'Mean Gap' quantifies how close our hybrid optimizer gets to the true global 
minimum, providing empirical backing for the ~3% gap claim in the paper.
"""
import argparse, json, torch, importlib, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add project root to path to enable imports
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.core import Action
from t_train_Transformer import setup_data_and_model_components, split_state, lyapunov_V, InvertableColumnTransformer

# Dynamically import the policy network module to access the optimization logic
try:
    policy_net_module = importlib.import_module("scripts.05_train_policy_network")
    compute_optimal_action = policy_net_module.compute_optimal_action
    _action_tensor_to_object = policy_net_module._action_tensor_to_object
    _action_to_tensor = policy_net_module._action_to_tensor
except ImportError as e:
    print(f"CRITICAL: Failed to import scripts.05_train_policy_network: {e}")
    sys.exit(1)

def analyze_optim_gap(args):
    """
    Main function to run the gap analysis.
    """
    print("--- Starting Optimization Gap Analysis ---")
    
    # 1. Load Components (Model, Data, V function)
    exp_dir = Path(args.experiment_dir)
    config_path = exp_dir / "config.json"
    checkpoint_path = exp_dir / "checkpoint_best.pt"
    
    if not config_path.exists(): raise FileNotFoundError("Config not found")
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        train_args_dict = json.load(f)
    
    train_args = argparse.Namespace(**train_args_dict)
    train_args.device = args.device
    
    # Explicitly patch t_train_Transformer global variables.
    # The module relies on these globals (S_DIM, indices) to construct the model correctly.
    # We must cast them to native Python types (int, list) to avoid JIT/serialization issues.
    import t_train_Transformer as t_train
    
    t_train.STATE_COLS_DEFAULT = list(train_args.STATE_COLS_DEFAULT)
    t_train.S_DIM = int(train_args.S_DIM)
    t_train.E_DIM = int(train_args.E_DIM)
    t_train.PHI_DIM = int(train_args.PHI_DIM)
    t_train.H_DIM = int(train_args.H_DIM)
    
    # Indices must be lists/tuples of ints for slicing: [start, end]
    t_train.E_INDEX = [int(x) for x in train_args.E_INDEX]
    t_train.PHI_INDEX = [int(x) for x in train_args.PHI_INDEX]
    t_train.H_INDEX = [int(x) for x in train_args.H_INDEX]

    print(f"[INFO] t_train_Transformer Globals Updated:")
    print(f"       E_INDEX={t_train.E_INDEX}, PHI_INDEX={t_train.PHI_INDEX}, H_INDEX={t_train.H_INDEX}")
    print(f"       S_DIM={t_train.S_DIM}")
    
    # Setup model components using the standard helper function
    components = setup_data_and_model_components(train_args, np.random.default_rng(42))
    model = components['model']
    val_loader = components['val_loader']
    V_fn = components['V_fn']
    
    # Load Model Weights
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    # Unwrap compiled model keys if necessary
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Define Parameters for Brute-force Grid Search
    # We define a dense grid to approximate the global optimum.
    # K: 1 to 16 (discrete steps)
    # Lambda: 0.0 to 1.0 with step 0.05
    GRID_K = list(range(1, 17))
    GRID_LAMBDA = np.linspace(0.0, 1.0, 21).tolist() 
    INDEX_TYPES = ['exact', 'hnsw', 'ivf_pq']
    
    results = []
    
    print(f"Sampling {args.num_samples} states for gap analysis...")
    
    iterator = iter(val_loader)
    samples_processed = 0
    
    with torch.no_grad():
        while samples_processed < args.num_samples:
            try:
                batch = next(iterator)
            except StopIteration:
                break
                
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            B = batch['S_t'].shape[0]
            
            # --- A. Run Hybrid Strategy (The "Our Method") ---
            # This executes the same optimization logic used to generate Teacher labels.
            hybrid_results = compute_optimal_action(
                model, V_fn, 
                batch['S_t'], batch['S_hist'], batch['A_hist'],
                batch.get('decoder_hidden_state'),
                batch.get('source_embeddings'),
                batch.get('prefix_embeddings'),
                train_args_dict, # Pass config dict containing optimization params
                args.device
            )
            
            # --- B. Run Brute-force Grid Search (The "Global Optimum") ---
            # We iterate manually over each sample to perform the grid search.
            for i in range(B):
                if samples_processed >= args.num_samples: break
                
                # 1. Retrieve the optimal V found by the Hybrid Strategy
                # hybrid_results[i] contains results for all index types; we take the best one.
                best_type_hybrid = hybrid_results[i]['best_knn_type']
                v_hybrid = hybrid_results[i]['details'][best_type_hybrid]['cost']
                
                # 2. Compute Global V via Grid Search
                v_global_min = float('inf')
                
                # Extract inputs for the single sample
                s_t = batch['S_t'][i:i+1]
                s_hist = batch['S_hist'][i:i+1]
                a_hist = batch['A_hist'][i:i+1]
                hs = batch['decoder_hidden_state'][i:i+1] if 'decoder_hidden_state' in batch else None
                src = batch['source_embeddings'][i:i+1] if 'source_embeddings' in batch else None
                pref = batch['prefix_embeddings'][i:i+1] if 'prefix_embeddings' in batch else None
                
                # Calculate the "Heavy" encoder part once (cacheable)
                cache = model.forward_heavy_cache(s_t, s_hist, a_hist, hs, src, pref)
                
                # Iterate over the dense grid of actions
                for idx_type in INDEX_TYPES:
                    # Construct one-hot vector for index type
                    idx_vec = torch.zeros(1, 4, device=args.device)
                    # Map: none=0, exact=1, hnsw=2, ivf_pq=3
                    type_map = {'exact': 1, 'hnsw': 2, 'ivf_pq': 3}
                    idx_vec[0, type_map[idx_type]] = 1.0
                    
                    for k_val in GRID_K:
                        k_norm = k_val / 16.0
                        for l_val in GRID_LAMBDA:
                            # Construct full action tensor: [1, 6] -> [one_hot(4), k_norm, lambda]
                            cont_vec = torch.tensor([[k_norm, l_val]], device=args.device)
                            action = torch.cat([idx_vec, cont_vec], dim=1)
                            
                            # Run the "Light" head to get prediction
                            E_pred, H_pred, _ = model.forward_light_head(action, cache)
                            
                            # Reconstruct Next State (using Inertia Assumption for Phi)
                            S_next = torch.cat([E_pred, cache['Phi_t'], H_pred], dim=1)
                            
                            # Calculate Lyapunov Cost V(S_next)
                            v_curr = V_fn(S_next).item()
                            
                            if v_curr < v_global_min:
                                v_global_min = v_curr
                
                # 3. Calculate Relative Gap
                # Gap = (V_hybrid - V_global) / V_hybrid
                # Ideally V_hybrid >= V_global. If hybrid is better (due to grid granularity), gap is 0.
                if v_hybrid < v_global_min:
                    gap = 0.0 # Hybrid found a point between grid steps that was better
                else:
                    # Add epsilon to denominator to avoid division by zero
                    gap = (v_hybrid - v_global_min) / (v_hybrid + 1e-9)
                
                results.append(gap)
                samples_processed += 1
                
                if samples_processed % 50 == 0:
                    print(f"Processed {samples_processed}/{args.num_samples}. Current Mean Gap: {np.mean(results):.4f}")

    # --- Report Results ---
    mean_gap = np.mean(results)
    p95_gap = np.percentile(results, 95)
    
    print("\n" + "="*50)
    print("RESULTS: Sub-optimality Gap Analysis")
    print("="*50)
    print(f"Total Samples: {len(results)}")
    print(f"[Key Result] Mean Gap (epsilon_sub): {mean_gap:.4f} ({(mean_gap*100):.2f}%)")
    print(f"95th Percentile Gap: {p95_gap:.4f}")
    print("-" * 50)
    
    if mean_gap <= 0.035:
        print("[Successful] Empirical gap is VERIFIED within the claimed ~3% range.")
    else:
        print(f"[Warning] Empirical gap {mean_gap:.4f} exceeds the claimed 3%!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    analyze_optim_gap(args)
