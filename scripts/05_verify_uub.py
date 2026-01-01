# -*- coding: utf-8 -*-
"""
scripts/05_verify_uub.py

[Task 5: UUB Assumption Verification - PHYSICAL SPACE FIXED]

Fixes logical fallacy by performing verification in the ORIGINAL (Physical) space.
Includes fix for NameError: batch_norms -> batch_delta_norms.
"""

import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# Import internal modules
import t_train_Transformer
from t_train_Transformer import (
    setup_data_and_model_components,
    split_state,
    lyapunov_V,
    InvertableColumnTransformer,
    to_device
)

def _set_rand_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def verify_assumptions(args):
    exp_dir = Path(args.experiment_dir)
    print(f"=== UUB Assumption Verification (Physical Space) ===\nTarget: {exp_dir.name}")
    
    # 1. Load Experiment Config
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        train_config = json.load(f)

    # --- INJECT GLOBALS ---
    print("[Info] Injecting state configurations into t_train_Transformer globals...")
    if 'STATE_COLS_DEFAULT' not in train_config:
        raise ValueError("Config missing 'STATE_COLS_DEFAULT'.")

    t_train_Transformer.STATE_COLS_DEFAULT = train_config['STATE_COLS_DEFAULT']
    t_train_Transformer.E_INDEX = train_config['E_INDEX']
    t_train_Transformer.PHI_INDEX = train_config['PHI_INDEX']
    t_train_Transformer.H_INDEX = train_config['H_INDEX']
    t_train_Transformer.S_DIM = train_config['S_DIM']
    t_train_Transformer.E_DIM = train_config['E_DIM']
    t_train_Transformer.PHI_DIM = train_config['PHI_DIM']
    t_train_Transformer.H_DIM = train_config['H_DIM']
    
    state_cols = train_config['STATE_COLS_DEFAULT']
    E_indices = list(range(train_config['E_INDEX'][0], train_config['E_INDEX'][1] + 1))
    
    # 2. Pipeline Init
    if args.data_path:
        train_config['val_path'] = str(args.data_path)
    
    train_config['device'] = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    train_config['batch_size'] = args.batch_size
    pipeline_args = argparse.Namespace(**train_config)
    
    print("\nStep 1: Initializing Pipeline...")
    rng = np.random.default_rng(args.seed)
    components = setup_data_and_model_components(pipeline_args, rng)
    
    val_loader = components['val_loader']
    model = components['model']
    scaler = components['scaler'] # Need scaler for inverse transform
    
    print(f"  Validation Set: {len(val_loader.dataset)} samples")

    # 3. Load Weights
    print("\nStep 2: Loading Weights...")
    checkpoint_path = exp_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists(): checkpoint_path = exp_dir / "checkpoint_last.pt"
    
    checkpoint = torch.load(checkpoint_path, map_location=pipeline_args.device)
    state_dict = checkpoint.get('model', checkpoint.get('model_state_dict'))
    if state_dict and list(state_dict.keys())[0].startswith('_orig_mod.'):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 4. Define Physical Space Lyapunov Function
    device = torch.device(pipeline_args.device)
    
    # Standard L2 Norm Squared in Physical Space as Lyapunov Candidate
    def V_fn_physical(E_physical_batch):
        return torch.sum(E_physical_batch ** 2, dim=1)

    # 5. Verification Loop
    print("\nStep 3: Verifying in PHYSICAL Space...")
    
    delta_norms = []
    delta_E_norms = []
    clf_satisfied = 0
    total_samples = 0
    rho = train_config.get('rho', 0.1)
    predict_delta = train_config.get('predict_delta', False)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = to_device(batch, device)
            
            S_t_scaled = batch["S_t"]
            S_hist_scaled = batch["S_hist"]
            A_t = batch["A_t"]
            A_hist = batch["A_hist"]
            
            # Forward Pass (Scaled Space)
            model_output = model(
                S_t_scaled, S_hist_scaled, A_t, A_hist,
                decoder_hidden_state=batch.get("decoder_hidden_state"),
                source_embeddings=batch.get("source_embeddings"), 
                prefix_embeddings=batch.get("prefix_embeddings")
            )
            E_pred_scaled, H_pred_scaled, _ = model_output
            
            # --- RECONSTRUCTION IN SCALED SPACE ---
            S_t_base_scaled = S_t_scaled[:, :t_train_Transformer.S_DIM]
            E_t_scaled, Phi_t_scaled, H_t_scaled = split_state(S_t_base_scaled)
            
            if predict_delta:
                E_next_scaled = E_t_scaled + E_pred_scaled
                H_next_scaled = H_t_scaled + H_pred_scaled
            else:
                E_next_scaled = E_pred_scaled
                H_next_scaled = H_pred_scaled
            
            # Construct Full Predicted State (Scaled)
            S_next_pred_scaled = torch.cat([E_next_scaled, Phi_t_scaled, H_next_scaled], dim=1)
            
            # Get Ground Truth Next State (Scaled)
            S_tp1_target_raw = batch["S_tp1"][:, :t_train_Transformer.S_DIM]
            if predict_delta:
                S_next_true_scaled = S_t_base_scaled + S_tp1_target_raw
            else:
                S_next_true_scaled = S_tp1_target_raw

            # --- INVERSE TRANSFORM TO PHYSICAL SPACE ---
            S_t_np = S_t_base_scaled.cpu().numpy()
            S_next_pred_np = S_next_pred_scaled.cpu().numpy()
            S_next_true_np = S_next_true_scaled.cpu().numpy()
            
            df_S_t = pd.DataFrame(S_t_np, columns=state_cols)
            df_S_pred = pd.DataFrame(S_next_pred_np, columns=state_cols)
            df_S_true = pd.DataFrame(S_next_true_np, columns=state_cols)
            
            S_t_phys = scaler.inverse_transform(df_S_t)
            S_pred_phys = scaler.inverse_transform(df_S_pred)
            S_true_phys = scaler.inverse_transform(df_S_true)
            
            S_t_phys = torch.tensor(S_t_phys, device=device, dtype=torch.float32)
            S_pred_phys = torch.tensor(S_pred_phys, device=device, dtype=torch.float32)
            S_true_phys = torch.tensor(S_true_phys, device=device, dtype=torch.float32)
            
            # Extract E components (Physical)
            E_t_phys = S_t_phys[:, E_indices]
            E_pred_phys = S_pred_phys[:, E_indices]
            E_true_phys = S_true_phys[:, E_indices]
            
            # --- B1: Model Error (Physical) ---
            diff = S_pred_phys - S_true_phys
            batch_delta_norms = torch.norm(diff, dim=1).cpu().numpy()
            delta_norms.extend(batch_delta_norms) # <--- FIXED HERE
            
            batch_E_norms = torch.norm(E_pred_phys - E_true_phys, dim=1).cpu().numpy()
            delta_E_norms.extend(batch_E_norms)

            # --- B3: CLF Condition (Physical) ---
            V_curr = V_fn_physical(E_t_phys)
            V_next_pred = V_fn_physical(E_pred_phys)
            
            # Relaxed check for numerical stability
            threshold = 1e-4
            target = (1 - rho) * V_curr
            satisfied = (V_next_pred <= target) | (V_next_pred < threshold)
            
            clf_satisfied += satisfied.float().sum().item()
            total_samples += S_t_scaled.shape[0]

    # 7. Statistics
    delta_norms = np.array(delta_norms)
    delta_E_norms = np.array(delta_E_norms)
    
    b1_total_mean = np.mean(delta_norms)
    b1_total_p99 = np.percentile(delta_norms, 99)
    b1_total_max = np.max(delta_norms)
    
    b1_e_p99 = np.percentile(delta_E_norms, 99)
    
    # Noise Proxy (Heuristic: Noise ~ 0.5 * Prediction Error)
    xi_proxy = np.std(delta_norms) * 0.5
    
    clf_rate = clf_satisfied / total_samples
    
    # UUB Bound b = (delta + xi)^2 / rho
    b_val = (b1_total_p99 + xi_proxy)**2 / rho
    
    print(f"\n=== Physical Space Verification Results ===")
    print(f"B1: Model Error P99 (Mean)  = {b1_total_mean:.4f}")
    print(f"B1: Model Error P99 (Total) = {b1_total_p99:.4f}")
    print(f"B1: Model Error P99 (Max)   = {b1_total_max:.4f}")
    print(f"B1: Model Error P99 (Error) = {b1_e_p99:.4f}")
    print(f"B2: Noise Proxy (Est)       = {xi_proxy:.4f}")
    print(f"B3: CLF Satisfaction        = {clf_rate*100:.2f}%")
    print(f"--> UUB Bound b             ≈ {b_val:.4f}")
    
    report = {
        "b1_total_p99": float(b1_total_p99),
        "b2_noise_proxy": float(xi_proxy),
        "b3_clf_rate": float(clf_rate),
        "uub_bound_b": float(b_val)
    }
    
    out_path = exp_dir / "assumption_verification_report_physical.json"
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    _set_rand_seed(args.seed)
    verify_assumptions(args)
