# -*- coding: utf-8 -*-
"""
scripts/04_analysis_02_hessian.py

[Task 2: Hessian Strong Convexity Analysis]

This script empirically validates the strong convexity assumption required for 
Theorem 4.4 in the PAEC framework. It calculates the Hessian matrix of the 
optimization objective (Lyapunov value V of the predicted next state) with 
respect to the continuous action parameters (k, lambda).

The analysis involves:
1. Loading a trained dynamics model (T_theta) and its scaler.
2. Iterating through validation samples.
3. Defining the objective function: J(A) = V(T_theta(S_t, A)).
4. Computing the Hessian matrix using PyTorch's autograd.
5. Calculating the minimum eigenvalue of the Hessian to check for positive definiteness.
"""
import argparse, json, torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# Import the model definition and data setup utilities.
# These modules must be accessible in the path.
import t_train_Transformer
from t_train_Transformer import InvertableColumnTransformer, setup_data_and_model_components, PAECDataset, PAECTransition, lyapunov_V, split_state

def analyze_hessian(args):
    """
    Main execution function for Hessian analysis.
    
    Args:
        args: Command line arguments containing experiment directory and device settings.
    """
    exp_dir = Path(args.experiment_dir)
    print(f"--- Starting Hessian Convexity Analysis for experiment: {exp_dir.name} ---")

    # --- 1. Load Components ---
    print("Step 1: Loading model, scaler, and validation data...")
    config_path = exp_dir / "config.json"
    checkpoint_path = exp_dir / "checkpoint_best.pt"
    scaler_path = exp_dir / "scaler.joblib"

    # Ensure all required artifacts exist.
    if not all([config_path.exists(), checkpoint_path.exists(), scaler_path.exists()]):
        raise FileNotFoundError("Experiment directory must contain config.json, checkpoint_best.pt, and scaler.joblib")

    # Load configuration used during training.
    with open(config_path, 'r', encoding='utf-8') as f:
        saved_args_dict = json.load(f)
    # Convert dictionary to Namespace object for compatibility with setup functions.
    train_args = argparse.Namespace(**saved_args_dict)
    
    # Override device if specified by the user.
    if args.device:
        train_args.device = args.device

    # The `t_train_Transformer` module relies on global variables (S_DIM, E_DIM, etc.) 
    # to define model architecture. We must inject the values from the loaded config 
    # into the module's namespace before initializing the model.
    print("Setting up global context from config.json for t_train_Transformer module...")
    t_train_Transformer.STATE_COLS_DEFAULT = train_args.STATE_COLS_DEFAULT
    t_train_Transformer.S_DIM = train_args.S_DIM
    t_train_Transformer.E_DIM = train_args.E_DIM
    t_train_Transformer.PHI_DIM = train_args.PHI_DIM
    t_train_Transformer.H_DIM = train_args.H_DIM
    t_train_Transformer.E_INDEX = train_args.E_INDEX
    t_train_Transformer.PHI_INDEX = train_args.PHI_INDEX
    t_train_Transformer.H_INDEX = train_args.H_INDEX

    # Initialize data loaders and model architecture using the shared setup function.
    # We use a fixed seed for reproducibility.
    mock_rng = np.random.default_rng(train_args.seed)
    components = setup_data_and_model_components(train_args, mock_rng)

    model = components['model']
    val_loader = components['val_loader']
    V_fn = components['V_fn']
    scaler = components['scaler']
    
    # Load learned weights into the model.
    checkpoint = torch.load(checkpoint_path, map_location=train_args.device)
    state_dict = checkpoint['model']
    # Handle potential prefix issues if the model was saved from a compiled version.
    if list(state_dict.keys())[0].startswith('_orig_mod.'):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Components loaded successfully.")

    # --- 2. Iterate and Compute Hessian ---
    print(f"Step 2: Iterating through validation set ({len(val_loader.dataset)} samples)...")
    results = []
    
    # Limit the analysis to a subset if specified to save time.
    num_samples_to_process = min(args.num_samples, len(val_loader.dataset))
    
    pbar = tqdm(total=num_samples_to_process)
    samples_processed = 0

    for batch in val_loader:
        if samples_processed >= num_samples_to_process:
            break

        # Move batch data to the appropriate device.
        batch = {k: v.to(train_args.device) for k, v in batch.items() if torch.is_tensor(v)}
        
        S_t_batch = batch['S_t']
        A_t_batch = batch['A_t']
        
        # Calculate the unscaled error norm for categorization (Low vs High Error regions).
        S_t_unscaled_df = pd.DataFrame(S_t_batch.cpu().numpy(), columns=train_args.STATE_COLS_DEFAULT)
        S_t_unscaled = scaler.inverse_transform(S_t_unscaled_df)
        E_t_unscaled = S_t_unscaled[:, train_args.E_INDEX[0]:train_args.E_INDEX[1]+1]
        E_norms = np.linalg.norm(E_t_unscaled, axis=1)

        # Process each sample in the batch individually.
        for i in range(S_t_batch.shape[0]):
            if samples_processed >= num_samples_to_process:
                break

            # Isolate single sample tensors.
            S_t, A_t = S_t_batch[i], A_t_batch[i]
            S_hist, A_hist = batch['S_hist'][i], batch['A_hist'][i]
            src_emb = batch['source_embeddings'][i] if 'source_embeddings' in batch else None
            pref_emb = batch['prefix_embeddings'][i] if 'prefix_embeddings' in batch else None
            dec_hs = batch['decoder_hidden_state'][i] if 'decoder_hidden_state' in batch else None
            
            # Define the scalar objective function: Action -> Next State -> Lyapunov Value.
            # This closure captures the static context (S_t, history, embeddings).
            def objective_function(action_tensor):
                # Run model forward pass (unsqueeze to add batch dimension of 1)
                model_output = model(
                    S_t.unsqueeze(0), S_hist.unsqueeze(0),
                    action_tensor.unsqueeze(0), A_hist.unsqueeze(0),
                    source_embeddings=src_emb.unsqueeze(0) if src_emb is not None else None,
                    prefix_embeddings=pref_emb.unsqueeze(0) if pref_emb is not None else None,
                    decoder_hidden_state=dec_hs.unsqueeze(0) if dec_hs is not None else None
                )
                E_pred, H_pred, _ = model_output
                
                # Reconstruct full state using Inertia Assumption for Phi (Pressure state).
                _, Phi_t, _ = split_state(S_t.unsqueeze(0))
                S_pred = torch.cat([E_pred, Phi_t, H_pred], dim=1)
                
                # Return the scalar Lyapunov value V(S_pred).
                return V_fn(S_pred).squeeze()

            # Compute the Hessian matrix of V w.r.t Action A using numerical differentiation.
            hessian_matrix = torch.autograd.functional.hessian(objective_function, A_t)
            
            # Calculate eigenvalues to determine convexity.
            # Positive definite Hessian (all eigenvalues > 0) implies local convexity.
            min_eigenvalue = torch.linalg.eigvalsh(hessian_matrix).min().item()
            
            results.append({
                "min_eigenvalue": min_eigenvalue,
                "e_norm": E_norms[i]
            })
            
            samples_processed += 1
            pbar.update(1)
            
    pbar.close()

    # --- 3. Analyze and Report Results ---
    print("\nStep 3: Analyzing results...")
    if not results:
        print("No results to analyze.")
        return

    df = pd.DataFrame(results)

    # Calculate overall statistics.
    total_samples = len(df)
    positive_definite_count = (df['min_eigenvalue'] > 0).sum()
    positive_definite_ratio = positive_definite_count / total_samples if total_samples > 0 else 0
    avg_min_eigenvalue = df['min_eigenvalue'].mean()

    # Calculate statistics specifically for the high-error region (where convexity matters most for control).
    high_error_df = df[df['e_norm'] > 0.5]
    high_error_samples = len(high_error_df)
    high_error_pd_count = (high_error_df['min_eigenvalue'] > 0).sum()
    high_error_pd_ratio = high_error_pd_count / high_error_samples if high_error_samples > 0 else 0

    # Output summary report.
    print("\n--- Hessian Convexity Analysis Summary ---")
    print(f"Total Samples Analyzed: {total_samples}")
    print(f"Overall Positive Definite Ratio (λ_min > 0): {positive_definite_ratio:.4f} ({positive_definite_count}/{total_samples})")
    print(f"Average Minimum Eigenvalue (E[λ_min]): {avg_min_eigenvalue:.4f}")
    print(f"Positive Definite Ratio in High-Error Region (||E|| > 0.5): {high_error_pd_ratio:.4f} ({high_error_pd_count}/{high_error_samples})")
    
    # Save detailed CSV for further plotting or verification.
    output_path = exp_dir / "hessian_analysis_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hessian convexity analysis for a trained model.")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment directory (e.g., './drive/...')")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of validation samples to analyze.")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., 'cpu' or 'cuda')")
    
    args = parser.parse_args()
    analyze_hessian(args)
