import os, sys, traceback
import json
import argparse
import torch
from pathlib import Path

# --- Path Setup and Import ---
# Add the project root directory to sys.path to enable importing modules from 'src'.
try:
    sys.path.append(str(Path(__file__).parent.parent.resolve()))
    from src.config import *
except ImportError:
    # Critical error handling if configuration cannot be loaded.
    print("[Error] Could not import DATA_LOADER_PARAMS from src.config.")
    print("  Ensure 'src/config.py' exists and defines DATA_LOADER_PARAMS.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    # Catch-all for other import errors.
    print(f"[Error] An unexpected error occurred during config import: {e}")
    traceback.print_exc()
    sys.exit(1)

def get_golden_config_defaults():
    """
    Constructs the canonical ("golden") configuration dictionary.
    
    This function replicates the argument parsing logic found in the training scripts
    (e.g., t_train_Transformer.py) to establish a complete set of default hyperparameters
    and structural definitions (like state dimensions). This serves as the standard
    against which existing configuration files will be normalized.
    
    Returns:
        dict: A dictionary containing all standard arguments and their default values.
    """
    parser = argparse.ArgumentParser()
    
    # ==============================================================================
    # 1. Define all standard arguments to match the training script interface
    # ==============================================================================
    
    # --- Dataset Parameters ---
    # Paths for training data, caching, and validation.
    parser.add_argument("--train_path", type=str, default=str(PATHS["processed_data_dir"] / "training_data_stepwise" / "strategy_comparison_stepwise_1000.csv"))
    parser.add_argument("--train_path_base_cut_num", type=int, default=2)
    parser.add_argument("--cache_path", type=str, default=str(PATHS["cache_dir"]))
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="When val_path is empty or invalid, the ratio of the validation set to the training set")
    
    # Data loading and processing settings.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=min(8, (os.cpu_count() or 1) // 2))
    parser.add_argument("--disable_autocast", action="store_true", help="Disable autocast of model training")
    parser.add_argument("--disable_compile", action="store_true", help="Disable compilation of model training")
    parser.add_argument("--target_total_steps", type=int, default=3000)
    
    
    # --- Model Architecture Parameters ---
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--hid_dim", type=int, default=64)
    model_group.add_argument("--layers", type=int, default=3)
    model_group.add_argument("--nhead", type=int, default=4, help="Number of attention heads for the Transformer.")
    model_group.add_argument("--history_len", type=int, default=4, help="Number of historical states to use as input for the sequence model.")
    model_group.add_argument("--predict_delta", action="store_true", help="Change the model to predict the state change (S_tp1 - S_t) instead of the full next state.")
    model_group.add_argument("--use_text_embeddings", action="store_true", help="Enhance model input with source and prefix text embeddings.")
    model_group.add_argument("--use_separate_heads_eh", action="store_true", help="Use separate heads for predicting E, (Phi), and H vectors.")
    model_group.add_argument("--use_multi_heads", action="store_true", help="Use Multi-Head transformer for predicting E, (Phi), H vectors aligning with actions.")
    
    # Decoder hidden state integration settings.
    decoder_hidden_state_group = parser.add_argument_group("Hidden State of Decoder")
    decoder_hidden_state_group.add_argument("--use_decoder_hidden_state", action="store_true", help="Enhance model input with the NMT decoder's hidden state from the previous step.")
    decoder_hidden_state_group.add_argument("--decoder_hidden_state_dim", type=int, default=1024, help="Dimensionality of the NMT decoder's hidden state.")

    # --- General Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=str(PATHS["dynamics_model_dir"] / "current"), help="Output directory (checkpoints/logs/config/Scaler)")
    parser.add_argument("--use_uncertainty_weights", action="store_true", help=f"Use self-learning uncertainty to weigh the component losses with same dimension of S.")

    # --- Stability and Control Loss Parameters (CLF) ---
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--lambda_clf", type=float, default=0.0)
    parser.add_argument("--jacobian_reg", type=float, default=0.0)
    parser.add_argument("--teacher_reweight_alpha", type=float, default=1.5)
    parser.add_argument("--softmin_tau", type=float, default=0.5)
    parser.add_argument("--rollout_H", type=int, default=12)
    parser.add_argument("--num_delta_dirs", type=int, default=2, help="The number of directions of the teacher action ± perturbation (<= dA)")
    parser.add_argument("--action_delta", type=float, default=0.25, help="The step size of the teacher action ± perturbation")

    # --- Control Barrier Function (CBF) Parameters ---
    parser.add_argument("--phi_crit", type=float, default=0.0, help="If it is positive, the CBF violation rate is counted; if it is zero or negative, it won't counted.")

    # --- Lyapunov Matrix P Learning Parameters ---
    # Configuration for V(E) = E^T P E.
    P_group = parser.add_argument_group("P Matrix Learning")
    P_group.add_argument("--learn_P", action="store_true", help="If set, then learn V(E)=E^T P E in the P diagonal.")
    P_group.add_argument("--P_init", type=float, nargs='*', default=None, help=f"Initial values for the diagonal of P in V(E)=E^T P E. Expects values in same dimension of E.")
    parser.add_argument("--skip_cols", type=float, nargs='*', default=[], help=f"Droppable columns in train_data.")

    # --- Dynamic Scheduler Parameters ---
    # Schedules for annealing loss weights and hyperparameters over epochs.
    parser.add_argument("--rho_final", type=float, default=None)
    parser.add_argument("--rho_warmup_ep", type=int, default=10)
    parser.add_argument("--lambda_clf_final", type=float, default=None)
    parser.add_argument("--lambda_warmup_ep", type=int, default=12)
    parser.add_argument("--softmin_tau_final", type=float, default=None)
    parser.add_argument("--softmin_anneal_ep", type=int, default=12)

    # --- N-Step CLF Parameters ---
    nclf_group = parser.add_argument_group("N-step CLF")
    nclf_group.add_argument("--use_nstep_clf", action="store_true")
    nclf_group.add_argument("--nstep_H", type=int, default=3)
    nclf_group.add_argument("--nstep_gamma", type=float, default=0.98)
    nclf_group.add_argument("--nstep_lambda", type=float, default=1.0)
    nclf_group.add_argument("--nstep_bptt_window", type=int, default=-1)
    
    # Action selection strategy within N-step rollout.
    nclf_selector_group = parser.add_argument_group("N-step CLF Selector")
    nclf_selector_group.add_argument("--nstep_selector", type=str, default="softmin", choices=["softmin", "gumbel_st", "hard_greedy"])
    nclf_selector_group.add_argument("--gumbel_tau_init", type=float, default=1.0)
    nclf_selector_group.add_argument("--gumbel_tau_final", type=float, default=0.1)
    nclf_selector_group.add_argument("--gumbel_anneal_ep", type=int, default=15)
    
    # CVaR (Conditional Value at Risk) settings.
    cvar_group = parser.add_argument_group("N-step CLF")
    cvar_group.add_argument("--use_cvar_loss", action="store_true")
    cvar_group.add_argument("--cvar_alpha", type=float, default=0.8)
    
    # Epsilon-Greedy exploration settings.
    epsilon_greedy_group = parser.add_argument_group("N-step CLF")
    epsilon_greedy_group.add_argument("--use_epsilon_greedy", action="store_true")
    epsilon_greedy_group.add_argument("--epsilon_init", type=float, default=0.3)
    epsilon_greedy_group.add_argument("--epsilon_final", type=float, default=0.01)
    epsilon_greedy_group.add_argument("--epsilon_decay_ep", type=int, default=15)
    epsilon_greedy_group.add_argument("--policy_entropy_weight", type=float, default=0.0)

    # --- Advanced Control Loss Parameters ---
    # Control Barrier Function settings.
    cbf_group = parser.add_argument_group("CBF Control")
    cbf_group.add_argument("--lambda_cbf", type=float, default=0.0)
    cbf_group.add_argument("--cbf_alpha", type=float, default=0.5)
    
    # Cox Proportional Hazards settings.
    cox_group = parser.add_argument_group("Cox Control")
    cox_group.add_argument("--lambda_cox", type=float, default=0.0)
    cox_group.add_argument("--cox_event_threshold", type=float, default=2.0)
    
    # Lipschitz continuity (Spectral Norm) and Action Dissimilarity (ADT).
    parser.add_argument("--lambda_adt", type=float, default=0.0)
    parser.add_argument("--use_spectral_norm", action="store_true")
    
    # --- S8 Theoretical Validation Suite Parameters ---
    # Flags to enable specific theoretical checks.
    s8_group = parser.add_argument_group("S8: Theoretical Validation Suite")
    s8_group.add_argument("--s8_enable", action="store_true")
    s8_group.add_argument("--s8_only", action="store_true", help="Skip training and run S8 validation on a matching existing model.")
    s8_group.add_argument("--s8_jacobian_robust", action="store_true")
    s8_group.add_argument("--s8_lyapunov_full", action="store_true")
    s8_group.add_argument("--s8_cbf_invariance", action="store_true")
    s8_group.add_argument("--s8_error_bounds", action="store_true")
    s8_group.add_argument("--s8_multistep_decay", action="store_true")
    s8_group.add_argument("--s8_jacobian_samples", type=int, default=256)
    s8_group.add_argument("--s8_jacobian_iters", type=int, default=30)
    s8_group.add_argument("--s8_jacobian_restarts", type=int, default=5)
    s8_group.add_argument("--s8_jacobian_rtol", type=float, default=1e-4)
    s8_group.add_argument("--s8_cbf_horizon", type=int, default=10)
    s8_group.add_argument("--s8_multistep_horizon", type=int, default=20)
    s8_group.add_argument("--s8_use_last_ckpt", action="store_true")

    # --- Learning Rate Scheduler Parameters ---
    lr_group = parser.add_argument_group("Learning Rate")
    lr_group.add_argument('--use_lr_scheduler', action='store_true')
    lr_group.add_argument('--lr_scheduler_type', type=str, default='reduce_on_plateau', choices=['reduce_on_plateau', 'cosine', 'step'])
    lr_group.add_argument('--lr_scheduler_mode', type=str, default='min', choices=['min', 'max'])
    lr_group.add_argument('--lr_scheduler_factor', type=float, default=0.5)
    lr_group.add_argument('--lr_scheduler_patience', type=int, default=3)
    lr_group.add_argument('--lr_scheduler_min_lr', type=float, default=1e-6)
    lr_group.add_argument('--lr_scheduler_monitor', type=str, default='rmse', choices=['rmse', 'loss', 'clf_violation_rate'])
    lr_group.add_argument('--lr_scheduler_t_max', type=int, default=None)
    lr_group.add_argument('--lr_scheduler_eta_min', type=float, default=1e-6)
    lr_group.add_argument('--lr_scheduler_step_size', type=int, default=5)
    lr_group.add_argument('--lr_scheduler_gamma', type=float, default=0.5)
    
    # --- Curriculum Learning Parameters ---
    curriculum_group = parser.add_argument_group("Curriculum Learning")
    curriculum_group.add_argument("--use_curriculum", action="store_true", help="Enable two-phase curriculum learning: prediction-first, then control.")
    curriculum_group.add_argument("--curriculum_phase1_epochs", type=int, default=10, help="Number of epochs for Phase 1 (prediction-only training).")
    
    # Early stopping and export settings.
    parser.add_argument("--stab_lexi_eps", type=float, default=1e-4)
    parser.add_argument("--earlystop_mode", type=str, default="rmse", choices=["rmse", "stability_first"], help="Mode for early stopping: 'rmse' or 'stability_first'.")
    parser.add_argument("--export_action_stats", action="store_true")
    parser.add_argument("--export_rollout_csv", action="store_true")
    
    parser.add_argument('--earlystop_monitor_after_epoch', type=int, default=0, help='Start monitoring for the best model only after this epoch. Crucial for curriculum learning to bypass Phase 1.')

    # Retrieve default values from argparse.
    defaults = vars(parser.parse_args([]))
    
    # ==============================================================================
    # 2. Inject Dynamic State Space Dimensions and Column Names
    # ==============================================================================
    # These values define the 11-dimensional state space:
    # Error (4D) + Pressure (3D) + Context (4D) = 11D
    # E_INDEX, PHI_INDEX, and H_INDEX define the slice boundaries for each component.
    defaults.update({
        "S_DIM": 11,
        "E_DIM": 4,
        "PHI_DIM": 3,
        "H_DIM": 4,
        "E_INDEX": [0, 3],
        "PHI_INDEX": [4, 6],
        "H_INDEX": [7, 10],
        "STATE_COLS_DEFAULT": [
            "error_semantic",
            "error_coverage",
            "error_fluency_surprisal",
            "error_fluency_repetition",
            "pressure_latency",
            "pressure_memory",
            "pressure_throughput",
            "context_faith_focus",
            "context_consistency",
            "context_stability",
            "context_confidence_volatility"
        ]
    })
    
    return defaults

def normalize_config(existing_config: dict, golden_config: dict) -> tuple[dict, bool]:
    """
    Standardizes a given configuration dictionary against the golden schema.
    
    This function ensures that the input configuration contains all required keys
    (adding missing ones from defaults) and removes any obsolete keys that are no
    longer present in the golden standard.
    
    Args:
        existing_config (dict): The configuration dictionary loaded from a file.
        golden_config (dict): The canonical configuration schema.
        
    Returns:
        tuple: (normalized_config, has_changed_flag)
    """
    normalized_config = {}
    has_changed = False
    
    existing_keys = set(existing_config.keys())
    golden_keys = set(golden_config.keys())
    
    # 1. Identify and add missing keys using default values from golden_config.
    missing_keys = golden_keys - existing_keys
    if missing_keys:
        has_changed = True
        for key in sorted(list(missing_keys)): # Sort for deterministic logging
            normalized_config[key] = golden_config[key]
            print(f"\t[Add] Added missing key: '{key}' with default value: {golden_config[key]}")

    # 2. Copy existing valid keys and filter out obsolete ones.
    for key in sorted(list(existing_keys)):
        value = existing_config[key]
        if key in golden_keys:
            normalized_config[key] = value
        else:
            has_changed = True
            print(f"\t[Delete] Removed obsolete key: '{key}'")

    # 3. Final consistency check. Even if keys match, ensure the structure/types are consistent.
    # Note: simple json dump comparison helps detect subtle changes or ordering issues.
    if json.dumps(normalized_config, sort_keys=True) != json.dumps(existing_config, sort_keys=True) and not has_changed:
        has_changed = True

    return normalized_config, has_changed

def process_directory(dir_path: Path, golden_config: dict):
    """
    Scans and updates configuration files within a specific experiment directory.
    
    This function targets `config.json` as well as pytorch checkpoints (`checkpoint_best.pt`
    and `checkpoint_last.pt`). It ensures that the configuration embedded within checkpoints
    matches the golden standard, preventing errors during model loading/inference.
    
    Args:
        dir_path (Path): The directory to process.
        golden_config (dict): The canonical configuration schema.
    """
    print(f"\nScanning: {dir_path}")
    
    config_file = dir_path / "config.json"
    best_ckpt_file = dir_path / "checkpoint_best.pt"
    last_ckpt_file = dir_path / "checkpoint_last.pt"

    # Verify that this is a valid experiment directory containing all necessary files.
    if not all([config_file.exists(), best_ckpt_file.exists(), last_ckpt_file.exists()]): return

    # --- 1. Normalize config.json ---
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            current_config = json.load(f)
        
        print("  - Processing config.json...")
        new_config, changed = normalize_config(current_config, golden_config)
        
        if changed:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, indent=2) # indent=2 for readability
            print("    [Success] Saved updated config.json.")
        else:
            print("    [Success] config.json is already up-to-date.")

    except Exception as e:
        print(f"  [Error] processing {config_file}: {e}")

    # --- 2. Update Configurations Embedded in Checkpoints ---
    for ckpt_file in [best_ckpt_file, last_ckpt_file]:
        try:
            print(f"  - Processing {ckpt_file.name}...")
            # Load the checkpoint. Map to CPU to avoid OOM or CUDA requirement errors during maintenance tasks.
            checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
            
            # Check if the checkpoint contains a 'config' dictionary.
            if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
                ckpt_config = checkpoint['config']
                new_ckpt_config, changed = normalize_config(ckpt_config, golden_config)
                
                if changed:
                    checkpoint['config'] = new_ckpt_config
                    torch.save(checkpoint, ckpt_file)
                    print(f"\t[Success] Saved updated {ckpt_file.name}.")
                else:
                    print(f"\t[Success] {ckpt_file.name} config is already up-to-date.")
            else:
                # If config is missing entirely, inject the golden config to repair the checkpoint.
                print("\t[Warning] 'config' key not found or not a dict in checkpoint. Adding it.")
                checkpoint['config'] = golden_config
                torch.save(checkpoint, ckpt_file)
                print(f"\t[Success] Saved {ckpt_file.name} with new config dictionary.")
                
        except Exception as e:
            print(f"  [Error] processing {ckpt_file}: {e}")

def main():
    """
    Main entry point for the configuration fixing script.
    
    It recursively traverses the dynamics model directory, generating a golden configuration,
    and standardizing all experiment configs found to match this schema.
    """
    base_dir = PATHS["dynamics_model_dir"]
    if not base_dir.is_dir():
        print(f"[Error] Base directory not found at '{base_dir}'. Please check the path.")
        return

    print("[Begin] Generating golden configuration standard...")
    golden_config = get_golden_config_defaults()
    print("[Success] Golden configuration created successfully.")
    
    print("\n" + "="*50)
    print(f"[Begin] Starting scan and update process in: {base_dir}")
    print("="*50)

    # Walk through the directory tree to find experiment folders.
    for root, _, _ in os.walk(base_dir):
        process_directory(Path(root), golden_config)

    print("\n" + "="*50)
    print("[Success] Scan and update process completed!")
    print("="*50)

if __name__ == "__main__":
    main()
