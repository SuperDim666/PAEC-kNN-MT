#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/05_train_policy_network.py

[PAEC Framework - Phase 3: Policy Learning]

This script implements the "Teacher-Student" distillation process to train the lightweight
Student Policy Network (pi_phi).

The process consists of two main phases:
1. PHASE A (Teacher Generation):
   - Uses the heavy, pre-trained Dynamics Model (T_theta) as the "Teacher".
   - For each state S_t in the training data, it solves an online optimization problem
     to find the optimal action A_t* that minimizes the Lyapunov value V(S_t+1).
   - Generates a labelled dataset {(S_t, A_t*)}.

2. PHASE B (Student Training):
   - Trains the Student Policy Network (pi_phi) to mimic the Teacher's optimal actions.
   - The student learns to predict the optimal index type (classification) and continuous
     parameters k and lambda (regression) directly from the state S_t.
   - This distills the complex online optimization into a fast forward pass.
"""

import argparse, logging, traceback
import os, sys, json, joblib, warnings, random, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional, Callable, Any

# Add the project root directory to sys.path to enable absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as project_config

# Import core data structures
try: from src.core import Action
except ImportError: exit(1)

# Import dynamics model components and utilities
try:
    from t_train_Transformer import (
        _build_pairs_from_df, lyapunov_V, _compute_file_hash, get_train_path_base, data_cache_and_split,
        InvertableColumnTransformer, PAECDataset, PAECTransition
    )
    from t_train_Transformer import PAECTransition
    import t_train_Transformer
except ImportError as e:
    print(f"CRITICAL: Failed to import required components from t_train_Transformer.py: {e}")
    print("Please ensure `t_train_Transformer.py` is in the 'scripts' directory and accessible.")
    exit(1)

# Import the Student Policy Network model definition
try:
    from src.models.paec_policy_network import PAECPolicyNetwork
except ImportError as e:
    print(f"CRITICAL: Failed to import PAECPolicyNetwork from src.models.paec_policy_network: {e}")
    exit(1)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable TensorCore precision if available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

IS_DEBUG = False
ACTION_DIM = 6  # [one_hot(4), k_norm(1), lambda(1)]

def set_rand_seed(seed: int):
    """Sets random seeds for reproducibility."""
    if IS_DEBUG: print(f"[DEBUG] Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_teacher_components(teacher_model_dir: str, use_last_ckpt: bool, device: str) -> Dict:
    """
    Loads the pre-trained Dynamics Model (Teacher) and its associated components.

    This function performs a critical "context patch": it injects the configuration
    parameters from the saved model (e.g., S_DIM, E_INDEX) into the global scope of
    the `t_train_Transformer` module. This is necessary because the Dynamics Model class
    relies on these module-level globals.

    Args:
        teacher_model_dir: Path to the directory containing the trained dynamics model.
        use_last_ckpt: If True, load 'checkpoint_last.pt'; otherwise 'checkpoint_best.pt'.
        device: 'cuda' or 'cpu'.

    Returns:
        A dictionary containing the loaded model, scaler, config, P matrix, and V function.
    """
    logger.info(f"Loading Teacher components from: {teacher_model_dir}")
    if IS_DEBUG: print(f"[DEBUG][load_teacher_components] Starting load from {teacher_model_dir} on device {device}")
    
    model_dir = Path(teacher_model_dir)
    if not model_dir.is_dir(): raise FileNotFoundError(f"Teacher model directory not found: {model_dir}")
    
    # Load the configuration JSON used during dynamics model training
    config_path = model_dir / 'config.json'
    if not config_path.exists(): raise FileNotFoundError(f"Teacher config.json not found in {model_dir}")
    with open(config_path, 'r') as f:
        teacher_config = json.load(f)
    
    logger.info("Loaded Teacher config.json")
    if IS_DEBUG: print(f"[DEBUG][load_teacher_components] Loaded teacher_config keys: {list(teacher_config.keys())}")
    
    ckpt_name = "checkpoint_last.pt" if use_last_ckpt else "checkpoint_best.pt"
    checkpoint_path = model_dir / ckpt_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint {ckpt_name} not found in {model_dir}")
    
    logger.info(f"Loading Teacher checkpoint: {ckpt_name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if IS_DEBUG: print(f"[DEBUG][load_teacher_components] Loaded checkpoint keys: {list(checkpoint.keys())}")
    
    if IS_DEBUG: print("[DEBUG][load_teacher_components] Patching t_train_Transformer globals from teacher_config...")
    
    # Inject dimensions and indices into t_train_Transformer global scope
    try:
        S_DIM_cfg = teacher_config['S_DIM']
        E_DIM_cfg = teacher_config['E_DIM']
        PHI_DIM_cfg = teacher_config['PHI_DIM']
        H_DIM_cfg = teacher_config['H_DIM']
        E_INDEX_cfg = tuple(teacher_config['E_INDEX'])
        PHI_INDEX_cfg = tuple(teacher_config['PHI_INDEX'])
        H_INDEX_cfg = tuple(teacher_config['H_INDEX'])
        STATE_COLS_DEFAULT_cfg = teacher_config['STATE_COLS_DEFAULT']
        
        t_train_Transformer.S_DIM = S_DIM_cfg
        t_train_Transformer.E_DIM = E_DIM_cfg
        t_train_Transformer.PHI_DIM = PHI_DIM_cfg
        t_train_Transformer.H_DIM = H_DIM_cfg
        t_train_Transformer.E_INDEX = E_INDEX_cfg
        t_train_Transformer.PHI_INDEX = PHI_INDEX_cfg
        t_train_Transformer.H_INDEX = H_INDEX_cfg
        t_train_Transformer.STATE_COLS_DEFAULT = STATE_COLS_DEFAULT_cfg

        logger.info(f"Patched t_train_Transformer globals: S_DIM={t_train_Transformer.S_DIM}, E_DIM={t_train_Transformer.E_DIM}, H_DIM={t_train_Transformer.H_DIM}") # Log patched values
        logger.info(f"Patched STATE_COLS_DEFAULT: {t_train_Transformer.STATE_COLS_DEFAULT}")
        if IS_DEBUG: print("[DEBUG][load_teacher_components] Global patching successful.")

    except KeyError as e:
        if IS_DEBUG: print(f"[DEBUG][load_teacher_components] ERROR during global patching: {e}")
        raise KeyError(f"Teacher config.json is missing a required dimension key needed for patching: {e}")
    
    # Determine text embedding settings
    if IS_DEBUG: print("[DEBUG][load_teacher_components] Determining text embedding dimension...")
    use_text_embeddings_flag = teacher_config.get('use_text_embeddings', False)
    text_embedding_dim_from_config = teacher_config.get('text_embedding_dim', 0)
    final_text_embedding_dim = 0
    sbert_model = None
    sbert_model_name = None

    if use_text_embeddings_flag:
        if IS_DEBUG: print("[DEBUG][load_teacher_components] Teacher config indicates use_text_embeddings=True.")
        if text_embedding_dim_from_config > 0:
            final_text_embedding_dim = text_embedding_dim_from_config
            logger.info(f"[INFO] Using text embedding dimension from config: {final_text_embedding_dim}")
            if IS_DEBUG: print(f"[DEBUG][load_teacher_components] text_embedding_dim found in config: {final_text_embedding_dim}")
        else:
            # Fallback: Infer dimension by loading the model if config is missing it
            logger.warning(f"[WARNING] PAEC config indicates text embeddings were used ('use_text_embeddings': True), "
                          f"but 'text_embedding_dim' is missing or zero in {config_path}. "
                          f"Attempting to infer dimension by loading the SentenceTransformer.")
            if IS_DEBUG: print("[DEBUG][load_teacher_components] text_embedding_dim missing/zero in config, attempting inference.")
            try:
                sbert_model_name = project_config.MODEL_NAMES.get("sentence_encoder", 'sentence-transformers/LaBSE')
                logger.info(f"Loading SentenceTransformer '{sbert_model_name}' to infer dimension...")
                if IS_DEBUG: print(f"[DEBUG][load_teacher_components] Loading SBERT model: {sbert_model_name}")
                sbert_model = SentenceTransformer(sbert_model_name, device='cpu')
                actual_dim = sbert_model.get_sentence_embedding_dimension()
                final_text_embedding_dim = actual_dim
                logger.info(f"[INFO] Inferred text embedding dimension as {final_text_embedding_dim} from {sbert_model_name}.")
                if IS_DEBUG: print(f"[DEBUG][load_teacher_components] Inferred dimension: {final_text_embedding_dim}")
            except Exception as e:
                sbert_model = None
                sbert_model_name = None
                logger.error(f"[ERROR] Failed to dynamically get embedding dimension: {e}. Cannot proceed with text embeddings.")
                if IS_DEBUG: print(f"[DEBUG][load_teacher_components] ERROR inferring dimension: {e}")
                final_text_embedding_dim = 0
                use_text_embeddings_flag = False
                logger.warning("[WARNING] Disabling text embeddings due to failure in dimension retrieval.") 
                exit(1)
    else:
         final_text_embedding_dim = 0
         sbert_model = None
         logger.info("[INFO] Text embeddings are disabled according to config.")
         if IS_DEBUG: print("[DEBUG][load_teacher_components] Text embeddings explicitly disabled in config.")

    logger.info(f"[INFO] Final settings for model init: use_text_embeddings={use_text_embeddings_flag}, text_embedding_dim={final_text_embedding_dim}") 
    if IS_DEBUG: print(f"[DEBUG][load_teacher_components] Final embedding settings: use={use_text_embeddings_flag}, dim={final_text_embedding_dim}")
    if IS_DEBUG: print("[DEBUG][load_teacher_components] Instantiating PAECTransition model...")
    
    # Initialize the Dynamics Model architecture
    try:
        action_dim_config = teacher_config.get('action_dim', ACTION_DIM)
        if action_dim_config != ACTION_DIM:
            logger.warning(f"Teacher config action_dim ({action_dim_config}) differs from expected {ACTION_DIM}. Using {ACTION_DIM}.")
        model_args = {
            'action_dim': ACTION_DIM,
            'hid_dim': teacher_config['hid_dim'],
            'layers': teacher_config['layers'],
            'history_len': teacher_config['history_len'],
            'predict_delta': teacher_config['predict_delta'],
            'use_text_embeddings': use_text_embeddings_flag,
            'text_embedding_dim': final_text_embedding_dim,
            'use_decoder_hidden_state': teacher_config['use_decoder_hidden_state'],
            'decoder_hidden_state_dim': teacher_config['decoder_hidden_state_dim'],
            'use_separate_heads_eh': teacher_config['use_separate_heads_eh'],
            'use_multi_heads': teacher_config['use_multi_heads'],
            'use_spectral_norm': teacher_config['use_spectral_norm'],
            'nhead': teacher_config['nhead']
        }
        if IS_DEBUG: print(f"[DEBUG][load_teacher_components] PAECTransition args: {model_args}")
        teacher_model = PAECTransition(**model_args).to(device)
    except KeyError as e:
        if IS_DEBUG: print(f"[DEBUG][load_teacher_components] KeyError during model instantiation: {e}")
        raise KeyError(f"Teacher config.json is missing a required key for PAECTransition: {e}")
    
    if IS_DEBUG: print("[DEBUG][load_teacher_components] Loading model state_dict...")
    state_dict = checkpoint['model']
    
    # Unwrap state_dict if it was compiled
    if list(state_dict.keys())[0].startswith('_orig_mod.'):
        logger.info("Unwrapping compiled state_dict for Teacher model...")
        if IS_DEBUG: print("[DEBUG][load_teacher_components] State dict appears compiled, unwrapping...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Load weights
    try:
        if IS_DEBUG: print("[DEBUG][load_teacher_components] Attempting strict state_dict loading...")
        teacher_model.load_state_dict(state_dict, strict=True)
        if IS_DEBUG: print("[DEBUG][load_teacher_components] Strict loading successful.")
    except RuntimeError as e:
        logger.error(f"Failed to load Teacher state_dict strictly: {e}. Trying non-strict loading...")
        if IS_DEBUG: print(f"[DEBUG][load_teacher_components] Strict loading failed: {e}. Trying non-strict...")
        try:
             teacher_model.load_state_dict(state_dict, strict=False)
             if IS_DEBUG: print("[DEBUG][load_teacher_components] Non-strict loading successful.")
        except RuntimeError as e2:
            if IS_DEBUG: print(f"[DEBUG][load_teacher_components] Non-strict loading also failed: {e2}")
            raise RuntimeError(f"Strict and non-strict loading failed for Teacher model: {e2}") 
            exit(1)
    
    teacher_model.eval()
    logger.info("Teacher model (T_theta) instantiated and weights loaded.")
    
    # Load the data scaler
    if IS_DEBUG: print("[DEBUG][load_teacher_components] Loading scaler...")
    scaler_path = model_dir / "scaler.joblib" 
    if not scaler_path.exists(): 
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}") 
    scaler = joblib.load(scaler_path) 
    logger.info(f"Scaler loaded. State Dim (S_DIM) used for model was: {t_train_Transformer.S_DIM}")
    
    # Determine the P matrix for the Lyapunov function V(S) = E^T P E
    if IS_DEBUG: print("[DEBUG][load_teacher_components] Loading P matrix...")
    if 'P' in checkpoint and checkpoint['P'] is not None:
        # Load learned P from checkpoint
        P_values = torch.tensor(checkpoint['P'], dtype=torch.float32, device=device) 
        if P_values.shape[0] != t_train_Transformer.E_DIM: 
             raise ValueError(f"Loaded P matrix from checkpoint dim ({P_values.shape[0]}) != E_DIM ({t_train_Transformer.E_DIM})") 
        logger.info(f"Loaded P matrix directly from checkpoint (Epoch {checkpoint.get('epoch', 'N/A')})") 
        if IS_DEBUG: print(f"[DEBUG][load_teacher_components] P matrix loaded from checkpoint. Shape: {P_values.shape}")
    elif 'P_init' in teacher_config:
        # Fallback to initial config P if not found in checkpoint
        p_init_values = teacher_config['P_init']
        # Check compatibility with dimensions
        if not isinstance(p_init_values, list) or len(p_init_values) != t_train_Transformer.E_DIM + t_train_Transformer.H_DIM: 
             raise ValueError(f"Invalid 'P_init' in config.json. Expected a list of length {t_train_Transformer.E_DIM + t_train_Transformer.H_DIM}, got {p_init_values}")
        P_values = torch.tensor(p_init_values, dtype=torch.float32, device=device)
        logger.info("Loaded P matrix using 'P_init' from config.json (P not found in checkpoint).")
        if IS_DEBUG: print(f"[DEBUG][load_teacher_components] P matrix created from config P_init. Shape: {P_values.shape}")
    else:
        if IS_DEBUG: print("[DEBUG][load_teacher_components] ERROR: P matrix not found in checkpoint or config.")
        raise ValueError("Could not determine P matrix: 'P' key not found or is None in checkpoint, AND 'P_init' key not found in config.json.")
    
    # Define the Lyapunov function closure
    P_fixed = P_values.detach()
    if IS_DEBUG: print(f"[DEBUG][load_teacher_components] P_fixed for V_fn: {P_fixed.tolist()}")
    def V_fn(S: torch.Tensor) -> torch.Tensor:
        if IS_DEBUG: print(f"[DEBUG][V_fn] Input S shape: {S.shape}")
        # V(S) operates on the first S_DIM elements (state vector)
        S_base = S[:, :t_train_Transformer.S_DIM]
        v_result = lyapunov_V(S_base, P_fixed)
        if IS_DEBUG: print(f"[DEBUG][V_fn] Output V shape: {v_result.shape}, first 5 values: {v_result[:5].tolist()}")
        return v_result
    
    if IS_DEBUG: print("[DEBUG][load_teacher_components] Load complete. Returning components.")
    
    # Retry loading SBERT if initial attempt failed but flag is set
    if use_text_embeddings_flag and sbert_model is None:
        logger.warning("Text embeddings enabled but SBERT model failed to load. Retrying load...")
        try:
             sbert_model_name = project_config.MODEL_NAMES.get("sentence_encoder", 'sentence-transformers/LaBSE')
             sbert_model = SentenceTransformer(sbert_model_name, device='cpu')
        except Exception as e_retry:
            logger.error(f"Failed to retry loading SBERT: {e_retry}. Phase A will likely fail.")
            
    return {
        "teacher_model": teacher_model,       # The loaded and initialized PAECTransition model
        "teacher_config": teacher_config,     # The configuration dictionary from config.json
        "scaler": scaler,                     # The loaded scaler object
        "P_matrix": P_fixed,                  # The fixed P matrix diagonal tensor
        "V_fn": V_fn,                         # The Lyapunov function V(S)
        "final_text_embedding_dim": final_text_embedding_dim,
        "use_text_embeddings_flag": use_text_embeddings_flag,
        "text_embedder": sbert_model,
        "sbert_model_name": sbert_model_name
    }

def compute_optimal_action(
    teacher_model: PAECTransition, # T_theta
    V_fn: Callable,                # V(S) function
    # Inputs (already scaled batch data)
    S_t_batch: torch.Tensor, S_hist_batch: torch.Tensor, A_hist_batch: torch.Tensor,
    # Optional embeddings (already batched)
    decoder_hidden_state_batch: Optional[torch.Tensor],
    source_embedding_batch: Optional[torch.Tensor],
    prefix_embedding_batch: Optional[torch.Tensor],
    # Config parameters needed for optimization
    teacher_config: Dict,
    device: str
) -> List[Dict[str, Any]]:
    """
    A standalone teacher policy solver (Online Planner).
    
    For a given batch of states, it performs online optimization to find the
    optimal continuous parameters (k, lambda) for each index type, and then
    selects the global best action A* that minimizes the predicted Lyapunov
    value V(S_t+1) according to the Dynamics Model.

    Args:
        teacher_model: Loaded T_theta (PAECTransition) model.
        V_fn: Lyapunov function V(S).
        S_t_batch: Batch of the current state (scaled).
        S_hist_batch: Batch of the state history (scaled).
        A_hist_batch: Batch of the action history (unscaled).
        decoder_hidden_state_batch: Optional decoder hidden state batch.
        source_embedding_batch: Optional source text embedding batch.
        prefix_embedding_batch: Optional prefix text embedding batch.
        teacher_config: Configuration dictionary for the Teacher model.
        device: Computational device ('cuda' or 'cpu').
    
    Returns:
        A list of result dictionaries for each sample in the batch. Each contains:
        - `rank`: Sorted list of index types by cost.
        - `best_knn_type`: The optimal index type.
        - `details`: Optimal k, lambda, and cost for each index type.
    """
    if IS_DEBUG: print(f"[DEBUG][compute_optimal_action_v2] Start. Batch size: {S_t_batch.shape[0]}")
    
    # 1. Obtain optimization parameters from config
    optim_config = teacher_config.get("OPTIM_STEPS_CONFIG", {})
    TOLERANCE = optim_config.get("tolerance", 1e-4)
    PATIENCE = optim_config.get("patience", 1)
    MAX_STEPS = optim_config.get("max_steps", 10)       # Gradient descent steps
    RHO = teacher_config.get("rho", 0.2)                # Lyapunov convergence parameter
    LEARNING_RATE = teacher_config.get("lr", 1e-2)      # Learning rate for online optimizer

    B = S_t_batch.shape[0] # Batch size

    # 2. Pre-calculate Heavy Cache (Encoder Output)
    # This optimization avoids re-running the heavy Transformer Encoder for every optimization step
    model_kwargs = {}
    if teacher_config.get('use_decoder_hidden_state', False):
        model_kwargs['decoder_hidden_state'] = decoder_hidden_state_batch
    if teacher_config.get('use_text_embeddings', False):
         model_kwargs['source_embeddings'] = source_embedding_batch
         model_kwargs['prefix_embeddings'] = prefix_embedding_batch
         if model_kwargs.get('source_embeddings') is None or model_kwargs.get('prefix_embeddings') is None:
              raise ValueError("compute_optimal_action requires text embeddings but they were not provided.")

    with torch.no_grad():
        cache_batch_raw = teacher_model.forward_heavy_cache(
            S_t_batch, S_hist_batch, A_hist_batch, **model_kwargs
        )
        # Detach cache to stop gradients flowing back into the encoder (we only optimize action)
        cache_batch = {k: v.detach() for k, v in cache_batch_raw.items() if v is not None}
        if "Phi_t" not in cache_batch:  # Ensure Phi_t is present for state reconstruction
            raise KeyError("Critical error: Phi_t not found in heavy cache results.")
        if IS_DEBUG: print(f"[DEBUG][compute_optimal_action_v2] Heavy cache computed.")

    # 3. Calculate baseline cost for 'none' action (k=0)
    with torch.no_grad():
        none_action_obj = Action(k=0, index_type='none', lambda_weight=0.0)
        none_action_tensor = _action_to_tensor(none_action_obj, device).repeat(B, 1) # Shape: [B, 6]
        # Predict next state components using the light decoder head
        E_pred_none, H_pred_none, _ = teacher_model.forward_light_head(none_action_tensor, cache_batch)
        # Reconstruct full S_next using Inertia assumption for Phi (Phi_t+1 approx Phi_t)
        S_next_none = torch.cat([E_pred_none, cache_batch["Phi_t"], H_pred_none], dim=1) # Shape: [B, S_DIM]
        v_none_batch = V_fn(S_next_none) # Calculate Lyapunov Cost V(S_next)
        if IS_DEBUG: print(f"[DEBUG][compute_optimal_action_v2] Cost for 'none' action calculated.")

    # 4. Initialize results storage
    results_per_index = {
        'none': {'cost': v_none_batch, 'action_tensor': none_action_tensor}
    }

    # 5. Optimize for each active index type independently
    index_types_to_optimize = ['exact', 'hnsw', 'ivf_pq']

    for index_type in index_types_to_optimize:
        if IS_DEBUG: print(f"[DEBUG][compute_optimal_action_v2] Optimizing for index_type = {index_type}")
        
        # Enable gradient calculation for action parameters
        with torch.enable_grad():
            try:
                # Initialize learnable parameters for k and lambda (start from center 0.5)
                initial_k_norm = torch.full((B, 1), 0.5, device=device)
                initial_lambda = torch.full((B, 1), 0.5, device=device)
                
                # Create leaf tensors requiring gradient
                k_norm_param = initial_k_norm.clone().requires_grad_(True)
                lambda_param = initial_lambda.clone().requires_grad_(True)
                
                # Optimizer for this specific index type's parameters
                optimizer = torch.optim.Adam([k_norm_param, lambda_param], lr=LEARNING_RATE)

                # Tracking best found parameters
                best_losses_this_index = torch.full((B,), float('inf'), device=device)
                
                # Initial best action tensor (placeholder)
                index_one_hot = _action_to_tensor(Action(index_type=index_type, k=8, lambda_weight=0.5), device)[:, :4].repeat(B, 1)
                best_action_tensor_this_index = torch.cat([index_one_hot, initial_k_norm, initial_lambda], dim=1).detach()
                
                patience_counter = 0
                index_type_tensor = _action_to_tensor(Action(index_type=index_type, k=0, lambda_weight=0.0), device)[:, :4].repeat(B, 1) # Fixed discrete part

                optim_steps_ran_this_index = 0
                
                # Optimization Loop
                for optim_step in range(MAX_STEPS):
                    optim_steps_ran_this_index = optim_step + 1
                    optimizer.zero_grad()

                    # Enforce constraints [0, 1] on parameters
                    with torch.no_grad():
                        k_norm_param.data.clamp_(0.0, 1.0)
                        lambda_param.data.clamp_(0.0, 1.0)

                    # Construct candidate action batch
                    action_tensor_batch = torch.cat([index_type_tensor, k_norm_param, lambda_param], dim=1) # [B, 6]

                    # Forward pass through light head (using cached encoder outputs)
                    E_pred_batch, H_pred_batch, _ = teacher_model.forward_light_head(action_tensor_batch, cache_batch)
                    
                    # Reconstruct S_next
                    S_next_pred_batch = torch.cat([E_pred_batch, cache_batch["Phi_t"], H_pred_batch], dim=1)
            
                    # Compute Loss: V(S_next)
                    losses_batch = V_fn(S_next_pred_batch)

                    # Stability check
                    if torch.isnan(losses_batch).any():
                        logger.warning(f"NaN detected in loss for {index_type} at step {optim_step}. Stopping opt.")
                        break
                    
                    # Backward pass (sum of losses allows individual gradients)
                    losses_batch.sum().backward()
                    optimizer.step()

                    # Update best results per sample
                    current_losses = losses_batch.detach()
                    improvement = best_losses_this_index - current_losses
                    improved_mask = improvement > TOLERANCE

                    # Store best parameters found so far
                    best_action_tensor_this_index[improved_mask] = action_tensor_batch[improved_mask].detach()
                    best_losses_this_index = torch.min(best_losses_this_index, current_losses)

                    # Early stopping logic
                    if torch.any(improved_mask):
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= PATIENCE:
                        if IS_DEBUG: print(f"[DEBUG][compute_optimal_action_v2] Patience exceeded for {index_type} at step {optim_step}.")
                        break

                # Handle failures
                best_losses_this_index[torch.isnan(best_losses_this_index)] = float('inf')
                if IS_DEBUG: print(f"[DEBUG][compute_optimal_action_v2] Optimization for {index_type} finished after {optim_steps_ran_this_index} steps.")

            except Exception as e:
                logger.error(f"Optimization loop failed for {index_type}: {e}")
                traceback.print_exc()
                # Fallback to infinity cost on error
                best_losses_this_index = torch.full((B,), float('inf'), device=device)
                index_one_hot = _action_to_tensor(Action(index_type=index_type, k=8, lambda_weight=0.5), device)[:, :4].repeat(B, 1)
                best_action_tensor_this_index = torch.cat(
                    [index_one_hot, torch.full((B, 1), 0.5, device=device), torch.full((B, 1), 0.5, device=device)], dim=1
                ).detach()

        # 6. Store optimized result for this index type
        results_per_index[index_type] = {
            'cost': best_losses_this_index,           
            'action_tensor': best_action_tensor_this_index
        }

    # 7. Aggregate and Rank Results
    final_batch_results = []
    for i in range(B):
        sample_results_details = {}
        
        # Only rank k>0 indices to find the "Active A*" (Teacher's best active retrieval)
        k_gt_zero_costs = {} 

        for index_type in ['none', 'exact', 'hnsw', 'ivf_pq']:
            cost = results_per_index[index_type]['cost'][i].item()
            action_tensor = results_per_index[index_type]['action_tensor'][i]
            
            # Decode tensor to object
            action_obj = _action_tensor_to_object(action_tensor)

            # Collect active costs for sorting
            if index_type != 'none':
                k_gt_zero_costs[index_type] = cost
                
            sample_results_details[index_type] = {
                'k': action_obj.k,
                'lambda': action_obj.lambda_weight,
                'cost': cost
            }

        # Identify the best active retrieval strategy
        ranked_kNN_indices = sorted(k_gt_zero_costs, key=lambda k: k_gt_zero_costs[k])
        
        # Rank everything (including none) for logging purposes
        all_costs = {**k_gt_zero_costs, 'none': sample_results_details['none']['cost']}
        ranked_indices_all = sorted(all_costs, key=lambda k: all_costs[k])

        final_batch_results.append({
            'rank': ranked_indices_all,
            'best_knn_type': ranked_kNN_indices[0],     # The target for the student policy (active retrieval)
            'details': sample_results_details
        })

    if IS_DEBUG: print(f"[DEBUG][compute_optimal_action_v2] Finished. Returning results for {len(final_batch_results)} samples.")
    return final_batch_results

def _action_to_tensor(action: Action, device: str) -> torch.Tensor:
    """Converts an Action object to a normalized tensor [1, 6]."""
    action_tensor = torch.zeros(1, ACTION_DIM, device=device)
    index_map = {'none': 0, 'exact': 1, 'hnsw': 2, 'ivf_pq': 3}
    if action.index_type in index_map:
        action_tensor[0, index_map[action.index_type]] = 1.0
    action_tensor[0, 4] = action.k / float(project_config.KNN_MAX_K)
    action_tensor[0, 5] = action.lambda_weight
    if IS_DEBUG and random.random() < 0.01: print(f"[DEBUG][_action_to_tensor] Input: {action}, Output tensor: {action_tensor.tolist()}")
    return action_tensor

def _action_tensor_to_object(action_tensor: torch.Tensor, max_k: float=float(project_config.KNN_MAX_K)) -> Action:
    """Decodes a tensor [1, 6] back to an Action object."""
    if IS_DEBUG and random.random() < 0.01: print(f"[DEBUG][_action_tensor_to_object] Input tensor: {action_tensor.tolist()}")
    index_map_inv = {0: 'none', 1: 'exact', 2: 'hnsw', 3: 'ivf_pq'}
    k_norm = torch.clamp(action_tensor[4], 0.0, 1.0).item()
    lambda_w = torch.clamp(action_tensor[5], 0.0, 1.0).item()
    final_k = round(k_norm * max_k)
    
    # Handle logic for 'none' or k=0
    if final_k == 0:
        final_index_type = 'none'
        final_lambda_w = 0.0
        final_k = 0
    else:
        best_idx_val = torch.argmax(action_tensor[:4]).item()
        intended_index_type = index_map_inv.get(int(best_idx_val), 'none')
        if intended_index_type == 'none':
            final_index_type = 'exact' # Fallback for k>0 but index=0
            final_lambda_w = lambda_w
        else:
            final_index_type = intended_index_type
            final_lambda_w = lambda_w
        if final_k == 0: final_k = 1
        
    final_action = Action(index_type=final_index_type, k=final_k, lambda_weight=final_lambda_w)
    if IS_DEBUG and random.random() < 0.01: print(f"[DEBUG][_action_tensor_to_object] Output action: {final_action}")
    return final_action

class PolicyDataset(Dataset):
    def __init__(self, npz_path: str, teacher_config: Dict):
        """
        Dataset class for training the Student Policy Network.
        Loads the dataset generated by `generate_teacher_dataset` (Phase A).
        """
        logger.info(f"Loading PolicyDataset from {npz_path}...")
        if IS_DEBUG: print(f"[DEBUG][PolicyDataset.__init__] Loading from {npz_path}")
        try:
            data = np.load(npz_path)
            if IS_DEBUG: print(f"[DEBUG][PolicyDataset.__init__] NPZ loaded. Keys: {list(data.keys())}")
        except Exception as e:
            logger.error(f"Failed to load npz file {npz_path}: {e}")
            raise

        self.teacher_config = teacher_config
        self.action_dim = ACTION_DIM
        if IS_DEBUG: print("[DEBUG][PolicyDataset.__init__] Converting loaded numpy arrays to tensors...")

        # Validation of required keys
        required_keys = ['S_t', 'S_hist', 'A_hist', 'mask', 'A_star']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required keys in npz file {npz_path}: {missing_keys}")

        # Input Features (X)
        self.S_t = torch.from_numpy(data['S_t'].astype(np.float32))
        self.S_hist = torch.from_numpy(data['S_hist'].astype(np.float32))
        self.A_hist = torch.from_numpy(data['A_hist'].astype(np.float32))
        self.mask = torch.from_numpy(data['mask'].astype(np.bool_))
        if IS_DEBUG: print(f"[DEBUG][PolicyDataset.__init__] Processed S_t ({self.S_t.shape}), S_hist ({self.S_hist.shape}), A_hist ({self.A_hist.shape}), mask ({self.mask.shape})")

        # Optional Inputs (decoder hidden state, embeddings)
        if IS_DEBUG: print("[DEBUG][PolicyDataset.__init__] Processing optional embeddings...")
        self.H_dec_t = None
        if teacher_config.get('use_decoder_hidden_state', False):
            if 'H_dec_t' in data:
                self.H_dec_t = torch.from_numpy(data['H_dec_t'].astype(np.float32))
                if IS_DEBUG: print(f"[DEBUG][PolicyDataset.__init__] Processed H_dec_t ({self.H_dec_t.shape})")
            else:
                logger.warning("Teacher config expected 'H_dec_t' but it was not found in the npz file.")
                if IS_DEBUG: print("[DEBUG][PolicyDataset.__init__] WARNING: H_dec_t missing in npz.")

        self.Src_emb_t, self.Pref_emb_t = None, None
        if teacher_config.get('use_text_embeddings', False):
            if 'Src_emb_t' in data and 'Pref_emb_t' in data:
                self.Src_emb_t = torch.from_numpy(data['Src_emb_t'].astype(np.float32))
                self.Pref_emb_t = torch.from_numpy(data['Pref_emb_t'].astype(np.float32))
                if IS_DEBUG: print(f"[DEBUG][PolicyDataset.__init__] Processed Src_emb_t ({self.Src_emb_t.shape}), Pref_emb_t ({self.Pref_emb_t.shape})")
            else:
                logger.warning("Teacher config expected text embeddings ('Src_emb_t', 'Pref_emb_t') but they were not found in the npz file.")
                if IS_DEBUG: print("[DEBUG][PolicyDataset.__init__] WARNING: Text embeddings missing in npz.")

        # Target Labels (Y)
        # The student learns A_star (the optimal action computed by the teacher)
        if IS_DEBUG: print("[DEBUG][PolicyDataset.__init__] Processing labels...")
        self.A_star = torch.from_numpy(data['A_star'].astype(np.float32))
        if IS_DEBUG: print(f"[DEBUG][PolicyDataset.__init__] Processed A_star ({self.A_star.shape})")

        # Consistency Check
        expected_len = len(self.S_t)
        all_lengths_match = all(
            (tensor is None or len(tensor) == expected_len)
            for tensor_name, tensor in [
                ('S_hist', self.S_hist), ('A_hist', self.A_hist), ('mask', self.mask), ('A_star', self.A_star),
                ('H_dec_t', self.H_dec_t if teacher_config.get('use_decoder_hidden_state', False) else None),
                ('Src_emb_t', self.Src_emb_t if teacher_config.get('use_text_embeddings', False) else None),
                ('Pref_emb_t', self.Pref_emb_t if teacher_config.get('use_text_embeddings', False) else None)
            ] if tensor_name in required_keys or (tensor is not None)
        )
        if not all_lengths_match:
            lengths = {
                'S_t': len(self.S_t), 'S_hist': len(self.S_hist), 'A_hist': len(self.A_hist),
                'mask': len(self.mask), 'A_star': len(self.A_star),
                'H_dec_t': len(self.H_dec_t) if self.H_dec_t is not None else 'N/A',
                'Src_emb_t': len(self.Src_emb_t) if self.Src_emb_t is not None else 'N/A',
                'Pref_emb_t': len(self.Pref_emb_t) if self.Pref_emb_t is not None else 'N/A'
            }
            mismatched = {k: v for k, v in lengths.items() if isinstance(v, int) and v != expected_len}
            raise ValueError(f"Data length mismatch in {npz_path} after loading! Expected {expected_len}, but found mismatches: {mismatched}")

        logger.info(f"PolicyDataset loaded with {expected_len} samples.")
        
        if IS_DEBUG:
            print("[DEBUG][PolicyDataset.__init__] Verifying if loaded S_t tensor appears scaled...")
            s_t_mean = torch.mean(self.S_t, dim=0)
            s_t_std = torch.std(self.S_t, dim=0)
            print(f"\tLoaded S_t Mean: {s_t_mean.tolist()}")
            print(f"\tLoaded S_t Std: {s_t_std.tolist()}")
            print("[DEBUG][PolicyDataset.__init__] Loaded data statistics logged. Proceeding without strict mean check.")
            print("[DEBUG][PolicyDataset.__init__] Initialization complete.")

    def __len__(self):
        return len(self.S_t)

    def __getitem__(self, idx: int):
        H_dec_t = self.H_dec_t[idx] if self.H_dec_t is not None else torch.empty(0)
        Src_emb_t = self.Src_emb_t[idx] if self.Src_emb_t is not None else torch.empty(0)
        Pref_emb_t = self.Pref_emb_t[idx] if self.Pref_emb_t is not None else torch.empty(0)
        if IS_DEBUG and idx == 0:
            print(
                f"[DEBUG][PolicyDataset.__getitem__] shapes for idx 0: S_t={self.S_t[idx].shape}, "
                f"S_hist={self.S_hist[idx].shape}, A_hist={self.A_hist[idx].shape}, H_dec_t={H_dec_t.shape}, "
                f"Src_emb_t={Src_emb_t.shape}, Pref_emb_t={Pref_emb_t.shape}, mask={self.mask[idx].shape}, A_star={self.A_star[idx].shape}"
            )
        return (
            self.S_t[idx], self.S_hist[idx], self.A_hist[idx],
            H_dec_t, Src_emb_t, Pref_emb_t,
            self.mask[idx],
            self.A_star[idx]
        )

class CompositePolicyLoss(nn.Module):
    def __init__(self, index_weight: float=1.0, k_lambda_weight: float=1.0):
        super().__init__()
        self.index_weight = index_weight
        self.k_lambda_weight = k_lambda_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        if IS_DEBUG: print(f"[DEBUG][CompositePolicyLoss] Initialized with index_weight={index_weight}, k_lambda_weight={k_lambda_weight}")

    def forward(
        self,
        pred_index_logits: torch.Tensor,
        pred_k_lambda_values: torch.Tensor,
        target_A_star: torch.Tensor
    ):
        if IS_DEBUG and random.random() < 0.05:
            print(f"[DEBUG][CompositePolicyLoss.forward] Input shapes: logits={pred_index_logits.shape}, k_lambda={pred_k_lambda_values.shape}, target={target_A_star.shape}")

        # Note: The Student is trained to discriminate only among ACTIVE index types (k>0).
        # The 0-th logit corresponds to 'none' and is ignored/sliced out because the decision
        # to use 'none' is handled by the upstream hard safety valve, not this network.
        
        # 1. Slice logits to exclude index 0 ('none')
        if pred_index_logits.shape[1] != 4:
             logger.warning(f"Expected 4 logits from policy network, but got {pred_index_logits.shape[1]}. Loss calculation may be incorrect.")
             pred_logits_k_gt_0 = pred_index_logits
             target_labels_k_gt_0 = torch.argmax(target_A_star[:, :4], dim=1)
        else:
            # Select logits for indices 1, 2, 3 (Exact, HNSW, IVF_PQ)
            pred_logits_k_gt_0 = pred_index_logits[:, 1:] # Shape: [B, 3]

            # 2. Slice target one-hot vector similarly
            target_one_hot_k_gt_0 = target_A_star[:, 1:4] # Shape: [B, 3]
            
            # 3. Create target labels [0, 1, 2] corresponding to [Exact, HNSW, IVF_PQ]
            target_labels_k_gt_0 = torch.argmax(target_one_hot_k_gt_0, dim=1) # Shape: [B]

        # 4. Extract continuous targets (k and lambda)
        target_k_lambda = target_A_star[:, 4:] # Shape: [B, 2]
        
        # 5. Compute Component Losses
        loss_index = self.ce_loss(pred_logits_k_gt_0, target_labels_k_gt_0)
        loss_k_lambda = self.mse_loss(pred_k_lambda_values, target_k_lambda)
        
        total_loss = (self.index_weight * loss_index) + \
                     (self.k_lambda_weight * loss_k_lambda)
        
        if IS_DEBUG and random.random() < 0.05:
            print(
                f"[DEBUG][CompositePolicyLoss.forward] Losses: total={total_loss.item():.4f}, "
                f"index (k>0)={loss_index.item():.4f}, k_lambda={loss_k_lambda.item():.4f}"
            )
        return total_loss, loss_index, loss_k_lambda

def train_policy_network(args: argparse.Namespace, teacher_config: Dict):
    """
    PHASE B: Trains the Student Policy Network using the dataset generated in Phase A.
    """
    logger.info("=" * 80)
    logger.info("PHASE B: Training 'Student' Policy Network (pi_phi)")
    if IS_DEBUG: print("[DEBUG][train_policy_network] Starting Phase B.")
    logger.info("Epoch-level average train/validation losses will be logged regardless of debug mode.")
    logger.info("=" * 80)
    
    device = torch.device(args.device)
    if IS_DEBUG: print(f"[DEBUG][train_policy_network] Using device: {device}")
    
    # Load dataset
    policy_data_path = Path(args.policy_data_path)
    if IS_DEBUG: print(f"[DEBUG][train_policy_network] Attempting to load dataset from: {policy_data_path}")
    if not policy_data_path.exists():
        logger.error(f"Policy training data not found at {policy_data_path}")
        logger.error("Please run PHASE A (generate_teacher_dataset) first.")
        exit(1)
    
    full_dataset = PolicyDataset(str(policy_data_path), teacher_config)
    
    # Split into Train/Val
    if IS_DEBUG: print("[DEBUG][train_policy_network] Splitting dataset...")
    val_size = int(len(full_dataset) * args.val_ratio_student)
    train_size = len(full_dataset) - val_size
    if val_size == 0 and len(full_dataset) > 0:
        val_size = 1
        train_size = len(full_dataset) - 1
    elif len(full_dataset) == 0:
         logger.error("Teacher dataset is empty. Cannot train.")
         exit(1)
         
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info(f"Split data: {train_size} train, {val_size} validation samples.")
    if IS_DEBUG: print(f"[DEBUG][train_policy_network] Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}")
    
    # Initialize Student Model
    if IS_DEBUG: print("[DEBUG][train_policy_network] Instantiating student model (PAECPolicyNetwork)...")
    try:
        s_dim_teacher = teacher_config['S_DIM']
        use_text_embeddings_flag = teacher_config.get('use_text_embeddings', False)
        text_emb_dim_teacher = 0 

        if use_text_embeddings_flag:
            text_emb_dim_from_config = teacher_config.get('text_embedding_dim', 0)
            
            if text_emb_dim_from_config > 0:
                if IS_DEBUG: print(f"[DEBUG][train_policy_network] Using 'text_embedding_dim' from teacher_config: {text_emb_dim_from_config}")
                text_emb_dim_teacher = text_emb_dim_from_config
            else:
                logger.warning(f"[WARNING][train_policy_network] 'text_embedding_dim' is missing or zero in teacher_config. Attempting to infer.")
                if IS_DEBUG: print(f"[DEBUG][train_policy_network] Inferring text_embedding_dim by loading SBERT model...")
                
                sbert_model_name = project_config.MODEL_NAMES.get("sentence_encoder", 'sentence-transformers/LaBSE')
                try:
                    sbert_model_name = teacher_config.get('sentence_encoder_model_name', sbert_model_name)
                    if IS_DEBUG: print(f"[DEBUG][train_policy_network] Loading SBERT model: {sbert_model_name}")
                    # Fast check on CPU to avoid GPU allocation for this simple check
                    sbert_model_dim_checker = SentenceTransformer(sbert_model_name, device='cpu') 
                    inferred_dim = sbert_model_dim_checker.get_sentence_embedding_dimension()
                    text_emb_dim_teacher = inferred_dim
                    if IS_DEBUG: print(f"[DEBUG][train_policy_network] Inferred dimension: {inferred_dim}")
                    del sbert_model_dim_checker
                except Exception as e:
                    logger.error(f"Failed to load SentenceTransformer ('{sbert_model_name}') to infer text_embedding_dim: {e}")
                    logger.error("Cannot initialize PAECPolicyNetwork without 'text_embedding_dim'.")
                    exit(1)
        dec_hs_dim_teacher = teacher_config.get('decoder_hidden_state_dim', 0) if teacher_config.get('use_decoder_hidden_state', False) else 0
        if IS_DEBUG: print(f"[DEBUG][train_policy_network] Student model dims: S_DIM={s_dim_teacher}, TextEmbDim={text_emb_dim_teacher}, DecHsDim={dec_hs_dim_teacher}")
        
        student_model_args = {
            'S_DIM':s_dim_teacher,
            'ACTION_DIM':ACTION_DIM,
            'hid_dim':teacher_config['hid_dim'],
            'nhead':teacher_config['nhead'],
            'layers':teacher_config['layers'],
            'history_len':teacher_config['history_len'],
            'use_text_embeddings':teacher_config.get('use_text_embeddings', False),
            'text_embedding_dim':text_emb_dim_teacher,
            'use_decoder_hidden_state':teacher_config.get('use_decoder_hidden_state', False),
            'decoder_hidden_state_dim':dec_hs_dim_teacher
        }
        if IS_DEBUG: print(f"[DEBUG][train_policy_network] PAECPolicyNetwork args: {student_model_args}")
        model = PAECPolicyNetwork(**student_model_args).to(device)
    except KeyError as e:
        logger.error(f"Teacher config.json is missing a required key: {e}")
        exit(1)
    except ValueError as e:
         logger.error(f"Error initializing PAECPolicyNetwork: {e}")
         exit(1)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = CompositePolicyLoss()
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Initialized PAECPolicyNetwork (pi_phi) with {num_params} parameters.")
    if IS_DEBUG: print(f"[DEBUG][train_policy_network] Student model initialized with {num_params} params.")
    
    best_val_loss = float('inf')
    output_dir = Path(args.output_model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs_student = args.epochs_student
    
    if IS_DEBUG: print(f"[DEBUG][train_policy_network] Starting training loop for {epochs_student} epochs.")
    
    for epoch in range(epochs_student):
        if IS_DEBUG: print(f"[DEBUG][train_policy_network] Starting Epoch {epoch+1}/{epochs_student}")
        
        # --- TRAINING LOOP ---
        model.train()
        train_loss, train_loss_idx, train_loss_kl = 0.0, 0.0, 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_student} [TRAIN]")
        
        for batch_idx, batch in enumerate(pbar_train):
            if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Train Batch {batch_idx+1}/{len(train_loader)} - Unpacking data...")
            S_t, S_hist, A_hist, H_dec_t, Src_emb_t, Pref_emb_t, mask, A_star = \
                [b.to(device) if torch.is_tensor(b) else None for b in batch]
            
            H_dec_t = H_dec_t if H_dec_t is not None else None
            Src_emb_t = Src_emb_t if Src_emb_t is not None else None
            Pref_emb_t = Pref_emb_t if Pref_emb_t is not None  else None
            
            if IS_DEBUG and batch_idx==0:
                print(
                    f"[DEBUG][train_policy_network] Train Batch 0 data shapes on device: "
                    f"S_t={S_t.shape if S_t is not None else 'None'}, S_hist={S_hist.shape if S_hist is not None else 'None'}, A_hist={A_hist.shape if A_hist is not None else 'None'}, H_dec_t={H_dec_t.shape if H_dec_t is not None else 'None'}, Src_emb_t={Src_emb_t.shape if Src_emb_t is not None else 'None'}, Pref_emb_t={Pref_emb_t.shape if Pref_emb_t is not None else 'None'}, mask={mask.shape if mask is not None else 'None'}, A_star={A_star.shape if A_star is not None else 'None'}")
            
            # Reshape inputs if necessary (batch dimension handling)
            if S_t is not None and S_t.dim() == 2:
                S_t = S_t.unsqueeze(1)
                if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Reshaped S_t to: {S_t.shape}")
            if Src_emb_t is not None and Src_emb_t.dim() == 2:
                Src_emb_t = Src_emb_t.unsqueeze(1)
                if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Reshaped Src_emb_t to: {Src_emb_t.shape}")
            if Pref_emb_t is not None and Pref_emb_t.dim() == 2:
                Pref_emb_t = Pref_emb_t.unsqueeze(1)
                if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Reshaped Pref_emb_t to: {Pref_emb_t.shape}")
            
            optimizer.zero_grad()
            try:
                if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Train Batch 0 - Forward pass...")
                pred_index_logits, pred_k_lambda_values = model(
                    S_t, S_hist, A_hist, H_dec_t, Src_emb_t, Pref_emb_t,
                    src_key_padding_mask=mask
                )
                if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Train Batch 0 - Forward pass successful. Output shapes: logits={pred_index_logits.shape}, k_lambda={pred_k_lambda_values.shape}")
            except Exception as e:
                logger.error(f"Model forward pass failed: {e}")
                if IS_DEBUG: print(f"[DEBUG][train_policy_network] ERROR during forward pass: {e}")
                logger.error(f"Input shapes: S_t={S_t.shape if S_t is not None else 'None'}, S_hist={S_hist.shape if S_hist is not None else 'None'}, A_hist={A_hist.shape if A_hist is not None else 'None'}")
                if H_dec_t is not None: logger.error(f"H_dec_t shape: {H_dec_t.shape}")
                if Src_emb_t is not None: logger.error(f"Src_emb_t shape: {Src_emb_t.shape}")
                if Pref_emb_t is not None: logger.error(f"Pref_emb_t shape: {Pref_emb_t.shape}")
                exit(1)

            if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Train Batch 0 - Calculating loss...")
            loss, loss_idx, loss_kl = criterion(
                pred_index_logits, pred_k_lambda_values, A_star
            )
            if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Train Batch 0 - Loss: {loss.item():.4f}, Idx: {loss_idx.item():.4f}, KL: {loss_kl.item():.4f}")

            if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Train Batch 0 - Backward pass...")
            loss.backward()
            if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Train Batch 0 - Optimizer step...")
            optimizer.step()

            train_loss += loss.item()
            train_loss_idx += loss_idx.item()
            train_loss_kl += loss_kl.item()
            pbar_train.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_train_loss_idx = train_loss_idx / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_train_loss_kl = train_loss_kl / len(train_loader) if len(train_loader) > 0 else 0.0
        
        if IS_DEBUG: print(
            f"[DEBUG][train_policy_network] Epoch {epoch+1} Train Avg Loss: {avg_train_loss:.4f} "
            f"(Idx: {avg_train_loss_idx:.4f}, KL: {avg_train_loss_kl:.4f})"
        )
        
        # --- VALIDATION LOOP ---
        if IS_DEBUG: print(f"[DEBUG][train_policy_network] Epoch {epoch+1} Starting Validation...")
        model.eval()
        val_loss, val_loss_idx, val_loss_kl = 0.0, 0.0, 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs_student} [VALID]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar_val):
                if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Val Batch {batch_idx+1}/{len(val_loader)} - Unpacking data...")
                S_t, S_hist, A_hist, H_dec_t, Src_emb_t, Pref_emb_t, mask, A_star = \
                    [b.to(device) if torch.is_tensor(b) else None for b in batch]
                
                H_dec_t = H_dec_t if H_dec_t is not None else None
                Src_emb_t = Src_emb_t if Src_emb_t is not None else None
                Pref_emb_t = Pref_emb_t if Pref_emb_t is not None else None
                
                if IS_DEBUG and batch_idx==0:
                     print(
                        f"[DEBUG][train_policy_network] Val Batch 0 data shapes on device: S_t={S_t.shape if S_t is not None else 'None'}, "
                        f"S_hist={S_hist.shape if S_hist is not None else 'None'}, A_hist={A_hist.shape if A_hist is not None else 'None'}, "
                        f"H_dec_t={H_dec_t.shape if H_dec_t is not None else 'None'}, Src_emb_t={Src_emb_t.shape if Src_emb_t is not None else 'None'}, "
                        f"Pref_emb_t={Pref_emb_t.shape if Pref_emb_t is not None else 'None'}, mask={mask.shape if mask is not None else 'None'}, "
                        f"A_star={A_star.shape if A_star is not None else 'None'}"
                    )
                
                if S_t is not None and S_t.dim() == 2:
                    S_t = S_t.unsqueeze(1)
                    if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Val Reshaped S_t to: {S_t.shape}")
                if Src_emb_t is not None and Src_emb_t.dim() == 2:
                    Src_emb_t = Src_emb_t.unsqueeze(1)
                    if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Val Reshaped Src_emb_t to: {Src_emb_t.shape}")
                if Pref_emb_t is not None and Pref_emb_t.dim() == 2:
                    Pref_emb_t = Pref_emb_t.unsqueeze(1)
                    if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Val Reshaped Pref_emb_t to: {Pref_emb_t.shape}")
                
                try:
                    if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Val Batch 0 - Forward pass...")
                    pred_index_logits, pred_k_lambda_values = model(
                        S_t, S_hist, A_hist, H_dec_t, Src_emb_t, Pref_emb_t,
                        src_key_padding_mask=mask
                    )
                    if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Val Batch 0 - Forward pass successful.")
                except Exception as e:
                     logger.error(f"Validation forward pass failed: {e}")
                     if IS_DEBUG: print(f"[DEBUG][train_policy_network] ERROR during validation forward pass: {e}")
                     exit(1)
                
                if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Val Batch 0 - Calculating loss...")
                loss, loss_idx, loss_kl = criterion(
                    pred_index_logits, pred_k_lambda_values, A_star
                )
                if IS_DEBUG and batch_idx==0: print(f"[DEBUG][train_policy_network] Val Batch 0 - Loss: {loss.item():.4f}")
                
                val_loss += loss.item()
                val_loss_idx += loss_idx.item()
                val_loss_kl += loss_kl.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        avg_val_loss_idx = val_loss_idx / len(val_loader) if len(val_loader) > 0 else float('inf')
        avg_val_loss_kl = val_loss_kl / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        if IS_DEBUG: print(
            f"[DEBUG][train_policy_network] Epoch {epoch+1} Val Avg Loss: {avg_val_loss:.4f} "
            f"(Idx: {avg_val_loss_idx:.4f}, KL: {avg_val_loss_kl:.4f})"
        )
        logger.info(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} (Idx: {avg_train_loss_idx:.4f}, KL: {avg_train_loss_kl:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (Idx: {avg_val_loss_idx:.4f}, KL: {avg_val_loss_kl:.4f})"
        )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = output_dir / "pi_phi_best.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best student model saved to {model_path} (Val Loss: {best_val_loss:.4f})")
            if IS_DEBUG: print(f"[DEBUG][train_policy_network] New best model saved at epoch {epoch+1}")
            
            # Save associated config for the Student model
            config_path = output_dir / "pi_phi_config.json"
            with open(config_path, 'w') as f:
                json.dump(teacher_config, f, indent=4)
            logger.info(f"Associated Teacher config saved to {config_path}")
            if IS_DEBUG: print(f"[DEBUG][train_policy_network] Teacher config saved alongside best model.")
            
    logger.info("PHASE B: Training 'Student' Policy Network (pi_phi) COMPLETE.")
    if IS_DEBUG: print("[DEBUG][train_policy_network] Finished Phase B.")

def generate_teacher_dataset(args: argparse.Namespace, teacher_components: Dict):
    """
    PHASE A: Generates the training dataset for the Student Policy.
    
    It uses the loaded Dynamics Model (T_theta) to act as a Teacher. It processes the
    raw training data, scales it, and for each sample, runs an online optimization
    algorithm to find the best action A_t* that minimizes V(S_t+1).
    
    The resulting (Input, A_t*) pairs are saved as an NPZ file.
    """
    logger.info("=" * 80)
    logger.info("PHASE A: Generating 'Teacher' (pi*) Dataset with Detailed Results")
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Starting Phase A.")
    logger.info("=" * 80)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Extracting parameters from teacher_components...")

    # 1. Extract the Teacher components and config
    teacher_config = teacher_components["teacher_config"]
    scaler = teacher_components["scaler"]
    teacher_model = teacher_components["teacher_model"].to(device).eval() # T_theta
    logger.info(f"Using scaler assumed to be loaded from: {Path(args.teacher_model_dir) / "scaler.joblib"}")
    if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] Scaler object type: {type(scaler)}")
    V_fn = teacher_components["V_fn"]

    history_len = teacher_config['history_len']
    predict_delta = teacher_config.get('predict_delta', False)
    final_text_embedding_dim = teacher_components['final_text_embedding_dim']
    use_text_embeddings_flag = teacher_components['use_text_embeddings_flag']
    use_decoder_hidden_state_flag = teacher_config['use_decoder_hidden_state']
    action_dim_config = teacher_config.get('action_dim', ACTION_DIM)

    # Ensure global dimensions are set for correct data loading
    if not hasattr(t_train_Transformer, 'STATE_COLS_DEFAULT') or t_train_Transformer.S_DIM == 0:
         raise RuntimeError("Global state dimensions were not set correctly during teacher loading.")
    
    # 2. Prepare Data Loader
    logger.info("Preparing to load data using unified data processing function...")
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Setting up for data_cache_and_split...")
    
    if not os.path.exists(args.source_csv):
        raise FileNotFoundError(f"Source CSV not found: {args.source_csv}")
    
    try:
        # Determine necessary columns
        required_load_cols = set(
            t_train_Transformer.STATE_COLS_DEFAULT + 
            [
                'action_index_type', 'action_k', 'action_lambda', 'Strategy', 'sample_id', 
                'beam_id', 'step', 'error_norm', 'source_text', 'generated_prefix' 
            ]
        )
        
        if use_decoder_hidden_state_flag:
            required_load_cols.add('decoder_hidden_state')
        
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] Required columns: {required_load_cols}")
        df_cols = pd.read_csv(args.source_csv, nrows=0).columns
        cols_to_read = [col for col in required_load_cols if col in df_cols]
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] Columns available to read: {cols_to_read}")
        
        main_df = pd.read_csv(args.source_csv, usecols=cols_to_read)
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] CSV loaded. Shape: {main_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load CSV from {args.source_csv}: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    val_df_external = None
    if args.t_train_val_path and os.path.exists(args.t_train_val_path):
        logger.info(f"Loading external validation CSV from: {args.t_train_val_path}")
        val_df_external = pd.read_csv(args.t_train_val_path, usecols=cols_to_read)
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] External val CSV loaded. Shape: {val_df_external.shape}")
    
    # 3. Setup Sentence Embedder
    logger.info("Loading SentenceTransformer for text embeddings...")
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Loading SentenceTransformer...")
    try:
        sbert_model_name = project_config.MODEL_NAMES.get("sentence_encoder", 'sentence-transformers/LaBSE')
        text_embedder = SentenceTransformer(sbert_model_name, device=str(device))
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] SentenceTransformer loaded: {sbert_model_name}")
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # 4. Execute unified data loading
    logger.info("Using data_cache_and_split function...")
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Calling data_cache_and_split...")
    try:
        data_results = data_cache_and_split(
            main_df=main_df,
            val_df_external=val_df_external,
            text_embedder=text_embedder,
            sbert_model_name=sbert_model_name,
            save_dir=Path(args.teacher_model_dir),
            cache_path_base=args.cache_path,
            source_csv_path=args.source_csv,
            val_csv_path=args.t_train_val_path,
            val_ratio=teacher_config.get('val_ratio', 0.2),
            seed=args.seed,
            use_decoder_hidden_state=use_decoder_hidden_state_flag,
            history_len=history_len,
            train_path_base_cut_num=args.source_csv_base_cut_num,
            device=str(device),
            cox_event_threshold=teacher_config.get('cox_event_threshold', 2.0),
            action_dim=action_dim_config,
            state_cols_default = teacher_config['STATE_COLS_DEFAULT'],
            s_dim = teacher_config.get('S_DIM', len(teacher_config['STATE_COLS_DEFAULT']))
        )
        
        logger.info("Unified data processing complete.")
        if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] data_cache_and_split completed successfully")
    except Exception as e:
        logger.error(f"Error in data_cache_and_split: {e}")
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] ERROR in data_cache_and_split: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract training data (student learns from train set only)
    logger.info("Extracting training data from unified processing results...")
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Extracting train data only...")
    
    S_t_all, S_hist_all, S_tp1_target_all = data_results['S_t_tr'], data_results['S_hist_tr'], data_results['S_tp1_tr']
    A_all, A_hist_all = data_results['A_tr'], data_results['A_hist_tr']
    seq_id_all = data_results['seq_tr']
    T_all, E_all = data_results['T_tr'], data_results['E_tr']
    source_texts_all, prefixes_all = data_results['src_tr'], data_results['pref_tr']
    hs_all, src_emb_all, pref_emb_all = data_results['hs_tr'], data_results['src_emb_tr'], data_results['pref_emb_tr']
    
    # Update scaler reference to the one returned by data processing
    scaler = data_results['scaler']
    
    logger.info(f"Using {len(S_t_all)} training samples for policy network dataset generation")
    if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] Train data extracted: {len(S_t_all)} samples")
    
    # 5. Scale state data using the Teacher's Scaler
    logger.info("Scaling state data (S_t, S_hist) using loaded Teacher scaler...")
    try:
        if IS_DEBUG: 
            print(f"[DEBUG][generate_teacher_dataset] Scaling S_t (shape: {S_t_all[:, :t_train_Transformer.S_DIM].shape}) using columns: {t_train_Transformer.STATE_COLS_DEFAULT}") 
        
        # Scale S_t
        S_t_all_scaled = scaler.transform(pd.DataFrame(S_t_all[:, :t_train_Transformer.S_DIM], columns=t_train_Transformer.STATE_COLS_DEFAULT)) 

        # Verify S_t scaling in debug mode
        if IS_DEBUG:
            s_t_scaled_mean = np.mean(S_t_all_scaled, axis=0)
            s_t_scaled_std = np.std(S_t_all_scaled, axis=0)
            print(f"[DEBUG][generate_teacher_dataset] S_t_all_scaled stats (ALL {len(s_t_scaled_mean)}) - Mean: {np.round(s_t_scaled_mean, 6)}, Std: {np.round(s_t_scaled_std, 6)}")
            
            # Check StandardScaler columns for mean approx 0
            standard_col_names = t_train_Transformer.TRANSFORM_COLS.get("standard", [])
            standard_indices = [i for i, col in enumerate(t_train_Transformer.STATE_COLS_DEFAULT) 
                                if col in standard_col_names]
            
            if standard_indices:
                standard_means = s_t_scaled_mean[standard_indices]
                is_close = np.isclose(standard_means, 0.0, atol=1.0)
                if not np.all(is_close):
                    failing_local_idx = np.where(~is_close)[0]
                    failing_global_idx = [standard_indices[i] for i in failing_local_idx]
                    failing_cols = [t_train_Transformer.STATE_COLS_DEFAULT[i] for i in failing_global_idx]
                    failing_means = standard_means[failing_local_idx]
                    error_msg = (
                        f"StandardScaler columns have mean too far from 0! "
                        f"Failing columns: {failing_cols}. Their means: {failing_means}"
                    )
                    logger.error(error_msg)
                    raise AssertionError(error_msg)
                print("[DEBUG][generate_teacher_dataset] StandardScaler columns check passed.")
            
            # Log info for other transforms
            power_cols = t_train_Transformer.TRANSFORM_COLS.get("power", [])
            quantile_cols = t_train_Transformer.TRANSFORM_COLS.get("quantile", [])
            
            for i, col in enumerate(t_train_Transformer.STATE_COLS_DEFAULT):
                mean_val = s_t_scaled_mean[i]
                std_val = s_t_scaled_std[i]
                if col in power_cols:
                    logger.info(f"Power-transformed '{col}': mean={mean_val:.4f}, std={std_val:.4f}")
                elif col in quantile_cols:
                    if abs(mean_val) > 2.0:
                        logger.warning(f"Quantile-transformed '{col}': mean={mean_val:.4f}, std={std_val:.4f} (large deviation)")
                    else:
                        logger.info(f"Quantile-transformed '{col}': mean={mean_val:.4f}, std={std_val:.4f}")

        # Scale S_hist
        n_samples = S_hist_all.shape[0]
        hist_len = S_hist_all.shape[1]
        S_hist_2d = S_hist_all.reshape(-1, t_train_Transformer.S_DIM)
        S_hist_2d_scaled = scaler.transform(pd.DataFrame(S_hist_2d, columns=t_train_Transformer.STATE_COLS_DEFAULT))
        S_hist_all_scaled = S_hist_2d_scaled.reshape(n_samples, hist_len, -1)
        
        if IS_DEBUG:
            print(f"[DEBUG][generate_teacher_dataset] Scaled shapes: S_t={S_t_all_scaled.shape}, S_hist={S_hist_all_scaled.shape}")
            print("[DEBUG][generate_teacher_dataset] S_t scaling completed successfully.")
        
        logger.info("State data scaled successfully.")
        
    except Exception as e:
        logger.error(f"Failed to scale data using the loaded scaler: {e}")
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] ERROR during scaling: {e}")
        logger.error("Ensure the scaler was trained on the correct columns/data and history_len is handled.")
        raise

    # 6. Create DataLoader for Teacher Inference
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Creating temporary PAECDataset for inference...")
    try:
        if IS_DEBUG:
            print(f"  Passing to PAECDataset - src_emb_all type: {type(src_emb_all)}, shape: {src_emb_all.shape if src_emb_all is not None else 'None'}")
            print(f"  Passing to PAECDataset - pref_emb_all type: {type(pref_emb_all)}, shape: {pref_emb_all.shape if pref_emb_all is not None else 'None'}")

        temp_dataset = PAECDataset(
            S_t=S_t_all_scaled,
            S_hist=S_hist_all_scaled,
            A=A_all,
            A_hist=A_hist_all,
            S_tp1=S_tp1_target_all,
            T=np.array(T_all), E=np.array(E_all),
            source_embeddings=src_emb_all,
            prefix_embeddings=pref_emb_all,
            decoder_hidden_states=hs_all,
            seq_id=seq_id_all
        )
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] PAECDataset created. Length: {len(temp_dataset)}") 
    except Exception as e: 
        logger.error(f"Failed to create PAECDataset: {e}") 
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] ERROR creating PAECDataset: {e}") 
        raise 

    if IS_DEBUG:
        print("\n ***** ACTION REQUIRED: Please verify PAECDataset.__getitem__ in t_train_Transformer.py *****")
        print(" Ensure it returns a dictionary containing keys 'source_embeddings' and 'prefix_embeddings' when embeddings are present.")

    teacher_inference_loader = DataLoader( 
        temp_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=(device == torch.device('cuda')) 
    )
    if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] DataLoader created. Number of batches: {len(teacher_inference_loader)}") 

    # 7. Iteratively calculate A_t* (Teacher's optimal actions)
    policy_data_inputs = [] 
    policy_data_labels_detailed = [] 

    logger.info("Starting computation of optimal actions (A_t*) using Teacher pi*...") 
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Starting A_t* computation loop...") 
    pbar = tqdm(teacher_inference_loader, desc="Generating A_t* Labels") 
    
    for batch_idx, batch in enumerate(pbar): 
        if IS_DEBUG: print(f"\n[DEBUG][generate_teacher_dataset] Processing batch {batch_idx+1}/{len(teacher_inference_loader)}...") 
        S_t_scaled_batch = batch["S_t"].to(device) 
        S_hist_scaled_batch = batch["S_hist"].to(device) 
        A_hist_batch = batch["A_hist"].to(device) 

        if IS_DEBUG and batch_idx == 0:
            print("[DEBUG][generate_teacher_dataset] Inspecting first batch from DataLoader:")
            print(f"  Batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"  Key '{key}': type={type(value)}, shape={value.shape}, device={value.device}")
                else:
                    print(f"  Key '{key}': type={type(value)}")
            print("-" * 20)

        # Handle optional inputs
        hs_batch = batch.get("decoder_hidden_state") 
        hs_is_valid = hs_batch is not None and torch.is_tensor(hs_batch) and hs_batch.numel() > 0
        hs_batch = hs_batch.to(device) if hs_is_valid else None
        
        src_emb_batch = batch.get("source_embeddings") 
        src_is_valid = src_emb_batch is not None and torch.is_tensor(src_emb_batch) and src_emb_batch.numel() > 0
        src_emb_batch = src_emb_batch.to(device) if src_is_valid else None

        pref_emb_batch = batch.get("prefix_embeddings") 
        pref_is_valid = pref_emb_batch is not None and torch.is_tensor(pref_emb_batch) and pref_emb_batch.numel() > 0
        pref_emb_batch = pref_emb_batch.to(device) if pref_is_valid else None

        # Masking
        mask_batch = batch.get("mask", torch.zeros(S_t_scaled_batch.shape[0], history_len + 1, dtype=torch.bool)).to(device)

        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] Calling compute_optimal_action for batch {batch_idx+1}...")

        try:
            # === Teacher Optimization Step ===
            # Compute the optimal action A* for each sample in the batch by running gradient descent on T_theta
            A_t_star_batch_details_list = compute_optimal_action(
                teacher_model=teacher_model,
                V_fn=V_fn,
                S_t_batch=S_t_scaled_batch,
                S_hist_batch=S_hist_scaled_batch,
                A_hist_batch=A_hist_batch,
                decoder_hidden_state_batch=hs_batch,
                source_embedding_batch=src_emb_batch,
                prefix_embedding_batch=pref_emb_batch,
                teacher_config=teacher_config,
                device=str(device) 
            )
            
            # Process results
            batch_labels_dict = defaultdict(list)
            overall_best_action_tensors_batch = [] 

            for sample_detail in A_t_star_batch_details_list: 
                details = sample_detail['details'] 
                rank_all = sample_detail['rank']
                best_knn_type = sample_detail['best_knn_type'] # The best Active retrieval index
                
                # Check for completeness
                current_sample_index = len(batch_labels_dict.get('rank', []))
                if IS_DEBUG and current_sample_index == 0 and (batch_idx % 50 == 0):
                    if 'ivf_pq' in details:
                        print(f"[DEBUG][generate_teacher_dataset] Batch {batch_idx+1}, Sample 0: 'ivf_pq' details = {details['ivf_pq']}")
                    else:
                        print(f"[DEBUG][generate_teacher_dataset] Batch {batch_idx+1}, Sample 0: WARNING! 'ivf_pq' key MISSING in 'details' dict!")

                # Store costs and params for all indices
                for index_type in ['exact', 'hnsw', 'ivf_pq', 'none']: 
                    detail = details.get(index_type, {'k': -1, 'lambda': -1.0, 'cost': float('inf')}) 
                    batch_labels_dict[f'cost_{index_type}'].append(detail['cost']) 
                    if index_type != 'none': 
                        batch_labels_dict[f'k_{index_type}'].append(detail['k']) 
                        batch_labels_dict[f'lambda_{index_type}'].append(detail['lambda']) 

                batch_labels_dict['rank'].append(",".join(rank_all))

                # Identify A_star (Teacher's best ACTIVE action)
                best_details = details[best_knn_type] 
                best_action_obj = Action(k=best_details['k'], index_type=best_knn_type, lambda_weight=best_details['lambda'])
                
                # Ensure valid k
                if best_action_obj.k == 0:
                    logger.warning(f"Best kNN action '{best_knn_type}' resulted in k=0. Forcing k=1.")
                    best_action_obj.k = 1
               
                overall_best_action_tensors_batch.append(_action_to_tensor(best_action_obj, str(device))) 

            # Debug check for missing keys in result dict
            if IS_DEBUG and (batch_idx % 50 == 0 or batch_idx == len(pbar) - 1):
                if 'k_ivf_pq' not in batch_labels_dict:
                    print(f"[DEBUG][generate_teacher_dataset] ERROR: 'k_ivf_pq' IS MISSING from batch_labels_dict keys!")
            
            # Aggregate batch labels
            A_t_star_tensors = torch.cat(overall_best_action_tensors_batch, dim=0)
            batch_labels_dict['A_star'] = A_t_star_tensors.cpu().numpy() # type: ignore

            for key in batch_labels_dict: 
                 if key == 'rank': 
                     batch_labels_dict[key] = np.array(batch_labels_dict[key], dtype=object) # type: ignore
                 else: 
                     batch_labels_dict[key] = np.array(batch_labels_dict[key]) # type: ignore

            if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] Processed detailed A* results for batch {batch_idx+1}.") 

        except Exception as e: 
            logger.error(f"Error during compute_optimal_action or result processing: {e}") 
            if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] ERROR in compute_optimal_action/processing: {e}") 
            traceback.print_exc() 
            logger.warning("Skipping batch due to error.") 
            continue 

        # Store input features and labels for this batch
        policy_data_inputs.append({ 
            'S_t': S_t_scaled_batch.cpu().numpy(), 
            'S_hist': S_hist_scaled_batch.cpu().numpy(), 
            'A_hist': A_hist_batch.cpu().numpy(), 
            'H_dec_t': hs_batch.cpu().numpy() if hs_batch is not None else None, 
            'Src_emb_t': src_emb_batch.cpu().numpy() if src_emb_batch is not None else None, 
            'Pref_emb_t': pref_emb_batch.cpu().numpy() if pref_emb_batch is not None else None, 
            'mask': mask_batch.cpu().numpy() 
        })
        policy_data_labels_detailed.append(batch_labels_dict) 

    logger.info("A_t* label generation complete.") 
    if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] A_t* loop finished. Collected {len(policy_data_inputs)} input batches and {len(policy_data_labels_detailed)} label batches.") 

    if not policy_data_inputs or not policy_data_labels_detailed: 
        logger.error("No data generated for policy training. Exiting.") 
        if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] ERROR: No data generated.") 
        exit(1) 

    # 8. Consolidate and Save (.npz)
    logger.info(f"Saving policy training data to {args.policy_data_path} (using .npz format)...") 
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Flattening data for npz saving...")
    try: 
        def flatten_policy_data(data_list: List[Dict[str, np.ndarray]], key: str): 
            """Helper to concatenate batches of numpy arrays."""
            if not data_list or not isinstance(data_list[0], dict) or key not in data_list[0] or data_list[0][key] is None: 
                if IS_DEBUG: print(f"[DEBUG][flatten_policy_data] Key '{key}' not found or data is None in input list.") 
                return None 
            try: 
                valid_arrays = [item[key] for item in data_list if item[key] is not None] 
                if not valid_arrays: 
                    if IS_DEBUG: print(f"[DEBUG][flatten_policy_data] No valid arrays found for key '{key}'.") 
                    return None 
                shapes = {arr.shape[1:] for arr in valid_arrays if isinstance(arr, np.ndarray)} 
                if len(shapes) > 1: 
                    logger.warning(f"Inconsistent non-batch dimensions detected for key '{key}': {shapes}. Skipping concatenation.") 
                    if IS_DEBUG: print(f"[DEBUG][flatten_policy_data] Inconsistent non-batch dimensions for key '{key}'.") 
                    return None 
                concatenated = np.concatenate(valid_arrays, axis=0) 
                if IS_DEBUG: print(f"[DEBUG][flatten_policy_data] Successfully concatenated '{key}'. Shape: {concatenated.shape}") 
                return concatenated 
            except ValueError as e: 
                # Handle object arrays (e.g. rank strings)
                if key == 'rank' and all(isinstance(item[key], np.ndarray) and item[key].dtype == object for item in data_list if item[key] is not None): 
                     try: 
                         concatenated = np.concatenate(valid_arrays, axis=0) 
                         if IS_DEBUG: print(f"[DEBUG][flatten_policy_data] Successfully concatenated object array '{key}'. Shape: {concatenated.shape}") 
                         return concatenated 
                     except ValueError as e_obj: 
                         logger.warning(f"Could not concatenate object array {key}: {e_obj}") 
                         return None 
                else: 
                    logger.warning(f"Could not concatenate {key}, possibly empty batches or shape mismatches: {e}") 
                    return None 

        S_t_flat = flatten_policy_data(policy_data_inputs, 'S_t') 
        if S_t_flat is None: 
            raise ValueError("Generated S_t data is empty or could not be concatenated.")

        # Data check
        if IS_DEBUG:
            s_t_flat_mean = np.mean(S_t_flat, axis=0)
            s_t_flat_std = np.std(S_t_flat, axis=0)
            print(f"[DEBUG][generate_teacher_dataset] Final S_t_flat statistics (ALL {len(s_t_flat_mean)} columns):")
            print(f"  Mean: {s_t_flat_mean}")
            print(f"  Std: {s_t_flat_std}")
            
            # Check standard columns
            standard_col_names = t_train_Transformer.TRANSFORM_COLS.get("standard", [])
            standard_indices = [i for i, col in enumerate(t_train_Transformer.STATE_COLS_DEFAULT) if col in standard_col_names]
            
            if standard_indices:
                standard_means = s_t_flat_mean[standard_indices]
                is_close = np.isclose(standard_means, 0.0, atol=1.0)
                if not np.all(is_close):
                    failing_local_idx = np.where(~is_close)[0]
                    failing_global_idx = [standard_indices[i] for i in failing_local_idx]
                    failing_cols = [t_train_Transformer.STATE_COLS_DEFAULT[i] for i in failing_global_idx]
                    failing_means = standard_means[failing_local_idx]
                    logger.warning(
                        f"Some StandardScaler columns in final data have mean far from 0: "
                        f"{failing_cols} with means {failing_means}. "
                        f"This is acceptable for train/val subsets."
                    )
                else:
                    print("[DEBUG][generate_teacher_dataset] Final S_t_flat StandardScaler columns check passed.")
        
        num_samples = len(S_t_flat) 
        logger.info(f"Total samples generated: {num_samples}") 
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] Total samples after flattening: {num_samples}") 

        # Prepare dict for npz saving
        save_dict = {'S_t': S_t_flat} 

        for key in ['S_hist', 'A_hist', 'mask', 'H_dec_t', 'Src_emb_t', 'Pref_emb_t']: 
            flat_array = flatten_policy_data(policy_data_inputs, key) 
            if flat_array is not None: 
                if len(flat_array) == num_samples: 
                    save_dict[key] = flat_array 
                    if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] Added input '{key}' to save_dict. Shape: {flat_array.shape}") 
                else: 
                    logger.warning(f"Length mismatch for input '{key}' ({len(flat_array)} vs {num_samples}). Skipping.") 
            elif key in ['S_hist', 'A_hist', 'mask']: 
                 if key == 'S_hist' and history_len == 0: 
                     save_dict[key] = np.zeros((num_samples, 0, t_train_Transformer.S_DIM), dtype=np.float32) 
                 elif key == 'A_hist' and history_len == 0: 
                     save_dict[key] = np.zeros((num_samples, 0, ACTION_DIM), dtype=np.float32) 
                 elif key == 'mask': 
                     logger.warning(f"Flattening failed for essential input 'mask'. Using default.") 
                     save_dict[key] = np.zeros((num_samples, history_len + 1), dtype=bool) 
                 else: 
                    raise ValueError(f"Flattening failed for essential input '{key}'.") 

        label_keys = ['A_star', 'k_exact', 'lambda_exact', 'cost_exact', 
                      'k_hnsw', 'lambda_hnsw', 'cost_hnsw', 
                      'k_ivf_pq', 'lambda_ivf_pq', 'cost_ivf_pq', 
                      'cost_none', 'rank'] 

        def flatten_label_data(label_batches_list: List[Dict[str, np.ndarray]], key: str): 
            try: 
                arrays_to_concat = [batch_dict[key] for batch_dict in label_batches_list if key in batch_dict and batch_dict[key] is not None] 
                if not arrays_to_concat: 
                    logger.warning(f"Label key '{key}' not found or was None in all batch results.") 
                    return None 
                
                # Check types and shapes
                first_shape = arrays_to_concat[0].shape[1:] if arrays_to_concat[0].ndim > 1 else () 
                first_dtype = arrays_to_concat[0].dtype 
                if not all(arr.shape[1:] == first_shape and arr.dtype == first_dtype for arr in arrays_to_concat): 
                    if key == 'rank' and all(arr.dtype == object for arr in arrays_to_concat): 
                         pass 
                    else: 
                         shapes_dtypes = {(arr.shape[1:], str(arr.dtype)) for arr in arrays_to_concat} 
                         logger.warning(f"Inconsistent shape or dtype found for label key '{key}'. Found: {shapes_dtypes}. Skipping concatenation.") 
                         return None 
                concatenated = np.concatenate(arrays_to_concat, axis=0) 
                return concatenated 
            except Exception as e: 
                logger.error(f"Error flattening label key '{key}': {e}") 
                return None 

        for key in label_keys: 
            flattened_label_array = flatten_label_data(policy_data_labels_detailed, key) 
            if flattened_label_array is not None: 
                if len(flattened_label_array) == num_samples: 
                    save_dict[key] = flattened_label_array 
                    if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] Added label '{key}' to save_dict. Shape: {flattened_label_array.shape}, Dtype: {flattened_label_array.dtype}") 
                else: 
                    logger.warning(f"Length mismatch for label '{key}' ({len(flattened_label_array)} vs {num_samples}). Skipping.") 
            elif key == 'A_star': 
                raise ValueError(f"Essential label 'A_star' could not be flattened or is missing.") 

        # Save to file
        output_path_npz = Path(args.policy_data_path).with_suffix('.npz') 
        output_path_npz.parent.mkdir(parents=True, exist_ok=True) 
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] Saving final data to NPZ: {output_path_npz} with keys: {list(save_dict.keys())}") 

        np.savez_compressed(output_path_npz, **save_dict) # type: ignore 
        logger.info(f"Policy network training data saved successfully to {output_path_npz} ({num_samples} samples)") 

    except Exception as e: 
        logger.error(f"Error occurred during final data preparation or saving (NPZ): {e}") 
        if IS_DEBUG: print(f"[DEBUG][generate_teacher_dataset] ERROR during NPZ prep/saving: {e}") 
        traceback.print_exc() 
        sys.exit(1) 

    logger.info("PHASE A: Generating 'Teacher' (pi*) Dataset COMPLETE.") 

    # Cleanup memory
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Cleaning up memory...") 
    if 'main_df' in locals() and main_df is not None: del main_df 
    if 'S_t_all' in locals() and S_t_all is not None: del S_t_all, S_hist_all, A_all, A_hist_all, S_tp1_target_all, seq_id_all, T_all, E_all 
    if 'source_texts_all' in locals() and source_texts_all is not None: del source_texts_all, prefixes_all, hs_all 
    if 'src_emb_all' in locals() and src_emb_all is not None: del src_emb_all 
    if 'pref_emb_all' in locals() and pref_emb_all is not None: del pref_emb_all 
    if 'S_t_all_scaled' in locals() and S_t_all_scaled is not None: del S_t_all_scaled 
    if 'S_hist_all_scaled' in locals() and S_hist_all_scaled is not None: del S_hist_all_scaled 
    if 'temp_dataset' in locals() and temp_dataset is not None: del temp_dataset 
    if 'teacher_inference_loader' in locals() and teacher_inference_loader is not None: del teacher_inference_loader 
    if 'policy_data_inputs' in locals() and policy_data_inputs is not None: del policy_data_inputs 
    if 'policy_data_labels_detailed' in locals() and policy_data_labels_detailed is not None: del policy_data_labels_detailed 
    if 'save_dict' in locals() and save_dict is not None: del save_dict 
    import gc
    gc.collect() 
    torch.cuda.empty_cache() 
    if IS_DEBUG: print("[DEBUG][generate_teacher_dataset] Cleanup complete.") 

def main(): 
    global IS_DEBUG 
    
    parser = argparse.ArgumentParser(description="Train PAEC Policy Network (pi_phi) using Teacher-Student (pi*)")
    parser.add_argument('--teacher-model-dir', type=str, 
                        default=str(project_config.PATHS["dynamics_model_dir"] / 'Champion'), 
                        help='Path to the *existing* Teacher T_theta model directory (e.g., "Champion") ' 
                             'which contains checkpoint_best.pt (or _last.pt), scaler.joblib, and config.json.')
    parser.add_argument('--use-last-ckpt', action='store_true', help='Use checkpoint_last.pt instead of checkpoint_best.pt for the Teacher model.')
    parser.add_argument("--t_train_val_path", type=str, default="")
    parser.add_argument("--cache_path", type=str, default=project_config.PATHS["cache_dir"])
    parser.add_argument('--source-csv', type=str, 
                        default=str(project_config.PATHS["processed_data_dir"] / 'training_data_stepwise' / 'strategy_comparison_stepwise_1000.csv'), 
                        help='Path to the original T_theta training data CSV.')
    parser.add_argument("--source-csv-base-cut-num", type=int, default=2)
 
    parser.add_argument('--datastore-path', type=str, 
                        default=str(project_config.PATHS["datastore_dir"]), 
                        help='Path to the kNN datastore (needed *only* if Teacher model used datastore during its training/data gen). Currently UNUSED in Phase A.') 
    parser.add_argument('--policy-data-path', type=str, 
                        default=str(project_config.PATHS["processed_data_dir"] / 'policy_network_training_data.npz'),
                        help='Output path for the generated (inputs, A_t*) teacher dataset.') 
    parser.add_argument('--output-model-dir', type=str,
                        default=str(project_config.PATHS["policy_model_dir"]),
                        help='Directory to save the new student model (pi_phi_best.pt) and its config.') 
    parser.add_argument('--val-ratio-student', type=float, default=0.1, help='Validation split ratio for *student* training dataset.') 
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for *both* teacher data generation and student training.') 
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for *student* training.') 
    parser.add_argument('--epochs-student', type=int, default=50, help='Number of epochs for *student* training.') 
    parser.add_argument('--num-workers', type=int, default=min(4, (os.cpu_count() or 1) // 2), help='Number of dataloader workers.') 
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device used for training (cuda or cpu).') 
    parser.add_argument('--seed', type=int, default=42, help='Random seed.') 
    parser.add_argument('--skip-generation', action='store_true', help='Skip PHASE A (Teacher data generation) and go directly to student training.') 
    parser.add_argument('--skip-training', action='store_true', help='Skip PHASE B (Student training) and only generate teacher data.') 
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug printing throughout the script.')
    
    args = parser.parse_args()
    IS_DEBUG = args.debug
    if IS_DEBUG: print("[DEBUG][main] Debug mode enabled.")
    if IS_DEBUG: print(f"[DEBUG][main] Parsed args: {args}")
    set_rand_seed(args.seed)

    # 1. Load the Teacher model components (needed for both phases to ensure consistency)
    if IS_DEBUG: print("[DEBUG][main] Loading teacher components...") 
    try: 
        teacher_components = load_teacher_components(args.teacher_model_dir, args.use_last_ckpt, args.device) 
        teacher_config = teacher_components["teacher_config"]
        if IS_DEBUG: print("[DEBUG][main] Teacher components loaded successfully.") 
    except Exception as e: 
        logger.error(f"Failed to load Teacher components: {e}") 
        if IS_DEBUG: print(f"[DEBUG][main] ERROR loading teacher components: {e}") 
        traceback.print_exc() 
        exit(1)
    
    # 2. Phase A: Generate dataset from Teacher
    if not args.skip_generation: 
        if IS_DEBUG: print("[DEBUG][main] Starting Phase A: generate_teacher_dataset...") 
        generate_teacher_dataset(args, teacher_components) 
    else: 
        logger.info("Skipping PHASE A (Teacher data generation) as requested.") 
        if IS_DEBUG: print("[DEBUG][main] Skipped Phase A.")

    # 3. Phase B: Train Student Policy
    if not args.skip_training: 
        if IS_DEBUG: print("[DEBUG][main] Starting Phase B: train_policy_network...") 
        train_policy_network(args, teacher_config) 
    else: 
        logger.info("Skipping PHASE B (Student training) as requested.") 
        if IS_DEBUG: print("[DEBUG][main] Skipped Phase B.") 
        
    if IS_DEBUG: print("[DEBUG][main] Script finished.")

if __name__ == "__main__": 
    main()
