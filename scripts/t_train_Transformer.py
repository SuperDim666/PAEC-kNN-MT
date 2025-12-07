# -*- coding: utf-8 -*-

import os, sys, locale, argparse, shutil, math, json, random, hashlib, locale, traceback, joblib, torch

from typing import Dict, Tuple, List, Optional, Union, Any, Callable
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.utils import parametrizations
from torch.utils.data import Dataset, DataLoader
# Hardware Acceleration
from torch import jit

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.impute import SimpleImputer
from scipy.stats import norm
from sentence_transformers import SentenceTransformer

# Allow argparse.Namespace to be pickled safely
torch.serialization.add_safe_globals([argparse.Namespace])

# Attempt to import project-specific configuration
try:
    sys.path.append(str(Path(__file__).parent.parent.resolve()))
    from src.config import *
except ImportError as e:
    print("[Error] Failed to import project modules. Ensure the script is run from the 'scripts' directory.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print("[Error] An unexpected error occurred during import of src/config.")
    traceback.print_exc()
    sys.exit(1)

# Configure Pandas display options for better debugging visibility
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# Definition of feature groups for different scaling strategies.
# These keys map to lists of column names that require specific preprocessing techniques.
TRANSFORM_COLS = {
    # Group for features with approximately normal/symmetric distributions
    "standard": [
        'error_semantic',
        'context_consistency',
        'context_stability',
        'context_confidence_volatility',
        'error_fluency_surprisal',
        'error_fluency_repetition',
    ],
    # "power": [ ],
    # Group for features with complex, multi-modal distributions or significant outliers
    "quantile": [
        'error_coverage',
        'pressure_latency',
        'pressure_memory',
        'pressure_throughput',
        'context_faith_focus',
    ]
}

# Base columns (original state S_t)
# These globals will be populated dynamically based on the configuration file.
STATE_COLS_DEFAULT = [""]
E_INDEX = PHI_INDEX = H_INDEX = [0,0]
S_DIM = E_DIM = PHI_DIM = H_DIM = 0

def get_float_precision(arr):
    """
    Determines the machine epsilon (precision) for a given numpy array based on its dtype.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        float: The smallest representable positive number such that 1.0 + eps != 1.0.
    """
    dtype = arr.dtype
    if dtype == np.float16: return 1e-3
    elif dtype == np.float32: return 1.5e-07
    elif dtype == np.float64: return 3e-16
    elif dtype == np.float128: return 2e-19
    else: raise ValueError(f"[Error] Unsupported float dtype: {dtype}")

class NullContext:
    """
    A simple context manager that does nothing.
    Used as a fallback when autocast is disabled or unavailable.
    """
    def __enter__(self): return self
    def __exit__(self, *args): args=args; pass

class InvertableColumnTransformer(ColumnTransformer):
    """
    Extends sklearn.compose.ColumnTransformer to support inverse transformations.

    Standard ColumnTransformers do not easily support `inverse_transform` when features
    are reordered or processed in subsets. This class implements the logic to reconstruct
    the original input space from the transformed feature space, which is critical for
    interpreting model predictions and calculating metrics in the original scale.
    """
    def inverse_transform(self, X):
        """
        Inverse transforms the input X back to the original feature space.

        Args:
            X (array-like or pd.DataFrame): Transformed data.

        Returns:
            np.ndarray: Data in the original feature space.
        """
        is_df = isinstance(X, pd.DataFrame)
        if not is_df:
            X = pd.DataFrame(X, columns=STATE_COLS_DEFAULT)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Received {X.shape[1]} columns but transformer expected {self.n_features_in_}")

        retarr = np.zeros((X.shape[0], self.n_features_in_))
        used_original_indices = []
        for name, transformer, original_cols in self.transformers_:
            if name == 'remainder': continue
            # To avoid warnings, make arr a DF with original group column names
            group_cols = [STATE_COLS_DEFAULT[i] for i in original_cols]
            arr_df = pd.DataFrame(X, columns=group_cols)

            # Special handling for NaN prevention
            arr_np = arr_df.to_numpy()
            if isinstance(transformer, PowerTransformer):
                lambdas = transformer.lambdas_
                epsilon = get_float_precision(arr_np)
                for j in range(arr_np.shape[1]):
                    lambda_j = lambdas[j]
                    if lambda_j != 0:
                        bound = -1.0 / lambda_j
                        if lambda_j > 0: arr_np[:, j] = np.maximum(arr_np[:, j], bound + epsilon)
                        else: arr_np[:, j] = np.minimum(arr_np[:, j], bound - epsilon)
                        unsafe_idx = (arr_np[:, j] * lambda_j + 1) < epsilon
                        if np.any(unsafe_idx):
                            safe_x = (epsilon - 1) / lambda_j
                            arr_np[unsafe_idx, j] = safe_x
                arr_np = np.clip(arr_np, -1e5, 1e5)
                inverted = transformer.inverse_transform(arr_np)
            elif isinstance(transformer, QuantileTransformer):
                arr_np = np.clip(arr_np, -8, 8)
                uniform = norm.cdf(arr_np)
                inverted = np.zeros_like(arr_np)
                for j in range(arr_np.shape[1]):
                    inverted[:, j] = np.interp(
                        uniform[:, j],
                        np.linspace(0, 1,len(transformer.quantiles_[:, j])),
                        transformer.quantiles_[:, j],
                        left=transformer.quantiles_[0, j],
                        right=transformer.quantiles_[-1, j]
                    )
            else:
                inverted = transformer.inverse_transform(arr_df)

            # If inverted is DF, to np
            inverted = inverted.to_numpy() if isinstance(inverted, pd.DataFrame) else inverted

            # Assign to original positions
            retarr[:, original_cols] = inverted
            used_original_indices.extend(original_cols)
        if hasattr(self, 'remainder'):
            remainder = getattr(self, 'remainder')
            if remainder == 'passthrough':
                if 'remainder' in self.output_indices_:
                    remainder_slice = self.output_indices_['remainder']
                    arr = X.iloc[:, remainder_slice.start:remainder_slice.stop]
                    remainder_cols = sorted(set(range(self.n_features_in_)) - set(used_original_indices))
                    group_cols = [STATE_COLS_DEFAULT[i] for i in remainder_cols]
                    arr_df = pd.DataFrame(arr, columns=group_cols)
                    retarr[:, remainder_cols] = arr_df.to_numpy()
        return retarr

# Ensure reproducibility and optimized backend settings
# torch.autograd.set_detect_anomaly(True)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
if hasattr(torch, '_functorch'):
    _functorch = getattr(torch, '_functorch')
    if hasattr(_functorch, 'config') and hasattr(_functorch.config, 'donated_buffer'):
        _functorch.config.donated_buffer = False
if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
    if hasattr(torch.backends.cudnn, 'deterministic'):
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = False

def to_device(batch, device):
    """
    Moves a batch of data (dictionary) to the specified device.
    """
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

def _set_rand_seed(seed: int):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enable scalar capture for PAEC model optimizations
    if hasattr(torch, '_dynamo') and \
       hasattr(torch._dynamo, 'config') and \
       hasattr(torch._dynamo.config, 'capture_scalar_outputs'):
        torch._dynamo.config.capture_scalar_outputs = True

    # # Allow CuDNN to automatically find the fastest convolution algorithm.
    # # May sacrifice some strict reproducibility across hardware or software versions.
    # # 1. Enable during parameter tuning and final training;
    # # 2. Disable when strictly reproducing the results in the paper is required.
    # if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
    #     if hasattr(torch.backends.cudnn, 'deterministic'):
    #         torch.backends.cudnn.deterministic = False
    #     if hasattr(torch.backends.cudnn, 'benchmark'):
    #         torch.backends.cudnn.benchmark = True

# -----------------------------
# Tool functions: numerical, measurement, stability analysis
# -----------------------------

def build_action_candidates(
        S_t: torch.Tensor, A_t: torch.Tensor, num_delta_dirs: int, action_delta: float
    ) -> torch.Tensor:
    """
    Constructs a set of candidate actions by applying local, continuous perturbations
    around the teacher's action A_t. This creates a neighborhood of actions for calculating
    local stability and gradients.

    Perturbations are applied to the continuous components (k_norm, lambda) while
    keeping the discrete component (Index Type) constant.

    Args:
        S_t: Current state tensor (unused in logic but kept for interface consistency).
        A_t: Current action tensor [Batch, Action_Dim].
        num_delta_dirs: Number of continuous dimensions to perturb (0 to 2).
        action_delta: Magnitude of the perturbation.

    Returns:
        torch.Tensor: Candidate actions tensor of shape [K, Batch, Action_Dim].
    """
    device = S_t.device
    B, dA = A_t.shape

    if dA != 6:
        raise ValueError(f"build_action_candidates expects a 6D action tensor, but received {dA}D.")
    if not (0 <= num_delta_dirs <= 2):
        raise ValueError(f"num_delta_dirs must be between 0 and 2 for a 2D continuous space, but got {num_delta_dirs}.")

    # Start with the original teacher action
    candidate_list = [A_t.unsqueeze(0)]  # [1, B, 6]
    A_discrete = A_t[:, :4]   # [B, 4]
    A_continuous = A_t[:, 4:] # [B, 2] (k_norm, lambda)
    num_continuous_dims = A_continuous.shape[1]
    
    # Apply perturbations along specified dimensions
    for i in range(num_delta_dirs):
        
        # Create delta vector for the current dimension
        delta_vec = torch.zeros(1, num_continuous_dims, device=device)
        delta_vec[0, i] = action_delta
        
        # Apply positive and negative perturbations
        perturbed_plus = A_continuous + delta_vec
        perturbed_minus = A_continuous - delta_vec
        
        # Clamp values to valid range [0, 1]
        perturbed_plus = torch.clamp(perturbed_plus, 0.0, 1.0)
        perturbed_minus = torch.clamp(perturbed_minus, 0.0, 1.0)

        # Reconstruct full action vectors
        action_plus = torch.cat([A_discrete, perturbed_plus], dim=1).unsqueeze(0)
        action_minus = torch.cat([A_discrete, perturbed_minus], dim=1).unsqueeze(0)
        
        candidate_list.append(action_plus)
        candidate_list.append(action_minus)
    
    # Concatenate all candidates into a single tensor
    A_cands = torch.cat(candidate_list, dim=0)

    return A_cands.contiguous()

def ljung_box_Q(residuals: np.ndarray, lags: int = 10) -> float:
    """
    Calculates the Ljung-Box Q statistic to test for autocorrelation in the model's
    prediction residuals. A low Q value suggests the residuals are white noise,
    indicating the model has captured the deterministic dynamics well.

    Args:
        residuals: Array of prediction residuals.
        lags: Number of lags to test.

    Returns:
        float: The Q statistic (or NaN if insufficient data).
    """
    r = np.asarray(residuals).reshape(-1)
    n = len(r)
    if n < lags + 2:
        return float("nan")
    r = r - r.mean()
    acfs = []
    denom = np.sum(r ** 2)
    if denom <= 1e-12:
        return 0.0
    for k in range(1, lags + 1):
        acf = np.sum(r[k:] * r[:-k]) / denom
        acfs.append(acf)
    Q = n * (n + 2) * np.sum([(acfs[k - 1] ** 2) / (n - k) for k in range(1, lags + 1)])
    return float(Q)

def spectral_norm_jacobian_mean(
        model, 
        S_t: torch.Tensor, S_hist: torch.Tensor, 
        A_t: torch.Tensor, A_hist: torch.Tensor,
        sample_size: int = 64,
        source_embeddings: Optional[torch.Tensor] = None,
        prefix_embeddings: Optional[torch.Tensor] = None,
        decoder_hidden_state: Optional[torch.Tensor] = None
    ) -> float:
    """
    Estimates the mean spectral norm of the Jacobian of the dynamics model (dT/dS).
    This serves as a proxy for the Lipschitz constant of the model, quantifying its stability.

    Args:
        model: The dynamics model.
        S_t, S_hist, A_t, A_hist: Input batches.
        sample_size: Number of samples to use for estimation.
        source_embeddings, prefix_embeddings, decoder_hidden_state: Optional inputs.

    Returns:
        float: The mean spectral norm across the sampled batch.
    """
    B = S_t.shape[0]
    # Ensure sample_size does not exceed batch size
    sample_size = min(sample_size, B)
    idx = torch.randperm(B, device=S_t.device)[:sample_size]

    # Sample inputs and enable gradients for S_t
    S = S_t[idx].detach().clone().requires_grad_(True)
    Sh = S_hist[idx].detach().clone()
    A = A_t[idx].detach().clone()
    Ah = A_hist[idx].detach().clone()
    src_emb_sub = source_embeddings[idx] if source_embeddings is not None else None
    pref_emb_sub = prefix_embeddings[idx] if prefix_embeddings is not None else None
    hs_sub = decoder_hidden_state[idx] if decoder_hidden_state is not None else None

    spectral_norms = []
    for i in range(sample_size):
        def func(s_single):
            # Forward pass for a single sample
            model_output = model(
                s_single.unsqueeze(0), Sh[i:i+1], A[i:i+1], Ah[i:i+1],
                decoder_hidden_state=hs_sub[i:i+1] if hs_sub is not None else None,
                source_embeddings=src_emb_sub[i:i+1] if src_emb_sub is not None else None,
                prefix_embeddings=pref_emb_sub[i:i+1] if pref_emb_sub is not None else None
            )
            # Concatenate E and H predictions for the Jacobian
            E_pred, H_pred, _ = model_output
            out_single = torch.cat([E_pred, H_pred], dim=1)
            
            return out_single.squeeze(0)

        # Compute exact Jacobian for this sample
        J = torch.autograd.functional.jacobian(func, S[i])
        
        # Compute spectral norm (max singular value)
        singular_values = torch.linalg.svdvals(J)
        sigma = singular_values.max().item()
        spectral_norms.append(sigma)
    
    return sum(spectral_norms) / len(spectral_norms) if spectral_norms else float("nan")

def split_state(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decomposes the full state vector S into its components: Error (E), Pressure (Phi), and Context (H).
    Relies on global index variables.

    Args:
        S: The state tensor [Batch, S_DIM].

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: E, Phi, H tensors.
    """
    E = S[:, E_INDEX[0]:E_INDEX[1]+1]
    Phi = S[:, PHI_INDEX[0]:PHI_INDEX[1]+1]
    H = S[:, H_INDEX[0]:H_INDEX[1]+1]
    return E, Phi, H

def lyapunov_V(S: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Lyapunov scalar value V(S) = E^T P E.
    Used to quantify the "energy" or error magnitude of the system state.

    Args:
        S: The full state tensor [Batch, S_DIM].
        P: The diagonal elements of the positive definite matrix P [E_DIM].

    Returns:
        torch.Tensor: The Lyapunov values [Batch].
    """
    E = S[:, E_INDEX[0] : E_INDEX[1] + 1]
    return torch.sum((E * P.view(1, E_DIM)) * E, dim=1)

def clf_loss(
        model, 
        S_t: torch.Tensor, S_hist: torch.Tensor,
        A_t: torch.Tensor, A_hist: torch.Tensor,
        V_fn: Callable[[torch.Tensor], torch.Tensor], rho: float,
        num_delta_dirs: int = 2, 
        action_delta: float = 0.1, softmin_tau: float = 0.5,
        source_embeddings: Optional[torch.Tensor] = None,
        prefix_embeddings: Optional[torch.Tensor] = None,
        decoder_hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
    """
    Computes the Single-Step Control Lyapunov Function (CLF) loss.
    It samples candidate actions around the current action and penalizes the model if
    *no* candidate action satisfies the Lyapunov decrease condition: V(S_t+1) <= (1-rho)V(S_t).

    Args:
        model: The dynamics model.
        S_t, S_hist, A_t, A_hist: Current state and history batches.
        V_fn: Function to compute Lyapunov value.
        rho: Required convergence rate.
        num_delta_dirs, action_delta: Parameters for action sampling.
        softmin_tau: Temperature for softmin weighting of violations.
        source_embeddings, prefix_embeddings, decoder_hidden_state: Optional inputs.

    Returns:
        Tuple[torch.Tensor, float]: The calculated loss and the violation rate.
    """
    
    # Generate candidate actions
    A_cands = build_action_candidates(
        S_t=S_t, A_t=A_t,
        num_delta_dirs=num_delta_dirs,
        action_delta=action_delta
    )  # Shape: [K, B, dA]
    K = A_cands.shape[0]

    with torch.enable_grad():
        
        # Detach inputs to stop gradients flowing into data history
        S_t_detached = S_t.detach()
        S_hist_detached = S_hist.detach()
        A_hist_detached = A_hist.detach()
        hs_detached = decoder_hidden_state.detach() if decoder_hidden_state is not None else None
        
        V_t = V_fn(S_t_detached)  # [B]
        
        # Inertia assumption: Phi does not change instantly.
        _, Phi_t_detached, _ = split_state(S_t_detached)
        
        deltaV = []
        all_S_next_k = [] 
        for k in range(K):
            
            # Predict next state components
            model_output = model(
                S_t_detached, S_hist_detached,
                A_cands[k], A_hist_detached,
                decoder_hidden_state=hs_detached,
                source_embeddings=source_embeddings, 
                prefix_embeddings=prefix_embeddings
            )
            
            E_tp1_k, H_tp1_k, _ = model_output
            # Reconstruct full next state using Phi inertia
            S_tp1_k = torch.cat([E_tp1_k, Phi_t_detached, H_tp1_k], dim=1)

            all_S_next_k.append(S_tp1_k)

            # Calculate violation: V(S_t+1) - (1-rho)V(S_t)
            V_tp1_k = V_fn(S_tp1_k)
            deltaV_k = V_tp1_k - (1.0 - rho) * V_t  # [B]
            deltaV.append(deltaV_k)
        
        deltaV_stack = torch.stack(deltaV, dim=0)  # [K, B]
        all_S_next_k_stack = torch.stack(all_S_next_k, dim=0)

        # Use softmin to focus on the 'best' candidate action (minimum violation)
        softmin_weights = F.softmax(-deltaV_stack / softmin_tau, dim=0).t() # [B, K]
        softmin_vals = torch.einsum("bk,kb->b", softmin_weights, deltaV_stack) # [B]
        
        # Only penalize positive violations
        violations = F.relu(softmin_vals)
        loss = violations.mean()
        violation_rate = (violations > 0).float().mean().item()

        return loss, violation_rate

def cvar_loss(violations: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes the Conditional Value at Risk (CVaR) loss.
    This focuses the loss on the worst-case (tail) violations, improving robustness.

    Args:
        violations: Tensor of violation values (after ReLU).
        alpha: Quantile level (e.g., 0.1 for worst 10%).

    Returns:
        torch.Tensor: The scalar CVaR loss.
    """
    n = violations.numel()
    if n == 0:
        return violations.new_tensor(0.0)
    tail = max(int(math.ceil((1.0 - alpha) * n)), 1)
    
    # Filter for non-zero violations
    positive_violations = violations[violations > 1e-6]
    if positive_violations.numel() == 0:
        return violations.new_tensor(0.0)
    
    tail = min(tail, positive_violations.numel())
    
    # Average the top-k violations
    topk, _ = torch.topk(positive_violations, tail, largest=True, sorted=False)
    return topk.mean()

# --- S5: Cox Loss Function ---
@jit.script
def cox_ph_loss(log_risks: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative partial log-likelihood for the Cox Proportional Hazards model.
    Encourages the model to assign higher risk scores to samples that 'fail' earlier.

    Args:
        log_risks: Predicted log-risk scores.
        times: Time to event (or censoring).
        events: Binary indicator (1 if event occurred, 0 if censored).

    Returns:
        torch.Tensor: The scalar loss.
    """

    # Sort by time descending to efficiently calculate risk sets
    times, sort_indices = torch.sort(times.contiguous(), descending=True)
    events = events.contiguous()[sort_indices]
    log_risks = log_risks.contiguous()[sort_indices]

    # Calculate log-sum-exp of risks for the risk set at each time step
    risk_set_log_sum = torch.log(torch.cumsum(torch.exp(log_risks), dim=0))

    # Loss is calculated only for uncensored events
    loss = -torch.sum((log_risks - risk_set_log_sum)[events == 1])

    num_events = torch.sum(events)
    return loss / num_events if num_events > 0 else torch.tensor(0.0, device=log_risks.device)

# --- S5: CBF Soft-Constraint Loss ---
def control_barrier_loss(S_t: torch.Tensor, S_tp1: torch.Tensor, phi_crit: float, cbf_alpha: float) -> torch.Tensor:
    """
    Computes the Control Barrier Function (CBF) loss for safety constraints.
    Penalizes the system if it moves closer to or crosses a safety boundary defined by phi_crit.

    Args:
        S_t, S_tp1: Current and next state tensors.
        phi_crit: Critical threshold for resource pressure.
        cbf_alpha: CBF relaxation parameter.

    Returns:
        torch.Tensor: The scalar CBF loss.
    """
    _, Phi_t, _ = split_state(S_t)
    _, Phi_tp1, _ = split_state(S_tp1)

    # Barrier function h(S) = (phi_crit^2 - ||Phi||^2)
    h_t   = phi_crit**2 - torch.sum(Phi_t**2,   dim=1)
    h_tp1 = phi_crit**2 - torch.sum(Phi_tp1**2, dim=1)

    # Condition: h(S_next) >= (1 - alpha) * h(S_curr)
    violations = F.relu(-(h_tp1 - (1.0 - cbf_alpha) * h_t))
    return violations.mean()

def n_step_clf_loss(
    model: nn.Module,
    S_t: torch.Tensor, S_hist: torch.Tensor, 
    A_t: torch.Tensor, A_hist: torch.Tensor, 
    V_fn: Callable[[torch.Tensor], torch.Tensor],
    nstep_H: int, nstep_gamma: float, nstep_selector: str, nstep_bptt_window: int,
    use_cvar_loss: bool, cvar_alpha: float,
    use_epsilon_greedy: bool, epsilon: float, policy_entropy_weight: float,
    rho: float, num_delta_dirs: int, action_delta: float,
    gumbel_tau: float,
    lambda_adt: float,
    lambda_cbf: float, phi_crit: float, cbf_alpha: float,
    return_dv_trajectory: bool = False,
    source_embeddings: Optional[torch.Tensor] = None,
    prefix_embeddings: Optional[torch.Tensor] = None,
    decoder_hidden_state: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Computes the N-step Control Lyapunov Function loss via trajectory rollout.
    Simulates the system forward for 'nstep_H' steps, selecting optimal local actions,
    and penalizing cumulative stability violations. Incorporates CBF, ADT, and Entropy terms.

    Args:
        model: Dynamics model.
        S_t, S_hist, A_t, A_hist: Initial batch inputs.
        V_fn: Lyapunov function.
        nstep_H: Number of steps to unroll.
        nstep_gamma: Discount factor for future steps.
        nstep_selector: Strategy for action selection ('softmin', 'gumbel_st', 'hard_greedy').
        nstep_bptt_window: Truncated BPTT window size.
        ... (various loss weighting and config parameters) ...

    Returns:
        Tuple[torch.Tensor, dict]: Total loss and a dictionary of statistics.
    """
    B, _ = S_t.shape
    all_step_losses = []  
    action_indices_hist = []
    
    dv_trajectory = [] if return_dv_trajectory else None

    # Initialize simulation state
    S_current, S_hist_current = S_t, S_hist
    A_anchor, A_hist_current = A_t, A_hist

    history_S_bptt = [S_current]
    adt_switch_rates = []

    # Initialize previous indices for ADT (Action Dissimilarity) calculation
    with torch.no_grad():
        prev_indices = torch.zeros(B, dtype=torch.long, device=S_t.device)\

    # Rollout loop
    for i in range(nstep_H):
        
        # 1. Generate candidate actions around the current anchor
        cand_actions = build_action_candidates(
            S_t=S_current, A_t=A_anchor,
            num_delta_dirs=num_delta_dirs,
            action_delta=action_delta
        ) # Shape: [K, B, dA]
        K = cand_actions.shape[0]

        # 2. Calculate Lyapunov delta for all candidates
        with torch.no_grad():
            V_current_no_grad = V_fn(S_current)

        _, Phi_current, _ = split_state(S_current)

        all_S_next_k, all_dV_k = [], []
        for k in range(K):
            # Predict next state components
            model_output = model(
                S_current, S_hist_current, 
                cand_actions[k], A_hist_current,
                decoder_hidden_state=decoder_hidden_state,
                source_embeddings=source_embeddings, 
                prefix_embeddings=prefix_embeddings
            )
            
            E_next_k, H_next_k, _ = model_output
                
            # Reconstruct full state (Inertia assumption for Phi)
            S_next_k = torch.cat([E_next_k, Phi_current, H_next_k], dim=1)

            V_next_k = V_fn(S_next_k)
            dV_k = V_next_k - (1.0 - rho) * V_current_no_grad
            all_S_next_k.append(S_next_k)
            all_dV_k.append(dV_k)

        all_S_next_k_stack = torch.stack(all_S_next_k, dim=0) # [K, B, dS]
        all_dV_k_stack = torch.stack(all_dV_k, dim=0)       # [K, B]

        # 3. Select Action based on predicted stability
        if nstep_selector == 'softmin':
            weights = F.softmax(-all_dV_k_stack / gumbel_tau, dim=0).t()
            with torch.no_grad():
                best_k_indices_no_grad = torch.argmin(all_dV_k_stack, dim=0)
        elif nstep_selector == 'gumbel_st':
            y_soft = F.gumbel_softmax(logits=-all_dV_k_stack, tau=gumbel_tau, hard=False, dim=0).t()
            y_hard = F.gumbel_softmax(logits=-all_dV_k_stack, tau=gumbel_tau, hard=True, dim=0).t()
            weights = y_hard - y_soft.detach() + y_soft
            with torch.no_grad():
                best_k_indices_no_grad = torch.argmax(weights, dim=-1)
        else: # 'hard_greedy'
            best_k_indices = torch.argmin(all_dV_k_stack, dim=0)
            weights = F.one_hot(best_k_indices, num_classes=K).float()
            best_k_indices_no_grad = best_k_indices

        # Apply epsilon-greedy exploration if enabled
        if use_epsilon_greedy and model.training:
            random_indices = torch.randint(0, K, (B,), device=S_t.device)
            use_random = (torch.rand(B, device=S_t.device) < epsilon)
            final_indices = torch.where(use_random, random_indices, best_k_indices_no_grad)
            weights = F.one_hot(final_indices, num_classes=K).float()
        else:
            final_indices = best_k_indices_no_grad

        action_indices_hist.append(final_indices.detach().cpu().numpy())

        # 4. Compute expected next state and loss for this step
        best_dV_this_step = torch.einsum("bk,kb->b", weights, all_dV_k_stack)
        best_S_next = torch.einsum("bk,kbj->bj", weights, all_S_next_k_stack)

        if return_dv_trajectory and dv_trajectory is not None:
            dv_trajectory.append(best_dV_this_step.detach())

        # 5. Accumulate component losses
        violations = F.relu(best_dV_this_step)
        
        step_loss_components = []
        # Main CLF violation loss
        clf_violation_loss = cvar_loss(violations, cvar_alpha) if use_cvar_loss else violations.mean()
        step_loss_components.append(clf_violation_loss)

        # Control Barrier Function (CBF) loss for safety
        if lambda_cbf > 0 and phi_crit > 0:
            cbf_l = control_barrier_loss(S_current, best_S_next, phi_crit, cbf_alpha)
            step_loss_components.append(lambda_cbf * cbf_l)

        # Action Dissimilarity Term (ADT) to prevent chattering
        if lambda_adt > 0 or not model.training:
            switch_indicators = (final_indices != prev_indices).float()
            adt_l = switch_indicators.mean()
            if not model.training:
                adt_switch_rates.append(adt_l.item())
            if lambda_adt > 0:
                step_loss_components.append(lambda_adt * adt_l)
        
        step_loss = torch.sum(torch.stack(step_loss_components))
        all_step_losses.append((nstep_gamma ** i) * step_loss)

        prev_indices = final_indices.detach()

        # 6. Update state for next step
        A_current = cand_actions[final_indices, torch.arange(B)]
        
        if S_hist_current.shape[1] > 0:
            # Update history windows
            next_S_hist = torch.cat([S_hist_current[:, 1:, :], S_current.unsqueeze(1)], dim=1)
            next_A_hist = torch.cat([A_hist_current[:, 1:, :], A_current.unsqueeze(1)], dim=1)
            S_hist_current = next_S_hist.detach()
            A_hist_current = next_A_hist.detach()
        S_current = best_S_next
        
        # Handle Truncated BPTT
        history_S_bptt.append(S_current)
        if nstep_bptt_window > 0 and len(history_S_bptt) > nstep_bptt_window:
            history_S_bptt[0] = history_S_bptt[0].detach()
            history_S_bptt.pop(0)

        A_anchor = A_current
    
    total_loss = torch.sum(torch.stack(all_step_losses))

    # 7. Policy Entropy Regularization (encourage diversity)
    if policy_entropy_weight > 0 and len(action_indices_hist) > 0:
        all_actions = np.concatenate(action_indices_hist)
        counts = np.bincount(all_actions, minlength=K)
        if counts.sum() > 0:
            probs = counts / counts.sum()
            probs = torch.from_numpy(probs).float().to(S_t.device)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9))
            total_loss = total_loss - (policy_entropy_weight * entropy) 

    # Collect diagnostics
    stats = {}
    if action_indices_hist:
        stats = {
            f"clf_violation_rate_step{i}": (all_dV_k_stack[action_indices_hist[i], torch.arange(B)] > 0).float().mean().item()
            for i in range(nstep_H)
        }
        stats["clf_violation_rate_mean"] = np.mean(list(stats.values()))

    if adt_switch_rates:
        stats["adt_switch_rate"] = np.mean(adt_switch_rates)

    if return_dv_trajectory:
        stats["dv_trajectory"] = torch.stack(dv_trajectory, dim=1) # [B, H]

    return total_loss, stats

def lyapunov_positive_rate(
    model, 
    S_t: torch.Tensor, S_hist: torch.Tensor, 
    A_t: torch.Tensor, A_hist: torch.Tensor,
    V_fn: Callable[[torch.Tensor], torch.Tensor],
    source_embeddings: Optional[torch.Tensor] = None,
    prefix_embeddings: Optional[torch.Tensor] = None,
    decoder_hidden_state: Optional[torch.Tensor] = None
) -> float:
    """
    Computes the rate at which the Lyapunov function increases (positive drift)
    for the single-step teacher action. Used as a validation metric.

    Args:
        model: Dynamics model.
        S_t, S_hist, A_t, A_hist: Inputs.
        V_fn: Lyapunov function.
        ... (optional embeddings) ...

    Returns:
        float: Fraction of samples where V(S_t+1) > V(S_t).
    """
    with torch.no_grad():
        V_t = V_fn(S_t)
        
        # Get model output
        model_output = model(
            S_t, S_hist, A_t, A_hist,
            decoder_hidden_state=decoder_hidden_state,
            source_embeddings=source_embeddings, 
            prefix_embeddings=prefix_embeddings
        )
        
        # Reconstruct next state
        E_pred, H_pred, _ = model_output
        _, Phi_t, _ = split_state(S_t)
        S_tp1 = torch.cat([E_pred, Phi_t, H_pred], dim=1)
        
        # Check drift
        V_tp1 = V_fn(S_tp1)
        dv = V_tp1 - V_t
        return (dv > 0).float().mean().item()
    
def multi_step_negative_drift_coverage(
    model, 
    S_t: torch.Tensor, S_hist: torch.Tensor, 
    A_t: torch.Tensor, A_hist: torch.Tensor,
    V_fn: Callable[[torch.Tensor], torch.Tensor],
    rho: float, rollout_H: int = 5, action_delta: float = 0.1, num_delta_dirs: int = 2,
    source_embeddings: Optional[torch.Tensor] = None,
    prefix_embeddings: Optional[torch.Tensor] = None,
    decoder_hidden_state: Optional[torch.Tensor] = None
) -> float:
    """
    Computes the coverage of negative drift (stability) over a multi-step rollout
    when the optimal local action is chosen at each step. This evaluates the
    feasibility of stabilization.

    Args:
        model: Dynamics model.
        ... (inputs and config) ...
        rollout_H: Horizon length.

    Returns:
        float: Percentage of steps in the rollout where stability was achieved.
    """
    B, _ = A_t.shape
    S_curr = S_t.clone()
    S_hist_curr = S_hist.clone()
    A_hist_curr = A_hist.clone()
    
    totals, neg_counts = [], []

    for _ in range(rollout_H):
        # Generate candidates around teacher action
        A_cands = build_action_candidates(
            S_t=S_curr, A_t=A_t,
            num_delta_dirs=num_delta_dirs,
            action_delta=action_delta
        )
        K = A_cands.shape[0]

        with torch.no_grad():
            V_now = V_fn(S_curr)
            _, Phi_curr, _ = split_state(S_curr)
            
            all_dV_k = []
            all_S_next_k = []
            for k in range(K):
                model_output = model(
                    S_curr, S_hist_curr, 
                    A_cands[k], A_hist_curr,
                    decoder_hidden_state=decoder_hidden_state,
                    source_embeddings=source_embeddings, 
                    prefix_embeddings=prefix_embeddings
                )

                E_next_k, H_next_k, _ = model_output
                S_next_k = torch.cat([E_next_k, Phi_curr, H_next_k], dim=1)
                
                V_next_k = V_fn(S_next_k)
                dv_k = V_next_k - (1.0 - rho) * V_now
                all_dV_k.append(dv_k)
                all_S_next_k.append(S_next_k)
            
            all_dV_k_stack = torch.stack(all_dV_k, dim=0)
            all_S_next_k_stack = torch.stack(all_S_next_k, dim=0)

            # Choose best action
            best_k_indices = torch.argmin(all_dV_k_stack, dim=0)
            
            best_dv = all_dV_k_stack[best_k_indices, torch.arange(B)]
            best_S_next = all_S_next_k_stack[best_k_indices, torch.arange(B)]
            best_A_curr = A_cands[best_k_indices, torch.arange(B)]

            neg_counts.append((best_dv <= 0).float().sum().item())
            totals.append(B)
            
            # Update state/history
            S_curr = best_S_next
            
            if S_hist_curr.shape[1] > 0:
                next_S_hist = torch.cat([S_hist_curr[:, 1:, :], S_curr.unsqueeze(1)], dim=1)
                next_A_hist = torch.cat([A_hist_curr[:, 1:, :], best_A_curr.unsqueeze(1)], dim=1)
            else:
                next_S_hist = S_hist_curr
                next_A_hist = A_hist_curr

            S_hist_curr = next_S_hist
            A_hist_curr = next_A_hist
    
    if not totals:
        return float("nan")
    return float(sum(neg_counts) / sum(totals))

def multi_step_negative_drift_coverage_teacher(
    model, 
    S_t: torch.Tensor, S_hist: torch.Tensor, 
    A_t: torch.Tensor, A_hist: torch.Tensor,
    V_fn: Callable[[torch.Tensor], torch.Tensor], 
    rho: float,
    rollout_H: int = 5,
    source_embeddings: Optional[torch.Tensor] = None,
    prefix_embeddings: Optional[torch.Tensor] = None,
    decoder_hidden_state: Optional[torch.Tensor] = None
) -> float:
    """
    Computes rollout stability coverage using ONLY the fixed teacher action A_t
    repeatedly. Serves as a baseline comparison.
    """
    with torch.no_grad():
        S_curr = S_t.clone()
        S_hist_curr = S_hist.clone()
        A_hist_curr = A_hist.clone()
        totals = []
        neg_counts = []
        
        for _ in range(rollout_H):
            V_now = V_fn(S_curr)
            _, Phi_curr, _ = split_state(S_curr)

            model_output = model(
                S_curr, S_hist_curr, 
                A_t, A_hist_curr,
                decoder_hidden_state=decoder_hidden_state,
                source_embeddings=source_embeddings, 
                prefix_embeddings=prefix_embeddings
            )
            
            E_next, H_next, _ = model_output
            S_next = torch.cat([E_next, Phi_curr, H_next], dim=1)

            V_next = V_fn(S_next)
            dv = V_next - (1.0 - rho) * V_now
            neg_counts.append((dv <= 0).float().sum().item())
            totals.append(S_curr.shape[0])
            
            S_curr = S_next
            
            if S_hist_curr.shape[1] > 0:
                S_hist_curr = torch.cat([S_hist_curr[:, 1:, :], S_curr.unsqueeze(1)], dim=1)
                A_hist_curr = torch.cat([A_hist_curr[:, 1:, :], A_t.unsqueeze(1)], dim=1)

        if not totals:
            return float("nan")
        return float(sum(neg_counts) / sum(totals))

def multi_step_negative_drift_coverage_fixed(
    model, 
    S_t: torch.Tensor, S_hist: torch.Tensor, A_hist: torch.Tensor,
    V_fn: Callable[[torch.Tensor], torch.Tensor], 
    rho: float, rollout_H: int = 5,
    fixed_idx: int = 0, action_dim: int = 6,
    source_embeddings: Optional[torch.Tensor] = None,
    prefix_embeddings: Optional[torch.Tensor] = None,
    decoder_hidden_state: Optional[torch.Tensor] = None
) -> float:
    """
    Computes rollout stability coverage using a fixed constant action (e.g., 'none').
    """
    with torch.no_grad():
        B = S_t.shape[0]
        device = S_t.device
        A_fixed = torch.zeros(B, action_dim, device=device)
        A_fixed[:, int(fixed_idx)] = 1.0

        S_curr = S_t.clone()
        S_hist_curr = S_hist.clone()
        A_hist_curr = A_hist.clone()
        neg_counts = []
        
        for _ in range(rollout_H):
            V_now = V_fn(S_curr)
            _, Phi_curr, _ = split_state(S_curr)

            model_output = model(
                S_curr, S_hist_curr, 
                A_fixed, A_hist_curr,
                decoder_hidden_state=decoder_hidden_state,
                source_embeddings=source_embeddings, 
                prefix_embeddings=prefix_embeddings
            )
            
            E_next, H_next, _ = model_output
            S_next = torch.cat([E_next, Phi_curr, H_next], dim=1)

            V_next = V_fn(S_next)
            dv = V_next - (1.0 - rho) * V_now
            neg_counts.append((dv <= 0).float().sum().item())
            
            S_curr = S_next

            if S_hist_curr.shape[1] > 0:
                S_hist_curr = torch.cat([S_hist_curr[:, 1:, :], S_curr.unsqueeze(1)], dim=1)
                A_hist_curr = torch.cat([A_hist_curr[:, 1:, :], A_fixed.unsqueeze(1)], dim=1)
            
        total = B * rollout_H
        if total == 0:
            return float("nan")
        return float(sum(neg_counts) / total)

# -----------------------------
# Dataset
# -----------------------------

def _build_pairs_from_df(
    df: pd.DataFrame,
    history_len: int = 0,
    action_dim: int=6,
    event_threshold: float = 2.0,
    use_decoder_hidden_state: bool = False,
    predict_delta: bool = False
):
    """
    Processes the raw DataFrame into structured numpy arrays for training.
    Handles sequence alignment, history generation, and action vector encoding.

    Args:
        df: Input DataFrame containing trajectory data.
        history_len: Number of past steps to include in history.
        action_dim: Dimension of action vector.
        event_threshold: Threshold for Cox event generation.
        use_decoder_hidden_state: Flag to parse hidden states.
        predict_delta: Flag to calculate delta states.

    Returns:
        Tuple of numpy arrays (S_t, S_hist, A, A_hist, S_tp1, ...) suitable for dataset creation.
    """
    action_cols = ['action_index_type', 'action_k', 'action_lambda']
    group_cols = ['Strategy', 'sample_id', 'beam_id', 'step']
    
    required_cols = set(STATE_COLS_DEFAULT + action_cols + group_cols)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.sort_values(group_cols)
    
    if 'error_faithfulness' in df.columns:
        df['error_faithfulness'] = df['error_faithfulness'].clip(0.0, 1.0)

    group_keys = df[['Strategy', 'sample_id', 'beam_id']].apply(tuple, axis=1)
    
    if 'error_norm' not in df.columns:
        raise ValueError("Key Error: 'error_norm' column is missing from the input CSV file.")

    # Create next state columns
    for c in STATE_COLS_DEFAULT + ['error_norm']:
        df[f'next_{c}'] = df.groupby(group_keys)[c].shift(-1)
    
    # Construct Action Matrix
    A_full = np.zeros((len(df), action_dim), dtype=np.float32)
    index_map = {'none': 0, 'exact': 1, 'hnsw': 2, 'ivf_pq': 3}
    type_indices = df['action_index_type'].map(index_map).values
    A_full[np.arange(len(df)), type_indices.astype(int)] = 1.0
    A_full[:, 4] = (df['action_k'].astype(np.float32).to_numpy() / float(KNN_MAX_K)).astype(np.float32)
    A_full[:, 5] = df['action_lambda'].astype(np.float32).to_numpy()
    
    action_col_names = [f'A_dim_{i}' for i in range(action_dim)]
    for i, col in enumerate(action_col_names):
        df[col] = A_full[:, i]

    history_cols_flat = []
    history_action_cols_flat = []
    
    # Create History Columns via Shifting
    if history_len > 0:
        for k in range(1, history_len + 1):
            for col_name in STATE_COLS_DEFAULT:
                new_col_name = f'S_t_minus_{k}_{col_name}'
                df[new_col_name] = df.groupby(group_keys)[col_name].shift(k)
                if k <= history_len:
                    history_cols_flat.append(new_col_name)
            
            for col_name in action_col_names:
                new_action_col_name = f'A_t_minus_{k}_{col_name}'
                df[new_action_col_name] = df.groupby(group_keys)[col_name].shift(k)
                if k <= history_len:
                    history_action_cols_flat.append(new_action_col_name)

    # Filter invalid rows (due to shifting)
    cols_to_check_na = [f'next_{c}' for c in STATE_COLS_DEFAULT]
    if history_len > 0:
        cols_to_check_na.extend(history_cols_flat)
        cols_to_check_na.extend(history_action_cols_flat)
             
    valid = ~pd.isna(df[cols_to_check_na]).any(axis=1)
    df = df.loc[valid].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"No valid data points remained after creating history. "
            f"Max shift required: {history_len}. "
            f"Try reducing history_len or checking sequence lengths."
        )
    
    group_keys_valid = df[['Strategy', 'sample_id', 'beam_id']].apply(tuple, axis=1)
    seq_id, _ = pd.factorize(group_keys_valid)

    # Reshape history into 3D tensors
    S_hist = np.zeros((len(df), history_len, len(STATE_COLS_DEFAULT)), dtype=np.float32)
    if history_len > 0:
        S_hist = df[history_cols_flat].values.reshape(len(df), history_len, len(STATE_COLS_DEFAULT))

    A_hist = np.zeros((len(df), history_len, action_dim), dtype=np.float32)
    if history_len > 0:
        A_hist = df[history_action_cols_flat].values.reshape(len(df), history_len, action_dim)
    
    S_t_base = df[STATE_COLS_DEFAULT].astype(np.float32).values
    S_tp1_base = df[[f'next_{c}' for c in STATE_COLS_DEFAULT]].astype(np.float32).values
    
    A = df[action_col_names].astype(np.float32).values

    # Determine prediction target (Delta vs Absolute)
    if predict_delta:
        final_S_tp1 = S_tp1_base - S_t_base
        print("[Info] Target `S_tp1` is set to be the state delta (S_tp1 - S_t).")
    else:
        final_S_tp1 = S_tp1_base
        print("[Info] Target `S_tp1` is set to be the absolute next state (S_tp1).")

    # Cox Event Generation
    df['event_occurred'] = df['error_norm'] > event_threshold
    df['group_id'] = seq_id
    first_event_times = df[df['event_occurred']].groupby('group_id')['step'].min()
    df = df.merge(first_event_times.rename('first_event_step'), on='group_id', how='left')
    max_steps = df.groupby('group_id')['step'].max()
    df = df.merge(max_steps.rename('max_step'), on='group_id', how='left')

    df['T'] = df['first_event_step'].fillna(df['max_step'])
    df['E'] = (~df['first_event_step'].isna()).astype(np.int64)
    T = df['T'].astype(np.float32).values
    E = df['E'].astype(np.int64).values
    
    source_texts = df['source_text'].astype(str).tolist()
    generated_prefixes = df['generated_prefix'].fillna('').astype(str).tolist()

    # Parse Hidden States (if available)
    decoder_hidden_states = None
    if use_decoder_hidden_state:
        if 'decoder_hidden_state' not in df.columns:
            raise ValueError("Column 'decoder_hidden_state' is required but not found in the CSV. "
                             "Please re-run the data generation script from the paec project.")
        print("[Info] Parsing 'decoder_hidden_state' column...")
        def robust_json_loads(x):
            try:
                return json.loads(x)
            except (json.JSONDecodeError, TypeError):
                return None

        hidden_states_list = df['decoder_hidden_state'].apply(robust_json_loads).tolist()

        first_valid_hs = next((hs for hs in hidden_states_list if hs is not None), None)
        if first_valid_hs is None:
            raise ValueError("Could not parse any valid 'decoder_hidden_state' entries.")

        hs_dim = len(first_valid_hs)
        filled_hidden_states = [hs if hs is not None else [0.0] * hs_dim for hs in hidden_states_list]
        decoder_hidden_states = np.array(filled_hidden_states, dtype=np.float32)

    return S_t_base, S_hist, A, A_hist, final_S_tp1, seq_id.astype(np.int64), T, E, source_texts, generated_prefixes, decoder_hidden_states

class PAECDataset(Dataset):
    """
    PyTorch Dataset for the PAEC model. Wraps pre-processed numpy arrays.
    """
    def __init__(
        self,
        S_t: np.ndarray, S_hist: np.ndarray, A: np.ndarray, A_hist: np.ndarray, S_tp1: np.ndarray,
        T: Optional[np.ndarray], E: Optional[np.ndarray],
        source_embeddings: Optional[np.ndarray], 
        prefix_embeddings: Optional[np.ndarray],
        decoder_hidden_states: Optional[np.ndarray],
        seq_id: Optional[np.ndarray] = None,
        source_texts: Optional[List[str]] = None,
        generated_prefixes: Optional[List[str]] = None,
    ):
        # All incoming arrays are assumed to be pre-processed, scaled, and padded.
        self.S_t    = torch.from_numpy(S_t.astype(np.float32))
        self.S_hist = torch.from_numpy(S_hist.astype(np.float32))
        self.A_t    = torch.from_numpy(A.astype(np.float32))
        self.A_hist = torch.from_numpy(A_hist.astype(np.float32))
        self.S_tp1  = torch.from_numpy(S_tp1.astype(np.float32))
        
        self.seq_id = torch.from_numpy(seq_id) if seq_id is not None else None
        self.T = torch.from_numpy(T) if T is not None else None
        self.E = torch.from_numpy(E) if E is not None else None
        
        self.source_embeddings = torch.from_numpy(source_embeddings.astype(np.float32)) if source_embeddings is not None else None
        self.prefix_embeddings = torch.from_numpy(prefix_embeddings.astype(np.float32)) if prefix_embeddings is not None else None
        self.decoder_hidden_states = torch.from_numpy(decoder_hidden_states.astype(np.float32)) if decoder_hidden_states is not None else None
        self.source_texts = source_texts
        self.generated_prefixes = generated_prefixes

    def __len__(self): 
        return self.S_t.shape[0]
    
    def __getitem__(self, i: int) -> Dict[str, Any]:
        item : Dict[str, Any] = {
            "S_t": self.S_t[i], 
            "S_hist": self.S_hist[i],
            "A_t": self.A_t[i],
            "A_hist": self.A_hist[i],
            "S_tp1": self.S_tp1[i]
        }
        if self.source_embeddings is not None:
            item["source_embeddings"] = self.source_embeddings[i]
        if self.prefix_embeddings is not None:
            item["prefix_embeddings"] = self.prefix_embeddings[i]
        if self.seq_id is not None: 
            item["seq_id"] = self.seq_id[i]
        if self.T is not None: 
            item["T"] = self.T[i]
        if self.E is not None: 
            item["E"] = self.E[i]
        if self.decoder_hidden_states is not None:
            item["decoder_hidden_state"] = self.decoder_hidden_states[i]
        if self.source_texts is not None:
            item["source_text"] = self.source_texts[i]
        if self.generated_prefixes is not None:
            item["generated_prefix"] = self.generated_prefixes[i]
        return item
 
# --- Saving Tools ---
def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _save_scaler_joblib(scaler: Pipeline, path: Path):
    """
    Saves the entire scikit-learn pipeline object using joblib.
    This preserves the complete state of the transformer chain.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)

# -----------------------------
# Model Construction: PAEC Transfer Dynamics
# -----------------------------

class NodeMLP(nn.Module):
    """
    A configurable Multi-Layer Perceptron (MLP) block used as a building block for
    various components of the PAEC model (embeddings, projection heads, etc.).

    Key Feature:
    - Supports Spectral Normalization (SN) on linear layers to constrain the
      Lipschitz constant of the network, which is crucial for the theoretical
      stability guarantees of the PAEC framework.
    """
    def __init__(self, in_dim, hid_dim, out_dim, num_layers=2, act=nn.SiLU, use_spectral_norm=False):
        """
        Initializes the NodeMLP.

        Args:
            in_dim (int): Input dimension.
            hid_dim (int): Hidden layer dimension.
            out_dim (int): Output dimension.
            num_layers (int): Total number of linear layers (default: 2).
            act (nn.Module): Activation function class (default: nn.SiLU).
            use_spectral_norm (bool): If True, applies spectral normalization to weights.
        """
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            linear_layer = nn.Linear(d, hid_dim)
            # S5 (Lipschitz): Apply Lipschitz spectral normalization to enforce 1-Lipschitz continuity per layer
            if use_spectral_norm:
                parametrizations.spectral_norm(linear_layer, name='weight')
            layers.append(linear_layer)
            layers.append(act())
            d = hid_dim

        output_layer = nn.Linear(d, out_dim)
        # S5 (Lipschitz): Apply Lipschitz spectral normalization to the output layer as well
        if use_spectral_norm:
            parametrizations.spectral_norm(output_layer, name='weight')
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).contiguous()

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of tokens in the sequence.
    Essential for Transformer models since they possess no inherent notion of order.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]

        Returns:
            Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # The stored `pe` has shape [max_len, 1, embedding_dim].
        # We need to transpose x to [seq_len, batch_size, embedding_dim]
        # to broadcast the addition with pe[:x.size(1)].

        # Note that the PAECTransition model uses batch_first=True for the Transformer,
        # so the input x to this function is [batch_size, seq_len, dim].
        # However, the standard PositionalEncoding implementation expects [seq_len, batch_size, dim].
        # We will follow the standard implementation's permutation logic for correctness.

        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        if hasattr(self, 'pe'):
            pe = getattr(self, 'pe')
            x = x + pe[:x.size(0)]
        x = self.dropout(x)
        return x.permute(1, 0, 2)  # [batch_size, seq_len, embedding_dim]

class PAECTransition(nn.Module):
    """
    The core Dynamics Model (T_theta) for PAEC.

    Architecture:
    - Sequence-based: Processes a history of system states (S) and actions (A) using a Transformer Encoder.
    - Feature Fusion: Combines the encoded history context with the current proposed action (A_t)
      and optional embeddings (text, decoder hidden state).
    - Prediction Heads: Outputs the predicted next state components (Error 'E' and Context 'H')
      and an auxiliary risk score (Cox).

    This model serves as the "Physics Engine" for the control system, predicting how the
    translation state evolves under specific actions.
    """
    def __init__(
        self,
        action_dim: int,
        hid_dim: int,
        layers: int,
        history_len: int,
        predict_delta: bool,
        use_text_embeddings: bool,
        text_embedding_dim: int,
        use_decoder_hidden_state: bool,
        decoder_hidden_state_dim: int,
        use_moe_heads: bool,
        use_multi_heads: bool,
        use_spectral_norm: bool,
        nhead: int
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hid_dim = hid_dim
        self.history_len = history_len
        self.predict_delta = predict_delta
        self.use_text_embeddings = use_text_embeddings
        self.text_embedding_dim = text_embedding_dim
        self.use_decoder_hidden_state = use_decoder_hidden_state
        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.use_moe_heads = use_moe_heads
        self.use_multi_heads = use_multi_heads

        # 1. Input Embedding Layers
        # The history sequence S_hist consists of the base dimensions (S_DIM) plus the action dimension.
        # Only S_t is augmented with extra embeddings later in the pipeline.
        history_item_dim = S_DIM + self.action_dim
        self.history_seq_embed = NodeMLP(history_item_dim, hid_dim, hid_dim, num_layers=2, use_spectral_norm=use_spectral_norm)

        # S_t is the current state vector dimension.
        self.state_embed_t = NodeMLP(S_DIM, hid_dim, hid_dim, num_layers=2, use_spectral_norm=use_spectral_norm)

        # 2. Positional Encoding for the sequence (History + Current Step)
        self.pos_encoder = PositionalEncoding(hid_dim, max_len=history_len + 2)

        # 3. Transformer Encoder
        # Captures temporal dependencies in the state-action trajectory.
        encoder_layer = TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=nhead,
            dim_feedforward=hid_dim * 4,
            dropout=0.1,
            activation=F.silu,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=layers)

        # 4. Action and Feature Projections
        self.action_proj = nn.Linear(action_dim, hid_dim)

        # State-Action Fusion: Can use Cross-Attention (Multi-head) or Concatenation.
        if self.use_multi_heads:
            self.state_action_attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=nhead, batch_first=True)
        self.fusion_layer_norm = nn.LayerNorm(hid_dim)

        # Optional: Project text embeddings (source/prefix) into hidden space
        if self.use_text_embeddings:
            self.text_embed_proj = NodeMLP(
                in_dim=self.text_embedding_dim * 2, # source_emb + prefix_emb
                hid_dim=hid_dim,
                out_dim=hid_dim,
                use_spectral_norm=use_spectral_norm
            )
        # Optional: Project NMT decoder hidden state into hidden space
        if self.use_decoder_hidden_state:
            self.hidden_state_proj = NodeMLP(
                in_dim=self.decoder_hidden_state_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                use_spectral_norm=use_spectral_norm
            )

        # 5. Output Heads
        # Determine the input dimension for the prediction heads based on enabled features.
        out_head_in_dim = hid_dim
        if not self.use_multi_heads: out_head_in_dim += hid_dim
        if self.use_text_embeddings: out_head_in_dim += hid_dim
        if self.use_decoder_hidden_state: out_head_in_dim += hid_dim

        # Support for Mixture-of-Experts (MoE) heads or a shared head.
        if self.use_moe_heads:
            # Separate heads for Error (E) and Context (H) components
            self.out_head_E = NodeMLP(out_head_in_dim, hid_dim, E_INDEX[1] - E_INDEX[0] + 1, use_spectral_norm=use_spectral_norm)
            self.out_head_H = NodeMLP(out_head_in_dim, hid_dim, H_INDEX[1] - H_INDEX[0] + 1, use_spectral_norm=use_spectral_norm)
        else:
            # Single shared head for all state components (E + H)
            output_dim = E_DIM + H_DIM
            self.out_head = NodeMLP(out_head_in_dim, hid_dim, output_dim, use_spectral_norm=use_spectral_norm)

        # 6. Cox Risk Prediction Head (Auxiliary task for survival analysis)
        self.risk_head = NodeMLP(hid_dim, hid_dim // 2, 1, num_layers=2, use_spectral_norm=use_spectral_norm)

        # Apply spectral normalization to all linear layers if configured (for Lipschitz bounds)
        if use_spectral_norm: self.apply(self._apply_spectral_norm)

    def _apply_spectral_norm(self, module):
        """
        Recursively applies spectral normalization to all nn.Linear weights in the module.
        Ensures Lipschitz continuity for the Transformer components.
        """
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight'): parametrizations.spectral_norm(module, name='weight')
        if isinstance(module, nn.MultiheadAttention):
            # Handle fused vs separate projection weights in MHA
            if hasattr(module, 'in_proj_weight'):
                parametrizations.spectral_norm(module, name='in_proj_weight')
            else:
                if hasattr(module, 'q_proj_weight'): parametrizations.spectral_norm(module, name='q_proj_weight')
                if hasattr(module, 'k_proj_weight'): parametrizations.spectral_norm(module, name='k_proj_weight')
                if hasattr(module, 'v_proj_weight'): parametrizations.spectral_norm(module, name='v_proj_weight')
        for name, child in module.named_children():
            self._apply_spectral_norm(child)

    def forward_heavy_cache(
        self,
        S_t: torch.Tensor, S_hist: torch.Tensor, A_hist: torch.Tensor,
        decoder_hidden_state: Optional[torch.Tensor] = None,
        source_embeddings: Optional[torch.Tensor] = None,
        prefix_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Performs the computationally expensive part of the forward pass (History Encoding + Context Embedding).
        Returns a cache of tensors that do NOT depend on the current action A_t.
        This allows efficient evaluation of multiple candidate actions for the same state.
        """
        if self.history_len > 0:
            if S_hist.shape[1] != self.history_len:
                 raise ValueError(f"Model configured with history_len={self.history_len}, but received S_hist with shape {S_hist.shape}")
            if A_hist.shape[1] != self.history_len:
                 raise ValueError(f"Model configured with history_len={self.history_len}, but received A_hist with shape {A_hist.shape}")

        if self.use_text_embeddings and (source_embeddings is None or prefix_embeddings is None):
            raise ValueError("Model is configured to use text embeddings, but they were not provided.")
        if self.use_decoder_hidden_state and decoder_hidden_state is None:
            raise ValueError("Model is configured to use the decoder hidden state, but it was not provided.")

        # 1. Combine and embed history sequence (States + Actions)
        history_seq = torch.cat([S_hist[:, :, :S_DIM], A_hist], dim=-1)
        hist_embed = self.history_seq_embed(history_seq)

        # 2. Embed current state S_t
        t_embed = self.state_embed_t(S_t).unsqueeze(1)

        # 3. Concatenate and apply Positional Encoding
        seq_embed = torch.cat([hist_embed, t_embed], dim=1)
        seq_embed_pos = self.pos_encoder(seq_embed)

        # 4. Process with Transformer Encoder to get the context-aware hidden state
        transformer_out = self.transformer_encoder(seq_embed_pos)
        final_hidden_state = transformer_out[:, -1, :] # [B, hid_dim]

        # 5. Process static embeddings (Text, Decoder State)
        text_emb, hs_emb = None, None
        if self.use_text_embeddings and source_embeddings is not None and prefix_embeddings is not None:
            combined_text_emb = torch.cat([source_embeddings, prefix_embeddings], dim=1)
            text_emb = self.text_embed_proj(combined_text_emb) # [B, hid_dim]

        if self.use_decoder_hidden_state and decoder_hidden_state is not None:
            if decoder_hidden_state.dim() == 3 and decoder_hidden_state.shape[1] == 1:
                decoder_hidden_state = decoder_hidden_state.squeeze(1)
            hs_emb = self.hidden_state_proj(decoder_hidden_state) # [B, hid_dim]

        # 6. Extract Phi_t for Inertia Assumption (Pressure state is assumed constant for next step prediction)
        _, Phi_t, _ = split_state(S_t)

        # Cache baseline E_t and H_t if the model is predicting deltas (changes)
        E_t, H_t = None, None
        if self.predict_delta:
            E_t = S_t[:, E_INDEX[0]:E_INDEX[1]+1]
            H_t = S_t[:, H_INDEX[0]:H_INDEX[1]+1]

        return {
            "final_hidden_state": final_hidden_state,
            "text_emb": text_emb,
            "hs_emb": hs_emb,
            "Phi_t": Phi_t,
            "E_t": E_t,
            "H_t": H_t
        }

    def forward_light_head(
        self,
        A_t: torch.Tensor,
        cache: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the lightweight part of the forward pass: Action Embedding, Feature Fusion,
        and Prediction Heads. Designed to be called repeatedly with different actions A_t
        using the pre-computed cache.

        Returns:
            Tuple: (E_pred, H_pred, log_risk)
        """
        # 1. Retrieve cached tensors
        final_hidden_state = cache["final_hidden_state"]
        text_emb = cache["text_emb"]
        hs_emb = cache["hs_emb"]

        # 2. Embed the Action
        a_emb = torch.tanh(self.action_proj(A_t))

        # 3. Fuse State and Action
        if self.use_multi_heads:
            state_q = final_hidden_state.unsqueeze(1)
            action_kv = a_emb.unsqueeze(1)
            
            # Cross-Attention: State attends to Action
            action_aware_embedding, _ = self.state_action_attention(
                query=state_q, 
                key=action_kv, 
                value=action_kv
            )
            fused_embedding = self.fusion_layer_norm(final_hidden_state + action_aware_embedding.squeeze(1))
            prediction_inputs = [fused_embedding]
        else:
            # Simple Concatenation
            prediction_inputs = [final_hidden_state, a_emb]

        # Add auxiliary features if available
        if self.use_text_embeddings and text_emb is not None:
            prediction_inputs.append(text_emb)

        if self.use_decoder_hidden_state and hs_emb is not None:
            prediction_inputs.append(hs_emb)

        combined_features = torch.cat(prediction_inputs, dim=1)

        # 4. Predict Cox Risk (based on state context only, logic simplified here to use shared hidden state)
        log_risk = self.risk_head(final_hidden_state).view(-1)

        # 5. Predict Next State Components (E and H)
        if self.use_moe_heads:
            out_E = self.out_head_E(combined_features)
            out_H = self.out_head_H(combined_features)
        else:
            out_EH = self.out_head(combined_features)
            out_E, out_H = out_EH[:, :E_INDEX[1] + 1], out_EH[:, E_INDEX[1] + 1:E_INDEX[1] + H_DIM + 1]

        # Handle Delta Prediction logic
        if self.predict_delta:
            E_t = cache.get("E_t")
            H_t = cache.get("H_t")
            # If E_t/H_t weren't cached (e.g., due to different cache creation path), reconstruct them
            if E_t is None or H_t is None:
                 # Note: This fallback requires S_t to be in the cache, which is not standard in 'forward_heavy_cache'
                 # Assuming cache has been augmented if this path is taken.
                 E_t, _, H_t = split_state(cache["S_t_for_delta"])

            final_E = E_t + out_E
            final_H = H_t + out_H
        else:
            final_E = out_E
            final_H = out_H

        return final_E, final_H, log_risk
    
    def forward(
        self,
        S_t: torch.Tensor, S_hist: torch.Tensor,
        A_t: torch.Tensor, A_hist: torch.Tensor,
        decoder_hidden_state: Optional[torch.Tensor] = None,
        source_embeddings: Optional[torch.Tensor] = None,
        prefix_embeddings: Optional[torch.Tensor] = None
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Standard forward pass used during training.
        Combines heavy encoding and light prediction in one go.

        Returns:
            E_pred [B, dim_E], H_pred [B, dim_H], log_risk [B]
        """
        # ==============================================================================
        # STAGE 1: FEATURE EXTRACTION (Encoder)
        # ==============================================================================
        if self.history_len > 0:
            if S_hist.shape[1] != self.history_len:
                 raise ValueError(f"Model configured with history_len={self.history_len}, but received S_hist with shape {S_hist.shape}")
            if A_hist.shape[1] != self.history_len:
                 raise ValueError(f"Model configured with history_len={self.history_len}, but received A_hist with shape {A_hist.shape}")

        if self.use_text_embeddings and (source_embeddings is None or prefix_embeddings is None):
            raise ValueError("Model is configured to use text embeddings, but they were not provided.")
        if self.use_decoder_hidden_state and decoder_hidden_state is None:
            raise ValueError("Model is configured to use the decoder hidden state, but it was not provided.")

        # 1. Embed History
        history_seq = torch.cat([S_hist[:, :, :S_DIM], A_hist], dim=-1)
        hist_embed = self.history_seq_embed(history_seq)

        # 2. Embed Current State
        t_embed = self.state_embed_t(S_t).unsqueeze(1)

        # 3. Transformer Encoding
        seq_embed = torch.cat([hist_embed, t_embed], dim=1)
        seq_embed_pos = self.pos_encoder(seq_embed)
        transformer_out = self.transformer_encoder(seq_embed_pos)
        final_hidden_state = transformer_out[:, -1, :]

        # 4. Embed Action
        a_emb = torch.tanh(self.action_proj(A_t))

        # 5. Fuse Features
        if self.use_multi_heads:
            state_q = final_hidden_state.unsqueeze(1)
            action_kv = a_emb.unsqueeze(1)
            action_aware_embedding, _ = self.state_action_attention(
                query=state_q, 
                key=action_kv, 
                value=action_kv
            )
            fused_embedding = self.fusion_layer_norm(final_hidden_state + action_aware_embedding.squeeze(1))
            prediction_inputs = [fused_embedding]
        else:
            prediction_inputs = [final_hidden_state, a_emb]

        # Add Auxiliary Embeddings
        if self.use_text_embeddings and source_embeddings is not None and prefix_embeddings is not None:
            combined_text_emb = torch.cat([source_embeddings, prefix_embeddings], dim=1)
            text_emb = self.text_embed_proj(combined_text_emb)
            prediction_inputs.append(text_emb)

        if self.use_decoder_hidden_state and decoder_hidden_state is not None:
            if decoder_hidden_state.dim() == 3 and decoder_hidden_state.shape[1] == 1:
                decoder_hidden_state = decoder_hidden_state.squeeze(1)
            hs_emb = self.hidden_state_proj(decoder_hidden_state)
            prediction_inputs.append(hs_emb)

        combined_features = torch.cat(prediction_inputs, dim=1)

        # Predict Risk
        log_risk = self.risk_head(final_hidden_state).view(-1)

        # ==============================================================================
        # STAGE 2: OUTPUT PREDICTION (Heads)
        # ==============================================================================

        if self.use_moe_heads:
            out_E = self.out_head_E(combined_features)
            out_H = self.out_head_H(combined_features)
        else:
            out_EH = self.out_head(combined_features)
            out_E, out_H = out_EH[:, :E_INDEX[1] + 1], out_EH[:, E_INDEX[1] + 1:E_INDEX[1] + H_DIM + 1]

        # Apply Delta if configured
        if self.predict_delta:
            final_E = S_t[:, :E_INDEX[1] + 1] + out_E
            final_H = S_t[:, H_INDEX[0]:H_INDEX[1] + 1] + out_H
        else:
            final_E = out_E
            final_H = out_H

        return final_E, final_H, log_risk

# -----------------------------
# Training & Evaluations
# -----------------------------
def get_master_stability_weight(epoch: int, args) -> float:
    """
    Computes the curriculum weight for stability-related losses based on the current epoch.

    Phase 1: Pure prediction training (weight = 0).
    Phase 2: Gradual ramp-up of stability constraints (weight 0 -> 1).

    Args:
        epoch (int): Current training epoch.
        args: Argument namespace with curriculum settings.

    Returns:
        float: The weighting factor [0.0, 1.0] for stability losses.
    """
    if not args.use_curriculum:
        return 1.0

    if epoch <= args.curriculum_phase1_epochs:
        return 0.0

    # Start ramping after phase 1
    ramp_start_epoch = args.curriculum_phase1_epochs + 1
    epochs_into_ramp = epoch - ramp_start_epoch

    # Linearly interpolate based on warmup period
    master_stability_weight = _lin_schedule(
        ep=epochs_into_ramp,
        total_ep=args.lambda_warmup_ep,  # Total ramp duration
        start=0.0,
        end=1.0,
        warmup_ep=args.lambda_warmup_ep
    )

    return master_stability_weight

def compute_metrics(
    model, loader, device, V_fn: Callable[[torch.Tensor], torch.Tensor], 
    rho: float, phi_crit: Optional[float],
    rollout_H: int, num_delta_dirs: int, action_delta: float, softmin_tau: float,
    args, scaler: Pipeline
) -> Dict[str, float]:
    """
    Evaluates the model on the validation set, computing both prediction accuracy
    and control/stability metrics.

    Metrics include:
    - RMSE/R2/W1 for Error and Context state predictions.
    - Lyapunov stability violation rates.
    - Multi-step rollout stability coverage.
    - Jacobian spectral norm (Lipschitz proxy).
    """
    model.eval()

    state_component_names = STATE_COLS_DEFAULT
    metrics_collectors = defaultdict(list)
    all_nstep_dvs = []
    residuals_by_seq: Dict[int, List[float]] = {}

    batch_idx = 0
    # Use efficient attention if available
    backend_choice = (
        SDPBackend.EFFICIENT_ATTENTION if torch.cuda.is_available() else SDPBackend.MATH
    )

    with sdpa_kernel(backend_choice):
        with torch.no_grad():
            for batch in loader:
                batch_idx += 1

                batch = to_device(batch, device)
                S_t = batch["S_t"]; S_hist = batch["S_hist"]
                A_t = batch["A_t"]; A_hist = batch["A_hist"]
                S_tp1_target_scaled = batch["S_tp1"][:, :S_DIM] # This is either ΔS or S_tp1 depending on config

                source_embeddings = batch.get("source_embeddings")
                prefix_embeddings = batch.get("prefix_embeddings")
                decoder_hidden_state = batch.get("decoder_hidden_state")

                # Forward pass
                model_output = model(
                    S_t, S_hist, A_t, A_hist,
                    decoder_hidden_state=decoder_hidden_state,
                    source_embeddings=source_embeddings,
                    prefix_embeddings=prefix_embeddings
                )

                # =================== [START] METRIC RECONSTRUCTION LOGIC ===================

                S_t_base = S_t[:, :S_DIM]

                # --- 1. Reconstruct GROUND TRUTH absolute next state ---
                if args.predict_delta:
                    # S_tp1_target_scaled is ΔS, add to S_t
                    S_tp1_abs_scaled = S_t_base + S_tp1_target_scaled
                else:
                    S_tp1_abs_scaled = S_tp1_target_scaled

                # --- 2. Reconstruct PREDICTED absolute next state ---
                S_pred_abs_scaled = torch.zeros_like(S_t_base)
                E_pred_out, H_pred_out, log_risk = model_output

                if args.predict_delta:
                    E_t_scaled, _, H_t_scaled = split_state(S_t_base)
                    E_tp1_pred_scaled = E_t_scaled + E_pred_out
                    H_tp1_pred_scaled = H_t_scaled + H_pred_out
                else:
                    E_tp1_pred_scaled = E_pred_out
                    H_tp1_pred_scaled = H_pred_out

                # INERTIA ASSUMPTION: Phi (Pressure) does not change instantly.
                # Copy Phi from current state to predicted state.
                _, Phi_t_scaled, _ = split_state(S_t_base)
                S_pred_abs_scaled = torch.cat([E_tp1_pred_scaled, Phi_t_scaled, H_tp1_pred_scaled], dim=1)

                # --- 3. Inverse Transform (Back to Original Scale) for Interpretable Metrics ---
                s_pred_abs_np = S_pred_abs_scaled.cpu().numpy()
                s_tp1_abs_np = S_tp1_abs_scaled.cpu().numpy()

                s_pred_abs_df = pd.DataFrame(s_pred_abs_np, columns=STATE_COLS_DEFAULT)
                s_tp1_abs_df = pd.DataFrame(s_tp1_abs_np, columns=STATE_COLS_DEFAULT)

                S_pred_original_np = scaler.inverse_transform(s_pred_abs_df)
                S_tp1_original_np = scaler.inverse_transform(s_tp1_abs_df)

                S_pred_original = torch.from_numpy(S_pred_original_np.astype(np.float32)).to(device)
                S_tp1_original = torch.from_numpy(S_tp1_original_np.astype(np.float32)).to(device)

                # Eval specific components (E and H)
                e_indices = list(range(E_INDEX[0], E_INDEX[1] + 1))
                h_indices = list(range(H_INDEX[0], H_INDEX[1] + 1))
                components_to_eval = e_indices + h_indices

                for i in components_to_eval:
                    pred_comp = S_pred_original[:, i]
                    true_comp = S_tp1_original[:, i]
                    comp_name = state_component_names[i]

                    # RMSE
                    comp_rmse = torch.sqrt(torch.mean((pred_comp - true_comp) ** 2)).item()
                    metrics_collectors[f"rmse_{comp_name}"].append(comp_rmse)

                    # R2 Score
                    mean_true = torch.mean(true_comp)
                    ss_tot = torch.sum((true_comp - mean_true) ** 2)
                    ss_res = torch.sum((true_comp - pred_comp) ** 2)
                    comp_r2 = 1.0 - (ss_res / ss_tot).item() if ss_tot > 1e-8 else 0.0
                    metrics_collectors[f"r2_{comp_name}"].append(comp_r2)

                    # Wasserstein-1 (Mean Absolute Error)
                    comp_w1 = torch.mean(torch.abs(pred_comp - true_comp)).item()
                    metrics_collectors[f"w1_{comp_name}"].append(comp_w1)

                metrics_collectors["rmse_total"].append(torch.sqrt(torch.mean((S_pred_original - S_tp1_original) ** 2)).item())
                metrics_collectors["w1_total"].append(torch.mean(torch.abs(S_pred_original - S_tp1_original)).item())

                # --- Control and Stability Metrics ---

                # Calculate Lyapunov drift rate
                metrics_collectors["lyapunov_pos_rate_mean"].append(
                    lyapunov_positive_rate(
                        model, S_t, S_hist, A_t, A_hist, V_fn,
                        decoder_hidden_state=decoder_hidden_state,
                        source_embeddings=source_embeddings,
                        prefix_embeddings=prefix_embeddings
                    )
                )

                # Check if the teacher's action (ground truth) violated the CLF condition
                V_t = V_fn(S_t)
                V_next = V_fn(S_pred_abs_scaled) # V(S_t+1) based on prediction
                tv = (V_next - (1.0 - rho) * V_t > 0).float().mean().item()
                metrics_collectors["clf_violation_rate_teacher"].append(tv)

                # Validation Cox Loss
                if args.lambda_cox > 0:
                    T, E = batch["T"], batch["E"]
                    loss_cox_val = cox_ph_loss(log_risk, T, E)
                    metrics_collectors["cox_loss_val"].append(loss_cox_val.item())

                # Validation CBF Loss (Teacher)
                if args.lambda_cbf > 0 and phi_crit is not None and phi_crit > 0:
                     cbf_loss_teacher = control_barrier_loss(S_t, S_pred_abs_scaled, phi_crit, args.cbf_alpha)
                     metrics_collectors["cbf_loss_teacher_val"].append(cbf_loss_teacher.item())

                # Calculate optimal local CLF violation (if we chose the best action)
                _, _clf_violation = clf_loss(
                    model, S_t, S_hist, A_t, A_hist, V_fn, rho=rho,
                    num_delta_dirs=num_delta_dirs,
                    action_delta=action_delta,
                    softmin_tau=softmin_tau,
                    decoder_hidden_state=decoder_hidden_state,
                    source_embeddings=source_embeddings,
                    prefix_embeddings=prefix_embeddings
                )
                metrics_collectors["clf_violation_rate"].append(_clf_violation)

                # N-Step Rollout Validation
                if args.use_nstep_clf:
                    _, nstep_stats = n_step_clf_loss(
                        model=model, S_t=S_t, S_hist=S_hist, A_t=A_t, A_hist=A_hist, V_fn=V_fn,
                        decoder_hidden_state=decoder_hidden_state,
                        source_embeddings=source_embeddings, prefix_embeddings=prefix_embeddings,
                        nstep_H=args.nstep_H, nstep_gamma=1.0,
                        nstep_selector='hard_greedy',
                        nstep_bptt_window=-1,
                        use_cvar_loss=False, cvar_alpha=0.0,
                        use_epsilon_greedy=False, epsilon=0.0,
                        policy_entropy_weight=0.0,
                        rho=rho,
                        num_delta_dirs=num_delta_dirs, action_delta=action_delta,
                        gumbel_tau=0.01,
                        return_dv_trajectory=True,
                        lambda_adt=0.0,
                        lambda_cbf=args.lambda_cbf if args.lambda_cbf > 0 else 0.0,
                        phi_crit=args.phi_crit if args.phi_crit > 0 else 0.0,
                        cbf_alpha=args.cbf_alpha
                    )
                    for k, v in nstep_stats.items():
                        if k == "dv_trajectory":
                            all_nstep_dvs.append(v.cpu())
                        else:
                            metrics_collectors[k].append(v)

                # Multi-Step Negative Drift Coverage (H=5)
                # Checks if the model can consistently find stabilizing actions over a rollout.
                neg_cov = multi_step_negative_drift_coverage(
                    model, S_t, S_hist, A_t, A_hist, V_fn, rho=rho, rollout_H=rollout_H,
                    action_delta=action_delta,
                    num_delta_dirs=num_delta_dirs,
                    decoder_hidden_state=decoder_hidden_state,
                    source_embeddings=source_embeddings,
                    prefix_embeddings=prefix_embeddings
                )
                if not math.isnan(neg_cov):
                    metrics_collectors["multi_step_neg_drift_coverage"].append(neg_cov)

                # Compare with Teacher coverage
                cov_teacher = multi_step_negative_drift_coverage_teacher(
                    model, S_t, S_hist, A_t, A_hist, V_fn, rho=rho, rollout_H=rollout_H,
                    decoder_hidden_state=decoder_hidden_state,
                    source_embeddings=source_embeddings,
                    prefix_embeddings=prefix_embeddings
                )
                if not math.isnan(cov_teacher):
                    metrics_collectors["multi_step_neg_drift_coverage_teacher"].append(cov_teacher)

                # Compare with Fixed Action coverage
                cov_fixed0 = multi_step_negative_drift_coverage_fixed(
                    model, S_t, S_hist, A_hist, V_fn, rho=rho, rollout_H=rollout_H,
                    fixed_idx=0, action_dim=A_t.shape[1],
                    decoder_hidden_state=decoder_hidden_state,
                    source_embeddings=source_embeddings,
                    prefix_embeddings=prefix_embeddings
                )
                if not math.isnan(cov_fixed0):
                    metrics_collectors["multi_step_neg_drift_coverage_fixed0"].append(cov_fixed0)

                # Collect Ljung-Box residuals
                if "seq_id" in batch:
                    # Calculate mean residual per sequence to test for autocorrelation
                    resid = (S_pred_abs_scaled - S_tp1_abs_scaled).mean(dim=1).detach().cpu().numpy()
                    ids = batch["seq_id"].detach().cpu().numpy().tolist()
                    for r, sid in zip(resid, ids):
                        residuals_by_seq.setdefault(int(sid), []).append(float(r))

    # Aggregate metrics
    metrics = {}
    for key, val_list in metrics_collectors.items():
        if val_list:
            clean_list = [v for v in val_list if v is not None and not math.isnan(v)]
            if clean_list:
                metrics[key] = float(np.mean(clean_list))
            else:
                metrics[key] = float("nan")
        else:
            metrics[key] = float("nan")

    # Flatten specific stats
    if "clf_violation_rate_mean" in metrics:
        metrics["nstep_clf_violation_rate_mean"] = metrics.pop("clf_violation_rate_mean")
    for i in range(args.nstep_H):
        step_key = f"clf_violation_rate_step{i}"
        if step_key in metrics:
            metrics[f"nstep_{step_key}"] = metrics.pop(step_key)

    # Compute Q-Statistic
    Q_list = [ljung_box_Q(np.array(rseq), lags=10) for sid, rseq in residuals_by_seq.items()]
    Q_list = [q for q in Q_list if not math.isnan(q)]
    metrics["ljung_box_Q"] = float(np.mean(Q_list)) if Q_list else float("nan")

    # Compute Jacobian Spectral Norm (Lipschitz Proxy)
    try:
        first_batch = next(iter(loader))
        batch = to_device(first_batch, device)
        source_embeddings = batch.get("source_embeddings")
        prefix_embeddings = batch.get("prefix_embeddings")
        decoder_hidden_state = batch.get("decoder_hidden_state")
        
        metrics["jacobian_spectral_norm_mean"] = spectral_norm_jacobian_mean(
            model, batch["S_t"], batch["S_hist"], batch["A_t"], batch["A_hist"],
            sample_size=64,
            source_embeddings=source_embeddings,
            prefix_embeddings=prefix_embeddings,
            decoder_hidden_state=decoder_hidden_state
        )
    except Exception as e:
        print(f"[Error] calculating jacobian_spectral_norm_mean: {e}")
        traceback.print_exc()
        metrics["jacobian_spectral_norm_mean"] = float("nan")

    if len(all_nstep_dvs) > 0:
        all_nstep_dvs_tensor = torch.cat(all_nstep_dvs, dim=0)
        metrics["nstep_endpoint_violation_rate"] = (all_nstep_dvs_tensor[:, -1] > 0).float().mean().item()
        all_violations = F.relu(all_nstep_dvs_tensor.flatten())
        metrics["nstep_cvar_violation"] = cvar_loss(all_violations, args.cvar_alpha).item()

    return metrics

def train_one_epoch(
    model, model_for_jac,
    loader, optimizer, device, grad_scaler,
    V_fn: Callable[[torch.Tensor], torch.Tensor], args, epoch: int,
    log_vars_param: Optional[nn.Parameter] = None
) -> Dict[str, float]:
    """
    Performs one full epoch of training.
    Optimizes prediction loss, stability loss (CLF/CBF), and auxiliary risks.
    Handles curriculum learning for stability weights.
    """
    model.train()

    state_component_names = STATE_COLS_DEFAULT
    pred_loss_collectors = defaultdict(list)
    clf_violation_rates, clf_losses, total_losses, cox_losses = [], [], [], []
    
    # --- Curriculum Learning for Stability ---
    # Phase 1: Prediction only (weight = 0).
    # Phase 2: Linearly ramp up stability weight to 1.
    if args.use_curriculum:
        master_stability_weight = get_master_stability_weight(epoch, args)
    else:
        master_stability_weight = 1.0

    # Calculate effective epoch for hyperparameter schedules
    if args.use_curriculum:
        control_phase_epoch = max(0, epoch - args.curriculum_phase1_epochs)
    else:
        control_phase_epoch = epoch

    # Anneal Hyperparameters
    rho_ep = _lin_schedule(control_phase_epoch, args.epochs, args.rho, args.rho_final, args.rho_warmup_ep)
    lclf_base = args.lambda_clf 
    gumbel_tau_ep = _exp_anneal(control_phase_epoch, args.epochs, args.gumbel_tau_init, args.gumbel_tau_final, args.gumbel_anneal_ep)
    epsilon_ep = _lin_schedule(control_phase_epoch, args.epochs, args.epsilon_init, args.epsilon_final, args.epsilon_decay_ep)

    huber_loss_fn = nn.HuberLoss(reduction='none') # Per-element loss for weighting
    
    backend_choice = (
        SDPBackend.EFFICIENT_ATTENTION if torch.cuda.is_available() else SDPBackend.MATH
    )

    with sdpa_kernel(backend_choice):

        for i, batch in enumerate(loader):
            batch = to_device(batch, device)
            S_t = batch["S_t"]; S_hist = batch["S_hist"]
            A_t = batch["A_t"]; A_hist = batch["A_hist"]
            S_tp1 = batch["S_tp1"][:, :S_DIM]
            T = batch["T"]
            E = batch["E"]
            decoder_hidden_state = batch.get("decoder_hidden_state")

            # Pre-compute weights for high-pressure or divergent samples
            with torch.no_grad():
                V_t_true = V_fn(S_t)
                V_tp1_true = V_fn(S_tp1)
                dV_true = V_tp1_true - V_t_true
                _, Phi_t, _ = split_state(S_t)
                high_pressure = (torch.norm(Phi_t, dim=1) > 1.0).float()
                # Reweight samples where V increases or pressure is high
                w = torch.ones_like(dV_true)
                w = torch.where(dV_true > 0, torch.full_like(w, args.teacher_reweight_alpha), w)
                w = torch.where(high_pressure > 0, torch.full_like(w, args.teacher_reweight_alpha), w)
                w = w.detach()

            enable_autocast = not args.disable_autocast and args.device == 'cuda' and not sys.platform.startswith('win')
            
            # Setup Autocast Context
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                autocast = getattr(torch.amp, 'autocast')
                autocast_context = autocast(device_type=args.device, enabled=enable_autocast)
            elif hasattr(torch, 'autocast'):
                autocast_context = torch.autocast(device_type=args.device, enabled=enable_autocast)
            else:
                autocast_context = torch.cuda.amp.autocast(enabled=enable_autocast) if args.device == 'cuda' else NullContext()

            with autocast_context:
                
                source_embeddings = batch.get("source_embeddings")
                prefix_embeddings = batch.get("prefix_embeddings")
                
                # S_tp1_target is either the delta or absolute state depending on configuration
                S_tp1_target = batch["S_tp1"][:, :S_DIM]

                # Model Forward Pass
                model_output = model(
                    S_t, S_hist, A_t, A_hist,
                    decoder_hidden_state=decoder_hidden_state,
                    source_embeddings=source_embeddings, 
                    prefix_embeddings=prefix_embeddings
                )

                pred_E, pred_H, log_risk = model_output
                pred_E = pred_E.contiguous(); pred_H = pred_H.contiguous()
                
                # Split target into components
                target_E, _, target_H = split_state(S_tp1_target)
                    
                # Calculate Huber Loss per component
                loss_E = huber_loss_fn(pred_E, target_E)
                loss_H = huber_loss_fn(pred_H, target_H)
                    
                per_element_loss = torch.cat([loss_E, loss_H], dim=1)
                    
                # Logging component losses
                state_component_names_EH = state_component_names[E_INDEX[0]:E_INDEX[1]+1] + state_component_names[H_INDEX[0]:H_INDEX[1]+1]
                for i in range(per_element_loss.shape[1]):
                    comp_name = state_component_names_EH[i]
                    pred_loss_collectors[f"huber_{comp_name}"].append(torch.mean(per_element_loss[:, i]).detach().item())

                # Apply Uncertainty Weighting or Simple Mean
                if args.use_uncertainty_weights and log_vars_param is not None:
                    if log_vars_param.numel() != E_DIM + H_DIM:
                        raise ValueError(f"log_vars_param must be ({E_DIM}+{H_DIM}={E_DIM+H_DIM})-dim, but found {log_vars_param.numel()}")
                    weighted_losses = per_element_loss * w.unsqueeze(1)
                    mean_losses = torch.mean(weighted_losses, dim=0)
                    # Adaptive loss weighting
                    final_losses = torch.exp(-log_vars_param) * mean_losses + log_vars_param
                    huber_loss = torch.sum(final_losses)
                else:
                    huber_loss = torch.mean(w * (torch.mean(loss_E, dim=1) + torch.mean(loss_H, dim=1)))
                
                pred_loss_collectors["huber_total"].append(huber_loss.detach().item())

                # Cox Loss for Risk Prediction
                loss_cox = torch.tensor(0.0, device=device)
                if args.lambda_cox > 0:
                    loss_cox = cox_ph_loss(log_risk, T, E)
                    if not torch.isnan(loss_cox):
                        cox_losses.append(loss_cox.detach().item())

                # Calculate Stability Losses (CLF/CBF/ADT/Jac)
                clf_l = torch.tensor(0.0, device=device)
                clf_violation = 0.0
                if master_stability_weight > 0:
                    
                    if args.use_nstep_clf:
                        clf_l, clf_stats = n_step_clf_loss(
                            model=model, S_t=S_t, S_hist=S_hist, A_t=A_t, A_hist=A_hist, V_fn=V_fn,
                            decoder_hidden_state=decoder_hidden_state,
                            source_embeddings=source_embeddings, prefix_embeddings=prefix_embeddings,
                            nstep_H=args.nstep_H, nstep_gamma=args.nstep_gamma,
                            nstep_selector=args.nstep_selector, nstep_bptt_window=args.nstep_bptt_window,
                            use_cvar_loss=args.use_cvar_loss, cvar_alpha=args.cvar_alpha,
                            use_epsilon_greedy=args.use_epsilon_greedy, epsilon=epsilon_ep,
                            policy_entropy_weight=args.policy_entropy_weight,
                            rho=rho_ep,
                            num_delta_dirs=args.num_delta_dirs, action_delta=args.action_delta,
                            gumbel_tau=gumbel_tau_ep,
                            lambda_adt=args.lambda_adt, lambda_cbf=args.lambda_cbf,
                            phi_crit=args.phi_crit, cbf_alpha=args.cbf_alpha
                        )
                        clf_l = args.nstep_lambda * clf_l
                        clf_violation = clf_stats.get("clf_violation_rate_mean", 0.0)
                    else: # Single-step CLF
                        clf_l, clf_violation = clf_loss(
                            model, S_t, S_hist, A_t, A_hist, V_fn, rho=rho_ep,
                            num_delta_dirs=args.num_delta_dirs,
                            action_delta=args.action_delta, softmin_tau=args.softmin_tau,
                            decoder_hidden_state=decoder_hidden_state,
                            source_embeddings=source_embeddings, prefix_embeddings=prefix_embeddings
                        )
                        clf_l = lclf_base * clf_l
                    
                clf_violation_rates.append(clf_violation)

                # Jacobian Spectral Norm Regularization
                jac_pen = torch.tensor(0.0, device=device)
                if master_stability_weight > 0 and args.jacobian_reg > 0.0:
                    # Switch to MATH backend for double-backward support if needed
                    with sdpa_kernel(SDPBackend.MATH):
                        with torch.enable_grad():
                            # Sample a subset to save compute
                            sub_indices = torch.randperm(S_t.shape[0], device=device)[:min(64, S_t.shape[0])]
                            S_sub = S_t[sub_indices].detach().clone().requires_grad_(True)
                            S_hist_sub, A_sub, A_hist_sub = S_hist[sub_indices], A_t[sub_indices], A_hist[sub_indices]
                            src_emb_sub = source_embeddings[sub_indices] if source_embeddings is not None else None
                            pref_emb_sub = prefix_embeddings[sub_indices] if prefix_embeddings is not None else None
                            hs_sub = decoder_hidden_state[sub_indices] if decoder_hidden_state is not None else None
                            
                            spectral_norms = []
                            for i in range(S_sub.shape[0]):
                                # Helper for computing Jacobian per sample
                                def jac_func_single(s_single):
                                    model_output = model_for_jac(
                                        s_single.unsqueeze(0), S_hist_sub[i].unsqueeze(0), 
                                        A_sub[i].unsqueeze(0), A_hist_sub[i].unsqueeze(0),
                                        decoder_hidden_state=hs_sub[i].unsqueeze(0) if hs_sub is not None else None,
                                        source_embeddings=src_emb_sub[i].unsqueeze(0) if src_emb_sub is not None else None,
                                        prefix_embeddings=pref_emb_sub[i].unsqueeze(0) if pref_emb_sub is not None else None,
                                    )
                                    return torch.cat(model_output[:2], dim=1).squeeze(0)

                                J = torch.autograd.functional.jacobian(
                                    jac_func_single,
                                    S_sub[i],
                                    create_graph=True # Enable graph for penalty optimization
                                )
                                sigma = torch.linalg.svdvals(J).max()
                                spectral_norms.append(sigma)
                                    
                            if spectral_norms:
                                jac_sigma = torch.mean(torch.stack(spectral_norms))
                                jac_pen = args.jacobian_reg * jac_sigma

                # Combine all losses
                stability_loss_component = clf_l + jac_pen
                loss = huber_loss + (master_stability_weight * stability_loss_component) + (args.lambda_cox * loss_cox if not torch.isnan(loss_cox) else 0.0)

            # Optimization Step
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            if not torch.isnan(clf_l):
                clf_losses.append(clf_l.detach().item())
            total_losses.append(loss.detach().item())
    
    log_dict = {
        "loss": float(np.mean(total_losses)) if total_losses else float("nan"),
        "clf_loss": float(np.mean(clf_losses)) if clf_losses else float("nan"),
        "clf_violation_rate": float(np.mean(clf_violation_rates)) if clf_violation_rates else float("nan"),
        "cox_loss": float(np.mean(cox_losses)) if cox_losses else float("nan")
    }
    for key, values in pred_loss_collectors.items():
        log_dict[key] = float(np.mean(values)) if values else float("nan")

    return log_dict

# --- S2: Simple Scheduler ---
def _lin_schedule(ep, total_ep, start, end, warmup_ep):
    """Linearly interpolates a value from start to end over a warmup period."""
    if end is None or warmup_ep <= 0:
        return start
    t = min(max(ep / max(1, warmup_ep), 0.0), 1.0)
    return float(start + (end - start) * t)

def _exp_anneal(ep, total_ep, start, end, anneal_ep):
    """Exponentially anneals a value from start to end."""
    if end is None or anneal_ep <= 0:
        return start
    r = min(max(ep / max(1, anneal_ep), 0.0), 1.0)
    return float(start * (end / max(1e-12, start)) ** r)

# --- S2: Validation set export: optimal action statistics and rolling trajectory ---
def _argmin_softmin_deltaV(
    model, 
    S: torch.Tensor, S_hist: torch.Tensor, 
    A_teacher: torch.Tensor, A_hist: torch.Tensor, 
    V_fn: Callable[[torch.Tensor], torch.Tensor],
    rho, cand,
    source_embeddings: Optional[torch.Tensor] = None,
    prefix_embeddings: Optional[torch.Tensor] = None,
    decoder_hidden_state: Optional[torch.Tensor] = None
):
    """
    Searches for the optimal candidate action by minimizing the predicted
    Lyapunov delta (V_next - (1-rho)V_curr).

    Used during validation to check if *any* stable action exists locally.
    """
    # Returns: best_action_idx [B], best_dV [B], best_Snext [B, S_DIM]
    K, B, dA = cand.shape
    with torch.no_grad():
        V_now = V_fn(S) # [B]
        _, Phi_t, _ = split_state(S)
        
        best_dv = torch.full((B,), float('inf'), device=S.device)
        best_Sn = torch.empty_like(S)
        best_k = torch.zeros(B, dtype=torch.long, device=S.device)
        
        for k in range(K):
            model_output = model(
                S, S_hist, cand[k], A_hist,
                decoder_hidden_state=decoder_hidden_state,
                source_embeddings=source_embeddings, 
                prefix_embeddings=prefix_embeddings
            )
            
            E_next, H_next, _ = model_output
            S_next = torch.cat([E_next, Phi_t, H_next], dim=1)

            V_next = V_fn(S_next)
            dv = V_next - (1.0 - rho) * V_now
            
            mask = dv < best_dv
            best_k = torch.where(mask, k, best_k)
            best_dv = torch.where(mask, dv, best_dv)
            best_Sn = torch.where(mask.view(-1,1), S_next, best_Sn)

        return best_k, best_dv, best_Sn

def export_val_action_stats_and_rollout(
    model, 
    loader, 
    device, 
    V_fn: Callable[[torch.Tensor], torch.Tensor], 
    rho,
    num_delta_dirs, action_delta, save_dir,
    export_stats: bool=False, export_rollout: bool=False
):
    """
    Calculates and exports statistics on optimal actions and single-step rollouts
    for the validation set. Useful for analyzing the control landscape.
    """
    if not (export_stats or export_rollout):
        return
    model.eval()
    all_k = []
    rollout_rows = []

    backend_choice = (
        SDPBackend.EFFICIENT_ATTENTION if torch.cuda.is_available() else SDPBackend.MATH
    )

    with sdpa_kernel(backend_choice):
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, device)
                S = batch["S_t"]; S_hist = batch["S_hist"]
                A = batch["A_t"]; A_hist = batch["A_hist"]
                source_embeddings = batch.get("source_embeddings")
                prefix_embeddings = batch.get("prefix_embeddings")
                seq_ids = batch.get("seq_id", None)
                decoder_hidden_state = batch.get("decoder_hidden_state")

                cand = build_action_candidates(
                    S_t=S, A_t=A,
                    num_delta_dirs=num_delta_dirs,
                    action_delta=action_delta
                )

                best_k, best_dV, _ = _argmin_softmin_deltaV(
                    model, S, S_hist, A, A_hist, V_fn, rho, cand, 
                    decoder_hidden_state=decoder_hidden_state,
                    source_embeddings=source_embeddings, 
                    prefix_embeddings=prefix_embeddings
                )
                if export_stats:
                    all_k.append(best_k.detach().cpu().numpy())

                if export_rollout:
                    V_now = V_fn(S)
                    for i in range(S.shape[0]):
                        sid = int(seq_ids[i].item()) if seq_ids is not None else -1
                        rollout_rows.append({
                            "seq_id": sid,
                            "step": 0,
                            "best_k": int(best_k[i].item()),
                            "V_t": float(V_now[i].item()),
                            "dV": float(best_dV[i].item())
                        })

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if export_stats and len(all_k) > 0:
        ks = np.concatenate(all_k, axis=0)
        unique_k, counts = np.unique(ks, return_counts=True)
        hist = dict(zip(map(int, unique_k), map(int, counts)))
        with open(save_dir / "val_best_action_hist.json", "w", encoding="utf-8") as f:
            json.dump({"hist": hist, "note": "Histogram of best candidate action indices 'k' on validation set."}, f, ensure_ascii=False, indent=2)

    if export_rollout and len(rollout_rows) > 0:
        pd.DataFrame(rollout_rows).to_csv(save_dir / "val_rollout_one_step.csv", index=False)
        
# ============================================================================
# S8: Theoretical Validation Suite Functions (COMPLETE & MODULAR)
# ============================================================================

def s8_robust_spectral_norm_jacobian(
    model,
    S_t: torch.Tensor, S_hist: torch.Tensor,
    A_t: torch.Tensor, A_hist: torch.Tensor,
    sample_size: int = 256,
    num_random_starts: int = 5,
    source_embeddings: Optional[torch.Tensor] = None,
    prefix_embeddings: Optional[torch.Tensor] = None,
    decoder_hidden_state: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> Dict[str, Union[float, List[float]]]:
    """
    Calculates the robust spectral norm of the model's Jacobian.
    Validates Lipschitz continuity assumptions by ensuring the gradient norm is bounded near 1.

    Returns: Statistics (mean, max, percentiles) of the spectral norms.
    """
    B = S_t.shape[0]
    sample_size = min(sample_size, B)
    all_results = []

    # Move tensors to device
    S_t, S_hist, A_t, A_hist = S_t.to(device), S_hist.to(device), A_t.to(device), A_hist.to(device)
    if source_embeddings is not None: source_embeddings = source_embeddings.to(device)
    if prefix_embeddings is not None: prefix_embeddings = prefix_embeddings.to(device)
    if decoder_hidden_state is not None: decoder_hidden_state = decoder_hidden_state.to(device)

    print(f"[S8 Jacobian] Running validation on {num_random_starts} random samples of size {sample_size}...")

    for restart_idx in range(num_random_starts):
        # Sample batch
        idx = torch.randperm(B, device=device)[:sample_size]
        
        # Prepare input with gradients enabled
        S_sample = S_t[idx].detach().clone().requires_grad_(True)
        Sh_sample, A_sample, Ah_sample = S_hist[idx], A_t[idx], A_hist[idx]
        src_emb_sample = source_embeddings[idx] if source_embeddings is not None else None
        pref_emb_sample = prefix_embeddings[idx] if prefix_embeddings is not None else None
        hs_sub_sample = decoder_hidden_state[idx] if decoder_hidden_state is not None else None

        # Function wrapper for Jacobian calculation
        def func_to_jac(s_single, sh_single, a_single, ah_single, src_emb_single, pref_emb_single, hs_single):
            model_output = model(
                s_single.unsqueeze(0), sh_single.unsqueeze(0),
                a_single.unsqueeze(0), ah_single.unsqueeze(0),
                decoder_hidden_state=hs_single.unsqueeze(0) if hs_single is not None else None,
                source_embeddings=src_emb_single.unsqueeze(0) if src_emb_single is not None else None,
                prefix_embeddings=pref_emb_single.unsqueeze(0) if pref_emb_single is not None else None
            )

            out_single = torch.cat(model_output[:2], dim=1)
            return out_single.squeeze(0)

        # Compute spectral norms per sample
        spectral_norms_for_this_restart = []
        for i in range(sample_size):
            J = torch.autograd.functional.jacobian(
                lambda s: func_to_jac(
                    s, Sh_sample[i], A_sample[i], Ah_sample[i],
                    src_emb_sample[i] if src_emb_sample is not None else None,
                    pref_emb_sample[i] if pref_emb_sample is not None else None,
                    hs_sub_sample[i] if hs_sub_sample is not None else None
                ),
                S_sample[i], create_graph=False, strict=False
            )
            singular_values = torch.linalg.svdvals(J)
            sigma = singular_values.max().item()
            spectral_norms_for_this_restart.append(sigma)

        all_results.append({
            'restart_idx': restart_idx,
            'sample_indices': idx.cpu().numpy(),
            'final_values': np.array(spectral_norms_for_this_restart),
            'converged': True,
            'iterations': 1,
            'iteration_history': np.array(spectral_norms_for_this_restart)[np.newaxis, :]
        })
        print(f"[S8 Jacobian] Sample {restart_idx+1}/{num_random_starts} processed.")

    if not all_results:
        print("[Warning] S8: No spectral norms were calculated.")
        return {
            'mean': float('nan'), 'std': float('nan'), 'max': float('nan'), 'min': float('nan'),
            'percentile_50': float('nan'), 'percentile_90': float('nan'),
            'percentile_95': float('nan'), 'percentile_99': float('nan'),
            'converged_rate': 0.0, 'avg_iterations': 0.0, 'all_results': []
        }

    all_values = np.concatenate([r['final_values'] for r in all_results])

    return {
        'mean': float(np.mean(all_values)), 'std': float(np.std(all_values)),
        'max': float(np.max(all_values)), 'min': float(np.min(all_values)),
        'percentile_50': float(np.percentile(all_values, 50)),
        'percentile_90': float(np.percentile(all_values, 90)),
        'percentile_95': float(np.percentile(all_values, 95)),
        'percentile_99': float(np.percentile(all_values, 99)),
        'converged_rate': sum(r['converged'] for r in all_results) / num_random_starts,
        'avg_iterations': sum(r['iterations'] for r in all_results) / num_random_starts,
        'all_results': all_results
    }

def s8_lyapunov_validation(
    model, 
    loader, 
    V_fn: Callable[[torch.Tensor], torch.Tensor], 
    rho: float,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Validates Lyapunov function behavior.
    Checks if predicted next states satisfy the decay condition: V(S_pred) <= (1-rho)V(S_curr).
    """
    model.eval()
    all_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = to_device(batch, device)
            S_t = batch["S_t"]; S_hist = batch["S_hist"]
            A_t = batch["A_t"]; A_hist = batch["A_hist"]
            S_tp1_true = batch["S_tp1"]
            source_embeddings = batch.get("source_embeddings")
            prefix_embeddings = batch.get("prefix_embeddings")
            decoder_hidden_state = batch.get("decoder_hidden_state")
            
            model_output = model(
                S_t, S_hist, A_t, A_hist,
                decoder_hidden_state=decoder_hidden_state,
                source_embeddings=source_embeddings, 
                prefix_embeddings=prefix_embeddings
            )
            
            E_tp1_pred, H_tp1_pred, _ = model_output
            # Use ground truth Phi for pure prediction validation
            _, Phi_t, _ = split_state(S_t)
            S_tp1_pred = torch.cat([E_tp1_pred, Phi_t, H_tp1_pred], dim=1)

            V_t = V_fn(S_t)
            V_tp1_pred = V_fn(S_tp1_pred)
            V_tp1_true = V_fn(S_tp1_true)
            
            target_V = (1 - rho) * V_t
            clf_satisfied_pred = (V_tp1_pred <= target_V)
            clf_satisfied_true = (V_tp1_true <= target_V)
            
            decay_rate_pred = V_tp1_pred / (V_t + 1e-8)
            decay_rate_true = V_tp1_true / (V_t + 1e-8)
            
            for i in range(S_t.shape[0]):
                all_data.append({
                    'batch_idx': batch_idx, 'sample_idx': i, 'V_t': V_t[i].item(),
                    'V_tp1_pred': V_tp1_pred[i].item(), 'V_tp1_true': V_tp1_true[i].item(),
                    'target_V': target_V[i].item(), 'clf_satisfied_pred': clf_satisfied_pred[i].item(),
                    'clf_satisfied_true': clf_satisfied_true[i].item(), 'decay_rate_pred': decay_rate_pred[i].item(),
                    'decay_rate_true': decay_rate_true[i].item(), 'theoretical_decay': 1 - rho,
                    'E_norm':   torch.norm(S_t[i, E_INDEX[0]:E_INDEX[1]+1]).item(),
                    'Phi_norm': torch.norm(S_t[i, PHI_INDEX[0]:PHI_INDEX[1]+1]).item(),
                    'H_norm':   torch.norm(S_t[i, H_INDEX[0]:H_INDEX[1]+1]).item()
                })
    
    df = pd.DataFrame(all_data)
    if df.empty:
        return {
            'clf_satisfaction_rate_pred': 0.0, 'clf_satisfaction_rate_true': 0.0,
            'mean_decay_rate_pred': 0.0, 'mean_decay_rate_true': 0.0,
            'std_decay_rate_pred': 0.0, 'theoretical_decay': 1 - rho, 'dataframe': df
        }
        
    return {
        'clf_satisfaction_rate_pred': df['clf_satisfied_pred'].mean(),
        'clf_satisfaction_rate_true': df['clf_satisfied_true'].mean(),
        'mean_decay_rate_pred': df['decay_rate_pred'].mean(),
        'mean_decay_rate_true': df['decay_rate_true'].mean(),
        'std_decay_rate_pred': df['decay_rate_pred'].std(),
        'theoretical_decay': 1 - rho,
        'dataframe': df
    }

def s8_cbf_invariance_validation(
    model, 
    loader, 
    phi_crit: float, 
    cbf_alpha: float,
    horizon: int = 10,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Validates Control Barrier Function (CBF) invariance.
    Checks if the system state remains within the safe set over a rollout horizon.
    """
    model.eval()
    all_data = []
    
    def compute_h(S):
        Phi = S[:, PHI_INDEX[0]:PHI_INDEX[1]+1]
        return phi_crit**2 - torch.sum(Phi**2, dim=1)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = to_device(batch, device)
            S_t = batch["S_t"]; S_hist = batch["S_hist"]
            A_t = batch["A_t"]; A_hist = batch["A_hist"]
            source_embeddings = batch.get("source_embeddings")
            prefix_embeddings = batch.get("prefix_embeddings")
            decoder_hidden_state = batch.get("decoder_hidden_state")
            
            h_values = [compute_h(S_t)]
            
            S_curr = S_t
            S_hist_curr = S_hist
            A_hist_curr = A_hist
            state_trajectory = [S_t]

            for step in range(horizon):
                # Assume no further action (or teacher action) for rollout
                A_curr = A_t if step == 0 else torch.zeros_like(A_t)
                
                _, Phi_curr, _ = split_state(S_curr)
                
                model_output = model(
                    S_curr, S_hist_curr, 
                    A_curr, A_hist_curr,
                    decoder_hidden_state=decoder_hidden_state,
                    source_embeddings=source_embeddings, 
                    prefix_embeddings=prefix_embeddings
                )

                E_next, H_next, _ = model_output
                S_next = torch.cat([E_next, Phi_curr, H_next], dim=1)
                
                h_next = compute_h(S_next)
                
                h_values.append(h_next)
                state_trajectory.append(S_next)
                
                if S_hist_curr.shape[1] > 0:
                    S_hist_curr = torch.cat([S_hist_curr[:, 1:, :], S_curr.unsqueeze(1)], dim=1)
                    A_hist_curr = torch.cat([A_hist_curr[:, 1:, :], A_curr.unsqueeze(1)], dim=1)
                
                S_curr = S_next
            
            h_trajectory = torch.stack(h_values, dim=1)
            full_state_trajectory = torch.stack(state_trajectory, dim=1)
            
            for i in range(S_t.shape[0]):
                for step in range(horizon):
                    h_t = h_trajectory[i, step].item()
                    h_tp1 = h_trajectory[i, step + 1].item()
                    target_h = (1 - cbf_alpha) * h_t
                    
                    all_data.append({
                        'batch_idx': batch_idx, 'sample_idx': i, 'step': step,
                        'h_t': h_t, 'h_tp1': h_tp1, 'target_h': target_h,
                        'cbf_satisfied': h_tp1 >= target_h,
                        'in_safe_set_t': h_t >= 0, 'in_safe_set_tp1': h_tp1 >= 0,
                        'Phi_norm_t':   torch.norm(full_state_trajectory[i, step,     PHI_INDEX[0]:PHI_INDEX[1]+1]).item(),
                        'Phi_norm_tp1': torch.norm(full_state_trajectory[i, step + 1, PHI_INDEX[0]:PHI_INDEX[1]+1]).item()
                    })
    
    if not all_data:
        return {'cbf_satisfaction_rate': 0.0, 'forward_invariance_rate': 0.0, 'safe_set_maintenance': 0.0, 'dataframe': pd.DataFrame()}

    df = pd.DataFrame(all_data)
    
    initial_safe_mask = (df['step'] == 0) & (df['in_safe_set_t'])
    safe_start_trajectories = df[initial_safe_mask]
    
    if not safe_start_trajectories.empty:
        invariance_results = safe_start_trajectories.groupby(['batch_idx', 'sample_idx']).apply(
            lambda group: df[(df['batch_idx'] == group.name[0]) & (df['sample_idx'] == group.name[1])]['in_safe_set_tp1'].all()
        )
        invariance_rate = invariance_results.mean()
    else:
        invariance_rate = float('nan')
    
    return {
        'cbf_satisfaction_rate': df['cbf_satisfied'].mean(),
        'forward_invariance_rate': invariance_rate,
        'safe_set_maintenance': df['in_safe_set_tp1'].mean(),
        'dataframe': df
    }

def s8_error_boundedness_validation(
    model, loader, scaler: Pipeline, device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Validates that the predicted error state remains bounded.
    Checks the norm of E_pred against theoretical bounds.
    """
    model.eval()
    all_data = []

    feature_names = STATE_COLS_DEFAULT

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = to_device(batch, device)
            
            model_output = model(
                S_t=batch["S_t"], S_hist=batch["S_hist"],
                A_t=batch["A_t"], A_hist=batch["A_hist"],
                decoder_hidden_state=batch.get("decoder_hidden_state"),
                source_embeddings=batch.get("source_embeddings"),
                prefix_embeddings=batch.get("prefix_embeddings")
            )
            
            E_pred_std, H_pred_std, _ = model_output
            _, Phi_t_std, _ = split_state(batch["S_t"][:, :S_DIM])
            S_pred_standardized = torch.cat([E_pred_std, Phi_t_std, H_pred_std], dim=1)

            # Inverse transform to get interpretable metrics
            s_pred_np = S_pred_standardized.cpu().numpy()
            s_pred_df = pd.DataFrame(s_pred_np, columns=feature_names)
            S_pred_original = scaler.inverse_transform(s_pred_df)
            E_pred_original = S_pred_original[:, E_INDEX[0]:E_INDEX[1]+1]

            for i in range(S_pred_standardized.shape[0]):
                dict_append = {
                    'batch_idx': batch_idx, 'sample_idx': i,
                    'pred_e_norm': np.linalg.norm(E_pred_original[i])
                }
                for idx, col in enumerate(STATE_COLS_DEFAULT):
                    if col.startswith("error_"):
                        error_pref = col.lstrip("error_")
                        dict_append[f"pred_e_{error_pref}"] = E_pred_original[i][idx]
                all_data.append(dict_append)
    
    df = pd.DataFrame(all_data)
    if df.empty:
        return {'EMPTY RESULT': 'EMPTY_RESULT'}
        
    return {
        'max_e_norm': df['pred_e_norm'].max(),
        'percentile_99_e_norm': df['pred_e_norm'].quantile(0.99),
        'dataframe': df
    }

def s8_multistep_decay_validation(
    model, 
    loader, 
    V_fn: Callable[[torch.Tensor], torch.Tensor], 
    rho: float, 
    horizon: int = 20,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Validates exponential decay of the Lyapunov function over a multi-step rollout.
    """
    model.eval()
    all_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = to_device(batch, device)
            S_t = batch["S_t"]; S_hist = batch["S_hist"]
            A_t = batch["A_t"]; A_hist = batch["A_hist"]
            source_embeddings = batch.get("source_embeddings")
            prefix_embeddings = batch.get("prefix_embeddings")
            decoder_hidden_state = batch.get("decoder_hidden_state")
            
            V_trajectory = [V_fn(S_t)]
            S_curr = S_t; S_hist_curr = S_hist; A_hist_curr = A_hist
            
            for step in range(horizon):
                A_curr = A_t if step == 0 else torch.zeros_like(A_t)
                _, Phi_curr, _ = split_state(S_curr)

                model_output = model(
                    S_curr, S_hist_curr, A_curr, A_hist_curr,
                    decoder_hidden_state=decoder_hidden_state,
                    source_embeddings=source_embeddings, 
                    prefix_embeddings=prefix_embeddings
                )

                E_next, H_next, _ = model_output
                S_next = torch.cat([E_next, Phi_curr, H_next], dim=1)
                
                V_trajectory.append(V_fn(S_next))

                if S_hist_curr.shape[1] > 0:
                    S_hist_curr = torch.cat([S_hist_curr[:, 1:, :], S_curr.unsqueeze(1)], dim=1)
                    A_hist_curr = torch.cat([A_hist_curr[:, 1:, :], A_curr.unsqueeze(1)], dim=1) 
                S_curr = S_next

            V_trajectory = torch.stack(V_trajectory, dim=1)
            
            for i in range(S_t.shape[0]):
                V_0 = V_trajectory[i, 0].item()
                
                for step in range(horizon):
                    V_t_val = V_trajectory[i, step].item()
                    V_tp1_val = V_trajectory[i, step + 1].item()
                    
                    theoretical_V = V_0 * ((1 - rho) ** (step + 1))
                    actual_decay = V_tp1_val / (V_t_val + 1e-8) if V_t_val > 1e-8 else 0
                    
                    all_data.append({
                        'batch_idx': batch_idx, 'sample_idx': i, 'step': step,
                        'V_t': V_t_val, 'V_tp1': V_tp1_val, 'V_0': V_0,
                        'theoretical_V': theoretical_V, 'actual_decay_rate': actual_decay,
                        'theoretical_decay_rate': 1 - rho,
                        'cumulative_actual': V_tp1_val / (V_0 + 1e-8) if V_0 > 1e-8 else 0,
                        'cumulative_theoretical': (1 - rho) ** (step + 1),
                        'monotonic': V_tp1_val <= V_t_val
                    })
    
    if not all_data:
        return {'monotonic_rate': 0.0, 'mean_decay_deviation': 0.0, 'cumulative_deviation': 0.0, 'dataframe': pd.DataFrame()}

    df = pd.DataFrame(all_data)
    
    return {
        'monotonic_rate': df['monotonic'].mean(),
        'mean_decay_deviation': (df['actual_decay_rate'] - df['theoretical_decay_rate']).abs().mean(),
        'cumulative_deviation': (df['cumulative_actual'] - df['cumulative_theoretical']).abs().mean(),
        'dataframe': df
    }

def s8_run_validation_suite(
    model, val_loader, scaler: Pipeline,
    V_fn: Callable[[torch.Tensor], torch.Tensor], 
    args, epoch, save_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    Executes all S8 validation tests and saves results.
    Acts as a comprehensive health check for the trained model's physics.
    """
    
    results = {}
    csv_dir = save_dir / f"s8_validation_epoch_{epoch}"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Robust Jacobian validation
    if args.s8_jacobian_robust:
        print(f"[Info] S8: Running robust Jacobian validation...")
        jac_val = s8_robust_spectral_norm_jacobian(
            model,
            val_loader.dataset.S_t, val_loader.dataset.S_hist,
            val_loader.dataset.A_t, val_loader.dataset.A_hist,
            sample_size=args.s8_jacobian_samples,
            num_random_starts=args.s8_jacobian_restarts, 
            source_embeddings=val_loader.dataset.source_embeddings, 
            prefix_embeddings=val_loader.dataset.prefix_embeddings,
            decoder_hidden_state=val_loader.dataset.decoder_hidden_states,
            device=args.device
        )
        flat_results = [];
        # Check if 'all_results' exists and is not empty
        if jac_val and 'all_results' in jac_val and jac_val['all_results']:
            for restart_data in jac_val['all_results']:
                for i, val in enumerate(restart_data['final_values']):
                    flat_results.append({
                        'restart_idx': restart_data['restart_idx'], 'converged': restart_data['converged'],
                        'iterations': restart_data['iterations'], 'sample_idx_in_batch': i,
                        'final_sigma': val
                    })
            pd.DataFrame(flat_results).to_csv(csv_dir / "jacobian_validation.csv", index=False)
            jacobian_summary = {k: v for k, v in jac_val.items() if k != 'all_results'}
            results['jacobian'] = jacobian_summary
            print(f"[Info] S8: Jacobian Validation Done. Mean: {results['jacobian'].get('mean', float('nan')):.4f}, Max: {results['jacobian'].get('max', float('nan')):.4f}")
        else:
            print("[Info] S8: Jacobian validation did not return expected results.")
            results['jacobian'] = {}


    # 2. Lyapunov validation
    if args.s8_lyapunov_full:
        print(f"[Info] S8: Running Lyapunov validation...")
        lyap_val = s8_lyapunov_validation(
            model, val_loader, V_fn, args.rho,
            device=args.device
        )
        lyap_val['dataframe'].to_csv(csv_dir / "lyapunov_validation.csv", index=False)
        results['lyapunov'] = {k: v for k, v in lyap_val.items() if k != 'dataframe'}
    
    # 3. CBF invariance validation
    if args.s8_cbf_invariance and args.lambda_cbf > 0:
        print(f"[Info] S8: Running CBF invariance validation...")
        cbf_val = s8_cbf_invariance_validation(
            model, val_loader,
            args.phi_crit, args.cbf_alpha, horizon=args.s8_cbf_horizon,
            device=args.device
        )
        cbf_val['dataframe'].to_csv(csv_dir / "cbf_invariance.csv", index=False)
        results['cbf'] = {k: v for k, v in cbf_val.items() if k != 'dataframe'}
    
    # 4. Error boundedness
    if args.s8_error_bounds:
        print(f"[Info] S8: Running error boundedness validation...")
        error_val = s8_error_boundedness_validation(
            model, val_loader, scaler, device=args.device
        )
        error_val['dataframe'].to_csv(csv_dir / "error_bounds.csv", index=False)
        results['error_bounds'] = {k: v for k, v in error_val.items() if k != 'dataframe'}
    
    # 5. Multi-step decay
    if args.s8_multistep_decay:
        print(f"[Info] S8: Running multi-step decay validation...")
        decay_val = s8_multistep_decay_validation(
            model, val_loader, V_fn, args.rho,
            horizon=args.s8_multistep_horizon,
            device=args.device
        )
        decay_val['dataframe'].to_csv(csv_dir / "multistep_decay.csv", index=False)
        results['multistep'] = {k: v for k, v in decay_val.items() if k != 'dataframe'}
    
    # Save summary
    summary_path = csv_dir / "summary.json"
    
    def convert_numpy_types(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    serializable_results = convert_numpy_types(results)
    
    with open(summary_path, 'w') as f:
        json.dump({'epoch': epoch, 'results': serializable_results}, f, indent=2)
    print(f"[Info] S8: Validation suite finished. Results saved in {csv_dir}")
    return serializable_results

# -----------------------------
# Main Function
# -----------------------------
def robust_log_writer(log_data: dict, log_path: Path, cache_path: Path):
    """
    Robustly writes a log entry to a jsonl file, with a caching mechanism for failures.
    Ensures that training logs are preserved even if network I/O fails transiently.
    """
    log_line = json.dumps(log_data, ensure_ascii=False) + "\n"
    
    try:
        # First, try to flush any cached entries
        if cache_path.exists() and cache_path.stat().st_size > 0:
            with open(cache_path, 'r', encoding='utf-8') as cache_f:
                cached_content = cache_f.read()
            
            with open(log_path, 'a', encoding='utf-8') as log_f:
                log_f.write(cached_content)
            
            # Clear cache after successful flush
            with open(cache_path, 'w', encoding='utf-8') as cache_f:
                cache_f.truncate(0)
            print(f"[Info] Successfully flushed cache from {cache_path} to {log_path}")

        # Now, write the current log line
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_line)

    except (IOError, OSError) as e:
        print(f"[Warning] Failed to write to {log_path}: {e}. Caching log entry to {cache_path}.")
        try:
            with open(cache_path, 'a', encoding='utf-8') as cache_f:
                cache_f.write(log_line)
        except (IOError, OSError) as cache_e:
            print(f"[Error] CRITICAL: Failed to write to cache file {cache_path} as well: {cache_e}")

def get_train_path_base(
    args, train_key:str="train_path", cut_num_key:str="train_path_base_cut_num"
) -> str:
    """
    Extracts the base directory path for training data to help with caching organization.
    """
    args_dict = vars(args)
    train_path = args_dict[train_key].rstrip('/')
    cut_num = args_dict[cut_num_key]
    if cut_num <= 0 or not train_path: return train_path
    parts = train_path.split('/')
    separator_count = len(parts) - 1
    if separator_count <= 1: return train_path
    result = '/'.join(parts[:cut_num + 1])
    return result
    
def _compute_file_hash(file_path: str) -> str:
    """Computes MD5 hash of a file for cache invalidation."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def data_cache_and_split(
    main_df: pd.DataFrame, val_df_external: Optional[pd.DataFrame],
    text_embedder: SentenceTransformer,
    sbert_model_name: str,
    save_dir: Path, 
    cache_path_base: str,
    source_csv_path: str, val_csv_path: Optional[str],
    val_ratio: float,
    seed: int,
    use_decoder_hidden_state: bool,
    history_len: int,
    train_path_base_cut_num: int = 2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    cox_event_threshold: float = 0.1,
    action_dim: int = 6,
    state_cols_default: List[str] = STATE_COLS_DEFAULT,
    s_dim: int = S_DIM
) -> Dict[str, Any]:
    """
    Unified data processing function for both t_train_Transformer.py and 05_train_policy_network.py.
    
    This function:
    1. Computes cache fingerprint based on config and data source.
    2. Loads cached data if available; otherwise processes CSVs and computes embeddings.
    3. Handles train/validation splitting (using indices or group IDs).
    4. Fits and saves a new Scaler if training, or uses existing if cached.
    5. Returns processed datasets and scaler.
    """
    
    print("[data_cache_and_split] Starting unified data processing...")
    
    # --- 1. COMPUTE CACHE FINGERPRINT ---
    text_embedding_dim = text_embedder.get_sentence_embedding_dimension()
    
    config_fingerprint = {
        "sbert_model": sbert_model_name,
        "train_csv_hash": _compute_file_hash(source_csv_path),
        "val_csv_hash": _compute_file_hash(val_csv_path) if val_csv_path else None,
        "val_path": val_csv_path,
        "val_ratio": val_ratio,
        "seed": seed,
        "use_decoder_hidden_state": use_decoder_hidden_state,
        "history_len": history_len
    }
    
    fingerprint_str = json.dumps(config_fingerprint, sort_keys=True)
    current_config_hash = hashlib.sha256(fingerprint_str.encode('utf-8')).hexdigest()
    
    # --- 2. SETUP CACHE PATHS ---
    input_path = Path(source_csv_path)
    # Construct cache directory path
    parts = input_path.parent.parts
    if len(parts) > train_path_base_cut_num:
        relative_path = Path(*parts[train_path_base_cut_num:])
    else:
        relative_path = Path('.')
    
    cache_dir = Path(cache_path_base) / relative_path / current_config_hash[:35]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "data_cache_all.npz"
    metadata_file = cache_dir / "metadata.json"
    
    # --- 3. CHECK CACHE VALIDITY ---
    is_cache_valid = False
    if cache_file.exists() and metadata_file.exists():
        with open(metadata_file, 'r') as f:
            saved_fingerprint = json.load(f)
        if saved_fingerprint == config_fingerprint:
            is_cache_valid = True
            print(f"[Cache] Found valid cache at {cache_file}")
        else:
            print("[Cache] Found outdated cache. Regenerating data...")
    else:
        print("[Cache] No valid cache found. Generating data and embeddings...")
    
    # --- 4. LOAD OR GENERATE ALL DATA ---
    if is_cache_valid:
        print(f"[Cache] Loading all processed data from {cache_file}...")
        with np.load(cache_file, allow_pickle=True) as data:
            S_t_all = data['S_t_all']
            S_hist_all = data['S_hist_all']
            A_all = data['A_all']
            A_hist_all = data['A_hist_all']
            S_tp1_all = data['S_tp1_all']
            seq_id_all = data['seq_id_all']
            T_all = data['T_all']
            E_all = data['E_all']
            source_texts_all = data['source_texts_all'].tolist()
            prefixes_all = data['prefixes_all'].tolist()
            hs_all = data['hs_all'] if 'hs_all' in data else None
            src_emb_all = data['src_emb_all'] if 'src_emb_all' in data else None
            pref_emb_all = data['pref_emb_all'] if 'pref_emb_all' in data else None
        print(f"[Cache] Loaded {len(S_t_all)} cached samples")
    else:
        # Generate data from CSV
        print("[Processing] Building data pairs from CSV...")
        S_t_all, S_hist_all, \
        A_all, A_hist_all, \
        S_tp1_all, seq_id_all, T_all, E_all, \
        source_texts_all, prefixes_all, hs_all = _build_pairs_from_df(
            main_df,
            history_len=history_len,
            action_dim=action_dim,
            event_threshold=cox_event_threshold,
            use_decoder_hidden_state=use_decoder_hidden_state
        )
        
        # Compute text embeddings
        print("[Cache] Pre-computing text embeddings for ALL data...")
        src_emb_all  = text_embedder.encode(source_texts_all, show_progress_bar=True, device=device)
        pref_emb_all = text_embedder.encode(prefixes_all,     show_progress_bar=True, device=device)
        
        # Save cache
        print(f"[Cache] Saving processed data (ALL) to {cache_file}...")
        cache_save_dict = {
            'S_t_all': S_t_all, 'S_hist_all': S_hist_all, 'S_tp1_all': S_tp1_all,
            'A_all': A_all, 'A_hist_all': A_hist_all,
            'seq_id_all': seq_id_all,
            'T_all': T_all, 'E_all': E_all,
            'source_texts_all': np.array(source_texts_all),
            'prefixes_all': np.array(prefixes_all),
        }
        if hs_all is not None:       cache_save_dict['hs_all']       = hs_all
        if src_emb_all is not None:  cache_save_dict['src_emb_all']  = src_emb_all
        if pref_emb_all is not None: cache_save_dict['pref_emb_all'] = pref_emb_all
        
        np.savez_compressed(cache_file, **cache_save_dict)
        with open(metadata_file, 'w') as f:
            json.dump(config_fingerprint, f, indent=2)
        print("[Cache] Cache saved successfully.")
    
    # --- 5. HANDLE SPLIT INDICES ---
    split_indices_path = save_dir / "split_indices.npz"
    use_saved_split = split_indices_path.exists()
    
    if val_df_external is not None:
        # External validation set provided
        print("[Split] Using external validation set...")
        train_mask = np.ones(len(S_t_all), dtype=bool)  # All data is training
        val_mask = np.zeros(len(S_t_all), dtype=bool)  # No validation from main data
        
        # Process external validation data
        print("[Processing] Building validation pairs from external CSV...")
        S_t_va, S_hist_va, A_va, A_hist_va, S_tp1_va, seq_va, T_va, E_va, src_va, pref_va, hs_va = _build_pairs_from_df(
            val_df_external,
            history_len=history_len,
            action_dim=action_dim,
            event_threshold=cox_event_threshold,
            use_decoder_hidden_state=use_decoder_hidden_state
        )
        
        # Use all main data as training
        S_t_tr, S_hist_tr, A_tr, A_hist_tr, S_tp1_tr, seq_tr, T_tr, E_tr = (
            S_t_all, S_hist_all, A_all, A_hist_all, S_tp1_all, seq_id_all, T_all, E_all
        )
        src_tr, pref_tr, hs_tr = source_texts_all, prefixes_all, hs_all
        
        # Compute embeddings for training data (use cached if available)
        if src_emb_all is not None and pref_emb_all is not None:
            print("[Cache] Using cached embeddings for training data...")
            src_emb_tr = src_emb_all
            pref_emb_tr = pref_emb_all
        else:
            print("[Cache] Pre-computing text embeddings for training data...")
            src_emb_tr = text_embedder.encode(src_tr, show_progress_bar=True, device=device)
            pref_emb_tr = text_embedder.encode(pref_tr, show_progress_bar=True, device=device)
        
        # Compute validation embeddings (always compute for external validation)
        print("[Cache] Pre-computing text embeddings for external validation set...")
        src_emb_va = text_embedder.encode(src_va, show_progress_bar=True, device=device)
        pref_emb_va = text_embedder.encode(pref_va, show_progress_bar=True, device=device)
        
    else:
        # Use split indices (load existing or create new)
        if use_saved_split:
            print(f"[Split] Found and loading existing split indices from {split_indices_path}")
            split_data = np.load(split_indices_path, allow_pickle=True)
            train_mask = split_data['train_mask']
            val_mask = split_data['val_mask']
            
            # Validate mask length to ensure compatibility
            if len(train_mask) != len(seq_id_all) or len(val_mask) != len(seq_id_all):
                raise ValueError(
                    f"Saved split masks length mismatch! "
                    f"train_mask: {len(train_mask)}, val_mask: {len(val_mask)}, "
                    f"current data: {len(seq_id_all)}. "
                    f"Delete {split_indices_path} and retrain."
                )
            
            print(f"[Split] Using saved split: {train_mask.sum()} train, {val_mask.sum()} val samples")
        else:
            # Create new split based on sequence groups to avoid data leakage
            print("[Split] Creating new split indices based on sequence groups...")
            unique_group_ids = np.unique(seq_id_all)
            print(f"Found {len(unique_group_ids)} unique groups/sequences.")
            
            # Shuffle group IDs
            np.random.seed(seed)
            np.random.shuffle(unique_group_ids)
            
            # Calculate split point
            n_groups = len(unique_group_ids)
            
            if n_groups == 1 and val_ratio > 0:
                # Edge case: single group. Use for both train and val.
                print(f"[Split] Warning: Only 1 group found. Using it for BOTH train and val.")
                train_groups = unique_group_ids 
                val_groups = unique_group_ids   
                
            elif n_groups > 1:
                # Standard split logic
                val_group_count = max(1, int(n_groups * val_ratio))
                train_cutoff = n_groups - val_group_count
                train_groups = unique_group_ids[:train_cutoff]
                val_groups = unique_group_ids[train_cutoff:]
            else:
                # n_groups == 0
                train_groups = np.array([])
                val_groups = np.array([])
            
            print(f"Split: {len(train_groups)} train groups, {len(val_groups)} val groups")
            
            # Create masks based on group IDs
            train_mask = np.isin(seq_id_all, train_groups)
            val_mask = np.isin(seq_id_all, val_groups)
            
            # Save split indices
            print(f"[Split] Saving split indices to {split_indices_path}")
            np.savez_compressed(
                split_indices_path,
                train_mask=train_mask, val_mask=val_mask,
                train_groups=train_groups, val_groups=val_groups,
                val_ratio=val_ratio,
                n_total=len(seq_id_all), n_groups=n_groups
            )
            print("[Split] Split indices saved successfully.")
        
        # Apply masks to split data
        print("[Split] Applying masks to split ALL data into train/val...")
        S_t_tr, S_hist_tr, A_tr, A_hist_tr, S_tp1_tr, seq_tr, T_tr, E_tr = (
            S_t_all[train_mask], S_hist_all[train_mask], A_all[train_mask],
            A_hist_all[train_mask], S_tp1_all[train_mask], seq_id_all[train_mask],
            T_all[train_mask], E_all[train_mask]
        )
        S_t_va, S_hist_va, A_va, A_hist_va, S_tp1_va, seq_va, T_va, E_va = (
            S_t_all[val_mask], S_hist_all[val_mask], A_all[val_mask],
            A_hist_all[val_mask], S_tp1_all[val_mask], seq_id_all[val_mask],
            T_all[val_mask], E_all[val_mask]
        )
        
        # Convert arrays to lists for text data
        source_texts_all_arr = np.array(source_texts_all)
        prefixes_all_arr     = np.array(prefixes_all)
        
        src_tr, pref_tr = source_texts_all_arr[train_mask].tolist(), prefixes_all_arr[train_mask].tolist()
        src_va, pref_va = source_texts_all_arr[val_mask].tolist(),   prefixes_all_arr[val_mask].tolist()
        
        # Split embeddings and hidden states
        if use_decoder_hidden_state and hs_all is not None:
            hs_tr, hs_va = hs_all[train_mask], hs_all[val_mask]
        else:
            hs_tr = hs_va = None
        
        if src_emb_all is not None and pref_emb_all is not None:
            src_emb_tr,  src_emb_va = src_emb_all[train_mask],  src_emb_all[val_mask]
            pref_emb_tr, pref_emb_va = pref_emb_all[train_mask], pref_emb_all[val_mask]
        else:
            # Recompute embeddings if not in cache (shouldn't happen with new cache logic)
            print("[Cache] Computing text embeddings for train/val splits...")
            src_emb_tr, pref_emb_tr = \
                text_embedder.encode(src_tr,  show_progress_bar=True, device=device), \
                text_embedder.encode(pref_tr, show_progress_bar=True, device=device)
            src_emb_va, pref_emb_va = \
                text_embedder.encode(src_va,  show_progress_bar=True, device=device), \
                text_embedder.encode(pref_va, show_progress_bar=True, device=device)
    
    # --- 6. HANDLE SCALER ---
    scaler_path = save_dir / "scaler.joblib"
    use_saved_scaler = scaler_path.exists()
    
    # Define transform configuration
    transform_config = {
        "standard": sorted(
            [c for c in TRANSFORM_COLS["standard"] if c in state_cols_default],
            key=lambda x: state_cols_default.index(x)
        ),
        # "power": sorted(
        #     [c for c in TRANSFORM_COLS["power"] if c in state_cols_default],
        #     key=lambda x: state_cols_default.index(x)
        # ),
        "quantile": sorted(
            [c for c in TRANSFORM_COLS["quantile"] if c in state_cols_default],
            key=lambda x: state_cols_default.index(x)
        ),
    }
    
    col_to_idx = {col: idx for idx, col in enumerate(state_cols_default)}
    transform_config_idx = {
        "standard": [col_to_idx[c] for c in transform_config["standard"]],
        # "power": [col_to_idx[c] for c in transform_config["power"]],
        "quantile": [col_to_idx[c] for c in transform_config["quantile"]]
    }
    
    # Configure the preprocessor pipeline
    preprocessor = InvertableColumnTransformer(
        transformers=[
            ('standard', StandardScaler(), transform_config_idx["standard"]),
            # ('power', PowerTransformer(method='yeo-johnson', standardize=False), transform_config_idx["power"]),
            ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=max(1, min(len(S_t_tr) // 10, 1000))), transform_config_idx["quantile"])
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    scaler = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean', add_indicator=True)),
        ('preprocessor', preprocessor)
    ])
    
    if use_saved_scaler:
        print(f"[Scaler] Found and loading existing scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        print("[Scaler] Loaded successfully.")
    else:
        # Fit on training data only
        print(f"[Scaler] Fitting scaler on training data (shape: {S_t_tr[:, :s_dim].shape})")
        scaler.fit(pd.DataFrame(S_t_tr[:, :s_dim], columns=state_cols_default))
        
        # Save scaler immediately
        print(f"[Scaler] Saving fitted scaler to {scaler_path}")
        joblib.dump(scaler, scaler_path)
        print("[Scaler] Scaler saved successfully.")
    
    # --- 7. RETURN RESULTS ---
    return {
        # Training data
        'S_t_tr': S_t_tr, 'S_hist_tr': S_hist_tr, 'S_tp1_tr': S_tp1_tr,
        'A_tr': A_tr, 'A_hist_tr': A_hist_tr,
        'seq_tr': seq_tr,
        'T_tr': T_tr, 'E_tr': E_tr,
        'src_tr': src_tr, 'pref_tr': pref_tr,
        'hs_tr': hs_tr, 'src_emb_tr': src_emb_tr, 'pref_emb_tr': pref_emb_tr,
        # Validation data
        'S_t_va': S_t_va, 'S_hist_va': S_hist_va, 'S_tp1_va': S_tp1_va,
        'A_va': A_va, 'A_hist_va': A_hist_va,
        'seq_va': seq_va,
        'T_va': T_va, 'E_va': E_va,
        'src_va': src_va, 'pref_va': pref_va,
        'hs_va': hs_va, 'src_emb_va': src_emb_va, 'pref_emb_va': pref_emb_va,
        # Scaler and metadata
        'scaler': scaler,
        'text_embedding_dim': text_embedding_dim
    }

def setup_data_and_model_components(args, rng):
    """
    Refactored helper to set up all common data and model objects for training.
    
    This function initializes:
    1. SentenceTransformer for text embeddings.
    2. DataLoaders (via data_cache_and_split).
    3. The PAEC Dynamics Model (T_theta).
    4. The Optimizer (AdamW).
    5. The Lyapunov function (V_fn) and P matrix.
    6. Learning Rate Schedulers.

    Args:
        args: ArgumentNamespace containing all config parameters.
        rng: Random number generator for reproducibility.

    Returns:
        dict: A dictionary containing all initialized components.
    """

    # --- Setup SentenceTransformer ---
    print("[Info] Loading SentenceTransformer model for text embeddings...")
    sbert_model_name = 'sentence-transformers/LaBSE'
    if MODEL_NAMES and isinstance(MODEL_NAMES, dict):
        sbert_model_name = MODEL_NAMES.get("sentence_encoder", 'sentence-transformers/LaBSE')
    text_embedder = SentenceTransformer(sbert_model_name, device=args.device)
    print(f"[Info] SentenceTransformer model '{sbert_model_name}' loaded.")

    # --- Load Data ---
    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"Invalid training set path: {args.train_path}")
    main_df = pd.read_csv(args.train_path).drop(columns=args.skip_cols, errors='ignore')
    val_df_external = None
    if args.val_path and os.path.exists(args.val_path):
        val_df_external = pd.read_csv(args.val_path)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- USE UNIFIED DATA PROCESSING FUNCTION ---
    print("[Info] Using unified data cache and split function...")
    data_results = data_cache_and_split(
        main_df=main_df, val_df_external=val_df_external,
        text_embedder=text_embedder,
        sbert_model_name=sbert_model_name,
        save_dir=save_dir,
        cache_path_base=args.cache_path,
        source_csv_path=args.train_path,
        val_csv_path=args.val_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_decoder_hidden_state=args.use_decoder_hidden_state,
        history_len=args.history_len,
        train_path_base_cut_num=args.train_path_base_cut_num,
        device=args.device,
        cox_event_threshold=args.cox_event_threshold,
        action_dim=6,
        state_cols_default=STATE_COLS_DEFAULT,
        s_dim=S_DIM
    )
    
    # Extract results for Datasets
    S_t_tr, S_hist_tr, S_tp1_tr = data_results['S_t_tr'], data_results['S_hist_tr'], data_results['S_tp1_tr']
    A_tr, A_hist_tr = data_results['A_tr'], data_results['A_hist_tr']
    seq_tr = data_results['seq_tr']
    T_tr, E_tr = data_results['T_tr'], data_results['E_tr']
    src_tr, pref_tr = data_results['src_tr'], data_results['pref_tr']
    hs_tr, src_emb_tr, pref_emb_tr = data_results['hs_tr'], data_results['src_emb_tr'], data_results['pref_emb_tr']
    
    S_t_va, S_hist_va, S_tp1_va = data_results['S_t_va'], data_results['S_hist_va'], data_results['S_tp1_va']
    A_va, A_hist_va = data_results['A_va'], data_results['A_hist_va']
    seq_va = data_results['seq_va']
    T_va, E_va = data_results['T_va'], data_results['E_va']
    src_va, pref_va = data_results['src_va'], data_results['pref_va']
    hs_va, src_emb_va, pref_emb_va = data_results['hs_va'], data_results['src_emb_va'], data_results['pref_emb_va']
    
    scaler = data_results['scaler']
    text_embedding_dim = data_results['text_embedding_dim']
    
    print(f"[Info] Unified processing complete. Text embedding dim: {text_embedding_dim}")
    print(f"[Info] Train samples: {len(S_t_tr)}, Val samples: {len(S_t_va)}")

    # --- SCALING ---
    # Apply scaler to all state tensors (Current, History, Target)
    print("[Processing] Scaling data arrays...")
    
    S_t_tr_scaled = scaler.transform(pd.DataFrame(S_t_tr, columns=STATE_COLS_DEFAULT))
    S_t_va_scaled = scaler.transform(pd.DataFrame(S_t_va, columns=STATE_COLS_DEFAULT))
    
    # Reshape history to 2D for scaling, then back to 3D
    S_hist_tr_scaled = scaler.transform(pd.DataFrame(S_hist_tr.reshape(-1, S_hist_tr.shape[2]), columns=STATE_COLS_DEFAULT)).reshape(S_hist_tr.shape)
    S_hist_va_scaled = scaler.transform(pd.DataFrame(S_hist_va.reshape(-1, S_hist_va.shape[2]), columns=STATE_COLS_DEFAULT)).reshape(S_hist_va.shape)

    S_tp1_tr_scaled = scaler.transform(pd.DataFrame(S_tp1_tr, columns=STATE_COLS_DEFAULT))
    S_tp1_va_scaled = scaler.transform(pd.DataFrame(S_tp1_va, columns=STATE_COLS_DEFAULT))

    # --- DATASETS AND DATALOADERS ---
    train_ds = PAECDataset(
        S_t_tr_scaled, S_hist_tr_scaled, A_tr, A_hist_tr, S_tp1_tr_scaled,
        T=T_tr, E=E_tr,
        source_embeddings=src_emb_tr, prefix_embeddings=pref_emb_tr,
        decoder_hidden_states=hs_tr, seq_id=seq_tr,
        source_texts=src_tr, generated_prefixes=pref_tr
    )
    val_ds = PAECDataset(
        S_t_va_scaled, S_hist_va_scaled, A_va, A_hist_va, S_tp1_va_scaled,
        T_va, E_va,
        source_embeddings=src_emb_va, prefix_embeddings=pref_emb_va,
        decoder_hidden_states=hs_va, seq_id=seq_va,
        source_texts=src_va, generated_prefixes=pref_va
    )
    
    # Persistent workers improve throughput for small batches
    use_persistent_workers = args.num_workers > 0 and not sys.platform.startswith('win')
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(args.device == 'cuda'),
        persistent_workers=use_persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=(args.device == 'cuda'),
        persistent_workers=use_persistent_workers
    )
    
    action_dim = train_ds.A_t.shape[1]
    
    # --- Model ---
    model = PAECTransition(
        action_dim=int(action_dim),
        hid_dim=args.hid_dim,
        layers=args.layers,
        history_len=args.history_len,
        predict_delta=args.predict_delta,
        use_text_embeddings=args.use_text_embeddings,
        text_embedding_dim=(text_embedding_dim or 0),
        use_decoder_hidden_state=args.use_decoder_hidden_state,
        decoder_hidden_state_dim=args.decoder_hidden_state_dim,
        use_moe_heads=args.use_moe_heads,
        use_multi_heads=args.use_multi_heads,
        use_spectral_norm=args.use_spectral_norm,
        nhead=args.nhead
    ).to(args.device)
    print("\n[Info] Model initialized in {E,H} MODE: Phi as condition (predicts E+H only).")
        
    
    # --- V_fn and P Parameter ---
    # Add fused=True if on CUDA and supported.
    # This fuses multiple optimizer operations into a single GPU kernel for speed.
    use_fused = args.device == 'cuda'
    print(f"[Info] Fused optimizer {'enabled' if use_fused else 'disabled'}.")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        fused=use_fused
    )
    
    # Initialize P matrix (weights for Lyapunov function components)
    P_init = torch.tensor(args.P_init, dtype=torch.float32, device=args.device)
    P_param = None
    if args.learn_P:
        # Learnable P matrix
        if len(P_init) != E_DIM:
            print(f"[Warning] --P_init has {len(P_init)} elements, but {E_DIM} are needed for V(E). Expanding with 1.0's...")
            P_init_expanded = torch.ones(E_DIM, device=args.device)
            P_init_expanded[:len(P_init)] = P_init
            P_init = P_init_expanded

        # Use log-space parameterization to ensure P remains positive definite via softplus
        P_param = nn.Parameter(torch.log(torch.expm1(P_init) + 1e-8))
        optimizer.add_param_group({"params": [P_param], "lr": args.lr})
        def V_fn(S: torch.Tensor) -> torch.Tensor:
            # V_fn always operates on the base S_DIM features
            S_base = S[:, :S_DIM]
            P = F.softplus(P_param) + 1e-6
            return lyapunov_V(S_base, P)
    else:
        # Fixed P matrix
        if len(P_init) != E_DIM:
            print(f"[Warning] --P_init has {len(P_init)} elements, but {E_DIM} are needed for V(E). Expanding with 1.0's...")
            P_init_expanded = torch.ones(E_DIM, device=args.device)
            P_init_expanded[:len(P_init)] = P_init
            P_init = P_init_expanded

        P_fixed = P_init.detach()
        def V_fn(S: torch.Tensor) -> torch.Tensor:
            # V_fn always operates on the base S_DIM features
            S_base = S[:, :S_DIM]
            return lyapunov_V(S_base, P_fixed)
    
    # Setup uncertainty weighting for loss components
    log_vars_param = None
    if args.use_uncertainty_weights:
        print("[Info] Uncertainty weighting enabled. Creating learnable log_vars parameter.")
        # Dynamically set the dimension of log_vars_param based on the modeling mode.
        log_vars_dim = E_DIM + H_DIM
        log_vars_param = nn.Parameter(torch.zeros(log_vars_dim, device=args.device))
        optimizer.add_param_group({"params": [log_vars_param], "lr": args.lr})
        print(f"[Info] log_vars dimension set to: {log_vars_dim}")
    
    # Initialize LR Scheduler
    scheduler = None
    if args.use_lr_scheduler:
        if args.lr_scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=args.lr_scheduler_mode, 
                factor=args.lr_scheduler_factor, 
                patience=args.lr_scheduler_patience, 
                min_lr=args.lr_scheduler_min_lr
            )
        elif args.lr_scheduler_type == 'cosine':
            t_max = args.lr_scheduler_t_max if args.lr_scheduler_t_max else args.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=t_max, 
                eta_min=args.lr_scheduler_eta_min
            )
        elif args.lr_scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=args.lr_scheduler_step_size, 
                gamma=args.lr_scheduler_gamma
            )

    return {
        "train_loader": train_loader, "val_loader": val_loader,
        "scaler": scaler, "model": model, "V_fn": V_fn,
        "P_param": P_param, "P_init": P_init, "optimizer": optimizer, "action_dim": action_dim,
        "scheduler": scheduler, "text_embedder": text_embedder,
        "log_vars_param": log_vars_param
    }

def check_training_configs_match(saved_args: dict, current_args: argparse.Namespace) -> bool:
    """
    Checks if the current run configuration matches a previously saved configuration.
    Ignores keys that do not affect the model's structural definition (e.g., num_workers, s8 flags).

    Args:
        saved_args (dict): The configuration dictionary loaded from disk.
        current_args (argparse.Namespace): The current runtime arguments.

    Returns:
        bool: True if configurations match for resuming training, False otherwise.
    """
    current_args_dict = vars(current_args)
    
    # Keys that do NOT define the trained model's identity/weights
    non_training_keys = {
        's8_enable', 's8_only', 's8_jacobian_robust', 's8_jacobian_samples',
        's8_jacobian_restarts', 's8_lyapunov_full',
        's8_cbf_invariance', 's8_error_bounds',
        's8_multistep_decay', 's8_multistep_horizon', 's8_cbf_horizon', 's8_use_last_ckpt'
        
        'export_action_stats', 'export_rollout_csv', 'num_workers', 'cache_path',
        'disable_autocast', 'disable_compile', 'device', 'save_dir', 'train_path_base_cut_num',
        'earlystop_monitor_after_epoch'
    }

    # Programmatically generate the set of training keys from current args
    training_keys = {k for k in current_args_dict if k not in non_training_keys}
    saved_training_keys = {k for k in saved_args if k not in non_training_keys}

    # 1. Check for added or removed keys
    if training_keys != saved_training_keys:
        added = training_keys - saved_training_keys
        removed = saved_training_keys - training_keys
        print("[Info] Config mismatch due to added/removed training keys.")
        if added: print(f"  - Added to current config: {added}")
        if removed: print(f"  - Removed from current config: {removed}")
        return False
    
    # 2. Check for value mismatches
    mismatched_keys = []
    for key in training_keys:
        # Correctly handle boolean flags (like store_true) by providing a default
        is_bool_flag = isinstance(current_args_dict.get(key), bool)
        
        saved_val = saved_args.get(key, False if is_bool_flag else None)
        current_val = current_args_dict.get(key)
    
        if saved_val != current_val:
            mismatched_keys.append({
                "key": key,
                "saved": saved_val,
                "current": current_val
            })
            
    if mismatched_keys:
        print("[Info] Config mismatch on training key values:")
        for item in mismatched_keys:
            print(f"  - Key: '{item['key']}' | Saved: {item['saved']} | Current: {item['current']}")
        return False

    return True

def argparse_list_str(arg) -> List[str]: return arg.split(',')

def main():
    """
    Main execution function.
    Handles argument parsing, experiment resumption, data loading, training loop,
    and validation execution.
    """
    global STATE_COLS_DEFAULT, E_INDEX, PHI_INDEX, H_INDEX, S_DIM, E_DIM, PHI_DIM, H_DIM
    
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--train_path", type=str, default=str(PATHS["processed_data_dir"] / "training_data_stepwise" / "strategy_comparison_stepwise_1000.csv"))
    parser.add_argument("--train_path_base_cut_num", type=int, default=2)
    parser.add_argument("--cache_path", type=str, default=str(PATHS["cache_dir"]))
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="When val_path is empty or invalid, the ratio of the validation set to the training set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=min(8, (os.cpu_count() or 1) // 2))
    parser.add_argument("--disable_autocast", action="store_true", help="Disable autocast of model training")
    parser.add_argument("--disable_compile", action="store_true", help="Disable compilation of model training")
    parser.add_argument("--target_total_steps", type=int, default=3000)

    # Models Arguments
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--hid_dim", type=int, default=64)
    model_group.add_argument("--layers", type=int, default=3)
    model_group.add_argument("--nhead", type=int, default=4, help="Number of attention heads for the Transformer.")
    model_group.add_argument("--history_len", type=int, default=4, help="Number of historical states to use as input for the sequence model.")
    model_group.add_argument("--predict_delta", action="store_true", help="Change the model to predict the state change (S_tp1 - S_t) instead of the full next state.")
    model_group.add_argument("--use_text_embeddings", action="store_true", help="Enhance model input with source and prefix text embeddings.")
    model_group.add_argument("--use_moe_heads", action="store_true", help="Use separate Mixture-of-Experts heads for predicting E, Phi, and H vectors.")
    model_group.add_argument("--use_multi_heads", action="store_true", help="Use Multi-Head transformer for predicting E, Phi, H vectors aligning with actions.")

    decoder_hidden_state_group = parser.add_argument_group("Hidden State of Decoder")
    decoder_hidden_state_group.add_argument("--use_decoder_hidden_state", action="store_true", help="Enhance model input with the NMT decoder's hidden state from the previous step.")
    decoder_hidden_state_group.add_argument("--decoder_hidden_state_dim", type=int, default=(BUILD_PIPELINE_SETTINGS or {}).get("decoder-embed-dim", 512), help="Dimensionality of the NMT decoder's hidden state.")

    # Training Arguments
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=str(PATHS["dynamics_model_dir"] / "current"), help="Output directory (checkpoints/logs/config/Scaler)")
    parser.add_argument("--use_uncertainty_weights", action="store_true", help=f"Use self-learning uncertainty to weigh the component losses with same dimension of S.")

    # CLF/Stability Arguments
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--lambda_clf", type=float, default=0.0)
    parser.add_argument("--jacobian_reg", type=float, default=0.0)
    parser.add_argument("--teacher_reweight_alpha", type=float, default=1.5)
    parser.add_argument("--softmin_tau", type=float, default=0.5)
    parser.add_argument("--rollout_H", type=int, default=12)
    parser.add_argument("--num_delta_dirs", type=int, default=2, help="The number of directions of the teacher action ± perturbation (<= dA)")
    parser.add_argument("--action_delta", type=float, default=0.25, help="The step size of the teacher action ± perturbation")

    # CBF Arguments
    parser.add_argument("--phi_crit", type=float, default=0.0, help="If it is positive, the CBF violation rate is counted; if it is zero or negative, it won't counted.")

    # P Matrix Learning Arguments
    P_group = parser.add_argument_group("P Matrix Learning")
    P_group.add_argument("--learn_P", action="store_true", help="If set, then learn V(E)=E^T P E in the P diagonal.")
    P_group.add_argument("--P_init", type=float, nargs='*', default=None, help=f"Initial values for the diagonal of P in V(E)=E^T P E. Expects values in same dimension of E.")
    parser.add_argument("--skip_cols", type=float, nargs='*', default=[], help=f"Droppable columns in train_data.")

    # Schedulers
    parser.add_argument("--rho_final", type=float, default=None)
    parser.add_argument("--rho_warmup_ep", type=int, default=10)
    parser.add_argument("--lambda_clf_final", type=float, default=None)
    parser.add_argument("--lambda_warmup_ep", type=int, default=12)
    parser.add_argument("--softmin_tau_final", type=float, default=None)
    parser.add_argument("--softmin_anneal_ep", type=int, default=12)

    # N-Step CLF
    nclf_group = parser.add_argument_group("N-step CLF")
    nclf_group.add_argument("--use_nstep_clf", action="store_true")
    nclf_group.add_argument("--nstep_H", type=int, default=3)
    nclf_group.add_argument("--nstep_gamma", type=float, default=0.98)
    nclf_group.add_argument("--nstep_lambda", type=float, default=1.0)
    nclf_group.add_argument("--nstep_bptt_window", type=int, default=-1)
    
    nclf_selector_group = parser.add_argument_group("N-step CLF Selector")
    nclf_selector_group.add_argument("--nstep_selector", type=str, default="softmin", choices=["softmin", "gumbel_st", "hard_greedy"])
    nclf_selector_group.add_argument("--gumbel_tau_init", type=float, default=1.0)
    nclf_selector_group.add_argument("--gumbel_tau_final", type=float, default=0.1)
    nclf_selector_group.add_argument("--gumbel_anneal_ep", type=int, default=15)
    
    cvar_group = parser.add_argument_group("N-step CLF")
    cvar_group.add_argument("--use_cvar_loss", action="store_true")
    cvar_group.add_argument("--cvar_alpha", type=float, default=0.8)
    
    epsilon_greedy_group = parser.add_argument_group("N-step CLF")
    epsilon_greedy_group.add_argument("--use_epsilon_greedy", action="store_true")
    epsilon_greedy_group.add_argument("--epsilon_init", type=float, default=0.3)
    epsilon_greedy_group.add_argument("--epsilon_final", type=float, default=0.01)
    epsilon_greedy_group.add_argument("--epsilon_decay_ep", type=int, default=15)
    epsilon_greedy_group.add_argument("--policy_entropy_weight", type=float, default=0.0)

    # Control Loss Groups
    cbf_group = parser.add_argument_group("CBF Control")
    cbf_group.add_argument("--lambda_cbf", type=float, default=0.0)
    cbf_group.add_argument("--cbf_alpha", type=float, default=0.5)
    
    cox_group = parser.add_argument_group("Cox Control")
    cox_group.add_argument("--lambda_cox", type=float, default=0.0)
    cox_group.add_argument("--cox_event_threshold", type=float, default=2.0)
    
    parser.add_argument("--lambda_adt", type=float, default=0.0)
    parser.add_argument("--use_spectral_norm", action="store_true")
    
    # S8 Validation Suite
    s8_group = parser.add_argument_group("S8: Theoretical Validation Suite")
    s8_group.add_argument("--s8_enable", action="store_true")
    s8_group.add_argument("--s8_only", action="store_true", help="Skip training and run S8 validation on a matching existing model.")
    s8_group.add_argument("--s8_jacobian_robust", action="store_true")
    s8_group.add_argument("--s8_lyapunov_full", action="store_true")
    s8_group.add_argument("--s8_cbf_invariance", action="store_true")
    s8_group.add_argument("--s8_error_bounds", action="store_true")
    s8_group.add_argument("--s8_multistep_decay", action="store_true")
    s8_group.add_argument("--s8_jacobian_samples", type=int, default=256)
    s8_group.add_argument("--s8_jacobian_restarts", type=int, default=5)
    s8_group.add_argument("--s8_cbf_horizon", type=int, default=10)
    s8_group.add_argument("--s8_multistep_horizon", type=int, default=20)
    s8_group.add_argument("--s8_use_last_ckpt", action="store_true")

    # LR Scheduler
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
    
    # Curriculum Learning
    curriculum_group = parser.add_argument_group("Curriculum Learning")
    curriculum_group.add_argument("--use_curriculum", action="store_true", help="Enable two-phase curriculum learning: prediction-first, then control.")
    curriculum_group.add_argument("--curriculum_phase1_epochs", type=int, default=10, help="Number of epochs for Phase 1 (prediction-only training).")
    
    # Early Stopping & Export
    parser.add_argument("--stab_lexi_eps", type=float, default=1e-4)
    parser.add_argument("--earlystop_mode", type=str, default="rmse", choices=["rmse", "stability_first"], help="Mode for early stopping: 'rmse' or 'stability_first'.")
    parser.add_argument("--export_action_stats", action="store_true")
    parser.add_argument("--export_rollout_csv", action="store_true")
    
    parser.add_argument('--earlystop_monitor_after_epoch', type=int, default=0,
                        help='Start monitoring for the best model only after this epoch. '
                             'Crucial for curriculum learning to bypass Phase 1.')
    parser.add_argument('--force_orig_continue', action='store_true')

    args, _ = parser.parse_known_args()
    
    # --- Dynamic Epoch/Batch Size Calculation ---
    try:
        # Fast row count
        with open(args.train_path, 'rb') as f: num_samples = sum(1 for _ in f) - 1 
    except FileNotFoundError:
        raise FileNotFoundError(f"[Error] Training file not found at \"{args.train_path}\"")
    
    # --- Auto-Detect State Columns ---
    df_columns = pd.read_csv(args.train_path, nrows=0).drop(columns=args.skip_cols, errors='ignore').columns
    col_prefixes = ['error_', 'pressure_', 'context_']
    STATE_COLS_DEFAULT = [
        col for col in df_columns if any(
            col.startswith(p) for p in col_prefixes) and
            not col.endswith('_norm')
    ]
    # Enforce order E -> Phi -> H
    prefix_order = {prefix: idx for idx, prefix in enumerate(col_prefixes)}
    def get_prefix_key(col_name):
        for prefix, order in prefix_order.items():
            if col_name.startswith(prefix):
                return order
        return 99 # Should not happen for selected columns

    STATE_COLS_DEFAULT.sort(key=get_prefix_key)
    print("[Info] Enforced state column order E->Phi->H:", STATE_COLS_DEFAULT)
    
    # Identify indices for each component group
    indices = {}
    for prefix in col_prefixes:
        prefix_indices = [i for i, col in enumerate(STATE_COLS_DEFAULT) if col.startswith(prefix)]
        if prefix_indices: indices[prefix] = (prefix_indices[0], prefix_indices[-1])
        else: indices[prefix] = (None, None)
    E_INDEX = indices['error_']; E_DIM = E_INDEX[1] - E_INDEX[0] + 1
    PHI_INDEX = indices['pressure_']; PHI_DIM = PHI_INDEX[1] - PHI_INDEX[0] + 1
    H_INDEX = indices['context_']; H_DIM = H_INDEX[1] - H_INDEX[0] + 1
    S_DIM = len(STATE_COLS_DEFAULT)
    
    # Update args with dimensions
    args.S_DIM = S_DIM
    args.E_DIM = E_DIM
    args.PHI_DIM = PHI_DIM
    args.H_DIM = H_DIM
    args.E_INDEX = list(E_INDEX)
    args.PHI_INDEX = list(PHI_INDEX)
    args.H_INDEX = list(H_INDEX)
    args.STATE_COLS_DEFAULT = STATE_COLS_DEFAULT
    
    if args.P_init is None or not isinstance(args.P_init, List) or len(args.P_init) != E_DIM: args.P_init = [1.0] * E_DIM
    if args.epochs is None:
        steps_per_epoch = max(1, math.ceil(num_samples / args.batch_size))
        args.epochs = max(1, math.ceil(args.target_total_steps / steps_per_epoch))
        print(f"[Info] epochs not specified. Calculated epochs={args.epochs} to reach ~{args.target_total_steps} total steps with batch_size={args.batch_size}.")

    _set_rand_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f'Device: {args.device}, Num Workers: {args.num_workers}')

    # --- Setup Directories ---
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "config.json"
    last_ckpt_path = save_dir / "checkpoint_last.pt"
    best_ckpt_path = save_dir / "checkpoint_best.pt"
    log_jsonl_path = save_dir / "metrics.jsonl"
    log_cache_path = save_dir / "cached_metrics.jsonl"
    
    start_epoch = 1
    epochs_to_run = args.epochs
    resume_checkpoint = None
    
    # --- Primary Logic: Check for existing experiment and handle user intent ---
    if config_path.exists():
        print(f"\n[Info] Found existing experiment in {save_dir}. Checking configuration...")
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_args = json.load(f)
            
        if not check_training_configs_match(saved_args, args):
            print("[Action] Configuration mismatch! Deleting old experiment directory and starting fresh.")
            try:
                shutil.rmtree(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True) # Recreate dir after deletion
            except OSError as e:
                raise OSError(f"[Error] Could not remove directory {save_dir}: {e}")
        else:
            # Configs match. Now determine user's intent: S8-only or Training.
            print("[Info] Configuration matches.")

            # INTENT 1: S8 VALIDATION ONLY
            if args.s8_only:
                selected_ckpt_path = last_ckpt_path if args.s8_use_last_ckpt else best_ckpt_path
                
                print("[Action] `s8_only` flag detected. Proceeding to S8 validation.")
                if not selected_ckpt_path.exists():
                    print(f"[Error] Cannot run S8 validation: the {'last' if args.s8_use_last_ckpt else 'best'} model not found at {selected_ckpt_path}.")
                    return # Exit cleanly

                components = setup_data_and_model_components(args, rng)
                print(f"[Info] Loading {'last' if args.s8_use_last_ckpt else 'best'} model from {selected_ckpt_path} for S8 validation...")
                selected_checkpoint = torch.load(selected_ckpt_path, map_location=args.device)

                state_dict = selected_checkpoint['model']
                is_compiled = list(state_dict.keys())[0].startswith('_orig_mod.')
                if is_compiled:
                    state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
                
                try:
                    components['model'].load_state_dict(state_dict, strict=True)
                except RuntimeError as e:
                    print(f"[Error] Failed to load model state for S8 validation: {e}")
                    print("        This is likely due to a model architecture mismatch. Ensure configs are identical.")
                    exit(1)
                
                print("[Info] S8: Reconstructing V_fn from the loaded checkpoint's P matrix for --s8_only mode...")
                V_fn_for_s8_only = components['V_fn']
                if 'P' in selected_checkpoint and selected_checkpoint['P'] is not None:
                    P_from_ckpt = torch.tensor(selected_checkpoint['P'], dtype=torch.float32, device=args.device)
                    
                    P_fixed_for_s8_only = P_from_ckpt.detach()
                    def V_fn_s8_only_specific(S: torch.Tensor) -> torch.Tensor:
                        S_base = S[:, :S_DIM]
                        return lyapunov_V(S_base, P_fixed_for_s8_only)
                    
                    V_fn_for_s8_only = V_fn_s8_only_specific # Use the newly created, correct V_fn
                    print(f"[Info]   - S8: V_fn for --s8_only validation created successfully using P values from epoch {selected_checkpoint.get('epoch', 'N/A')}.")
                else:
                    print("[Warning] S8: 'P' matrix not found in checkpoint. Falling back to default V_fn. "
                          "Stability results in summary.json may be inconsistent.")

                s8_run_validation_suite(
                    components['model'], components['val_loader'], components['scaler'], 
                    V_fn_for_s8_only,
                    args,
                    epoch=f"pretrained_{'last' if args.s8_use_last_ckpt else 'best'}", save_dir=save_dir
                )
                
                print("\n[Info] S8 validation on pre-trained model complete. Exiting.")
                return # Exit after validation

            # INTENT 2: RESUME TRAINING
            if last_ckpt_path.exists():
                print("[Action] Preparing to resume training from the last checkpoint.")
                resume_checkpoint = torch.load(last_ckpt_path, map_location=args.device)
                if 'epoch' in resume_checkpoint:
                    start_epoch = resume_checkpoint['epoch'] + 1
                else:
                    print("[Warning] 'epoch' key not found in checkpoint. Will start from epoch 1.")
                if args.force_orig_continue: epochs_to_run = args.epochs - (start_epoch - 1)
            else:
                print("[Warning] Config matches, but no 'checkpoint_last.pt' found. Starting fresh from epoch 1.")

    # --- Setup for Training ---
    # This block is reached if it's a fresh start or a resume.
    if start_epoch == 1:
        print("\n[Info] Starting a new training run.")
        _save_json(vars(args), config_path)
    else:
        print(f"\n[Info] This run will train for {epochs_to_run} epochs (from {start_epoch} to {start_epoch + epochs_to_run - 1}).")

    components = setup_data_and_model_components(args, rng)
    model = components['model']; model: nn.Module
    train_loader, val_loader = components['train_loader'], components['val_loader']
    state_scaler = components['scaler']
    lr_scheduler = components['scheduler']
    action_dim = components['action_dim']
    V_fn, optimizer = components['V_fn'], components['optimizer']
    P_init, P_param = components['P_init'], components['P_param']
    text_embedder = components['text_embedder']
    log_vars_param = components["log_vars_param"]
    
    _save_scaler_joblib(state_scaler, save_dir / "scaler.joblib")

    best_val_rmse = float("inf")
    best_pack = {"rmse": float("inf"), "lyap": float("inf"), "clf_v": float("inf")}

    if resume_checkpoint:
        print("[Info] Loading model and optimizer state from checkpoint...")
        
        # Load model state
        state_dict = resume_checkpoint['model']
        is_compiled = list(state_dict.keys())[0].startswith('_orig_mod.')
        if is_compiled:
            print("[Info] Unwrapping model state_dict from torch.compile()...")
            state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"[Error] Failed to load checkpoint state_dict: {e}")
            exit(1)
        
        # Load optimizer state
        if 'optimizer' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
        else:
            print("[Warning] Optimizer state not found in checkpoint.")

        # Load early stopping criteria from the BEST checkpoint to ensure consistency
        if best_ckpt_path.exists():
            print("[Info] Loading best metrics from best checkpoint for early stopping.")
            best_checkpoint = torch.load(best_ckpt_path, map_location=args.device)
            if 'best_val_metrics' in best_checkpoint:
                best_pack = best_checkpoint['best_val_metrics']
                best_val_rmse = best_pack.get("rmse", float("inf"))
                print(f"[Info] Resuming with best_val_rmse: {best_val_rmse:.4f}")
            else:
                print("[Warning] 'best_val_metrics' not found in best checkpoint.")
        else:
            print("[Warning] Best checkpoint not found. Early stopping criteria reset.")
    
    # Calculate the total number of epochs this session should run
    end_epoch = start_epoch + epochs_to_run - 1
    
    print(f"[Info] This run will train for {epochs_to_run} epochs, from {start_epoch} to {end_epoch}.")
    epoch = start_epoch
    
    # model_for_jac always points to the uncompiled original model
    # for state preservation and Jacobian calculation
    model_for_jac = model
    
    # training_model is the model that will be used for the training loop,
    # it can be compiled or raw
    training_model = model
    if not args.disable_compile and sys.version_info >= (3, 8) and torch.__version__ >= "2.0.0" and not sys.platform.startswith('win'):
        print("PyTorch 2.0+ detected, compiling the model for training...")
        compile_mode = "reduce-overhead" if args.device == "cpu" else "default"
        try:
            training_model = torch.compile(model, mode=compile_mode)
            print("Model compilation successful!")
        except Exception as e:
            print(f"[Warning] Model compilation failed with error: {e}. Falling back to uncompiled model.")
            training_model = model

    if hasattr(torch.amp, 'GradScaler'):
        grad_scaler_func = getattr(torch.amp, 'GradScaler')
        grad_scaler = grad_scaler_func(args.device, enabled=(args.device == 'cuda' and not args.disable_autocast))
    else:
        grad_scaler = None
    
    for epoch in range(start_epoch, end_epoch + 1):
        
        # Reset the best model criteria when the monitoring period begins.
        # This ensures that the first model in the monitoring window is always saved as the initial "best",
        # establishing a correct baseline for subsequent epochs.
        if epoch == args.earlystop_monitor_after_epoch:
            print(f"\n[Info] Starting to monitor for best model at epoch {epoch}. Resetting best metrics.")
            best_val_rmse = float("inf")
            best_pack = {"rmse": float("inf"), "lyap": float("inf"), "clf_v": float("inf")}
        
        # Curriculum: Calculate master weight
        if args.use_curriculum:
            control_phase_epoch = max(0, epoch - args.curriculum_phase1_epochs)
            master_stability_weight = _lin_schedule(
                ep=control_phase_epoch,
                total_ep=args.epochs,
                start=0.0,
                end=1.0,
                warmup_ep=args.lambda_warmup_ep
            )
        else:
            control_phase_epoch = epoch
            master_stability_weight = 1.0
        
        rho_ep = _lin_schedule(control_phase_epoch, args.epochs, args.rho, args.rho_final, args.rho_warmup_ep)
        
        # For logging, show the effective lambda which is the base lambda scaled by the master weight.
        lclf_eff = (args.nstep_lambda if args.use_nstep_clf else args.lambda_clf) * master_stability_weight
        tau_ep = _exp_anneal(epoch, args.epochs, args.softmin_tau, args.softmin_tau_final, args.softmin_anneal_ep)
        a_dirs, a_delta = args.num_delta_dirs, args.action_delta
        
        # --- Train Step ---
        train_log = train_one_epoch(
            training_model, model_for_jac, train_loader, optimizer, args.device, grad_scaler,
            V_fn=V_fn, args=args, epoch=epoch, log_vars_param=log_vars_param
        )
        
        # --- Validation Step ---
        val_metrics = compute_metrics(
            training_model, val_loader, args.device, V_fn=V_fn, rho=rho_ep,
            phi_crit=(args.phi_crit if args.phi_crit > 0 else None),
            rollout_H=args.rollout_H,
            num_delta_dirs=a_dirs, action_delta=a_delta,
            softmin_tau=tau_ep, args=args,
            scaler=state_scaler
        )
        
        if lr_scheduler:
            if args.lr_scheduler_type == 'reduce_on_plateau':
                monitor_val = val_metrics.get(args.lr_scheduler_monitor, float('inf'))
                lr_scheduler.step(monitor_val)
            else:
                lr_scheduler.step()
                
        # --- Checkpointing ---
        current_rmse = val_metrics.get("rmse_total", float("inf"))
        improved = False
        if args.earlystop_mode == "rmse":
            if current_rmse < best_val_rmse:
                improved = True
                best_val_rmse = current_rmse
        else: # stability_first
            eps = args.stab_lexi_eps
            lyap = val_metrics.get("lyapunov_pos_rate_mean", float("inf"))
            clf_v = val_metrics.get("clf_violation_rate", float("inf"))
            r = current_rmse
            if (lyap + eps < best_pack["lyap"] or
                (abs(lyap - best_pack["lyap"]) <= eps and (clf_v + eps < best_pack["clf_v"] or
               (abs(clf_v - best_pack["clf_v"]) <= eps and r < best_pack["rmse"])))):
                improved = True
        
        if improved and epoch >= args.earlystop_monitor_after_epoch:
            print(f"[Info] New best model found at epoch {epoch}. Saving checkpoint...")
            best_pack["rmse"] = current_rmse
            best_pack["lyap"] = val_metrics.get("lyapunov_pos_rate_mean", float("inf"))
            best_pack["clf_v"] = val_metrics.get("clf_violation_rate", float("inf"))
            
            P_to_save = (F.softplus(P_param).detach().cpu().numpy() + 1e-6).tolist() if args.learn_P else P_init.detach().cpu().numpy().tolist()
            
            torch.save({
                "model": model_for_jac.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(args),
                "epoch": epoch,
                "best_val_metrics": best_pack,
                "P": P_to_save
            }, best_ckpt_path)

        # Logging
        log_vars_log = log_vars_param.detach().cpu().numpy().tolist() if log_vars_param is not None else None
        P_log = (F.softplus(P_param).detach().cpu().numpy() + 1e-6).tolist() if args.learn_P else P_init.detach().cpu().numpy().tolist()
        log = {
            "epoch": epoch, "train": train_log, "val": val_metrics, "P": P_log,
            "sched": {
                "rho": rho_ep, 
                "lambda_clf_effective": lclf_eff, # Log the effective lambda
                "master_stability_weight": master_stability_weight, # Log the master weight
                "softmin_tau": tau_ep, 
                "num_delta_dirs": a_dirs, 
                "action_delta": a_delta
            },
            "log_vars": log_vars_log,
            "save_dir": str(save_dir)
        }
        state_component_names = STATE_COLS_DEFAULT
        train_huber_e = [train_log.get(f'huber_{state_component_names[i]}', -1) for i in range(E_INDEX[0], E_INDEX[1] + 1)]
        train_huber_h = [train_log.get(f'huber_{state_component_names[i]}', -1) for i in range(H_INDEX[0], H_INDEX[1] + 1)]
        val_rmse_e = [val_metrics.get(f'rmse_{state_component_names[i]}', -1) for i in range(E_INDEX[0], E_INDEX[1] + 1)]
        val_rmse_h = [val_metrics.get(f'rmse_{state_component_names[i]}', -1) for i in range(H_INDEX[0], H_INDEX[1] + 1)]
        val_w1_e = [val_metrics.get(f'w1_{state_component_names[i]}', -1) for i in range(E_INDEX[0], E_INDEX[1] + 1)]
        val_w1_h = [val_metrics.get(f'w1_{state_component_names[i]}', -1) for i in range(H_INDEX[0], H_INDEX[1] + 1)]
        val_r2_e = [val_metrics.get(f'r2_{state_component_names[i]}', -1) for i in range(E_INDEX[0], E_INDEX[1] + 1)]
        val_r2_h = [val_metrics.get(f'r2_{state_component_names[i]}', -1) for i in range(H_INDEX[0], H_INDEX[1] + 1)]
        
        train_huber_e_txt = ','.join([f"{train_huber_e[i]:.6f}" for i in range(len(train_huber_h))])
        train_huber_h_txt = ','.join([f"{train_huber_h[i]:.6f}" for i in range(len(train_huber_h))])
        rmse_r2_e_txt = ','.join([f'{val_rmse_e[i]:.6f} (R²:{val_r2_e[i]:.2f})' for i in range(min(len(val_rmse_e), len(val_r2_e)))])
        rmse_r2_h_txt = ','.join([f'{val_rmse_h[i]:.6f} (R²:{val_r2_h[i]:.2f})' for i in range(min(len(val_rmse_h), len(val_r2_h)))])
        w1_e_txt = ','.join([f'{val_w1_e[i]:.6f}' for i in range(len(val_w1_e))])
        w1_h_txt = ','.join([f'{val_w1_h[i]:.6f}' for i in range(len(val_w1_h))])
        
        if int(epoch) > 1: print("="*40)
        print(f"Epoch {epoch:02d}/{args.epochs} ({(epoch/args.epochs)*100:2f}%):\n"
              f"\tTrain Loss: {log['train']['loss']:.6f}, Total Huber Loss: {log['train']['huber_total']:.6f}\n"
              f"\t\tHuber Loss (E): {train_huber_e_txt}\n\t\tHuber Loss (H): {train_huber_h_txt}\n\n"
              f"\tVal. Total RMSE: {log['val'].get('rmse_total', -1):.6f}, Total W1: {log['val'].get('w1_total', -1):.6f}\n"
              f"\t\tRMSE(E): {rmse_r2_e_txt})\n\t\tRMSE(H): {rmse_r2_h_txt})\n"
              f"\t\t"+"-"*(40-len("\t\t"))+"\n"
              f"\t\tW1(E): {w1_e_txt}\n\t\tW1(H): {w1_h_txt}"
        )
        
        robust_log_writer(log, log_jsonl_path, log_cache_path)
        
        torch.save({
            "model": model_for_jac.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
            "epoch": epoch,
            "best_val_metrics": best_pack,
            "P": P_log
        }, last_ckpt_path)

    print("\nTraining finished.")
    print(f"Saving final checkpoint from epoch {epoch}...")
    P_to_save = (F.softplus(P_param).detach().cpu().numpy() + 1e-6).tolist() if args.learn_P else P_init.detach().cpu().numpy().tolist()
    torch.save({
        "model": model_for_jac.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(args),
        "epoch": epoch,
        "P": P_to_save
    }, save_dir / "checkpoint_last.pt")
    _save_json(val_metrics, save_dir / "metrics_last.json")

    # Export analysis statistics for the final model
    export_val_action_stats_and_rollout(
        model, val_loader, args.device, V_fn, rho=args.rho,
        num_delta_dirs=args.num_delta_dirs, action_delta=args.action_delta,
        save_dir=save_dir,
        export_stats=args.export_action_stats, export_rollout=args.export_rollout_csv
    )
    
    # Run final comprehensive S8 validation suite if enabled
    if args.s8_enable:
        print(f"\n[Info] S8: Loading {'last' if args.s8_use_last_ckpt else 'best'} model for final theoretical validation...")
        
        selected_ckpt_path = last_ckpt_path if args.s8_use_last_ckpt else best_ckpt_path
        
        if selected_ckpt_path.exists():
            checkpoint = torch.load(selected_ckpt_path, map_location=args.device, weights_only=False)
            # Re-create model to load state dict with potentially different args than current run
            saved_config = argparse.Namespace(**checkpoint['config'])
            validation_model = PAECTransition(
                action_dim=action_dim,
                hid_dim=saved_config.hid_dim,
                layers=saved_config.layers,
                history_len=saved_config.history_len,
                predict_delta=saved_config.predict_delta,
                use_text_embeddings=saved_config.use_text_embeddings,
                text_embedding_dim=text_embedder.get_sentence_embedding_dimension(),
                use_decoder_hidden_state=saved_config.use_decoder_hidden_state,
                decoder_hidden_state_dim=saved_config.decoder_hidden_state_dim,
                use_moe_heads=saved_config.use_moe_heads,
                use_multi_heads=saved_config.use_multi_heads,
                use_spectral_norm=saved_config.use_spectral_norm,
                nhead=saved_config.nhead
            ).to(args.device)
            
            state_dict = checkpoint['model']
            is_compiled = list(state_dict.keys())[0].startswith('_orig_mod.')
            if is_compiled:
                state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
            validation_model.load_state_dict(state_dict, strict=True)
            
            print(f"[Info] S8: Reconstructing V_fn from the {'last' if args.s8_use_last_ckpt else 'best'} checkpoint's P matrix...")
            V_fn_for_s8 = V_fn
            if 'P' in checkpoint and checkpoint['P'] is not None:
                P_from_ckpt = torch.tensor(checkpoint['P'], dtype=torch.float32, device=args.device)
                
                P_fixed_for_s8 = P_from_ckpt.detach()
                def V_fn_s8_specific(S: torch.Tensor) -> torch.Tensor:
                    S_base = S[:, :S_DIM]
                    return lyapunov_V(S_base, P_fixed_for_s8)
                
                V_fn_for_s8 = V_fn_s8_specific
                print(f"[Info]   - S8: V_fn for validation created successfully using P values from epoch {checkpoint.get('epoch', 'N/A')}.")
            else:
                print("[Warning] S8: 'P' matrix not found in checkpoint. Falling back to V_fn from the end of training. "
                      "Stability results in summary.json may be inconsistent.")
            
            s8_run_validation_suite(
                validation_model, val_loader, components['scaler'], 
                V_fn_for_s8,
                args,
                epoch='last' if args.s8_use_last_ckpt else 'best',
                save_dir=save_dir
            )
        else:
            print(f"[Warning] S8: The {'last' if args.s8_use_last_ckpt else 'best'} checkpoint not found. Skipping final S8 validation.")

if __name__ == "__main__":
    main()
