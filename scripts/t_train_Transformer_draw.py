# -*- coding: utf-8 -*-

import os, sys, json, shutil, warnings, traceback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, cast
from matplotlib.container import BarContainer
from matplotlib.patches import Rectangle
import shutil
import warnings

# --- CONFIGURATION ---

try:
    # Add the parent directory of this script to the system path to allow importing 'src'
    sys.path.append(str(Path(__file__).parent.parent.resolve()))
    from src.config import *
except ImportError as e:
    print("[ERROR] Failed to import project modules. Ensure the script is run from the 'scripts' directory.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print("[ERROR] An unexpected error occurred during import of src/config.")
    traceback.print_exc()
    sys.exit(1)

# Define the base directory where all experiment runs are saved.
# This directory should contain the results of various experiment milestones.
# Use a raw string (r"...") or forward slashes for cross-platform compatibility.
base_dir = PATHS["dynamics_model_dir"]

# Directories to skip during processing or deletion
# This list can be populated with paths that should be excluded from batch operations.
jump_dirs = []

# --- [FIXED] Fully populated list of all experiments to analyze ---
# This list defines the specific experiments to process, grouped by milestone.
# Each tuple contains: (Experiment Name, Relative Path within base_dir)
EXPERIMENTS = [
    # Milestone 0: Baseline model without specific stability or accuracy enhancements.
    ("00_Base", "00_Base"),
    
    # Milestone 1: Focus on Lipschitz Continuity and Spectral Normalization.
    ("01_OnlySN", "01_Continuity/01_OnlySN"),
    ("01_SN_jac_1e-3", "01_Continuity/02_SN_1e-3"),
    
    # Milestone 2: Focus on Prediction Accuracy improvements.
    ("02_LearnP_Scheduler", "02_Accuracy/01_Learn_P_lr_Scheduler"),
    ("02_Text_Decoder", "02_Accuracy/02_TextEmb_DecoderState"),
    ("02_Uncertainty", "02_Accuracy/03_UncertaintyWeights"),
    ("02_MOE", "02_Accuracy/04_MOE"),
    ("02_Predict_Delta", "02_Accuracy/05_Predict_Delta"),
    ("02_MultiHead", "02_Accuracy/06_MultiHead"),
    
    # Milestone 3: Focus on Stability mechanisms (CLF, CBF, etc.).
    # 3.1: Single-Step Control Lyapunov Function (CLF) with varying weights.
    ("03_Single_CLF_0.4", "03_Stability/01_Single_CLF/01_CLF_0.4"),
    ("03_Single_CLF_0.8", "03_Stability/01_Single_CLF/02_CLF_0.8"),
    ("03_Single_CLF_1.2", "03_Stability/01_Single_CLF/03_CLF_1.2"),
    # 3.2: N-Step CLF with different horizons and Gumbel-Softmax variations.
    ("03_NCLF_0.7", "03_Stability/02_NCLF/01_NCLF_0.7"),
    ("03_NCLF_1.0", "03_Stability/02_NCLF/02_NCLF_1.0"),
    ("03_NCLF_1.3", "03_Stability/02_NCLF/03_NCLF_1.3"),
    ("03_Gumbel_Softmax", "03_Stability/02_NCLF/04_Gumble_Softmax"),
    # 3.3: Epsilon-Greedy exploration strategies.
    ("03_Epsilon_Default", "03_Stability/03_epsilon_Greedy/01_default"),
    ("03_Epsilon_PolicyEntropy_0.01", "03_Stability/03_epsilon_Greedy/02_PolicyEntropy_0.01"),
    ("03_Epsilon_PolicyEntropy_0.1", "03_Stability/03_epsilon_Greedy/03_PolicyEntropy_0.1"),
    # 3.4: Backpropagation Through Time (BPTT) Window adjustments.
    ("03_BPTT_H5_W2", "03_Stability/04_bptt/01_H_5_bptt_2"),
    ("03_BPTT_H6_W3", "03_Stability/04_bptt/02_H_6_bptt_3"),
    ("03_BPTT_H8_W4", "03_Stability/04_bptt/03_H_8_bptt_4"),
    # 3.5: Conditional Value-at-Risk (CVaR) loss for tail risk management.
    ("03_CVaR_alpha_0.7", "03_Stability/05_CVaR/01_alpha_0.7"),
    ("03_CVaR_alpha_1.0", "03_Stability/05_CVaR/02_alpha_1.0"),
    ("03_CVaR_alpha_1.3", "03_Stability/05_CVaR/03_alpha_1.3"),
    # 3.6.1: Control Barrier Functions (CBF) for safety constraints.
    ("03_CBF_L0.1_C0.7_A0.5", "03_Stability/061_CBF/01_λ_0.1_crit_0.7_alpha_0.5"),
    ("03_CBF_L0.5_C0.7_A0.5", "03_Stability/061_CBF/02_λ_0.5_crit_0.7_alpha_0.5"),
    ("03_CBF_L1.0_C0.7_A0.5", "03_Stability/061_CBF/03_λ_1.0_crit_0.7_alpha_0.5"),
    ("03_CBF_L0.5_C0.7_A0.1", "03_Stability/061_CBF/04_λ_0.5_crit_0.7_alpha_0.1"),
    ("03_CBF_L0.5_C0.7_A0.9", "03_Stability/061_CBF/05_λ_0.5_crit_0.7_alpha_0.9"),
    ("03_CBF_L0.5_C0.5_A0.5", "03_Stability/061_CBF/06_λ_0.5_crit_0.5_alpha_0.5"),
    ("03_CBF_L0.5_C0.9_A0.5", "03_Stability/061_CBF/07_λ_0.5_crit_0.9_alpha_0.5"),
    # 3.6.2: Cox Proportional Hazards model for risk prediction.
    ("03_Cox_L0.5_T0.5", "03_Stability/062_Cox/01_λ_0.5_event_thr_0.5"),
    ("03_Cox_L0.5_T0.8", "03_Stability/062_Cox/02_λ_0.5_event_thr_0.8"),
    ("03_Cox_L1.0_T0.5", "03_Stability/062_Cox/03_λ_1.0_event_thr_0.5"),
    ("03_Cox_L1.0_T0.8", "03_Stability/062_Cox/04_λ_1.0_event_thr_0.8"),
    # 3.6.3: Action Dissimilarity Term (ADT) for temporal consistency.
    ("03_ADT_L0.7", "03_Stability/063_ADT/01_λ_0.7"),
    ("03_ADT_L1.0", "03_Stability/063_ADT/02_λ_1.0"),
    ("03_ADT_L1.3", "03_Stability/063_ADT/03_λ_1.3"),
    
    # Milestone 4: Curriculum Learning strategies.
    ("04_Curriculum_10", "04_Curriculum/01_phase1_epochs_10"),
    ("04_No_Curriculum", "04_Curriculum/02_no_curriculum"),
    
    # Champion Model: The final selected best-performing model.
    ("Champion", "Champion")
]

# State variable component groups, mapping metric categories to specific data columns.
# Used for aggregating and plotting related metrics.
STATE_COMPONENTS = {
    'Error': ["error_semantic", "error_coverage", "error_fluency_surprisal", "error_fluency_repetition"],
    'Pressure': ["pressure_latency", "pressure_memory", "pressure_throughput"],
    'Context': ["context_faith_focus", "context_consistency", "context_stability", "context_confidence_volatility"]
}

# --- DIRECTORY UTILITIES ---

def delete_all_analysis_plots(start_path: Path):
    """
    Recursively deletes all 'analysis_plots' folders within a given directory tree.
    This is useful for cleaning up previous analysis outputs before a fresh run.
    It respects the 'jump_dirs' list to avoid deleting data in excluded directories.

    Args:
        start_path (Path): The root directory to start the search from.

    Returns:
        int: The number of 'analysis_plots' directories deleted.
    """
    start_path = Path(start_path).resolve()
    deleted_count = 0
    
    # Walk from the bottom up to safely delete directories
    for root, dirs, _ in os.walk(start_path, topdown=False):
        current_root = Path(root)
        # Check if the current directory or any of its parents are in the skip list
        if any(jump_dir in current_root.parents or jump_dir == current_root for jump_dir in [Path(j) for j in jump_dirs]):
            continue
        
        # If 'analysis_plots' directory exists, delete it
        if "analysis_plots" in dirs:
            target_path = current_root / "analysis_plots"
            try:
                shutil.rmtree(target_path)
                print(f"✅ Deleted: {target_path}")
                deleted_count += 1
                dirs.remove("analysis_plots") # Update dirs list for the walker to avoid errors
            except Exception as e:
                print(f"❌ Deletion Failed: {target_path} - {str(e)}")
    return deleted_count

# --- CORE PROCESSING LOGIC ---

def process_run_directory(run_dir: Path, experiment_name: str):
    """
    Orchestrates the analysis tasks for a single experiment run directory.
    It generates various plots and summaries based on the available data files.

    Args:
        run_dir (Path): The directory path of the specific experiment run.
        experiment_name (str): The name of the experiment for labeling outputs.
    """
    print(f"Processing: {run_dir.relative_to(base_dir)}")
    
    # Define and create the output directory for plots
    output_dir = run_dir / "analysis_plots"
    output_dir.mkdir(exist_ok=True)

    # --- Analysis Step 1: Action Histogram ---
    # Visualizes the distribution of selected actions (k values) from validation data.
    hist_path = run_dir / "val_best_action_hist.json"
    if hist_path.exists():
        try:
            with open(hist_path, 'r', encoding='utf-8') as f:
                hist_data_json = json.load(f)
            plot_action_histogram(hist_data_json, output_dir / f"{experiment_name}_action_histogram.png")
        except Exception as e:
            print(f"  - Failed to process action histogram: {e}")

    # --- Analysis Step 2: dV Distribution ---
    # Analyzes the distribution of Lyapunov value changes (dV) from single-step rollouts.
    rollout_path = run_dir / "val_rollout_one_step.csv"
    if rollout_path.exists():
        try:
            rollout_data_df = pd.read_csv(rollout_path)
            plot_dV_distribution_and_analyze(
                rollout_data_df, 
                output_dir / f"{experiment_name}_dV_distribution.png", 
                output_dir / f"{experiment_name}_dV_distribution.json"
            )
        except Exception as e:
            print(f"  - Failed to process dV distribution: {e}")

    # --- Analysis Step 3: Training Curves from metrics.jsonl ---
    # plots training progress metrics like RMSE, loss components, and stability rates over epochs.
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        try:
            metrics_data_df = load_metrics_data(metrics_path)
            if not metrics_data_df.empty:
                plot_training_curves(metrics_data_df, output_dir, experiment_name)
            else:
                print(f"  - Metrics file is empty: {metrics_path}")
        except Exception as e:
            print(f"  - Failed to process training curves: {e}")

    # --- Analysis Step 4: S8 Validation Summary ---
    # Creates a summary table from the theoretical validation suite (S8) results.
    # It checks for 'best' epoch first, then falls back to 'pretrained_best'.
    summary_path_best = run_dir / "s8_validation_epoch_best" / "summary.json"
    summary_path_pretrained = run_dir / "s8_validation_epoch_pretrained_best" / "summary.json"
    summary_path = None
    if summary_path_best.exists():
        summary_path = summary_path_best
    elif summary_path_pretrained.exists():
        summary_path = summary_path_pretrained
    
    if summary_path:
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            create_summary_visual(summary_data, output_dir / f"{experiment_name}_S8_summary.png", experiment_name)
        except Exception as e:
            print(f"  - Failed to process S8 summary: {e}")


def scan_and_process_runs(base_dir: str, experiments_list: List[Tuple[str, str]]):
    """
    Iterates through the defined list of experiments and triggers processing for each.
    It handles directory checks and skips invalid or incomplete runs.

    Args:
        base_dir (str): The root directory containing experiment outputs.
        experiments_list (List[Tuple[str, str]]): List of (name, relative_path) tuples.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        print(f"❌ Base directory not found: {base_path}")
        return

    for exp_name, rel_path in experiments_list:
        run_dir = base_path / rel_path
        
        # Skip if the directory doesn't exist
        if not run_dir.is_dir():
            print(f"⏭️ Skipping (Not Found): {run_dir.relative_to(base_path)}")
            continue

        # Check for essential summary files to determine if the run was completed/valid
        summary_exists = (run_dir / "s8_validation_epoch_best" / "summary.json").exists() or \
                         (run_dir / "s8_validation_epoch_pretrained_best" / "summary.json").exists()
        
        if not summary_exists:
            print(f"⏭️ Skipping (Incomplete - no summary.json): {run_dir.relative_to(base_path)}")
            continue

        # Skip if directory is in the jump_dirs list
        if any(jump_dir in run_dir.parents or jump_dir == run_dir for jump_dir in [Path(j) for j in jump_dirs]):
            print(f"⏭️ Skipping (jump_dirs): {run_dir.relative_to(base_path)}")
            continue
        
        try:
            process_run_directory(run_dir, exp_name)
        except Exception as e:
            print(f"  - ❗ Top-level Processing Failed for {run_dir.relative_to(base_path)}: {e}")


# --- DATA LOADING ---

def load_metrics_data(path: Path):
    """
    Parses the 'metrics.jsonl' file, extracting relevant metrics into a pandas DataFrame.
    It flattens nested structures for easier plotting.

    Args:
        path (Path): Path to the metrics.jsonl file.

    Returns:
        pd.DataFrame: DataFrame containing metrics per epoch.
    """
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Infer epoch number if missing
            epoch = data.get('epoch', len(records) + 1)
            val_metrics = data.get('val', {})
            sched_params = val_metrics.get('sched', {})
            
            record = {'epoch': epoch}

            # Extract standard regression metrics (RMSE, R2, W1) for state components
            for metric_prefix in ['rmse', 'r2', 'w1']:
                record[f'{metric_prefix}_total'] = val_metrics.get(f'{metric_prefix}_total')
                for _, components in STATE_COMPONENTS.items():
                    for comp in components:
                        record[f'{metric_prefix}_{comp}'] = val_metrics.get(f'{metric_prefix}_{comp}')

            # Extract specific stability and control metrics
            other_metrics = [
                'multi_step_neg_drift_coverage', 'multi_step_neg_drift_coverage_teacher', 'multi_step_neg_drift_coverage_fixed0',
                'nstep_clf_violation_rate_step0', 'nstep_clf_violation_rate_step1', 'nstep_clf_violation_rate_step2',
                'nstep_clf_violation_rate_mean', 'nstep_endpoint_violation_rate', 'nstep_cvar_violation',
                'lyapunov_pos_rate_mean', 'adt_switch_rate', 'cox_loss'
            ]
            for metric in other_metrics:
                record[metric] = val_metrics.get(metric)

            # Extract scheduler parameter 'rho'
            record['rho_sched'] = sched_params.get('rho')
            records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Ensure numeric types for plotting
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- PLOTTING FUNCTIONS ---

def plot_action_histogram(hist_data: Dict[str, Dict[str, float]], save_path: Path):
    """
    Generates a bar chart showing the frequency of selected actions (k values).
    
    Args:
        hist_data (Dict): Dictionary containing histogram data under 'hist' key.
        save_path (Path): Destination path for the saved image.
    """
    hist = hist_data.get('hist', {})
    if not hist: return
    
    # Sort keys for consistent x-axis ordering
    keys_int = sorted([int(k) for k in hist.keys()])
    keys_str = sorted([str(k) for k in hist.keys()])
    values = [hist[k] for k in keys_str]
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=keys_int, y=values, palette="viridis", hue=keys_int, legend=False)
    plt.title('Distribution of Best Stabilizing Actions (Validation Set)', fontsize=16)
    plt.xlabel('Candidate Action Index k', fontsize=12)
    plt.ylabel('Selection Frequency', fontsize=12)
    plt.xticks(ticks=range(len(keys_int)), labels=[k for k in keys_str])
    
    # Add value labels on top of bars
    for container in ax.containers:
        casted_container = cast(BarContainer, container)
        ax.bar_label(casted_container, fmt='%d', label_type='edge', size=11, padding=5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_dV_distribution_and_analyze(rollout_df: pd.DataFrame, save_path_plot: Path, save_path_json: Path):
    """
    Plots the histogram of dV (Lyapunov change) values and saves descriptive statistics.
    
    Args:
        rollout_df (pd.DataFrame): DataFrame containing rollout data with 'dV' column.
        save_path_plot (Path): Path to save the distribution plot.
        save_path_json (Path): Path to save the statistical summary.
    """
    if 'dV' not in rollout_df or rollout_df['dV'].isnull().all(): return
    dv_data = rollout_df['dV'].dropna()
    if dv_data.empty: return

    plt.figure(figsize=(12, 7))
    sns.histplot(x=dv_data, kde=True, bins=50, color='royalblue')
    mean_dv = dv_data.mean()
    plt.axvline(mean_dv, color='red', linestyle='--', label=f'Average dV = {mean_dv:.3f}')
    plt.title('Lyapunov Decay (dV) Distribution of Optimal Single-Step Action', fontsize=16)
    plt.xlabel('dV = V(S_t+1) - (1-rho)*V(S_t)', fontsize=12)
    plt.ylabel('Sample Density', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path_plot)
    plt.close()

    # Calculate and save statistics
    desc_stats = dv_data.describe().to_dict()
    analysis_results = {"analysis_target": "dV_distribution", "descriptive_statistics": desc_stats}
    with open(save_path_json, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

def plot_training_curves(metrics_df: pd.DataFrame, output_dir: Path, experiment_name: str):
    """
    Generates a comprehensive set of training curve plots from the metrics DataFrame.
    Includes component-wise accuracy, coverage rates, violation rates, and other stability metrics.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing training metrics over epochs.
        output_dir (Path): Directory to save the plots.
        experiment_name (str): Name of the experiment for titles.
    """
    epoch_col = 'epoch'
    
    def all_nan(df: pd.DataFrame, cols: List[str]):
        """Helper to check if a list of columns contains only NaN values."""
        cols_exist = [c for c in cols if c in df.columns]
        if not cols_exist: return True
        return df[cols_exist].isnull().all().all()

    # --- Plot Component-wise Metrics (RMSE, W1, R2) ---
    # Iterates through metric types (rmse, w1, r2) and state component groups (Error, Pressure, Context)
    def plot_metric_components(metric_prefix: str, y_label: str):
        for group_name, components in STATE_COMPONENTS.items():
            cols = [f"{metric_prefix}_{comp}" for comp in components]
            if not all_nan(metrics_df, cols):
                plt.figure(figsize=(12, 7))
                for col in cols:
                    if col in metrics_df and not metrics_df[col].isnull().all():
                        # Remove prefix for cleaner legend labels
                        plt.plot(metrics_df[epoch_col], metrics_df[col], marker='o', linestyle='-', label=col.replace(f"{metric_prefix}_", ""))
                
                # Plot Total metric if available
                total_col = f"{metric_prefix}_total"
                if total_col in metrics_df and not metrics_df[total_col].isnull().all():
                     plt.plot(metrics_df[epoch_col], metrics_df[total_col], marker='s', linestyle='--', color='black', label='Total')

                plt.title(f'{experiment_name} - {group_name} Prediction Accuracy ({metric_prefix.upper()})', fontsize=16)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel(y_label, fontsize=12)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(output_dir / f"{experiment_name}_curves_{metric_prefix.upper()}_{group_name}.png")
                plt.close()

    plot_metric_components('rmse', 'RMSE')
    plot_metric_components('w1', 'Wasserstein-1 Distance')
    plot_metric_components('r2', 'R² Score')

    # --- Plot Coverage Metrics ---
    # Visualizes multi-step negative drift coverage alongside the 'rho' scheduler.
    coverage_cols = ['multi_step_neg_drift_coverage', 'multi_step_neg_drift_coverage_teacher', 'multi_step_neg_drift_coverage_fixed0']
    if not all_nan(metrics_df, coverage_cols):
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.plot(metrics_df[epoch_col], metrics_df[coverage_cols[0]], marker='o', linestyle='-', label='Coverage')
        ax1.plot(metrics_df[epoch_col], metrics_df[coverage_cols[1]], marker='s', linestyle='--', label='Coverage (Teacher)')
        ax1.plot(metrics_df[epoch_col], metrics_df[coverage_cols[2]], marker='^', linestyle=':', label='Coverage (Fixed-0)')
        ax1.set_title(f'{experiment_name} - Multi-Step Negative Drift Coverage (H=12)', fontsize=16)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Coverage Rate', fontsize=12)
        ax1.legend(loc='upper left')

        # Plot rho on secondary y-axis
        if 'rho_sched' in metrics_df and not metrics_df['rho_sched'].isnull().all():
            ax2 = ax1.twinx()
            ax2.plot(metrics_df[epoch_col], metrics_df['rho_sched'], color='red', linestyle='-.', label='rho (Stability Pressure)')
            ax2.set_ylabel('rho Value', color='red', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')
        
        fig.tight_layout()
        plt.savefig(output_dir / f"{experiment_name}_curves_Coverage.png")
        plt.close()

    # --- Plot N-Step Violation Rates with specific styling ---
    # Visualizes various N-step violation metrics and CVaR violation.
    vio_cols = [
        'nstep_endpoint_violation_rate', 'nstep_clf_violation_rate_mean',
        'nstep_clf_violation_rate_step0', 'nstep_clf_violation_rate_step1',
        'nstep_clf_violation_rate_step2'
    ]
    cvar_col = 'nstep_cvar_violation'
    
    if not all_nan(metrics_df, vio_cols + [cvar_col]):
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Define styles for different lines
        style_map = {
            'nstep_endpoint_violation_rate': {'color': 'crimson', 'linestyle': '-', 'linewidth': 2.5, 'marker': 'o', 'zorder': 10, 'label': 'Endpoint Violation (Most Important)'},
            'nstep_clf_violation_rate_mean': {'color': 'cornflowerblue', 'linestyle': '-', 'linewidth': 1.5, 'marker': 's', 'alpha': 0.8, 'label': 'Mean Violation'},
            'nstep_clf_violation_rate_step0': {'color': 'grey', 'linestyle': '--', 'linewidth': 1, 'marker': '.', 'alpha': 0.7, 'label': 'Step 0 Violation'},
            'nstep_clf_violation_rate_step1': {'color': 'darkgrey', 'linestyle': ':', 'linewidth': 1, 'marker': '.', 'alpha': 0.7, 'label': 'Step 1 Violation'},
            'nstep_clf_violation_rate_step2': {'color': 'silver', 'linestyle': '-.', 'linewidth': 1, 'marker': '.', 'alpha': 0.7, 'label': 'Step 2 Violation'}
        }
        
        for col in vio_cols:
            if col in metrics_df and not metrics_df[col].isnull().all():
                style = style_map.get(col, {})
                ax1.plot(metrics_df[epoch_col], metrics_df[col], **style)

        ax1.set_title(f'{experiment_name} - N-Step Violation Rates', fontsize=16)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Violation Rate', fontsize=12)
        ax1.legend(loc='upper left')
        
        # Plot CVaR violation on secondary y-axis
        if cvar_col in metrics_df and not metrics_df[cvar_col].isnull().all():
            ax2 = ax1.twinx()
            ax2.plot(metrics_df[epoch_col], metrics_df[cvar_col], color='purple', marker='x', linestyle='--', label='CVaR Violation')
            ax2.set_ylabel('CVaR Violation Value', color='purple', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='purple')
            ax2.legend(loc='upper right')
        
        fig.tight_layout()
        plt.savefig(output_dir / f"{experiment_name}_curves_ViolationRates.png")
        plt.close()


    # --- Other Individual Plots ---
    # Plots specialized metrics like ADT Switch Rate, Lyapunov Positive Rate, and Cox Loss.
    single_metric_plots = {
        'ADTSwitchRate': ('adt_switch_rate', 'Switch Rate'),
        'LyapunovPosRate': ('lyapunov_pos_rate_mean', 'Positive Rate'),
        'CoxLoss': ('cox_loss', 'Loss Value')
    }

    for name, (col, ylabel) in single_metric_plots.items():
        if col in metrics_df and not metrics_df[col].isnull().all():
            plt.figure(figsize=(12, 7))
            
            # Handle gaps in data (NaNs) by plotting segments
            if metrics_df[col].isnull().any():
                y = metrics_df[col]
                x = metrics_df[epoch_col]
                nan_indices = np.where(np.isnan(y))[0]
                segments_x = np.split(x.to_numpy(), nan_indices)
                segments_y = np.split(y.to_numpy(), nan_indices)
                for i, (x_seg, y_seg) in enumerate(zip(segments_x, segments_y)):
                    if i > 0:
                        y_seg = y_seg[1:]
                        x_seg = x_seg[1:]
                    if len(y_seg) > 0:
                        plt.plot(x_seg, y_seg, marker='o', linestyle='-')
            else:
                 plt.plot(metrics_df[epoch_col], metrics_df[col], marker='o', linestyle='-')

            plt.title(f'{experiment_name} - {name}', fontsize=16)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / f"{experiment_name}_curves_{name}.png")
            plt.close()

def create_summary_visual(summary_data: Dict[str, Dict], save_path: Path, experiment_name: str):
    """
    Creates a visual table summary from the S8 validation data.
    Displays metrics related to Jacobian, Lyapunov stability, Error Bounds, and Multi-Step Decay.

    Args:
        summary_data (Dict): Data loaded from summary.json.
        save_path (Path): Path to save the image.
        experiment_name (str): Experiment name for the title.
    """
    results = summary_data.get("results", {})
    
    table_data = []
    # Define sections and their corresponding JSON keys
    sections = {
        "Jacobian": [
            ("Mean Norm", "jacobian.mean"), ("Std Dev", "jacobian.std"),
            ("Max Norm", "jacobian.max"), ("99th Percentile", "jacobian.percentile_99"),
            ("Converged Rate", "jacobian.converged_rate"), ("Avg Iterations", "jacobian.avg_iterations")
        ],
        "Lyapunov": [
            ("CLF Satisfied (Pred)", "lyapunov.clf_satisfaction_rate_pred"),
            ("CLF Satisfied (True)", "lyapunov.clf_satisfaction_rate_true"),
            ("Mean Decay (Pred)", "lyapunov.mean_decay_rate_pred"),
            ("Mean Decay (True)", "lyapunov.mean_decay_rate_true"),
            ("Theoretical Decay", "lyapunov.theoretical_decay")
        ],
        "Error Bounds": [
            ("Max E-Norm", "error_bounds.max_e_norm"),
            ("99th Percentile E-Norm", "error_bounds.percentile_99_e_norm")
        ],
        "Multi-Step Decay": [
            ("Monotonic Rate", "multistep.monotonic_rate"),
            ("Mean Decay Deviation", "multistep.mean_decay_deviation"),
            ("Cumulative Deviation", "multistep.cumulative_deviation")
        ]
    }
    
    # Populate table data
    for section_name, metrics in sections.items():
        table_data.append([f'-- {section_name} --', ''])
        for metric_name, data_path in metrics:
            keys = data_path.split('.')
            value = results
            try:
                for key in keys: value = value[key]
                # Format float values
                table_data.append([metric_name, f"{value:.4f}" if isinstance(value, float) else value])
            except (KeyError, TypeError):
                table_data.append([metric_name, "N/A"])
    
    # Draw table using matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], loc='center', cellLoc='left', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Styling: Bold headers and section dividers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        if '--' in cell.get_text().get_text():
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')

    plt.title(f'S8 Validation Summary: {experiment_name}', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Suppress matplotlib warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

    # Setup plotting aesthetics
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Verdana', 'Times New Roman', 'Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # Optional cleanup step based on user input
    user_input = input("--- Would you like to delete all existing 'analysis_plots' folders first? [y/N]: ")
    if user_input.lower() == 'y':
        print("\n--- Starting Cleanup Phase ---")
        print(f"Scanning for 'analysis_plots' in: {base_dir}")
        deleted_count = delete_all_analysis_plots(base_dir)
        print(f"\nCleanup Complete: Deleted {deleted_count} 'analysis_plots' directories.\n")

    print("--- Starting Analysis Phase ---")
    # Trigger the batch processing of experiments
    scan_and_process_runs(base_dir, EXPERIMENTS)
    print("\nAll batch processing complete!")
