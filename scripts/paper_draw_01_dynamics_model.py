# -*- coding: utf-8 -*-
# /scripts/paper_draw_01_dynamics_model.py

import os, sys, traceback, json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

# --- CONFIGURATION ---

try:
    # Resolve the project root directory to allow absolute imports from 'src'
    project_root = Path(__file__).parent.parent.resolve()
    sys.path.append(str(project_root))
    from src.config import *

except ImportError:
    print("[Error] Could not import elements from src.config.")
    print("  Ensure 'src/config.py' exists.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"[Error] An unexpected error occurred during config import: {e}")
    traceback.print_exc()
    sys.exit(1)

# List of experiments to analyze, mapping display names to their relative directory paths.
# These correspond to the folder structure created during the training phases.
# The experiments are organized by Milestones, reflecting the ablation study progression.
EXPERIMENTS = [

    # Milestone 0: The baseline unconstrained Transformer model
    ("00_Base", "00_Base"),

    # Milestone 1: Continuity Analysis (Lipschitz constraints)
    # 01_OnlySN: Spectral Normalization only
    # 01_SN_jac_1e-3: Spectral Normalization + Jacobian Regularization
    ("01_OnlySN", "01_Continuity/01_OnlySN"),
    ("01_SN_jac_1e-3", "01_Continuity/02_SN_1e-3"),

    # Milestone 2: Accuracy Improvements (Architecture & Training Dynamics)
    # Includes learnable P matrix, text embeddings, uncertainty weights, separate heads for E and H, delta prediction, etc.
    ("02_LearnP_Scheduler", "02_Accuracy/01_Learn_P_lr_Scheduler"),
    ("02_Text_Decoder", "02_Accuracy/02_TextEmb_DecoderState"),
    ("02_Uncertainty", "02_Accuracy/03_UncertaintyWeights"),
    ("02_SeparateHeadEH", "02_Accuracy/04_SeparateHeadEH"),
    ("02_Predict_Delta", "02_Accuracy/05_Predict_Delta"),
    ("02_MultiHead", "02_Accuracy/06_MultiHead"),

    # Milestone 3: Stability Mechanisms
    # 3.1: Single-Step Control Lyapunov Function (CLF) with varying weights
    ("03_Single_CLF_0.4", "03_Stability/01_Single_CLF/01_CLF_0.4"),
    ("03_Single_CLF_0.8", "03_Stability/01_Single_CLF/02_CLF_0.8"),
    ("03_Single_CLF_1.2", "03_Stability/01_Single_CLF/03_CLF_1.2"),

    # 3.2: Multi-Step (N-Step) CLF
    ("03_NCLF_0.7", "03_Stability/02_NCLF/01_NCLF_0.7"),
    ("03_NCLF_1.0", "03_Stability/02_NCLF/02_NCLF_1.0"),
    ("03_NCLF_1.3", "03_Stability/02_NCLF/03_NCLF_1.3"),
    ("03_Gumbel_Softmax", "03_Stability/02_NCLF/04_Gumble_Softmax"),

    # 3.3: Epsilon-Greedy Exploration Strategies
    ("03_Epsilon_Default", "03_Stability/03_epsilon_Greedy/01_default"),
    ("03_Epsilon_PolicyEntropy_0.01", "03_Stability/03_epsilon_Greedy/02_PolicyEntropy_0.01"),
    ("03_Epsilon_PolicyEntropy_0.1", "03_Stability/03_epsilon_Greedy/03_PolicyEntropy_0.1"),

    # 3.4: BPTT Window Variations (Backpropagation Through Time)
    ("03_BPTT_H5_W2", "03_Stability/04_bptt/01_H_5_bptt_2"),
    ("03_BPTT_H6_W3", "03_Stability/04_bptt/02_H_6_bptt_3"),
    ("03_BPTT_H8_W4", "03_Stability/04_bptt/03_H_8_bptt_4"),

    # 3.5: Conditional Value-at-Risk (CVaR) Loss
    ("03_CVaR_alpha_0.7", "03_Stability/05_CVaR/01_alpha_0.7"),
    ("03_CVaR_alpha_1.0", "03_Stability/05_CVaR/02_alpha_1.0"),
    ("03_CVaR_alpha_1.3", "03_Stability/05_CVaR/03_alpha_1.3"),

    # 3.6.1: Control Barrier Function (CBF) Safety Constraints
    ("03_CBF_L0.1_C0.7_A0.5", "03_Stability/061_CBF/01_λ_0.1_crit_0.7_alpha_0.5"),
    ("03_CBF_L0.5_C0.7_A0.5", "03_Stability/061_CBF/02_λ_0.5_crit_0.7_alpha_0.5"),
    ("03_CBF_L1.0_C0.7_A0.5", "03_Stability/061_CBF/03_λ_1.0_crit_0.7_alpha_0.5"),
    ("03_CBF_L0.5_C0.7_A0.1", "03_Stability/061_CBF/04_λ_0.5_crit_0.7_alpha_0.1"),
    ("03_CBF_L0.5_C0.7_A0.9", "03_Stability/061_CBF/05_λ_0.5_crit_0.7_alpha_0.9"),
    ("03_CBF_L0.5_C0.5_A0.5", "03_Stability/061_CBF/06_λ_0.5_crit_0.5_alpha_0.5"),
    ("03_CBF_L0.5_C0.9_A0.5", "03_Stability/061_CBF/07_λ_0.5_crit_0.9_alpha_0.5"),

    # 3.6.2: Cox Proportional Hazards Loss
    ("03_Cox_L0.5_T0.5", "03_Stability/062_Cox/01_λ_0.5_event_thr_0.5"),
    ("03_Cox_L0.5_T0.8", "03_Stability/062_Cox/02_λ_0.5_event_thr_0.8"),
    ("03_Cox_L1.0_T0.5", "03_Stability/062_Cox/03_λ_1.0_event_thr_0.5"),
    ("03_Cox_L1.0_T0.8", "03_Stability/062_Cox/04_λ_1.0_event_thr_0.8"),

    # 3.6.3: Action Dissimilarity Term (ADT) Loss
    ("03_ADT_L0.7", "03_Stability/063_ADT/01_λ_0.7"),
    ("03_ADT_L1.0", "03_Stability/063_ADT/02_λ_1.0"),
    ("03_ADT_L1.3", "03_Stability/063_ADT/03_λ_1.3"),

    # Milestone 4: Curriculum Learning Strategy
    ("04_Curriculum_10", "04_Curriculum/01_phase1_epochs_10"),
    ("04_No_Curriculum", "04_Curriculum/02_no_curriculum"),

    # Final Champion Model
    ("Champion", "Champion")
]

# --- PLOTTING FUNCTIONS ---

def plot_milestone_1_continuity(experiments_data, save_dir, data_save_dir):
    """
    Generates a violin plot comparing the Jacobian Spectral Norm distribution across Milestone 1 models.
    This validates the effectiveness of Spectral Normalization in enforcing Lipschitz continuity.
    """
    plot_data = []
    # Identify relevant experiments for this milestone
    exp_names = ["00_Base", "01_OnlySN", "01_SN_jac_1e-3"]
    for exp_name in exp_names:
        if exp_name in experiments_data and 'jacobian_df' in experiments_data[exp_name]:
            # Load the Jacobian validation data CSV
            df = experiments_data[exp_name]['jacobian_df'].copy()
            df['experiment'] = exp_name
            plot_data.append(df)
    if not plot_data:
        print("Skipping Milestone 1 plot: No data found.")
        return
    # Combine data and save source CSV
    full_df = pd.concat(plot_data, ignore_index=True)
    full_df.to_csv(data_save_dir / 'milestone_1_continuity_data.csv', index=False)

    # Generate Violin Plot
    plt.figure(figsize=(12, 7))
    sns.violinplot(data=full_df, x='experiment', y='final_sigma', cut=0)
    plt.title('Milestone 1: Jacobian Spectral Norm Distribution', fontsize=16)
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel(r'Jacobian Spectral Norm ($\sigma_{\max}$)', fontsize=12)
    # Add reference line for theoretical upper bound (Lipschitz constant = 1)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Theoretical Upper Bound (1.0)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'milestone_1_continuity.png', dpi=300)
    plt.close()

def plot_milestone_2_accuracy(experiments_data, save_dir, data_save_dir):
    """
    Generates a line plot comparing Validation RMSE over epochs for Milestone 2 models.
    This visualizes the impact of various architectural improvements on prediction accuracy.
    """
    # Select experiments focused on accuracy improvements
    exp_names = ["01_OnlySN", "02_LearnP_Scheduler", "02_Text_Decoder", "02_Uncertainty", "02_SeparateHeadEH", "02_Predict_Delta", "02_MultiHead"]
    plot_data = []
    plt.figure(figsize=(14, 8))
    for exp_name in exp_names:
        if exp_name in experiments_data and 'metrics_df' in experiments_data[exp_name]:
            df = experiments_data[exp_name]['metrics_df']
            if 'val.rmse_total' in df.columns:
                # Plot RMSE curve
                plt.plot(df['epoch'], df['val.rmse_total'], label=exp_name, marker='o', linestyle='-')
                temp_df = df[['epoch', 'val.rmse_total']].copy()
                temp_df['experiment'] = exp_name
                plot_data.append(temp_df)
    if not plot_data:
        print("Skipping Milestone 2 plot: No data found.")
        plt.close()
        return
    # Save source data
    pd.concat(plot_data, ignore_index=True).to_csv(data_save_dir / 'milestone_2_accuracy_data.csv', index=False)
    
    # Finalize plot details
    plt.title('Milestone 2: Validation RMSE vs. Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Validation RMSE', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / 'milestone_2_accuracy.png', dpi=300)
    plt.close()

def plot_milestone_3_stability(experiments_data, save_dir, data_save_dir):
    """
    Generates multiple plots to analyze system stability metrics across Milestone 3 models.
    Metrics include CLF satisfaction, Lyapunov drift, Monotonic decay, and N-step violations.
    """
    # Select key representative experiments for stability analysis
    exp_names = [
        "02_MultiHead", "03_NCLF_1.0", "03_Epsilon_PolicyEntropy_0.01", 
        "03_BPTT_H5_W2", "03_CVaR_alpha_0.7", "03_CBF_L0.5_C0.7_A0.5", 
        "03_Cox_L0.5_T0.5", "03_ADT_L1.3", "04_No_Curriculum", "04_Curriculum_10"
    ]
    
    # --- Plot 1: CLF Satisfaction Rate Trend ---
    plot_data_clf = []
    for name in exp_names:
        if name in experiments_data and 'metrics_df' in experiments_data[name]:
            df = experiments_data[name]['metrics_df']
            if 'val.clf_violation_rate' in df.columns:
                # Calculate Satisfaction Rate = 1 - Violation Rate
                temp_df = pd.DataFrame({
                    'epoch': df['epoch'],
                    'CLF Satisfaction Rate': 1 - df['val.clf_violation_rate'],
                    'Experiment': name
                })
                plot_data_clf.append(temp_df)
    if plot_data_clf:
        full_df_clf = pd.concat(plot_data_clf, ignore_index=True)
        full_df_clf.to_csv(data_save_dir / 'milestone_3_clf_satisfaction_trend_data.csv', index=False)
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=full_df_clf, x='epoch', y='CLF Satisfaction Rate', hue='Experiment', marker='o', palette='turbo')
        plt.title('Milestone 3: CLF Satisfaction Rate vs. Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('CLF Satisfaction Rate (1 - Violation Rate)', fontsize=12)
        plt.grid(True, which='both', linestyle='--')
        plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / 'milestone_3_clf_satisfaction_trend.png', dpi=300)
        plt.close()

    # --- Plot 2: Lyapunov Positive Drift Rate Trend ---
    plot_data_pdr = []
    for name in exp_names:
        if name in experiments_data and 'metrics_df' in experiments_data[name]:
            df = experiments_data[name]['metrics_df']
            if 'val.lyapunov_pos_rate_mean' in df.columns:
                temp_df = df[['epoch', 'val.lyapunov_pos_rate_mean']].copy()
                temp_df.rename(columns={'val.lyapunov_pos_rate_mean': 'Lyapunov Positive Drift Rate'}, inplace=True)
                temp_df['Experiment'] = name
                plot_data_pdr.append(temp_df)
    if plot_data_pdr:
        full_df_decay = pd.concat(plot_data_pdr, ignore_index=True)
        full_df_decay.to_csv(data_save_dir / 'milestone_3_mean_decay_trend_data.csv', index=False)
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=full_df_decay, x='epoch', y='Lyapunov Positive Drift Rate', hue='Experiment', marker='o', palette='turbo')
        plt.title('Milestone 3: Lyapunov Positive Drift Rate vs. Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(r'Positive Drift Probability $\left(\mathbb{P}\left[\Delta V > 0\right]\right)$', fontsize=12)
        plt.grid(True, which='both', linestyle='--')
        plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / 'milestone_3_mean_decay_trend.png', dpi=300)
        plt.close()

    # --- Plot 3: Monotonic Decay Rate Comparison (Bar Chart) ---
    plot_data_mono = []
    for name in exp_names:
        if name in experiments_data and 'summary' in experiments_data[name]:
            # Extract final summary metrics calculated by S8 validation
            summary = experiments_data[name]['summary']['results']
            mono_rate = summary.get('multistep', {}).get('monotonic_rate', np.nan)
            plot_data_mono.append({'Experiment': name, 'Monotonic Decay Rate': mono_rate})
    if plot_data_mono:
        df_mono = pd.DataFrame(plot_data_mono).dropna()
        df_mono.to_csv(data_save_dir / 'milestone_3_monotonic_decay_comparison_data.csv', index=False)
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=df_mono, x='Experiment', y='Monotonic Decay Rate', hue='Experiment', palette='viridis', dodge=False)
        plt.title('Milestone 3: Final Monotonic Decay Rate Comparison', fontsize=16)
        plt.xlabel('Experiment', fontsize=12)
        plt.ylabel('Monotonic Decay Rate', fontsize=12)
        plt.xticks(rotation=15, ha='right')
        # Annotate bars with values
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        plt.tight_layout()
        plt.savefig(save_dir / 'milestone_3_monotonic_decay_comparison.png', dpi=300)
        plt.close()

    # --- Plot 4: N-Step Endpoint Violation Rate Trend ---
    plot_data_nstep = []
    for name in exp_names:
        if name in experiments_data and 'metrics_df' in experiments_data[name]:
            df = experiments_data[name]['metrics_df']
            if 'val.nstep_endpoint_violation_rate' in df.columns:
                temp_df = df[['epoch', 'val.nstep_endpoint_violation_rate']].copy()
                temp_df.rename(columns={'val.nstep_endpoint_violation_rate': 'N-Step Violation Rate'}, inplace=True)
                temp_df['Experiment'] = name
                plot_data_nstep.append(temp_df)
    if plot_data_nstep:
        full_df_nstep = pd.concat(plot_data_nstep, ignore_index=True)
        full_df_nstep.to_csv(data_save_dir / 'milestone_3_nstep_violation_trend_data.csv', index=False)
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=full_df_nstep, x='epoch', y='N-Step Violation Rate', hue='Experiment', marker='o', palette='turbo')
        plt.title('Milestone 3: N-Step Endpoint Violation Rate (Endpoint) vs. Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('N-Step Violation Rate (Endpoint)', fontsize=12)
        plt.grid(True, which='both', linestyle='--')
        plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / 'milestone_3_nstep_violation_trend.png', dpi=300)
        plt.close()
    
    # --- Plot 5: Multi-Step Negative Drift Coverage Trend ---
    # Measures how often the Lyapunov function decreases over a multi-step horizon
    plot_data_nstep = []
    for name in exp_names:
        if name in experiments_data and 'metrics_df' in experiments_data[name]:
            df = experiments_data[name]['metrics_df']
            if 'val.multi_step_neg_drift_coverage' in df.columns:
                temp_df = df[['epoch', 'val.multi_step_neg_drift_coverage']].copy()
                temp_df.rename(columns={'val.multi_step_neg_drift_coverage': 'Multi-Step Negative Drift Coverage'}, inplace=True)
                temp_df['Experiment'] = name
                plot_data_nstep.append(temp_df)
    if plot_data_nstep:
        full_df_nstep = pd.concat(plot_data_nstep, ignore_index=True)
        full_df_nstep.to_csv(data_save_dir / 'milestone_3_multistep_neg_drift_coverage.csv', index=False)
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=full_df_nstep, x='epoch', y='Multi-Step Negative Drift Coverage', hue='Experiment', marker='o', palette='turbo')
        plt.title('Milestone 3: Multi-Step Negative Drift Coverage vs. Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Multi-Step Negative Drift Coverage', fontsize=12)
        plt.grid(True, which='both', linestyle='--')
        plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / 'milestone_3_multistep_neg_drift_coverage.png', dpi=300)
        plt.close()

def plot_milestone_4_curriculum(experiments_data, save_dir, data_save_dir):
    """
    Generates a line plot comparing RMSE convergence between Curriculum Learning and No Curriculum approaches.
    Marks the start of Phase 2 (stability constraints) for the curriculum model.
    """
    exp_names = [
        "04_No_Curriculum",
        "04_Curriculum_10"
    ]
    plot_data = []
    plt.figure(figsize=(14, 8))
    for exp_name in exp_names:
        if exp_name in experiments_data and 'metrics_df' in experiments_data[exp_name]:
            df = experiments_data[exp_name]['metrics_df'].copy()
            if 'val.rmse_total' in df.columns:
                p = plt.plot(df['epoch'], df['val.rmse_total'], label=exp_name, marker='o', linestyle='-')
                # Add vertical line indicating the start of Phase 2 for curriculum models
                if 'config' in experiments_data[exp_name] and 'curriculum_phase1_epochs' in experiments_data[exp_name]['config']:
                    phase1_epochs = experiments_data[exp_name]['config']['curriculum_phase1_epochs']
                    if phase1_epochs != 0:
                        plt.axvline(x=phase1_epochs, linestyle='--', color=p[0].get_color(), label=f'{exp_name} Phase 2 Start')
                temp_df = df[['epoch', 'val.rmse_total']].copy(); temp_df['experiment'] = exp_name
                plot_data.append(temp_df)
    if not plot_data:
        print("Skipping Milestone 4 plot: No data found.")
        plt.close()
        return
    # Save source data
    pd.concat(plot_data, ignore_index=True).to_csv(data_save_dir / 'milestone_4_curriculum_data.csv', index=False)
    
    # Finalize plot
    plt.title('Milestone 4: Curriculum Learning vs. No Curriculum', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Validation RMSE', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / 'milestone_4_curriculum.png', dpi=300)
    plt.close()

def plot_champion_summary(experiments_data, save_dir, data_save_dir):
    """
    Generates a Radar Chart summarizing the performance of top models across multiple dimensions.
    Features:
    - Dynamic normalization for axes to highlight subtle differences.
    - Decomposition of R^2 scores into 8 component dimensions.
    - Comparison of Accuracy, Stability, and Robustness metrics.
    """
    exp_names = [
        "03_BPTT_H5_W2", "03_CVaR_alpha_0.7",
        "03_CBF_L0.5_C0.7_A0.5", "03_Cox_L0.5_T0.5", "03_ADT_L1.3",
        "04_Curriculum_10", "04_No_Curriculum",
    ]
    
    # Define R^2 components and map them to readable labels
    r2_components = {
        'val.r2_error_semantic': 'R² Semantic',
        'val.r2_error_coverage': 'R² Coverage',
        'val.r2_error_fluency_surprisal': 'R² Surprisal',
        'val.r2_error_fluency_repetition': 'R² Repetition',
        'val.r2_context_faith_focus': 'R² Faithfulness',
        'val.r2_context_consistency': 'R² Consistency',
        'val.r2_context_stability': 'R² Stability',
        'val.r2_context_confidence_volatility': 'R² Volatility'
    }

    # Define all metrics to plot and their optimization direction
    metrics_to_plot = {
        'Avg. Jacobian Norm': {'lower_is_better': True},
        'CLF Satisfaction': {'lower_is_better': False},
        'Monotonic Decay Rate': {'lower_is_better': False}
    }
    for short_name in r2_components.values():
        metrics_to_plot[short_name] = {'lower_is_better': False}

    # Extract metrics for each experiment
    plot_data = []
    for exp_name in exp_names:
        if exp_name not in experiments_data:
            print(f"Warning: Data for '{exp_name}' not found. Skipping.")
            continue

        row = {'Experiment': exp_name}
        metrics_df = experiments_data[exp_name].get('metrics_df')
        summary = experiments_data[exp_name].get('summary', {}).get('results', {})

        # Prefer final epoch data from metrics log
        if metrics_df is not None and not metrics_df.empty:
            last_epoch_metrics = metrics_df.iloc[-1]
            
            for col_name, short_name in r2_components.items():
                row[short_name] = last_epoch_metrics.get(col_name, np.nan)
            
            row['Avg. Jacobian Norm'] = last_epoch_metrics.get('val.jacobian_spectral_norm_mean', 
                                                               summary.get('jacobian', {}).get('mean', np.nan))
            row['CLF Satisfaction'] = summary.get('lyapunov', {}).get('clf_satisfaction_rate_pred', np.nan)
        else:
            # Fallback to summary.json if metrics log is missing
            print(f"Warning: metrics.jsonl not found for {exp_name}. Using summary.json for all metrics.")
            for short_name in r2_components.values():
                row[short_name] = np.nan
            row['Avg. Jacobian Norm'] = summary.get('jacobian', {}).get('mean', np.nan)
            row['CLF Satisfaction'] = summary.get('lyapunov', {}).get('clf_satisfaction_rate_pred', np.nan)

        # Monotonic decay rate is always in summary
        row['Monotonic Decay Rate'] = summary.get('multistep', {}).get('monotonic_rate', np.nan)
        plot_data.append(row)
    
    if not plot_data:
        print("Skipping Champion summary plot: No data found.")
        return

    df = pd.DataFrame(plot_data).set_index('Experiment')
    
    # Define a logical visual order for the radar axes
    column_order = [
        'Avg. Jacobian Norm',
        'R² Semantic', 'R² Coverage', 'R² Surprisal', 'R² Repetition',
        'CLF Satisfaction',
        'R² Faithfulness', 'R² Consistency', 'R² Stability', 'R² Volatility',
        'Monotonic Decay Rate'
    ]
    final_column_order = [col for col in column_order if col in df.columns]
    df = df[final_column_order]

    df.to_csv(data_save_dir / 'champion_summary_decomposed_data.csv')
    
    df.dropna(inplace=True)
    if df.empty: 
        print("Skipping Champion summary plot: Not enough complete data after dropping NaNs.")
        return

    # --- Dynamic Normalization Logic ---
    # Normalize each axis to [0, 1] relative to the min/max values in the set.
    # Adds a small buffer (padding) to avoid flat lines if variance is low.
    norm_bounds = {}
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        diff = max_val - min_val
        
        if diff < 1e-9:
            padding = 0.1 * abs(min_val) if abs(min_val) > 1e-9 else 0.1
            norm_bounds[col] = {'min': min_val - padding, 'max': max_val + padding}
        else:
            padding = diff * 0.05
            norm_bounds[col] = {'min': min_val - padding, 'max': max_val + padding}

    normalized_df = df.copy()
    for col, props in metrics_to_plot.items():
        if col not in df.columns: continue
        
        min_b = norm_bounds[col]['min']
        max_b = norm_bounds[col]['max']
        clipped_data = df[col].clip(min_b, max_b)

        if abs(max_b - min_b) < 1e-9:
            normalized_df[col] = 0.5; continue
            
        if props['lower_is_better']:
            normalized_df[col] = (max_b - clipped_data) / (max_b - min_b)
        else:
            normalized_df[col] = (clipped_data - min_b) / (max_b - min_b)

    # --- Generate Radar Plot ---
    labels = normalized_df.columns
    num_vars = len(labels)
    # Calculate angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_closed = angles + [angles[0]] # Close the loop

    fig, ax = plt.subplots(figsize=(18, 18), subplot_kw=dict(polar=True))

    for i, row in normalized_df.iterrows():
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles_closed, values, label=i, marker='o', zorder=10)
        ax.fill(angles_closed, values, alpha=0.1, zorder=10)

    # Configure grid and labels
    grid_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_yticks(grid_levels)
    ax.set_yticklabels([])
    ax.set_ylim(-0.1, 1.15) # Leave space for labels

    # Determine label rotation
    angles_deg = np.rad2deg(np.array(angles))
    left_idx = np.argmin(np.abs(angles_deg - 180))
    right_idx = np.argmin(np.minimum(np.abs(angles_deg - 0), np.abs(angles_deg - 360)))

    # Draw axis labels and value indicators
    for i, (angle, label) in enumerate(zip(angles, labels)):
        # Axis title
        ax.text(angle, 1.22, label.replace(' ', '\n'), ha='center', va='center', fontsize=12, weight='bold')
        
        props = metrics_to_plot[label]
        min_b, max_b = norm_bounds[label]['min'], norm_bounds[label]['max']
        
        rotation = 0
        if i == left_idx: rotation = 90
        if i == right_idx: rotation = -90

        # Draw actual values on the grid lines
        for r in grid_levels:
            if props['lower_is_better']:
                original_value = max_b - r * (max_b - min_b)
            else:
                original_value = min_b + r * (max_b - min_b)
            
            label_text = f"{original_value:.5f}"
            ax.text(angle, r, label_text, ha='center', va='center', fontsize=9, color='darkblue', rotation=rotation,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)

    plt.title('Champion Model Performance Summary (Full R²)', size=22, color='black', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / 'champion_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def load_all_experiment_data(base_dir, experiments_list):
    """
    Traverses the directory structure to load all relevant results for the listed experiments.
    Returns a nested dictionary containing config, metrics DataFrame, summary JSON, and validation CSVs.
    """
    all_data = {}
    for exp_name, rel_path in experiments_list:
        exp_dir = base_dir / rel_path
        if not exp_dir.is_dir():
            print(f"Warning: Experiment directory '{exp_dir}' does not exist. Skipping.")
            continue

        # Look for the summary file in the 'best' epoch folder first
        summary_path = exp_dir / "s8_validation_epoch_best" / "summary.json"
        if not summary_path.exists():
            # If 'best' folder is missing, try finding any other available validation folder
            summary_path = next(exp_dir.glob("s8_validation_epoch_*/summary.json"), None)
            
        if summary_path:
            exp_data = {}
            try:
                # Load summary metrics
                with open(summary_path, 'r') as f: exp_data['summary'] = json.load(f)

                # Load training metrics logs
                metrics_path = exp_dir / 'metrics.jsonl'
                if metrics_path.exists():
                    records = [json.loads(line) for line in open(metrics_path, 'r')]
                    exp_data['metrics_df'] = pd.json_normalize(records)
                
                # Load specific CSV reports (Jacobian, Multistep decay)
                for key, fname in [('jacobian_df', 'jacobian_validation.csv'), ('multistep_df', 'multistep_decay.csv')]:
                    if (csv_path := summary_path.parent / fname).exists():
                        exp_data[key] = pd.read_csv(csv_path)
                    
                # Load model config
                if (config_path := exp_dir / 'config.json').exists():
                    with open(config_path, 'r') as f: exp_data['config'] = json.load(f)

                all_data[exp_name] = exp_data
            except Exception as e:
                print(f"Error loading data for '{exp_name}': {e}")
    return all_data

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Suppress matplotlib warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
    
    # Configure global plotting style
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Verdana']
    plt.rcParams['axes.unicode_minus'] = False

    # Setup output directories
    save_directory = PATHS["results_dir"] / "ablation_plots"
    data_save_directory = save_directory / "data"
    save_directory.mkdir(parents=True, exist_ok=True)
    data_save_directory.mkdir(parents=True, exist_ok=True)
    
    print("Loading all experiment data...")
    # Load all data from the file system
    all_experiments_data = load_all_experiment_data(PATHS["dynamics_model_dir"], EXPERIMENTS)
    print(f"Successfully loaded data for {len(all_experiments_data)} out of {len(EXPERIMENTS)} specified experiments.")

    print("Generating plots and exporting data...")
    # Execute plotting functions for each milestone
    plot_milestone_1_continuity(all_experiments_data, save_directory, data_save_directory)
    plot_milestone_2_accuracy(all_experiments_data, save_directory, data_save_directory)
    plot_milestone_3_stability(all_experiments_data, save_directory, data_save_directory)
    plot_milestone_4_curriculum(all_experiments_data, save_directory, data_save_directory)
    plot_champion_summary(all_experiments_data, save_directory, data_save_directory)
    
    print(f"--- All tasks completed! ---")
    print(f"Plots saved to:      {save_directory.resolve()}")
    print(f"Plot data saved to:    {data_save_directory.resolve()}")
