# -*- coding: utf-8 -*-
"""
scripts/01_generate_training_data.py

[PAEC Framework - Phase 1: Data Generation]

This script is the entry point for generating the training dataset required to train
the Dynamics Model (T_theta). It executes the translation pipeline using a variety
of heuristic strategies (policies) within a simulated production environment.

The process involves:
1. Iterating through defined decoding strategies (e.g., beam search with various widths).
2. Initializing the DataGenerationPipeline, which orchestrates the interaction between
   the NMT model, the kNN datastore, and the ProductionConstraintSimulator.
3. Running translation samples where heuristic policies make decisions (A_t) based on
   simulated system states (S_t).
4. Logging the resulting state transitions (S_t, A_t, S_t+1) and outcomes.
5. Consolidating all generated trajectory data into a single CSV file for subsequent training.
"""

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import pandas as pd
from typing import List, Dict, Tuple, Optional
import argparse

# --- Path Setup ---
# Add the 'src' directory to the system path to enable importing project modules
# regardless of the current working directory.
SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_PATH not in sys.path: sys.path.append(SRC_PATH)

# --- Project Imports ---
# config: Contains global hyperparameters, paths, and simulator settings.
# DecodingStrategy: Enum defining supported decoding methods (e.g., BEAM_SEARCH).
# DataGenerationPipeline: The core controller that runs the simulation loop.
# InvertableColumnTransformer: Used here to ensure dependency availability, though mainly for model training.
from src import config
from src.core import DecodingStrategy
from src.pipeline import DataGenerationPipeline
from t_train_Transformer import InvertableColumnTransformer

def generate_training_data_only(
    strategies_to_test: List[Tuple[DecodingStrategy, Dict]],
    num_samples: int, output_filename: str,
    policies_mix: Optional[Dict[str, float]] = None,
    is_debug: bool = False,
    enable_tqdm: bool = True
):
    """
    Executes the data generation pipeline for a list of specified decoding strategies.
    
    This function acts as a high-level driver that:
    1. Initializes a separate pipeline for each strategy configuration.
    2. Generates 'num_samples' of translation trajectories.
    3. Aggregates the resulting dataframes from all strategies.
    4. Saves the consolidated dataset to disk.

    Args:
        strategies_to_test: A list of tuples, each containing a DecodingStrategy enum
                            and a dictionary of parameters (e.g., beam_size).
        num_samples: The number of source sentences to process per strategy.
        output_filename: The absolute path where the final CSV will be saved.
        policies_mix: A dictionary defining the probability distribution of heuristic
                      policies (e.g., Balanced, QualityFirst) used during generation.
        is_debug: If True, enables verbose logging for debugging purposes.
        enable_tqdm: If True, displays progress bars during generation.
    """
    
    print("="*80)
    print("🔬 Starting Data Generation for Dynamics Model Training")
    print(f"   [Target File]: {output_filename}")
    print("="*80)

    # List to hold the DataFrames generated from each strategy run.
    all_step_data = []

    # Iterate through each defined decoding strategy configuration.
    for strategy, params in strategies_to_test:
        # Create a human-readable label for the strategy (e.g., "beam_search_b4").
        beam_size = params.get('beam_size', 1)
        strategy_label = f"beam_search_b{beam_size}" if beam_size > 1 else "greedy"
        
        print(f"\n{'='*60}")
        print(f"📊 Processing Strategy: {strategy.value} (Label: {strategy_label})")
        print(f"   Parameters: {params}")
        print(f"{'='*60}")

        try:
            # Initialize the pipeline. This sets up the NMT model, kNN system, and
            # the ProductionConstraintSimulator.
            # optimize_simulator_params=True allows the simulator to auto-tune its
            # parameters (alpha, beta, etc.) to match realistic traffic distributions.
            pipeline = DataGenerationPipeline(
                decoding_strategy=strategy,
                **params,
                policies_mix=policies_mix,
                is_debug=is_debug,
                optimize_simulator_params=config.PRODUCTION_SIMULATOR_PARAMS_RESET
            )
            
            # Execute the simulation loop to generate trajectory data.
            # This calls the internal logic that samples traffic, computes states (E, P, H),
            # executes heuristic policies, and records the results.
            strategy_df = pipeline.generate_sample_data(num_samples=num_samples)

            # Clean up resources (GPU memory, FAISS indices) after the run.
            pipeline.teardown()
            
            if not strategy_df.empty:
                # Tag the data with the strategy label for later analysis.
                strategy_df['Strategy'] = strategy_label
                all_step_data.append(strategy_df)
                print(f"\n[SUCCESS] Strategy '{strategy_label}' complete. Generated {len(strategy_df)} data points.")
            else:
                print(f"\n[WARNING] Strategy '{strategy_label}' produced no data.")
                

        except Exception as e:
            # Catch and log any runtime errors to prevent the entire batch from failing.
            print(f"\n[ERROR] An error occurred while processing strategy '{strategy_label}': {e}")
            import traceback
            traceback.print_exc()
            exit(1)

    # --- Final Data Consolidation and Saving ---
    if not all_step_data:
        print("\n[ERROR] No data was generated across all strategies. Aborting save.")
        return

    # Concatenate all individual DataFrames into one large training set.
    final_df = pd.concat(all_step_data, ignore_index=True)

    # Ensure the target directory structure exists.
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Persist the final dataset to CSV.
    final_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*80)
    print("[SUCCESS] Data Generation Task Finished!")
    print(f"\t- Total data points generated: {len(final_df)}")
    print(f"\t- Number of strategies processed: {len(strategies_to_test)}")
    print(f"\t- Output file saved to: {output_filename}")
    print("="*80)


if __name__ == "__main__":
    # Setup argument parser to handle optional debug flag.
    argparser = argparse.ArgumentParser(description="Generate training data for the dynamics model.")
    argparser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging.")
    
    args = argparser.parse_args()
    is_debug = args.debug or False
    
    # --- Configuration Setup ---
    # Retrieve the sample count defined in the global configuration.
    # This determines how many sentences are processed per strategy run.
    NUM_SAMPLES_PER_STRATEGY = config.EXPERIMENT_PARAMS[f'num_samples_per_strategy_{config.EXPERIMENT_PARAMS["using_num_samples"]}']
    
    # Calculate a unique hash of the current configuration (data loader, simulator, policies).
    # This hash is used to version the generated data and ensure reproducibility.
    _config_dict = OrderedDict([
        ("DATA_LOADER_PARAMS", dict(sorted(config.DATA_LOADER_PARAMS.items()))),
        ("SIMULATOR_PARAMS", dict(sorted(config.SIMULATOR_PARAMS.items()))),
        ("POLICIES_MIX", dict(sorted(config.POLICIES_MIX.items()))),
    ])
    CONFIG_HASH = hashlib.sha256(
        json.dumps(_config_dict, sort_keys=False, indent=None).encode()
    ).hexdigest()
    del _config_dict
    config.CONFIG_HASH = CONFIG_HASH

    # Prepare the list of decoding strategies to simulate.
    # Diverse strategies (different beam sizes) ensure the dynamics model learns from
    # a wide variety of translation behaviors and state trajectories.
    DECODING_STRATEGIES = config.EXPERIMENT_PARAMS["decoding_strategies_to_test"]
    STRATEGIES_FOR_TRAINING = []
    for decoding_strategy in DECODING_STRATEGIES:
        STRATEGIES_FOR_TRAINING.append(
            (DecodingStrategy.BEAM_SEARCH, {
                'beam_size': decoding_strategy['beam_size'],
                'length_penalty': decoding_strategy['length_penalty'],
                'use_datastore': config.USE_REAL_DATASTORE,
                'datastore_path': config.PATHS["datastore_dir"],
                # In this phase, we are generating training data, not evaluating.
                # Therefore, we do not load pre-trained dynamics or policy models yet.
                'evaluation_mode': False,
                'dynamics_model_dir': None,
                'offline_policy_model_dir': None
            })
        )

    # Define the output directory and filename based on the sample size configuration.
    OUTPUT_DIR = config.PATHS["processed_data_dir"] / 'training_data_stepwise'
    strategy_key = config.EXPERIMENT_PARAMS['using_num_samples']
    num_samples = config.EXPERIMENT_PARAMS[f"num_samples_per_strategy_{strategy_key}"]
    OUTPUT_FILENAME = OUTPUT_DIR / f'strategy_comparison_stepwise_{num_samples}.csv'

    # Create the output directory and archive the current configuration file.
    # This is crucial for experiment tracking, allowing future reference to the exact
    # parameters used to generate this dataset.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SRC_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'src' / 'config.py'
    DEST_CONFIG_PATH = OUTPUT_DIR / 'config.py'
    if DEST_CONFIG_PATH.exists():
        os.remove(str(DEST_CONFIG_PATH))
    shutil.copy(str(SRC_CONFIG_PATH), str(DEST_CONFIG_PATH))
    print(f"[Successful] Copied config file to: {DEST_CONFIG_PATH}")

    # --- Execute Data Generation ---
    # Trigger the main generation logic with the prepared configurations.
    generate_training_data_only(
        strategies_to_test=STRATEGIES_FOR_TRAINING,
        num_samples=NUM_SAMPLES_PER_STRATEGY,
        output_filename=str(OUTPUT_FILENAME),
        policies_mix=config.POLICIES_MIX,
        is_debug=is_debug
    )
 
    # Write a completion marker file to indicate the process finished successfully.
    processed_data_dir = config.PATHS.get("processed_data_dir", './drive/MyDrive/PAEC_proj/data/processed')
    os.makedirs(processed_data_dir, exist_ok=True)
    with open(os.path.join(processed_data_dir, 'complete_all.txt'), 'w') as file: file.write(f'{1}\n')
