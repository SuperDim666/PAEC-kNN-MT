# -*- coding: utf-8 -*-
"""
src/pipeline/data_generation_pipeline.py

[PAEC Framework - Data Generation & Evaluation Pipeline]

This module defines the `DataGenerationPipeline` class, which serves as the central
orchestrator for the PAEC system. It handles two distinct modes of operation:

1. Training Data Generation (Phase 1):
   - Simulates a production environment using `ProductionConstraintSimulator`.
   - Iterates through a dataset using diverse heuristic policies (Teacher strategies).
   - Generates trajectories of (State, Action, Next State) tuples to train the Dynamics Model (T_theta).
   - Automatically optimizes simulation parameters to ensure realistic traffic patterns.

2. System Evaluation (Validation Phase):
   - Connects to the `RealtimeResourceMonitor` to measure actual hardware performance.
   - Executes translation tasks using the trained PAEC policies (Online or Offline).
   - Captures detailed metrics (BLEU, Latency, Memory) and step-by-step trajectories.
"""

import time
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Any, Tuple
from difflib import SequenceMatcher
from sacrebleu import sentence_bleu
from sacrebleu.metrics.bleu import BLEUScore
from tqdm import tqdm

import re
import importlib
from itertools import product
from datetime import datetime
from pathlib import Path
import logging

# Configure logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Imports ---
from src import config
from src.core import SystemState, DecodingStrategy
from src.core.structs import Action
from src.simulation.resource_monitor import RealtimeResourceMonitor
from src.system import kNNMTSystem
from src.simulation import ProductionConstraintSimulator
from src.data_processing import RealDatasetLoader


class DataGenerationPipeline:
    """
    Orchestrates the end-to-end process of translation, resource simulation, and data logging.

    This class integrates the Neural Machine Translation (NMT) system, the kNN retrieval module,
    and the resource monitoring/simulation components. It is capable of generating large-scale
    synthetic training data under simulated constraints or evaluating the system's performance
    under real-world constraints.
    """

    def __init__(
        self,
        decoding_strategy: DecodingStrategy,
        beam_size: int = 3,
        length_penalty: float = 1.0,
        use_datastore: bool = False,
        datastore_path: Optional[str] = None,
        evaluation_mode: bool = False,
        dynamics_model_dir: Optional[str] = None,
        offline_policy_model_dir: Optional[str] = None,
        max_length: int = 64,
        policies_mix: Optional[Dict[str, float]] = None,
        is_debug: bool = False,
        optimize_simulator_params: bool = False
    ):
        """
        Initializes the Data Generation Pipeline.

        Args:
            decoding_strategy (DecodingStrategy): The algorithm used for text generation (e.g., Beam Search).
            beam_size (int): Number of beams for beam search.
            length_penalty (float): Penalty factor for sequence length during scoring.
            use_datastore (bool): Whether to enable the kNN datastore.
            datastore_path (Optional[str]): Path to the FAISS index and datastore files.
            evaluation_mode (bool): If True, runs in real-time evaluation mode. If False, runs in data generation mode.
            dynamics_model_dir (Optional[str]): Path to the trained dynamics model (T_theta).
            offline_policy_model_dir (Optional[str]): Path to the trained policy network (Pi_phi).
            max_length (int): Maximum length of generated sequences.
            policies_mix (Optional[Dict]): Distribution of heuristic policies to use during generation.
            is_debug (bool): Enable verbose debug logging.
            optimize_simulator_params (bool): If True, runs a grid search to tune simulator parameters before generation.
        """
        self.is_debug = is_debug
        
        # If requested, optimize the traffic simulation parameters to match target distributions
        if optimize_simulator_params and not evaluation_mode:
            self._optimize_simulator_params()
        
        print("="*60)
        print("[Begin]  Initializing Data Generation Pipeline...")
        self.evaluation_mode = evaluation_mode
        self.decoding_strategy = decoding_strategy
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.policies_mix = policies_mix if policies_mix else config.POLICIES_MIX
        
        # Initialize the core kNN-MT system which handles the model and datastore interactions
        self.knn_system = kNNMTSystem(
            use_datastore=use_datastore,
            datastore_path=datastore_path,
            evaluation_mode=evaluation_mode,
            dynamics_model_dir=dynamics_model_dir,
            offline_policy_model_dir=offline_policy_model_dir,
            policies_mix=self.policies_mix,
            is_debug=self.is_debug
        )
        
        # Initialize the appropriate resource tracking mechanism based on the mode
        if self.evaluation_mode:
            print("\t[Info] Resource monitor mode: Performance evaluation")
            # In evaluation mode, measure actual system metrics (Real Hardware)
            self.pressure_computer = RealtimeResourceMonitor()
        else:
            print("\t[Info] Resource monitor mode: Training data generation")
            # In data generation mode, simulate production pressure (Virtual Environment)
            self.pressure_computer = ProductionConstraintSimulator(knn_system=self.knn_system, is_debug=self.is_debug)
        
        # Initialize the dataset loader to fetch source sentences
        self.dataset_loader = RealDatasetLoader()

        # Storage for the generated trajectory logs
        self.data_log: List[Dict] = []

        print("[Begin] Loading real translation datasets...")
        # Load the subset of data designated for datastore/generation
        self.real_samples = self.dataset_loader.load_all_datasets(split="train", size="ds")
        if not self.real_samples:
            raise RuntimeError("Failed to load any datasets. Please check network connection or data paths.")
        
        print(f"[Success] Pipeline ready for decoding strategy: {decoding_strategy.value}")
        if decoding_strategy == DecodingStrategy.BEAM_SEARCH:
            print(f"\t- Beam size: {beam_size}")
            print(f"\t- Length penalty: {length_penalty}")
        print("="*60)

    def _update_config_file(self, best_params: Dict[str, float]):
        """
        Updates the global `config.py` file with the optimized simulator parameters.
        
        This ensures that subsequent runs use the parameters that produce the most
        realistic traffic patterns. It performs a safe regex replacement to preserve
        comments and file structure.

        Args:
            best_params (Dict[str, float]): The optimized parameters (alpha, beta, etc.).
        """
        config_path = Path(__file__).resolve().parent.parent / 'config.py'
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Format the new dictionary as a python string
            new_params_str = "PRODUCTION_SIMULATOR_PARAMS = {\n"
            for key, value in best_params.items():
                new_params_str += f"    \"{key}\": {value},\n"
            new_params_str += "}"

            # Regex to find the PRODUCTION_SIMULATOR_PARAMS dictionary in the file
            pattern = re.compile(r"PRODUCTION_SIMULATOR_PARAMS\s*=\s*\{.*?\n\}", re.DOTALL)
            if not pattern.search(content):
                print(f"[Error] Could not find PRODUCTION_SIMULATOR_PARAMS block in {config_path}. Update skipped.")
                return

            # Replace the old block with the new one
            new_content = pattern.sub(new_params_str, content)

            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"[Success] Updated {config_path} with optimized simulator parameters.")

        except Exception as e:
            print(f"[Error] Failed to update config.py: {e}")

    def _optimize_simulator_params(self):
        """
        Performs a grid search to optimize the `ProductionConstraintSimulator` parameters.
        
        Objective:
            Find parameters (alpha, beta, amplitude, noise) that generate latency and
            throughput distributions matching the predefined target goals (e.g., specific
            mean, std, and skewness). This ensures the training data covers a realistic
            and diverse range of pressure states.
        """
        print("\n" + "="*80)
        print("[Begin]  Starting Automatic Simulator Parameter Optimization...")
        print("="*80)
        
        # Target statistical ranges for Latency and Throughput distributions
        target_range = {'latency': (13, 204), 'throughput': (800, 3000)}

        # Define search space for Gamma distribution parameters and noise
        param_grid = {
            'mean_load_alpha': np.round(np.arange(1.5, 4.005, 0.5), 1),
            'mean_load_beta': np.round(np.arange(0.6, 2.005, 0.2), 1),
            'variation_amplitude': np.round(np.arange(0.4, 2.005, 0.2), 1),
            'noise_std': np.round(np.arange(0.05, 0.5, 0.05), 1)
        }
        
        # Fixed bounds for traffic multipliers
        fixed_params = {
            "mean_load_min": 0.2, "mean_load_max": 1.5,
            "traffic_min": 0.05, "traffic_max": 2.0
        }

        N_STEPS = 1000  # Number of simulation steps per configuration
        best_params = None
        best_score = -np.inf  # Score metric to maximize (Combined Standard Deviation)
        
        # Initialize a temporary system just for the simulation
        temp_knn_system = kNNMTSystem(
            use_datastore=True,
            datastore_path=str(config.PATHS["datastore_dir"]),
            evaluation_mode=False,
            policies_mix=config.POLICIES_MIX,
            is_debug=False
        )

        # Iterate over all parameter combinations
        for alpha, beta, amp, noise in product(*param_grid.values()):
            
            simulator = ProductionConstraintSimulator(temp_knn_system, is_debug=False)
            
            # Apply current parameters
            simulator.mean_load_alpha, simulator.mean_load_beta = alpha, beta
            simulator.variation_amplitude, simulator.noise_std = amp, noise
            simulator.mean_load_min, simulator.mean_load_max = fixed_params["mean_load_min"], fixed_params["mean_load_max"]
            simulator.traffic_min, simulator.traffic_max = fixed_params["traffic_min"], fixed_params["traffic_max"]
            
            # Run simulation loop
            for _ in range(N_STEPS):
                # Simulate a 'none' action to observe baseline traffic
                simulator.update_resource_metrics(Action(k=0, index_type='none', lambda_weight=0.0), pressure_norm=np.random.rand())
            
            # Calculate distribution statistics from the buffer
            stats = simulator.compute_distribution_stats(target_range)
            
            ideal_ranges = config.PRODUCTION_SIMULATOR_GOALS
            
            is_valid = True
            if not stats:
                is_valid = False
            else:
                # Validate statistics against all criteria (min, max, mean, std, skew)
                for metric_name, criteria in ideal_ranges.items():
                    for stat_name, (lower_bound, upper_bound) in criteria.items():
                        
                        calculated_value = stats.get(metric_name, {}).get(stat_name, None)

                        if calculated_value is None:
                            if self.is_debug: print(f"\t- [Rejected] Missing statistic: {metric_name} -> {stat_name}")
                            is_valid = False
                            break

                        if not (lower_bound <= calculated_value <= upper_bound):
                            if self.is_debug: print(f"\t- [Rejected] {metric_name.capitalize()} '{stat_name}' ({calculated_value:.2f}) is outside the ideal range [{lower_bound}, {upper_bound}].")
                            is_valid = False
                            break 
                    
                    if not is_valid:
                        break
            
            if is_valid:
                # Score based on coverage (Standard Deviation) to prefer diverse simulations
                score = (stats['latency']['std'] + stats['throughput']['std']) / 2
                if self.is_debug: print(f"\t- [Accepted] Score (Combined Std): {score:.2f}")
                if score > best_score:
                    best_score = score
                    best_params = {
                        "variation_amplitude": amp, "noise_std": noise,
                        "mean_load_alpha": alpha, "mean_load_beta": beta,
                        **fixed_params
                    }
        
        # Clean up temporary resources
        temp_knn_system.teardown()
        del temp_knn_system
        
        # Apply optimal parameters if found
        if best_params:
            print("\n[Info] Optimizer - Best parameters found:")
            print(json.dumps(best_params, indent=4))
            self._update_config_file(best_params)
            # Reload config to apply changes immediately
            importlib.reload(config)
        else:
            print("\n[Warning] Optimizer: No fully valid parameter set found. Using defaults from config.py.")
        
        print("="*80)

    def generate_sample_data(self, num_samples: int, enable_tqdm: bool = True) -> pd.DataFrame:
        """
        Executes the main data generation loop.

        This method:
        1. Selects a random subset of source sentences.
        2. Iterates through them, translating each using the configured strategy.
        3. Records the detailed state-action trajectory for each step.
        4. Handles exceptions to ensure the pipeline is robust.

        Args:
            num_samples (int): Total number of sentences to process.
            enable_tqdm (bool): Whether to show a progress bar.

        Returns:
            pd.DataFrame: A DataFrame containing all logged trajectory steps.
        """
        print(f"[Begin] Starting data generation for {num_samples} samples...")
        print(f"num_samples: {num_samples}\t|\tavailable_samples:{len(self.real_samples)}")
        available_samples = len(self.real_samples)
        num_to_process = min(num_samples, available_samples)

        # Randomly select a subset of samples to process
        rng = np.random.default_rng(config.RANDOM_SEED)
        selected_indices = rng.choice(available_samples, size=num_to_process, replace=False)

        if enable_tqdm:
            iterator = tqdm(selected_indices, total=num_to_process, desc="Generating samples")
        else:
            iterator = selected_indices

        for i, sample_idx in enumerate(iterator):
            sample = self.real_samples[sample_idx]
            source_text, reference_text = sample['source_text'], sample['target_text']
            
            # Reset simulator state for each new sentence to simulate independent requests
            if not self.evaluation_mode:
                self.pressure_computer.reset()
            
            if not enable_tqdm: print(f"\n--- Processing sample {i+1}/{num_to_process} (Dataset: {sample['dataset']}) ---")

            try:
                # Execute translation
                if self.decoding_strategy == DecodingStrategy.BEAM_SEARCH:
                    # translate_with_paec_control handles the step-by-step logic and logging
                    result, trajectory = self.knn_system.translate_with_paec_control(
                        source_text, self.pressure_computer,
                        max_length=self.max_length,
                        beam_size=self.beam_size,
                        length_penalty=self.length_penalty
                    )
                else:
                    raise NotImplementedError(f"Decoding strategy {self.decoding_strategy} not implemented.")
                
                # Process and store the resulting trajectory logs
                for beam_id, steps in trajectory.items():
                    for step_data in steps:
                        self._log_step_data(step_data, result, sample, i, beam_id)
                
                if not enable_tqdm: print(f"[Success] Sample {i+1} processed. Final Translation: {result['translation']} | Skip Rate: {result['knn_skip_rate']:.2f}")

            except Exception as e:
                print(f"[Error] Failed to process sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                exit(1)
        
        if not self.data_log:
            print("[Warning] No data points were generated.")
            return pd.DataFrame()

        return pd.DataFrame(self.data_log)

    def _log_step_data(
        self,
        step_data: Dict, result: Dict,
        sample: Dict, sample_id: int,
        beam_id: int
    ):
        """
        Processes a single step from the translation trajectory and logs it.

        This method constructs the full system state (Error, Pressure, Context),
        simulates the resource impact of the action, performs an optional
        kNN retrieval (if active), and compiles all metrics into a dictionary.

        Args:
            step_data (Dict): Raw data for the current step (from kNN system).
            result (Dict): Final translation result metadata.
            sample (Dict): Original source sample metadata.
            sample_id (int): Unique ID for the sample in this run.
            beam_id (int): ID of the beam this step belongs to.
        """
        
        if not isinstance(self.pressure_computer, ProductionConstraintSimulator):
            raise RuntimeError("Resource metrics can only be updated in data generation mode using simulation.")

        # Reconstruct the SystemState object for this step
        system_state = SystemState(
            error_state=step_data['error_state'],
            pressure_state=step_data['pressure_state'],
            context_state=step_data['context_state'],
            timestamp=step_data.get('timestamp', time.time())
        )
        
        action = step_data['action']

        # Update the simulator to reflect the cost of the chosen action
        # This calculates the pressure state for the *next* step based on current actions
        pressure_norm = system_state.pressure_state.norm()
        resource_metrics = self.pressure_computer.update_resource_metrics(action, pressure_norm)

        # Simulate or perform kNN retrieval to get distance metrics for logging
        # If action dictates no retrieval, return empty results
        retrieval_distances, retrieved_values, retrieval_time = self.knn_system.perform_knn_retrieval(
            step_data['query_embedding'], action
        ) if action.k > 0 and action.index_type != 'none' else (np.array([]), np.array([]), 0.0)

        # Calculate partial BLEU score for the prefix generated so far
        partial_translation = " ".join(step_data['generated_tokens'])
        bleu = sentence_bleu(partial_translation, [sample['target_text']]) if partial_translation else 0.0
        
        if isinstance(bleu, BLEUScore):
            bleu_score = bleu.score
        else:
            bleu_score = 0.0

        # Construct the final log entry
        log_entry = {
            'sample_id': sample_id,
            'step': step_data['step'],
            'beam_id': beam_id,
            'dataset': sample['dataset'],
            'domain': sample['domain'],
            'source_text': sample['source_text'],
            'reference_text': sample['target_text'],
            'generated_prefix': partial_translation,
            'final_translation': result['translation'],
            'bleu_score': bleu_score,
            'decoding_strategy': self.decoding_strategy.value,
            # Decomposed Error State
            'error_semantic': system_state.error_state.error_semantic,
            'error_coverage': system_state.error_state.error_coverage,
            'error_fluency_surprisal': system_state.error_state.error_fluency_surprisal,
            'error_fluency_repetition': system_state.error_state.error_fluency_repetition,
            'error_norm': system_state.error_state.norm(),
            # Decomposed Pressure State
            'pressure_latency': system_state.pressure_state.latency_pressure,
            'pressure_memory': system_state.pressure_state.memory_pressure,
            'pressure_throughput': system_state.pressure_state.throughput_pressure,
            'pressure_norm': pressure_norm,
            # Decomposed Context State
            'context_faith_focus': system_state.context_state.context_faith_focus,
            'context_consistency': system_state.context_state.context_consistency,
            'context_stability': system_state.context_state.context_stability,
            'context_confidence_volatility': system_state.context_state.context_confidence_volatility,
            # Action Parameters
            'action_k': action.k,
            'action_index_type': action.index_type,
            'action_lambda': action.lambda_weight,
            # Physical Resources
            'resource_latency': resource_metrics.get('latency', 0.0),
            'resource_memory': self.pressure_computer.current_total_memory_mb,
            'resource_throughput': resource_metrics.get('throughput', 0.0),
            'retrieval_time': retrieval_time,
            'num_retrieved': len(retrieved_values),
            'min_distance': float(np.min(retrieval_distances)) if len(retrieval_distances) > 0 else 0.0,
            'timestamp': system_state.timestamp
        }
        
        # Add beam-specific metrics if applicable
        if self.decoding_strategy == DecodingStrategy.BEAM_SEARCH:
            log_entry['beam_score'] = step_data.get('beam_score', 0.0)
            log_entry['num_active_beams'] = step_data.get('num_active_beams', 1)
        
        # Serialize the high-dimensional hidden state vector to JSON for storage
        # This allows reconstruction of the state embedding during training
        if 'decoder_hidden_state' in step_data and step_data['decoder_hidden_state'] is not None:
            log_entry['decoder_hidden_state'] = json.dumps(step_data['decoder_hidden_state'].tolist())
        else:
            log_entry['decoder_hidden_state'] = None

        self.data_log.append(log_entry)

    def _calculate_beam_diversity(self, main_translation: str, alternatives: List[str]) -> float:
        """
        Calculates a diversity score representing how different the beam hypotheses are.
        Used for analysis of beam search effectiveness.

        Args:
            main_translation (str): The top hypothesis.
            alternatives (List[str]): Other active hypotheses.

        Returns:
            float: Diversity score (0.0 to 1.0), where 1.0 is completely different.
        """
        if not alternatives:
            return 0.0
        
        # Calculate sequence similarity ratio for each alternative against the main one
        similarities = [SequenceMatcher(None, main_translation, alt).ratio() for alt in alternatives]
        
        # Return average dissimilarity
        return float(np.mean([1.0 - s for s in similarities]))

    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Persists the generated data and summary reports to disk.

        Args:
            df (pd.DataFrame): The accumulated data logs.
            filename (str): Base filename for the output.
        """
        strategy_suffix = f"_{self.decoding_strategy.value}"
        if self.decoding_strategy == DecodingStrategy.BEAM_SEARCH:
            strategy_suffix += f"_beam{self.beam_size}"
        
        base_name = filename.replace('.csv', '')
        csv_path = config.PATHS["processed_data_dir"] / 'training_data_stepwise' / f"{base_name}{strategy_suffix}.csv"
        json_path = config.PATHS["processed_data_dir"] / 'training_data_stepwise' / f"{base_name}{strategy_suffix}.json"
        summary_path = config.PATHS["processed_data_dir"] / 'training_data_stepwise' / f"{base_name}{strategy_suffix}_summary.txt"

        # Save main CSV file
        df.to_csv(csv_path, index=False)
        print(f"[Success] Data successfully saved to: {csv_path}")

        # Save detailed JSON logs (preserves types better than CSV)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data_log, f, indent=2, default=str, ensure_ascii=False)
        print(f"[Success] Detailed log saved to: {json_path}")

        # Generate and save a human-readable summary report
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Data Generation Summary ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write("="*50 + "\n")
            f.write(f"Decoding Strategy: {self.decoding_strategy.value}\n")
            if self.decoding_strategy == DecodingStrategy.BEAM_SEARCH:
                f.write(f"Beam Size: {self.beam_size}\n")
            f.write(f"Total Data Points: {len(df)}\n")
            f.write(f"Unique Samples Processed: {df['sample_id'].nunique()}\n")
            f.write(f"Average BLEU Score: {df['bleu_score'].mean():.3f}\n\n")
            f.write("Dataset Distribution:\n" + df['dataset'].value_counts().to_string() + "\n\n")
            f.write("Domain Distribution:\n" + df['domain'].value_counts().to_string() + "\n")
        print(f"[Success] Summary report saved to: {summary_path}")

    def run_paec_evaluation_for_sentence(self, sample: Dict) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Executes a translation task for a single sentence in evaluation mode.

        This method is used during the validation phase to assess system performance
        under real hardware constraints using the RealtimeResourceMonitor. It captures
        the full trajectory of the translation for detailed analysis.

        Args:
            sample (Dict): Contains 'source_text' and 'target_text'.

        Returns:
            Tuple: 
                - Final metrics dictionary (BLEU, Latency, Memory, etc.).
                - List of step dictionaries (trajectory) for the best beam.
        """
        if not self.evaluation_mode or not isinstance(self.pressure_computer, RealtimeResourceMonitor):
            raise RuntimeError("This method can only be called in evaluation_mode with a RealtimeResourceMonitor.")

        source_text = sample['source_text']
        reference_text = sample['target_text']

        # Reset monitor for independent measurement
        self.pressure_computer.reset()

        # Synchronize GPU to ensure accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize(config.DEVICE)
            torch.cuda.reset_peak_memory_stats(config.DEVICE)
        
        start_time = time.perf_counter()

        # Execute translation with PAEC control enabled
        # Returns result and the full trajectory tree
        result, trajectory_dict = self.knn_system.translate_with_paec_control(
            source_text, self.pressure_computer,
            max_length=self.max_length,
            beam_size=self.beam_size,
            length_penalty=self.length_penalty
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize(config.DEVICE)
        
        end_time = time.perf_counter()
        
        # --- Collect Performance Metrics ---
        latency_ms = (end_time - start_time) * 1000.0
        memory_mb = torch.cuda.max_memory_allocated(config.DEVICE) / (1024**2) if torch.cuda.is_available() else 0.0
        
        hypothesis = result.get('translation', "Translation failed.")
        
        # Log warning if hypothesis perfectly matches reference (possible data leakage or overfitting)
        if hypothesis == reference_text:
            if self.is_debug: logger.warning(f"Hypothesis matches reference for source: {source_text}")

        final_metrics = {
            'hypothesis': hypothesis,
            'latency_ms': latency_ms,
            'memory_mb': memory_mb,
            'knn_skip_rate': result.get('knn_skip_rate', 0.0),
            'source': source_text,
            'reference': reference_text
        }
        
        # Extract the trajectory of the best beam (Beam ID 0)
        best_trajectory = []
        if trajectory_dict:
            best_trajectory = trajectory_dict.get(0, [])
        
        return final_metrics, best_trajectory
    
    def teardown(self):
        """
        Releases all resources held by the pipeline, including GPU memory,
        FAISS indices, and monitor threads.
        """
        if self.is_debug: 
            print("[Info] Tearing down DataGenerationPipeline and releasing all resources...")
        
        if hasattr(self, 'knn_system'):
            if self.is_debug: print("\t- Tearing down kNNMTSystem...")
            self.knn_system.teardown()
            del self.knn_system
        
        if hasattr(self, 'pressure_computer'):
            if self.is_debug: print("\t- Releasing pressure computer resources...")
            del self.pressure_computer
        
        if hasattr(self, 'dataset_loader'):
            if self.is_debug: print("\t- Releasing dataset loader resources...")
            del self.dataset_loader
            del self.real_samples
        
        if hasattr(self, 'data_log'):
            if self.is_debug: print("\t- Clearing data logs...")
            self.data_log.clear()
            del self.data_log
        
        if self.is_debug: print("\t- Cleaning up temporary objects...")
        if hasattr(self, 'real_samples'):
            del self.real_samples
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            if self.is_debug: print("\t- Clearing PyTorch CUDA cache...")
            torch.cuda.empty_cache()
        
        if self.is_debug: 
            print("[Success] DataGenerationPipeline resources have been fully released.")
