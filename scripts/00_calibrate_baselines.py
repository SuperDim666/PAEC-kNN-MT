# -*- coding: utf-8 -*-
"""
scripts/00_calibrate_baselines.py

PAEC System Baseline Calibration Script

This script is the first step in the PAEC pipeline. It benchmarks the underlying hardware
(GPU/CPU) to establish physical performance limits. These baselines are crucial for the
ProductionConstraintSimulator (and RealtimeResourceMonitor) to normalize pressure vectors correctly.

Key Objectives:
1. Measure R_opt (Optimal Throughput): The maximum sustainable requests per second.
2. Measure M_avail: Detect total available memory (implicitly handled by PyTorch/psutil).
3. Reference L_SLA: Measure best-case latency to help users set realistic Service Level Agreements.
4. Update ./src/config.py: Automatically inject measured values into the global configuration.
"""

import re
import time
import torch
import psutil
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from pathlib import Path
import fileinput

def update_config_file(config_path, updates):
    """
    Updates specific keys in the src/config.py file with measured baseline values.
    
    This function performs an in-place edit to preserve the original file structure,
    comments, and formatting. It searches for keys like "Throughput_opt" and replaces
    their values with the newly calibrated statistics.
    
    Args:
        config_path (Path): Path to the target config.py file.
        updates (dict): Dictionary mapping config keys to their new float values.
    """
    # Use fileinput to modify the file line-by-line in place, creating a backup just in case.
    with fileinput.FileInput(config_path, inplace=True, backup='.bak') as file:
        for line in file:
            # Flag to track if the current line matches a key to be updated
            found_key = None
            
            # Iterate through the updates dictionary to find matching keys in the file
            for key, value in updates.items():
                # Regex looks for lines starting with "key": value
                if re.search(fr'^\s*"{key}":', line):
                    found_key = key
                    # Format the new line with the updated value (3 decimal places)
                    new_line = f'    "{key}": {value:.3f},\n'
                    print(new_line, end='')
                    break
            
            # If the line was not updated, print it exactly as is to preserve the file
            if not found_key:
                print(line, end='')

    print(f"[Success] Configuration file updated: {config_path}")


def measure_r_opt(
  model, tokenizer, device,
  test_sentence,
  num_warmup=10, num_measure=50, max_batch_exp=14
):
    """
    Empirically measures the Optimal Throughput (R_opt) of the system.
    
    This function simulates increasing load by exponentially growing the batch size (2^i).
    It identifies the 'sweet spot' where throughput is maximized before hardware limits
    (like GPU memory exhaustion or thermal throttling) cause performance degradation.
    This R_opt value defines the denominator for the throughput pressure calculation in the PAEC state.
    
    Args:
        model: The loaded NMT model (Transformer) for inference benchmarking.
        tokenizer: The associated tokenizer for input processing.
        device: Computational device ('cuda' or 'cpu').
        test_sentence: A standardized input string used for uniform benchmarking.
        num_warmup: Number of forward passes to run before timing (to warm up CUDA kernels).
        num_measure: Number of forward passes to average for the final measurement.
        max_batch_exp: Maximum power of 2 to test for batch size (e.g., 14 -> 16384).
    
    Returns:
        r_opt (float): The peak measured throughput (sentences/second).
        optimal_bs (int): The batch size that achieved this peak throughput.
    """
    print("\t- Finding optimal throughput (R_opt) by testing batch sizes...")
    
    # Define batch sizes to test: [1, 2, 4, ..., 2^max_batch_exp]
    batch_sizes = [2**i for i in range(max_batch_exp + 1)]
    throughputs = {}
    
    for bs in batch_sizes:
        print(f"\t- Testing batch size: {bs}")
        # Replicate the test sentence to fill the batch
        sentences = [test_sentence] * bs
        
        try:
            # Phase 1: Warmup
            # Runs inference without gradient tracking to stabilize hardware caches/clocks.
            with torch.no_grad():
                for _ in range(num_warmup):
                    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
                    model.generate(**inputs, max_new_tokens=20)
            
            # Phase 2: Measurement
            # Ensure GPU synchronization before starting the timer for accurate results.
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                for _ in range(num_measure):
                    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
                    model.generate(**inputs, max_new_tokens=20)
            
            # Ensure GPU synchronization after finishing to capture full execution time.
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Calculate throughput: Total Sentences Processed / Total Time
            total_time = end_time - start_time
            rps = (bs * num_measure) / total_time
            throughputs[bs] = rps
            print(f"\t\t-> Throughput: {rps:.2f} sentences/sec")
        
        except RuntimeError as e:
            # Handle Out-Of-Memory (OOM) errors gracefully.
            # If OOM occurs, we have hit the physical limit; stop increasing batch size.
            if "out of memory" in str(e).lower():
                print(f"\t\t[Error] -> OOM detected. Stopping further increases.")
                break
            else:
                raise e
        except Exception as e:
            print(f"\t\t[Error] -> Error occurs at batch size {bs}: {e}")
            break
    
    if not throughputs:
        raise ValueError("No valid measurements obtained.")
    
    # Identify the optimal configuration: the batch size yielding the highest RPS.
    optimal_bs = max(throughputs, key=throughputs.get)
    r_opt = throughputs[optimal_bs]
    print(f"  - [Info] Real measured optimal throughput (R_opt): {r_opt:.3f} requests/sec (at batch size {optimal_bs})")
    return r_opt, optimal_bs


def main():
    """
    Main entry point for the calibration process (PAEC v2.0).
    
    Workflow:
    1. Setup environment (paths, device, model).
    2. Measure 'Best-Case Latency' to guide L_SLA configuration.
    3. Measure 'Optimal Throughput' (R_opt) via stress testing.
    4. Update 'src/config.py' with the measured R_opt.
    """
    print("🚀 Starting system baseline calibration (v2.0)...")
    
    # Resolve the project root and config path relative to this script.
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "src" / "config.py"
    
    # Determine the computing device (prefer CUDA for production simulation).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load a standard NMT model (MarianMT) to serve as the workload proxy.
    model_name = "Helsinki-NLP/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name, device_map=device)
    model.eval()
    
    updates = {}

    # --- 1. Measure Best-Case Latency as a REFERENCE for L_SLA ---
    # L_SLA (Service Level Agreement) is a business constraint, not a purely physical one.
    # However, measuring the minimum possible latency (batch size = 1) helps the user set a realistic SLA.
    
    print("\n  - Measuring best-case latency for reference...")
    single_sentence = ["Dieser Satz wird zur Leistungsmessung verwendet."]
    latencies = []
    
    with torch.no_grad():
        for _ in range(100): # Run 100 iterations to get a stable average.
            if device == "cuda": torch.cuda.synchronize()
            start_iter_time = time.perf_counter()
            
            inputs = tokenizer(single_sentence, return_tensors="pt").to(device)
            _ = model.generate(**inputs, max_new_tokens=20)
            
            if device == "cuda": torch.cuda.synchronize()
            end_iter_time = time.perf_counter()
            latencies.append((end_iter_time - start_iter_time) * 1000)
    
    best_case_latency = np.mean(latencies)
    print(f"  - [Info] Measured best-case single sentence latency is {best_case_latency:.3f} ms.")
    print("  - [Action] The 'Latency_SLA' in config.py acts as a user-defined L_SLA (Service Level Agreement).")
    print("             It will NOT be updated automatically. Please set it manually based on your performance targets.")
    print("             The current measured best-case latency can be a good reference point.")

    # --- 2. Measure Optimal Throughput (R_opt) ---
    print("  - Loading lightweight model for performance measurement...")

    test_sentence = "Dieser Satz wird zur Leistungsmessung verwendet."

    # Execute the throughput stress test to find the system's peak capacity.
    r_opt_value, optimal_batch_size = measure_r_opt(model, tokenizer, device, test_sentence)

    # Prepare the update dictionary for the config file.
    updates["Throughput_opt"] = r_opt_value

    # --- 3. Update Config File ---
    # Inject the measured physical limits into the project configuration.
    # This ensures the ResourcePressureVector is normalized against real hardware capabilities.
    update_config_file(config_path, updates)
    print("\nCalibration complete.")

if __name__ == "__main__":
    main()
