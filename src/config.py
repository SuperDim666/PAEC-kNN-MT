# -*- coding: utf-8 -*-
"""
src/config.py

Central configuration file for the PAEC (Production-Aware Exposure Compensation) project.
This file consolidates all hyperparameters, model identifiers, file paths, and simulation
parameters to facilitate easy management and experimentation.

It serves as the single source of truth for:
1. File system paths (Project root, data, models).
2. NMT model architecture and training hyperparameters (Fairseq).
3. Production environment simulation parameters (Latency, Memory, Throughput baselines).
4. Heuristic policy distributions for training data generation.
5. FAISS datastore index configurations.
"""

from pathlib import Path

# --- Project Directory Structure ---
# Defines the root directory of the project to construct absolute paths.
# This makes the project portable across different machines.
PROJ_DIR: Path = Path(__file__).resolve().parent.parent / "drive" / "MyDrive" / "PAEC_proj"

# --- Corpus Size Configuration ---
# Defines the sample sizes (upper limit) for NMT training and Datastore creation.
# Note: The 'datastore' size is intentionally smaller than 'nmt_training' in this experimental setup
# to simulate a "Strong Model, Weak Retrieval" scenario for rigorous stability testing.
TARGET_TOTAL_SAMPLES_CORPUS = {
    "nmt_training": float('inf'),   # (~2-3M) Used for training the core NMT model (use all available).
    "datastore": 50000              # (50k) Used for building the kNN Datastore and all subsequent tasks.
}

# Configuration for corpus sampling and similarity verification scripts.
CORPUS_SAMPLING_SETTINGS = {
    "num_proc": 8,                  # Number of processes for parallel data processing.
    "batch_size": 1000,             # Batch size for mapping functions.
    "similarity_check_model": "sentence-transformers/LaBSE", # Model for semantic deduplication.
    "similarity_threshold": 0.75,   # Cosine similarity threshold for filtering duplicates.
    
    # Settings for test sentence similarity analysis (`scripts/01C_test_sentence_similarity.py`)
    "top_n_to_print": 1000,
    "total_sample_size_to_check": 10000,
    "output_plot_filename": "similarity_distribution.png",
    "target_line_nums": []
}

# Maximum number of neighbors to retrieve during kNN-MT inference.
KNN_MAX_K = 16

# --- DAgger (Dataset Aggregation) Parameters ---
# Parameters controlling the iterative data collection process (if used).
DAGGER_MAX_ITERS = 10               # Max number of DAgger iterations (data batches from train_ds).
DAGGER_EPSILON_PI = 0.3             # Epsilon for e-greedy exploration in 'Policy_Explore' mode.
DAGGER_REPLAY_LIMITS = -1           # Max history of iterations to train on (-1 for all, 5 for last 5).

# Hyperparameter: min attention weight to be considered "covered" 
# in `compute_error_state()` for the Entity Coverage metric.
ATTN_THRESHOLD = 0.6 

# Engineering rigor check: Datastore must be a subset of the NMT training data.
assert TARGET_TOTAL_SAMPLES_CORPUS["datastore"] <= TARGET_TOTAL_SAMPLES_CORPUS["nmt_training"], \
    "Datastore size cannot be larger than NMT training size."

# --- Fairseq & SentencePiece Configuration ---
# This dictionary controls the arguments passed to `fairseq-train` and `spm_train`
# in the shell scripts (e.g., `scripts/02_build_pipeline.sh`).
BUILD_PIPELINE_SETTINGS = {
    
    # General Settings
    "source-lang": "de",
    "target-lang": "en",
    "arch": "transformer",          # Base NMT architecture.
    "dataset-impl": "mmap",         # Use memory-mapped files for large datasets.
    "workers": 12,                  # FASTBPE Number of Multiprocessing Workers.
    "num-workers": 12,              # Model Training Workers of CPU.
    "max-tokens": 8192,             # Maximum number of tokens in a batch (Training).
    "max-tokens-valid": 2048,       # Maximum number of tokens in a batch (Validation).
    
    # BPE (Byte Pair Encoding) Settings
    "bpe": "sentencepiece",
    "bpe-vocab-size": 48000,
    
    # SentencePiece Specific Settings
    "spm-model-type": "unigram",    # Unigram language model for subword segmentation.
    "spm-character-coverage": 0.9995,
    "spm-input-sentence-size": 1e8,
    "spm-shuffle-input-sentence": True,
    
    # Unigram Regularization (improves robustness)
    "spm-normalization-rule-name": "nmt_nfkc",
    "spm-add-dummy-prefix": True,
    "spm-remove-extra-whitespaces": True,
    "spm-enable-regularization": True,
    "spm-split-by-whitespace": True,
    "spm-nbest-size": 64,
    "spm-alpha": 0.05,
    
    # Model Checkpoint Filenames
    "nmt-model": "checkpoint_best.pt",
    "nmt-model-dir": "self-trained",

    # Preprocessing Settings
    "joined-dictionary": False,     # Separate dictionaries for source and target languages.
    "thresholdsrc": 0,
    "thresholdtgt": 0,
    
    # NMT Training Hyperparameters
    "self-nmt-arch": True,
    "share-all-embeddings": True,   # Share encoder, decoder, and output embeddings.
    "encoder-layers": 6,
    "decoder-layers": 6,
    "encoder-embed-dim": 512,
    "decoder-embed-dim": 512,
    "encoder-ffn-embed-dim": 2048,
    "decoder-ffn-embed-dim": 2048,
    "encoder-attention-heads": 8,
    "decoder-attention-heads": 8,
    "dropout": 0.1,
    "attention-dropout": 0.1,
    "activation-dropout": 0.0,
    "activation-fn": "relu",
    "max-source-positions": 1024,
    "max-target-positions": 1024,
    "optimizer": "adam",
    "adam-betas": "(0.9, 0.98)",
    "lr": 0.0005,
    "lr-scheduler": "inverse_sqrt",
    "warmup-updates": 4000,
    "weight-decay": 0.0001,
    "criterion": "label_smoothed_cross_entropy",
    "label-smoothing": 0.1,
    "update-freq": 4,               # Gradient accumulation steps.
    "precision": "memory-efficient-fp16",
    "max-epoch": 50,
    "patience": 5,                  # Early stopping patience.
    "no-epoch-checkpoints": True,   # Do not save a checkpoint every epoch (save space).
    "no-last-checkpoints": False,
    "no-save-optimizer-state": False,
    "no-progress-bar": True,
    "log-interval": 400,
    # "max-update": 20000,

    # kNN / Validation Settings
    "model-overrides": {
        "eval_bleu": False,
        "required_seq_len_multiple": 1,
        "load_alignments": False,
    },
    "skip-invalid-size-inputs-valid-test": True,
    "knn-arch-model": "adaptive_knn_mt",
    
    # Combiner Model Settings (Adaptive kNN-MT Baseline)
    # These settings control the training of the lightweight meta-network used in
    # standard Adaptive kNN-MT, which PAEC compares against.
    "log-interval-adaptive": 1000,
    "knn-max-k-adaptive": KNN_MAX_K,
    "knn-mode-adaptive": "train_metak",
    "knn-k-type-adaptive": "trainable",
    "knn-lambda-type-adaptive": "trainable",
    "knn-temperature-type-adaptive": "fixed",
    "knn-temperature-adaptive": 10.0,
    # "batch-size-adaptive": 32,
    "no-epoch-checkpoints-adaptive": True,
    "no-last-checkpoints-adaptive": True,
    "no-save-optimizer-state-adaptive": True,
    "save-interval-updates-adaptive": 100,
    "lr-adaptive": 3e-4,
    "lr-scheduler-adaptive": "reduce_lr_on_plateau",
    "min-lr-adaptive": 3e-05,
    "label-smoothing-adaptive": 0.001,
    "lr-patience-adaptive": 5,
    "lr-shrink-adaptive": 0.5,
    "patience-adaptive": 10,
    "max-epoch-adaptive": 200,
    "max-update-adaptive": 2000,
    "criterion-adaptive": "label_smoothed_cross_entropy",
    "clip-norm-adaptive": 1.0,
    "optimizer-adaptive": "adam",
    "adam-betas-adaptive": '(0.9, 0.98)',
    "adam-eps-adaptive": 1e-8,
    "best-checkpoint-metric-adaptive": "loss",
    "train-subset-adaptive": "valid",   # Train combiner on validation set (standard practice).
    "valid-subset-adaptive": "valid",
}

# --- Path Configurations ---
# Explicitly define paths for data, models, results, and libraries.
# `USE_REAL_DATASTORE` toggles between simulated and actual FAISS retrieval.
USE_REAL_DATASTORE = True
PATHS = {
    "proj_dir": PROJ_DIR,
    "data_dir": PROJ_DIR / "data",
    "cache_dir": PROJ_DIR / "cache",
    "processed_data_dir": PROJ_DIR / "data" / "processed",
    "models_dir": PROJ_DIR / "models",
    "performance_models_dir": PROJ_DIR / "models" / "performance_models", # Linear regression models for cost estimation.
    "dynamics_model_dir": PROJ_DIR / "models" / "dynamics_model",         # T_theta checkpoints.
    "policy_model_dir": PROJ_DIR / "models" / "policy_model",             # Pi_phi checkpoints.
    "nmt_model_dir": PROJ_DIR / "models" / "nmt_models",                  # Fairseq NMT checkpoints.
    "combiner_model_dir": PROJ_DIR / "models" / "combiners",              # Adaptive kNN-MT checkpoints.
    "results_dir": PROJ_DIR / "results",
    "libs_dir": PROJ_DIR / "libs",
    "knn_box_dir": PROJ_DIR / "libs" / "knn-box",
    "datastore_dir": PROJ_DIR / "data" / "datastores" / "paec_corpus_datastores", # FAISS indices.
    "raw_corpus_dir": PROJ_DIR / "data" / "raw" / "paec_corpus_combined", # Raw text for NMT.
    "data_bin_dir": PROJ_DIR / "data" / "bin" / "paec_corpus_combined",     # Binary data for NMT using full corpus.
    "data_bin_dir_ds": PROJ_DIR / "data" / "bin" / "paec_corpus_combined_ds", # Binary data for Datastore using subset.
    "bpe_dir": PROJ_DIR / "data" / "bpe" / "paec_corpus_combined", # BPE model and codes.
}
PATHS["impl"] = BUILD_PIPELINE_SETTINGS["dataset-impl"]
PATHS["impl_data_dir"] = PATHS["data_dir"] / PATHS["impl"]

# --- Policies Ratios ---
# Defines the mixture of heuristic policies used during the Data Generation Phase.
# This ensures the Dynamics Model (T_theta) learns from a diverse distribution of behaviors,
# covering both safe operation and edge cases (e.g., near-OOM or high error).
POLICIES_MIX = {
    'Policy_Default_Balanced': 0.4,         # Default policy used in main experiments (Balanced logic).
    'Policy_Quality_First': 0.3,            # Prioritizes translation quality over resource usage.
    'Policy_Resource_Guardian': 0.2,        # Extremely conservative, prioritizes low resource usage.
    'Policy_Stability_Averse': 0.07,        # Focuses on stabilizing the internal context vector.
    'Policy_Dangerous_Perturbator': 0.03,   # Aggressively tests system limits (Negative Sampling).
}

def validate_policies_mix():
    """Validates that the policy probabilities sum to 1.0."""
    total_prob = sum(POLICIES_MIX.values())
    if abs(total_prob - 1.0) > 1e-6:
        raise ValueError(f"Policies probabilities sum to {total_prob}, must be 1.0")
validate_policies_mix()

# --- Model Identifiers ---
# Central repository for Hugging Face model names used for metrics and auxiliary tasks.
MODEL_NAMES = {
    "sentence_encoder": "sentence-transformers/LaBSE",  # For semantic similarity (Error state).
    "fluency_scorer": "distilgpt2"                      # For surprisal calculation (Error state).
}

# --- Data Loader Parameters ---
# Configuration for the RealDatasetLoader class.
# Defines which datasets to load, their sampling ratios, and filtering rules.
DATA_LOADER_PARAMS = {
    "datasets": [
        # {"name": "wmt/wmt19", "config": "de-en", "split": "train", "ratio": 0.1, "domain": "news",
        #  "ner_filter": True, "min_entities": 1, "trust_remote_code": False},
        # {"name": "bentrevett/multi30k", "split": "train", "ratio": 1.0, "domain": "description",
        #  "ner_filter": False, "min_entities": 0, "trust_remote_code": False},
        # {"name": "Helsinki-NLP/opus_books", "config": "de-en", "split": "train", "ratio": 0.1, "domain": "literary",
        #  "ner_filter": True, "min_entities": 1, "trust_remote_code": False},
        {"name": "Helsinki-NLP/opus-100", "config": "de-en", "split": "train", "ratio": 0.1, "domain": "mixed",
         "ner_filter": True, "min_entities": 2, "trust_remote_code": False, "split_only_from_train": False},
        {"name": "europarl_bilingual", "config": "de-en", "split": "train", "ratio": 0.9, "domain": "legal",
         "ner_filter": True, "min_entities": 2, "trust_remote_code": False, "split_only_from_train": True},  # Legal entities from EU parliament texts.
    ],
    "ner_filter": False,  # Global default for NER filtering.
    "min_entities": 1,
    "min_sentence_length": 5, "max_sentence_length": 35,
    "valid_split_ratio": 0.10, "test_split_ratio": 0.10
}

# The ratios of dataset must sum to 1 to be correctly splitted in `scripts/01_data_preparations/01_prepare_corpus.py`
if abs(sum([dataset.get("ratio", 0.0) for dataset in DATA_LOADER_PARAMS["datasets"]]) - 1.0) > 1e-6:
    raise ValueError(f"The ratios of dataset must sum to 1.0 to be correctly splitted in `scripts/01_data_preparations/01_prepare_corpus.py`")

# --- Production Constraint Simulator Parameters ---
# Parameters for the ProductionConstraintSimulator and RealtimeResourceMonitor.
# These values define the "safe" operating limits and the coefficients for the
# Pressure State (Phi_t) calculation.
# Note: R_opt and M_avail are updated by `00_calibrate_baselines.py`.
SIMULATOR_PARAMS = {
    # System baseline resource configuration (will be overwritten by calibration).
    "Latency_SLA": 100.0,                   # Target max latency (ms).
    "Memory_Avail": 15095.062,              # Available GPU Memory (MB).
    "Throughput_opt": 296.706,              # Optimal Throughput (RPS).

    # Sliding window for calculating derivatives (e.g., dLatency/dt).
    "sliding_window_size": 10,

    # Pressure vector calculation weights (Sigmoid inputs).
    # See Chapter 2.3 of the paper for definitions.
    "w_latency_current": 2.0,               # w1: Weight for current latency.
    "w_latency_derivative": 1.5,            # w2: Weight for latency change rate.
    "w_memory_current": 3.0,                # w3: Weight for current memory usage.
    "w_memory_derivative": 2.0,             # w4: Weight for memory change rate.
    "w_throughput_current_deficit": 2.0,    # w5_deficit: Penalty when throughput < R_opt.
    "w_throughput_current_surplus": 0.1,    # w5_surplus: Bonus when throughput >= R_opt.
    "w_throughput_offset": 1.0,             # w6: Offset for the sigmoid function.

    # Mapping from abstract pressure [0,1] to concrete concurrency for cost model.
    "min_concurrency": 1,
    "max_concurrency": 96,
    
    "decay_factor": 0.7                     # Decay factor for history smoothing.
}

# --- Experiment & Analysis Parameters ---
# Parameters for running scientific experiments and generating analyses.
EXPERIMENT_PARAMS = {
    "using_num_samples": "full",
    "num_samples_per_strategy_test": 10,
    "num_samples_per_strategy_quick": 100,
    "num_samples_per_strategy_full": 1000,
    "decoding_strategies_to_test": [
        # {'beam_size': 1, 'length_penalty': 1.0, 'use_datastore': True, 'datastore_path': 'data/datastores'}, # Greedy Search
        # {'beam_size': 2, 'length_penalty': 1.0, 'use_datastore': True, 'datastore_path': 'data/datastores'},
        # {'beam_size': 3, 'length_penalty': 1.0, 'use_datastore': True, 'datastore_path': 'data/datastores'},
        {'beam_size': 4, 'length_penalty': 1.0, 'use_datastore': True, 'datastore_path': 'data/datastores'},
    ],

    # Coherence Horizon Detector parameters for scientific analysis (unused in main training).
    "coherence_horizon": {
        "window_size": 5,
        "continuity": 2,
        "threshold": 0.001,
    }
}

DEVICE = "cuda"

# --- General Settings ---
# Seed for random number generators to ensure reproducibility.
RANDOM_SEED = 42
CONFIG_HASH = "null"  # To be set dynamically based on scripts

# Constraints for normalizing specific state variables (used in Policy base class).
CONSTRAINT_LIMIT = {
    "error_semantic": {
        "min": 0.0,
        "max": 0.9465634877234697 # Empirical max from data analysis.
    },
}

# Flag to reset simulator parameters during initialization.
PRODUCTION_SIMULATOR_PARAMS_RESET = False

# Traffic Pattern Simulation Parameters
# Controls the stochastic process generating Latency/Throughput loads in `ProductionConstraintSimulator`.
PRODUCTION_SIMULATOR_PARAMS = {
    "variation_amplitude": 1.8, # Amplitude of the daily sine wave cycle.
    "noise_std": 0.2,           # Standard deviation of Gaussian noise.
    "mean_load_alpha": 3.5,     # Alpha parameter for Gamma distribution (Mean Load).
    "mean_load_beta": 1.0,      # Beta parameter for Gamma distribution.
    "mean_load_min": 0.2,       # Minimum base load multiplier.
    "mean_load_max": 1.5,       # Maximum base load multiplier.
    "traffic_min": 0.05,        # Minimum traffic multiplier floor.
    "traffic_max": 2.0          # Maximum traffic multiplier ceiling.
}

# Target statistical distributions for simulator optimization/validation.
PRODUCTION_SIMULATOR_GOALS = {
    'latency': {
        'min': (5, 15), 'max': (190, 250), 'mean': (65, 115),
        'std': (60, 120), 'skew': (-0.5, 1.5)
    },
    'throughput': {
        'min': (100, 900), 'max': (2800, 6500), 'mean': (1600, 2200),
        'std': (1500, 3000), 'skew': (-0.5, 1.5)
    }
}

# Configuration for the Online Optimization (Teacher Policy generation).
OPTIM_STEPS_CONFIG = {
    "tolerance": 1e-4,          # Early stopping tolerance for loss improvement.
    "patience": 1,              # Patience for early stopping.
    "max_steps": 4,             # Max gradient descent steps per action search.
    "lr": 1e-2,                 # Learning rate for optimizing action parameters (k, lambda).
}

# --- FAISS Index Configurations ---
# Specify which FAISS index types are available and should be considered during evaluation.
ENABLED_INDICES = ['exact', 'hnsw', 'ivf_pq']   # 'exact' (FlatL2), 'ivf_pq' (IVFPQ).
DEFAULT_INDEX = 'ivf_pq'                # Default index type to use when not specified.

# Mapping baseline models to their preferred index type.
KNN_BASELINE_INDEX = {
    'vanilla': 'ivf_pq',
    'adaptive': 'ivf_pq'
}

# Validation checks for index configuration.
if 'none' in ENABLED_INDICES and len(ENABLED_INDICES) == 1:
    raise ValueError("ENABLED_INDICES cannot be only 'none'. At least one valid index type (`exact`, `hnsw`, `ivf_pq`, etc.) must be enabled.")
if DEFAULT_INDEX not in ENABLED_INDICES or DEFAULT_INDEX == 'none':
    raise ValueError("DEFAULT_INDEX must be one of the ENABLED_INDICES and cannot be 'none'.")
for keys in KNN_BASELINE_INDEX:
    if KNN_BASELINE_INDEX[keys] not in ENABLED_INDICES or KNN_BASELINE_INDEX[keys] == 'none':
        raise ValueError(f"KNN_BASELINE_INDEX for {keys} must be one of the ENABLED_INDICES and cannot be 'none'.")

# --- Adaptive Robust kNN-MT Strategy Parameters ---
ADAPTIVE_ROBUST_KNN_PARAMS = {
    # kNN Distribution Smoothing/Weighting (kNN-SW)
    "KNNSW_ENABLE": False,
    "KNNSW_TEMPERATURE": 50.0,      # Softmax temperature (lower -> sharper, higher -> smoother).
    # Hard thresholds for the "Resource Safety Valve".
    # If pressure exceeds these values, the system forces k=0 (skip retrieval).
    "KNN_PRESSURE_THRESHOLD": {     
        "latency": 0.9,
        "memory": 0.9,
        "throughput": 0.9
    }
}
