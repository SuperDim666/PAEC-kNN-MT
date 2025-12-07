#!/bin/bash
set -e

# ==============================================================================
# 1. Configuration Loading & Environment Setup
# ==============================================================================

# Resolve the project root directory
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# Dynamically load the 'PATHS' configuration from src/config.py
CONFIG_PATHS_JSON=$(python -c "import sys; sys.path.append('$PROJECT_ROOT'); from src.config import PATHS; import json; print(json.dumps({k: str(v) for k, v in PATHS.items()}, indent=4))")
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to load paths from src/config.py. Ensure it's runnable."
    exit 1
fi

# Dynamically load the 'BUILD_PIPELINE_SETTINGS' configuration from src/config.py
CONFIG_BUILD_PIPELINE_SETTINGS_JSON=$(python -c "import sys; sys.path.append('$PROJECT_ROOT'); from src.config import BUILD_PIPELINE_SETTINGS; import json; print(json.dumps({k: str(v) for k, v in BUILD_PIPELINE_SETTINGS.items()}, indent=4))")
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to load pipeline settings from src/config.py. Ensure it's runnable."
    exit 1
fi

# Define helper functions to extract values from the loaded JSON configurations
get_path() { echo "$CONFIG_PATHS_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('$1', ''))"; }
get_nmt_model_settings() { echo "$CONFIG_BUILD_PIPELINE_SETTINGS_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('$1', ''))"; }

# Define environment and library paths
FAIRSEQ_ENV_DIR="/content/fairseq_env"
KNNBOX_DATASTORE_UTIL_FILE_DIR="$(get_path "knn_box_dir")/knnbox/datastore/utils.py"

# Activate the dedicated Fairseq virtual environment (Python 3.8)
source "$FAIRSEQ_ENV_DIR/bin/activate"

# ==============================================================================
# 2. Patching knn-box for Hardware Compatibility
# ==============================================================================
# Patch the knn-box utility file to safely handle FAISS index type checks.
# This prevents crashes on Colab GPUs where 'swigfaiss_avx2' or 'swigfaiss_avx512' might be undefined.
if [ -n "$COLAB_GPU" ]; then
    sed -i 's/isinstance(index,faiss.swigfaiss_avx2.IndexPreTransform)/hasattr(faiss, "swigfaiss_avx2") and isinstance(index, faiss.swigfaiss_avx2.IndexPreTransform)/g' "$KNNBOX_DATASTORE_UTIL_FILE_DIR"
    sed -i 's/isinstance(index,faiss.swigfaiss_avx512.IndexPreTransform)/hasattr(faiss, "swigfaiss_avx512") and isinstance(index, faiss.swigfaiss_avx512.IndexPreTransform)/g' "$KNNBOX_DATASTORE_UTIL_FILE_DIR"
fi

# ==============================================================================
# 3. Training Adaptive kNN-MT Combiner
# ==============================================================================
# Construct the argument list for `fairseq-train`.
# This process trains the "Combiner" (Meta-k network) for Adaptive kNN-MT.
# It fine-tunes the base NMT model and learns to dynamically adjust 'k' and 'lambda'.
args=(
    "$(get_path "data_bin_dir_ds")"                                                        # Input binary data (Datastore subset)
    --task translation                                                                      # Task type
    --train-subset "$(get_nmt_model_settings "train-subset-adaptive")"                      # Subset used for training the combiner
    --valid-subset "$(get_nmt_model_settings "valid-subset-adaptive")"                      # Subset used for validation
    --best-checkpoint-metric "$(get_nmt_model_settings "best-checkpoint-metric-adaptive")"  # Metric to track best checkpoint
    --finetune-from-model "$(get_path "nmt_model_dir")/$(get_nmt_model_settings "nmt-model-dir")/$(get_nmt_model_settings "nmt-model")" # Load weights from the base NMT model
    --max-epoch "$(get_nmt_model_settings "max-epoch-adaptive")"                            # Maximum training epochs
    --max-update "$(get_nmt_model_settings "max-update-adaptive")"                          # Maximum training updates
    --save-interval-updates "$(get_nmt_model_settings "save-interval-updates-adaptive")"    # Interval for saving checkpoints
    --tensorboard-logdir "$(get_path "combiner_model_dir")/adaptive_knn_mt/log"             # TensorBoard log directory
    --save-dir "$(get_path "combiner_model_dir")/adaptive_knn_mt"                           # Output directory for the trained model
    --user-dir "$(get_path "knn_box_dir")/knnbox/models"                                    # Path to custom knn-box model implementations
    --arch "adaptive_knn_mt@$(get_nmt_model_settings "arch")"                              # Architecture wrapper (Adaptive kNN over Transformer)
    --max-tokens "$(get_nmt_model_settings "max-tokens")"                                   # Maximum tokens per batch
    --num-workers "$(get_nmt_model_settings "num-workers")"                                 # Number of data loading workers
    --knn-mode "$(get_nmt_model_settings "knn-mode-adaptive")"                              # kNN mode (e.g., train_metak)
    --knn-datastore-path "$(get_path "datastore_dir")/ivf_pq"                               # Path to the pre-built IVF_PQ FAISS index
    --knn-max-k "$(get_nmt_model_settings "knn-max-k-adaptive")"                            # Maximum neighbors (k) to consider
    --knn-k-type "$(get_nmt_model_settings "knn-k-type-adaptive")"                          # Learning strategy for k (e.g., trainable)
    --knn-lambda-type "$(get_nmt_model_settings "knn-lambda-type-adaptive")"                # Learning strategy for lambda
    --knn-temperature-type "$(get_nmt_model_settings "knn-temperature-type-adaptive")"      # Learning strategy for temperature
    --knn-temperature "$(get_nmt_model_settings "knn-temperature-adaptive")"                # Softmax temperature value
    --log-interval "$(get_nmt_model_settings "log-interval-adaptive")"                      # Logging interval
    --criterion "$(get_nmt_model_settings "criterion-adaptive")"                            # Loss function
    --optimizer "$(get_nmt_model_settings "optimizer-adaptive")"                            # Optimizer
    --adam-betas "$(get_nmt_model_settings "adam-betas-adaptive")"                          # Adam betas
    --adam-eps "$(get_nmt_model_settings "adam-eps-adaptive")"                              # Adam epsilon
    --clip-norm "$(get_nmt_model_settings "clip-norm-adaptive")"                            # Gradient clipping norm
    --lr "$(get_nmt_model_settings "lr-adaptive")"                                          # Learning rate
    --lr-scheduler "$(get_nmt_model_settings "lr-scheduler-adaptive")"                      # Learning rate scheduler
    --min-lr "$(get_nmt_model_settings "min-lr-adaptive")"                                  # Minimum learning rate
    --label-smoothing "$(get_nmt_model_settings "label-smoothing-adaptive")"                # Label smoothing factor
    --lr-patience "$(get_nmt_model_settings "lr-patience-adaptive")"                        # Patience for LR reduction
    --lr-shrink "$(get_nmt_model_settings "lr-shrink-adaptive")"                            # Factor for LR reduction
    --patience "$(get_nmt_model_settings "patience-adaptive")"                              # Early stopping patience
    --update-freq "$(get_nmt_model_settings "update-freq")"                                 # Gradient accumulation steps
    --knn-combiner-path "$(get_path "combiner_model_dir")/adaptive_knn_mt"                  # Path to save the specific combiner state
)

if [ "$(get_nmt_model_settings "no-epoch-checkpoints-adaptive")" = "True" ]; then
    args+=(--no-epoch-checkpoints)
fi
if [ "$(get_nmt_model_settings "no-last-checkpoints-adaptive")" = "True" ]; then
    args+=(--no-last-checkpoints)
fi
if [ "$(get_nmt_model_settings "no-save-optimizer-state-adaptive")" = "True" ]; then
    args+=(--no-save-optimizer-state)
fi

# Run the training command with Fused LayerNorm enabled for performance optimization
FAIRSEQ_USE_FUSED_LAYERNORM=1 fairseq-train "${args[@]}"
