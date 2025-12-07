#!/bin/bash
#
# This script orchestrates the end-to-end NMT pipeline preparation.
# It executes the following phases:
# 1. BPE Model Training: Learns subword segmentation rules (FastBPE or SentencePiece).
# 2. BPE Application: Applies segmentation to training, validation, and test sets.
# 3. Preprocessing: Binarizes text data for Fairseq efficient loading.
# 4. NMT Training: Trains the Transformer model if a checkpoint does not exist.
# 5. Datastore Construction: Builds FAISS indices (Exact, HNSW, IVF_PQ) for kNN-MT.

# Exit immediately if any command exits with a non-zero status.
set -e 

# --- 0. Configuration and Path Retrieval ---

# Resolve the project root directory.
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# Extract path configurations from src/config.py using an inline Python script.
CONFIG_PATHS_JSON=$(python -c "import sys; sys.path.append('$PROJECT_ROOT'); from src.config import PATHS; import json; print(json.dumps({k: str(v) for k, v in PATHS.items()}, indent=4))")
if [ $? -ne 0 ]; then
    echo "[Error] Failed to load paths from src/config.py. Ensure it's runnable."
    exit 1
fi

# Extract pipeline settings (hyperparameters, architecture) from src/config.py.
CONFIG_BUILD_PIPELINE_SETTINGS_JSON=$(python -c "import sys; sys.path.append('$PROJECT_ROOT'); from src.config import BUILD_PIPELINE_SETTINGS; import json; print(json.dumps({k: str(v) for k, v in BUILD_PIPELINE_SETTINGS.items()}, indent=4))")
if [ $? -ne 0 ]; then
    echo "[Error] Failed to load pipeline settings from src/config.py. Ensure it's runnable."
    exit 1
fi

# Helper functions to query specific keys from the loaded JSON configurations.
get_path() { echo "$CONFIG_PATHS_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('$1', ''))"; }
get_nmt_model_settings() { echo "$CONFIG_BUILD_PIPELINE_SETTINGS_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('$1', ''))"; }
get_ds_model_overrides_str() { echo "$CONFIG_BUILD_PIPELINE_SETTINGS_JSON" | python -c "import sys, json;print(str(json.load(sys.stdin).get('model-overrides', {})).replace('true', 'True').replace('false', 'False').replace('null', 'None'))"; }

# Assign shell variables from the Python configuration.
PAEC_PROJECT_DIR=$(get_path "proj_dir")
RAW_CORPUS_DIR=$(get_path "raw_corpus_dir")
BPE_DIR=$(get_path "bpe_dir")
DATA_BIN_DIR=$(get_path "data_bin_dir")         # Directory for full NMT binary data
DATA_BIN_DIR_DS=$(get_path "data_bin_dir_ds")   # Directory for Datastore subset binary data
MODEL_SAVE_DIR=$(get_path "nmt_model_dir")      # Directory to save NMT checkpoints
DATASTORE_SAVE_PATH=$(get_path "datastore_dir") # Directory to save FAISS indices
FAIRSEQ_ENV_DIR="$PROJECT_ROOT/fairseq_env"     # Dedicated Python environment for Fairseq
KNNBOX_LIB_DIR=$(get_path "knn_box_dir")
KNNBOX_DATASTORE_UTIL_FILE_DIR="$KNNBOX_LIB_DIR/knnbox/datastore/utils.py"

# Set library paths for the custom environment.
LD_LIBRARY_PATH="$FAIRSEQ_ENV_DIR/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Validate that critical paths were successfully retrieved.
if [ -z "$RAW_CORPUS_DIR" ] || [ -z "$BPE_DIR" ] || [ -z "$DATA_BIN_DIR" ] || [ -z "$DATA_BIN_DIR_DS" ] || [ -z "$MODEL_SAVE_DIR" ] || [ -z "$DATASTORE_SAVE_PATH" ]; then
    echo "[Error] One or more critical paths could not be loaded from src/config.py."
    # echo "Loaded paths JSON: $CONFIG_PATHS_JSON" # Debug output
    exit 1
fi

# Activate the dedicated Fairseq Python environment.
export PATH="$FAIRSEQ_ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$FAIRSEQ_ENV_DIR/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
source "$FAIRSEQ_ENV_DIR/bin/activate"
export PATH="$FAIRSEQ_ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$FAIRSEQ_ENV_DIR/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Helper function to verify Python module availability.
check_python_module() {
    python3.8 -c "import $1" 2>/dev/null
    return $?
}

# Helper function to verify NVIDIA Apex installation (required for mixed precision).
check_apex_installation() {
    # Check if apex module is importable
    export LD_LIBRARY_PATH="$FAIRSEQ_ENV_DIR/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
    if ! python3.8 -c "
import sys
try:
    import torch
    import apex
    if not torch.cuda.is_available():
        print('CUDA not available', file=sys.stderr)
        sys.exit(1)
    print('Apex is installed. PyTorch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
except ImportError as e:
    print(f'Import failed: {str(e)}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error: {str(e)}', file=sys.stderr)
    sys.exit(1)
" >/dev/null 2>&1; then
        return 1
    fi
    # Verify that Apex's AMP functionality is available
    python3.8 -c "import amp_C; import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('Apex AMP is functional')" 2>/dev/null
    return $?
}

echo "Checking system requirements..."

# Ensure PyTorch is installed and compatible.
if check_python_module torch; then
    echo "[Success] PyTorch is installed."
    TORCH_VERSION=$(python3.8 -c "import torch; print(torch.__version__)")
    echo "PyTorch version: $TORCH_VERSION"
else
    echo "PyTorch is not installed. Installing a compatible version..."
    pip install torch==2.4.1cu121 -f https://download.pytorch.org/whl/torch_stable.html --quiet --progress-bar on
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install PyTorch."
        exit 1
    fi
    echo "PyTorch installation completed."
fi

# Ensure CUDA is available for GPU acceleration.
CUDA_AVAILABLE=$(python3.8 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "Error: CUDA is not available. Apex requires CUDA for --cuda_ext."
    exit 1
fi
CUDA_VERSION=$(python3.8 -c "import torch; print(torch.version.cuda)")
echo "CUDA version used by PyTorch: $CUDA_VERSION"

# Validate system CUDA toolkit version matches PyTorch.
if command -v nvcc &> /dev/null; then
    SYSTEM_CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "System CUDA toolkit version: $SYSTEM_CUDA_VERSION"
    if [ "$SYSTEM_CUDA_VERSION" != "$CUDA_VERSION" ]; then
        echo "[Warning] System CUDA version ($SYSTEM_CUDA_VERSION) does not match PyTorch CUDA version ($CUDA_VERSION). This may cause issues with Apex."
    fi
else
    echo "[Warning] nvcc not found. Ensure the CUDA toolkit is installed for Apex compilation."
fi

# Install ninja build system for faster C++ extension compilation.
if ! command -v ninja &> /dev/null; then
    echo "Installing ninja-build..."
    pip install ninja --quiet --progress-bar on
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install ninja."
        exit 1
    fi
fi

# Verify or Install NVIDIA Apex for optimization.
if check_apex_installation; then
    echo "[Success] NVIDIA Apex is ready to use with functional AMP."
else
    echo "Installing Apex..."
    # Clone Apex repository
    pip uninstall apex -y
    rm -rf ./apex
    git clone https://github.com/NVIDIA/apex
    cd apex || { echo "Error: Failed to enter apex directory."; exit 1; }
    MAX_JOBS=$(nproc)
    APEX_PARALLEL_BUILD=8
    CUDA_NVCC_FLAGS="-O3"
    CFLAGS="-O3 -march=native"
    CXXFLAGS="-O3 -march=native"
    APEX_DISABLE_EXTENSION_CHECKS=1
    TORCH_CUDA_ARCH_LIST="8.0"
    MKL_THREADING_LAYER=GNU
    NVCC_APPEND_FLAGS="--threads 8"
    rm -rf build
    # Install Apex with CUDA and C++ extensions
    sed -i '/^def check_cuda_torch_binary_vs_bare_metal(/a\    return' setup.py
    sed -i 's/parallel: int | None = None/parallel: typing.Optional[int] = None/g' setup.py
    sed -i 's/import sys/import sys, typing/g' setup.py
    python3.8 setup.py install --cuda_ext --cpp_ext > ../apex_install.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Apex installation failed. Check apex_install.log for details."
        cat apex_install.log
        cd ..
        exit 1
    fi
    cd ..
    echo "Apex installation completed!"
    
    # Verify Apex installation
    if ! check_apex_installation; then
        echo "Error: Apex installed but AMP functionality is not working."
        exit 1
    fi
    # Re-package the environment to include Apex
    pip cache purge
    deactivate
    rm -rf "$PROJECT_ROOT/fairseq_env.tar.gz"
    tar -czf "$PROJECT_ROOT/fairseq_env.tar.gz" \
        --exclude="__pycache__" \
        --exclude="*.pyc" \
        "$PROJECT_ROOT/fairseq_env"
    cp -rf "$PROJECT_ROOT/fairseq_env.tar.gz" "$PAEC_PROJECT_DIR"
    source "$FAIRSEQ_ENV_DIR/bin/activate"
    echo "[Success] Apex verified with functional AMP."
fi

# Ensure fastBPE is installed (C++ implementation).
if ! command -v fastbpe &> /dev/null; then
    echo "Installing fastBPE..."
    rm -rf "$PROJECT_ROOT/fastBPE"
    git clone https://github.com/glample/fastBPE.git
    cd fastBPE
    g -O3 -std=c11 -pthread fastBPE/main.cc -IfastBPE -o fastbpe
    sudo mv fastbpe /usr/local/bin/
    python3.8 setup.py install
    cd ..
fi

# Ensure SentencePiece binary is installed.
if ! command -v spm_train &> /dev/null; then
    echo "Installing SentencePiece (spm_train)..."
    sudo apt-get install -y sentencepiece
    if [ $? -ne 0 ]; then
        echo "[Error] Failed to install sentencepiece via apt-get. Trying pip..."
        pip install sentencepiece
    fi
fi

# Ensure SentencePiece Python module is available in global and venv.
deactivate
if ! check_python_module sentencepiece; then
    echo "Installing SentencePiece (python)..."
    pip install sentencepiece --quiet --progress-bar on
    echo "Installed SentencePiece (python)."
fi
source "$FAIRSEQ_ENV_DIR/bin/activate"
if ! check_python_module sentencepiece; then
    echo "Installing SentencePiece (python) on fairseq_env..."
    pip install sentencepiece --quiet --progress-bar on
    echo "Installed SentencePiece (python) on fairseq_env."
fi

# Final validation of tools.
if ! command -v spm_train &> /dev/null || ! check_python_module sentencepiece; then
    echo "[Error] SentencePiece command 'spm_train' or python module not found after installation."
    exit 1
fi
if ! command -v fairseq-preprocess &> /dev/null; then
    echo "[Error] 'fairseq-preprocess' command not found. Please install Fairseq."
    exit 1
fi

echo "Starting BPE, Preprocessing, and Datastore Build"
echo "------------------------------------------------------------"
echo "Using Project Root: $PROJECT_ROOT"
echo "Using Raw Corpus: $RAW_CORPUS_DIR"
echo "Using New Data Bin: $DATA_BIN_DIR"
echo "Using New NMT Model: $MODEL_SAVE_DIR"
echo "Using New Datastore: $DATASTORE_SAVE_PATH"
echo "------------------------------------------------------------"

# --- 1. Train BPE Model ---
# Create directory for BPE artifacts.
mkdir -p "$BPE_DIR"

# Define paths for FastBPE artifacts.
BPE_CODES_FASTBPE="$BPE_DIR/fastbpe.codes"
BPE_VOCAB_FASTBPE_RAW="$BPE_DIR/fastbpe.vocab.raw"
# Define paths for SentencePiece artifacts.
SPM_MODEL="$BPE_DIR/spm.model"
SPM_VOCAB_SPM_RAW="$BPE_DIR/spm.vocab"
# Define path for the final dictionary used by Fairseq.
FINAL_DICT_FOR_FAIRSEQ="$BPE_DIR/dict.txt"

# Retrieve BPE settings from config.
BPE_TYPE=$(get_nmt_model_settings "bpe")
BPE_VOCAB_SIZE=$(get_nmt_model_settings "bpe-vocab-size")

# Check and set defaults for BPE settings.
if [ -z "$BPE_TYPE" ]; then
    echo "[Warning] 'bpe' not set in BUILD_PIPELINE_SETTINGS. Defaulting to 'fastbpe'."
    BPE_TYPE="fastbpe"
fi
if [ -z "$BPE_VOCAB_SIZE" ]; then
    echo "[Error] 'bpe-vocab-size' not set in BUILD_PIPELINE_SETTINGS. Cannot train BPE/SPM."
    exit 1
fi
echo "[Success] Using BPE type: $BPE_TYPE"
echo "[Success] Using BPE vocab size: $BPE_VOCAB_SIZE"

# Check if BPE training is already complete.
if [ -e "$FINAL_DICT_FOR_FAIRSEQ" ]; then
    echo "[Success] Step 1: Final dictionary '$FINAL_DICT_FOR_FAIRSEQ' already exists. Skipping BPE/SPM training."
    # Verify corresponding model files exist.
    if [ "$BPE_TYPE" = "fastbpe" ] && [ ! -e "$BPE_CODES_FASTBPE" ]; then
        echo "[Error] Dict exists but '$BPE_CODES_FASTBPE' is missing."
        exit 1
    elif [[ ( "$BPE_TYPE" = "sentencepiece" ||  "$BPE_TYPE" = "sentence_piece" ) && ! -e "$SPM_MODEL" ]]; then
        echo "[Error] Dict exists but '$SPM_MODEL' is missing."
        exit 1
    fi
else
    echo "--- Step 1: Learning BPE/SPM from ALL raw data ---"
    
    # Locate all raw source and target language files (excluding datastore subsets).
    ALL_RAW_FILES=$(find "$RAW_CORPUS_DIR" -maxdepth 1 -type f \( -name "*.de" -o -name "*.en" \) | grep -v "_ds")
    
    if [ -z "$ALL_RAW_FILES" ]; then
        echo "[Error] No raw .de or .en files found in $RAW_CORPUS_DIR"
        exit 1
    fi

    # Concatenate all files to create a unified corpus for BPE learning.
    ALL_RAW_CONCAT_TMP="$BPE_DIR/all_raw_concat.tmp"
    echo "Concatenating $(echo $ALL_RAW_FILES | wc -w) files into $ALL_RAW_CONCAT_TMP..."
    cat $ALL_RAW_FILES > "$ALL_RAW_CONCAT_TMP"

    # Execute BPE training based on the selected type.
    if [ "$BPE_TYPE" = "fastbpe" ]; then
        echo "Learning BPE (fastbpe) from the single concatenated file..."
        fastbpe learnbpe "$BPE_VOCAB_SIZE" "$ALL_RAW_CONCAT_TMP" > "$BPE_CODES_FASTBPE"

        # Apply BPE temporarily to generate the vocabulary.
        BPE_CONCAT_TMP="$BPE_DIR/all_raw_concat.bpe.tmp"
        echo "Applying fastbpe to all data to generate master vocab..."
        fastbpe applybpe "$BPE_CONCAT_TMP" "$ALL_RAW_CONCAT_TMP" "$BPE_CODES_FASTBPE"
        
        echo "Generating master BPE vocab (raw) from BPE'd data..."
        fastbpe getvocab "$BPE_CONCAT_TMP" > "$BPE_VOCAB_FASTBPE_RAW"
        
        # FastBPE vocab is directly compatible with Fairseq.
        cp "$BPE_VOCAB_FASTBPE_RAW" "$FINAL_DICT_FOR_FAIRSEQ"

        rm "$BPE_CONCAT_TMP"
        echo "[Success] fastbpe training and master vocab creation complete."

    elif [ "$BPE_TYPE" = "sentencepiece" ] || [ "$BPE_TYPE" = "sentence_piece" ]; then
        echo "Learning SentencePiece (SPM) from the single concatenated file..."
        
        # Construct arguments for spm_train.
        spm_train_args=(
            --input="$ALL_RAW_CONCAT_TMP"
            --model_prefix="$BPE_DIR/spm"
            --vocab_size="$BPE_VOCAB_SIZE"
            --character_coverage="$(get_nmt_model_settings "spm-character-coverage")"
            --model_type="$(get_nmt_model_settings "spm-model-type")"
            --input_sentence_size="$(get_nmt_model_settings "spm-input-sentence-size")"
            $( [ "$(get_nmt_model_settings "spm-shuffle-input-sentence")" = "True" ] && echo "--shuffle_input_sentence=true" )
            --unk_id=3
            --bos_id=0
            --pad_id=1
            --eos_id=2
        )
        
        # Add unigram-specific settings if applicable.
        if [ "$(get_nmt_model_settings "spm-model-type")" = "unigram" ]; then
            spm_train_args+=(
                --num_threads="$(nproc)"
                --normalization_rule_name="$(get_nmt_model_settings "spm-normalization-rule-name")"
            )
            # Helper to convert Python bools to SPM CLI format.
            add_bool_arg() {
                local value="$(get_nmt_model_settings "$1")"
                [[ "${value,,}" =~ ^(true|1|yes|on)$ ]] && echo "--$2=true" || echo "--$2=false"
            }
            spm_train_args+=(
                $(add_bool_arg "spm-add-dummy-prefix" "add_dummy_prefix")
                $(add_bool_arg "spm-remove-extra-whitespaces" "remove_extra_whitespaces")
                $(add_bool_arg "spm-split-by-whitespace" "split_by_whitespace")
            )
        fi

        # Run training.
        spm_train "${spm_train_args[@]}"

        if [ ! -e "$SPM_MODEL" ] || [ ! -e "$SPM_VOCAB_SPM_RAW" ]; then
            echo "[Error] SentencePiece model training failed. '$SPM_MODEL' or '$SPM_VOCAB_SPM_RAW' not found."
            exit 1
        fi

        # Convert SPM vocab format to Fairseq dictionary format (token count).
        # Skip control tokens (top 4 lines) as Fairseq adds them automatically.
        echo "Converting SPM vocab to Fairseq dictionary format..."
        tail -n +5 "$SPM_VOCAB_SPM_RAW" | cut -f1 | awk '{ print $1 " 99" }' > "$FINAL_DICT_FOR_FAIRSEQ"

        echo "[Success] SentencePiece model training and vocab conversion complete."

    else
        echo "[Error] Unknown BPE type '$BPE_TYPE' in Step 1."
        exit 1
    fi
    
    # Remove the massive concatenated raw file.
    rm "$ALL_RAW_CONCAT_TMP"
fi

# --- 2. Apply BPE/SPM ---
echo "--- Step 2: Applying BPE/SPM ---"

# Check if regularization is enabled for SentencePiece (Unigram mode).
echo "Applying to NMT splits (train, valid, test)..."
spm_reg_args=""
if [[ ( "$BPE_TYPE" = "sentencepiece"  ||  "$BPE_TYPE" = "sentence_piece" ) && "$(get_nmt_model_settings "spm-model-type")" = "unigram" && "$(get_nmt_model_settings "spm-enable-regularization")" = "True" ]]; then
    NBEST_SIZE=$(get_nmt_model_settings "spm-nbest-size")
    ALPHA=$(get_nmt_model_settings "spm-alpha")
    spm_reg_args="--nbest_size=${NBEST_SIZE} --alpha=${ALPHA}"
    echo "[Success] Unigram regularization enabled for TRAINING splits (nbest=$NBEST_SIZE, alpha=$ALPHA)"
else
    echo "[Info] Unigram regularization is OFF."
fi

# Apply BPE to main NMT splits.
for SPLIT in train valid test; do
    for LANG in de en; do
        RAW_FILE="$RAW_CORPUS_DIR/$SPLIT.$LANG"
        BPE_FILE="$BPE_DIR/$SPLIT.bpe.$LANG"
        if [ ! -e "$RAW_FILE" ]; then
            echo "  [Warning] Raw file missing for NMT: $RAW_FILE. Skipping BPE application."
            continue
        fi
        if [ ! -e "$BPE_FILE" ]; then
            if [ "$BPE_TYPE" = "fastbpe" ]; then
                echo "  Applying fastbpe to NMT $SPLIT.$LANG..."
                fastbpe applybpe "$BPE_FILE" "$RAW_FILE" "$BPE_CODES_FASTBPE"
            elif [ "$BPE_TYPE" = "sentencepiece" ] || [ "$BPE_TYPE" = "sentence_piece" ]; then
                # Apply regularization only to the training split if enabled.
                if [ "$SPLIT" = "train" ] && [ -n "$spm_reg_args" ]; then
                    echo "  Applying sentencepiece to NMT $SPLIT.$LANG WITH regularization..."
                    spm_encode --model="$SPM_MODEL" --output_format=piece $spm_reg_args < "$RAW_FILE" > "$BPE_FILE"
                else
                    echo "  Applying sentencepiece to NMT $SPLIT.$LANG WITHOUT regularization..."
                    spm_encode --model="$SPM_MODEL" --output_format=piece < "$RAW_FILE" > "$BPE_FILE"
                fi
            fi
        else
            echo "  NMT $SPLIT.bpe.$LANG already exists."
        fi
    done
done

# Apply BPE to Datastore splits (subsets).
echo "Applying to Datastore splits (train_ds, valid_ds, test_ds)..."
for SPLIT_PREFIX in train_ds valid_ds test_ds; do 
    for LANG in de en; do
        RAW_FILE_DS="$RAW_CORPUS_DIR/$SPLIT_PREFIX.$LANG"
        BPE_FILE_DS="$BPE_DIR/$SPLIT_PREFIX.bpe.$LANG"
        if [ ! -e "$RAW_FILE_DS" ]; then
            echo "  [Warning] Raw file missing for DS: $RAW_FILE_DS. Skipping BPE application."
            continue
        fi
        if [ ! -e "$BPE_FILE_DS" ]; then
            if [ "$BPE_TYPE" = "fastbpe" ]; then
                echo "  Applying fastbpe to DS $SPLIT_PREFIX.$LANG..."
                fastbpe applybpe "$BPE_FILE_DS" "$RAW_FILE_DS" "$BPE_CODES_FASTBPE"
            elif [ "$BPE_TYPE" = "sentencepiece" ] || [ "$BPE_TYPE" = "sentence_piece" ]; then
                # Apply regularization to datastore training set if enabled.
                if [ "$SPLIT_PREFIX" = "train_ds" ] && [ -n "$spm_reg_args" ]; then
                    echo "  Applying sentencepiece to DS $SPLIT_PREFIX.$LANG WITH regularization..."
                    spm_encode --model="$SPM_MODEL" --output_format=piece $spm_reg_args < "$RAW_FILE_DS" > "$BPE_FILE_DS"
                else
                    echo "  Applying sentencepiece to DS $SPLIT_PREFIX.$LANG WITHOUT regularization..."
                    spm_encode --model="$SPM_MODEL" --output_format=piece < "$RAW_FILE_DS" > "$BPE_FILE_DS"
                fi
            fi
        else
            echo "  DS $SPLIT_PREFIX.bpe.$LANG already exists."
        fi
    done
done
echo "[Success] BPE/SPM application complete for both NMT and Datastore splits."

# --- 3. Fairseq Preprocessing ---

# 3.1 Preprocess for NMT Training (Full dataset).
mkdir -p "$DATA_BIN_DIR"
DICT_DE="$DATA_BIN_DIR/dict.de.txt"
DICT_EN="$DATA_BIN_DIR/dict.en.txt"

if [ -e "$DATA_BIN_DIR/preprocess.log" ]; then
    echo "[Success] Step 3.1: NMT $DATA_BIN_DIR already exists. Check existence of dict.de.txt and dict.en.txt before skipping fairseq-preprocess."
    # Ensure dictionaries exist.
    if [ ! -e "$DICT_DE" ] || [ ! -e "$DICT_EN" ]; then
        echo "[Error] NMT preprocess log exists, but dictionaries are missing in $DATA_BIN_DIR. Please clean and re-run."
        exit 1
    fi
else
    echo "--- Step 3.1: Running fairseq-preprocess for NMT (Creating dicts) ---"
    # Ensure master dictionary exists.
    if [ "$(get_nmt_model_settings "joined-dictionary")" = "False" ] && [ ! -e "$FINAL_DICT_FOR_FAIRSEQ" ]; then
        echo "[Error] Master dictionary '$FINAL_DICT_FOR_FAIRSEQ' not found when 'joined-dictionary'='False'."
        echo "        Re-run Step 1 or enable 'joined-dictionary'."
        exit 1
    fi
    
    # Run fairseq-preprocess.
    fairseq-preprocess \
        --source-lang "$(get_nmt_model_settings "source-lang")" \
        --target-lang "$(get_nmt_model_settings "target-lang")" \
        --trainpref "$BPE_DIR/train.bpe" \
        --validpref "$BPE_DIR/valid.bpe" \
        --testpref "$BPE_DIR/test.bpe" \
        --destdir "$DATA_BIN_DIR" \
        --workers "$(get_nmt_model_settings "workers")" \
        --dataset-impl "$(get_nmt_model_settings "dataset-impl")" \
        $( [ "$(get_nmt_model_settings "joined-dictionary")" = "True" ] && echo "--joined-dictionary" ) \
        $( [ "$(get_nmt_model_settings "joined-dictionary")" = "False" ] && echo "--srcdict $FINAL_DICT_FOR_FAIRSEQ" ) \
        $( [ "$(get_nmt_model_settings "joined-dictionary")" = "False" ] && echo "--tgtdict $FINAL_DICT_FOR_FAIRSEQ" ) \
        --thresholdsrc 0 \
        --thresholdtgt 0
    echo "[Success] NMT Fairseq preprocessing complete. Dictionaries and data in $DATA_BIN_DIR"
    
    # Validation check.
    if [ ! -e "$DICT_DE" ] || [ ! -e "$DICT_EN" ]; then
        echo "[Error] NMT preprocessing finished, but dictionaries were not created in $DATA_BIN_DIR."
        exit 1
    fi
fi

# 3.2 Preprocess for Datastore (Reuse dictionaries).
mkdir -p "$DATA_BIN_DIR_DS"
if [ -e "$DATA_BIN_DIR_DS/preprocess.log" ]; then
    echo "[Success] Step 3.2: Datastore $DATA_BIN_DIR_DS already exists. Skipping fairseq-preprocess."
else
    echo "--- Step 3.2: Running fairseq-preprocess for Datastore (Reusing NMT dicts) ---"
    # Ensure NMT dictionaries exist.
    if [ ! -e "$DICT_DE" ] || [ ! -e "$DICT_EN" ]; then
        echo "[Error] Cannot preprocess datastore data. NMT dictionaries not found in $DATA_BIN_DIR."
        exit 1
    fi
    fairseq-preprocess \
        --source-lang "$(get_nmt_model_settings "source-lang")" \
        --target-lang "$(get_nmt_model_settings "target-lang")" \
        --trainpref "$BPE_DIR/train_ds.bpe" \
        --validpref "$BPE_DIR/valid_ds.bpe" \
        --testpref "$BPE_DIR/test_ds.bpe" \
        --destdir "$DATA_BIN_DIR_DS" \
        --workers "$(get_nmt_model_settings "workers")" \
        --dataset-impl "$(get_nmt_model_settings "dataset-impl")" \
        --srcdict "$DICT_DE" \
        --tgtdict "$DICT_EN"
    echo "[Success] Datastore Fairseq preprocessing complete. Data in $DATA_BIN_DIR_DS"
fi

# --- 4. NMT Model Training ---
echo "--- Step 4: Checking for NMT Model ---"
NMT_BEST_MODEL="$MODEL_SAVE_DIR/$(get_nmt_model_settings "nmt-model-dir")/checkpoint_best.pt"
NMT_LAST_MODEL="$MODEL_SAVE_DIR/$(get_nmt_model_settings "nmt-model-dir")/checkpoint_last.pt"
NMT_COMPLETE_DIR="$MODEL_SAVE_DIR/$(get_nmt_model_settings "nmt-model-dir")/complete.txt"

echo "Checking for completion status of NMT training..."
# Check if training is already marked complete or if existing checkpoints match configuration requirements.
if [ -e "$NMT_COMPLETE_DIR" ] || { [ -e "$NMT_BEST_MODEL" ] && [ -e "$NMT_LAST_MODEL" ] && python -c "
import torch
import sys
best_ckpt = torch.load('$NMT_BEST_MODEL', weights_only=False)
last_ckpt = torch.load('$NMT_LAST_MODEL', weights_only=False)
best_epoch = best_ckpt['extra_state']['train_iterator']['epoch']
last_epoch = last_ckpt['extra_state']['train_iterator']['epoch']
best_max_epoch = best_ckpt['args'].max_epoch
best_patience = best_ckpt['args'].patience
last_max_epoch = last_ckpt['args'].max_epoch
last_patience = last_ckpt['args'].patience
ext_max_epoch = $(get_nmt_model_settings "max-epoch")
ext_patience = $(get_nmt_model_settings "patience")
if best_max_epoch != last_max_epoch or best_patience != last_patience:
    sys.exit(1)
if best_max_epoch != ext_max_epoch or best_patience != ext_patience:
    sys.exit(1)
diff = last_epoch - best_epoch
if last_epoch == ext_max_epoch or not (diff >= ext_patience):
    sys.exit(0)
else:
    sys.exit(1)
"; }; then
    echo "[Success] Found Complete NMT BEST Model: $NMT_BEST_MODEL"
else
    echo "NMT Model not found or broken. Trying to start training..."
    
    # Optimize I/O by copying data to a local SSD temporary directory.
    echo "Copying data to local SSD for faster I/O..."
    LOCAL_DATA_BIN="/tmp/paec_corpus_combined"
    LOCAL_MODEL_DIR="/tmp/paec_corpus_model"
    
    # Patch Fairseq options for BF16 compatibility if needed.
    if [[ "$(get_nmt_model_settings "precision")" =~ ^(bf16|memory-efficient-bf16)$ ]]; then
        sed -i '/^    if args\.bf16:$/ { N; /\n        args\.tpu = True$/ d }' "$KNNBOX_LIB_DIR/fairseq/options.py"
    fi
    
    # Setup local directories and copy data.
    rm -rf "$LOCAL_DATA_BIN" "$LOCAL_MODEL_DIR"
    mkdir -p "$LOCAL_DATA_BIN" "$LOCAL_MODEL_DIR" "$MODEL_SAVE_DIR/$(get_nmt_model_settings "nmt-model-dir")"
    find "$DATA_BIN_DIR" -maxdepth 1 -type f | grep -q . && cp -rf "$DATA_BIN_DIR"/* "$LOCAL_DATA_BIN"/
    find "$MODEL_SAVE_DIR/$(get_nmt_model_settings "nmt-model-dir")" -maxdepth 1 -type f | grep -q . && cp -rf "$MODEL_SAVE_DIR/$(get_nmt_model_settings "nmt-model-dir")"/* "$LOCAL_MODEL_DIR"/
    
    # Configure Fused LayerNorm based on precision settings.
    if [ "$(get_nmt_model_settings "precision")" = "bf16" ] || [ "$(get_nmt_model_settings "precision")" = "memory-efficient-bf16" ]; then
        export FAIRSEQ_USE_FUSED_LAYERNORM=0
    else
        export FAIRSEQ_USE_FUSED_LAYERNORM=1
    fi
    
    # Construct arguments for fairseq-train.
    nmt_args=(
        "$LOCAL_DATA_BIN"
        --source-lang "$(get_nmt_model_settings "source-lang")"
        --target-lang "$(get_nmt_model_settings "target-lang")"
        --arch "$(get_nmt_model_settings "arch")"
        --max-source-positions "$(get_nmt_model_settings "max-source-positions")"
        --max-target-positions "$(get_nmt_model_settings "max-target-positions")"
        --optimizer "$(get_nmt_model_settings "optimizer")"
        --lr "$(get_nmt_model_settings "lr")"
        --lr-scheduler "$(get_nmt_model_settings "lr-scheduler")"
        --warmup-updates "$(get_nmt_model_settings "warmup-updates")"
        --weight-decay "$(get_nmt_model_settings "weight-decay")"
        --criterion "$(get_nmt_model_settings "criterion")"
        --label-smoothing "$(get_nmt_model_settings "label-smoothing")"
        --max-tokens "$(get_nmt_model_settings "max-tokens")"
        --max-tokens-valid "$(get_nmt_model_settings "max-tokens-valid")"
        --update-freq "$(get_nmt_model_settings "update-freq")"
        "--$(get_nmt_model_settings "precision")"
        --num-workers "$(get_nmt_model_settings "num-workers")"
        --max-epoch "$(get_nmt_model_settings "max-epoch")"
        --patience "$(get_nmt_model_settings "patience")"
        --log-interval "$(get_nmt_model_settings "log-interval")"
        --save-dir "$LOCAL_MODEL_DIR"
    )
    
    # Add architectural overrides if training a custom model.
    if [ "$(get_nmt_model_settings "self-nmt-train")" = "True" ]; then
        nmt_args+=(
            --encoder-layers "$(get_nmt_model_settings "encoder-layers")"
            --decoder-layers "$(get_nmt_model_settings "decoder-layers")"
            --encoder-embed-dim "$(get_nmt_model_settings "encoder-embed-dim")"
            --decoder-embed-dim "$(get_nmt_model_settings "decoder-embed-dim")"
            --encoder-ffn-embed-dim "$(get_nmt_model_settings "encoder-ffn-embed-dim")"
            --decoder-ffn-embed-dim "$(get_nmt_model_settings "decoder-ffn-embed-dim")"
            --encoder-attention-heads "$(get_nmt_model_settings "encoder-attention-heads")"
            --decoder-attention-heads "$(get_nmt_model_settings "decoder-attention-heads")"
             --dropout "$(get_nmt_model_settings "dropout")"
            --attention-dropout "$(get_nmt_model_settings "attention-dropout")"
            --activation-dropout "$(get_nmt_model_settings "activation-dropout")"
            --activation-fn "$(get_nmt_model_settings "activation-fn")"
        )
        if [ "$(get_nmt_model_settings "share-all-embeddings")" = "True" ]; then
            nmt_args+=(--share-all-embeddings)
        fi
    fi
    
    # Add optimizer arguments.
    if [ "$(get_nmt_model_settings "optimizer")" = "adam" ]; then
        nmt_args+=(--adam-betas "$(get_nmt_model_settings "adam-betas")")
    fi
    
    # Add logging and checkpointing arguments.
    if [ "$(get_nmt_model_settings "no-epoch-checkpoints")" = "True" ]; then
        nmt_args+=("--no-epoch-checkpoints")
    fi
    if [ "$(get_nmt_model_settings "no-last-checkpoints")" = "True" ]; then
        nmt_args+=("--no-last-checkpoints")
    fi
    if [ "$(get_nmt_model_settings "no-progress-bar")" = "True" ]; then
        nmt_args+=(--no-progress-bar)
    fi
    if [ "$(get_nmt_model_settings "no-save-optimizer-state")" = "True" ]; then
        nmt_args+=(--no-save-optimizer-state)
    fi
    
    # Execute Training.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_WAIT_POLICY=PASSIVE MKL_THREADING_LAYER=GNU fairseq-train "${nmt_args[@]}"
    
    # Sync trained model back to permanent storage.
    find "$LOCAL_MODEL_DIR" -maxdepth 1 -type f | grep -q . && cp -rf "$LOCAL_MODEL_DIR"/* "$MODEL_SAVE_DIR/$(get_nmt_model_settings "nmt-model-dir")"/
    [ ! -f "$NMT_COMPLETE_DIR" ] && touch "$NMT_COMPLETE_DIR"
    echo "[Success] NMT Model training complete. Model saved to $MODEL_SAVE_DIR/$(get_nmt_model_settings "nmt-model-dir")"
fi

# --- 5. Build Datastore ---
# Set up datastore building paths and basic arguments.
DS_MODEL_PATH="$MODEL_SAVE_DIR/$(get_nmt_model_settings "nmt-model-dir")/$(get_nmt_model_settings "nmt-model")"
DS_ARGS=(
    "$DATA_BIN_DIR_DS"
    --task translation
    --path "$DS_MODEL_PATH"
    --model-overrides "$(get_ds_model_overrides_str)"
    --dataset-impl "$(get_nmt_model_settings "dataset-impl")"
    --valid-subset train
    $( [ "$(get_nmt_model_settings "skip-invalid-size-inputs-valid-test")" = "True" ] && echo "--skip-invalid-size-inputs-valid-test" )
    --max-tokens "$(get_nmt_model_settings "max-tokens")"
    --user-dir "$KNNBOX_LIB_DIR/knnbox/models"
    --arch "$(get_nmt_model_settings "knn-arch-model")@$(get_nmt_model_settings "arch")"
    --knn-mode build_datastore
)

# Append BPE arguments for datastore building.
if [ "$BPE_TYPE" = "sentencepiece" ] ||  [ "$BPE_TYPE" = "sentence_piece" ]; then
    DS_ARGS+=(
        --bpe sentencepiece
        --sentencepiece-model "$SPM_MODEL"
    )
elif [ "$BPE_TYPE" = "fastbpe" ]; then
    DS_ARGS+=(
        --bpe fastbpe
        --bpe-codes "$BPE_CODES_FASTBPE"
    )
else
    echo "[Error] Unknown/unsupported BPE type '$BPE_TYPE' in Step 5."
    exit 1
fi

# 5.1 Construct 'exact' Datastore (Flat L2 Index).
if [ -e "$DATASTORE_SAVE_PATH/exact/config.json" ]; then
    echo "[Success] Step 5-1: Datastore 'exact' already exists. Skipping build."
else
    echo "--- Step 5-1: Building Datastore 'exact' (using 'faiss.IndexFlatL2()' and ONLY on 'train' split) ---"
    # Patch utils.py to use IndexFlatL2.
    sed -i -e '/^        index_dim = pca_dim if do_pca else dimension$/a \
        index = faiss.IndexFlatL2(index_dim)\
        index = faiss.IndexIDMap(index)\
' -e '/^        index_dim = pca_dim if do_pca else dimension$/,/^        if use_gpu:$/{
/        index_dim = pca_dim if do_pca else dimension/b
/        if use_gpu:/b
d
}' "$KNNBOX_DATASTORE_UTIL_FILE_DIR"
    
    if [ "$(get_nmt_model_settings "precision")" = "bf16" ] || [ "$(get_nmt_model_settings "precision")" = "memory-efficient-bf16" ]; then
        export FAIRSEQ_USE_FUSED_LAYERNORM=0
    else
        export FAIRSEQ_USE_FUSED_LAYERNORM=1
    fi
    
    # Execute datastore build.
    OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=0 \
    python "$KNNBOX_LIB_DIR/knnbox-scripts/common/validate.py" --knn-datastore-path "$DATASTORE_SAVE_PATH/exact" "${DS_ARGS[@]}"
    echo "[Success] Datastore 'EXACT' build complete. Files are in $DATASTORE_SAVE_PATH/exact"
fi

# 5.2 Construct 'hnsw' Datastore (HNSW Graph Index).
mkdir -p "$DATASTORE_SAVE_PATH/hnsw"
if [ -e "$DATASTORE_SAVE_PATH/hnsw/config.json" ]; then
    echo "[Success] Step 5-2: Datastore 'hnsw' already exists. Skipping build."
else
    echo "--- Step 5-2: Building Datastore 'hnsw' (using 'faiss.IndexHNSWFlat()' and ONLY on 'train' split) ---"
    # Patch utils.py to use IndexHNSWFlat.
    sed -i -e '/^        index_dim = pca_dim if do_pca else dimension$/a \
        index = faiss.IndexHNSWFlat(index_dim, 32)\
        index = faiss.IndexIDMap(index)\
' -e '/^        index_dim = pca_dim if do_pca else dimension$/,/^        if use_gpu:$/{
/        index_dim = pca_dim if do_pca else dimension/b
/        if use_gpu:/b
d
}' "$KNNBOX_DATASTORE_UTIL_FILE_DIR"
    
    if [ "$(get_nmt_model_settings "precision")" = "bf16" ] || [ "$(get_nmt_model_settings "precision")" = "memory-efficient-bf16" ]; then
        export FAIRSEQ_USE_FUSED_LAYERNORM=0
    else
        export FAIRSEQ_USE_FUSED_LAYERNORM=1
    fi
    
    # Execute datastore build.
    OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=0 \
    python "$KNNBOX_LIB_DIR/knnbox-scripts/common/validate.py" --knn-datastore-path "$DATASTORE_SAVE_PATH/hnsw" "${DS_ARGS[@]}"
    echo "[Success] Datastore 'hnsw' build complete. Files are in $DATASTORE_SAVE_PATH/hnsw"
fi

# 5.3 Construct 'ivf_pq' Datastore (Inverted File Product Quantization Index).
mkdir -p "$DATASTORE_SAVE_PATH/ivf_pq"
if [ -e "$DATASTORE_SAVE_PATH/ivf_pq/config.json" ]; then
    echo "[Success] Step 5-3: Datastore 'ivf_pq' already exists. Skipping build."
else
    echo "--- Step 5-3: Building Datastore 'ivf_pq' (using 'faiss.IndexIVFPQ()' and ONLY on 'train' split) ---"
    # Patch utils.py to use IndexIVFPQ.
    sed -i -e '/^        index_dim = pca_dim if do_pca else dimension$/a \
        quantizer = faiss.IndexFlatL2(index_dim)\
        index = faiss.IndexIVFPQ(quantizer, index_dim, n_centroids, code_size, 8)\
        index.nprobe = n_probe
' -e '/^        index_dim = pca_dim if do_pca else dimension$/,/^        if use_gpu:$/{
/        index_dim = pca_dim if do_pca else dimension/b
/        if use_gpu:/b
d
}' "$KNNBOX_DATASTORE_UTIL_FILE_DIR"
    
    if [ "$(get_nmt_model_settings "precision")" = "bf16" ] || [ "$(get_nmt_model_settings "precision")" = "memory-efficient-bf16" ]; then
        export FAIRSEQ_USE_FUSED_LAYERNORM=0
    else
        export FAIRSEQ_USE_FUSED_LAYERNORM=1
    fi
    
    # Execute datastore build.
    OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=0 \
    python "$KNNBOX_LIB_DIR/knnbox-scripts/common/validate.py" --knn-datastore-path "$DATASTORE_SAVE_PATH/ivf_pq" "${DS_ARGS[@]}"
    echo "[Success] Datastore 'ivf_pq' build complete. Files are in $DATASTORE_SAVE_PATH/ivf_pq"
fi

echo "-------------------------------------"
echo "Pipeline Build Finished Successfully!"
echo "-------------------------------------"
