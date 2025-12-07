#!/bin/bash
# Exit immediately if a command exits with a non-zero status to ensure setup integrity.
set -e

echo "1. Setting up environment..."

# Determine the absolute path to the project root directory.
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# Dynamically load the 'PATHS' configuration from 'src/config.py' and convert it to a JSON string for Bash parsing.
# This ensures path consistency between Python scripts and Shell scripts.
CONFIG_PATHS_JSON=$(python -c "import sys; sys.path.append('$PROJECT_ROOT'); from src.config import PATHS; import json; print(json.dumps({k: str(v) for k, v in PATHS.items()}, indent=4))")
if [ $? -ne 0 ]; then
    echo "[Error] Failed to load paths from src/config.py. Ensure it's runnable."
    exit 1
fi

# Dynamically load 'BUILD_PIPELINE_SETTINGS' from 'src/config.py' to retrieve model training configurations.
CONFIG_BUILD_PIPELINE_SETTINGS_JSON=$(python -c "import sys; sys.path.append('$PROJECT_ROOT'); from src.config import BUILD_PIPELINE_SETTINGS; import json; print(json.dumps({k: str(v) for k, v in BUILD_PIPELINE_SETTINGS.items()}, indent=4))")
if [ $? -ne 0 ]; then
    echo "[Error] Failed to load pipeline settings from src/config.py. Ensure it's runnable."
    exit 1
fi

# Define helper functions to parse specific keys from the loaded JSON configuration strings.
get_path() { echo "$CONFIG_PATHS_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('$1', ''))"; }
get_nmt_model_settings() { echo "$CONFIG_BUILD_PIPELINE_SETTINGS_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('$1', ''))"; }

# Retrieve critical directory paths from the configuration.
PAEC_PROJECT_DIR=$(get_path "proj_dir")
KNNBOX_LIB_DIR=$(get_path "knn_box_dir")
FAIRSEQ_ENV_DIR="$PROJECT_ROOT/fairseq_env"

# Grant execution permissions to scripts and write permissions to project directories.
chmod -R 755 "$PROJECT_ROOT/src"
chmod -R 755 "$PROJECT_ROOT/scripts"
chmod -R 755 "$PROJECT_ROOT/tmps"
chmod -R 755 "$PAEC_PROJECT_DIR"

# Update system package lists and install necessary system-level dependencies.
# This includes build tools (gcc, g++), Python versions (3.8 for legacy Fairseq, 3.12 for main logic), and math libraries (MKL).
sudo apt-get update > /dev/null 2>&1
sudo apt-get install -y software-properties-common > /dev/null 2>&1
sudo add-apt-repository ppa:deadsnakes/ppa -y > /dev/null 2>&1
sudo apt-get update > /dev/null 2>&1
sudo apt-get install -y pv build-essential gcc g++ libomp-dev python3.8 python3.8-dev python3.8-venv python3.12 python3.12-dev python3.12-venv libmkl-dev libmkl-intel-lp64 dos2unix sentencepiece > /dev/null 2>&1
sudo apt-get upgrade > /dev/null 2>&1

echo "Cloning and patching knn-box library..."

# Clone the 'knn-box' library if it doesn't exist, which provides the base implementation for kNN-MT.
if [ ! -e "$PAEC_PROJECT_DIR/libs/knn-box" ]; then
    cd "$PAEC_PROJECT_DIR/libs"
    git clone https://github.com/NJUNLP/knn-box.git
    cd $PAEC_PROJECT_DIR/libs/knn-box
    
    # Patch 'knn-box' source code to replace deprecated NumPy types (np.int, np.float) with native Python types.
    # This prevents AttributeErrors with newer NumPy versions.
    sed -i -e 's/\bnp\.int\b/int/g' -e 's/\bnp\.float\b/float/g' $KNNBOX_LIB_DIR/knnbox/datastore/pck_datastore.py
    sed -i -e 's/\bnp\.int\b/int/g' -e 's/\bnp\.float\b/float/g' $KNNBOX_LIB_DIR/fairseq/data/data_utils.py
    sed -i -e 's/\bnp\.int\b/int/g' -e 's/\bnp\.float\b/float/g' $KNNBOX_LIB_DIR/fairseq/data/indexed_dataset.py
    sed -i -e 's/\bnp\.int\b/int/g' -e 's/\bnp\.float\b/float/g' $KNNBOX_LIB_DIR/fairseq/modules/dynamic_crf_layer.py
    
    # Patch 'dataclass' field definitions to resolve syntax errors in newer Python versions.
    sed -i -E 's/([a-zA-Z_]+: [a-zA-Z_]+) = \b([a-zA-Z_]+)\(\)/\1 = field(default_factory=\2)/g' $KNNBOX_LIB_DIR/fairseq/dataclass/data_class.py
    
    # Build C++ extensions for knn-box.
    python setup.py build_ext --inplace
    cd $PROJECT_ROOT
fi

echo "2. Setting up Python 3.12 environment..."

# Upgrade pip for the main Python 3.12 environment.
python3.12 -m pip install --upgrade pip --quiet --progress-bar on

echo "Installing required Python packages for Python 3.12..."

# Install main project requirements for Python 3.12, pointing to the CUDA 12.6 extra index for PyTorch.
python3.12 -m pip install -r "$PAEC_PROJECT_DIR/requirements_linux_main.txt" --quiet --progress-bar on --extra-index-url https://download.pytorch.org/whl/cu126

echo "Installing fastBPE and other dependencies for Python 3.12..."

# Install auxiliary tools: tensorboardX and fastBPE (C++ implementation for speed).
python3.12 -m pip install tensorboardX > /dev/null 2>&1
rm -rf fastBPE
git clone "https://github.com/glample/fastBPE.git"

# Conditionally clone NVIDIA Apex if 'self-train' is enabled in the config, required for optimized mixed-precision training.
if [ "$(get_nmt_model_settings "self-train")" = "True" ]; then
    git clone https://github.com/NVIDIA/apex
fi

# Install fastBPE for Python 3.12.
cd $PROJECT_ROOT/fastBPE
python3.12 setup.py install
cd $PROJECT_ROOT

# Download spaCy models for German and English NLP tasks (NER, etc.).
spacy download en_core_web_lg
spacy download de_core_news_lg
spacy download de_core_news_sm

echo "3. Setting up fairseq_env (Python 3.8)..."

# Define a function to verify the installation of NVIDIA Apex and its AMP (Automatic Mixed Precision) functionality.
check_apex_installation() {
    # Check if apex module is importable and CUDA is available.
    export LD_LIBRARY_PATH="$FAIRSEQ_ENV_DIR/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
    if ! python -c "
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
    # Verify that Apex's C++ extension (amp_C) is loadable.
    python -c "import amp_C; import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('Apex AMP is functional')" 2>/dev/null
    return $?
}

# Check if the pre-packaged 'fairseq_env' tarball exists. If not, build the environment from scratch.
if [ ! -e "$PAEC_PROJECT_DIR/fairseq_env.tar.gz" ]; then
    echo "Start setup process of fairseq_env (Python 3.8)"
    
    # Create a Python 3.8 virtual environment to support legacy Fairseq dependencies.
    python3.8 -m venv $FAIRSEQ_ENV_DIR
    
    # Set 'python' command to point to the venv's Python 3.8 to ensure script compatibility.
    sudo update-alternatives --install /usr/bin/python python "$FAIRSEQ_ENV_DIR/bin/python" 101 > /dev/null 2>&1
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 51 > /dev/null 2>&1
    sudo update-alternatives --auto python > /dev/null 2>&1
    
    # Activate the environment.
    source "$FAIRSEQ_ENV_DIR/bin/activate"
    
    # Install pip dependencies for the legacy environment.
    python3.8 -m pip install --upgrade pip --quiet --progress-bar on
    python3.8 -m pip install -e $PAEC_PROJECT_DIR/libs/knn-box --force-reinstall --quiet --progress-bar on
    python3.8 -m pip install -r $PAEC_PROJECT_DIR/requirements_linux_3.8.txt --quiet --progress-bar on
    
    # Re-install knn-box in editable mode to ensure local changes are reflected.
    python3.8 -m pip install -e $PAEC_PROJECT_DIR/libs/knn-box --quiet --progress-bar on
    python3.8 -m pip install tensorboardX > /dev/null 2>&1
    
    # Install fastBPE within the Python 3.8 environment.
    echo "Installing fastBPE..."
    rm -rf fastBPE
    git clone https://github.com/glample/fastBPE.git
    chmod -R u+rw $PROJECT_ROOT/fastBPE
    cd $PROJECT_ROOT/fastBPE
    export PATH="$FAIRSEQ_ENV_DIR/bin:$PATH"
    python3.8 setup.py install
    cd $PROJECT_ROOT
    rm -rf fastBPE
    echo "fastBPE installation completed!"
    
    # Install NVIDIA Apex if self-training is enabled in the configuration.
    if [ "$(get_nmt_model_settings "self-train")" = "True" ]; then
        echo "Installing Apex..."
        if [ -d "apex" ]; then
            echo "Removing existing 'apex' directory..."
            rm -rf apex
        fi
        # Ensure clean state.
        pip uninstall apex -y
        rm -rf $PROJECT_ROOT/apex
        git clone https://github.com/NVIDIA/apex
        cd apex || { echo "Error: Failed to enter apex directory."; exit 1; }
        
        # Configure compilation flags for performance (AVX, CUDA arch).
        MAX_JOBS=$(nproc)
        APEX_PARALLEL_BUILD=8
        CUDA_NVCC_FLAGS="-O3"
        CFLAGS="-O3 -march=native"
        CXXFLAGS="-O3 -march=native"
        APEX_DISABLE_EXTENSION_CHECKS=1
        TORCH_CUDA_ARCH_LIST="8.0" # Target Ampere architecture (A100).
        MKL_THREADING_LAYER=GNU
        NVCC_APPEND_FLAGS="--threads 8"
        rm -rf build
        
        # Patch Apex's setup.py to fix compatibility issues with newer Python versions (type hint syntax).
        sed -i '/^def check_cuda_torch_binary_vs_bare_metal(/a\    return' setup.py
        sed -i 's/parallel: int | None = None/parallel: typing.Optional[int] = None/g' setup.py
        sed -i 's/import sys/import sys, typing/g' setup.py
        
        # Build and install Apex with CUDA and C++ extensions.
        python3.8 setup.py install --cuda_ext --cpp_ext > ../apex_install.log 2>&1
        if [ $? -ne 0 ]; then
            echo "Error: Apex installation failed. Check apex_install.log for details."
            cat apex_install.log
            cd ..
            rm -rf apex
            exit 1
        fi
        cd ..
        rm -rf apex
        echo "Apex installation completed!"
        
        # Verify that Apex works as expected.
        if ! check_apex_installation; then
            echo "Error: Apex installed but AMP functionality is not working."
            exit 1
        fi
    else
        # If self-training is disabled, remove Apex to avoid conflicts.
        pip uninstall apex --quiet
    fi
    
    # Download spaCy models for the Python 3.8 environment.
    spacy download en_core_web_lg
    spacy download de_core_news_lg
    spacy download de_core_news_sm
    
    # Archive the configured environment into a tarball for future reuse/caching.
    tar -czf $PROJECT_ROOT/fairseq_env.tar.gz \
        --exclude="__pycache__" \
        --exclude="*.pyc" \
        $FAIRSEQ_ENV_DIR
    cp -rf $PROJECT_ROOT/fairseq_env.tar.gz $PAEC_PROJECT_DIR/
    deactivate
    echo "Setup process of fairseq_env (Python 3.8) complete!"
else
    # If the environment tarball exists, restore it to skip the build process.
    echo "fairseq_env (Python 3.8) detected successful, now setting up fairseq_env into environmental variables..."
    cp -rf $PAEC_PROJECT_DIR/fairseq_env.tar.gz $PROJECT_ROOT/
    pv $PROJECT_ROOT/fairseq_env.tar.gz | tar -xz -C /
    rm -rf $PROJECT_ROOT/fairseq_env.tar.gz
    
    # Fix permissions and activate the restored environment.
    chmod -R 755 $FAIRSEQ_ENV_DIR
    source $FAIRSEQ_ENV_DIR/bin/activate
    export PATH="$FAIRSEQ_ENV_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$FAIRSEQ_ENV_DIR/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
    
    # Ensure fastBPE binary is available in the system path.
    if ! command -v fastbpe &> /dev/null; then
        echo "Installing fastBPE..."
        rm -rf $PROJECT_ROOT/fastBPE
        git clone https://github.com/glample/fastBPE.git
        cd fastBPE
        g++ -O3 -std=c++11 -pthread fastBPE/main.cc -IfastBPE -o fastbpe
        sudo mv fastbpe /usr/local/bin/
        python3.8 setup.py install
        cd ..
    fi
    
    # Verify Apex if self-training is enabled, re-installing if necessary.
    if [ "$(get_nmt_model_settings "self-train")" = "True" ]; then
        if check_apex_installation; then
            echo "✅ NVIDIA Apex is ready to use with functional AMP."
        else
            echo "Installing Apex..."
            pip uninstall apex -y
            rm -rf $PROJECT_ROOT/apex
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
            
            # Apply patches again for re-installation.
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
            
            # Setup paths again after potential deactivation during install.
            deactivate
            export PATH="$FAIRSEQ_ENV_DIR/bin:$PATH"
            export LD_LIBRARY_PATH="$FAIRSEQ_ENV_DIR/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
            source "$FAIRSEQ_ENV_DIR/bin/activate"
            export PATH="$FAIRSEQ_ENV_DIR/bin:$PATH"
            export LD_LIBRARY_PATH="$FAIRSEQ_ENV_DIR/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
            
            if ! check_apex_installation; then
                echo "Error: Apex installed but AMP functionality is not working."
                exit 1
            fi
            
            # Update the cached tarball with the fixed Apex installation.
            deactivate
            rm -rf $PROJECT_ROOT/fairseq_env.tar.gz
            tar -czf $PROJECT_ROOT/fairseq_env.tar.gz --exclude="__pycache__" --exclude="*.pyc" $FAIRSEQ_ENV_DIR
            cp -rf $PROJECT_ROOT/fairseq_env.tar.gz $PAEC_PROJECT_DIR/
            echo "✅ Apex verified with functional AMP."
            source "$FAIRSEQ_ENV_DIR/bin/activate"
        fi
    fi
    
    # Finalize environment setup.
    export PATH="$FAIRSEQ_ENV_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$FAIRSEQ_ENV_DIR/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
    deactivate
    chmod -R 755 $FAIRSEQ_ENV_DIR
    export PATH="$FAIRSEQ_ENV_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$FAIRSEQ_ENV_DIR/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
    
    # Set default python alternatives for the shell context.
    sudo update-alternatives --install /usr/bin/python python "$FAIRSEQ_ENV_DIR/bin/python" 101 > /dev/null 2>&1
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 51 > /dev/null 2>&1
    sudo update-alternatives --auto python > /dev/null 2>&1
    echo "fairseq_env (Python 3.8) setup successful!"
fi

# Clean up system packages and pip cache to free up space.
sudo apt-get autoclean -y
sudo apt-get autoremove -y
pip cache purge
find / -name "pip_cache" 2>/dev/null
sudo rm -rf /usr/lib/python*/site-packages/pip/_vendor/cache/
