# -*- coding: utf-8 -*-
import sys, random, json, traceback, re, torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---

# Ensure src is in the path to import config
# Assumes this script runs from 'scripts' directory relative to project root
try:
    # Resolve the project root directory from the current script location.
    script_dir = Path(__file__).parent.parent.resolve()
    project_root = script_dir.parent
    
    # Add project root to sys.path to enable imports from the 'src' package.
    sys.path.append(str(project_root))
    # Dynamically import DATA_LOADER_PARAMS and other settings from src.config.
    # It's crucial that src/config.py exists and defines these dictionaries.
    from src.config import TARGET_TOTAL_SAMPLES_CORPUS, CORPUS_SAMPLING_SETTINGS, PATHS, DATA_LOADER_PARAMS
    print("[Success] Successfully imported required params from src.config.")

except ImportError:
    # Error handling for missing configuration imports.
    print("[Error] Could not import DATA_LOADER_PARAMS from src.config.")
    print("  Ensure 'src/config.py' exists and defines DAT_LOADER_PARAMS.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    # Catch-all for other import errors.
    print(f"[Error] An unexpected error occurred during config import: {e}")
    traceback.print_exc()
    sys.exit(1)

# Define canonical paths relative to project root.
# PROJ_DIR is typically defined in config.py, but re-constructed here for local reference if needed.
DRIVE_DIR = project_root / "drive" / "MyDrive" / "PAEC_proj"
# Directory to store cached Hugging Face datasets.
MODELS_CACHE_DIR = PATHS["models_dir"] / "dataset_cache"
# Directory where the final raw text files (.de, .en) will be saved.
RAW_CORPUS_DIR = PATHS["raw_corpus_dir"] # Changed output dir name
# Directory to save index mapping files (for traceability).
INDICES_DIR = RAW_CORPUS_DIR / "indices" # Directory to save index files

# Languages (can be made configurable if needed).
LANG_SRC = "de"
LANG_TGT = "en"

# Manual split ratios if valid/test are missing in the source dataset configuration.
MANUAL_VALID_RATIO = DATA_LOADER_PARAMS["valid_split_ratio"] if "valid_split_ratio" in DATA_LOADER_PARAMS else 0.10
MANUAL_TEST_RATIO  = DATA_LOADER_PARAMS["test_split_ratio"]  if "test_split_ratio"  in DATA_LOADER_PARAMS else 0.10

# Set random seeds for reproducibility.
SEED = DATA_LOADER_PARAMS.get("random_seed", 42)
random.seed(SEED)
np.random.seed(SEED)

# --- Helper Functions ---

def normalize_text_batch(texts: List[str], lang: str) -> List[str]:
    """
    Performs basic text normalization and filtering on a batch of strings.
    - Trust the source data (parallel corpora are already aligned)
    - Minimal intervention (let BPE and model handle complexity)
    - Fast processing (no expensive language detection)
    
    Returns:
        A list where valid texts are cleaned strings and invalid ones are None.
    """
    results = []
    # de_en_char_pattern = re.compile(r'^[§¶©®™°±×÷$%^&*_=+`~|/\\0-9a-zA-ZäöüÄÖÜßàáâãåæçèéêëìíîïðñòóôõøùúûýþÿÀÁÂÃÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕØÙÚÛÝÞß\s\.,!?;:\-\'"\(\)\[\]\{\}]+$')
    
    # Regex patterns to exclude specific noisy formats.
    exclude_patterns = [
        # re.compile(r'\d\s*:\s*\d'), re.compile(r'\d\s*/\s*\d'), re.compile(r'\d\s*,\s*\d'),
        # re.compile(r'\d\s*[.,;:!?(){}\[\]\'"«»„“”\-]?\s*\d{2,}\s*[.,;:!?(){}\[\]\'"«»„“”\-]?\s*\d')
    ]
    for text in texts:
        # 1. Strip whitespace.
        text = text.strip()
        
        # 2. Skip empty lines.
        if not text:
            results.append(None)
            continue
        
        # 3. standard length filter (words count).
        n_words = len(text.split())
        if n_words < 3 or n_words > 120:
            results.append(None)
            continue
        
        # 4. excluding sentences with website URLs, HTML tags, etc.
        excluded_texts = [
            "URL", "HTTP://", "HTTPS://", "WWW.", ".PHP", ".HTM", 
            "<A HREF=", "<S>", "</S>",
            # "_", "....", "+49", "UTC+", "''", "``", " TEL.",
        ]
        is_jump = False; text_upper = text.upper()
        for excluded_text in excluded_texts:
            if excluded_text in text_upper: results.append(None); is_jump = True; break
        if is_jump: continue
        
        # Check against exclusion patterns.
        for exclude_pattern in exclude_patterns:
            if exclude_pattern.search(text_upper): results.append(None); is_jump = True; break
        if is_jump: continue
        
        # Normalize multiple spaces to single space and trim non-alphanumeric edges.
        text = ' '.join(text.split())
        text = re.sub(r'^[\-\s]+', '', text)
        text = re.sub(r'[\-\s]+$', '', text)
        text = text.strip()
        results.append(text)
    
    return results

def _calculate_split_sizes(n_total: int, valid_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """Calculates the number of samples for train, valid, and test splits."""
    n_test = int(n_total * test_ratio)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_test - n_valid
    if n_train < 0: # Ensure train is not negative if ratios are large
        n_train = 0
        n_test = int(n_total * (test_ratio / (test_ratio + valid_ratio))) if (test_ratio + valid_ratio) > 0 else n_total // 2
        n_valid = n_total - n_test
    return n_train, n_valid, n_test

def run_advanced_filtering(split_ds: Dataset, lang_src: str, lang_tgt: str) -> Dataset:
    """
    Applies all filtering and normalization using datasets.map().
    - Runs normalize_text_batch() for basic cleaning.
    - Runs Length Ratio checks.
    - Runs Noise Ratio checks (alphanumeric density).
    """
    if split_ds is None: return None
    
    # 1. Add original_index column to track provenance after mapping/filtering.
    split_ds = split_ds.add_column("original_index", np.arange(len(split_ds)), new_fingerprint=f"with_index_{len(split_ds)}")
    
    original_count = len(split_ds)

    # 2. Define the comprehensive filter function to be called within .map().
    def _filter_batch_map(batch: Dict[str, List]) -> Dict[str, List]:
        # Detect column structure (nested 'translation' dict vs separated columns).
        if 'translation' in batch:
            # For datasets like opus-100 where translation is a list of dicts.
            src_texts = [item.get(lang_src, '') for item in batch['translation']]
            tgt_texts = [item.get(lang_tgt, '') for item in batch['translation']]
        else:
            # For datasets with direct language columns.
            src_texts = batch.get(lang_src, [])
            tgt_texts = batch.get(lang_tgt, [])
        
        # --- Filter 1: Basic Normalization ---
        # Handles empty strings, length limits (3-120 words), URLs, illegal chars.
        src_norm = normalize_text_batch(src_texts, lang_src)
        tgt_norm = normalize_text_batch(tgt_texts, lang_tgt)

        final_src = []
        final_tgt = []
        final_indices = []

        # Iterate through the batch.
        for i in range(len(src_norm)):
            s = src_norm[i]
            t = tgt_norm[i]
            
            # 2a. Check if basic normalization failed (returned None).
            if s is None or t is None: continue
                
            # --- Filter 2: Length Ratio ---
            s_words = s.split()
            t_words = t.split()
            s_len = len(s_words)
            t_len = len(t_words)
            
            # Safety check (bounds 3-120 already guaranteed by normalize_text_batch).
            if s_len == 0 or t_len == 0: continue 
            
            ratio = s_len / t_len
            # Allow length ratios between 1:5 and 5:1.
            if not (0.2 < ratio < 5.0): continue
            
            # --- Filter 3: Noise Ratio ---
            # Requirement: At least 40% of characters must be letters (isalpha).
            s_alpha = sum(c.isalpha() for c in s)
            s_total = len(s)
            if s_total == 0 or (s_alpha / s_total) < 0.40: continue
                
            t_alpha = sum(c.isalpha() for c in t)
            t_total = len(t)
            if t_total == 0 or (t_alpha / t_total) < 0.40: continue

            # If all filters pass, append to results.
            final_src.append(s)
            final_tgt.append(t)
            final_indices.append(batch['original_index'][i]) # Preserve original index

        # Return dict matching Dataset format.
        return {
            lang_src: final_src,
            lang_tgt: final_tgt,
            "original_index": final_indices
        }

    # 3. Apply filters via .map().
    print(f"\t- Applying advanced filtering (LangID, Ratio, Noise) using .map()...")
    # Using num_proc > 1 enables multiprocessing.
    # remove_columns deletes old 'translation' columns, keeping only new ones.
    filtered_ds = split_ds.map(
        _filter_batch_map,
        batched=("batch_size" in CORPUS_SAMPLING_SETTINGS and CORPUS_SAMPLING_SETTINGS.get("batch_size", 1000) > 0),
        batch_size=CORPUS_SAMPLING_SETTINGS.get("batch_size", 1000),
        remove_columns=split_ds.column_names, 
        num_proc=CORPUS_SAMPLING_SETTINGS.get("num_proc", 8)
    )
    
    discarded_count = original_count - len(filtered_ds)
    if discarded_count > 0:
         print(f"\t- [Warning] Discarded {discarded_count} invalid pairs during advanced filtering.")
    
    if len(filtered_ds) == 0:
        raise RuntimeError(f"\t- [Error] No valid pairs remain after filtering. Returning None.")
        
    print(f"\t- Kept {len(filtered_ds)} valid pairs")
    return filtered_ds

def _get_dataset_cache_dir(dataset_name: str) -> Path:
    """Creates a safe directory name for caching based on the dataset name."""
    safe_name = dataset_name.replace("/", "_").replace(".", "_").replace("-","_").strip()
    return MODELS_CACHE_DIR / safe_name

def filter_dataset_by_similarity(
    dataset: Dataset, 
    lang_src: str, 
    lang_tgt: str, 
    model_name: str, 
    threshold: float
) -> Dataset:
    """
    Filters a Dataset based on semantic similarity using SentenceTransformers.
    This is an expensive operation and should be run *after* deduplication.
    """
    print(f"  Loading similarity model '{model_name}'...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        print(f"[Error] Error loading similarity model: {e}")
        print(f"        Please ensure 'sentence-transformers' is installed: `pip install sentence-transformers`")
        # Fallback: return the dataset unfiltered if model fails
        return dataset
    print(f"  Model loaded on {device}. Starting similarity filtering...")

    good_indices = []
    batch_size = CORPUS_SAMPLING_SETTINGS.get("batch_size", 1000)

    try:
        # Process the dataset in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="  Filtering similarity"):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            de_batch = batch[lang_src]
            en_batch = batch[lang_tgt]
            
            with torch.no_grad():
                # Encode both batches
                # show_progress_bar=False to avoid nested tqdm bars
                de_embs = model.encode(de_batch, convert_to_tensor=True, device=device, show_progress_bar=False)
                en_embs = model.encode(en_batch, convert_to_tensor=True, device=device, show_progress_bar=False)
                
                # Calculate cosine similarity for corresponding (de[i], en[i]) pairs.
                scores = util.pytorch_cos_sim(de_embs, en_embs).diag()
                
                # Get the indices *within this batch* that are above the threshold.
                batch_good_indices = (scores >= threshold).nonzero(as_tuple=True)[0]
                
                # Append the *original dataset indices* to our good list.
                for batch_idx in batch_good_indices.cpu().tolist():
                    good_indices.append(i + batch_idx)

    except Exception as e:
        print(f"[Error] An error occurred during similarity filtering: {e}")
        traceback.print_exc()
        # Fallback: return the dataset unfiltered on error
        return dataset
    
    finally:
        # Clean up model and cache
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    filtered_dataset = dataset.select(good_indices)
    
    print(f"  Similarity filtering complete. Kept {len(filtered_dataset)} / {len(dataset)} pairs.")
    return filtered_dataset

def fetch_and_split_dataset(
    dataset_info: Dict[str, Any],
    lang_src: str,
    lang_tgt: str
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], Optional[Dict[str, List[int]]]]:
    """
    Fetches a dataset, performs manual splitting if necessary, and returns splits.
    Returns:
        Tuple containing train, valid, test Datasets (or None if loading failed),
        and a dictionary with indices used for manual splitting (or None).
    """
    name = dataset_info['name']
    config_name = dataset_info.get('config')
    split_only_from_train = dataset_info.get('split_only_from_train', False)
    split_indices = None # To store indices of manually created splits

    cache_dir = _get_dataset_cache_dir(name)

    print(f"  Fetching dataset: {name} (Config: {config_name}) -> Cache: {cache_dir}")

    try:
        # --- Attempt to load standard splits ---
        ds_splits = {}
        split_names = {"train"}
        # Only load 'validation' and 'test' if not forcing split from train
        if not split_only_from_train:
            print("\t- 'split_only_from_train' is False. Attempting to load existing 'validation' and 'test' splits.")
            split_names.update({"validation", "test"})
        else:
            print("\t- 'split_only_from_train' is True. Will skip loading 'validation' and 'test' and force manual split from 'train'.")
        loaded_successfully = True
        for split in {"train", "validation", "test"}:
            if split not in split_names:
                ds_splits[split] = None # Explicitly set to None if skipped
                continue
            try:
                # Load with streaming=False to get Dataset object, specify cache_dir
                ds_splits[split] = load_dataset(
                    name, config_name, split=split, cache_dir=str(cache_dir),
                    trust_remote_code=dataset_info["trust_remote_code"] # Default True for datasets requiring remote code execution
                )
                print(f"\t- Loaded '{split}' split ({len(ds_splits[split])} samples)")
            except ValueError as e: # Handle case where split doesn't exist
                 print(f"\t- Split '{split}' not found for {name}. Will attempt manual split if needed.")
                 ds_splits[split] = None
            except Exception as e:
                print(f"\t- ⚠️ Error loading '{split}' split for {name}: {e}")
                ds_splits[split] = None
                loaded_successfully = False # Mark dataset loading as failed

        if not loaded_successfully or ds_splits["train"] is None:
            print(f"\t- [Error] Failed to load required 'train' split for {name}. Skipping dataset.")
            raise ValueError("Failed to load required 'train' split.")

        train_ds = ds_splits["train"]
        valid_ds = ds_splits["validation"]
        test_ds = ds_splits["test"]

        # Apply filtering to all loaded splits.
        train_ds = run_advanced_filtering(train_ds, lang_src, lang_tgt)
        valid_ds = run_advanced_filtering(valid_ds, lang_src, lang_tgt)
        test_ds = run_advanced_filtering(test_ds, lang_src, lang_tgt)

        # Ensure we have at least a train dataset
        if train_ds is None or len(train_ds) == 0:
            print(f"\t - [Error] No valid training data after formatting for {name}. Skipping dataset.")
            return None, None, None, None

        return train_ds, valid_ds, test_ds, split_indices

    except Exception as e:
        print(f"\t - [Error] Unexpected error fetching/splitting {name}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None, None, None, None


def write_to_files(dataset: Dataset, file_prefix: Path, lang_src: str, lang_tgt: str):
    """Writes formatted pairs from a Dataset object to .src and .tgt files."""
    if dataset is None or len(dataset) == 0:
        print(f"  - Skipping writing for {file_prefix.name}: No data.")
        return 0

    # Ensure parent directory exists
    file_prefix.parent.mkdir(parents=True, exist_ok=True)

    path_src = str(file_prefix) + f".{lang_src}"
    path_tgt = str(file_prefix) + f".{lang_tgt}"

    count = 0
    with open(path_src, 'w', encoding='utf-8') as f_src, \
         open(path_tgt, 'w', encoding='utf-8') as f_tgt:
        for item in dataset:
            src = item[lang_src]
            tgt = item[lang_tgt]
            # Basic filtering for empty lines (should be minimal after formatting)
            if src and tgt:
                src = str(src).strip(); tgt = str(tgt).strip()
                f_src.write(src + "\n")
                f_tgt.write(tgt + "\n")
                count += 1
    print(f"  -> [Success] Wrote {count} sentence pairs to {file_prefix.name}.({lang_src}/{lang_tgt})")
    return count

def save_indices(indices_dict: Dict[str, Any], filepath: Path):
    """Saves the collected indices to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy arrays to lists for JSON serialization
    serializable_dict = {}
    for key, value in indices_dict.items():
        if isinstance(value, np.ndarray):
            serializable_dict[key] = value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], np.integer):
             serializable_dict[key] = [int(i) for i in value] # Convert numpy integers
        elif isinstance(value, dict): # Handle nested dictionaries (like manual_split_indices)
             serializable_dict[key] = {
                 k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in value.items()
             }
        else:
            serializable_dict[key] = value

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_dict, f, indent=2)
        print(f"  -> [Success] Saved indices data to {filepath}")
    except TypeError as e:
        print(f"  -> [Error] Error saving indices to {filepath}: {e}")
        print("     Attempting to save problematic dictionary for debugging:")
        try:
            debug_path = filepath.parent / (filepath.stem + "_debug_error.json")
            with open(debug_path, 'w', encoding='utf-8') as f_debug:
                 # Try saving again, maybe the error message helps identify the issue
                 json.dump(serializable_dict, f_debug, indent=2, default=lambda o: f"<non-serializable: {type(o).__name__}>")
            print(f"     Saved debug dictionary (with replacements) to {debug_path}")
        except Exception as e_debug:
            print(f"     Failed to save debug dictionary: {e_debug}")
            print(f"     Problematic keys might be: {[k for k, v in serializable_dict.items() if not isinstance(v, (list, dict, str, int, float, bool, type(None)))]}")


def main():
    """Main function to orchestrate data preparation."""
    print("="*60)
    print("Starting Data Collection and Preparation...")
    print(f"Output Directory: {RAW_CORPUS_DIR}")
    print(f"Indices Directory: {INDICES_DIR}")
    print(f"Models Cache Directory: {MODELS_CACHE_DIR}")
    print(f"Target Total Samples: {TARGET_TOTAL_SAMPLES_CORPUS.get("nmt_training", float('inf'))} (NMT), {TARGET_TOTAL_SAMPLES_CORPUS.get("datastore", 5e4)} (Datastore)")
    print("="*60)

    # Dictionary to store manual split indices (if any were performed).
    manual_split_indices: Dict[str, Optional[Dict[str, List[int]]]] = {}
    
    # List to hold all processed dataset chunks before global combination.
    all_datasets_to_combine: List[Dataset] = []
    
    # --- Step 1: Fetching and Cleaning All Datasets ---
    print("\n--- Step 1: Fetching and Cleaning All Datasets ---")
    for ds_info in DATA_LOADER_PARAMS["datasets"]:
        name = ds_info['name']
        print(f"\nProcessing dataset: {name}")
        
        # Load and clean specific dataset (train/valid/test splits).
        train_ds, valid_ds, test_ds, split_idx = fetch_and_split_dataset(ds_info, LANG_SRC, LANG_TGT)
        
        # Store index info if available.
        if split_idx:
            manual_split_indices[name] = split_idx 
        
        # Add valid splits to the combination list, tagging them with 'source_dataset'.
        if train_ds is not None and len(train_ds) > 0:
            print(f"  - Adding {len(train_ds)} samples from '{name}' (train split)")
            all_datasets_to_combine.append(train_ds.add_column("source_dataset", [name] * len(train_ds)))
        if valid_ds is not None and len(valid_ds) > 0:
            print(f"  - Adding {len(valid_ds)} samples from '{name}' (validation split)")
            all_datasets_to_combine.append(valid_ds.add_column("source_dataset", [name] * len(valid_ds)))
        if test_ds is not None and len(test_ds) > 0:
            print(f"  - Adding {len(test_ds)} samples from '{name}' (test split)")
            all_datasets_to_combine.append(test_ds.add_column("source_dataset", [name] * len(test_ds)))

    if not all_datasets_to_combine:
        print("\n[Error] No datasets were successfully processed. Exiting.")
        sys.exit(1)

    # --- Step 2: Global Concatenation and Deduplication ---
    print("\n--- Step 2: Global Concatenation and Deduplication ---")
    print(f"  Concatenating {len(all_datasets_to_combine)} dataset chunks...")
    
    # Concatenate all loaded chunks.
    # Note: 'original_index' is relative to the specific split, 'source_dataset' identifies source.
    all_data = concatenate_datasets(all_datasets_to_combine)
    del all_datasets_to_combine # Free memory
    print(f"  Total samples before deduplication: {len(all_data)}")

    # Strict Deduplication Logic: Track source and target sentences globally.
    seen_src_sentences = set()
    seen_tgt_sentences = set()
    unique_indices = []
    
    # Stores provenance info for unique pairs: List[Tuple(dataset_name, original_index)]
    unique_source_info: List[Tuple[str, int]] = [] 
    
    print(f"  Applying strict monolingual deduplication (src/tgt)...")
    for i in tqdm(range(len(all_data)), desc="  Global Deduplication"):
        item = all_data[i]
        src_text = item[LANG_SRC]
        tgt_text = item[LANG_TGT]
        
        # Keep pair only if NEITHER source nor target text has been seen.
        if src_text not in seen_src_sentences and tgt_text not in seen_tgt_sentences:
            seen_src_sentences.add(src_text)
            seen_tgt_sentences.add(tgt_text)
            unique_indices.append(i)
            unique_source_info.append((item["source_dataset"], item["original_index"]))

    unique_data = all_data.select(unique_indices)
    del all_data # Free memory
    del seen_src_sentences
    del seen_tgt_sentences
    print(f"  Total samples after deduplication: {len(unique_data)}")

    # --- Step 3: Stratified Sampling according to Ratio ---
    print("\n--- Step 3: Stratified Sampling according to ratio ---")

    # 1. Build index map: {dataset_name: [list of indices in unique_data]}
    print("  1. Building index map from unique data sources...")
    source_indices_map: Dict[str, List[int]] = {ds['name']: [] for ds in DATA_LOADER_PARAMS["datasets"]}
    
    for i, (source_name, original_idx) in enumerate(tqdm(unique_source_info, desc="  Mapping sources")):
        if source_name in source_indices_map:
            source_indices_map[source_name].append(i) # 'i' is the index in unique_data

    # 2. Count available unique data per source
    available_counts = {name: len(indices) for name, indices in source_indices_map.items()}
    print("  2. Available unique samples per source:")
    for name, count in available_counts.items():
        print(f"    - {name}: {count}")

    # 3. Calculate target totals based on ratios
    nmt_target_total_samples = TARGET_TOTAL_SAMPLES_CORPUS.get("nmt_training", float('inf'))
    
    nmt_corpus = None
    nmt_corpus_source_info = []
    
    # 4. Calculate bottleneck based on source ratios and availability.
    print(f"  3. Calculating max achievable total based on ratio constraints (Target: {nmt_target_total_samples})...")
    
    achievable_total_based_on_source = []
    for ds_info in DATA_LOADER_PARAMS["datasets"]:
        name = ds_info['name']
        ratio = ds_info.get('ratio', 0.0)
        available = available_counts.get(name, 0)
        
        if ratio > 0:
            # Calculate max total dataset size if this source is the limiting factor.
            max_total_from_this_source = available / ratio
            achievable_total_based_on_source.append(max_total_from_this_source)
            print(f"    - {name} (ratio={ratio:.2f}, avail={available}) -> max total: {int(max_total_from_this_source)}")
        else:
            print(f"    - {name} (ratio=0) -> No constraint.")
    
    if not achievable_total_based_on_source:
            print("  [Error] No datasets with ratio > 0 found. Cannot proceed.")
            sys.exit(1)
            
    # The bottleneck is the minimum of all potential max totals.
    max_achievable_total_due_to_ratio = min(achievable_total_based_on_source)
    
    # Final size is minimum of (User Target, Bottleneck).
    final_nmt_total_size = int(min(nmt_target_total_samples, max_achievable_total_due_to_ratio))
    
    print(f"  4. Max achievable total (ratio-constrained): {int(max_achievable_total_due_to_ratio)}")
    print(f"     Final target NMT total size (min(target, max_achievable)): {final_nmt_total_size}")

    # 5. Perform stratified sampling
    final_indices_to_select = []
    rng = np.random.RandomState(SEED)
    
    print("  5. Drawing samples per source (stratified)...")
    
    # 5a. Calculate exact sample counts per source
    dataset_sample_counts = {}
    total_sampled_so_far = 0
    largest_ratio_ds_name = ""
    max_ratio = -1.0
    
    for ds_info in DATA_LOADER_PARAMS["datasets"]:
        name = ds_info['name']
        ratio = ds_info.get('ratio', 0.0)
        if ratio > max_ratio:
            max_ratio = ratio
            largest_ratio_ds_name = name
        
        n_to_sample = int(round(ratio * final_nmt_total_size))
        dataset_sample_counts[name] = n_to_sample
        total_sampled_so_far += n_to_sample

    # 5b. Adjust for rounding errors
    diff = final_nmt_total_size - total_sampled_so_far
    if diff != 0 and largest_ratio_ds_name in dataset_sample_counts:
        print(f"     Adjusting for rounding error: {diff} samples added/removed from '{largest_ratio_ds_name}'.")
        dataset_sample_counts[largest_ratio_ds_name] += diff
    elif diff != 0:
            print(f"     Warning: Could not adjust rounding error (diff={diff}, largest_ds='{largest_ratio_ds_name}').")

    # 5c. Execute random sampling for each source
    for name, n_to_sample in dataset_sample_counts.items():
        if n_to_sample <= 0:
            ds_ratio = next((ds.get('ratio', 0.0) for ds in DATA_LOADER_PARAMS["datasets"] if ds['name'] == name), 0.0)
            if ds_ratio > 0:
                print(f"    - {name}: 0 samples (rounded down)")
            continue
            
        available_indices_for_source = source_indices_map.get(name, [])
        
        if n_to_sample > len(available_indices_for_source):
            print(f"    - ⚠️ Warning: {name} needs {n_to_sample} samples, but only {len(available_indices_for_source)} are available. Taking all.")
            n_to_sample = len(available_indices_for_source)
        
        print(f"    - {name}: Sampling {n_to_sample} / {len(available_indices_for_source)} available.")
        
        # Use rng.choice for sampling without replacement.
        sampled_indices_from_source_pool = rng.choice(available_indices_for_source, size=n_to_sample, replace=False).tolist()
        final_indices_to_select.extend(sampled_indices_from_source_pool)
        
    print(f"  Total samples selected (stratified): {len(final_indices_to_select)}")

    # 6. Assemble and Shuffle
    print("  6. Assembling and shuffling final NMT corpus...")
    
    # We must shuffle the final index list because it is currently grouped by source.
    # Create tuples: (index_in_unique_data, provenance_info)
    combined_list_to_shuffle = [
        (idx, unique_source_info[idx]) for idx in final_indices_to_select
    ]
    
    print(f"  Shuffling {len(combined_list_to_shuffle)} final selected pairs...")
    random.shuffle(combined_list_to_shuffle) # Uses seeded random
    
    # Unpack shuffled list.
    final_shuffled_indices = [item[0] for item in combined_list_to_shuffle]
    nmt_corpus_source_info = [item[1] for item in combined_list_to_shuffle]
    
    # Select from unique_data using the shuffled indices.
    nmt_corpus = unique_data.select(final_shuffled_indices)
    
    del unique_data # Free memory
    del unique_source_info

    # --- Step 4: Global Splitting (NMT Corpus) ---
    print("\n--- Step 4: Global Splitting (NMT Corpus) ---")
    
    if nmt_corpus is None:
         print("  [Error] NMT corpus was not created. Exiting.")
         sys.exit(1)

    # The NMT corpus is already shuffled.
    n_total = len(nmt_corpus)
    n_train, n_valid, n_test = _calculate_split_sizes(n_total, MANUAL_VALID_RATIO, MANUAL_TEST_RATIO)
    
    print(f"  Splitting NMT corpus: Train={n_train}, Valid={n_valid}, Test={n_test}")

    # Slice directly for splits.
    combined_datasets_nmt: Dict[str, Optional[Dataset]] = {}
    combined_datasets_nmt["train"] = nmt_corpus.select(range(n_train))
    combined_datasets_nmt["validation"] = nmt_corpus.select(range(n_train, n_train + n_valid))
    combined_datasets_nmt["test"] = nmt_corpus.select(range(n_train + n_valid, n_total))
    
    # Map provenance info to splits for index file saving.
    shuffled_final_indices_nmt: Dict[str, List[Tuple[str, int]]] = {}
    shuffled_final_indices_nmt["train"] = nmt_corpus_source_info[:n_train]
    shuffled_final_indices_nmt["validation"] = nmt_corpus_source_info[n_train : n_train + n_valid]
    shuffled_final_indices_nmt["test"] = nmt_corpus_source_info[n_train + n_valid :]

    # Clean up internal columns.
    for split_name in ["train", "validation", "test"]:
        if combined_datasets_nmt[split_name]:
            if "original_index" in combined_datasets_nmt[split_name].column_names:
                combined_datasets_nmt[split_name] = combined_datasets_nmt[split_name].remove_columns(["original_index"])
            if "source_dataset" in combined_datasets_nmt[split_name].column_names:
                combined_datasets_nmt[split_name] = combined_datasets_nmt[split_name].remove_columns(["source_dataset"])
    
    print("  Global split completed.")
    
    # --- Step 5: Creating Datastore Subsets from NMT Splits ---
    print("\n--- Step 5: Creating Datastore Subsets from NMT Splits ---")
    datastore_target_total_samples = TARGET_TOTAL_SAMPLES_CORPUS.get("datastore", 5e4)
    combined_datasets_ds: Dict[str, Dataset] = {}
    # Stores indices relative to the NMT splits.
    datastore_subset_indices: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
    # Stores provenance info for DS subsets.
    shuffled_final_indices_ds: Dict[str, List[Tuple[str, int]]] = {"train": [], "validation": [], "test": []}

    n_total_nmt_actual = sum(len(ds) for ds in combined_datasets_nmt.values() if ds is not None)

    if n_total_nmt_actual == 0:
        print("   - [Error] Cannot create datastore subsets, no NMT data available.")
    else:
        # Calculate NMT split sizes.
        n_train_nmt_actual = len(combined_datasets_nmt["train"]) if combined_datasets_nmt["train"] else 0
        n_valid_nmt_actual = len(combined_datasets_nmt["validation"]) if combined_datasets_nmt["validation"] else 0
        n_test_nmt_actual = len(combined_datasets_nmt["test"]) if combined_datasets_nmt["test"] else 0

        # Calculate DS subset sizes proportional to NMT split sizes.
        n_target_train_ds = round(n_train_nmt_actual / n_total_nmt_actual * datastore_target_total_samples) if n_total_nmt_actual > 0 else 0
        n_target_valid_ds = round(n_valid_nmt_actual / n_total_nmt_actual * datastore_target_total_samples) if n_total_nmt_actual > 0 else 0
        n_target_test_ds = round(n_test_nmt_actual / n_total_nmt_actual * datastore_target_total_samples) if n_total_nmt_actual > 0 else 0

        # Adjust for rounding errors on the train set.
        current_ds_total = n_target_train_ds + n_target_valid_ds + n_target_test_ds
        diff = datastore_target_total_samples - current_ds_total
        n_target_train_ds += diff

        print(f"  Target DS sizes: Train={n_target_train_ds}, Valid={n_target_valid_ds}, Test={n_target_test_ds} (Total={datastore_target_total_samples})")

        rng_ds = np.random.RandomState(SEED)

        for split_name, n_target_ds in [("train", n_target_train_ds), ("validation", n_target_valid_ds), ("test", n_target_test_ds)]:
            nmt_split_ds = combined_datasets_nmt[split_name]
            if nmt_split_ds and n_target_ds > 0:
                n_available = len(nmt_split_ds)
                n_select = min(n_target_ds, n_available)
                if n_select < n_target_ds:
                    print(f"    - Warning: Requested {n_target_ds} for DS {split_name}, but only {n_available} available in NMT {split_name}. Selecting {n_select}.")

                # Sample indices from the NMT split.
                indices_in_nmt_split = rng_ds.choice(n_available, size=n_select, replace=False).tolist()
                datastore_subset_indices[split_name] = sorted(indices_in_nmt_split) 

                # 1. Create subset using selected indices (already randomized by choice).
                combined_datasets_ds[split_name] = nmt_split_ds.select(indices_in_nmt_split)
                print(f"    - Created DS '{split_name}' subset with {len(combined_datasets_ds[split_name])} samples.")

                # --- Create final shuffled indices for DS subset ---
                # 2. Since `combined_datasets_ds` corresponds exactly to `indices_in_nmt_split` without further shuffling,
                # we just map back to provenance info.
                shuffled_order_indices_ds_subset = list(range(len(combined_datasets_ds[split_name])))

                shuffled_final_indices_ds[split_name] = []
                for idx_in_shuffled_subset in shuffled_order_indices_ds_subset:
                    if isinstance(idx_in_shuffled_subset, (int, np.integer)):
                        idx_int_subset = int(idx_in_shuffled_subset)
                        if 0 <= idx_int_subset < len(indices_in_nmt_split):
                                index_in_nmt = indices_in_nmt_split[idx_int_subset] 
                                if 0 <= index_in_nmt < len(shuffled_final_indices_nmt[split_name]):
                                    original_source_info = shuffled_final_indices_nmt[split_name][index_in_nmt]
                                    shuffled_final_indices_ds[split_name].append(original_source_info)
                                else:
                                    print(f"[WARNING] NMT Index {index_in_nmt} out of bounds for shuffled_final_indices_nmt['{split_name}'].")
                                    shuffled_final_indices_ds[split_name].append(("error", -2))
                        else:
                            print(f"[WARNING] Subset Index {idx_int_subset} out of bounds for indices_in_nmt_split['{split_name}'].")
                            shuffled_final_indices_ds[split_name].append(("error", -3))
                    else:
                        print(f"[WARNING] Invalid DS subset index type {type(idx_in_shuffled_subset)} for value {idx_in_shuffled_subset}. Skipping.")
            else:
                print(f"    - No DS subset created for '{split_name}' (target size {n_target_ds} or NMT split is None).")

    # --- Step 6: Write Output Files (NMT and DS) ---
    print("\n--- Step 4: Writing Output Files ---")
    split_to_filename_nmt = {"train": "train", "validation": "valid", "test": "test"}
    split_to_filename_ds = {"train": "train_ds", "validation": "valid_ds", "test": "test_ds"}
    final_counts_nmt = {}
    final_counts_ds = {}

    # Write NMT corpus files.
    print("  Writing NMT corpus files...")
    for split_name in ["train", "validation", "test"]:
        ds_nmt = combined_datasets_nmt[split_name]
        if ds_nmt:
            file_prefix_nmt = RAW_CORPUS_DIR / split_to_filename_nmt[split_name]
            final_counts_nmt[split_name] = write_to_files(ds_nmt, file_prefix_nmt, LANG_SRC, LANG_TGT)
        else:
            final_counts_nmt[split_name] = 0

    # Write DS subset files.
    print("  Writing Datastore subset files...")
    for split_name in ["train", "validation", "test"]:
        ds_ds = combined_datasets_ds[split_name]
        if ds_ds:
            file_prefix_ds = RAW_CORPUS_DIR / split_to_filename_ds[split_name]
            final_counts_ds[split_name] = write_to_files(ds_ds, file_prefix_ds, LANG_SRC, LANG_TGT)
        else:
            final_counts_ds[split_name] = 0

    print("\nFinal NMT Split Sizes:")
    print(f"  Train: {final_counts_nmt.get('train', 0)}")
    print(f"  Valid: {final_counts_nmt.get('validation', 0)}")
    print(f"  Test:  {final_counts_nmt.get('test', 0)}")
    print("\nFinal Datastore Subset Split Sizes:")
    print(f"  Train: {final_counts_ds.get('train', 0)}")
    print(f"  Valid: {final_counts_ds.get('validation', 0)}")
    print(f"  Test:  {final_counts_ds.get('test', 0)}")

    # --- Step 7: Saving Index Files ---
    print("\n--- Step 5: Saving Index Files ---")
    # Save provenance/mapping files for traceability.
    save_indices(manual_split_indices, INDICES_DIR / "manual_split_indices.json")
    save_indices(shuffled_final_indices_nmt, INDICES_DIR / "final_shuffled_indices_nmt.json")
    save_indices(datastore_subset_indices, INDICES_DIR / "datastore_subset_indices.json")
    save_indices(shuffled_final_indices_ds, INDICES_DIR / "final_shuffled_indices_ds.json")

    print("\n" + "="*60)
    print("[Success] Data preparation script finished successfully.")
    print(f"   Raw text files (NMT and DS subsets) saved in: {RAW_CORPUS_DIR}")
    print(f"   Index mapping files saved in: {INDICES_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
