#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/paec_mt_validation.py

[PAEC Framework - Phase 4: Validation & Benchmarking]

This script serves as the master entry point for evaluating and validating the
PAEC kNN-MT system against standard baselines (Vanilla kNN-MT, Adaptive kNN-MT,
and Pure NMT).

Key Responsibilities:
1.  Orchestration: It manages the evaluation of multiple models on a specified
    test dataset.
2.  PAEC Evaluation: It instantiates the PAEC system (using either the Online
    Dynamics Planner or the Offline Policy Network) and runs it in 'Evaluation Mode',
    where real-time resource usage is monitored.
3.  Baseline Evaluation: It wraps standard Fairseq/kNN-Box models but injects a
    custom inference loop. This allows it to capture detailed step-by-step state
    trajectories (Error, Pressure, Context) for baselines, enabling a fair,
    apples-to-apples comparison of stability metrics (Lyapunov energy) even
    though baselines do not control for them.
4.  Analysis: It aggregates metrics (BLEU, Latency, Memory, V(Error)) and
    generates the final comparative reports and visualization plots (including
    the critical 'Money Plot' showing stability over time).
"""

import os, sys, json, time, logging, argparse, torch, subprocess, re, psutil, warnings, traceback, copy, math, gc
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set, cast
from dataclasses import dataclass, asdict, field
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sacrebleu import corpus_bleu, corpus_chrf, corpus_ter
import matplotlib.pyplot as plt
from datasets import load_dataset
import sentencepiece as spm
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque, defaultdict
from scipy.optimize import linear_sum_assignment

# Allow argparse Namespace serialization for checkpoint loading
torch.serialization.add_safe_globals([argparse.Namespace])

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# PROJECT IMPORTS
# =====================================================================

# Import Fairseq and knn-box components for running baseline models
from fairseq.models import FairseqEncoder, FairseqDecoder
from knnbox.models import VanillaKNNMT, AdaptiveKNNMT
from knnbox.models.vanilla_knn_mt import VanillaKNNMTDecoder
from knnbox.models.adaptive_knn_mt import AdaptiveKNNMTDecoder
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders

# Add the project root 'src' directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core PAEC modules
# - InvertableColumnTransformer: For inverse scaling of states during analysis
# - config: Global project configuration (paths, params)
# - DataGenerationPipeline: The unified pipeline used for PAEC inference
# - kNNMTSystem: The core translation engine
# - RealtimeResourceMonitor: Monitors actual hardware usage during validation
# - RealDatasetLoader: Loads normalized test data
# - Structs: Core data classes defining the PAEC state space (E, P, H) and Actions
from t_train_Transformer import InvertableColumnTransformer
from src import config as paec_config
from src.pipeline import DataGenerationPipeline
from src.system.knn_mt import kNNMTSystem
from src.simulation.resource_monitor import RealtimeResourceMonitor
from src.data_processing.loader import RealDatasetLoader
from src.core.structs import DecodingStrategy, ErrorStateVector, GenerativeContextVector, ResourcePressureVector, SystemState, Action

# Configure global logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# DATA STRUCTURES & CONFIGURATION
# =====================================================================

@dataclass
class TranslationResult:
    """
    Container for the final output of a single sentence translation.
    Holds both the textual result and the aggregate performance metrics.
    """
    model_name: str
    source: str
    reference: str
    hypothesis: str
    latency_ms: float
    memory_mb: float
    knn_skip_rate: float
    # Detailed trajectory is usually omitted here to save memory in the final summary list
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class StepResult:
    """
    Container for granular, step-by-step telemetry data.
    This captures the exact state of the system (Error, Pressure, Context)
    at every decoding step t. This data powers the stability analysis plots.
    """
    model_name: str
    sample_id: int
    step: int
    error_semantic: float
    error_coverage: float
    error_fluency_surprisal: float
    error_fluency_repetition: float
    error_norm: float
    v_error: float # The calculated Lyapunov Energy V(E_t)

@dataclass
class AggregatedMetrics:
    """
    Container for corpus-level statistics derived after evaluating the full test set.
    Includes both standard quality metrics (BLEU) and control-theoretic metrics (V_Error).
    """
    Model: str
    BLEU: float
    CHRF: float
    TER: float
    Avg_Latency_ms: float
    Avg_Memory_MB: float
    Avg_kNN_Skip_Rate: float
    Avg_Peak_V_Error: float
    Avg_Final_V_Error: float
    Avg_Accumulated_V_Error: float
    Avg_Steps_to_Converge: float

@dataclass
class ExperimentConfig:
    """
    A unified configuration object holding all parameters parsed from CLI arguments.
    Passed down to evaluators and systems to configure behavior.
    """
    dataset_name: str
    source_lang: str
    target_lang: str
    test_size: int
    beam_size: int
    length_penalty: float
    dynamics_model_dir: str
    offline_policy_model_dir: str
    fairseq_checkpoint: str
    fairseq_data_bin: str
    bpe_type: str
    bpe_codes: Optional[str]
    sentencepiece_model: Optional[str]
    datastore_path: str
    adaptive_combiner_path: str
    output_dir: str
    cache_dir: str
    device: str
    seed: int
    vanilla_knn_k: int
    vanilla_knn_lambda: float
    vanilla_knn_temperature: float
    optimize_simulator_params: bool
    paec_test_mode: int  # 0=Both Online/Offline, 1=Online Only, 2=Offline Only, -1=Skip
    is_debug: bool = False

# =====================================================================
# PAEC TRANSLATION SYSTEM WRAPPER
# =====================================================================

class PAECTranslationSystem:
    """
    A wrapper around the core DataGenerationPipeline that configures it strictly
    for evaluation mode.
    
    In evaluation mode:
    1. It uses a RealtimeResourceMonitor instead of a simulator.
    2. It uses the actual trained Policy Network (or Dynamics Model planner) for actions.
    3. It performs translation on real hardware to measure true latency/memory.
    """
    def __init__(self, exp_config: ExperimentConfig):
        self.config = exp_config
        # Initialize the pipeline in evaluation mode.
        # This triggers the loading of the offline policy model (if provided) or
        # the dynamics model for online optimization.
        self.pipeline = DataGenerationPipeline(
            decoding_strategy=DecodingStrategy.BEAM_SEARCH,
            beam_size=self.config.beam_size,
            length_penalty=self.config.length_penalty,
            use_datastore=True,
            datastore_path=self.config.datastore_path,
            evaluation_mode=True, # Critical flag: Enables real-time monitoring and policy inference
            dynamics_model_dir=self.config.dynamics_model_dir,
            offline_policy_model_dir=self.config.offline_policy_model_dir,
            optimize_simulator_params=self.config.optimize_simulator_params,
            is_debug=self.config.is_debug
        )
        logger.info("Initialized PAECTranslationSystem with a dedicated evaluation pipeline.")

    def translate(self, source: str, reference: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Executes translation for a single source sentence using the PAEC system.
        
        Returns:
            Tuple containing:
            - Result dictionary (hypothesis, latency, memory, skip_rate)
            - Trajectory list (step-by-step state logs for the best beam)
        """
        # Delegate to the pipeline's specialized evaluation method which handles
        # the beam search loop, state tracking, and policy querying.
        result, trajectory = self.pipeline.run_paec_evaluation_for_sentence(
            sample={'source_text': source, 'target_text': reference}
        )
        
        # Sanity check for over-fitting or data leakage
        if result['hypothesis'] == reference:
            logger.warning(f"PAEC kNN-MT hypothesis matches reference for source: {source}")
        
        return result, trajectory
    
    def teardown(self):
        """Releases GPU memory and FAISS resources."""
        self.pipeline.teardown()
        if hasattr(self, "pipeline") and getattr(self, "pipeline"): del self.pipeline
        if hasattr(self, "config") and getattr(self, "config"): del self.config
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# =====================================================================
# BASELINE SYSTEMS WRAPPER (FOR VANILLA & ADAPTIVE)
# =====================================================================

@dataclass
class BaselineBeamHypothesis:
    """
    A lightweight hypothesis structure used for the manual beam search implementation
    within the baseline wrapper. Allows tracking state history alongside tokens.
    """
    tokens: torch.Tensor  # Sequence of token IDs
    log_prob: float       # Cumulative log probability
    incremental_state: Dict[str, Any] # Fairseq incremental state for caching
    hidden_state: Optional[torch.Tensor] = None # Last decoder hidden state
    covered_entities: Set[str] = field(default_factory=set) # Tracked for Error State calculation

    def score(self, length_penalty: float) -> float:
        """Calculates length-normalized score."""
        length = len(self.tokens)
        if length == 0: return -float('inf')
        return self.log_prob / (length ** length_penalty)

class KNNBoxBaselineSystem:
    """
    A wrapper for Vanilla and Adaptive kNN-MT baselines.
    
    CRITICAL: Unlike standard inference scripts, this class RE-IMPLEMENTS the
    beam search loop manually. 
    
    Why?
    Standard Fairseq/kNN-Box inference is a black box that outputs text. To compare
    stability fairly, we need to calculate the PAEC State Vector (Error, Pressure, Context)
    at *every step* of the baseline's generation, even though the baseline doesn't
    use these states for control. This allows us to visualize the baseline's "trajectory"
    in the same phase space as PAEC.
    """
    def __init__(self, model_type: str, exp_config: ExperimentConfig):
        self.model_type = model_type
        # Select appropriate index based on model type (usually 'ivf_pq' for baselines)
        self.knn_ds_selection = paec_config.KNN_BASELINE_INDEX.get(self.model_type, paec_config.DEFAULT_INDEX)
        self.config = exp_config
        self.device = torch.device(exp_config.device)
        self.is_debug = self.config.is_debug
        
        # --- 1. Load Fairseq Model Components ---
        logger.info(f"Loading Fairseq model and task from checkpoint: {self.config.fairseq_checkpoint}")
        
        # Load the base Transformer model first
        models, _saved_args, task = checkpoint_utils.load_model_ensemble_and_task(
            [self.config.fairseq_checkpoint],
            arg_overrides=self._get_arg_overrides() 
        )
        base_model = models[0]
        self.args = _saved_args
        self.task = task
        
        # Inject overridden arguments into the args namespace manually
        overrides = self._get_arg_overrides()
        for k, v in overrides.items():
            setattr(self.args, k, v)

        # Force inference mode for kNN operations
        if self.model_type != 'pure_nmt':
            if not hasattr(self.args, 'knn_mode') or self.args.knn_mode != 'inference':
                logger.info("Forcing knn_mode to 'inference' for initialization...")
                self.args.knn_mode = 'inference'

        # --- 2. Wrap Model with kNN Logic ---
        if self.model_type == 'vanilla':
            # For Vanilla kNN-MT, wrap the base decoder with VanillaKNNMTDecoder
            logger.info("🚀 Manually wrapping base NMT model with VanillaKNNMT...")
            
            knn_decoder = VanillaKNNMTDecoder(
                self.args, 
                base_model.decoder.dictionary, 
                base_model.decoder.embed_tokens
            )
            
            # Transfer weights from the pre-trained NMT model to the wrapper
            logger.info("📥 Transplanting weights from base decoder to Vanilla wrapper...")
            knn_decoder.load_state_dict(base_model.decoder.state_dict(), strict=False)
            
            self.model = VanillaKNNMT(self.args, base_model.encoder, knn_decoder)  # type: ignore

        elif self.model_type == 'adaptive':
            # For Adaptive kNN-MT, handle potential auto-wrapping by Fairseq
            if isinstance(base_model, AdaptiveKNNMT):
                logger.info("✅ Detect: Fairseq has already initialized AdaptiveKNNMT successfully.")
                logger.info("🚀 Using the auto-loaded AdaptiveKNNMT model (skipping manual re-wrapping).")
                self.model = base_model
                
                # Verify the Combiner network (Meta-k) is present
                if hasattr(self.model.decoder, 'combiner'):
                    logger.info(f"   Combiner Status: Loaded (Type: {type(self.model.decoder.combiner).__name__})") # type: ignore
                else:
                    logger.error("   [FATAL] AdaptiveKNNMT loaded but 'combiner' is missing!")
                    exit(1)
            else:
                # Manual wrapping fallback if checkpoint architecture didn't trigger auto-wrap
                logger.warning("⚠️ Detect: Base model is NOT AdaptiveKNNMT. Attempting manual wrap (Risk of weight loss).")
                
                knn_decoder = AdaptiveKNNMTDecoder(
                    self.args, 
                    base_model.decoder.dictionary, 
                    base_model.decoder.embed_tokens
                )
                logger.info("📥 Transplanting weights from base decoder to Adaptive wrapper...")
                knn_decoder.load_state_dict(base_model.decoder.state_dict(), strict=False)
                self.model = AdaptiveKNNMT(self.args, base_model.encoder, knn_decoder)  # type: ignore
            
        else:
            # Pure NMT uses the base model directly
            logger.warning(f"Unknown model type '{self.model_type}', using base transformer.")
            self.model = base_model
            
        # Ensure dictionaries are loaded for tokenization/detokenization
        if not self.task or not hasattr(self.task, 'source_dictionary') or not hasattr(self.task, 'target_dictionary'):
            raise ValueError("Failed to load Fairseq task or dictionaries. Check data bin path in config.")
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Move to GPU/CPU
        self.model.to(self.device)
        self.model.eval()

        # --- 3. Initialize BPE/SentencePiece ---
        self.bpe_type = self.config.bpe_type.lower()

        if self.bpe_type == "fastbpe":
            self.bpe = encoders.build_bpe(self.args)
            if self.bpe is None:
                raise RuntimeError(f"Failed to build fastBPE encoder from args.")
            if self.is_debug: logger.info("KNNBoxBaselineSystem: Initialized fastBPE handler.")
        elif self.bpe_type in ["sentencepiece", "sentence_piece"]:
            self.spm_processor = spm.SentencePieceProcessor()
            self.spm_processor.load(self.config.sentencepiece_model)    # type:ignore
            if not self.spm_processor:
                raise RuntimeError(f"Failed to load SentencePiece model from {self.config.sentencepiece_model}")
            self.bpe = None
            if self.is_debug: logger.info(f"KNNBoxBaselineSystem: Initialized SentencePiece handler from {self.config.sentencepiece_model}.")
        else:
            raise ValueError(f"Unsupported BPE type for baselines: {self.bpe_type}")
        
        # --- 4. Initialize Auxiliary Models for State Calculation ---
        # These models (SBERT, GPT-2, NER) are needed to calculate the PAEC state vector
        # (Semantic Error, Fluency, etc.) during the baseline's execution.
        if self.is_debug: logger.info(f"[{model_type}] Loading auxiliary models for state calculation...")
        self.sentence_encoder = SentenceTransformer(paec_config.MODEL_NAMES["sentence_encoder"], device=str(self.device or "cpu"))
        self.confidence_history = deque(maxlen=5)
        self.query_history = deque(maxlen=5)
        
        fluency_model_name = paec_config.MODEL_NAMES.get("fluency_scorer", "distilgpt2")
        self.lm_tokenizer = AutoTokenizer.from_pretrained(fluency_model_name)
        self.lm_model = AutoModelForCausalLM.from_pretrained(fluency_model_name, device_map=self.device)
        self.lm_model.eval()
        self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        self.fluency_loss_history = deque(maxlen=10)
        
        try:
            self.ner_de = spacy.load('de_core_news_lg')
            self.ner_en = spacy.load('en_core_web_lg')
        except (ImportError, OSError):
            logger.error("[ERROR] spaCy models not found. Please run: python -m spacy download de_core_news_lg en_core_web_lg")
            sys.exit(1)
            
        self.vocab_size = len(self.task.target_dictionary) # type: ignore

        logger.info(f"Successfully initialized {model_type} kNN-MT baseline system.")
        logger.info(f"Running on device: {self.device}")
    
    def _get_arg_overrides(self) -> Dict[str, Any]:
        """
        Constructs a dictionary of argument overrides to configure the model
        for inference (e.g., beam size, datastore path).
        """
        overrides = {
            'data': self.config.fairseq_data_bin,
            'beam': self.config.beam_size,
            'lenpen': self.config.length_penalty,
            'bpe': self.config.bpe_type,
            **({'bpe_codes': self.config.bpe_codes} if self.config.bpe_type == 'fastbpe' else {}),
            **({'sentencepiece_model': self.config.sentencepiece_model} if self.config.bpe_type in ['sentencepiece', 'sentence_piece'] else {}),
        }

        # Configure kNN-specific arguments if applicable
        if self.model_type != 'pure_nmt':
            overrides.update({
                'knn_mode': 'inference',
                'knn_datastore_path': str(Path(self.config.datastore_path) / self.knn_ds_selection),
                'knn_k': self.config.vanilla_knn_k,
                'knn_lambda': self.config.vanilla_knn_lambda,
                'knn_temperature': self.config.vanilla_knn_temperature,
            })

        if self.model_type == 'adaptive':
            overrides.update({
                'arch': f'adaptive_knn_mt@{paec_config.BUILD_PIPELINE_SETTINGS["arch"]}',
                'knn_combiner_path': self.config.adaptive_combiner_path,
                'knn_max_k': self.config.vanilla_knn_k
            })
        elif self.model_type == 'vanilla':
             overrides.update({
                'arch': f'vanilla_knn_mt@{paec_config.BUILD_PIPELINE_SETTINGS["arch"]}'
            })

        return overrides

    def translate(self, source_text: str, reference_text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Translates a single source sentence using a manual beam search loop.
        
        This loop mimics the model's standard inference behavior but injects
        instrumentation to calculate and log the PAEC State Vector at every step.
        This enables the "Money Plot" comparison where we visualize the evolution
        of Error/Pressure/Context for baselines vs PAEC.
        
        Args:
            source_text: The source sentence.
            reference_text: The reference translation (used for error calculation).
            
        Returns:
            Tuple of (Result Dictionary, Trajectory List).
        """
        with torch.no_grad():
            # --- 1. Pre-computation (Tokenization & Encoding) ---
            if self.bpe_type in ["sentencepiece", "sentence_piece"]:
                spm_encoded_tokens = self.spm_processor.encode(source_text, out_type=str)   # type: ignore
                bpe_source = " ".join(spm_encoded_tokens)
            elif self.bpe_type == "fastbpe":
                if self.bpe is None: raise RuntimeError("BPE tokenizer is not initialized for fastBPE.")
                bpe_source = self.bpe.encode(source_text)
            
            source_tokens = self.src_dict.encode_line(bpe_source, add_if_not_exist=False, append_eos=True).long().unsqueeze(0).to(self.device)
            
            # Precompute entities for coverage error calculation
            precomputed_source_entities = [ent.text for ent in self.ner_de(source_text).ents]
            source_entity_spans = self._identify_source_entity_token_spans(bpe_source, precomputed_source_entities)
            
            # Run encoder
            encoder_module = cast(FairseqEncoder, self.model.encoder)
            encoder_out = encoder_module(
                source_tokens, 
                src_lengths=torch.tensor([source_tokens.size(1)], device=self.device)
            )
            base_decoder = cast(FairseqDecoder, self.model.decoder)
            
            # Precompute source embedding for semantic error calculation
            precomputed_source_emb = self.sentence_encoder.encode([source_text], convert_to_tensor=True, show_progress_bar=False).to(self.device)

            # --- 2. Initialize Beam Search ---
            beam_size = self.config.beam_size
            length_penalty = self.config.length_penalty
            max_length = 64 # Hardcap for safety
            
            initial_token = self.tgt_dict.eos()
            self.confidence_history.clear()
            self.query_history.clear()

            active_beams: List[BaselineBeamHypothesis] = [
                BaselineBeamHypothesis(
                    tokens=torch.tensor([initial_token], device=self.device, dtype=torch.long),
                    log_prob=0.0, hidden_state=None, incremental_state={}
                )
            ]
            completed_hypotheses: List[BaselineBeamHypothesis] = []
            
            # Trajectory log for the best beam
            trajectory: List[Dict[str, Any]] = []

            total_steps = 0
            skip_steps = 0 # Not applicable for baselines as they always (conceptually) run their logic

            # --- 3. Decoding Loop ---
            for step in range(max_length):
                if not active_beams: break

                step_candidates: List[BaselineBeamHypothesis] = []

                for beam in active_beams:
                    decoder_input_tokens = beam.tokens.unsqueeze(0)

                    try:
                        # --- Forward Pass ---
                        # Calls the model's forward(). For kNN models, this internally handles
                        # datastore retrieval and probability interpolation.
                        decoder_out, extra = base_decoder(
                            prev_output_tokens=decoder_input_tokens,
                            encoder_out=encoder_out,
                            features_only=False,
                            return_all_hiddens=True,
                            incremental_state=beam.incremental_state
                        )
                        if 'inner_states' not in extra or not extra['inner_states']:
                            raise RuntimeError("No inner_states in decoder output")
                        
                        # --- Get Final Probabilities ---
                        # Use the specific get_normalized_probs method which handles
                        # model-specific logic (e.g., Adaptive's combiner).
                        net_output = (decoder_out, extra)
                        final_log_probs = base_decoder.get_normalized_probs(
                            net_output, 
                            log_probs=True, 
                            sample=None
                        )
                        
                        # Handle output shape variations
                        if final_log_probs.dim() == 3:
                            nmt_log_probs = final_log_probs[:, -1, :].squeeze(0)
                        else:
                            nmt_log_probs = final_log_probs.squeeze(0)
                        
                    except Exception as e:
                        logger.error(f"[{self.model_type}] Decoder forward pass failed at step {step}: {e}")
                        continue

                    # Extract internal states for metric calculation
                    cross_attention_weights = extra.get('attn', [None])[0]
                    current_step_attention = cross_attention_weights[0, -1, :] if cross_attention_weights is not None else None
                    current_hidden_state = extra['inner_states'][-1][:, -1:, :] # Shape [1, 1, H]
                    
                    # --- Compute PAEC State Vectors (Instrumentation) ---
                    # Reconstruct current partial translation
                    prefix_tokens = beam.tokens[1:].tolist() if len(beam.tokens) > 1 else []
                    generated_words = self._tokens_to_words(prefix_tokens)
                    query_embedding = self.project_to_query_embedding(current_hidden_state)
                    attn_tensor = current_step_attention.to(self.device) if current_step_attention is not None else None

                    # Calculate Context State H_t
                    context_state = self.compute_context_state(
                        decoder_out=decoder_out, query_embedding=query_embedding, attention_weights=torch.Tensor(attn_tensor),
                        encoder_hidden_states=encoder_out.encoder_out.permute(1, 0, 2),
                        uncovered_entities=set(precomputed_source_entities) - beam.covered_entities,
                        source_entity_spans=source_entity_spans
                    )
                    # Calculate Error State E_t
                    error_state, new_covered_entities = self.compute_error_state(
                        source_text=source_text, generated_prefix_words=generated_words, generated_tokens=prefix_tokens,
                        source_emb=precomputed_source_emb, source_entities=precomputed_source_entities, decoder_out=decoder_out
                    )
                    
                    # Create a dummy Pressure State P_t (Baselines don't feel pressure)
                    pressure_state = ResourcePressureVector(0.1, 0.1, 0.1)
                    
                    # Determine effective Action A_t
                    if self.model_type == 'pure_nmt':
                        baseline_action = Action(k=0, index_type='none', lambda_weight=0.0)
                    else:
                        baseline_action = Action(
                            k=self.config.vanilla_knn_k,
                            index_type=self.knn_ds_selection,
                            lambda_weight=self.config.vanilla_knn_lambda
                        )

                    # --- Log Step Data (for the best beam only) ---
                    if beam == active_beams[0]: 
                        step_dict = {
                            'step': step,
                            'generated_tokens': generated_words,
                            'error_state': error_state,
                            'pressure_state': pressure_state,
                            'context_state': context_state,
                            'action': baseline_action, 
                            'decoder_hidden_state': current_hidden_state.detach().squeeze(1).cpu().numpy(),
                        }
                        trajectory.append(step_dict)

                    # --- Expand Candidates ---
                    top_log_probs, top_indices = torch.topk(nmt_log_probs, k=beam_size)

                    for k_cand in range(beam_size):
                        token_id = top_indices[k_cand].item()
                        log_prob = top_log_probs[k_cand].item()
                        
                        candidate = BaselineBeamHypothesis(
                            tokens=torch.cat([beam.tokens, torch.tensor([token_id], device=self.device)]),
                            log_prob=beam.log_prob + log_prob,
                            hidden_state=current_hidden_state.detach(),
                            incremental_state=copy.deepcopy(beam.incremental_state)
                        )
                        candidate.covered_entities = beam.covered_entities.union(new_covered_entities)
                        step_candidates.append(candidate)

                # --- Prune Beams ---
                ordered_candidates = sorted(
                    step_candidates,
                    key=lambda h: h.score(length_penalty),
                    reverse=True
                )

                active_beams = []
                for hypo in ordered_candidates[:beam_size]:
                    if hypo.tokens[-1].item() == self.tgt_dict.eos():
                        # Completed sentence
                        hypo.tokens = hypo.tokens[:-1] # Remove EOS
                        completed_hypotheses.append(hypo)
                    else:
                        active_beams.append(hypo)

                if not active_beams:
                    break
            
            # --- 4. Finalize Translation ---
            final_hypotheses = completed_hypotheses + active_beams
            if not final_hypotheses:
                logger.warning(f"[{self.model_type}] No hypotheses generated for: {source_text}")
                return {'translation': "[EMPTY]", 'knn_skip_rate': 0.0}, []

            final_hypotheses.sort(key=lambda h: h.score(length_penalty), reverse=True)
            best_hypothesis = final_hypotheses[0]
            
            final_tokens = best_hypothesis.tokens.tolist()
            if final_tokens and final_tokens[0] == self.tgt_dict.eos(): final_tokens = final_tokens[1:]
            final_translation = self._tokens_to_words(final_tokens) if final_tokens else "[EMPTY]"

            return {
                'translation': final_translation,
                'knn_skip_rate': 0.0, # Baselines always theoretically run kNN logic
            }, trajectory

    def _tokens_to_words(self, tokens: List[int]) -> str:
        """Decodes token IDs into a string using the appropriate BPE processor."""
        if not tokens: return ""
        try:
            if self.bpe_type == "sentencepiece" or self.bpe_type == "sentence_piece":
                eos_id = self.tgt_dict.eos()
                decoded = self.spm_processor.decode([t for t in tokens if t != eos_id]).strip() # type: ignore
            else:
                bpe_string = self.tgt_dict.string(
                    torch.tensor(tokens, dtype=torch.long),
                    bpe_symbol=None,
                    escape_unk=True
                )
                if self.bpe and hasattr(self.bpe, 'decode'):
                    try:
                        decoded = self.bpe.decode(bpe_string)
                        decoded = decoded.replace("@@ ", "").replace("@-@", "").replace("@@", "").strip()
                    except Exception as e:
                        logger.warning(f"self.bpe.decode failed: {e}. Falling back to regex.")
                        decoded = re.sub(r'@@\s?|@-@\s?', '', bpe_string).strip()
                else:
                    decoded = re.sub(r'@@\s?|@-@\s?', '', bpe_string).strip()
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error decoding tokens: {tokens}, {e}")
            return "[DECODING_ERROR]"
        return decoded

    def _align_entities(self, source_entities: List[str], generated_entities: List[str]) -> Tuple[float, Set[str]]:
        """
        Aligns source entities with generated entities to calculate coverage.
        Uses exact matching followed by soft semantic alignment.
        """
        if not source_entities or not generated_entities:
            return 0.0, set()
        soft_alignment_score = 0.0
        covered_source_entities = set()
        remaining_src = list(source_entities)
        remaining_gen = list(generated_entities)
        src_indices_to_remove = []
        gen_indices_to_remove = []
    
        # Phase 1: Exact string matching (case-insensitive)
        for i, src_ent in enumerate(remaining_src):
            src_ent_lower = src_ent.lower()
            found_match_idx = -1
            for j, gen_ent in enumerate(remaining_gen):
                if j not in gen_indices_to_remove and src_ent_lower == gen_ent.lower():
                    found_match_idx = j
                    break
            if found_match_idx != -1:
                soft_alignment_score += 1.0
                covered_source_entities.add(src_ent)
                src_indices_to_remove.append(i)
                gen_indices_to_remove.append(found_match_idx)
        
        for i in sorted(src_indices_to_remove, reverse=True): del remaining_src[i]
        for j in sorted(gen_indices_to_remove, reverse=True): del remaining_gen[j]
        
        # Phase 2: Semantic alignment using Hungarian algorithm
        if not remaining_src or not remaining_gen:
            return soft_alignment_score, covered_source_entities
        
        src_embs = self.sentence_encoder.encode(remaining_src, convert_to_tensor=True, show_progress_bar=False)
        gen_embs = self.sentence_encoder.encode(remaining_gen, convert_to_tensor=True, show_progress_bar=False)
        similarity_matrix = util.pytorch_cos_sim(src_embs, gen_embs).cpu().numpy()
        cost_matrix = 1.0 - similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        similarity_threshold = 0.75
        for r, c in zip(row_ind, col_ind):
            similarity = similarity_matrix[r, c]
            if similarity >= similarity_threshold:
                soft_alignment_score += similarity
                covered_source_entities.add(remaining_src[r])
        return float(soft_alignment_score), covered_source_entities

    def _identify_source_entity_token_spans(self, bpe_source_text: str, source_entities: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """Finds the token indices corresponding to source entities."""
        if not source_entities: return {}
        bpe_source_tokens = bpe_source_text.split()
        entity_spans = defaultdict(list)
        for entity in sorted(source_entities, key=len, reverse=True):
            bpe_entity = ""
            if self.bpe_type == "fastbpe":
                if self.bpe is None: raise ValueError("BPE tokenizer not initialized for fastBPE.")
                try: bpe_entity = self.bpe.encode(entity).split()
                except Exception as e: raise RuntimeError(f"fastbpe.encode failed for entity '{entity}': {e}")
            elif self.bpe_type in ["sentencepiece", "sentence_piece"]:
                try: bpe_entity = self.spm_processor.encode(entity, out_type=str) # type: ignore
                except Exception as e: raise RuntimeError(f"sentencepiece processing failed for entity '{entity}': {e}")
            else: raise ValueError(f"Unsupported BPE rule: {self.bpe_type}")
            len_ent = len(bpe_entity)
            for i in range(len(bpe_source_tokens) - len_ent + 1):
                if bpe_source_tokens[i:i+len_ent] == bpe_entity:
                    entity_spans[entity].append((i, i + len_ent))
        return dict(entity_spans)

    def compute_error_state(self, source_text: str, generated_prefix_words: str, generated_tokens: List[int], source_emb: torch.Tensor, source_entities: List[str], decoder_out: torch.Tensor) -> Tuple[ErrorStateVector, Set[str]]:
        """Calculates the Error State Vector E_t (Semantic, Coverage, Surprisal, Repetition)."""
        generated_text = ""
        if isinstance(generated_prefix_words, list) and all(isinstance(item, str) for item in generated_prefix_words):
            generated_text = " ".join(generated_prefix_words)
        else:
            generated_text = generated_prefix_words
        try:
            if source_emb is None: source_emb = self.sentence_encoder.encode([source_text], convert_to_tensor=True, show_progress_bar=False)
            generated_emb = self.sentence_encoder.encode([generated_text], convert_to_tensor=True, show_progress_bar=False)
            cos_sim = F.cosine_similarity(source_emb, generated_emb).item()
            semantic_drift = np.clip(1.0 - cos_sim, 0.0, 2.0)
        except Exception as e:
            logger.error(f"Semantic drift calculation failed: {e}")
            semantic_drift = 1.0 # Penalize on failure
            
        generated_entities = [ent.text for ent in self.ner_en(generated_prefix_words).ents]
        _, covered_entities_set = self._align_entities(source_entities, generated_entities)
        coverage = len(covered_entities_set) / len(source_entities) if source_entities else 1.0
        error_coverage = 1.0 - coverage

        surprisal = 0.0
        logits = decoder_out[:, -1, :].squeeze(0)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9)).item()
        max_entropy = math.log2(self.vocab_size)
        if max_entropy > 0: surprisal = np.clip(entropy / max_entropy, 0.0, 1.0)

        repetition = 0.0
        window_size = 20
        if len(generated_tokens) > 1:
            window = generated_tokens[-window_size:]
            token_tensor = torch.tensor(window, dtype=torch.long, device=self.device)
            # Access embedding layer properly through Fairseq interface
            decoder_module = cast(FairseqDecoder, self.model.decoder)
            embedding_layer = cast(nn.Embedding, decoder_module.embed_tokens)
            embeddings = embedding_layer(token_tensor)
            
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
            upper_triangle_indices = torch.triu_indices(sim_matrix.shape[0], sim_matrix.shape[1], offset=1)
            if upper_triangle_indices.numel() > 0:
                sim_values = sim_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]
                repetition = sim_values.mean().item()
                repetition = np.clip(repetition, 0.0, 1.0)
        
        error_vector = ErrorStateVector(
            error_semantic=semantic_drift,
            error_coverage=error_coverage,
            error_fluency_surprisal=surprisal,
            error_fluency_repetition=repetition
        )
        return error_vector, covered_entities_set

    def compute_context_state(self, decoder_out: torch.Tensor, query_embedding: np.ndarray, attention_weights: torch.Tensor, encoder_hidden_states: torch.Tensor, uncovered_entities: Set[str], source_entity_spans: Dict[str, List[Tuple[int, int]]]) -> GenerativeContextVector:
        """Calculates the Context State Vector H_t (Focus, Consistency, Stability, Volatility)."""
        logits = decoder_out[:, -1, :].squeeze(0)
        faith_focus = 0.0
        if attention_weights is not None and uncovered_entities:
            source_len = attention_weights.shape[0]
            entity_mask = torch.zeros(source_len, device=self.device)
            for entity in uncovered_entities:
                if entity in source_entity_spans:
                    for start, end in source_entity_spans[entity]:
                        entity_mask[start:end] = 1.0
            faith_focus = torch.sum(attention_weights * entity_mask).item()
            faith_focus = np.clip(faith_focus, 0.0, 1.0)
        
        consistency = 0.0
        query_tensor = torch.from_numpy(query_embedding).to(self.device).squeeze()
        try:
            if attention_weights is not None and encoder_hidden_states is not None:
                attention_context_vector = torch.bmm(attention_weights.unsqueeze(0).unsqueeze(0), encoder_hidden_states).squeeze(0).squeeze(0)
                attention_context_vector_norm = torch.nn.functional.normalize(attention_context_vector, p=2, dim=0)
                consistency = F.cosine_similarity(query_tensor, attention_context_vector_norm, dim=0).item()
                consistency = np.clip((consistency + 1.0) / 2.0, 0.0, 1.0)
        except Exception as e:
            logger.warning(f"Consistency calculation failed: {e}")

        context_stability = 0.0
        try:
            if len(self.query_history) > 0:
                history_tensors = torch.stack(list(self.query_history), dim=0)
                similarities = F.cosine_similarity(query_tensor.unsqueeze(0), history_tensors, dim=1)
                context_stability = similarities.mean().item()
                context_stability = np.clip((context_stability + 1.0) / 2.0, 0.0, 1.0)
            self.query_history.append(query_tensor)
        except Exception as e:
            logger.warning(f"Stability calculation failed: {e}")

        volatility = 0.0
        try:
            top_k = 10
            top_k_logits, _ = torch.topk(logits, k=top_k)
            margin_k = torch.mean(top_k_logits[0] - top_k_logits[1:]) if top_k > 1 else 10.0
            current_confidence = torch.sigmoid(0.5 * torch.Tensor(margin_k)).item()
            self.confidence_history.append(current_confidence)
            if len(self.confidence_history) > 1:
                volatility = np.std(self.confidence_history)
        except Exception as e:
            if self.is_debug: logger.warning(f"Volatility calculation failed: {e}")
        
        context_vector = GenerativeContextVector(
            context_faith_focus=faith_focus,
            context_consistency=consistency,
            context_stability=context_stability,
            context_confidence_volatility=float(volatility)
        )
        return context_vector

    def project_to_query_embedding(self, hidden_state: torch.Tensor) -> np.ndarray:
        """Projects the hidden state to the kNN query space."""
        with torch.no_grad():
            last_hidden_vec = hidden_state[0, -1, :] if len(hidden_state.shape) == 3 else hidden_state
            query_embedding = last_hidden_vec.cpu().numpy()
            return query_embedding.astype('float32')

    def teardown(self):
        """Clean up auxiliary models to free memory."""
        if hasattr(self, "sentence_encoder") and getattr(self, "sentence_encoder"): del self.sentence_encoder
        if hasattr(self, "lm_tokenizer") and getattr(self, "lm_tokenizer"): del self.lm_tokenizer
        if hasattr(self, "lm_model") and getattr(self, "lm_model"): del self.lm_model
        if hasattr(self, "ner_de") and getattr(self, "ner_de"): del self.ner_de
        if hasattr(self, "ner_en") and getattr(self, "ner_en"): del self.ner_en
        if hasattr(self, "model") and getattr(self, "model"): del self.model
        if hasattr(self, "args") and getattr(self, "args"): del self.args
        if hasattr(self, "task") and getattr(self, "task"): del self.task
        if hasattr(self, "bpe") and getattr(self, "bpe"): del self.bpe
        if hasattr(self, "src_dict") and getattr(self, "src_dict"): del self.src_dict
        if hasattr(self, "tgt_dict") and getattr(self, "tgt_dict"): del self.tgt_dict
        if hasattr(self, "spm_processor") and getattr(self, "spm_processor"): del self.spm_processor
        if hasattr(self, "confidence_history") and getattr(self, "confidence_history"): del self.confidence_history
        if hasattr(self, "query_history") and getattr(self, "query_history"): del self.query_history
        if hasattr(self, "fluency_loss_history") and getattr(self, "fluency_loss_history"): del self.fluency_loss_history
        if hasattr(self, "config") and getattr(self, "config"): del self.config
        if hasattr(self, "device") and getattr(self, "device"): del self.device
        if hasattr(self, "model_type") and getattr(self, "model_type"): del self.model_type
        if hasattr(self, "bpe_type") and getattr(self, "bpe_type"): del self.bpe_type
        if hasattr(self, "vocab_size") and getattr(self, "vocab_size"): del self.vocab_size
        if hasattr(self, "is_debug") and getattr(self, "is_debug"): del self.is_debug
        if hasattr(self, "knn_ds_selection") and getattr(self, "knn_ds_selection"): del self.knn_ds_selection
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# =====================================================================
# MAIN EVALUATOR
# =====================================================================

class ModelEvaluator:
    """
    Orchestrates the high-level evaluation process. It runs the specified models
    (PAEC, Baselines) on the test dataset, collects results, computes aggregate
    metrics, and generates summary reports and plots.
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.config.device)
        self.is_debug = self.config.is_debug
        
        # Containers for storing detailed results for analysis
        self.all_step_results: List[StepResult] = []
        self.final_translation_results: List[TranslationResult] = []
        
        # The P matrix (Lyapunov weights) used to calculate V(E)
        self.P_matrix: Optional[torch.Tensor] = None

    def run_all_evaluations(self, dataset: List[Dict]):
        """
        Executes the full evaluation suite for all configured models.
        """
        # --- Evaluate PAEC kNN-MT (Online & Offline) ---
        offline_path = self.config.offline_policy_model_dir
        is_offline_available = offline_path is not None and str(offline_path).strip() != "" and os.path.exists(offline_path)
        mode = self.config.paec_test_mode

        run_online = False
        run_offline = False

        if mode == -1:
            logger.info("PAEC Test Mode is -1. Skipping PAEC kNN-MT evaluation.")
        elif mode == 1:
            run_online = True
        elif mode == 2:
            if is_offline_available:
                run_offline = True
            else:
                logger.warning(f"PAEC Test Mode is 2 (Offline Only), but offline path '{offline_path}' is invalid. Skipping PAEC kNN-MT.")
        else:
            # Mode 0 (Default): Run both if possible
            run_online = True
            if is_offline_available:
                run_offline = True
            else:
                logger.info(f"PAEC Test Mode is 0, but offline path '{offline_path}' is not valid/set. Falling back to Online Only.")

        if run_online:
            # Force online optimization by suppressing the offline policy path
            self._evaluate_paec(dataset, model_display_name="PAEC kNN-MT (Online)", force_online=True)
        
        if run_offline:
            # Run using the offline policy network
            self._evaluate_paec(dataset, model_display_name="PAEC kNN-MT (Offline π)", force_online=False)
        
        # --- Evaluate Baseline Models ---
        self._evaluate_baseline("pure_nmt", dataset)
        self._evaluate_baseline("vanilla", dataset)
        self._evaluate_baseline("adaptive", dataset)

        # --- Generate Final Report ---
        step_df = pd.DataFrame([asdict(r) for r in self.all_step_results])
        final_df = pd.DataFrame([asdict(r) for r in self.final_translation_results])
        
        self._verify_bpe_decoding(final_df)
        self._generate_report_and_plots(step_df, final_df)

    def _get_v_error(self, error_state: ErrorStateVector) -> float:
        """
        Calculates the Lyapunov Energy V(E) = E^T P E.
        Uses the globally loaded P_matrix.
        """
        if self.P_matrix is None:
            return -1.0 
        try:
            if self.P_matrix.device != self.device:
                self.P_matrix = self.P_matrix.to(self.device)
            
            e_vec = torch.tensor(error_state.to_vector(), dtype=torch.float32, device=self.device)
            
            # V(E) calculation: weighted sum of squares
            v_error = torch.sum((e_vec * self.P_matrix) * e_vec).item()
            return v_error
        except Exception as e:
            logger.error(f"Failed to compute V(E): {e}")
            return -2.0 

    def _evaluate_paec(self, dataset: List[Dict], model_display_name: str = "PAEC kNN-MT (Online)", force_online: bool = False):
        """
        Runs evaluation for the PAEC system.
        """
        if self.is_debug: logger.info("="*20 + f" Evaluating {model_display_name} " + "="*20)
        
        run_config = copy.deepcopy(self.config)
        if force_online:
            run_config.offline_policy_model_dir = ""
            if self.is_debug: logger.info(f"[{model_display_name}] Force online enabled. Offline policy path suppressed.")

        paec_system = PAECTranslationSystem(run_config)
        
        # Extract P_matrix for V(E) calculation from the loaded system
        if hasattr(paec_system.pipeline.knn_system, 'paec_P_values') and paec_system.pipeline.knn_system.paec_P_values is not None:
            self.P_matrix = paec_system.pipeline.knn_system.paec_P_values
            if self.is_debug: logger.info(f"Successfully loaded P_matrix (dim={self.P_matrix.shape[0]}) from PAEC system.")
        else:
            logger.error("Could not load P_matrix from PAEC system! V(E) calculations will fail.")
            # Fallback: Load from checkpoint
            try:
                ckpt_path = Path(self.config.dynamics_model_dir) / "checkpoint_best.pt"
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.P_matrix = torch.tensor(ckpt['P'], dtype=torch.float32, device=self.device)
                logger.info(f"Successfully loaded P_matrix manually from {ckpt_path}")
            except Exception as e:
                logger.error(f"Manual P_matrix load failed: {e}. V(E) will be -1.")
                self.P_matrix = None

        # Main Evaluation Loop
        for i, sample in enumerate(tqdm(dataset, desc=model_display_name)):
            source, reference = sample['source'], sample['reference']
            
            # Perform translation and capture trajectory
            result, trajectory = paec_system.translate(source, reference)
            
            if self.is_debug:
                print("-" * 30)
                print(f"[DEBUG] Raw {model_display_name} Result for Sample {i}")
                print(f"  Source: '{source}'")
                print(f"  Reference: '{reference}'")
                print(f"  Hypothesis Returned: '{result.get('translation', 'N/A')}'")
                print(f"  Latency: {result.get('latency_ms', -1):.2f} ms")
                print(f"  Memory: {result.get('memory_mb', -1):.2f} MB")
                print(f"  Skip Rate: {result.get('knn_skip_rate', -1):.3f}")
                print(f"  Trajectory steps: {len(trajectory)}")
                if result.get('translation') == reference:
                    print("  [DEBUG] >>> WARNING: Hypothesis is identical to Reference <<<")
                print("-" * 30)
            
            # Log detailed step data for analysis
            for step_data in trajectory:
                error_state = step_data['error_state']
                self.all_step_results.append(StepResult(
                    model_name=model_display_name,
                    sample_id=i,
                    step=step_data['step'],
                    error_semantic=error_state.error_semantic,
                    error_coverage=error_state.error_coverage,
                    error_fluency_surprisal=error_state.error_fluency_surprisal,
                    error_fluency_repetition=error_state.error_fluency_repetition,
                    error_norm=error_state.norm(),
                    v_error=self._get_v_error(error_state)
                ))

            # Log final summary result
            self.final_translation_results.append(TranslationResult(
                model_name=model_display_name, source=source, reference=reference,
                hypothesis=result.get('hypothesis', '[FAILED]'),
                latency_ms=result.get('latency_ms', -1.0),
                memory_mb=result.get('memory_mb', -1.0),
                knn_skip_rate=result.get('knn_skip_rate', -1.0),
                trajectory=[] 
            ))
            
        paec_system.teardown()
        return
    
    def _evaluate_baseline(self, model_type: str, dataset: List[Dict]):
        """Runs evaluation for baseline models (Pure, Vanilla, Adaptive)."""
        display_name = f"{model_type.capitalize()}-kNN-MT"
        if model_type == "pure_nmt":
            display_name = "Vanilla Pure NMT"

        if self.is_debug: logger.info("="*20 + f" Evaluating {display_name} " + "="*20)
        baseline_system = KNNBoxBaselineSystem(model_type, self.config)
        
        for i, sample in enumerate(tqdm(dataset, desc=display_name)):
            source, reference = sample['source'], sample['reference']
 
            # Manual timing for consistency
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start_time = time.perf_counter()
            
            # Translate using the baseline wrapper (which captures trajectory)
            result, trajectory = baseline_system.translate(source, reference)
            hypothesis = result.get('translation', '[FAILED]')
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000
            memory = torch.cuda.max_memory_allocated(self.device) / (1024**2) if self.device.type == 'cuda' else 0.0
            knn_skip_rate = 0.0 # Baselines do not skip based on policy
            
            if self.is_debug and i < 5:
                logger.info(f"[{model_type.capitalize()}-kNN-MT] Source: {source}")
                logger.info(f"[{model_type.capitalize()}-kNN-MT] Hypothesis: {hypothesis}")
                logger.info(f"[{model_type.capitalize()}-kNN-MT] Reference: {reference}")
                logger.info(f"[{model_type.capitalize()}-kNN-MT] Trajectory steps: {len(trajectory)}")

            # Log step-by-step results
            for step_data in trajectory:
                error_state = step_data['error_state']
                self.all_step_results.append(StepResult(
                    model_name=display_name,
                    sample_id=i,
                    step=step_data['step'],
                    error_semantic=error_state.error_semantic,
                    error_coverage=error_state.error_coverage,
                    error_fluency_surprisal=error_state.error_fluency_surprisal,
                    error_fluency_repetition=error_state.error_fluency_repetition,
                    error_norm=error_state.norm(),
                    v_error=self._get_v_error(error_state)
                ))

            # Log final result
            self.final_translation_results.append(TranslationResult(
                model_name=display_name,
                source=source,
                reference=reference,
                hypothesis=hypothesis,
                latency_ms=latency,
                memory_mb=memory,
                knn_skip_rate=knn_skip_rate,
                trajectory=[]
            ))
            
        baseline_system.teardown()
        torch.cuda.empty_cache()
        return

    def _verify_bpe_decoding(self, df: pd.DataFrame, sample_size: int = 5):
        """Checks a sample of outputs for residual BPE artifacts (e.g., '@@')."""
        if self.is_debug: logger.info("\nVerifying BPE decoding for all models...")
        models = df['model_name'].unique()
        all_clear = True

        for model in models:
            model_df = df[df['model_name'] == model]
            hypotheses_sample = model_df['hypothesis'].head(sample_size).tolist()
            
            found_artifact = False
            for i, hypo in enumerate(hypotheses_sample):
                if '@@' in hypo or '@' in hypo:
                    logger.warning(f"  [!] Potential BPE artifact found in '{model}' output (sample {i+1}): '{hypo}'")
                    found_artifact = True
                    all_clear = False
            
            if self.is_debug and not found_artifact:
                logger.info(f"  [✓] BPE check passed for '{model}' (checked {len(hypotheses_sample)} samples).")
        
        if self.is_debug:
            if all_clear:
                logger.info("BPE verification completed successfully for all models.")
            else:
                logger.warning("BPE verification completed with warnings. Please review logs.")
    
    def _generate_report_and_plots(self, step_df: pd.DataFrame, final_df: pd.DataFrame):
        """
        Calculates final metrics and generates visualization plots.
        """
        # --- Part 1: Corpus-Level Metrics (BLEU/TER/CHRF) ---
        if self.is_debug: logger.info("\nGenerating final report and plots (Part 1: BLEU/TER/CHRF)...")
        self._verify_bpe_decoding(final_df)
        
        models = final_df['model_name'].unique()
        summary_data = []

        for model in models:
            model_df = final_df[final_df['model_name'] == model]
            hypotheses = model_df['hypothesis'].tolist()
            references = model_df['reference'].tolist()
            
            if self.is_debug and model == 'PAEC kNN-MT':
                print("-" * 30)
                print(f"[DEBUG] Data passed to BLEU scorer for {model}:")
                print(f"  Number of Hypotheses: {len(hypotheses)}")
                print(f"  Number of References lists: {len(references)}")
                if len(hypotheses) > 0 and len(references) > 0:
                    print("  First 3 Hypotheses:")
                    for i in range(min(3, len(hypotheses))):
                        print(f"    {i+1}: '{hypotheses[i]}'")
                    print("  First 3 References:")
                    for i in range(min(3, len(references))):
                        print(f"     {i+1}: {references[i]}")
                    matches = [h == r[0] for h, r in zip(hypotheses, references)]
                    if any(matches):
                        print(f"  [DEBUG] >>> WARNING: {sum(matches)} out of {len(matches)} hypotheses EXACTLY match their reference! <<<")
                print("-" * 30)
            if self.is_debug:
                import pprint
                print("Before scoring:")
                print(f"  Hypotheses Samples (100): ")
                pprint.pprint(hypotheses)
                print(f"  References Samples (100): ")
                pprint.pprint(references)
            
            hypotheses = [h.strip() for h in hypotheses]
            references = [ref.strip() for ref in references]

            # Calculate translation quality scores
            bleu_score = corpus_bleu(hypotheses, [references])
            chrf_score = corpus_chrf(hypotheses, [references])
            ter_score = corpus_ter(hypotheses, [references])
            
            summary_data_append = {
                 'Model': model,
                'BLEU': bleu_score.score,
                'CHRF': chrf_score.score,
                'TER': ter_score.score,
                'Avg_Latency_ms': model_df['latency_ms'].mean(),
                'Avg_Memory_MB': model_df['memory_mb'].mean(),
                'Avg_kNN_Skip_Rate': model_df['knn_skip_rate'].mean()
            }
            summary_data.append(summary_data_append)
            
        summary_df = pd.DataFrame(summary_data)
        
        # --- Part 2: Step-Level Error Metrics ---
        if self.is_debug: logger.info("\nGenerating final report and plots (Part 2: Error State Metrics)...")
        
        # Save raw data to disk
        step_df.to_csv(self.output_dir / "detailed_step_results.csv", index=False)
        final_df.to_csv(self.output_dir / "detailed_final_results.csv", index=False)
        
        if step_df.empty:
            logger.warning("Step DataFrame is empty, skipping error state analysis.")
            summary_path = self.output_dir / "summary_report.csv"
            summary_df.to_csv(summary_path, index=False)
            return

        # Aggregate step metrics by model and step number
        step_agg_df = step_df.groupby(['model_name', 'step']).agg(
            Avg_V_Error=('v_error', 'mean'),
            Avg_Error_Semantic=('error_semantic', 'mean'),
            Avg_Error_Coverage=('error_coverage', 'mean'),
            Avg_Error_Fluency_Surprisal=('error_fluency_surprisal', 'mean'),
            Avg_Error_Fluency_Repetition=('error_fluency_repetition', 'mean'),
            Avg_Error_Norm=('error_norm', 'mean')
        ).reset_index()
        
        # Calculate summary statistics (Peak V, Final V, Accumulated V)
        error_summary_stats = step_df.groupby('model_name').agg(
            Avg_Peak_V_Error=('v_error', lambda x: step_df.loc[x.index].groupby('sample_id')['v_error'].max().mean()),
            Avg_Final_V_Error=('v_error', lambda x: step_df.loc[x.index].loc[step_df.loc[x.index].groupby('sample_id')['step'].idxmax()]['v_error'].mean()),
            Avg_Accumulated_V_Error=('v_error', lambda x: step_df.loc[x.index].groupby('sample_id')['v_error'].sum().mean())
        ).reset_index()
        
        # Calculate Convergence Time (Steps until V < 0.1)
        CONVERGENCE_THRESHOLD = 0.1 
        converged_steps = step_df[step_df['v_error'] < CONVERGENCE_THRESHOLD]
        if not converged_steps.empty:
            first_converged_step = converged_steps.groupby(['model_name', 'sample_id'])['step'].min()
            avg_steps_to_converge = first_converged_step.groupby('model_name').mean().reset_index(name='Avg_Steps_to_Converge')
            error_summary_stats = error_summary_stats.merge(avg_steps_to_converge, on='model_name', how='left')
        else:
            error_summary_stats['Avg_Steps_to_Converge'] = np.nan

        summary_df = summary_df.merge(error_summary_stats, left_on='Model', right_on='model_name', how='left').drop(columns=['model_name'])
        
        # Save final summary
        summary_path = self.output_dir / "summary_report.csv"
        summary_df.to_csv(summary_path, index=False)
        
        if self.is_debug:
            print("\n" + "="*80 + "\nEVALUATION SUMMARY\n" + "="*80)
            print(summary_df.to_string(index=False))
            print("\n" + "="*80 + "\nStep-by-Step Aggregates (Head)\n" + "="*80)
            print(step_agg_df.head(15))
        
        # --- Part 3: Generate Visualization Plots ---
        if self.is_debug: logger.info("\nGenerating final report and plots (Part 3: Visualizations)...")
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 24))
        fig.suptitle('PAEC kNN-MT Validation: Stability and Performance Analysis', fontsize=20, y=1.03)
        
        model_colors = {
            'PAEC kNN-MT': 'green',
            'Vanilla-kNN-MT': 'red',
            'Adaptive-kNN-MT': 'blue'
        }
        
        # 1. BLEU Bar Chart
        fig, ax = plt.subplots(figsize=(9, 8))
        bleu_values = [summary_df[summary_df['Model'] == model]['BLEU'].iloc[0] for model in models if not summary_df[summary_df['Model'] == model].empty]
        ax.bar(models, bleu_values, color=[model_colors.get(model, 'gray') for model in models])
        ax.set_title('Quality: Corpus BLEU Score by Model (Higher=Better)', fontsize=16)
        ax.set_xlabel('Model')
        ax.set_ylabel('BLEU Score')
        ax.grid(True, alpha=0.5, linestyle=':')
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_BLEU_bar.png", dpi=300)
        plt.close(fig)
        
        # 2. Latency Bar Chart
        fig, ax = plt.subplots(figsize=(9, 8))
        latency_values = [summary_df[summary_df['Model'] == model]['Avg_Latency_ms'].iloc[0] for model in models if not summary_df[summary_df['Model'] == model].empty]
        ax.bar(models, latency_values, color=[model_colors.get(model, 'gray') for model in models])
        ax.set_title('Latency: Average Latency by Model (Lower=Better)', fontsize=16)
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Latency (ms)')
        ax.grid(True, alpha=0.5, linestyle=':')
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_latency_bar.png", dpi=300)
        plt.close(fig)
        
        # 3. Lyapunov Error V(E) vs Step ("Money Plot")
        fig, ax = plt.subplots(figsize=(9, 8))
        for model in models:
            model_data = step_agg_df[step_agg_df['model_name'] == model]
            ax.plot(model_data['step'], model_data['Avg_V_Error'], marker='o', markersize=4,
                    label=model, color=model_colors.get(model, 'gray'))
        ax.set_title('Avg. Lyapunov Error $V\\left(\\mathcal{E}_t\\right)$ vs. Decoding Step (Stability)', fontsize=16)
        ax.set_xlabel('Decoding Step ($t$)')
        ax.set_ylabel('Average $V\\left(\\mathcal{E}_t\\right)$ (Lower=Better)')
        ax.legend()
        ax.grid(True, alpha=0.5, linestyle=':')
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_Lyapunov_error_vs_step.png", dpi=300)
        plt.close(fig)

        # 4. Semantic Error vs Step
        fig, ax = plt.subplots(figsize=(9, 8))
        for model in models:
            model_data = step_agg_df[step_agg_df['model_name'] == model]
            ax.plot(model_data['step'], model_data['Avg_Error_Semantic'], marker='.', markersize=4, linestyle='--',
                    label=model, color=model_colors.get(model, 'gray'))
        ax.set_title('Avg. Semantic Error vs. Step', fontsize=16)
        ax.set_xlabel('Decoding Step ($t$)')
        ax.set_ylabel('Avg $\\epsilon_t^{\\left(\\text{sem}\\right)}$ (Lower=Better)')
        ax.legend()
        ax.grid(True, alpha=0.5, linestyle=':')
        plt.tight_layout()
        plt.savefig(self.output_dir / "04_semantic_error.png", dpi=300)
        plt.close(fig)

        # 5. Coverage Error vs Step
        fig, ax = plt.subplots(figsize=(9, 8))
        for model in models:
            model_data = step_agg_df[step_agg_df['model_name'] == model]
            ax.plot(model_data['step'], model_data['Avg_Error_Coverage'], marker='.', markersize=4, linestyle='--',
                    label=model, color=model_colors.get(model, 'gray'))
        ax.set_title('Avg. Coverage Error vs. Step', fontsize=16)
        ax.set_xlabel('Decoding Step ($t$)')
        ax.set_ylabel('Avg $\\epsilon_t^{\\left(\\text{cov}\\right)}$ (Lower=Better)')
        ax.legend()
        ax.grid(True, alpha=0.5, linestyle=':')
        plt.tight_layout()
        plt.savefig(self.output_dir / "05_coverage_error.png", dpi=300)
        plt.close(fig)
        
        # 6. Surprisal Error vs Step
        fig, ax = plt.subplots(figsize=(9, 8))
        for model in models:
            model_data = step_agg_df[step_agg_df['model_name'] == model]
            ax.plot(model_data['step'], model_data['Avg_Error_Fluency_Surprisal'], marker='.', markersize=4, linestyle=':',
                    label=model, color=model_colors.get(model, 'gray'))
        ax.set_title('Avg. Fluency Error (Surprisal) vs. Step', fontsize=16)
        ax.set_xlabel('Decoding Step ($t$)')
        ax.set_ylabel('Avg $\\epsilon_t^{\\left(\\text{surp}\\right)}$ (Lower=Better)')
        ax.legend()
        ax.grid(True, alpha=0.5, linestyle=':')
        plt.tight_layout()
        plt.savefig(self.output_dir / "06_surprisal_error.png", dpi=300)
        plt.close(fig)

        # 7. Repetition Error vs Step
        fig, ax = plt.subplots(figsize=(9, 8))
        for model in models:
            model_data = step_agg_df[step_agg_df['model_name'] == model]
            ax.plot(model_data['step'], model_data['Avg_Error_Fluency_Repetition'], marker='.', markersize=4, linestyle=':',
                    label=model, color=model_colors.get(model, 'gray'))
        ax.set_title('Avg. Fluency Error (Repetition) vs. Step', fontsize=16)
        ax.set_xlabel('Decoding Step ($t$)')
        ax.set_ylabel('Avg $\\epsilon_t^{\\left(\\text{rep}\\right)}$ (Lower=Better)')
        ax.legend()
        ax.grid(True, alpha=0.5, linestyle=':')
        plt.tight_layout()
        plt.savefig(self.output_dir / "07_repetition_error.png", dpi=300)
        plt.close(fig)
        
        if self.is_debug: 
            logger.info(f"Saved visualizations to {self.config.output_dir}")

# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """
    Main entry point. Parses arguments, runs calibration (optional), sets up the
    experiment configuration, and triggers the evaluator.
    """
    parser = argparse.ArgumentParser(description='PAEC kNN-MT Validation')
    
    # Dataset Arguments
    parser.add_argument('--dataset', type=str, default='wmt19')
    parser.add_argument('--source-lang', type=str, default='de')
    parser.add_argument('--target-lang', type=str, default='en')
    parser.add_argument('--test-size', type=int, default=2000)
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--length-penalty', type=float, default=1.0)
    
    # Path Arguments (Defaults to project config structure)
    parser.add_argument('--dynamics-model-dir', type=str, default=str(paec_config.PATHS["dynamics_model_dir"] / "Champion"))
    parser.add_argument('--offline-policy-model-dir', type=str, default=str(paec_config.PATHS["policy_model_dir"]))
    parser.add_argument('--paec-data-path', type=str, default=str(paec_config.PATHS["processed_data_dir"] / "strategy_comparison_stepwise_1000.csv"))
    parser.add_argument('--fairseq-checkpoint', type=str, default=str(paec_config.PATHS["nmt_model_dir"] / 'self-trained' / 'checkpoint_best.pt'))
    parser.add_argument('--fairseq-data-bin', type=str, default=str(paec_config.PATHS["data_bin_dir"]))
    parser.add_argument('--use-datastore', action="store_true", help='Use real datastore if available')
    parser.add_argument('--datastore-path', type=str, default=str(paec_config.PATHS["datastore_dir"]))
    parser.add_argument('--adaptive-combiner-path', type=str, default=str(paec_config.PATHS["models_dir"] / 'combiners' / 'adaptive_knn_mt'))
    
    # BPE Configuration
    parser.add_argument('--bpe-type', type=str, default=paec_config.BUILD_PIPELINE_SETTINGS.get("bpe", "sentencepiece"), help='BPE type (fastbpe or sentencepiece)')
    parser.add_argument('--bpe-codes', type=str, default=str(paec_config.PATHS["bpe_dir"] / "bpe.codes"), help='Path to fastBPE codes file (if using fastbpe).')
    parser.add_argument('--sentencepiece-model', type=str, default=str(paec_config.PATHS["bpe_dir"] / "spm.model"), help='Path to SentencePiece model file (if using sentencepiece).')
    
    # Output and Execution Settings
    parser.add_argument('--output-dir', type=str, default=str(paec_config.PATHS["results_dir"] / 'paec_validation'))
    parser.add_argument('--cache-dir', type=str, default='cache')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    # Specific kNN Parameters
    parser.add_argument('--vanilla-knn-k', type=int, default=16)
    parser.add_argument('--vanilla-knn-lambda', type=float, default=0.7)
    parser.add_argument('--vanilla-knn-temperature', type=float, default=10.0)
    parser.add_argument('--optimize_simulator_params', action="store_true")
    parser.add_argument('--optimize_pressure_params', action="store_true")
    parser.add_argument('--paec-test-mode', type=int, default=0, help='0=Test Both (Online/Offline), 1=Online Only, 2=Offline Only, -1=Skip PAEC')
    parser.add_argument('--debug', action="store_true")
    
    args = parser.parse_args()
    
    # --- 1. RUN CALIBRATION SCRIPT (Optional) ---
    # Calibrates baseline system throughput/latency if requested.
    if args.optimize_pressure_params:
        if args.debug: logger.info("[Info] Step 1: Calibrating system baselines...")
        calibrate_script_path = Path(__file__).resolve().parent / "00_calibrate_baselines.py"
        if not calibrate_script_path.exists():
            logger.error(f"[Error] Calibration script not found at {calibrate_script_path}")
            sys.exit(1)
        subprocess.run([sys.executable, str(calibrate_script_path)], check=True)
        if args.debug: logger.info("[Success] Calibration complete. config.py has been updated.\n")
    elif args.debug:
        logger.info("[Skipped] Step 1: Calibrating system baselines SKIPPED")

    # --- 2. SETUP CONFIG AND DATASET ---
    # Create the unified configuration object
    exp_config = ExperimentConfig(
        dataset_name=args.dataset,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        test_size=args.test_size,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        dynamics_model_dir=args.dynamics_model_dir,
        offline_policy_model_dir=args.offline_policy_model_dir,
        fairseq_checkpoint=args.fairseq_checkpoint,
        fairseq_data_bin=args.fairseq_data_bin,
        bpe_type=args.bpe_type,
        bpe_codes=args.bpe_codes,
        sentencepiece_model=args.sentencepiece_model,
        datastore_path=args.datastore_path,
        adaptive_combiner_path=args.adaptive_combiner_path,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
        seed=args.seed,
        vanilla_knn_k=args.vanilla_knn_k,
        vanilla_knn_lambda=args.vanilla_knn_lambda,
        vanilla_knn_temperature=args.vanilla_knn_temperature,
        optimize_simulator_params=args.optimize_simulator_params,
        paec_test_mode=args.paec_test_mode,
        is_debug=args.debug
    )
    
    # Load the evaluation dataset using the project's standard loader
    if args.debug: logger.info("Loading validation dataset...")
    loader = RealDatasetLoader()
    # Load ALL available raw lines from the test file to ensure correct indexing
    all_samples_raw = loader.load_all_datasets(split="test", size="full")
    total_available = len(all_samples_raw)

    # Determine effective test size (subset or full)
    if exp_config.test_size == -1 or exp_config.test_size >= total_available:
        effective_size = total_available
        logger.info(f"[Info] `./src/config.py`: User requested FULL test set. Running on all {total_available} samples.")
    else:
        effective_size = exp_config.test_size
        logger.info(f"[Info] `./src/config.py`: User requested partial test set. Running on {effective_size} / {total_available} samples.")

    # Create the final dataset list
    test_dataset = [{'source': s['source_text'], 'reference': s['target_text']} for s in all_samples_raw[:effective_size]]
    
    # --- 3. RUN EVALUATION ---
    evaluator = ModelEvaluator(exp_config)
    evaluator.run_all_evaluations(test_dataset)
    
    if args.debug: logger.info(f"\nVALIDATION COMPLETE. Results saved to: {exp_config.output_dir}")

if __name__ == "__main__":
    main()
