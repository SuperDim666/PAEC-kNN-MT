# -*- coding: utf-8 -*-
"""
src/system/knn_mt.py

[PAEC Framework - Core System]

This module implements the central kNN-MT system controller for the PAEC framework.
It integrates the Neural Machine Translation (NMT) model with the kNN datastore
retrieval mechanism and the PAEC control logic.

Key Responsibilities:
1.  **Model Management**: Loads the Fairseq NMT model, auxiliary models (SBERT,
    fluency scorers), and PAEC components (Dynamics Model T_theta, Policy Network Pi_phi).
2.  **State Computation**: Calculates the multifaceted system state S_t = (E_t, P_t, H_t)
    at each decoding step.
3.  **Control Loop**: Executes the translation process (Beam Search), querying the
    policy (heuristic or learned) to determine optimal retrieval actions (A_t).
4.  **kNN Integration**: Manages FAISS indices and performs retrieval and probability
    interpolation based on the selected action.
5.  **Trajectory Tracking**: Records detailed step-by-step data for training the
    dynamics model or analyzing system performance.
"""

import time, sys, os, math, re, random, inspect, json, copy, traceback, joblib, faiss, torch, argparse, logging, gc
from pathlib import Path
import pandas as pd

import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple, Optional, Any, Union, Set
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
import sentencepiece as spm

# Allow argparse.Namespace to be pickled safely
torch.serialization.add_safe_globals([argparse.Namespace])

# --- Project Imports ---
from fairseq import checkpoint_utils
from fairseq.data import encoders
from src import config
from src.core import ErrorStateVector, ResourcePressureVector, GenerativeContextVector, Action, SystemState
from src.simulation.resource_monitor import RealtimeResourceMonitor
from src.simulation.constraint_simulator import ProductionConstraintSimulator
from src.simulation.policy import Policy_Default_Balanced
import src.simulation.policy as policy_module
from src.models.paec_policy_network import PAECPolicyNetwork

# Flag to control one-time warnings
ONE_PRINT = True

# Add script paths to allow importing training modules for loading the dynamics model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scripts import t_train_Transformer  # type: ignore
from scripts.t_train_Transformer import InvertableColumnTransformer  # type: ignore

class Hypothesis:
    """
    Represents a single partial translation hypothesis in the beam search.
    Stores the sequence of tokens, cumulative probability, and the history of
    PAEC states (E, P, H) for trajectory analysis.
    """
    def __init__(
        self,
        tokens: torch.Tensor, log_prob: float,
        hidden_state: Union[torch.Tensor, None],
        context: Optional[torch.Tensor],
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        trajectory: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initializes a hypothesis.

        Args:
            tokens (torch.Tensor): Sequence of token IDs generated so far.
            log_prob (float): Cumulative log probability of the sequence.
            hidden_state (torch.Tensor): The decoder's hidden state at the last step.
            context (torch.Tensor): Attention context vector (optional).
            incremental_state (dict): Fairseq's cache for incremental decoding.
            trajectory (list): History of state-action tuples for this beam.
        """
        self.tokens = tokens                        # Tensor of token IDs
        self.log_prob = log_prob                    # Total log probability
        self.hidden_state = hidden_state            # Last hidden state from the decoder
        self.context = context                      # Attention context (if needed, currently unused)
        self.incremental_state = incremental_state  # Caching for autoregressive efficiency
        self.covered_entities: Set[str] = set()     # Track source entities covered by this hypothesis
        self.trajectory = trajectory if trajectory is not None else []
        self.skip_steps = 0                         # Counter for steps where retrieval was skipped
        self.total_steps = 0                        # Total decoding steps
    
    def extend(
        self,
        token: int, log_prob: float,
        hidden_state: Union[torch.Tensor, None],
        context: Optional[torch.Tensor],
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        step_dict: Optional[Dict[str, Any]] = None,
        is_skip: bool = False
    ):
        """
        Creates a new Hypothesis by appending a token to the current one.

        Args:
            token (int): The new token ID to append.
            log_prob (float): Log probability of the new token.
            hidden_state (torch.Tensor): New hidden state.
            context (torch.Tensor): New context vector.
            incremental_state (dict): Updated cache.
            step_dict (dict): PAEC state data for the current step (to be logged).
            is_skip (bool): Whether kNN retrieval was skipped at this step.

        Returns:
            Hypothesis: A new extended hypothesis instance.
        """
        new_trajectory = self.trajectory + ([step_dict] if step_dict is not None else [])
        new_hypothesis = Hypothesis(
            tokens=torch.cat([self.tokens, torch.tensor([token], device=self.tokens.device)]),
            log_prob=self.log_prob + log_prob,
            hidden_state=hidden_state,
            context=context,
            incremental_state=incremental_state,
            trajectory=new_trajectory
        )
        new_hypothesis.skip_steps = self.skip_steps + (1 if is_skip else 0)
        new_hypothesis.total_steps = self.total_steps + 1
        new_hypothesis.covered_entities = self.covered_entities.copy() # Inherit covered entities
        return new_hypothesis

    @property
    def latest_token(self):
        """Returns the ID of the last generated token."""
        return self.tokens[-1]

class kNNMTSystem:
    """
    The main controller for the Production-Aware Exposure Compensation (PAEC) system.
    Handles NMT inference, Datastore retrieval, State calculation, and Policy execution.
    """
    def __init__(
        self,
        use_datastore: bool = False, datastore_path: Optional[str] = None,
        evaluation_mode: bool = False,
        dynamics_model_dir: Optional[str] = None,
        offline_policy_model_dir: Optional[str] = None,
        policies_mix: Optional[Dict[str, float]] = None,
        is_debug: bool = False
    ):
        """
        Initializes the kNN-MT system.

        Args:
            use_datastore (bool): Whether to enable kNN retrieval capabilities.
            datastore_path (str): Path to the FAISS datastore directories.
            evaluation_mode (bool): If True, uses trained models (T_theta/Pi_phi) for control.
                                    If False, uses heuristic policies for data generation.
            dynamics_model_dir (str): Path to the trained Dynamics Model (T_theta).
            offline_policy_model_dir (str): Path to the trained Policy Network (Pi_phi).
            policies_mix (dict): Ratios for mixing heuristic policies (Data Gen mode only).
            is_debug (bool): Enable verbose debug logging.
        """
        self.evaluation_mode = evaluation_mode
        self.is_debug = is_debug
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu")
        if self.is_debug: print(f"[Info] Using device: {self.device}")
        
        # In Data Generation Mode, prepare heuristic policies
        if not self.evaluation_mode:
            self.policies_mix = policies_mix if policies_mix else config.POLICIES_MIX
            # Load heuristic policy classes dynamically from the policy module
            self.policy_classes = {
                name: obj for name, obj in inspect.getmembers(policy_module, inspect.isclass)
                # if obj.__module__ == 'src.simulation.policy'
            }
            if self.is_debug: print(f"[Info] Available policy classes: {self.policy_classes}")
            # Validate that requested mix keys exist in loaded classes
            for policy_name in self.policies_mix.keys():
                if policy_name not in self.policy_classes:
                    raise ValueError(f"Policy {policy_name} not found in registry")

        # --- 1. Load Fairseq Model and Task ---
        # Configure BPE based on pipeline settings (fastbpe or sentencepiece)
        self.bpe_type = config.BUILD_PIPELINE_SETTINGS.get("bpe", "sentencepiece").lower()
        arg_overrides = {
            'data': str(config.PATHS["data_bin_dir"]),
            'bpe': self.bpe_type,
        }
        if self.bpe_type == "fastbpe":
            arg_overrides['bpe_codes'] = str(config.PATHS["bpe_dir"] / "fastbpe.codes")
        elif self.bpe_type == "sentencepiece" or self.bpe_type == "sentence_piece":
            arg_overrides['sentencepiece_model'] = str(config.PATHS["bpe_dir"] / "spm.model")
            # arg_overrides['remove_bpe'] = "sentencepiece"
        
        if self.is_debug: print("[Info] Loading Fairseq model, task, and dictionaries...")
        
        # Load the ensemble (usually a single model here)
        self.models, self.args, self.task = checkpoint_utils.load_model_ensemble_and_task(
            [str(config.PATHS["nmt_model_dir"] / config.BUILD_PIPELINE_SETTINGS["nmt-model-dir"] / config.BUILD_PIPELINE_SETTINGS["nmt-model"])],
            arg_overrides=arg_overrides
        )
        
        # Initialize BPE processor
        if self.bpe_type == "fastbpe":
            self.bpe = encoders.build_bpe(self.args)
            if self.bpe is None: raise RuntimeError("BPE tokenizer failed to build for fastBPE.\nCheck settings of bpe and bpe_codes paths in src/config.py.")
        elif self.bpe_type == "sentencepiece" or self.bpe_type == "sentence_piece":
            self.spm_processor = spm.SentencePieceProcessor()
            self.spm_processor.load(str(config.PATHS["bpe_dir"] / "spm.model")) # type: ignore
        
        self.model = self.models[0].to(self.device).eval()
        
        if not self.task:
            raise ValueError("Failed to load Fairseq task. Check the data bin path in src/config.py.")
        if not hasattr(self.task, 'source_dictionary') or not hasattr(self.task, 'target_dictionary'):
            raise ValueError("The loaded Fairseq task does not have source or target dictionaries.")
        
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary
        self.vocab_size = len(self.tgt_dict)
        if self.is_debug: print(f"[Success] Fairseq model '{self.args.arch}' loaded.")

        # --- 2. Load Auxiliary Models for State Computation ---
        if self.is_debug: print("[Info] Loading auxiliary models for PAEC state calculation...")
        
        # Sentence Encoder for Semantic Error (S_sem)
        self.sentence_encoder = SentenceTransformer(config.MODEL_NAMES["sentence_encoder"], device=str(self.device or "cpu"))
        
        # Buffers for history-dependent metrics
        self.confidence_history = deque(maxlen=5) # For volatility calculation
        self.query_history = deque(maxlen=5)      # For stability calculation
        
        # Causal LM for Fluency Error (S_surp)
        fluency_model_name = config.MODEL_NAMES.get("fluency_scorer", "distilgpt2")
        self.lm_tokenizer = AutoTokenizer.from_pretrained(fluency_model_name)
        self.lm_model = AutoModelForCausalLM.from_pretrained(fluency_model_name, device_map=self.device)
        self.lm_model.eval()
        self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        self.fluency_loss_history = deque(maxlen=10)

        if self.is_debug: print(f"[Success] Auxiliary models (including {fluency_model_name} for fluency) loaded.")

        # --- 3. Load Dynamics Model T_theta and Policy Pi_phi (if applicable) ---
        self.dynamics_model = None       # T_theta: Predicts S_t+1
        self.offline_policy_phi = None   # Pi_phi: Fast student policy
        self.paec_dynamics_scaler = None # Scaler from T_theta training (required for normalization)
        self.paec_config_dynamics = None # Config from T_theta (dimensions, etc.)
        
        # Loading logic for PAEC components
        if dynamics_model_dir:
            if self.is_debug: print(f"[Info] Loading components (Scaler, Config) from Dynamics Model dir: {dynamics_model_dir}")
            # Load scaler and config common to both online and offline modes
            self._load_common_paec_components(dynamics_model_dir)
        else:
            # dynamics_model_dir is mandatory for Evaluation Mode to ensure correct scaling
            if self.evaluation_mode and not offline_policy_model_dir:
                raise ValueError(f"No dynamics_model_dir of `{dynamics_model_dir}` provided. Scaler is required for *online* evaluation mode.")
            elif self.is_debug:
                print(f"[Info] No dynamics_model_dir provided, but it's OK (EvalMode={self.evaluation_mode}, OfflinePolicy={offline_policy_model_dir})")

        # Load the OFFLINE Policy Network (Pi_phi) if specified
        if offline_policy_model_dir:
            if self.is_debug: print(f"[Info] Loading OFFLINE Policy Network Pi_phi from: {offline_policy_model_dir} for EvalMode: {self.evaluation_mode}")
            if dynamics_model_dir is None:
                 raise ValueError("Loading an offline policy requires dynamics_model_dir to be set (for scaler/config).")
            if PAECPolicyNetwork is None:
                raise ImportError("PAECPolicyNetwork could not be imported. Cannot load offline policy.")
            if self.paec_dynamics_scaler is None:
                raise ValueError("Scaler must be loaded (from dynamics_model_dir) before loading the offline policy.")
            self._load_offline_policy_network(offline_policy_model_dir)
        
        # Load the ONLINE Dynamics Model (T_theta) if in Evaluation Mode but NO Offline Policy (Fallback to Online Optim)
        elif self.evaluation_mode and not offline_policy_model_dir:
            if self.is_debug: print(f"[Info] Evaluation mode: No offline policy path provided. Loading ONLINE Dynamics Model T_theta from: {dynamics_model_dir} for action optimization.")
            if dynamics_model_dir is None:
                raise ValueError("Cannot perform online optimization in evaluation mode without specifying dynamics_model_dir.")
            self._load_online_dynamics_model(dynamics_model_dir)
            
        elif not self.evaluation_mode:
            if self.is_debug: print("[Info] Data generation mode: Using heuristic policies.")
        
        # --- 4. Configure Datastore ---
        self.model_hidden_size = self.args.decoder_embed_dim
        self._real_datastore_loaded = False
        if use_datastore and datastore_path:
            self._real_datastore_loaded = self._try_load_real_datastore(datastore_path)

        # --- 5. Initialize FAISS Indexes and spaCy ---
        self.gpu_res = None
        self.use_faiss_gpu = self.device.type == 'cuda' and hasattr(faiss, "StandardGpuResources")
        if self.use_faiss_gpu:
            if self.is_debug: print("[Info] faiss-gpu detected. Initializing GPU resources for FAISS...")
            self.gpu_res = faiss.StandardGpuResources() # type: ignore

        self._initialize_indexes(use_real=self._real_datastore_loaded, datastore_path=datastore_path)
        self._initialize_spacy()
        
        # --- 6. Load kNN Robustness Parameters ---
        self.adaptive_robust_params = config.ADAPTIVE_ROBUST_KNN_PARAMS
        if self.is_debug: print(f"[Info] Loaded Adaptive Robust kNN Params: {self.adaptive_robust_params}")
        
        # Final check on BPE initialization
        if self.bpe_type == "fastbpe" and self.bpe is None:
            print("[kNNMTSystem.init] FastBPE processor not initialized properly, exiting...")
        elif self.bpe_type in ["sentencepiece", "sentence_piece"] and not hasattr(self, 'spm_processor'):
            print("[kNNMTSystem.init] SentencePiece processor not initialized properly, exiting...")
        
        if self.is_debug: print("[Success] kNN-MT system initialization complete.")
    
    def _initialize_spacy(self):
        """
        Initializes spaCy models for Named Entity Recognition (NER) in German and English.
        Used for the Entity Coverage Error (S_cov) metric.
        """
        try:
            import spacy
            self.ner_de = spacy.load('de_core_news_lg')
            self.ner_en = spacy.load('en_core_web_lg')
            if self.is_debug: print("[Success] spaCy models loaded.")
        except (ImportError, OSError):
            print("[Error] spaCy models not found. Please run: python -m spacy download de_core_news_lg en_core_web_lg")
            exit(1)
            
    def translate_with_paec_control(
        self,
        source_text: str,
        pressure_computer: Union[RealtimeResourceMonitor, ProductionConstraintSimulator],
        max_length: int = 64,
        beam_size: int = 3,
        length_penalty: float = 1.0
    ) -> Tuple[Dict[str, Any], Dict[int, List[Dict[str, Any]]]]:
        """
        Performs translation using a manual beam search loop that integrates the PAEC 
        control policy at each decoding step.
        
        This method is the core loop of the system, handling:
        1. NMT Decoding.
        2. State Computation (Error, Pressure, Context).
        3. Policy Querying (Heuristic or Neural).
        4. kNN Retrieval and Probability Interpolation.
        5. Trajectory Logging.

        Args:
            source_text: The input source sentence.
            pressure_computer: Monitor (Eval) or Simulator (DataGen) for resource metrics.
            max_length: Maximum generation length.
            beam_size: Beam width for search.
            length_penalty: Length penalty for beam scoring.

        Returns:
            Tuple containing:
            - Dict: Final translation results and metrics.
            - Dict: Full trajectories for each final beam.
        """
        # --- Input Validation ---
        if self.evaluation_mode and not isinstance(pressure_computer, RealtimeResourceMonitor):
            raise ValueError("RealtimeResourceMonitor is required in evaluation mode.")
        elif not self.evaluation_mode and not isinstance(pressure_computer, ProductionConstraintSimulator):
            raise ValueError("ProductionConstraintSimulator is required in data generation mode.")

        # Ensure required components are loaded for the current mode
        if self.evaluation_mode:
            if not self.offline_policy_phi and not self.dynamics_model:
                raise RuntimeError("Evaluation mode requires either offline_policy_phi or dynamics_model to be loaded.")
            if not self.paec_dynamics_scaler:
                raise RuntimeError("Evaluation mode requires paec_dynamics_scaler to be loaded.")
            if not self.paec_config_dynamics:
                raise RuntimeError("paec_config_dynamics (from dynamics model) is not loaded in Evaluation Mode.")
        else:
            if not self.policy_classes:
                raise RuntimeError("Data generation mode requires heuristic policy_classes to be loaded.")
        if self.bpe_type == "fastbpe" and self.bpe is None:
            raise RuntimeError("BPE tokenizer is not initialized for fastBPE.")

        with torch.no_grad():
            # --- 1. Pre-computation ---
            # Encode source text using BPE
            if self.bpe_type in ["sentencepiece", "sentence_piece"]:
                spm_encoded_tokens = self.spm_processor.encode(source_text, out_type=str)   # type: ignore
                bpe_source = " ".join(spm_encoded_tokens)
            elif self.bpe_type == "fastbpe":
                if self.bpe is None: raise RuntimeError("BPE tokenizer is not initialized for fastBPE.")
                bpe_source = self.bpe.encode(source_text)
            
            # Prepare source tokens and entity data
            source_tokens = self.src_dict.encode_line(bpe_source, add_if_not_exist=False, append_eos=True).long().unsqueeze(0).to(self.device)
            precomputed_source_entities = [ent.text for ent in self.ner_de(source_text).ents]
            source_entity_spans = self._identify_source_entity_token_spans(bpe_source, precomputed_source_entities)
            
            # Encode source sentence via NMT encoder
            encoder_out = self.model.encoder(source_tokens, src_lengths=torch.tensor([source_tokens.size(1)], device=self.device))
            base_decoder = self.model.decoder.decoder if hasattr(self.model.decoder, 'decoder') else self.model.decoder
            
            # Precompute SBERT embedding for semantic error calculation
            precomputed_source_emb = self.sentence_encoder.encode([source_text], convert_to_tensor=True, show_progress_bar=False).to(self.device)

            # --- 2. Initialize Beam Search ---
            initial_token = self.tgt_dict.eos()
            self.confidence_history.clear()
            self.query_history.clear()

            # Active beams stored as {beam_id: Hypothesis}
            active_beams: Dict[int, Hypothesis] = {
                0: Hypothesis(
                    tokens=torch.tensor([initial_token], device=self.device, dtype=torch.long),
                    log_prob=0.0, hidden_state=None, context=None, incremental_state={}
                )
            }
            completed_hypotheses: List[Hypothesis] = []
            
            # --- 3. Decoding Loop ---
            for step in range(max_length):
                if not active_beams: break # Stop if no active beams left

                # List to store candidates: (candidate_hypothesis, parent_beam_id)
                step_candidates_info: List[Tuple[Hypothesis, int]] = []

                # --- Iterate through current active beams ---
                current_beam_ids = list(active_beams.keys())
                for beam_id in current_beam_ids:
                    beam = active_beams[beam_id]
                    decoder_input_tokens = beam.tokens.unsqueeze(0)

                    # --- NMT Decoder Forward Pass ---
                    try:
                        decoder_out, extra = base_decoder(
                            prev_output_tokens=decoder_input_tokens, encoder_out=encoder_out,
                            features_only=False, return_all_hiddens=True, incremental_state=beam.incremental_state
                        )
                        if 'inner_states' not in extra or not extra['inner_states']:
                            raise RuntimeError("No inner_states in decoder output")
                    except Exception as e:
                        print(f"[Error] Decoder forward pass failed at step {step} for beam {beam_id}: {e}")
                        traceback.print_exc()
                        continue # Skip this beam if decoder fails

                    # Extract context and hidden state
                    cross_attention_weights = extra.get('attn', [None])[0]
                    current_step_attention = cross_attention_weights[0, -1, :] if cross_attention_weights is not None else None
                    current_hidden_state = extra['inner_states'][-1][:, -1:, :] # Shape [1, 1, H]

                    # --- Compute System State Vectors (S_t) ---
                    # 1. Prepare inputs for state computation
                    prefix_tokens = beam.tokens[1:].tolist() if len(beam.tokens) > 1 else []
                    generated_words = self._tokens_to_words(prefix_tokens)
                    query_embedding = self.project_to_query_embedding(current_hidden_state)
                    attn_tensor = current_step_attention.to(self.device) if current_step_attention is not None else None

                    # 2. Compute Context State (H_t)
                    context_state = self.compute_context_state(
                        decoder_out=decoder_out, query_embedding=query_embedding, attention_weights=torch.Tensor(attn_tensor),
                        encoder_hidden_states=encoder_out.encoder_out.permute(1, 0, 2),
                        uncovered_entities=set(precomputed_source_entities) - beam.covered_entities,
                        source_entity_spans=source_entity_spans
                    )
                    # 3. Compute Error State (E_t)
                    error_state, new_covered_entities = self.compute_error_state(
                        source_text=source_text, generated_prefix_words=generated_words, generated_tokens=prefix_tokens,
                        source_emb=precomputed_source_emb, source_entities=precomputed_source_entities, decoder_out=decoder_out
                    )
                    current_beam_covered_entities = beam.covered_entities.union(new_covered_entities)
                    
                    # 4. Compute Pressure State (Phi_t) from Monitor/Simulator
                    pressure_state = pressure_computer.compute_pressure_vector()
            
                    # --- Determine Action (A_t) ---
                    suggested_action: Action
                    if self.evaluation_mode:
                        # --- Evaluation Mode: Use PAEC Policy (Online Optim or Offline Net) ---
                        if self.paec_config_dynamics is None:
                             raise RuntimeError("paec_config_dynamics is not loaded in Evaluation Mode.")

                        # Prepare History Tensor for Policy Network
                        history_len = self.paec_config_dynamics.get('history_len', 0)
                        S_DIM = self.paec_config_dynamics.get('S_DIM', 0)
                        if S_DIM == 0: raise ValueError("S_DIM is not set in paec_config_dynamics.")

                        S_hist_tensor = torch.zeros((1, history_len, S_DIM), device=self.device)
                        A_hist_tensor = torch.zeros((1, history_len, 6), device=self.device) # Action dim is 6

                        # Fill history from beam trajectory
                        current_beam_trajectory = beam.trajectory
                        if history_len > 0 and len(current_beam_trajectory) > 0:
                            recent_steps = current_beam_trajectory[-history_len:]
                            start_idx = history_len - len(recent_steps)
                            for i, step_data in enumerate(recent_steps):
                                if not all(k in step_data for k in ['error_state', 'pressure_state', 'context_state', 'action']):
                                    print(f"[Warning] Incomplete step_data in trajectory beam {beam_id}, hist step {i}. Skipping history.")
                                    continue
                                try:
                                    s_vec = SystemState(step_data['error_state'], step_data['pressure_state'], step_data['context_state'], 0.0).to_vector()
                                    s_df = pd.DataFrame(s_vec.reshape(1, -1), columns=t_train_Transformer.STATE_COLS_DEFAULT)
                                    if self.paec_dynamics_scaler is None: raise RuntimeError("paec_dynamics_scaler is not loaded.")
                                    s_norm = self.paec_dynamics_scaler.transform(s_df)
                                    S_hist_tensor[0, start_idx + i, :] = torch.from_numpy(s_norm.astype(np.float32)).to(self.device)
                                    A_hist_tensor[0, start_idx + i, :] = self._action_to_tensor(step_data['action'])
                                except Exception as e:
                                    print(f"[Error] Failed preparing history step {i} for beam {beam_id}: {e}")

                        # Prepare auxiliary inputs (Decoder HS, Text Emb)
                        current_decoder_hidden_state_input = None
                        if self.paec_config_dynamics.get('use_decoder_hidden_state', False):
                            if current_hidden_state is not None and current_hidden_state.dim() == 3:
                                current_decoder_hidden_state_input = current_hidden_state.squeeze(1) # [1,1,H] -> [1,H]
                            else:
                                dec_hs_dim = self.paec_config_dynamics.get('decoder_hidden_state_dim', 1024)
                                current_decoder_hidden_state_input = torch.zeros((1, dec_hs_dim), device=self.device)

                        current_prefix_emb = None
                        if self.paec_config_dynamics.get('use_text_embeddings', False):
                            if generated_words:
                                current_prefix_emb = self.sentence_encoder.encode([generated_words], convert_to_tensor=True, show_progress_bar=False).to(self.device)
                            else:
                                emb_dim = self.text_embedding_dim_T
                                current_prefix_emb = torch.zeros((1, int(emb_dim or 0)), device=self.device)

                        # Dispatch to appropriate method based on loaded models
                        suggested_action = self._get_paec_action(
                            error_state, pressure_state, context_state,
                            S_hist_tensor, A_hist_tensor, current_decoder_hidden_state_input,
                            precomputed_source_emb, current_prefix_emb
                        )
                        if self.is_debug: print(f"  [Debug] PAEC Policy suggested: {suggested_action}")

                    else:
                        # --- Data Generation Mode: Use Heuristic Policy ---
                        policy_instance, policy_name = self._sample_policy_instance(self.policy_classes, self.policies_mix)
                        suggested_action = policy_instance.decide(
                            pressure_state.to_tuple(), error_state.to_tuple(), context_state.to_tuple()
                        )
                        if self.is_debug: print(f"  [Debug] Heuristic Policy '{policy_name}' suggested: {suggested_action}")

                    # Finalize Action (Consistency Checks)
                    action = Action(
                        index_type=suggested_action.index_type,
                        k=suggested_action.k,
                        lambda_weight=suggested_action.lambda_weight
                    )
                    if action.lambda_weight < 1e-6 or action.k == 0:
                        action = Action(k=0, index_type='none', lambda_weight=0.0)
                    elif action.index_type == 'none':
                            action = Action(k=0, index_type='none', lambda_weight=0.0)
                    
                    if self.is_debug: print(f"  [Debug] Final Action: {action}")

                    # Update Simulator State (if in Data Gen mode)
                    if not self.evaluation_mode:
                        pressure_norm = pressure_state.norm()
                        if not isinstance(pressure_computer, ProductionConstraintSimulator):
                            raise ValueError("Incorrect pressure computer for data gen mode.")
                        pressure_computer.update_resource_metrics(action, pressure_norm)

                    # --- NMT + kNN Probabilities Fusion ---
                    logits = decoder_out[:, -1, :]  # [1, vocab_size]
                    nmt_log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [vocab_size]

                    if self.is_debug and step < 5 and beam_id == 0:
                        base_top5_probs, base_top5_idx = torch.topk(torch.exp(nmt_log_probs), k=5)
                        base_top5_tokens = [self.tgt_dict.symbols[idx.item()] for idx in base_top5_idx]
                        print(f"  [Debug] Final Action used: k={action.k}, type={action.index_type}, lambda={action.lambda_weight:.3f}")
                        print(f"  Beam {beam_id} - Base NMT top 5: {list(zip(base_top5_tokens, base_top5_probs.tolist()))[:5]}")

                    final_log_probs = nmt_log_probs
                    distances, values = np.array([]), np.array([])
                    
                    # Apply kNN if action dictates
                    if action.index_type != 'none' and action.k > 0:
                        # Perform Retrieval
                        distances, values, _ = self.perform_knn_retrieval(query_embedding, action)
                        
                        if self.is_debug and step < 5 and beam_id == 0:
                            retrieved_tokens = [self.tgt_dict.string([v], escape_unk=True) for v in values[:10]]
                            print(f"  [Debug] Retrieved Values (Token IDs, Top 10): {values[:10].tolist()}")
                            print(f"  [Debug] Retrieved Tokens (Top 10): {retrieved_tokens}")
                            print(f"  [Debug] Retrieved Distances (Top 10): {[f'{d:.4f}' for d in distances[:10]]}")
                            if self.tgt_dict.unk() in values:
                                print(f"  [Debug] >>> WARNING: <<unk>> ID ({self.tgt_dict.unk()}) found in retrieved values!")
                        
                        if len(values) > 0:
                            # Apply kNN Probability Distribution logic (e.g., Softmax with Temp)
                            p_knn = self._get_knn_prob_dist(distances, values, nmt_log_probs.shape)
                            
                            if self.is_debug and step < 5 and beam_id == 0:
                                knn_top5_probs, knn_top5_idx = torch.topk(p_knn, k=5)
                                knn_top5_tokens = [self.tgt_dict.symbols[idx.item()] for idx in knn_top5_idx]
                                print(f"  [Debug] p_knn Top 5 Tokens: {list(zip(knn_top5_tokens, [f'{p:.4f}' for p in knn_top5_probs.tolist()]))}")
                                unk_prob_knn = p_knn[self.tgt_dict.unk()].item()
                                print(f"  [Debug] p_knn Probability at <<unk>> (ID={self.tgt_dict.unk()}): {unk_prob_knn:.6f}")
                                if torch.isnan(p_knn).any() or torch.isinf(p_knn).any():
                                     print("  [Debug] >>> CRITICAL: NaN or Inf detected in p_knn!")

                            # Interpolate NMT and kNN distributions
                            interpolated_probs = (
                                (1.0 - action.lambda_weight) * torch.exp(nmt_log_probs) +
                                action.lambda_weight * p_knn + 1e-10
                            )
                            final_log_probs = torch.log(interpolated_probs)
                            
                            if self.is_debug and step < 5 and beam_id == 0:
                                final_top5_probs, final_top5_idx = torch.topk(final_log_probs, k=5)
                                final_top5_tokens = [self.tgt_dict.symbols[idx.item()] for idx in final_top5_idx]
                                print(f"  [Debug] Final Log Probs Top 5 Tokens: {list(zip(final_top5_tokens, [f'{p:.4f}' for p in final_top5_probs.tolist()]))}")
                                unk_prob_final = torch.exp(final_log_probs[self.tgt_dict.unk()]).item()
                                print(f"  [Debug] Final Prob at <<unk>>: {unk_prob_final:.6f}")
                                if torch.isnan(final_log_probs).any() or torch.isinf(final_log_probs).any():
                                     print("  [Debug] >>> CRITICAL: NaN or Inf detected in final_log_probs!")

                    # --- Generate Candidates ---
                    actual_k_candidates = min(beam_size, final_log_probs.size(0))
                    top_log_probs, top_indices = torch.topk(final_log_probs, k=actual_k_candidates)

                    if self.is_debug and step < 5 and beam_id == 0:
                        print(f"  Final top {actual_k_candidates} tokens: {[self.tgt_dict.symbols[idx.item()] for idx in top_indices]}")
                        print(f"  Final top {actual_k_candidates} probs: {torch.exp(top_log_probs).tolist()}")

                    # --- Log Step Trajectory ---
                    # Prepare the data dictionary that will be stored in the hypothesis trajectory
                    base_step_dict = {
                        'step': step,
                        'generated_tokens': generated_words,
                        'query_embedding': query_embedding.tolist(),
                        'error_state': error_state,
                        'pressure_state': pressure_state,
                        'context_state': context_state,
                        'action': action,
                        'decoder_hidden_state': current_hidden_state.detach().squeeze(1).cpu().numpy(), # Shape [1, H]
                        'knn_distances': distances[:5].tolist(),
                        'knn_values': values[:5].tolist()
                    }

                    # Create new hypotheses for the top-k candidates
                    for k_cand in range(actual_k_candidates):
                        token_id = top_indices[k_cand].item()
                        log_prob = top_log_probs[k_cand].item()
                        new_incremental_state = copy.deepcopy(beam.incremental_state)
                        is_skip_for_this_beam = (action.index_type == 'none' or action.k == 0)
                        
                        candidate = beam.extend(
                            token=int(token_id), log_prob=log_prob,
                            hidden_state=current_hidden_state.detach(),
                            context=None, 
                            incremental_state=new_incremental_state,
                            step_dict=base_step_dict,
                            is_skip=is_skip_for_this_beam
                        )
                        candidate.covered_entities = current_beam_covered_entities.copy()
                        step_candidates_info.append((candidate, beam_id))

                # --- Beam Pruning & Selection ---
                # Sort all candidates by length-normalized score
                ordered_candidates_info = sorted(
                    step_candidates_info,
                    key=lambda info: info[0].log_prob / (len(info[0].tokens) ** length_penalty),
                    reverse=True
                )

                # Select top candidates to form next beams
                next_beams: Dict[int, Hypothesis] = {}
                new_beam_id_counter = 0
                temp_completed_this_step = []

                for cand_info in ordered_candidates_info:
                    cand, parent_beam_id = cand_info
                    is_complete = (cand.tokens[-1].item() == self.tgt_dict.eos())

                    if is_complete:
                        temp_completed_this_step.append(cand)
                    elif new_beam_id_counter < beam_size:
                        # Assign new sequential beam ID for the next step
                        current_new_beam_id = new_beam_id_counter
                        next_beams[current_new_beam_id] = cand
                        new_beam_id_counter += 1

                    # Break if we have enough active and completed hypotheses
                    if len(next_beams) >= beam_size and len(completed_hypotheses) + len(temp_completed_this_step) >= beam_size :
                         break

                completed_hypotheses.extend(temp_completed_this_step)
                active_beams = next_beams

                # --- Early Stopping ---
                if completed_hypotheses:
                    completed_hypotheses.sort(
                        key=lambda h: h.log_prob / (len(h.tokens) ** length_penalty),
                        reverse=True
                    )
                    completed_hypotheses = completed_hypotheses[:beam_size]

                # Stop if best completed hypothesis is better than best possible active hypothesis
                if len(completed_hypotheses) >= beam_size:
                    best_completed_score = completed_hypotheses[0].log_prob / (len(completed_hypotheses[0].tokens) ** length_penalty)
                    best_active_score = -float('inf')
                    if active_beams:
                        active_scores = [active_beams[bid].log_prob / (len(active_beams[bid].tokens) ** length_penalty) for bid in active_beams]
                        if active_scores: best_active_score = max(active_scores)

            # --- 4. Final Result Compilation ---
            final_hypotheses = completed_hypotheses + list(active_beams.values())
            if not final_hypotheses:
                print("[Warning] No hypotheses generated.")
                return {'translation': "", 'knn_skip_rate': 0.0}, {}

            final_hypotheses.sort(
                key=lambda h: h.log_prob / (len(h.tokens) ** length_penalty),
                reverse=True
            )
            
            best_hypothesis = final_hypotheses[0]

            # Prepare trajectory output for all returned beams
            final_trajectory_output: Dict[int, List[Dict[str, Any]]] = {}
            num_hypotheses_to_return = min(len(final_hypotheses), beam_size)
            
            for final_beam_id in range(num_hypotheses_to_return):
                hypo = final_hypotheses[final_beam_id]
                final_trajectory_output[final_beam_id] = hypo.trajectory

            if self.is_debug:
                 print(f"[Debug] Returning {len(final_trajectory_output)} trajectories.")
                 print(f"[Debug] Best hypothesis (Beam 0) has {len(final_trajectory_output.get(0, []))} steps.")

            # Decode final tokens to string
            final_tokens = best_hypothesis.tokens.tolist()
            if final_tokens and final_tokens[0] == self.tgt_dict.eos(): final_tokens = final_tokens[1:]
            if final_tokens and final_tokens[-1] == self.tgt_dict.eos(): final_tokens = final_tokens[:-1]
            final_translation = self._tokens_to_words(final_tokens) if final_tokens else ""

            if not final_translation or len(final_translation.strip()) == 0:
                print(f"[Error] Empty translation! Source: {source_text[:50]}...")
                print(f"[Error] Final tokens: {final_tokens}")
                final_translation = "[EMPTY_TRANSLATION]"

            # Final validation of output string (checking for BPE artifacts)
            if self.is_debug:
                print(f"[Info] Translation before fixing: '{final_translation}'")
                if '@@' in final_translation or '@-@' in final_translation:
                    final_translation.replace('@@', '').replace('@-@', '')
                    print("[Error] BPE marker detected in final translation!")
                elif '<<unk>>' in final_translation:
                    final_translation.replace('<<unk>>', '')
                    print("[Warning] UNK token detected in final translation!")
                elif '&apos;s' in final_translation:
                    final_translation = final_translation.replace('&apos;s', "'s")
                    print("[Info] Fixed `&apos;s` TO `'s` in final translation.")
                else:
                    print("[Info] No BPE or UNK detected in final translation.")
                print(f"[Info] Translation before fixing: '{final_translation}'")
                
                print("-" * 20)
                print("[Debug] Final Best Hypothesis Info:")
                print(f"  Source: {source_text}")
                print(f"  Final Tokens (IDs): {final_tokens}")
                print(f"  Final Translation (before return): '{final_translation}'")
                print(f"  Best Hypo Log Prob: {best_hypothesis.log_prob:.4f}")
                print(f"  Best Hypo Length: {len(best_hypothesis.tokens)}")
                print(f"  Score (len_pen={length_penalty}): {best_hypothesis.log_prob / (len(best_hypothesis.tokens) ** length_penalty):.4f}")
                print("-" * 20)
            
            # Compute skip rate for the best hypothesis
            best_skip_rate = 0.0
            if best_hypothesis.total_steps > 0:
                best_skip_rate = best_hypothesis.skip_steps / best_hypothesis.total_steps

            return {
                'translation': final_translation,
                'knn_skip_rate': best_skip_rate,
            }, final_trajectory_output
    
    def _sample_policy_instance(self, instantiated_policies: Dict[str, Any], policies_mix: Dict[str, float]) -> Any:
        """
        Randomly selects a heuristic policy based on the defined probability mix.
        Used during Data Generation to ensure diverse training data.
        """
        names = list(policies_mix.keys())
        probs = list(policies_mix.values())
        selected_name = random.choices(names, weights=probs, k=1)[0]
        SelectedPolicyClass = instantiated_policies[selected_name]
        
        # Inject fallback policy for the Perturbator
        if "Policy_Dangerous_Perturbator" in selected_name:
            return SelectedPolicyClass(fallback_policy=Policy_Default_Balanced()), selected_name
        
        return SelectedPolicyClass(), selected_name

    def _get_knn_prob_dist(self, distances: np.ndarray, values: np.ndarray, vocab_shape: torch.Size) -> torch.Tensor:
        """
        Converts kNN distances and values into a probability distribution over the vocabulary.
        Uses Softmax with a configurable temperature.
        """
        dists_tensor = torch.from_numpy(distances).to(self.device)
        
        # Retrieve temperature from config (Adaptive Robust Params)
        temperature = self.adaptive_robust_params.get("KNNSW_TEMPERATURE", 50.0) if self.adaptive_robust_params.get("KNNSW_ENABLED", True) else 1.0
        
        # Softmax over negative distances (closer is higher probability)
        probs_knn = F.softmax(-dists_tensor / temperature, dim=0)
        
        # Map probabilities to vocabulary indices
        p_knn = torch.zeros(vocab_shape, device=self.device)
        p_knn.scatter_add_(0, torch.from_numpy(values).to(self.device, dtype=torch.long), probs_knn)
        return p_knn

    def _tokens_to_words(self, tokens: List[int]) -> str:
        """
        Decodes a list of token IDs back into a string string using the appropriate BPE processor.
        """
        if not tokens: return ""
        
        if self.is_debug:
            print(f"  [Debug][_tokens_to_words] Input Tokens IDs (first 10): {tokens[:10]}")
        
        try:
            if self.bpe_type == "sentencepiece" or self.bpe_type == "sentence_piece":
                eos_id = self.tgt_dict.eos()
                decoded = self.spm_processor.decode([t for t in tokens if t != eos_id]).strip() # type: ignore
            else:
                # FastBPE / Standard BPE decoding
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
                        print(f"[Warning] self.bpe.decode failed: {e}. Falling back to regex.")
                        decoded = re.sub(r'@@\s?|@-@\s?', '', bpe_string).strip()
                else:
                    decoded = re.sub(r'@@\s?|@-@\s?', '', bpe_string).strip()
        except Exception as e:
            traceback.print_exc()
            raise e
        
        if self.is_debug:
            print(f"  [Debug][_tokens_to_words] Final decoded string: '{decoded[:100]}...'")
            if '@@' in decoded or '@-@' in decoded:
                print("  [Debug][_tokens_to_words] WARNING: Residual BPE marker detected in final output!")

        return decoded
    
    def _load_common_paec_components(self, dynamics_model_dir_path: str):
        """
        Loads shared PAEC components (Scaler, Config) from the Dynamics Model directory.
        Required for both Online Optimization and Offline Policy usage to ensure
        consistent state normalization.
        """
        model_dir = Path(dynamics_model_dir_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Dynamics model directory not found at: {model_dir}")

        # Load T_theta Config
        config_path = model_dir / 'config.json' 
        if not config_path.exists():
            raise FileNotFoundError(f"Dynamics model config.json not found in {model_dir}")
        with open(config_path, 'r') as f:
            self.paec_config_dynamics = json.load(f)

        if self.is_debug:
            print("[Info] Setting global dimensions based on Dynamics Model config...")
            print(f"[Debug] Loaded T_theta config: {self.paec_config_dynamics}")

        try:
            # Set global variables in training module to match loaded config
            S_DIM = self.paec_config_dynamics['S_DIM']
            E_DIM = self.paec_config_dynamics['E_DIM']
            PHI_DIM = self.paec_config_dynamics['PHI_DIM']
            H_DIM = self.paec_config_dynamics['H_DIM']
            E_INDEX = tuple(self.paec_config_dynamics['E_INDEX'])
            PHI_INDEX = tuple(self.paec_config_dynamics['PHI_INDEX'])
            H_INDEX = tuple(self.paec_config_dynamics['H_INDEX'])
            STATE_COLS_DEFAULT = self.paec_config_dynamics['STATE_COLS_DEFAULT']

            t_train_Transformer.S_DIM = S_DIM
            t_train_Transformer.E_DIM = E_DIM
            t_train_Transformer.PHI_DIM = PHI_DIM
            t_train_Transformer.H_DIM = H_DIM
            t_train_Transformer.E_INDEX = E_INDEX
            t_train_Transformer.PHI_INDEX = PHI_INDEX
            t_train_Transformer.H_INDEX = H_INDEX
            t_train_Transformer.STATE_COLS_DEFAULT = STATE_COLS_DEFAULT

            print(f"[Debug] Set global dimensions - S_DIM: {S_DIM}, E_DIM: {E_DIM}, H_DIM: {H_DIM}")

        except KeyError as e:
            raise KeyError(f"Required dimension key {e} not found in dynamics model config {config_path}") from e

        # Load Scaler
        scaler_path = model_dir / "scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        try:
            self.paec_dynamics_scaler = joblib.load(scaler_path)
            if self.is_debug: print("[Success] Scaler loaded successfully.")
        except ImportError as ie:
            print("[Error] Could not import InvertableColumnTransformer needed to load the scaler.")
            traceback.print_exc()
            raise ie
        except Exception as e:
            print(f"[Error] Failed to load scaler from {scaler_path}: {e}")
            traceback.print_exc()
            raise e

        # Infer text embedding dimension if needed
        self.text_embedding_dim_T = 0
        if self.paec_config_dynamics.get('use_text_embeddings', False):
            dim_from_config = self.paec_config_dynamics.get('text_embedding_dim', 0)
            if dim_from_config > 0:
                self.text_embedding_dim_T = int(dim_from_config)
            else:
                if hasattr(self, 'sentence_encoder'):
                    try:
                        self.text_embedding_dim_T = self.sentence_encoder.get_sentence_embedding_dimension()
                        if self.is_debug: print(f"[Info] Inferred text_embedding_dim for T_theta as {self.text_embedding_dim_T}")
                    except Exception as e:
                        print(f"Failed to infer text_embedding_dim: {e}")
                        traceback.print_exc()
                        raise e
                else:
                    raise RuntimeError("Cannot infer text_embedding_dim as sentence_encoder is not yet loaded.")
    
    def _load_offline_policy_network(self, policy_model_dir_path: str):
        """
        Loads the pre-trained Offline Policy Network (Pi_phi) for fast inference.
        """
        global ONE_PRINT
        if ONE_PRINT:
            ONE_PRINT = False
            print("[Warning] Using offline policy network Pi_phi for action selection. Ensure that the dynamics scaler is compatible with the policy's training data scaler.")
        model_dir = Path(policy_model_dir_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Offline policy model directory not found at: {model_dir}")
        if PAECPolicyNetwork is None:
            raise RuntimeError("PAECPolicyNetwork class is not available.")

        # Load Policy Config
        config_path = model_dir / 'pi_phi_config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Offline policy config pi_phi_config.json not found in {model_dir}")
        with open(config_path, 'r') as f:
            policy_config = json.load(f)

        # Instantiate Policy Network
        try:
            if not self.paec_config_dynamics:
                raise RuntimeError("Dynamics model config (T_theta) must be loaded before instantiating the offline policy.")
            
            # Dimensions from T_theta config
            s_dim_T = self.paec_config_dynamics['S_DIM']
            use_text_T = self.paec_config_dynamics.get('use_text_embeddings', False)
            text_emb_dim_T = self.text_embedding_dim_T
            use_dec_hs_T = self.paec_config_dynamics.get('use_decoder_hidden_state', False)
            dec_hs_dim_T = self.paec_config_dynamics.get('decoder_hidden_state_dim', 0) if use_dec_hs_T else 0

            # Architecture params from Policy config
            policy_hid_dim = policy_config.get('hid_dim', 64)
            policy_layers = policy_config.get('layers', 3)
            policy_nhead = policy_config.get('nhead', 4)
            policy_history_len = policy_config.get('history_len', 4)

            if policy_history_len != self.paec_config_dynamics.get('history_len', 4):
                print(f"[Warning] History length mismatch: Policy trained with {policy_history_len}, Dynamics data used {self.paec_config_dynamics.get('history_len', 4)}. Using policy's length.")

            student_model_args = {
                'S_DIM': s_dim_T,
                'ACTION_DIM': 6,
                'hid_dim': policy_hid_dim,
                'nhead': policy_nhead,
                'layers': policy_layers,
                'history_len': policy_history_len,
                'use_text_embeddings': use_text_T,
                'text_embedding_dim': text_emb_dim_T,
                'use_decoder_hidden_state': use_dec_hs_T,
                'decoder_hidden_state_dim': dec_hs_dim_T
            }
            if self.is_debug: print(f"[Debug] Instantiating PAECPolicyNetwork with args: {student_model_args}")

            self.offline_policy_phi = PAECPolicyNetwork(**student_model_args).to(self.device)

        except KeyError as e:
            raise KeyError(f"Missing key {e} in policy config '{config_path}' or dynamics config needed for policy instantiation.") from e
        except Exception as e:
            print(f"[Error] Failed to instantiate PAECPolicyNetwork: {e}")
            raise

        # Load Policy Weights
        weights_path = model_dir / "pi_phi_best.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Offline policy weights file not found at {weights_path}")

        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.offline_policy_phi.load_state_dict(state_dict)
            self.offline_policy_phi.eval()
            if self.is_debug: print("[Success] Offline policy network Pi_phi loaded successfully.")
        except Exception as e:
            print(f"[Error] Failed to load offline policy weights from {weights_path}: {e}")
            raise e
    
    def _load_online_dynamics_model(self, dynamics_model_dir_path: str):
        """
        Loads the Dynamics Model (T_theta) for Online Optimization.
        Used only if no offline policy is provided. Requires 'checkpoint_best.pt' and P matrix.
        """
        model_dir = Path(dynamics_model_dir_path)
        if not self.paec_config_dynamics:
            raise RuntimeError("Dynamics model config (T_theta) must be loaded before loading the model itself.")

        # Instantiate T_theta
        try:
            use_text_T = self.paec_config_dynamics.get('use_text_embeddings', False)
            text_emb_dim_T = self.text_embedding_dim_T
            use_dec_hs_T = self.paec_config_dynamics.get('use_decoder_hidden_state', False)
            dec_hs_dim_T = self.paec_config_dynamics.get('decoder_hidden_state_dim', 0) if use_dec_hs_T else 0

            model_args = {
                'action_dim': self.paec_config_dynamics.get('action_dim', 6),
                'hid_dim': self.paec_config_dynamics.get('hid_dim', 64),
                'layers': self.paec_config_dynamics.get('layers', 3),
                'history_len': self.paec_config_dynamics.get('history_len', 4),
                'predict_delta': self.paec_config_dynamics.get('predict_delta', False),
                'use_text_embeddings': use_text_T,
                'text_embedding_dim': text_emb_dim_T,
                'use_decoder_hidden_state': use_dec_hs_T,
                'decoder_hidden_state_dim': dec_hs_dim_T,
                'use_separate_heads_eh': self.paec_config_dynamics.get('use_separate_heads_eh', False),
                'use_multi_heads': self.paec_config_dynamics.get('use_multi_heads', False),
                'use_spectral_norm': self.paec_config_dynamics.get('use_spectral_norm', False),
                'nhead': self.paec_config_dynamics.get('nhead', 4)
            }
            if self.is_debug: print(f"[Debug] Instantiating PAECTransition (T_theta) with args: {model_args}")

            self.dynamics_model = t_train_Transformer.PAECTransition(**model_args).to(self.device)

        except KeyError as e:
            raise KeyError(f"Missing key {e} in dynamics model config needed for T_theta instantiation.") from e
        except Exception as e:
            print(f"[Error] Failed to instantiate PAECTransition (T_theta): {e}")
            raise

        # Load T_theta Weights and P Matrix
        checkpoint_path = model_dir / "checkpoint_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Dynamics model checkpoint not found at {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint['model']

            if list(state_dict.keys())[0].startswith('_orig_mod.'):
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

            model_state_dict = self.dynamics_model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
            missing_keys, unexpected_keys = self.dynamics_model.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys: print(f"[Warning] Missing keys when loading T_theta: {missing_keys}")
            if unexpected_keys: print(f"[Warning] Unexpected keys when loading T_theta: {unexpected_keys}")

            self.dynamics_model.eval()

            # Load P matrix (Lyapunov weights)
            if 'P' in checkpoint and checkpoint['P'] is not None:
                self.paec_P_values = torch.tensor(checkpoint['P'], dtype=torch.float32, device=self.device)
            elif 'P_init' in self.paec_config_dynamics:
                p_init_values = self.paec_config_dynamics['P_init']
                if len(p_init_values) != t_train_Transformer.E_DIM:
                    raise ValueError(f"P_init in config has wrong dimension: {len(p_init_values)} vs E_DIM {t_train_Transformer.E_DIM}")
                self.paec_P_values = torch.tensor(p_init_values, dtype=torch.float32, device=self.device)
            else:
                raise ValueError("Could not find P matrix in checkpoint or P_init in T_theta config.")

            if self.is_debug: print("[Success] Online Dynamics Model T_theta and P matrix loaded successfully.")

        except Exception as e:
            print(f"[Error] Failed to load T_theta weights or P matrix from {checkpoint_path}: {e}")
            raise
    
    def _parse_dtype_from_string(self, dtype_str: str) -> np.dtype:
        """Helper to parse numpy dtypes from string representations in config files."""
        match = re.search(r"\'(?:numpy\.)?(.*?)\'", dtype_str)
        if match:
            return np.dtype(match.group(1))
        else:
            return np.dtype(dtype_str)
    
    def _try_load_real_datastore(self, path: str) -> bool:
        """
        Attempts to load the kNN Datastore (keys and values) from disk via memory mapping.
        Verifies existence of all index subdirectories.
        """
        try:
            ds_path = Path(path)
            # Check for enabled index types
            for subdir in config.ENABLED_INDICES:
                config_path = ds_path / subdir / "config.json"
                keys_path = ds_path / subdir/ "keys.npy"
                vals_path = ds_path / subdir / "vals.npy"
           
                if not (ds_path / subdir).exists():
                    raise FileNotFoundError(f"Datastore subdirectory '{subdir}' not found in {ds_path}. Please ensure the datastore is built correctly.")

                required_files = [config_path, keys_path, vals_path]
                if not all(p.exists() for p in required_files):
                    missing = [p.name for p in required_files if not p.exists()]
                    raise FileNotFoundError(f"Missing required files in {ds_path / subdir}: {', '.join(missing)}. Please ensure the datastore is built correctly.")
        
            # Load metadata from one config (assuming consistent across indices)
            config_path = ds_path / config.ENABLED_INDICES[0] / "config.json"
            keys_path = ds_path / config.ENABLED_INDICES[0] / "keys.npy"
            vals_path = ds_path / config.ENABLED_INDICES[0] / "vals.npy"

            with open(config_path, 'r') as f:
                ds_config = json.load(f)
        
            if "data_infos" not in ds_config or "keys" not in ds_config["data_infos"] or "vals" not in ds_config["data_infos"]:
                raise ValueError("config.json is missing the required 'data_infos' structure.")
            
            key_info = ds_config["data_infos"]["keys"]
            val_info = ds_config["data_infos"]["vals"]
            
            key_dtype = self._parse_dtype_from_string(key_info["dtype"])
            val_dtype = self._parse_dtype_from_string(val_info["dtype"])

            key_shape = tuple(key_info["shape"])
            val_shape = tuple(val_info["shape"])

            if len(key_shape) != 2 or len(val_shape) != 1:
                raise ValueError("Invalid shapes in config.json. Keys should be 2D and Vals should be 1D.")
             
            self.datastore_size, self.embedding_dim = key_shape
            
            if self.datastore_size != val_shape[0]:
                raise ValueError(f"Inconsistent datastore size in config.json: keys have {self.datastore_size} but vals have {val_shape[0]}.")

            if self.is_debug: print("[Info] Loading 'keys.npy' using memory mapping with metadata from config.json...")
            self.datastore_embeddings = np.memmap(keys_path, dtype=key_dtype, mode='r', shape=key_shape)
            
            if self.is_debug: print("[Info] Loading 'vals.npy' using memory mapping with metadata from config.json...")
            self.datastore_values = np.memmap(vals_path, dtype=val_dtype, mode='r', shape=val_shape)

            if self.is_debug: print(f"[Success] Loaded real datastore: {self.datastore_size} entries, dim={self.embedding_dim}")
            return True
        
        except Exception as e:
            print(f"[Error] Failed to load real datastore: {str(e)}")
            traceback.print_exc()
            exit(1)
    
    def _initialize_indexes(self, use_real: bool, datastore_path: Optional[str] = None):
        """
        Loads the FAISS indexes (Exact, HNSW, IVF_PQ) from disk.
        Moves supported indexes to GPU if available.
        """
        if self.is_debug: print("[Info] Initializing multiple FAISS indexes...")
        
        if not (use_real and hasattr(self, 'datastore_embeddings') and self.datastore_embeddings is not None):
            print("[Error] Real datastore mode selected but embeddings not loaded.")
            exit(1)

        self.datastore_embeddings = np.ascontiguousarray(self.datastore_embeddings, dtype='float32')

        for index_type in config.ENABLED_INDICES:
            index_path = Path(datastore_path) / index_type / 'keys.faiss_index' if datastore_path else None
            
            if index_path and index_path.exists():
                if self.is_debug: print(f"\t- Loading pre-built '{index_type}' index from {index_path}...")
                cpu_index = faiss.read_index(str(index_path))
                index = cpu_index
            else:
                raise FileNotFoundError(f"Index file {index_type}.faiss_index not exist in {index_path}.")
            
            if self.use_faiss_gpu:
                if index_type in ['exact', 'ivf_pq']:
                    if self.is_debug: print(f"\t- Moving '{index_type}' index to GPU...")
                    co = faiss.GpuClonerOptions()   # type: ignore
                    co.useFloat16 = True
                    gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, self.device.index or 0, index, co) # type: ignore
                    setattr(self, f"{index_type}_index", gpu_index)
                    if self.is_debug: print(f"\t- Index '{index_type}' is ready on GPU with {gpu_index.ntotal} entries.")
                else:
                    if self.is_debug: print(f"\t- Index type '{index_type}' is not supported on GPU, keeping on CPU.")
                    setattr(self, f"{index_type}_index", index)
                    if self.is_debug: print(f"\t- Index '{index_type}' is ready on CPU with {index.ntotal} entries.")
            else:
                setattr(self, f"{index_type}_index", index)
                if self.is_debug: print(f"\t- Index '{index_type}' is ready on CPU with {index.ntotal} entries.")

    def project_to_query_embedding(self, hidden_state: torch.Tensor) -> np.ndarray:
        """
        Projects the NMT decoder's hidden state to the kNN query embedding space.
        Assumes the datastore keys are directly the hidden states (identity projection).
        """
        with torch.no_grad():
            last_hidden_vec = hidden_state[0, -1, :] if len(hidden_state.shape) == 3 else hidden_state
            query_embedding = last_hidden_vec.cpu().numpy()
            return query_embedding.astype('float32')

    def _align_entities(
        self,
        source_entities: List[str],
        generated_entities: List[str]
    ) -> Tuple[float, Set[str]]:
        """
        Matches source entities to generated entities to calculate Coverage Error.
        
        Uses a two-phase approach:
        1. Exact string matching (case-insensitive).
        2. Semantic similarity matching using Hungarian algorithm (Linear Sum Assignment)
           for remaining entities.

        Returns:
            Tuple[float, Set[str]]: Alignment score and set of covered source entity strings.
        """
        if not source_entities or not generated_entities:
            return 0.0, set()

        soft_alignment_score = 0.0
        covered_source_entities = set()

        # Phase 1: Exact Matching
        remaining_src = list(source_entities)
        remaining_gen = list(generated_entities)
        
        src_indices_to_remove = []
        gen_indices_to_remove = []

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

        for i in sorted(src_indices_to_remove, reverse=True):
            del remaining_src[i]
        for j in sorted(gen_indices_to_remove, reverse=True):
            del remaining_gen[j]

        # Phase 2: Semantic Matching
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
        """
        Maps source entities to their token index spans in the BPE-tokenized source sequence.
        Needed for 'Attention Focus' calculation.
        """
        if not source_entities: return {}
        
        bpe_source_tokens = bpe_source_text.split()
        entity_spans = defaultdict(list)
        
        for entity in sorted(source_entities, key=len, reverse=True):
            bpe_entity = ""
            
            if self.bpe_type == "fastbpe":
                if self.bpe is None: raise ValueError(f"[Error] self.bpe ERROR: {self.bpe}")
                try:
                    bpe_entity = self.bpe.encode(entity).split()
                except Exception as e:
                    raise RuntimeError(f"[Error] fastbpe.encode failed for entity '{entity}': {e}")
            
            elif self.bpe_type in ["sentencepiece", "sentence_piece"]:
                try:
                    bpe_entity = self.spm_processor.encode(entity, out_type=str) # type: ignore
                except Exception as e:
                    raise RuntimeError(f"[Error] sentencepiece processing failed for entity '{entity}': {e}")
            else:
                raise ValueError(f"[Error] Unsupported BPE rule: {self.bpe_type}")
            
            len_ent = len(bpe_entity)
            for i in range(len(bpe_source_tokens) - len_ent + 1):
                if bpe_source_tokens[i:i+len_ent] == bpe_entity:
                    entity_spans[entity].append((i, i + len_ent))
        
        return dict(entity_spans)

    def compute_error_state(
        self,
        source_text: str,
        generated_prefix_words: str,
        generated_tokens: List[int], # List of token IDs
        source_emb: torch.Tensor,
        source_entities: List[str],
        decoder_out: torch.Tensor
    ) -> Tuple[ErrorStateVector, Set[str]]:
        """
        Computes the Error State Vector E_t (Semantic Drift, Coverage, Surprisal, Repetition).

        Returns:
            Tuple[ErrorStateVector, Set[str]]: Computed error vector and set of currently covered entities.
        """
        generated_text = ""
        if isinstance(generated_prefix_words, list) and all(isinstance(item, str) for item in generated_prefix_words):
             generated_text = " ".join(generated_prefix_words)
        else:
            generated_text = generated_prefix_words

        # 1. Semantic Drift Error
        try:
            if source_emb is None: source_emb = self.sentence_encoder.encode([source_text], convert_to_tensor=True, show_progress_bar=False)
            generated_emb = self.sentence_encoder.encode([generated_text], convert_to_tensor=True, show_progress_bar=False)
            cos_sim = F.cosine_similarity(source_emb, generated_emb).item()
            semantic_drift = np.clip(1.0 - cos_sim, 0.0, 2.0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("[Error] Sematic Calculation ERROR")
            exit(1)
            
        # 2. Entity Coverage Error
        generated_entities = [ent.text for ent in self.ner_en(generated_prefix_words).ents]
        _, covered_entities_set = self._align_entities(source_entities, generated_entities)
        coverage = len(covered_entities_set) / len(source_entities) if source_entities else 1.0
        error_coverage = 1.0 - coverage

        # 3. Fluency: Surprisal (Entropy of prediction)
        surprisal = 0.0
        logits = decoder_out[:, -1, :].squeeze(0)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9)).item()
        max_entropy = math.log2(self.vocab_size)
        if max_entropy > 0: surprisal = np.clip(entropy / max_entropy, 0.0, 1.0)

        # 4. Fluency: Repetition (Semantic Self-Similarity)
        repetition = 0.0
        window_size = 20
        if len(generated_tokens) > 1:
            window = generated_tokens[-window_size:]
            token_tensor = torch.tensor(window, dtype=torch.long, device=self.device)
            # Retrieve internal embeddings to measure semantic similarity of output
            embeddings = self.model.decoder.embed_tokens(token_tensor)
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

    def compute_context_state(
        self,
        decoder_out: torch.Tensor,
        query_embedding: np.ndarray,
        attention_weights: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        uncovered_entities: Set[str],
        source_entity_spans: Dict[str, List[Tuple[int, int]]],
    ) -> GenerativeContextVector:
        """
        Computes the Context State Vector H_t (Focus, Consistency, Stability, Volatility).
        """

        logits = decoder_out[:, -1, :].squeeze(0)

        # 1. Attention Focus
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

        # 2. Consistency (between Query and Attention Context)
        consistency = 0.0
        try:
            query_tensor = torch.from_numpy(query_embedding).to(self.device).squeeze()

            if attention_weights is not None and encoder_hidden_states is not None:
                # Calculate attention-weighted source context
                attention_context_vector = torch.bmm(
                    attention_weights.unsqueeze(0).unsqueeze(0), 
                    encoder_hidden_states
                ).squeeze(0).squeeze(0)

                attention_context_vector_norm = torch.nn.functional.normalize(attention_context_vector, p=2, dim=0)
                consistency = F.cosine_similarity(query_tensor, attention_context_vector_norm, dim=0).item()
                consistency = np.clip((consistency + 1.0) / 2.0, 0.0, 1.0)

        except Exception as e:
            raise Exception(f"[Warning] Differentiable Consistency calculation failed: {e}")

        # 3. Stability (Similarity to recent queries)
        context_stability = 0.0
        try:
            if 'query_tensor' not in locals():
                query_tensor = torch.from_numpy(query_embedding).to(self.device).squeeze()

            if len(self.query_history) > 0:
                history_tensors = torch.stack(list(self.query_history), dim=0)
                similarities = F.cosine_similarity(query_tensor.unsqueeze(0), history_tensors, dim=1)
                context_stability = similarities.mean().item()
                context_stability = np.clip((context_stability + 1.0) / 2.0, 0.0, 1.0)

            self.query_history.append(query_tensor)

        except Exception as e:
            raise Exception(f"[Warning] Clarity calculation failed: {e}")

        # 4. Volatility (Std Dev of recent confidence scores)
        volatility = 0.0
        try:
            top_k = 10
            top_k_logits, _ = torch.topk(logits, k=top_k)
            margin_k = torch.mean(top_k_logits[0] - top_k_logits[1:]) if top_k > 1 else 10.0
            current_confidence = torch.sigmoid(0.5 * torch.Tensor(margin_k)).item()
            
            self.confidence_history.append(current_confidence)
            if len(self.confidence_history) > 1:
                volatility = np.sqrt(np.var(self.confidence_history) + 1e-6)
        except Exception as e:
            if self.is_debug: print(f"[Warning] Volatility calculation failed: {e}")

        context_vector = GenerativeContextVector(
            context_faith_focus=faith_focus,
            context_consistency=consistency,
            context_stability=context_stability,
            context_confidence_volatility=float(volatility)
        )
        return context_vector

    def perform_knn_retrieval(self, query_embedding: np.ndarray, action: Action) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executes kNN retrieval using the specified index type and k.
        
        Args:
            query_embedding: The query vector.
            action: The action determining index type and k.
            
        Returns:
            Tuple[distances, token_ids, time_taken]
        """
        if action.k == 0:
            return np.array([]), np.array([]), 0.0

        start_time = time.time()
        
        index_map = {}
        for i in config.ENABLED_INDICES:
            if hasattr(self, f"{i}_index"):
                index_map[i] = getattr(self, f"{i}_index")
        index = index_map.get(action.index_type, getattr(self, f"{config.DEFAULT_INDEX}_index"))

        try:
            distances, indices = index.search(np.asarray(query_embedding).reshape(1, -1), int(action.k))
            retrieved_values = self.datastore_values[indices[0]]
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("[Error] kNN retrieval ERROR")
            exit(1)

        retrieval_time = time.time() - start_time
        return distances[0], retrieved_values, retrieval_time

    def _action_to_tensor(self, action: Action) -> torch.Tensor:
        """Encodes an Action object into a 6D tensor for model input."""
        action_tensor = torch.zeros(1, 6, device=self.device)
        index_map = {'none': 0, 'exact': 1, 'hnsw': 2, 'ivf_pq': 3}
        if action.index_type in index_map:
            action_tensor[0, index_map[action.index_type]] = 1.0

        action_tensor[0, 4] = action.k / float(config.KNN_MAX_K)
        action_tensor[0, 5] = action.lambda_weight
        return action_tensor

    def _get_paec_action(
        self,
        error_state: ErrorStateVector,
        pressure_state: ResourcePressureVector,
        context_state: GenerativeContextVector,
        S_hist: torch.Tensor,
        A_hist: torch.Tensor,
        decoder_hidden_state: Optional[torch.Tensor] = None,
        source_embedding: Optional[torch.Tensor] = None,
        prefix_embedding: Optional[torch.Tensor] = None
    ) -> Action:
        """
        Dispatcher for PAEC Policy in Evaluation Mode.
        Implements the Hybrid Policy logic:
        1.  **Safety Valve**: Checks hard resource constraints. Returns 'none' if violated.
        2.  **Quality Optimizer**: If safe, queries the policy (Offline Pi_phi or Online T_theta)
            to find the best retrieval configuration (k>0).
        """
        if not self.evaluation_mode:
            return Action(k=0, index_type='none', lambda_weight=0.0)

        # --- 1. Resource Safety Valve ---
        latency_p, memory_p, throughput_p = pressure_state.to_tuple()
        
        if self.is_debug: print(f"[Debug] Pressure: latency={latency_p:.4f}, memory={memory_p:.4f}, throughput={throughput_p:.4f}")
        
        if memory_p > config.ADAPTIVE_ROBUST_KNN_PARAMS["KNN_PRESSURE_THRESHOLD"].get("memory") or \
           latency_p > config.ADAPTIVE_ROBUST_KNN_PARAMS["KNN_PRESSURE_THRESHOLD"].get("latency") or \
           throughput_p > config.ADAPTIVE_ROBUST_KNN_PARAMS["KNN_PRESSURE_THRESHOLD"].get("throughput"):
            if self.is_debug: print("  [Debug] Hybrid Policy: Resource Safety Valve ACTIVE. Forcing k=0.")
            return Action(k=0, index_type='none', lambda_weight=0.0)
        
        # --- 2. kNN Quality Optimizer ---
        if self.is_debug: print("  [Debug] Hybrid Policy: Resource Safety Valve PASSED. Evaluating k>0 options.")

        if self.offline_policy_phi:
            if self.is_debug: print("  Using Offline Policy (Pi_phi) for action.")
            return self._get_offline_policy_action(
                error_state, pressure_state, context_state,
                S_hist, A_hist, decoder_hidden_state,
                source_embedding, prefix_embedding
            )
        elif self.dynamics_model:
            if self.is_debug: print("  Using Online Optimization (T_theta) for action.")
            return self._get_paec_action_online_optim(
                error_state, pressure_state, context_state,
                S_hist, A_hist, decoder_hidden_state,
                source_embedding, prefix_embedding,
                force_knn_active=True
            )
        raise RuntimeError("No policy model (offline Pi_phi or online T_theta) loaded for evaluation mode.")

    def _get_paec_action_online_optim(
        self,
        error_state: ErrorStateVector,
        pressure_state: ResourcePressureVector,
        context_state: GenerativeContextVector,
        S_hist: torch.Tensor,
        A_hist: torch.Tensor,
        decoder_hidden_state: Optional[torch.Tensor] = None,
        source_embedding: Optional[torch.Tensor] = None,
        prefix_embedding: Optional[torch.Tensor] = None,
        force_knn_active: bool = False
    ) -> Action:
        """
        Performs Online Optimization using T_theta.
        Uses gradient descent to find the action (k, lambda) that minimizes the
        predicted Lyapunov value V(S_t+1) for each index type, then selects the global best.
        """
        if not all([self.dynamics_model, self.paec_dynamics_scaler, self.paec_P_values is not None]):
            print("[Error] Online optimization requires dynamics_model, scaler, and P_values.")
            return Action(k=0, index_type='none', lambda_weight=0.0)

        # Normalize state
        system_state = SystemState(error_state, pressure_state, context_state, timestamp=0.0)
        state_vec = system_state.to_vector().reshape(1, -1)
        if self.is_debug:
            print_txt = ', '.join([f"{v:.4f}" for v in state_vec.flatten().tolist()])
            print(f"  [Debug] Raw state_vec (before scaling): [{print_txt}]")
        state_df = pd.DataFrame(state_vec, columns=t_train_Transformer.STATE_COLS_DEFAULT)
        if self.paec_dynamics_scaler is None:
            raise RuntimeError("Dynamics scaler not loaded for online optimization.")
        state_norm = self.paec_dynamics_scaler.transform(state_df)
        state_tensor = torch.from_numpy(state_norm.astype(np.float32)).to(self.device)

        P_diag = self.paec_P_values

        # Optimization config
        optim_config = config.OPTIM_STEPS_CONFIG if hasattr(config, 'OPTIM_STEPS_CONFIG') else {}
        optim_steps = optim_config.get("max_steps", 10)
        learning_rate = 0.01
        patience = optim_config.get("patience", 1)
        tolerance = optim_config.get("tolerance", 1e-4)

        candidate_actions = {}
        candidate_costs = {}

        indices_to_optimize = [idx for idx in config.DEFAULT_INDEX if idx != 'none']

        if not indices_to_optimize:
            return Action(k=0, index_type='none', lambda_weight=0.0)

        with torch.enable_grad():
            for index_type in indices_to_optimize:
                k_norm_param = torch.tensor([8.0 / 16.0], device=self.device, requires_grad=True)
                lambda_param = torch.tensor([0.5], device=self.device, requires_grad=True)
                optimizer = torch.optim.Adam([k_norm_param, lambda_param], lr=learning_rate)

                index_type_tensor = self._action_to_tensor(Action(index_type=index_type, k=0, lambda_weight=0.0))[:, :4]

                best_loss_this_index = torch.tensor(float('inf'), device=self.device)
                best_action_tensor_this_index = torch.cat([
                     index_type_tensor,
                     k_norm_param.unsqueeze(0).detach().clone(),
                     lambda_param.unsqueeze(0).detach().clone()
                 ], dim=1)
                patience_counter = 0

                for step in range(optim_steps):
                    optimizer.zero_grad()
                    with torch.no_grad():
                        k_norm_param.clamp_(0.0, 1.0)
                        lambda_param.clamp_(0.0, 1.0)

                    action_tensor = torch.cat([
                        index_type_tensor,
                        k_norm_param.unsqueeze(0),
                        lambda_param.unsqueeze(0)
                    ], dim=1)

                    model_kwargs = {}
                    if self.paec_config_dynamics is None: self.paec_config_dynamics = {}
                    if self.paec_config_dynamics.get('use_decoder_hidden_state', False):
                        model_kwargs['decoder_hidden_state'] = decoder_hidden_state
                    if self.paec_config_dynamics.get('use_text_embeddings', False):
                        model_kwargs['source_embeddings'] = source_embedding
                        model_kwargs['prefix_embeddings'] = prefix_embedding
                        if model_kwargs['source_embeddings'] is None or model_kwargs['prefix_embeddings'] is None:
                            raise ValueError("_get_paec_action_online_optim requires text embeddings but they were not provided/computed.")

                    try:
                        if self.dynamics_model is None:
                            raise RuntimeError("Dynamics model is not loaded for online optimization.")
                        E_pred, H_pred, _ = self.dynamics_model(
                            state_tensor, S_hist, action_tensor, A_hist, **model_kwargs
                        )
                    except Exception as e:
                         print(f"[Error] OnlineOptim: Dynamics model forward pass failed for {index_type} at step {step}: {e}")
                         traceback.print_exc()
                         loss = torch.tensor(float('inf'), device=self.device)
                         break

                    _, Phi_t, _ = t_train_Transformer.split_state(state_tensor)
                    S_next_pred = torch.cat([E_pred, Phi_t, H_pred], dim=1)
                    loss = t_train_Transformer.lyapunov_V(S_next_pred, P_diag)

                    if torch.isnan(loss):
                         if self.is_debug: print(f"[Debug] OnlineOptim: NaN loss detected for {index_type} at step {step}. Stopping opt.")
                         loss = torch.tensor(float('inf'), device=self.device)
                         break

                    loss.backward()
                    optimizer.step()

                    # Early Stopping
                    current_loss_item = loss.item()
                    if best_loss_this_index.item() - current_loss_item > tolerance:
                        best_loss_this_index = loss.detach()
                        best_action_tensor_this_index = action_tensor.detach().clone()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                         if self.is_debug: print(f"[Debug] OnlineOptim: Patience exceeded for {index_type} at step {step}.")
                         break

                final_v = best_loss_this_index.item() if not torch.isinf(best_loss_this_index) else float('inf')
                candidate_actions[index_type] = best_action_tensor_this_index
                candidate_costs[index_type] = final_v

        # Evaluate 'none' action cost
        with torch.no_grad():
            none_action_tensor = self._action_to_tensor(Action(k=0, index_type='none', lambda_weight=0.0))
            model_kwargs_none = {}
            if self.paec_config_dynamics is None: self.paec_config_dynamics = {}
            if self.paec_config_dynamics.get('use_decoder_hidden_state', False):
                model_kwargs_none['decoder_hidden_state'] = decoder_hidden_state
            if self.paec_config_dynamics.get('use_text_embeddings', False):
                model_kwargs_none['source_embeddings'] = source_embedding
                model_kwargs_none['prefix_embeddings'] = prefix_embedding
                if model_kwargs_none['source_embeddings'] is None or model_kwargs_none['prefix_embeddings'] is None:
                    raise ValueError("_get_paec_action_online_optim requires text embeddings for 'none' check but they were not provided/computed.")

            try:
                if self.dynamics_model is None:
                    raise RuntimeError("Dynamics model is not loaded for 'none' action evaluation.")
                E_pred_none, H_pred_none, _ = self.dynamics_model(
                    state_tensor, S_hist, none_action_tensor, A_hist, **model_kwargs_none
                )
                _, Phi_t, _ = t_train_Transformer.split_state(state_tensor)
                S_next_none = torch.cat([E_pred_none, Phi_t, H_pred_none], dim=1)
                v_none = t_train_Transformer.lyapunov_V(S_next_none, P_diag).item()
            except Exception as e:
                print(f"[Error] OnlineOptim: Dynamics model forward pass failed for 'none' action: {e}")
                traceback.print_exc()
                raise e

        candidate_actions['none'] = none_action_tensor
        candidate_costs['none'] = v_none

        # Select Best Action
        ordered_types = ['none'] + indices_to_optimize
        filtered_costs = {k: candidate_costs.get(k, float('inf')) for k in ordered_types}
        
        # If forced to choose an active index, ignore 'none'
        if force_knn_active:
            if self.is_debug: print("  [Debug] Online Optim: force_knn_active=True. Ignoring 'none' cost.")
            filtered_costs['none'] = float('inf')

        min_cost = float('inf')
        best_type = 'none'

        for type_name, cost in filtered_costs.items():
             if not math.isinf(cost) and not math.isnan(cost) and cost < min_cost:
                  min_cost = cost
                  best_type = type_name

        final_action_tensor = candidate_actions.get(best_type, candidate_actions['none'])

        max_k = float(config.KNN_MAX_K)
        final_k_norm = final_action_tensor[0, 4].item()
        final_lambda = final_action_tensor[0, 5].item()
        final_k = round(final_k_norm * max_k)

        if best_type == 'none' or final_k == 0:
            final_index_type = 'none'
            final_k = 0
            final_lambda = 0.0
        else:
             final_index_type = best_type
             if final_k == 0: final_k = 1

        if self.is_debug:
             cost_str = ", ".join([f"{t}: {filtered_costs.get(t, float('inf')):.4f}" for t in ordered_types])
             print(f"  Online Optim Costs: [{cost_str}] -> Chose: {final_index_type} (k={final_k}, L={final_lambda:.3f})")

        return Action(index_type=final_index_type, k=final_k, lambda_weight=final_lambda)

    def _get_offline_policy_action(
        self,
        error_state: ErrorStateVector,
        pressure_state: ResourcePressureVector,
        context_state: GenerativeContextVector,
        S_hist: torch.Tensor,
        A_hist: torch.Tensor,
        decoder_hidden_state: Optional[torch.Tensor] = None,
        source_embedding: Optional[torch.Tensor] = None,
        prefix_embedding: Optional[torch.Tensor] = None
    ) -> Action:
        """
        Gets the action by performing a forward pass through the pre-trained
        offline Policy Network (Pi_phi).
        This is much faster than online optimization.
        Assumes the Resource Safety Valve has already passed.
        """
        if not self.offline_policy_phi or not self.paec_dynamics_scaler:
            print("[Error] Offline policy model or scaler not loaded. Cannot get offline action.")
            return Action(k=4, index_type=config.DEFAULT_INDEX, lambda_weight=0.3) # Failsafe

        # 1. Normalize State
        system_state = SystemState(error_state, pressure_state, context_state, timestamp=0.0)
        state_vec = system_state.to_vector().reshape(1, -1)
        state_df = pd.DataFrame(state_vec, columns=t_train_Transformer.STATE_COLS_DEFAULT)
        state_norm = self.paec_dynamics_scaler.transform(state_df)
        S_t_tensor = torch.from_numpy(state_norm.astype(np.float32)).to(self.device)

        # 2. Reshape Inputs
        if S_t_tensor.dim() == 2: S_t_tensor = S_t_tensor.unsqueeze(1) # Shape: [1, 1, S_DIM]
        if self.is_debug:
            print_txt = ', '.join([f"{v:.4f}" for v in S_t_tensor.flatten().tolist()])
            print(f"  [Debug] Scaled S_t_tensor (input to Pi_phi): [{print_txt}]")

        source_emb_input = source_embedding
        prefix_emb_input = prefix_embedding
        hidden_state_input = decoder_hidden_state

        history_len = S_hist.shape[1]
        mask = torch.zeros(1, history_len + 1, dtype=torch.bool, device=self.device)

        if self.is_debug:
            print(f"  [Debug] S_hist input shape: {S_hist.shape}, Mean: {S_hist.mean():.4f}, Std: {S_hist.std():.4f}")
            print(f"  [Debug] A_hist input shape: {A_hist.shape}, Mean: {A_hist.mean():.4f}, Std: {A_hist.std():.4f}")

            if hidden_state_input is not None:
                print(f"  [Debug] Decoder HS input shape: {hidden_state_input.shape}")
            if source_emb_input is not None:
                print(f"  [Debug] Source Emb input shape: {source_emb_input.shape}")
            if prefix_emb_input is not None:
                print(f"  [Debug] Prefix Emb input shape: {prefix_emb_input.shape}")
            print(f"  [Debug] Mask shape: {mask.shape}")

        # 3. Policy Forward Pass
        with torch.no_grad():
            try:
                if S_hist.dim() != 3: S_hist = S_hist.view(1, history_len, -1)
                if A_hist.dim() != 3: A_hist = A_hist.view(1, history_len, -1)
                if S_t_tensor.dim() != 3: S_t_tensor = S_t_tensor.view(1, 1, -1)

                pred_index_logits, pred_k_lambda_values = self.offline_policy_phi(
                    S_t_tensor, S_hist, A_hist,
                    hidden_state_input, source_emb_input, prefix_emb_input,
                    src_key_padding_mask=mask
                )
            except Exception as e:
                print(f"[Error] Offline policy forward pass failed: {e}")
                print(f"Input shapes: S_t={S_t_tensor.shape if S_t_tensor is not None else 'None'}, S_hist={S_hist.shape if S_hist is not None else 'None'}, A_hist={A_hist.shape if A_hist is not None else 'None'}")
                if hidden_state_input is not None: print(f"decoder_hs={hidden_state_input.shape}")
                if source_emb_input is not None: print(f"source_emb={source_emb_input.shape}")
                if prefix_emb_input is not None: print(f"prefix_emb={prefix_emb_input.shape}")
                traceback.print_exc()
                return Action(k=4, index_type=config.DEFAULT_INDEX, lambda_weight=0.3)

        # 4. Decode Output (Ignore 'none' logit as Safety Valve logic is separate)
        logits_k_gt_0 = pred_index_logits[:, 1:] # Shape: [1, 3]
        best_index_relative = torch.argmax(logits_k_gt_0.squeeze(0)).item()
        best_index_absolute = best_index_relative + 1 # +1 because we skipped index 0 ('none')
        
        k_norm_pred = torch.clamp(pred_k_lambda_values.squeeze(0)[0], 0.0, 1.0).item()
        lambda_pred = torch.clamp(pred_k_lambda_values.squeeze(0)[1], 0.0, 1.0).item()

        index_map_inv = {0: 'none', 1: 'exact', 2: 'hnsw', 3: 'ivf_pq'}
        final_index_type = index_map_inv.get(int(best_index_absolute), config.DEFAULT_INDEX)

        max_k = float(config.KNN_MAX_K)
        final_k = round(k_norm_pred * max_k)

        # Force valid action if safety valve passed
        if final_k == 0: final_k = 1
         
        final_lambda = lambda_pred
        action = Action(index_type=final_index_type, k=final_k, lambda_weight=final_lambda)

        if self.is_debug:
            logits_cpu = pred_index_logits.squeeze(0).cpu().tolist()
            logits_str = ", ".join([f"({i}) {logit:.2f}" for i, logit in enumerate(logits_cpu)])
            k_gt_0_str = ", ".join([f"{logit:.2f}" for logit in logits_k_gt_0.squeeze(0).cpu().tolist()])
            print(f"  Offline Policy Raw: Logits(All 4)=[{logits_str}]")
            print(f"  Offline Policy Decision: Logits(k>0)=[{k_gt_0_str}] -> Rel Idx={best_index_relative} -> Abs Idx={best_index_absolute}")
            print(f"  Offline Policy Action: Index={action.index_type}, k={action.k}, lambda={action.lambda_weight:.3f}")

        return action

    def teardown(self):
        """
        Releases resources, especially FAISS indices on GPU and loaded models.
        """
        if self.is_debug: print("[Info] Tearing down kNNMTSystem and releasing GPU resources...")
        if hasattr(self, "model") and self.model: del self.model
        if hasattr(self, "sentence_encoder") and self.sentence_encoder: del self.sentence_encoder
        if hasattr(self, "lm_model") and self.lm_model: del self.lm_model
        if hasattr(self, "lm_tokenizer") and self.lm_tokenizer: del self.lm_tokenizer
        if hasattr(self, "dynamics_model") and self.dynamics_model: del self.dynamics_model
        if hasattr(self, "offline_policy_phi") and self.offline_policy_phi: del self.offline_policy_phi
        if hasattr(self, "paec_dynamics_scaler") and self.paec_dynamics_scaler: del self.paec_dynamics_scaler
        if hasattr(self, "paec_config_dynamics") and self.paec_config_dynamics: del self.paec_config_dynamics
        if hasattr(self, "confidence_history") and self.confidence_history: del self.confidence_history
        if hasattr(self, "query_history") and self.query_history: del self.query_history
        
        if hasattr(self, "bpe") and self.bpe: del self.bpe
        if hasattr(self, "spm_processor") and self.spm_processor: del self.spm_processor
        if hasattr(self, "task") and self.task: del self.task
        if hasattr(self, "src_dict") and self.src_dict: del self.src_dict
        if hasattr(self, "tgt_dict") and self.tgt_dict: del self.tgt_dict
        if hasattr(self, "ner_de") and self.ner_de: del self.ner_de
        if hasattr(self, "ner_en") and self.ner_en: del self.ner_en
        
        # Clean up FAISS
        for name in config.ENABLED_INDICES:
            name = name + '_index'
            if hasattr(self, name):
                index = getattr(self, name)
                if hasattr(index, 'this'): # Check if it's a SWIG object
                    if self.is_debug: print(f"\t- Freeing FAISS index: {name}")
                    index.reset() # Clears the index data
                    del index
                setattr(self, name, None)
        if self.use_faiss_gpu and self.gpu_res is not None:
            if self.is_debug: print("\t- Freeing FAISS StandardGpuResources object.")
            del self.gpu_res
            self.gpu_res = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.is_debug: print("[Success] kNNMTSystem resources have been released.")
