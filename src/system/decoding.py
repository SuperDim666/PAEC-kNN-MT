# -*- coding: utf-8 -*-
"""
src/system/decoding.py

[PAEC Framework - Decoding Logic]

This module encapsulates generic decoding algorithms used within the kNN-MT system,
primarily focusing on the Beam Search implementation.

It separates the search strategy (maintenance of hypotheses, pruning, early stopping)
from the specific model logic (forward passes, state computation, policy decisions),
allowing for a modular design where the PAEC control loop interacts with the decoder
via a callback interface.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict

# Import core data structures for type hinting.
from src.core import GenerativeContextVector, ErrorStateVector

@dataclass
class BeamHypothesis:
    """
    Represents a single hypothesis (a potential translation path) within the beam
    search algorithm.

    This data structure acts as a container for the sequence generation history,
    tracking not just the tokens and scores, but also the specific state vectors
    (Context, Error) required by the PAEC framework to log trajectories and
    make control decisions.

    Attributes:
        tokens (List[int]): The sequence of generated token IDs (indices in the target vocabulary).
        log_prob (float): The cumulative log probability of this token sequence.
        hidden_states (List[torch.Tensor]): History of decoder hidden states for each step.
                                            Used for state reconstruction.
        context_states (List[GenerativeContextVector]): History of generative context vectors H_t.
                                                        Tracks the internal cognitive state over time.
        error_states (List[ErrorStateVector]): History of error state vectors E_t.
                                               Tracks the quality estimation over time.
        query_embeddings (List[np.ndarray]): History of embeddings used for kNN queries.
                                             Used for visualizing retrieval density/consistency.
        past_key_values (Optional[tuple]): Cached key-value pairs for Transformer inference optimization.
                                           Allows incremental decoding without re-processing the prefix.
    """
    tokens: List[int]
    log_prob: float

    # Use field(default_factory=list) to ensure each instance gets a new list,
    # avoiding issues with mutable default arguments shared across instances.
    hidden_states: List[torch.Tensor] = field(default_factory=list)
    context_states: List[GenerativeContextVector] = field(default_factory=list)
    error_states: List[ErrorStateVector] = field(default_factory=list)
    query_embeddings: List[np.ndarray] = field(default_factory=list)

    # Optional field for past key values (caching mechanism)
    past_key_values: Optional[tuple] = None

    def __lt__(self, other: 'BeamHypothesis') -> bool:
        """
        Comparison method for sorting and priority queue operations.

        Args:
            other (BeamHypothesis): Another hypothesis to compare against.

        Returns:
            bool: True if this hypothesis has a lower log probability than the other.
                  Note: Standard Python sort is ascending. For max-heap behavior (best first),
                  reverse sorting is typically used.
        """
        return self.log_prob < other.log_prob

    def score(self, length_penalty: float = 1.0) -> float:
        """
        Calculates the length-normalized score for the hypothesis.

        This normalization prevents the search algorithm from unconditionally favoring
        shorter sequences, as log probabilities are additive negative numbers (adding
        more tokens always decreases the unnormalized total log probability).

        Args:
            length_penalty (float): Factor to penalize/reward sequence length.
                                    values > 1.0 encourage longer sequences.
                                    values < 1.0 penalize longer sequences.

        Returns:
            float: The normalized score.
        """
        length = len(self.tokens)
        if length == 0:
            return -float('inf')
        return self.log_prob / (length ** length_penalty)

class BeamSearchDecoder:
    """
    A generic, modular Beam Search Decoder.

    This class orchestrates the iterative expansion of hypotheses. It manages the
    lifecycle of active beams and completed hypotheses (those ending with EOS).
    Crucially, the actual logic for generating the next set of candidates (including
    NMT forward pass, PAEC policy execution, and kNN retrieval) is delegated to
    an external `step_function`.
    """

    def __init__(self, beam_size: int, eos_token_id: int, pad_token_id: int,
                 length_penalty: float = 1.0, early_stopping: bool = True):
        """
        Initializes the BeamSearchDecoder configuration.

        Args:
            beam_size (int): The beam width (k), determining how many active hypotheses to maintain.
            eos_token_id (int): The token ID representing the End-Of-Sentence.
            pad_token_id (int): The token ID representing padding (used for batching).
            length_penalty (float): The length penalty factor used for scoring.
            early_stopping (bool): If True, the search terminates as soon as `beam_size`
                                   complete hypotheses are collected.
        """
        self.beam_size = beam_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

    def search(
        self,
        initial_beams: List[BeamHypothesis],
        step_function: Callable[[List[BeamHypothesis], int, int, Optional[Dict]], List[BeamHypothesis]],
        max_length: int,
        target_beam_size: int,
        step_hook: Optional[Callable[[List[BeamHypothesis], int], Dict]] = None
    ) -> List[BeamHypothesis]:
        """
        Executes the beam search algorithm.

        This method runs the main decoding loop. In every step, it calls the `step_function`
        to expand current beams into candidates, then prunes them based on scores.

        Args:
            initial_beams (List[BeamHypothesis]): The starting list of hypotheses (usually just BOS).
            step_function (Callable): A callback function that performs a single decoding step.
                Signature: step_function(active_beams, current_step, beam_width, hook_info) -> all_candidates
                This function encapsulates the PAEC logic: State Calculation -> Policy -> kNN Retrieval -> Scoring.
            max_length (int): The hard limit on the number of generated tokens.
            target_beam_size (int): The desired number of beams to maintain (usually same as self.beam_size).
            step_hook (Optional[Callable]): An optional callback executed at the start of each step.
                Can be used for logging, resource monitoring updates, or side-effects.

        Returns:
            List[BeamHypothesis]: The list of top-k completed hypotheses, sorted by score.
        """
        active_beams = initial_beams
        completed_hypotheses: List[BeamHypothesis] = []

        for step in range(max_length):
            # If no beams are active, search cannot proceed.
            if not active_beams:
                break

            # Execute the optional step hook (e.g., for updating resource monitors).
            hook_info = None
            if step_hook:
                hook_info = step_hook(active_beams, step)

            # Expand current beams into next-step candidates via the provided callback.
            # This is where the model inference and PAEC control logic reside.
            all_candidates = step_function(active_beams, step, target_beam_size, hook_info)

            # Sort candidates by length-normalized score (descending order).
            all_candidates.sort(key=lambda h: h.score(self.length_penalty), reverse=True)
            
            # Prune: Keep only the top-k best candidates.
            active_beams = all_candidates[:self.beam_size]

            # Separate completed hypotheses from active ones.
            new_active_beams: List[BeamHypothesis] = []
            for beam in active_beams:
                if beam.tokens[-1] == self.eos_token_id:
                    # If the last generated token is EOS, the hypothesis is complete.
                    completed_hypotheses.append(beam)
                else:
                    # Otherwise, it remains active for the next expansion step.
                    new_active_beams.append(beam)

            active_beams = new_active_beams

            # Check early stopping condition:
            # If we have enough completed sentences, we can stop generating.
            if self.early_stopping and len(completed_hypotheses) >= self.beam_size:
                break

        # If active beams remain after max_length, treat them as completed (force finish).
        completed_hypotheses.extend(active_beams)
        
        # Final sort of all completed hypotheses.
        completed_hypotheses.sort(key=lambda h: h.score(self.length_penalty), reverse=True)
        
        # Return the top-k results.
        return completed_hypotheses[:self.beam_size]
