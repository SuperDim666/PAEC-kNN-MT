# -*- coding: utf-8 -*-
"""
src/core/structs.py

[PAEC Framework - Core Data Structures]

This module defines the fundamental data structures used to represent the state
and control space of the PAEC (Production-Aware Exposure Compensation) kNN-MT system.
These classes provide a direct mapping from the theoretical mathematical framework
(State S_t, Action A_t) to Python objects used throughout the pipeline.

The structures defined here are used for:
1. Capturing real-time system metrics during translation.
2. Generating training data for the dynamics model.
3. Facilitating communication between the environment, policy, and dynamics model.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum

# ==============================================================================
# 1. State Space Vector Components (S_t)
# ==============================================================================

@dataclass(frozen=True)
class ErrorStateVector:
    """
    Represents the Error State Vector (E_t) in R^4.

    This vector quantifies the deviation of the current partial translation from
    an ideal translation across four distinct quality dimensions. It serves as the
    primary feedback signal for the control system to minimize.

    Attributes:
        error_semantic (float): Quantifies semantic drift from the source sentence.
                                Range: [0, 2] (based on cosine distance).
        error_coverage (float): Quantifies the ratio of uncovered source entities.
                                Range: [0, 1] (1 means no entities covered).
        error_fluency_surprisal (float): Normalized entropy of the model's next-token
                                         distribution. Indicates uncertainty/confusion.
                                         Range: [0, 1].
        error_fluency_repetition (float): Ratio of repetitive n-grams in the recent
                                          output history. Indicates degenerate loops.
                                          Range: [0, 1].
    """
    error_semantic: float
    error_coverage: float
    error_fluency_surprisal: float
    error_fluency_repetition: float

    def to_vector(self) -> np.ndarray:
        """
        Converts the state fields into a flat NumPy array.
        
        Returns:
            np.ndarray: A 1D array of shape (4,) containing [semantic, coverage, surprisal, repetition].
        """
        return np.array([
            self.error_semantic,
            self.error_coverage,
            self.error_fluency_surprisal,
            self.error_fluency_repetition
        ], dtype=np.float32)
    
    def to_tuple(self) -> tuple:
        """
        Converts the state fields into a standard Python tuple.
        Useful for unpacking or hashing.
        
        Returns:
            tuple: (semantic, coverage, surprisal, repetition)
        """
        return (
            self.error_semantic,
            self.error_coverage,
            self.error_fluency_surprisal,
            self.error_fluency_repetition
        )

    def norm(self) -> float:
        """
        Calculates the L2 Euclidean norm of the error vector.
        This provides a scalar magnitude of the total error 'energy'.
        
        Returns:
            float: The L2 norm.
        """
        return float(np.linalg.norm(self.to_vector()))

@dataclass
class ResourcePressureVector:
    """
    Represents the Resource Pressure State Vector (Phi_t) in (0, 1)^3.

    This vector quantifies the normalized dynamic load on the system's computational
    resources. These values are typically computed via a sigmoid function applied to
    raw metrics (latency, memory, throughput) relative to SLA thresholds.

    Attributes:
        latency_pressure (float): Normalized pressure from inference response time.
        memory_pressure (float): Normalized pressure from system memory usage.
        throughput_pressure (float): Normalized pressure from request processing rate gaps.
    """
    latency_pressure: float
    memory_pressure: float
    throughput_pressure: float

    def to_vector(self) -> np.ndarray:
        """
        Converts the pressure fields into a flat NumPy array.

        Returns:
            np.ndarray: A 1D array of shape (3,) containing [latency, memory, throughput].
        """
        return np.array([
            self.latency_pressure,
            self.memory_pressure,
            self.throughput_pressure
        ], dtype=np.float32)
    
    def to_tuple(self) -> tuple:
        """
        Converts the pressure fields into a standard Python tuple.

        Returns:
            tuple: (latency, memory, throughput)
        """
        return (
            self.latency_pressure,
            self.memory_pressure,
            self.throughput_pressure
        )

    def norm(self) -> float:
        """
        Calculates the L2 Euclidean norm of the pressure vector.

        Returns:
            float: The L2 norm.
        """
        return float(np.linalg.norm(self.to_vector()))

@dataclass(frozen=True)
class GenerativeContextVector:
    """
    Represents the Generative Context State Vector (H_t) in R^4.

    This vector captures the internal cognitive state of the NMT model, projecting
    high-dimensional hidden states into interpretable metrics regarding the
    generation process's stability and focus.

    Attributes:
        context_faith_focus (float): Measures attention concentration on source entities.
        context_consistency (float): Measures alignment between the query vector and attention context.
        context_stability (float): Measures the temporal stability of the query vector trajectory.
        context_confidence_volatility (float): Measures the variance in model confidence over recent steps.
    """
    context_faith_focus: float
    context_consistency: float
    context_stability: float
    context_confidence_volatility: float

    def to_vector(self) -> np.ndarray:
        """
        Converts the context fields into a flat NumPy array.

        Returns:
            np.ndarray: A 1D array of shape (4,) containing [focus, consistency, stability, volatility].
        """
        return np.array([
            self.context_faith_focus,
            self.context_consistency,
            self.context_stability,
            self.context_confidence_volatility
        ], dtype=np.float32)
    
    def to_tuple(self) -> tuple:
        """
        Converts the context fields into a standard Python tuple.

        Returns:
            tuple: (focus, consistency, stability, volatility)
        """
        return (
            self.context_faith_focus,
            self.context_consistency,
            self.context_stability,
            self.context_confidence_volatility
        )
    
# ==============================================================================
# 2. Total System State Vector (S_t)
# ==============================================================================

@dataclass
class SystemState:
    """
    Represents the Total System State Vector S_t = (E_t, Phi_t, H_t).

    This aggregate structure encapsulates the complete snapshot of the system
    at a specific discrete time step 't'. It serves as the primary input for
    both the Dynamics Model (T_theta) and the Policy Network (Pi_phi).

    Attributes:
        error_state (ErrorStateVector): The translation quality component.
        pressure_state (ResourcePressureVector): The resource constraint component.
        context_state (GenerativeContextVector): The internal model context component.
        timestamp (float): The absolute time when this state was recorded (for logging/debugging).
    """
    error_state: ErrorStateVector
    pressure_state: ResourcePressureVector
    context_state: GenerativeContextVector
    timestamp: float

    def to_vector(self) -> np.ndarray:
        """
        Concatenates all sub-state vectors into a single, unified state vector.
        
        This vector represents the raw input features before any normalization
        or processing by neural networks.

        Returns:
            np.ndarray: A 1D array of shape (11,) combining Error (4), Pressure (3), and Context (4).
        """
        return np.concatenate([
            self.error_state.to_vector(),
            self.pressure_state.to_vector(),
            self.context_state.to_vector()
        ], axis=0)

# ==============================================================================
# 3. Action Space and Decoding Strategy
# ==============================================================================

@dataclass
class Action:
    """
    Represents a discrete-continuous Control Action (A_t).

    An action defines the configuration for the kNN retrieval mechanism at the current step.
    It includes a categorical choice of index type and continuous parameters for the
    retrieval/fusion process.

    Attributes:
        k (int): The number of nearest neighbors to retrieve.
                 If k=0, retrieval is skipped.
        index_type (str): The type of FAISS index to query (e.g., 'none', 'exact', 'hnsw', 'ivf_pq').
        lambda_weight (float): The interpolation weight [0, 1] for fusing the kNN distribution.
    """
    k: int
    index_type: str
    lambda_weight: float

    def __post_init__(self):
        """
        Enforces logical consistency for the 'none' action.
        
        If the number of neighbors 'k' is 0, it implies no retrieval should happen.
        Therefore, index_type is forced to 'none' and lambda_weight to 0.0 to prevent
        invalid configurations.
        """
        if self.k == 0:
            self.index_type = 'none'
            self.lambda_weight = 0.0

class DecodingStrategy(Enum):
    """
    Enumeration of supported decoding strategies for the NMT system.
    
    Currently supports Beam Search, but extensible for strategies like
    Greedy, Nucleus Sampling, etc.
    """
    BEAM_SEARCH = "beam_search"
    # Future strategies like NUCLEUS_SAMPLING could be added here.
