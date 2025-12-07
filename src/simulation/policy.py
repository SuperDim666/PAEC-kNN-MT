# -*- coding: utf-8 -*-
"""
src/simulation/policy.py

[PAEC Framework - Heuristic Policy Definitions]

This module implements the Descriptive Policy Functions π(·|S_t), which map the
current system state vector S_t to a probability distribution over the discrete
action space A.

These heuristic policies are primarily used during Phase 1 (Data Generation) to
create a diverse training dataset for the Dynamics Model (T_theta). Each policy
represents a different decision-making philosophy (e.g., balanced, quality-first,
resource-conservative, or adversarial), ensuring the state-action space is
comprehensively explored.

The policies operate on the normalized 11-dimensional state vector components:
1. Error State (E_t): Semantic, Coverage, Surprisal, Repetition.
2. Pressure State (Phi_t): Latency, Memory, Throughput.
3. Context State (H_t): Focus, Consistency, Stability, Volatility.
"""

import numpy as np
from typing import Tuple, Union, Dict
import math

# Import project-specific modules
from src import config
from src.core import SystemState, Action

class BasePolicy:
    """
    Abstract base class for all heuristic policies.

    Responsibilities:
    1. Loading global constraint limits from the configuration.
    2. Normalizing raw state values into the [0, 1] range for consistent decision logic.
    3. Providing utility methods for lambda adjustment based on model confidence.
    4. Defining the standard interface `decide` for all derived policies.
    """
    _constraint_limit: Dict[str, Dict[str, float]] = {}
    LAMBDA_UPPER_BOUND = 0.6

    def __init__(self):
        """
        Initializes the policy instance. Loads normalization constraints from the
        global configuration on the first instantiation.
        """
        # Load constraints only once when the first policy object is created.
        if not BasePolicy._constraint_limit:
            if hasattr(config, 'CONSTRAINT_LIMIT'):
                BasePolicy._constraint_limit = config.CONSTRAINT_LIMIT
                print("[Info] Policy Normalizer: Loaded constraint limits from config.")
            else:
                raise ImportError("Policy Normalizer: 'CONSTRAINT_LIMIT' not found in config.")
    
    def _get_confidence_lambda(self, base_lambda: float, confidence: float) -> float:
        """
        Adjusts the interpolation weight (lambda) based on the NMT model's confidence.
        
        Principle: If the NMT model is highly confident in its own prediction,
        the need for external kNN intervention is reduced.

        Args:
            base_lambda (float): The initial lambda determined by the policy logic.
            confidence (float): The NMT model's confidence (0.0 to 1.0).

        Returns:
            float: The adjusted lambda, clamped to the global upper bound.
        """
        # The confidence penalty factor is inversely proportional to the model's confidence.
        # If confidence is 1.0, penalty is 0. If confidence is 0, penalty is 1.
        confidence_penalty = 1.0 - confidence
        
        # Apply the penalty to the base lambda.
        adjusted_lambda = base_lambda * confidence_penalty
        
        # Enforce the global upper bound as a final safety measure.
        return max(0, min(adjusted_lambda, self.LAMBDA_UPPER_BOUND))

    def _normalize_value(self, value: float, component_name: str) -> float:
        """
        Normalizes a raw state component value to the [0, 1] range based on
        pre-defined statistical limits (min/max).

        Args:
            value (float): The raw value of the state component.
            component_name (str): The key identifying the component in CONSTRAINT_LIMIT.

        Returns:
            float: The normalized value.
        """
        limits = BasePolicy._constraint_limit.get(component_name)
        if not limits: return value

        min_val, max_val = limits.get("min", 0.0), limits.get("max", 1.0)
        
        # Ensure max_val is greater than min_val to avoid division by zero
        if max_val <= min_val:
            return 0.5 # Return a neutral value if bounds are invalid

        # 1. Clamp the value to the robust range [P1, P99]
        clipped_value = np.clip(value, min_val, max_val)
        
        # 2. Apply Min-Max scaling to map to [0, 1]
        normalized_value = (clipped_value - min_val) / (max_val - min_val)
        
        return normalized_value

    def decide(
        self,
        pressure_state: Tuple[float, float, float], 
        error_state: Tuple[float, float, float, float], 
        context_state: Tuple[float, float, float, float]
    ) -> Action:
        """
        Main entry point for decision making. Normalizes specific components of the
        state vector and delegates to the concrete policy implementation.

        Args:
            pressure_state: Tuple (latency, memory, throughput).
            error_state: Tuple (semantic, coverage, surprisal, repetition).
            context_state: Tuple (focus, consistency, stability, volatility).

        Returns:
            Action: The selected action (IndexType, k, lambda).
        """
        
        # --- 1. Unpack all 9 new state vector components ---
        error_semantic, error_coverage, error_fluency_surprisal, error_fluency_repetition = error_state
        context_faith_focus, context_consistency, context_stability, context_confidence_volatility = context_state

        # --- 2. Normalize ONLY the necessary components ---
        # Most components (like coverage, pressure) are natively in [0, 1].
        # Semantic error is in [0, 2] and requires normalization.
        norm_error_semantic = self._normalize_value(error_semantic, "error_semantic")

        # --- 3. Create the state tuples for the heuristic policy ---
        # The heuristic logic expects all inputs to be roughly in the [0, 1] range.
        final_error_state = (
            norm_error_semantic,
            error_coverage,
            error_fluency_surprisal,
            error_fluency_repetition
        )
        
        final_context_state = (
            context_faith_focus,
            context_consistency,
            context_stability,
            context_confidence_volatility
        )

        # --- 5. Call the implementation-specific logic ---
        return self.decide_normalized(pressure_state, final_error_state, final_context_state)

    def decide_normalized(
        self,
        pressure_state: Tuple[float, float, float], 
        error_state: Tuple[float, float, float, float], 
        context_state: Tuple[float, float, float, float]
    ) -> Action:
        """
        Abstract method to be implemented by child policies.
        Contains the core heuristic logic mapping normalized states to actions.
        """
        raise NotImplementedError("Child policies must implement decide_normalized.")

class Policy_Default_Balanced(BasePolicy):
    """
    [Strategy: Balanced / Pragmatic Opportunist]
    
    A heuristic policy that attempts to strike a dynamic balance between system
    resources, translation quality, and contextual stability. It represents the
    default behavior for a production system.

    Logic Hierarchy:
    1. Pressure Check: Sets a baseline mode (safe vs. efficient) based on resource load.
    2. Error Intervention: Increases intensity if quality drops (e.g., semantic drift).
    3. Context Fine-tuning: Adjusts k/lambda based on reliability and model confidence.
    """
    
    def __init__(self):
        super().__init__()

    def decide_normalized(
        self,
        pressure_state: Tuple[float, float, float], 
        error_state:    Tuple[float, float, float, float], 
        context_state:  Tuple[float, float, float, float]
    ) -> Action:
        """
        Computes action based on a balanced assessment of all state components.
        """
        latency, memory, throughput = pressure_state
        semantic, coverage, surprisal, repetition = error_state
        focus, consistency, stability, volatility = context_state

        # --- Stage 1: Pressure-based Macro Mode ---
        # Determine the base search index and k based on system load.
        if memory >= 0.9:  # Red alert: Critical memory pressure
            return Action(index_type='none', k=0, lambda_weight=0.0)
        elif 0.75 <= memory < 0.9:  # Orange alert: High memory pressure
            # Downgrade to IVF_PQ (memory efficient) and reduce k
            index_type = 'ivf_pq'
            k_base = int(4 - 3 * (memory - 0.75) / 0.15)  # Linear scale k from 4 down to 1
            lambda_base = 0.5
        else:  # Safe zone: Resources are available
            if latency < 0.4 and throughput < 0.4:  # Idle state
                index_type = 'exact' # Use most accurate index
            elif latency > 0.65:  # High latency dominant
                index_type = 'hnsw'  # Use fast approximate index
            else:  # Default balanced
                index_type = 'hnsw'
            k_base = 8

        # --- Stage 2: Error Control Intervention ---
        # Override baseline parameters if significant errors are detected.
        
        # Macro failure: Severe semantic drift or missing entities
        if semantic > 0.8 or coverage > 0.9:
            # Force strong intervention to correct the trajectory
            k_base = 16
            lambda_base = 0.5 
            # Prioritize content correctness over fluency

        # Repetition loop detected
        elif repetition > 0.7:
            # Introduce diversity to break the loop
            k_base = 12
            lambda_base = 0.4

        # Local disfluency (high surprisal)
        elif surprisal > 0.8:
            # Use kNN to find more natural phrasings
            k_base = 8
            lambda_base = 0.3

        # Minor errors: Weighted combination
        else:
            weighted_error = 0.4 * semantic + 0.3 * coverage + 0.2 * surprisal + 0.1 * repetition
            if weighted_error > 0.5:
                k_base = 8
                lambda_base = 0.33
            else:
                # Low error: Maintenance mode
                k_base = 4
                lambda_base = 0.2

        # --- Stage 3: Context-based Fine-tuning ---
        
        # 1. kNN Reliability: High consistency implies the datastore is relevant.
        knn_reliability = consistency

        # 2. Necessity: Does the NMT model need help?
        # High need if: NMT is unstable, volatile, or unfocused.
        nmt_needs_help = 0.4 * (1.0 - stability) + 0.3 * volatility + 0.3 * (1.0 - focus)

        # 3. Adjust k and lambda
        # If kNN is reliable, we can afford to retrieve more neighbors.
        k_multiplier = 1.0 + 0.5 * (knn_reliability - 0.5)
        k_final = round(k_base * k_multiplier)
        
        # If NMT is struggling, increase the interpolation weight.
        lambda_adjustment = 0.1 * nmt_needs_help
        lambda_tuned = lambda_base + lambda_adjustment
        
        # 4. Final Confidence Penalty
        # Define effective confidence: High stability + Low volatility.
        effective_confidence = 0.7 * stability + 0.3 * (1.0 - volatility)
        lambda_final = self._get_confidence_lambda(lambda_tuned, effective_confidence)

        return Action(index_type=index_type, k=k_final, lambda_weight=lambda_final)

class Policy_Quality_First(BasePolicy):
    """
    [Strategy: Perfectionist / Idealist]

    Prioritizes translation quality (low Error) above all else.
    Priority Hierarchy: E > H > P (Error > Context > Pressure).
    
    Resources are considered secondary and only trigger overrides in critical
    situations (e.g., imminent OOM). This policy generates data representing
    "best possible quality" behavior.
    """
    def __init__(self):
        super().__init__()

    def decide_normalized(
        self,
        pressure_state: Tuple[float, float, float],
        error_state: Tuple[float, float, float, float],
        context_state: Tuple[float, float, float, float]
    ) -> Action:
        """
        Computes action focusing on minimizing error components.
        """
        # Unpack state vectors
        latency, memory, throughput = pressure_state
        semantic, coverage, surprisal, repetition = error_state
        focus, consistency, stability, volatility = context_state

        # --- Stage 1: Error-driven Macro Intervention (E > H > P) ---
        # Evaluate critical errors first.

        # Critical Failure: Semantic drift or Entity omission
        if semantic > 0.75 or coverage > 0.9:
            # Maximum Precision Correction Mode
            index_type = 'exact'
            k = 16
            base_lambda = 0.6 
        
        # Major Failure: Stuck in a repetition loop
        elif repetition > 0.7:
            # Loop Breaker Mode: High diversity needed
            index_type = 'hnsw'
            k = 12
            base_lambda = 0.5

        # Moderate Failure: Grammatical awkwardness
        elif surprisal > 0.8:
            # Grammatical Correction Mode
            index_type = 'hnsw'
            k = 8
            base_lambda = 0.4

        # --- Stage 2: Context-based Preventive Intervention ---
        # If errors are low, check for risky context signals.
        else:
            # High risk: Low focus, low stability, high volatility
            context_risk_score = (
                0.3 * (1.0 - focus) +
                0.3 * (1.0 - stability) +
                0.2 * volatility +
                0.2 * (1.0 - consistency)
            )

            if context_risk_score > 0.6:
                # Preventive High-Quality Retrieval
                index_type = 'hnsw'
                k = 10
                base_lambda = 0.45
            
            elif context_risk_score > 0.3:
                # Standard Safety Net
                index_type = 'ivf_pq'
                k = 4
                base_lambda = 0.3
            
            else:
                # Ideal State: High-Fidelity Maintenance
                index_type = 'exact'
                k = 8
                base_lambda = 0.25

        # --- Stage 3: Pressure-based Safety Override ---
        # Only override if system survival is at stake.

        # Emergency Shutdown: Imminent crash
        if memory > 0.95:
            return Action(index_type='none', k=0, lambda_weight=0.0)

        # Emergency Downgrade: Critical memory levels
        elif memory > 0.8:
            index_type = 'ivf_pq'
            k = max(1, min(k, 4)) # Cap k

        # --- Final Lambda Calculation ---
        effective_confidence = 1.0 - (0.7 * stability + 0.3 * volatility)
        final_lambda = self._get_confidence_lambda(base_lambda, effective_confidence)

        if index_type != 'none' and k == 0:
            k = 1

        return Action(index_type=index_type, k=k, lambda_weight=final_lambda)

class Policy_Resource_Guardian(BasePolicy):
    """
    [Strategy: Minimalist / Resource Guardian]

    Extremely conservative policy prioritizing system stability and efficiency.
    Priority Hierarchy: P >> E > H (Pressure far outweighs Error).

    Directives:
    1. If ANY pressure metric > 0.6, disable kNN immediately.
    2. Only enable minimal kNN if a critical error occurs while resources are abundant.
    3. Veto intervention if the datastore is unlikely to help (low consistency).
    """
    def __init__(self):
        super().__init__()
        
        # High threshold for error; intervene only when critical.
        self.CRITICAL_ERROR_THRESHOLD = 0.8
        
        # Veto threshold; if context consistency is low, don't waste resources.
        self.KNN_SUPPORT_THRESHOLD = 0.3
        
        # Minimal intervention weight.
        self.BASE_LAMBDA = 0.25

    def decide_normalized(
        self,
        pressure_state: Tuple[float, float, float],
        error_state: Tuple[float, float, float, float],
        context_state: Tuple[float, float, float, float]
    ) -> Action:
        """
        Computes action by aggressively minimizing resource usage.
        """
        latency, memory, throughput = pressure_state
        semantic, coverage, repetition, surprisal = error_state
        focus, consistency, stability, volatility = context_state

        # --- Stage 1: Mandatory Pressure Check (P >> E > H) ---
        # Immediate shutdown if any pressure is even moderately high.
        if memory > 0.6 or latency > 0.6 or throughput > 0.6:
            # Absolute Resource Priority Mode
            return Action(index_type='none', k=0, lambda_weight=0.0)

        # --- Stage 2: Critical Error Evaluation ---
        # Resources are safe. Check if intervention is strictly necessary.
        # Ignore local surprisal; focus on semantic/coverage failures.
        weighted_error = 0.5 * semantic + 0.4 * coverage + 0.1 * repetition

        if weighted_error > self.CRITICAL_ERROR_THRESHOLD:
            # Propose Minimum Cost Intervention
            action_to_consider = Action(index_type='ivf_pq', k=2, lambda_weight=self.BASE_LAMBDA)
        else:
            # Error is tolerable; save resources.
            return Action(index_type='none', k=0, lambda_weight=0.0)

        # --- Stage 3: Effectiveness Veto ---
        # Ensure the proposed kNN lookup isn't a waste of resources.
        knn_support_score = consistency

        if knn_support_score < self.KNN_SUPPORT_THRESHOLD:
            # Veto: Datastore unlikely to help.
            return Action(index_type='none', k=0, lambda_weight=0.0)
        else:
            # Approve minimal intervention.
            final_lambda = action_to_consider.lambda_weight
            return Action(
                index_type=action_to_consider.index_type,
                k=action_to_consider.k,
                lambda_weight=final_lambda
            )
        
class Policy_Stability_Averse(BasePolicy):
    """
    [Strategy: Psychologist / Stability Averse]

    Focuses on the NMT model's internal cognitive state (Context H).
    Priority Hierarchy: H >> P > E.

    Philosophy:
    - Intervene only when the model is "confused", "unstable", or "distracted".
    - If the model is stable and confident, trust it completely (ignore minor errors).
    """
    def __init__(self):
        super().__init__()

    def decide_normalized(
        self,
        pressure_state: Tuple[float, float, float],
        error_state: Tuple[float, float, float, float],
        context_state: Tuple[float, float, float, float]
    ) -> Action:
        """
        Computes action based on cognitive risk metrics derived from the context state.
        """
        latency, memory, throughput = pressure_state
        semantic, coverage, surprisal, repetition = error_state
        focus, consistency, stability, volatility = context_state

        # --- Stage 1: Cognitive Risk Assessment (H >> P > E) ---
        # Calculate risk based on instability, volatility, and lack of focus.
        cognitive_risk_score = (
            0.4 * (1.0 - stability) +   # Primary signal: Model confusion
            0.3 * volatility +          # Secondary signal: Decision instability
            0.2 * (1.0 - focus) +       # Distraction (missing entities)
            0.1 * (1.0 - consistency)   # Ambiguity
        )
       
        # If risk is low, the model is in a "Flow State". Do not disturb.
        if cognitive_risk_score < 0.3:
            return Action(index_type='none', k=0, lambda_weight=0.0)

        # --- Stage 2: Tool Selection (Pressure Constraint) ---
        # Risk is high, intervention needed. Select tool based on available memory.
        if memory > 0.75:
            # Low-Cost Assistance
            index_type = 'ivf_pq'
            k_base = 6
            lambda_base = 0.35
            k_max = 8
        else:
            # High-Quality Assistance
            index_type = 'hnsw'
            k_base = 10
            lambda_base = 0.45
            k_max = 16

        # --- Stage 3: Intensity Fine-tuning (Error Modulation) ---
        # Slight adjustment based on observed error magnitude.
        error_magnitude = (semantic + coverage + surprisal + repetition) / 4.0

        # Scale k and lambda based on error severity
        k_final = round(k_base * (1 + 0.5 * error_magnitude))
        lambda_tuned = lambda_base + 0.2 * error_magnitude

        k_final = max(1, min(k_final, k_max))
        
        # --- Final Lambda Calculation ---
        # Apply confidence penalty (High stability = High confidence).
        effective_confidence = 0.7 * stability + 0.3 * (1.0 - volatility)
        final_lambda = self._get_confidence_lambda(lambda_tuned, effective_confidence)

        return Action(index_type=index_type, k=k_final, lambda_weight=final_lambda)

class Policy_Dangerous_Perturbator(BasePolicy):
    """
    [Strategy: Chaos Monkey / Adversarial]

    Intentionally makes "bad" decisions in specific edge cases to generate
    negative training samples. This teaches the dynamics model the consequences
    of poor control (e.g., system crashes or quality degradation).
    
    Acts as a wrapper around a fallback policy, overriding decisions only
    when specific trigger conditions are met.
    """
    def __init__(
        self,
        fallback_policy: BasePolicy = Policy_Default_Balanced()
    ):
        """
        Args:
            fallback_policy: Policy to use when no dangerous condition is met.
        """
        super().__init__()
        self.fallback = fallback_policy

    def decide_normalized(
        self,
        pressure_state: Tuple[float, float, float],
        error_state: Tuple[float, float, float, float],
        context_state: Tuple[float, float, float, float]
    ) -> Action:
        """
        Checks for trigger scenarios to inject bad actions.
        """
        latency, memory, throughput = pressure_state
        semantic, coverage, surprisal, repetition = error_state
        focus, consistency, stability, volatility = context_state

        # --- Scenario 1: The "Explosion" (Resource Sabotage) ---
        # Condition: Memory is critically high.
        # Rational Action: Stop retrieval.
        # Dangerous Action: Maximize retrieval load to cause simulated OOM/High Pressure.
        if memory > 0.7:
            return Action(index_type='exact', k=16, lambda_weight=0.8)

        # --- Scenario 2: The "Abandon Ship" (Quality Sabotage) ---
        # Condition: Severe semantic drift or missing entities.
        # Rational Action: Strong retrieval intervention.
        # Dangerous Action: Disable retrieval completely, allowing error to propagate.
        if semantic > 0.75 or coverage > 0.9:
            return Action(index_type='none', k=0, lambda_weight=0.0)

        # --- Scenario 3: The "Ignorance" (Context Sabotage) ---
        # Condition: Model is unstable/confused, but Datastore is consistent (helpful).
        # Rational Action: Rely on Datastore.
        # Dangerous Action: Ignore Datastore.
        is_model_unstable = stability < 0.2
        is_knn_support_strong = consistency > 0.8
        if is_model_unstable and is_knn_support_strong:
            return Action(index_type='none', k=0, lambda_weight=0.0)
            
        # --- Scenario 4: The "Waste" (Efficiency Sabotage) ---
        # Condition: System state is perfect (low pressure, low error, stable).
        # Rational Action: Minimal/No retrieval.
        # Dangerous Action: Perform expensive 'exact' search for no reason.
        is_pressure_low = all(p < 0.2 for p in pressure_state)
        is_error_low = semantic < 0.1 and coverage < 0.1 and surprisal < 0.1 and repetition < 0.1
        is_context_strong = focus > 0.5 and consistency > 0.8 and stability > 0.8 and volatility < 0.1
        if is_pressure_low and is_error_low and is_context_strong:
            return Action(index_type='exact', k=16, lambda_weight=0.9)
        
        # Additional pressure triggers
        if latency > 0.7 or throughput > 0.7:
            return Action(index_type='exact', k=16, lambda_weight=0.8)

        # Fallback: Use reasonable policy
        return Action(index_type='exact', k=16, lambda_weight=0.9)
