# -*- coding: utf-8 -*-
"""
src/simulation/constraint_simulator.py

This module implements the ProductionConstraintSimulator, a core component of the PAEC
framework. It simulates the dynamic changes in resource pressure (latency, memory,
throughput) that a kNN-MT system would experience in a real production environment.

It acts as a "Virtual Environment" for Phase 1 (Data Generation), replacing the
real-time monitoring used in Phase 4 (Validation). It mathematically models:
1. Stochastic Traffic Patterns (Gamma distribution + Cyclical trends).
2. Theoretical Resource Consumption (Memory overhead of FAISS indices, latency cost of search).
3. Pressure State Calculation (Mapping physical metrics to normalized [0,1] state vectors).
"""

import os
import joblib
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, TYPE_CHECKING, Any
from scipy.stats import gamma, truncnorm  # for dynamic distribution sampling and truncation

# Import project-specific modules
from src import config
from src.core import Action, ResourcePressureVector

# Use TYPE_CHECKING to avoid circular imports at runtime, while allowing type hints
if TYPE_CHECKING:
    from src.system import kNNMTSystem


class ProductionConstraintSimulator:
    """
    Simulates resource pressure changes in a production environment.
    This class is responsible for calculating the ResourcePressureVector Φ_t based on
    system actions and simulated external factors like traffic patterns.

    It maintains the internal state of the simulated server (e.g., current memory usage,
    request latency history) and evolves it over time steps.
    """

    def __init__(self, knn_system: 'kNNMTSystem', is_debug: bool=False):
        """
        Initializes the simulator with baseline parameters and cost models.

        Args:
            knn_system (kNNMTSystem): Reference to the kNN system to access datastore properties (size, dimension).
            is_debug (bool): If True, enables verbose logging.
        """
        self.is_debug = is_debug
        if self.is_debug: print("[Info] Initializing Production-Aware Constraint Simulator...")

        # --- 1. Load Baseline and Weight Parameters from Config ---
        # These parameters define the Service Level Agreement (SLA) limits and
        # the weights used to calculate the normalized pressure vector.
        sim_params = config.SIMULATOR_PARAMS
        self.L_SLA = sim_params["Latency_SLA"]
        self.M_avail = sim_params["Memory_Avail"]
        self.R_opt = sim_params["Throughput_opt"]

        self.w1, self.w2 = sim_params["w_latency_current"], sim_params["w_latency_derivative"]
        self.w3, self.w4 = sim_params["w_memory_current"], sim_params["w_memory_derivative"]
        self.w5_deficit = sim_params["w_throughput_current_deficit"]
        self.w5_surplus = sim_params["w_throughput_current_surplus"]
        self.w6 = sim_params["w_throughput_offset"]

        # --- 2. Initialize Internal State Variables ---
        self.fixed_memory_cost_mb = 0.0
        self.current_total_memory_mb = 0.0
        self.last_total_memory_mb = 0.0
        self.current_time_step: int = 0

        # --- 3. Initialize Sliding Windows for Derivative Calculation ---
        # Used to smooth metrics and calculate rates of change (derivatives) for the pressure vector.
        self.window_size = sim_params["sliding_window_size"]
        self.decay_factor = sim_params["decay_factor"]
        self.latency_history = deque(maxlen=self.window_size)
        self.throughput_history = deque(maxlen=self.window_size)

        # --- Statistics Buffers ---
        # Used to collect data distributions during generation to verify against simulation goals.
        self.latency_stats_buffer = []  # Collects generated latency values
        self.throughput_stats_buffer = []  # Collects generated throughput values

        # --- 4. Link to kNN System for Datastore Properties ---
        self.n_vectors = knn_system.datastore_size
        self.dim = knn_system.embedding_dim

        # --- 5. Pre-calculate Theoretical Memory Costs ---
        # Estimates the static RAM usage for different index types to simulate realistic memory baselines.
        self.fixed_memory_costs_lookup: Dict[str, float] = {}
        self._calculate_all_theoretical_memory_costs()

        # Define M_base (baseline memory usage) and simulate server total memory.
        self.M_base = sum(self.fixed_memory_costs_lookup.values())
        if self.is_debug: print(f"[Info] Combined Base Memory (M_base) for all indices: {self.M_base:.2f} MB")

        # Define server memory range relative to M_base.
        # This allows simulating different hardware capacities.
        self.server_mem_range = (self.M_base * 1.1, self.M_base * 2.5) # Server RAM is 10%-150% larger than base index memory
        self.M_total = 0.0  # Placeholder, will be set in reset()

        # --- 6. Load Pre-trained Performance Cost Models ---
        # Loads regression models (e.g., Random Forest or Linear Regression) that predict
        # the latency and throughput cost of kNN searches based on 'k' and 'concurrency'.
        if self.is_debug: print("[Info] Loading pre-trained performance cost models...")
        self.models: Dict[str, Dict[str, Any]] = {}
        model_path = config.PATHS["performance_models_dir"]
        index_types = ['exact', 'hnsw', 'ivf_pq']

        for index_type in index_types:
            try:
                t_model_path = os.path.join(model_path, f"model_throughput_{index_type}.joblib")
                l_model_path = os.path.join(model_path, f"model_latency_{index_type}.joblib")
                self.models[index_type] = {
                    'throughput': joblib.load(t_model_path),
                    'latency': joblib.load(l_model_path)
                }
                if self.is_debug: print(f"\t- [Success] Loaded performance model for {index_type.upper()}")
            except FileNotFoundError:
                if self.is_debug: print(f"\t- [Warning] Performance model for {index_type.upper()} not found. Costs will be zero.")

        # Load simulation parameters for traffic generation
        self.variation_amplitude = config.PRODUCTION_SIMULATOR_PARAMS["variation_amplitude"]
        self.noise_std = config.PRODUCTION_SIMULATOR_PARAMS["noise_std"]
        self.mean_load_alpha = config.PRODUCTION_SIMULATOR_PARAMS["mean_load_alpha"]
        self.mean_load_beta = config.PRODUCTION_SIMULATOR_PARAMS["mean_load_beta"]
        self.mean_load_min = config.PRODUCTION_SIMULATOR_PARAMS["mean_load_min"]
        self.mean_load_max = config.PRODUCTION_SIMULATOR_PARAMS["mean_load_max"]
        self.traffic_min = config.PRODUCTION_SIMULATOR_PARAMS["traffic_min"]
        self.traffic_max = config.PRODUCTION_SIMULATOR_PARAMS["traffic_max"]

        self.reset()

    def reset(self):
        """
        Resets the simulator's state for a new translation episode.
        This simulates the task running on a new, randomly configured server instance
        to ensure diversity in the training data.
        """

        # Sample a new total server memory for this episode
        self.M_total = np.random.uniform(self.server_mem_range[0], self.server_mem_range[1])

        self.current_time_step = 0
        self.latency_history.clear()
        self.throughput_history.clear()

        # Clear the statistics buffer
        self.latency_stats_buffer.clear()
        self.throughput_stats_buffer.clear()

        # Initialize history with random baseline values
        for _ in range(self.window_size):
            initial_latency = np.random.uniform(0.3 * self.L_SLA, 0.5 * self.L_SLA)
            initial_throughput = np.random.uniform(0.4 * self.R_opt, 0.7 * self.R_opt)
            self.latency_history.append(initial_latency)
            self.throughput_history.append(initial_throughput)

        # Initial memory usage includes the base cost of loading indices plus some random overhead
        self.current_total_memory_mb = self.M_base + np.random.uniform(0, 0.2 * self.M_base)
        self.last_total_memory_mb = self.current_total_memory_mb - np.random.uniform(0, 0.1 * self.M_base)

    def _calculate_all_theoretical_memory_costs(self):
        """
        Pre-computes theoretical static memory costs for all available FAISS index types.
        Populates self.fixed_memory_costs_lookup.
        """
        if self.is_debug: print("[Info] Pre-calculating theoretical fixed memory costs for FAISS indexes...")
        for index_type in ['none', 'exact', 'hnsw', 'ivf_pq']:
            cost = self._calculate_theoretical_fixed_memory_mb(index_type)
            self.fixed_memory_costs_lookup[index_type] = cost
            if self.is_debug: print(f"\t- {index_type.upper()}: {cost:.2f} MB (Theoretical)")

    def _calculate_theoretical_fixed_memory_mb(self, index_type: str, params: dict = {}) -> float:
        """
        Calculates the estimated static RAM usage (in MB) for a specific FAISS index structure.

        Args:
            index_type (str): Type of index ('exact', 'hnsw', 'ivf_pq').
            params (dict): Optional parameters for the index (e.g., M for HNSW).

        Returns:
            float: Estimated size in MB.
        """
        if index_type == 'exact':
            # IndexFlatL2: Stores raw vectors. O(n * d).
            return (self.n_vectors * self.dim * 4) / (1024 * 1024)  # float32 = 4 bytes
        elif index_type == 'hnsw':
            # HNSW: Stores vectors + graph structure.
            M = params.get('M', 32) # Default M from kNNMTSystem
            # Rough estimate: vectors + adjacency lists (M links per node)
            bytes_per_vector = 1.1 * (4 * self.dim + 8 * M) # 1.1 as structural overhead factor
            return (self.n_vectors * bytes_per_vector) / (1024 * 1024)
        elif index_type == 'ivf_pq':
            # IVF-PQ: Stores quantized codes and inverted lists.
            nlist = params.get('nlist', min(100, self.n_vectors // 10))
            nbits = params.get('nbits', 8)
            m = params.get('m', self.dim // 32 if self.dim % 32 == 0 else 16)
            total_bytes = 0

            # 1. Coarse quantizer memory (Centroids)
            quantizer_bytes = nlist * self.dim * 4
            total_bytes += quantizer_bytes

            # 2. PQ codebook memory (Sub-quantizers)
            ksub = 1 << nbits
            pq_centroids_bytes = m * ksub * (self.dim // m) * 4
            total_bytes += pq_centroids_bytes

            # 3. Inverted lists memory (PQ code + ID)
            code_size = m * nbits // 8
            invlist_entry_size = code_size + 8 # 8 bytes for ID (int64)
            invlists_bytes = self.n_vectors * invlist_entry_size
            total_bytes += invlists_bytes

            # 4. Metadata and allocator overhead estimate
            total_bytes *= 1.065 # 1.5% metadata + 5% allocator overhead

            return total_bytes / (1024 * 1024)
        return 0.0

    def _calculate_search_memory_mb(self, action: Action, concurrency: float) -> float:
        """
        Calculates the dynamic memory cost of concurrent search operations.
        This models the temporary buffers used by FAISS during query execution.

        Args:
            action (Action): The current action being simulated.
            concurrency (float): Estimated number of concurrent users.

        Returns:
            float: Estimated dynamic memory usage in MB.
        """
        k = action.k
        if k == 0 or concurrency == 0:
            return 0.0

        # --- Faiss Search Parameters ---
        efSearch = 32   # HNSW: size of the dynamic list of candidates
        nprobe = 32     # IVFPQ: number of inverted lists to visit

        # --- Per-Query Dynamic Memory Calculation (in bytes) ---
        single_query_dynamic_mem_bytes = 0
        bytes_per_id = 8  # int64 for node IDs/indices
        bytes_per_dist = 4 # float32 for distances

        if action.index_type == 'hnsw':
            # 1. Candidate Pool (Priority Queue)
            candidate_pool_mem = efSearch * (bytes_per_dist + bytes_per_id)
            # 2. Visited Set (Hash Set)
            visited_set_mem = efSearch * bytes_per_id * 1.5 # 1.5 overhead factor
            single_query_dynamic_mem_bytes = candidate_pool_mem + visited_set_mem

        elif action.index_type == 'ivf_pq':
            # IVF-PQ involves scanning inverted lists and decoding vectors.
            nlist = min(100, self.n_vectors // 10)
            avg_vectors_per_list = self.n_vectors / nlist

            # Estimate worst-case list size to scan
            max_list_size = avg_vectors_per_list * 1.5
            decoded_vectors_buffer_mem = max_list_size * self.dim * bytes_per_dist

            # Final top-k selection heap
            result_heap_mem = k * (bytes_per_dist + bytes_per_id)
            single_query_dynamic_mem_bytes = decoded_vectors_buffer_mem + result_heap_mem

        elif action.index_type == 'exact':
            # IndexFlatL2 mainly uses memory for the result heap.
            result_heap_mem = k * (bytes_per_dist + bytes_per_id)
            single_query_dynamic_mem_bytes = result_heap_mem

        # Total dynamic search memory scales with concurrency
        total_search_mem_bytes = concurrency * single_query_dynamic_mem_bytes

        # Convert to Megabytes
        return total_search_mem_bytes / (1024 * 1024)

    def _map_pressure_to_concurrency(self, pressure_norm: float) -> float:
        """
        Maps the abstract pressure norm [0,1] to a concrete concurrency level.
        This provides a proxy for user load to estimate dynamic costs.

        Args:
            pressure_norm (float): L2 norm of the current pressure vector.

        Returns:
            float: Simulated concurrency value.
        """
        min_c = config.SIMULATOR_PARAMS["min_concurrency"]
        max_c = config.SIMULATOR_PARAMS["max_concurrency"]
        # Linear interpolation based on pressure
        return min_c + (max_c - min_c) * (pressure_norm)

    def _sigmoid(self, x: float) -> float:
        """
        Standard sigmoid function to map values to (0, 1).
        Used for normalizing pressure metrics.
        """
        x = np.clip(x, -700, 700)  # Prevent exp overflow
        return 1.0 / (1.0 + np.exp(-x))

    def simulate_traffic_pattern(self) -> Dict[str, float]:
        """
        Simulates realistic production traffic patterns using a stochastic process.
        The traffic model consists of:
        1. Base Load: Sampled from a Gamma distribution (long-tail).
        2. Cyclical Variation: A sinusoidal wave simulating daily peaks/troughs.
        3. Random Noise: Gaussian noise for unpredictability.

        Returns:
            Dict[str, float]: Current 'latency' (ms) and 'throughput' (RPS) baselines.
        """

        # 1. Sample a mean load from a Gamma distribution for realistic skew
        mean_load = gamma.rvs(self.mean_load_alpha, scale=1/self.mean_load_beta, size=1)[0]
        mean_load = np.clip(mean_load, self.mean_load_min, self.mean_load_max)

        # 2. Calculate daily cycle variation (24-hour period mapped to steps)
        hour_of_day = (self.current_time_step * 0.1) % 24 # Assume 1 step = 0.1 hour simulated time
        daily_variation = np.sin((hour_of_day - 8) * np.pi / 12) # Peak around 14:00 (2pm)

        # 3. Combine components into a traffic multiplier
        traffic_multiplier = mean_load * (1 + daily_variation * self.variation_amplitude) + np.random.normal(0, self.noise_std)

        # Clip multiplier to valid range
        traffic_multiplier = np.clip(traffic_multiplier, self.traffic_min, self.traffic_max)

        # Apply multiplier to baseline SLA/Optimal values
        # High traffic -> High Latency, Low Throughput capability relative to optimum
        current_latency = self.L_SLA * traffic_multiplier
        current_throughput = self.R_opt / traffic_multiplier

        # Collect statistics for coverage verification
        self.latency_stats_buffer.append(current_latency)
        self.throughput_stats_buffer.append(current_throughput)

        return {
            'latency': max(1.0, current_latency),
            'throughput': max(1.0, current_throughput),
        }

    def update_resource_metrics(self, action: Action, pressure_norm: float) -> Dict[str, float]:
        """
        Updates the internal resource metrics based on the simulation model and the chosen action.
        This advances the simulation by one time step.

        Args:
            action (Action): The action taken by the system (e.g., kNN search).
            pressure_norm (float): The current magnitude of system pressure.

        Returns:
            Dict[str, float]: Updated 'latency' and 'throughput' metrics.
        """
        self.current_time_step += 1

        # 1. Simulate base environmental traffic
        base_metrics = self.simulate_traffic_pattern()
        
        # 2. Calculate additional cost incurred by the specific action
        action_cost = self._calculate_action_cost(action, pressure_norm)

        # 3. Apply costs to metrics
        current_metrics = {
            'latency': min(5000, base_metrics['latency'] + action_cost['latency_cost']),
            'throughput': max(1.0, base_metrics['throughput'] - action_cost['throughput_cost']),
        }

        # 4. Simulate Memory Usage
        self.last_total_memory_mb = self.current_total_memory_mb
        concurrency = self._map_pressure_to_concurrency(pressure_norm)
        search_mem = self._calculate_search_memory_mb(action, concurrency)

        # Simulate background memory noise/fluctuations (e.g., other apps)
        time_factor = np.sin(self.current_time_step * np.pi / 120) 
        other_apps_space = self.M_total - self.M_base 
        other_apps_usage = other_apps_space * (0.5 + 0.3 * time_factor) 
        system_noise = np.random.uniform(-self.M_base * 0.005, self.M_base * 0.005)

        self.current_total_memory_mb = max(self.M_base + search_mem + other_apps_usage + system_noise, 0)

        # 5. Update History Windows with decay
        if self.latency_history:
            decayed_history = [val * (1 - self.decay_factor) for val in self.latency_history]
            self.latency_history = deque(decayed_history, maxlen=self.latency_history.maxlen)

        self.latency_history.append(current_metrics['latency'] * self.decay_factor)
        self.throughput_history.append(current_metrics['throughput'])

        return current_metrics

    def _calculate_action_cost(self, action: Action, pressure_norm: float) -> Dict[str, float]:
        """
        Predicts the latency and throughput penalty of an action using pre-trained cost models.

        Args:
            action (Action): The action to evaluate.
            pressure_norm (float): Current system pressure.

        Returns:
            Dict[str, float]: 'latency_cost' and 'throughput_cost'.
        """
        if action.k == 0 or action.index_type not in self.models:
            return {'latency_cost': 0, 'throughput_cost': 0}

        concurrency = self._map_pressure_to_concurrency(pressure_norm)
        model_input = pd.DataFrame([[action.k, concurrency]], columns=['k', 'concurrency'])

        try:
            models = self.models[action.index_type]
            latency_cost = models['latency'].predict(model_input)[0]
            throughput_cost = models['throughput'].predict(model_input)[0]
        except Exception as e:
            if self.is_debug: print(f"[ERROR] Cost prediction failed for {action.index_type}: {e}")
            return {'latency_cost': 0, 'throughput_cost': 0}

        return {
            'latency_cost': max(0, latency_cost),
            'throughput_cost': max(0, throughput_cost)
        }

    def compute_pressure_vector(self) -> ResourcePressureVector:
        """
        Computes the normalized ResourcePressureVector based on current metrics.
        This implements the mathematical definition of Pressure State (Phi) from the paper.

        Returns:
            ResourcePressureVector: Components (latency, memory, throughput) normalized to (0, 1).
        """
        if not self.latency_history:
            return ResourcePressureVector(0.1, 0.1, 0.1) # Default low pressure

        # --- Calculate current values and derivatives ---
        L_t = np.mean(list(self.latency_history))
        R_t = np.mean(list(self.throughput_history))
        dL_dt = abs(self.latency_history[-1] - self.latency_history[-2]) if len(self.latency_history) >= 2 else 0.0

        M_t = self.current_total_memory_mb
        M_dot_t = self.current_total_memory_mb - self.last_total_memory_mb

        # --- Apply Sigmoid Mappings ---
        # Latency Pressure: Weighted combination of absolute latency and its derivative relative to SLA.
        lat_pressure = self._sigmoid(self.w1 * L_t / self.L_SLA + self.w2 * dL_dt / self.L_SLA)
        # Memory Pressure: Weighted combination of absolute usage and derivative relative to total capacity.
        mem_pressure = self._sigmoid(self.w3 * M_t / self.M_total + self.w4 * M_dot_t / self.M_total)

        # Throughput Pressure: Asymmetric handling for deficit vs surplus.
        throughput_gap_ratio = (self.R_opt - R_t) / self.R_opt
        if throughput_gap_ratio > 0:
            # Case 1: Deficit (R_t < R_opt). Steep penalty.
            thr_pressure = self._sigmoid(self.w5_deficit * abs(throughput_gap_ratio) - self.w6)
        else:
            # Case 2: Surplus (R_t >= R_opt). Gentle relief.
            thr_pressure = self._sigmoid(self.w5_surplus * throughput_gap_ratio - self.w6)

        # --- Clip to ensure range (0, 1) ---
        return ResourcePressureVector(
            latency_pressure=np.clip(lat_pressure, 1e-6, 1 - 1e-6),
            memory_pressure=np.clip(mem_pressure, 1e-6, 1 - 1e-6),
            throughput_pressure=np.clip(thr_pressure, 1e-6, 1 - 1e-6)
        )

    # --- Distributed Verification Method ---
    def compute_distribution_stats(self, target_multi_env_range: Dict[str, tuple] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculates statistics of the generated simulation data to verify if it covers the
        intended range of operating conditions. Used by the optimization script to tune parameters.

        Args:
            target_multi_env_range: Optional target ranges to check against.

        Returns:
            Dict: Statistics (min, max, mean, std, skew) for latency and throughput.
        """
        if not self.latency_stats_buffer:
            if self.is_debug: print("[Warning] No data in buffer for stats computation.")
            return {}

        lat_array = np.array(self.latency_stats_buffer)
        thr_array = np.array(self.throughput_stats_buffer)

        stats = {
            'latency': {
                'min': lat_array.min(),
                'max': lat_array.max(),
                'mean': lat_array.mean(),
                'std': lat_array.std(),
                'skew': np.mean((lat_array - lat_array.mean())**3) / (lat_array.std()**3)
            },
            'throughput': {
                'min': thr_array.min(),
                'max': thr_array.max(),
                'mean': thr_array.mean(),
                'std': thr_array.std(),
                'skew': np.mean((thr_array - thr_array.mean())**3) / (thr_array.std()**3)
            }
        }

        # Check against target ranges if provided
        if target_multi_env_range:
            for metric in ['latency', 'throughput']:
                target_min, target_max = target_multi_env_range[metric]
                if stats[metric]['max'] < target_max * 0.95:
                    if self.is_debug: print(f"[Warning] Coverage insufficient for {metric}: max={stats[metric]['max']:.2f} < target {target_max:.2f}. Increase alpha/beta or amplitude.")
                if abs(stats[metric]['skew']) > 2.0:
                    if self.is_debug: print(f"[Warning] Skew too high for {metric}: {stats[metric]['skew']:.2f}. Reduce noise_std or use stricter clip.")

        return stats
