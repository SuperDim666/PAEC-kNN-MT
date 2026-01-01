# File: src/simulation/resource_monitor.py
# --- This is a new file (Revised Version) ---

import numpy as np
import torch
import psutil
from collections import deque
from typing import Tuple

import time, collections, threading

# Import project-specific modules
# config: Contains global simulation parameters (weights, baselines like L_SLA).
# ResourcePressureVector: Data structure for the pressure state component (Phi_t).
from src import config
from src.core import ResourcePressureVector

class ThroughputMonitor:
    """
    A thread-safe utility for monitoring real-time system throughput (Requests Per Second).

    It maintains a sliding window of request timestamps to calculate the instantaneous
    throughput (R_t) accurately. This is used during the evaluation phase to measure
    actual system performance against the optimal throughput baseline (R_opt).
    """
    def __init__(self, window_size=10.0):
        """
        Initializes the throughput monitor.

        Args:
            window_size (float): The duration of the sliding window in seconds.
                                 Requests older than this window are discarded.
        """
        self.window_size = window_size
        # Deque to store timestamps of completed requests
        self.requests = collections.deque()
        # Lock to ensure thread safety during recording and calculation
        self.lock = threading.Lock()
        self._running = True
        # Background thread to periodically clean up old timestamps to prevent memory growth
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def record_request_completion(self):
        """
        Records the timestamp of a completed request (or generation step).
        Should be called by the system immediately after a step is finished.
        """
        with self.lock:
            self.requests.append(time.time())

    def get_real_time_r(self):
        """
        Calculates the current real-time throughput (R_t).

        Returns:
            float: The number of requests processed in the last 'window_size' seconds,
                   normalized by the window size (requests/sec).
        """
        with self.lock:
            now = time.time()
            # Count requests that fall within the active window
            count = sum(1 for t in self.requests if now - t <= self.window_size)
            # Avoid division by zero
            return count / self.window_size if self.window_size > 0 else 0.0

    def _cleanup_loop(self):
        """
        Internal background loop that periodically removes expired timestamps
        from the deque to maintain memory efficiency.
        """
        while self._running:
            # Sleep for half the window size to avoid excessive locking
            time.sleep(self.window_size / 2)
            now = time.time()
            with self.lock:
                # Remove timestamps older than the window size
                while self.requests and now - self.requests[0] > self.window_size:
                    self.requests.popleft()

    def stop(self):
        """
        Stops the background cleanup thread gracefully.
        """
        self._running = False
        self._cleanup_thread.join()

class RealtimeResourceMonitor:
    """
    Monitors real hardware resource usage and computes the Pressure State Vector (Phi_t).

    Unlike the `ProductionConstraintSimulator`, which simulates an environment for training,
    this class measures *actual* system metrics (Latency, Memory, Throughput) during
    inference/validation. It acts as the sensor for the PAEC control loop in real deployments.
    """

    def __init__(self):
        """
        Initializes the monitor by loading baselines and weights from the configuration.
        It also detects the execution environment (CPU vs. GPU) to select the
        appropriate memory monitoring strategy.
        """
        # --- 1. Detect Environment and Load Calibrated Baselines ---
        # Determine if we are running on CUDA (and if configured to do so)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu")
        self.is_gpu_mode = (self.device.type == "cuda")
        
        # Load simulation parameters derived from `scripts/00_calibrate_baselines.py`
        sim_params = config.SIMULATOR_PARAMS
        self.l_sla = sim_params["Latency_SLA"]      # Latency Service Level Agreement (ms)
        # self.m_avail is determined dynamically based on hardware limits
        self.R_opt = sim_params["Throughput_opt"]   # Optimal Throughput Baseline (req/s)

        # --- 2. Load Calculation Weights ---
        # These weights control the sensitivity of the pressure vector to metric changes.
        self.w1, self.w2 = sim_params["w_latency_current"], sim_params["w_latency_derivative"]
        self.w3, self.w4 = sim_params["w_memory_current"], sim_params["w_memory_derivative"]
        self.w5_deficit = sim_params["w_throughput_current_deficit"]
        self.w5_surplus = sim_params["w_throughput_current_surplus"]
        self.w6 = sim_params["w_throughput_offset"]
        
        # --- 3. Initialize Sliding Windows for Derivative Calculation ---
        self.window_size = sim_params["sliding_window_size"]
        self.latency_history = deque(maxlen=self.window_size)
        # Initialize memory history with 0.0 to allow derivative calculation on first step
        self.memory_history = deque([0.0], maxlen=self.window_size)
        self.throughput_monitor = ThroughputMonitor(window_size=self.window_size)

    def _sigmoid(self, x: float) -> float:
        """
        Applies the standard sigmoid function to map values to the (0, 1) interval.
        Used to normalize raw physical metrics into abstract pressure scores.

        Args:
            x (float): Input value.

        Returns:
            float: Sigmoid(x).
        """
        return 1.0 / (1.0 + np.exp(-x))

    def _get_current_memory_state(self) -> Tuple[float, float]:
        """
        Measures the current memory usage (M_t) and total available memory (M_avail)
        directly from the hardware.

        Returns:
            Tuple[float, float]:
                - current_usage_mb: Current memory used by the process/device in MB.
                - total_available_mb: Total physical memory capacity in MB.
        """
        if self.is_gpu_mode:
            # For GPU, measure VRAM usage using PyTorch utilities.
            # M_t: Current memory allocated by PyTorch on the specific device.
            current_usage_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            # M_avail: Total VRAM capacity of the GPU.
            total_available_mb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)
        else:
            # For CPU, measure System RAM using psutil.
            # M_t: Resident Set Size (RSS) - actual physical memory used by the process.
            process = psutil.Process()
            current_usage_mb = process.memory_info().rss / (1024 ** 2)
            # M_avail: Total physical memory installed on the system.
            total_available_mb = psutil.virtual_memory().total / (1024 ** 2)
            
        return current_usage_mb, total_available_mb

    def record_step(self, latency_ms: float):
        """
        Records the metrics observed during a single translation step.
        This updates the history buffers used to calculate rates of change.

        Args:
            latency_ms (float): The measured latency of the step in milliseconds.
        """
        self.latency_history.append(latency_ms)
        self.throughput_monitor.record_request_completion()
        
        # Measure and record current memory usage at this specific moment
        current_memory_mb, _ = self._get_current_memory_state()
        self.memory_history.append(current_memory_mb)

    def compute_pressure_vector(self) -> ResourcePressureVector:
        """
        Computes the full Resource Pressure Vector (Phi_t) based on real-time hardware data.
        
        The calculation logic is mathematically identical to `ProductionConstraintSimulator`
        to ensure consistency between the training (simulated) and inference (real) phases.

        Returns:
            ResourcePressureVector: A dataclass containing normalized [0, 1] pressure
                                    scores for latency, memory, and throughput.
        """
        # If no history exists yet, return a default low-pressure state
        if not self.latency_history:
            return ResourcePressureVector(0.1, 0.1, 0.1)

        # --- Get real-time hardware memory state ---
        M_t, M_avail = self._get_current_memory_state()

        # --- Calculate Metrics and Derivatives ---
        # L_t: Average latency over the recent window
        L_t = np.mean(list(self.latency_history))
        # R_t: Current throughput from the monitor
        R_t = self.throughput_monitor.get_real_time_r()

        # Derivatives (Rate of Change)
        L_dot_t = abs(self.latency_history[-1] - self.latency_history[-2]) if len(self.latency_history) >= 2 else 0.0
        M_dot_t = self.memory_history[-1] - self.memory_history[-2] if len(self.memory_history) >= 2 else 0.0

        # --- Apply Theoretical Pressure Formulas ---
        
        # Latency Pressure: Proportional to current latency and its growth rate relative to SLA
        lat_pressure_arg = self.w1 * (L_t / self.l_sla) + self.w2 * (L_dot_t / self.l_sla)
        
        # Memory Pressure: Proportional to usage ratio and growth rate relative to capacity
        # Note: M_avail is retrieved dynamically here, unlike the static config in the simulator.
        mem_pressure_arg = self.w3 * (M_t / M_avail) + self.w4 * (M_dot_t / M_avail)
        
        # Throughput Pressure: Asymmetric response to deviations from optimal throughput
        throughput_gap_ratio = (self.R_opt - R_t) / self.R_opt
        
        if throughput_gap_ratio > 0:
            # Case 1: Throughput Deficit (R_t < R_opt) -> High Pressure
            # Use a steeper curve (w5_deficit) to penalize performance drops heavily.
            thr_pressure = self._sigmoid(self.w5_deficit * abs(throughput_gap_ratio) - self.w6)
        else:
            # Case 2: Throughput Surplus (R_t >= R_opt) -> Low Pressure
            # Use a gentler curve (w5_surplus) to indicate resource availability.
            thr_pressure = self._sigmoid(self.w5_surplus * throughput_gap_ratio - self.w6)
        
        # Apply sigmoid to normalize latency and memory arguments
        lat_pressure = self._sigmoid(lat_pressure_arg)
        mem_pressure = self._sigmoid(mem_pressure_arg)
        
        # --- Construct Result ---
        # Clip values to ensure numerical stability (epsilon range)
        return ResourcePressureVector(
            latency_pressure=np.clip(lat_pressure, 1e-6, 1 - 1e-6),
            memory_pressure=np.clip(mem_pressure, 1e-6, 1 - 1e-6),
            throughput_pressure=np.clip(thr_pressure, 1e-6, 1 - 1e-6)
        )
    
    def reset(self):
        """
        Resets the monitor's history and internal state.
        This is typically called between independent evaluation samples to ensure
        measurements are not contaminated by previous runs.
        """
        self.latency_history.clear()
        self.memory_history.clear()
        
        # Restart the throughput monitor to clear its internal request deque
        if hasattr(self, 'throughput_monitor') and self.throughput_monitor is not None:
            self.throughput_monitor.stop()
            del self.throughput_monitor
        self.throughput_monitor = ThroughputMonitor(window_size=self.window_size)

    def teardown(self):
        """
        Releases resources and stops background threads.
        """
        self.latency_history.clear()
        self.memory_history.clear()
        self.throughput_monitor.stop()
        del self.latency_history
        del self.memory_history
        del self.throughput_monitor
    
    def __del__(self):
        """
        Destructor to ensure threads are stopped if teardown wasn't called explicitly.
        """
        self.teardown()
