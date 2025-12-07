#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /config.py

import os

# Performance Models Directory
PERFORMANCE_MODELS_DIR = "./fitted_performance_models"
MEMORY_ANALYSIS_REPORT = "./comprehensive_memory_analysis_report.json"

# Benchmark Test Results
BENCHMARK_RESULTS = "./final_benchmark_results.csv"

# Make sure all required files exist
def validate_prerequisites():
    """Verify that all prerequisite files exist"""
    required_files = [
        PERFORMANCE_MODELS_DIR,
        MEMORY_ANALYSIS_REPORT,
        BENCHMARK_RESULTS
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Error] Missing required files: {file_path}")
    
    print("[Successful] All prerequisites verified passed.")