# ML Frameworks Benchmark for Avatar BCI

## Overview
Comprehensive benchmarking comparison of TensorFlow, PyTorch, and JAX frameworks for EEG brainwave classification on Kubernetes.

## Purpose
This benchmark addresses issue #406 by comparing ML framework performance for complex brainwave data signals in both CPU and GPU modes.

## Structure
```
ml-frameworks-benchmark/
├── README.md                      # This file
├── run_all_benchmarks.py          # Run all benchmarks
├── benchmarks/                    # Framework specific benchmark scripts
│   ├── pytorch_benchmark.py
│   ├── tensorflow_benchmark.py
│   └── jax_benchmark.py
├── utils/                         # Shared utilities
│   └── data_loader.py
└── results/                       # Benchmark results and reports
    ├── benchmark_results.json
    └── BENCHMARK_REPORT.md
```

## Frameworks Tested
- **PyTorch**: CNN model with Adam optimizer
- **TensorFlow**: Keras Sequential CNN
- **JAX**: Functional neural network with JIT compilation

## Metrics Measured
- Training time
- Inference latency
- Model accuracy
- CPU vs GPU performance

## Usage

### Run All Benchmarks
```bash
python3 ml-frameworks-benchmark/run_all_benchmarks.py
```

### Run Individual Framework Benchmarks
```bash
# PyTorch
python3 ml-frameworks-benchmark/benchmarks/pytorch_benchmark.py

# TensorFlow
python3 ml-frameworks-benchmark/benchmarks/tensorflow_benchmark.py

# JAX
python3 ml-frameworks-benchmark/benchmarks/jax_benchmark.py
```

### View Results
- **JSON Results**: `ml-frameworks-benchmark/results/benchmark_results.json`
- **Detailed Report**: `ml-frameworks-benchmark/results/BENCHMARK_REPORT.md`

## Results Summary

**Winners on CPU:**
- Fastest Training: PyTorch (0.68s)
- Fastest Inference: JAX (1.65ms)
- Most Accurate: TensorFlow (19.50%)

See full report in `results/BENCHMARK_REPORT.md`

## Requirements

Install dependencies:
```bash
pip install torch tensorflow jax jaxlib numpy pandas
```

## Issue
Closes #406 