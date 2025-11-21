"""
Run all framework benchmarks and generate comparison report
"""

import json
import sys
from pathlib import Path

# Add benchmarks to path
sys.path.append(str(Path(__file__).parent / 'benchmarks'))

from pytorch_benchmark import run_benchmark as pytorch_benchmark
from tensorflow_benchmark import run_benchmark as tensorflow_benchmark
from jax_benchmark import run_benchmark as jax_benchmark


def run_all_benchmarks(epochs=5):
    """Run benchmarks for all frameworks"""

    print("ML FRAMEWORKS BENCHMARK SUITE - Avatar BCI Project")
    
    
    results = []
    
    # PyTorch
    print("\n[1/3] Running PyTorch benchmark...")
    results.append(pytorch_benchmark(device_type='cpu', epochs=epochs))
    
    # TensorFlow
    print("\n[2/3] Running TensorFlow benchmark...")
    results.append(tensorflow_benchmark(device_type='cpu', epochs=epochs))
    
    # JAX
    print("\n[3/3] Running JAX benchmark...")
    results.append(jax_benchmark(device_type='cpu', epochs=epochs))
    
    return results


def generate_report(results):
    """Generate comparison report"""

    print("BENCHMARK COMPARISON REPORT")

    # Table header
    print(f"\n{'Framework':<12} {'Train Time':<12} {'Avg Epoch':<12} {'Accuracy':<12} {'Inference':<12}")

    
    # Sort by inference time (fastest first)
    sorted_results = sorted(results, key=lambda x: x['avg_inference_time'])
    
    for r in sorted_results:
        print(f"{r['framework']:<12} "
            f"{r['total_train_time']:<11.2f}s "
            f"{r['avg_epoch_time']:<11.2f}s "
            f"{r['accuracy']:<11.2f}% "
            f"{r['avg_inference_time']:<11.2f}ms")
    

    
    # Winner analysis
    fastest_train = min(results, key=lambda x: x['total_train_time'])
    fastest_inference = min(results, key=lambda x: x['avg_inference_time'])
    most_accurate = max(results, key=lambda x: x['accuracy'])
    
    print(f"\n WINNERS:")
    print(f"  Fastest Training:  {fastest_train['framework']} ({fastest_train['total_train_time']:.2f}s)")
    print(f"  Fastest Inference: {fastest_inference['framework']} ({fastest_inference['avg_inference_time']:.2f}ms)")
    print(f"  Most Accurate:     {most_accurate['framework']} ({most_accurate['accuracy']:.2f}%)")
    

    
    return sorted_results


def save_results(results, output_file='ml-frameworks-benchmark/results/benchmark_results.json'):
    """Save results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n Results saved to: {output_file}")


if __name__ == "__main__":
    # Run all benchmarks
    results = run_all_benchmarks(epochs=5)
    
    # Generate comparison report
    sorted_results = generate_report(results)
    
    # Save results
    save_results(sorted_results)