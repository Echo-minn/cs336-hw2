from cs336_basics.model import BasicsTransformerLM
import torch
import timeit
import argparse
import random
import numpy as np


def get_device():
    """Get the best available device (CUDA if available, otherwise CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def benchmark_model(d_model: int, d_ff: int, num_layers: int, num_heads: int, context_length: int, 
                   num_steps: int, num_warmups: int = 5, forward_only: bool = False, batch_size: int = 4):
    """
    Benchmark the forward and backward passes of a BasicsTransformerLM model.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        context_length: Maximum sequence length
        num_steps: Number of steps to benchmark
        num_warmups: Number of warm-up steps before timing
        forward_only: If True, only benchmark forward pass; if False, benchmark both forward and backward
        batch_size: Batch size for the benchmark
    """
    device = get_device()
    
    # Initialize the model
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0,
    )
    model = model.to(device)
    model.train()  # Set to training mode for backward pass
    
    # Generate a random batch of data
    # Create random token IDs within the vocabulary range
    input_ids = torch.randint(0, 10000, (batch_size, context_length), device=device)
    
    # Initialize optimizer for backward pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print(f"Benchmarking model with:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    print(f"  context_length: {context_length}")
    print(f"  batch_size: {batch_size}")
    print(f"  device: {device}")
    print(f"  forward_only: {forward_only}")
    print(f"  num_warmups: {num_warmups}")
    print(f"  num_steps: {num_steps}")
    print()
    
    # Warm-up steps
    print("Running warm-up steps...")
    for _ in range(num_warmups):
        optimizer.zero_grad()
        outputs = model(input_ids)
        if not forward_only:
            loss = outputs.mean()  # Simple loss for benchmarking
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    print("Warm-up complete. Starting benchmark...")
    
    # Benchmark timing
    times = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Start timing
        start_time = timeit.default_timer()
        
        # Forward pass
        outputs = model(input_ids)
        
        if not forward_only:
            # Backward pass
            loss = outputs.mean()  # Simple loss for benchmarking
            loss.backward()
            optimizer.step()
        
        # Synchronize if using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize() # Wait for CUDA threads to finish (important!)
        
        # End timing
        end_time = timeit.default_timer()
        step_time = end_time - start_time
        times.append(step_time)
        
        if (step + 1) % 10 == 0:
            print(f"Completed step {step + 1}/{num_steps}")
    
    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print("\nBenchmark Results:")
    print(f"  Mean time per step: {mean_time:.6f} seconds")
    print(f"  Std time per step: {std_time:.6f} seconds")
    print(f"  Min time per step: {min_time:.6f} seconds")
    print(f"  Max time per step: {max_time:.6f} seconds")
    print(f"  Total benchmark time: {np.sum(times):.6f} seconds")
    
    # Calculate throughput
    if forward_only:
        print(f"  Throughput: {batch_size / mean_time:.2f} sequences/second")
    else:
        print(f"  Throughput: {batch_size / mean_time:.2f} sequences/second (forward + backward)")
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'total_time': np.sum(times),
        'times': times
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark BasicsTransformerLM model')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--context_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps to benchmark')
    parser.add_argument('--num_warmups', type=int, default=5, help='Number of warm-up steps')
    parser.add_argument('--forward_only', action='store_true', help='Only benchmark forward pass')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    results = benchmark_model(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_length=args.context_length,
        num_steps=args.num_steps,
        num_warmups=args.num_warmups,
        forward_only=args.forward_only,
        batch_size=args.batch_size
    )
    
    return results


if __name__ == "__main__":
    main()
