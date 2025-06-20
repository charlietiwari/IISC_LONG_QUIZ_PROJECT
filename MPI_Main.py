#!/usr/bin/env python3
"""
MPI-Based Distributed Matrix Multiplication

Description: Implements distributed matrix multiplication using MPI for parallel computation
"""

import numpy as np
import time
from mpi4py import MPI
import sys
import argparse

class DistributedMatrixMultiplier:
    """
    A class to handle distributed matrix multiplication using MPI
    """
    
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
    def serial_matrix_multiply(self, A, B):
        """
        Standard serial matrix multiplication for benchmarking
        """
        return np.dot(A, B)
    
    def initialize_matrices(self, m, n, p, seed=42):
        """
        Initialize matrices A (m x n) and B (n x p) on root process
        """
        if self.rank == 0:
            np.random.seed(seed)
            A = np.random.rand(m, n).astype(np.float64)
            B = np.random.rand(n, p).astype(np.float64)
            return A, B
        else:
            return None, None
    
    def distribute_matrix_rows(self, A, B):
        """
        Distribute rows of matrix A among processes using scatter
        Matrix B is broadcasted to all processes
        """
        if self.rank == 0:
            m, n = A.shape
            _, p = B.shape
            
            # Calculate rows per process
            rows_per_process = m // self.size
            extra_rows = m % self.size
            
            # Create send counts and displacements for scatterv
            send_counts = [rows_per_process * n] * self.size
            displacements = [i * rows_per_process * n for i in range(self.size)]
            
            # Distribute extra rows to first few processes
            for i in range(extra_rows):
                send_counts[i] += n
                for j in range(i + 1, self.size):
                    displacements[j] += n
            
            # Flatten A for scatterv
            A_flat = A.flatten()
        else:
            A_flat = None
            send_counts = None
            displacements = None
            n = None
            p = None
        
        # Broadcast matrix dimensions
        n = self.comm.bcast(n, root=0)
        p = self.comm.bcast(p, root=0)
        
        # Broadcast send_counts to determine receive buffer size
        send_counts = self.comm.bcast(send_counts, root=0)
        
        # Prepare receive buffer
        recv_count = send_counts[self.rank]
        local_A_flat = np.empty(recv_count, dtype=np.float64)
        
        # Scatter rows of A
        self.comm.Scatterv([A_flat, send_counts, displacements, MPI.DOUBLE], 
                          local_A_flat, root=0)
        
        # Reshape local A
        local_rows = recv_count // n
        local_A = local_A_flat.reshape(local_rows, n)
        
        # Broadcast matrix B to all processes
        if self.rank == 0:
            B_bcast = B.copy()
        else:
            B_bcast = np.empty((n, p), dtype=np.float64)
        
        self.comm.Bcast(B_bcast, root=0)
        
        return local_A, B_bcast, local_rows
    
    def distributed_matrix_multiply(self, A, B):
        """
        Perform distributed matrix multiplication
        """
        start_time = time.time()
        
        if A is None or B is None:
            A = np.empty((0, 0))
            B = np.empty((0, 0))
            m, n, p = 0, 0, 0
        else:
            m, n = A.shape
            _, p = B.shape
        
        # Broadcast dimensions to all processes
        dims = self.comm.bcast((m, n, p) if self.rank == 0 else None, root=0)
        m, n, p = dims
        
        if m == 0 or n == 0 or p == 0:
            return np.empty((0, 0)), 0.0
        
        # Distribute matrix A and broadcast matrix B
        local_A, B_local, local_rows = self.distribute_matrix_rows(A, B)
        
        # Perform local matrix multiplication
        local_result = np.dot(local_A, B_local)
        
        # Gather results
        if self.rank == 0:
            # Prepare for gathering results
            rows_per_process = m // self.size
            extra_rows = m % self.size
            
            recv_counts = [rows_per_process * p] * self.size
            displacements = [i * rows_per_process * p for i in range(self.size)]
            
            for i in range(extra_rows):
                recv_counts[i] += p
                for j in range(i + 1, self.size):
                    displacements[j] += p
            
            result_flat = np.empty(m * p, dtype=np.float64)
        else:
            recv_counts = None
            displacements = None
            result_flat = None
        
        # Gather local results
        self.comm.Gatherv(local_result.flatten(), 
                         [result_flat, recv_counts, displacements, MPI.DOUBLE], 
                         root=0)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if self.rank == 0:
            result = result_flat.reshape(m, p)
            return result, execution_time
        else:
            return None, execution_time
    
    def benchmark_performance(self, matrix_sizes, num_runs=3):
        """
        Benchmark the distributed implementation against serial implementation
        """
        results = {
            'matrix_sizes': [],
            'serial_times': [],
            'distributed_times': [],
            'speedup': [],
            'efficiency': []
        }
        
        if self.rank == 0:
            print(f"Benchmarking with {self.size} processes...")
            print("Matrix Size | Serial Time | Distributed Time | Speedup | Efficiency")
            print("-" * 70)
        
        for m, n, p in matrix_sizes:
            serial_times = []
            distributed_times = []
            
            for run in range(num_runs):
                # Initialize matrices
                A, B = self.initialize_matrices(m, n, p, seed=42 + run)
                
                # Serial execution (only on root)
                if self.rank == 0:
                    start_time = time.time()
                    serial_result = self.serial_matrix_multiply(A, B)
                    serial_time = time.time() - start_time
                    serial_times.append(serial_time)
                
                # Distributed execution
                distributed_result, distributed_time = self.distributed_matrix_multiply(A, B)
                distributed_times.append(distributed_time)
                
                # Verify correctness (only on root)
                if self.rank == 0 and run == 0:
                    if np.allclose(serial_result, distributed_result, rtol=1e-10):
                        print(f"✓ Verification passed for {m}x{n} × {n}x{p}")
                    else:
                        print(f"✗ Verification failed for {m}x{n} × {n}x{p}")
            
            # Calculate averages
            avg_distributed_time = np.mean(distributed_times)
            
            if self.rank == 0:
                avg_serial_time = np.mean(serial_times)
                speedup = avg_serial_time / avg_distributed_time
                efficiency = speedup / self.size
                
                results['matrix_sizes'].append(f"{m}x{n}x{p}")
                results['serial_times'].append(avg_serial_time)
                results['distributed_times'].append(avg_distributed_time)
                results['speedup'].append(speedup)
                results['efficiency'].append(efficiency)
                
                print(f"{m:4}x{n:4}x{p:4} | {avg_serial_time:10.4f}s | {avg_distributed_time:15.4f}s | "
                      f"{speedup:7.2f} | {efficiency:9.2f}")
        
        return results if self.rank == 0 else None

def main():
    """
    Main function to run the distributed matrix multiplication
    """
    parser = argparse.ArgumentParser(description='Distributed Matrix Multiplication using MPI')
    parser.add_argument('--size', type=int, default=100, help='Matrix size (default: 100)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark tests')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs (default: 3)')
    
    args = parser.parse_args()
    
    multiplier = DistributedMatrixMultiplier()
    
    if args.benchmark:
        # Define test matrix sizes
        matrix_sizes = [
            (50, 50, 50),
            (100, 100, 100),
            (200, 200, 200),
            (300, 300, 300),
            (400, 400, 400)
        ]
        
        results = multiplier.benchmark_performance(matrix_sizes, args.runs)
        
        if multiplier.rank == 0:
            print("\nBenchmark completed!")
            print(f"Results saved with {multiplier.size} processes")
            
    else:
        # Single test run
        m = n = p = args.size
        A, B = multiplier.initialize_matrices(m, n, p)
        
        if multiplier.rank == 0:
            print(f"Testing {m}x{n} × {n}x{p} matrix multiplication with {multiplier.size} processes")
        
        result, exec_time = multiplier.distributed_matrix_multiply(A, B)
        
        if multiplier.rank == 0:
            print(f"Execution time: {exec_time:.4f} seconds")
            print(f"Result matrix shape: {result.shape}")

if __name__ == "__main__":
    main()