# IISC_LONG_QUIZ_PROJECT

# MPI-Based Distributed Matrix Multiplication Project

## Project Overview

This project implements a distributed matrix multiplication algorithm using the Message Passing Interface (MPI) in Python. The implementation demonstrates parallel computing concepts, performance optimization, and scalability analysis across multiple processes.

### Key Features

- **Distributed Computing**: Utilizes MPI for inter-process communication
- **Performance Benchmarking**: Compares serial vs. distributed execution
- **Scalability Analysis**: Tests performance across different process counts
- **Comprehensive Metrics**: Measures speedup, efficiency, and execution time
- **Automated Testing**: Includes test runners and analysis tools
- **Visualization**: Generates performance graphs and reports

## Architecture and Design

### Core Components

1. **DistributedMatrixMultiplier Class**
   - Handles MPI initialization and communication
   - Implements matrix distribution strategies
   - Performs parallel matrix multiplication
   - Collects and aggregates results

2. **Data Distribution Strategy**
   - **Row-wise partitioning**: Matrix A is divided by rows
   - **Broadcasting**: Matrix B is broadcast to all processes
   - **Load balancing**: Handles uneven matrix dimensions

3. **Communication Pattern**
   - `MPI.COMM_WORLD.Scatterv()`: Distributes matrix A rows
   - `MPI.COMM_WORLD.Bcast()`: Broadcasts matrix B
   - `MPI.COMM_WORLD.Gatherv()`: Collects partial results

### Algorithm Flow

```
1. Initialize MPI environment
2. Root process creates matrices A and B
3. Distribute rows of A among processes
4. Broadcast matrix B to all processes
5. Each process computes local matrix multiplication
6. Gather partial results at root process
7. Root process assembles final result matrix
```

## Mathematical Foundation

### Matrix Multiplication

For matrices A (m×n) and B (n×p), the result C (m×p) is computed as:

```
C[i][j] = Σ(k=0 to n-1) A[i][k] × B[k][j]
```

### Parallel Decomposition

The algorithm uses **row-wise decomposition**:
- Process 0 gets rows 0 to (m/p - 1)
- Process 1 gets rows (m/p) to (2m/p - 1)
- Process i gets rows (i×m/p) to ((i+1)×m/p - 1)

### Performance Metrics

1. **Speedup (S)**: `S = T_serial / T_parallel`
2. **Efficiency (E)**: `E = S / P` (where P = number of processes)
3. **Parallel Overhead**: Additional time due to communication

## Implementation Details

### Key Functions

#### `distribute_matrix_rows()`
```python
def distribute_matrix_rows(self, A, B):
    """
    Distributes rows of matrix A among processes using MPI Scatterv
    Broadcasts matrix B to all processes
    
    Returns:
        local_A: Local portion of matrix A
        B_local: Complete matrix B (broadcasted)
        local_rows: Number of rows assigned to this process
    """
```

#### `distributed_matrix_multiply()`
```python
def distributed_matrix_multiply(self, A, B):
    """
    Main distributed multiplication function
    
    Steps:
    1. Distribute matrix A and broadcast matrix B
    2. Perform local multiplication
    3. Gather results from all processes
    4. Return assembled result matrix
    """
```

### Error Handling

- **Matrix dimension validation**
- **Process count verification**
- **Memory allocation checks**
- **Communication timeout handling**

### Optimization Techniques

1. **Efficient Memory Usage**
   - Uses NumPy's contiguous arrays
   - Minimizes data copying
   - Proper memory alignment for MPI operations

2. **Communication Optimization**
   - Uses `Scatterv` for uneven data distribution
   - Broadcasts smaller matrix to reduce communication
   - Overlaps computation with communication where possible

3. **Load Balancing**
   - Distributes extra rows to first few processes
   - Handles matrices not evenly divisible by process count

## Performance Analysis

### Theoretical Performance

**Ideal Speedup**: Linear with number of processes
**Reality**: Limited by:
- Communication overhead
- Load imbalance
- Sequential portions (Amdahl's Law)

### Measured Performance Characteristics

1. **Small Matrices (< 100×100)**
   - Communication overhead dominates
   - Poor parallel efficiency
   - May be slower than serial

2. **Medium Matrices (100×100 to 500×500)**
   - Balanced computation vs. communication
   - Good speedup potential
   - Optimal for testing

3. **Large Matrices (> 500×500)**
   - Computation dominates
   - High parallel efficiency
   - Limited by memory

### Scalability Patterns

- **Strong Scaling**: Fixed problem size, increasing processes
- **Weak Scaling**: Problem size scales with processes
- **Efficiency Degradation**: Typically decreases with more processes

## File Structure

```
mpi_matrix_project/
├── distributed_matrix_mult.py
