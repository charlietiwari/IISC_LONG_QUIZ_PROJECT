#!/usr/bin/env python3
"""
Automated Test Runner and Benchmarking Script for MPI Matrix Multiplication
"""

import subprocess
import sys
import os
import time
import json
from datetime import datetime
import argparse
import platform

class MPITestRunner:
    """
    Automated test runner for MPI matrix multiplication benchmarks
    """
    
    def __init__(self):
        self.results_dir = "results"
        self.logs_dir = "logs"
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories"""
        for directory in [self.results_dir, self.logs_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def check_mpi_installation(self):
        """Check if MPI is properly installed"""
        try:
            result = subprocess.run(['mpiexec', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✓ MPI installation verified")
                return True
            else:
                print("✗ MPI installation check failed")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("✗ MPI not found or not responding")
            return False
    
    def check_python_dependencies(self):
        """Check if required Python packages are installed"""
        required_packages = ['numpy', 'mpi4py', 'matplotlib', 'pandas', 'seaborn']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package} is installed")
            except ImportError:
                print(f"✗ {package} is missing")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\nMissing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        return True
    
    def run_single_test(self, num_processes, matrix_size, timeout=300):
        """Run a single test with specified parameters"""
        cmd = [
            'mpiexec', '-n', str(num_processes),
            'python', 'distributed_matrix_mult.py',
            '--size', str(matrix_size)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✓ Test completed successfully in {end_time - start_time:.2f}s")
                return True, result.stdout, result.stderr
            else:
                print(f"✗ Test failed with return code {result.returncode}")
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"✗ Test timed out after {timeout}s")
            return False, "", "Test timed out"
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            return False, "", str(e)
    
    def run_benchmark_suite(self, process_counts, runs=3, timeout=600):
        """Run comprehensive benchmark suite"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_summary = {
            'timestamp': timestamp,
            'system_info': self.get_system_info(),
            'test_results': []
        }
        
        print(f"Starting benchmark suite at {datetime.now()}")
        print(f"Process counts: {process_counts}")
        print(f"Runs per test: {runs}")
        print("-" * 60)
        
        for num_processes in process_counts:
            print(f"\n=== Testing with {num_processes} processes ===")
            
            cmd = [
                'mpiexec', '-n', str(num_processes),
                'python', 'distributed_matrix_mult.py',
                '--benchmark', '--runs', str(runs)
            ]
            
            log_file = os.path.join(self.logs_dir, 
                                  f"benchmark_{num_processes}proc_{timestamp}.log")
            
            try:
                start_time = time.time()
                with open(log_file, 'w') as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, 
                                          text=True, timeout=timeout)
                end_time = time.time()
                
                test_result = {
                    'num_processes': num_processes,
                    'success': result.returncode == 0,
                    'execution_time': end_time - start_time,
                    'return_code': result.returncode,
                    'log_file': log_file
                }
                
                if result.returncode == 0:
                    print(f"✓ Benchmark completed in {end_time - start_time:.1f}s")
                    # Look for results file
                    results_file = self.find_latest_results_file(num_processes)
                    if results_file:
                        test_result['results_file'] = results_file
                        print(f"✓ Results saved to {results_file}")
                else:
                    print(f"✗ Benchmark failed with return code {result.returncode}")
                    print(f"Check log file: {log_file}")
                
                results_summary['test_results'].append(test_result)
                    
            except subprocess.TimeoutExpired:
                print(f"✗ Benchmark timed out after {timeout}s")
                test_result = {
                    'num_processes': num_processes,
                    'success': False,
                    'execution_time': timeout,
                    'return_code': -1,
                    'error': 'Timeout',
                    'log_file': log_file
                }
                results_summary['test_results'].append(test_result)
            
            except Exception as e:
                print(f"✗ Benchmark failed with exception: {e}")
                test_result = {
                    'num_processes': num_processes,
                    'success': False,
                    'execution_time': 0,
                    'return_code': -1,
                    'error': str(e),
                    'log_file': log_file
                }
                results_summary['test_results'].append(test_result)
        
        # Save summary
        summary_file = os.path.join(self.results_dir, f"test_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n=== Benchmark Suite Complete ===")
        print(f"Summary saved to: {summary_file}")
        
        return results_summary
    
    def find_latest_results_file(self, num_processes):
        """Find the most recent results file for given process count"""
        pattern = f"mpi_results_{num_processes}proc_"
        files = []
        
        for filename in os.listdir('.'):
            if filename.startswith(pattern) and filename.endswith('.json'):
                files.append(filename)
        
        if files:
            return max(files)  # Most recent by filename
        return None
    
    def get_system_info(self):
        """Collect system information"""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count()
        }
        
        try:
            # Try to get memory info on Windows
            if platform.system() == 'Windows':
                result = subprocess.run(['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        memory_bytes = int(lines[1].strip())
                        info['total_memory_gb'] = round(memory_bytes / (1024**3), 2)
        except:
            pass
        
        return info
    
    def run_scalability_analysis(self, max_processes=8, matrix_sizes=[100, 200, 300]):
        """Run scalability analysis with different process counts and matrix sizes"""
        print("=== Scalability Analysis ===")
        
        process_counts = [2**i for i in range(int(max_processes).bit_length()) if 2**i <= max_processes]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        scalability_results = []
        
        for size in matrix_sizes:
            for processes in process_counts:
                print(f"Testing {size}x{size} matrix with {processes} processes...")
                
                success, stdout, stderr = self.run_single_test(processes, size, timeout=300)
                
                result = {
                    'matrix_size': size,
                    'num_processes': processes,
                    'success': success,
                    'timestamp': datetime.now().isoformat()
                }
                
                if success:
                    # Extract timing information from stdout if available
                    lines = stdout.split('\n')
                    for line in lines:
                        if 'Execution time:' in line:
                            try:
                                time_str = line.split(':')[1].strip().split()[0]
                                result['execution_time'] = float(time_str)
                            except:
                                pass
                
                scalability_results.append(result)
                time.sleep(1)  # Brief pause between tests
        
        # Save scalability results
        scalability_file = os.path.join(self.results_dir, f"scalability_analysis_{timestamp}.json")
        with open(scalability_file, 'w') as f:
            json.dump(scalability_results, f, indent=2)
        
        print(f"Scalability analysis saved to: {scalability_file}")
        return scalability_results
    
    def generate_performance_report(self, results_files):
        """Generate performance analysis from results files"""
        if not results_files:
            print("No results files provided for analysis")
            return
        
        print("=== Generating Performance Analysis ===")
        
        # Create analysis script
        analysis_script = """
from performance_analyzer import PerformanceAnalyzer
import sys

analyzer = PerformanceAnalyzer()
results_files = sys.argv[1:]

for file in results_files:
    print(f"Analyzing {file}...")
    if analyzer.load_results(file):
        results = analyzer.results_data.get('results', {})
        num_processes = analyzer.results_data.get('metadata', {}).get('num_processes', 'unknown')
        
        # Generate plots
        analyzer.plot_performance_metrics(results, save_plots=True)
        
        # Generate report
        analyzer.generate_report(results, num_processes)

print("Performance analysis completed!")
"""
        
        # Write temporary analysis script
        with open('temp_analysis.py', 'w') as f:
            f.write(analysis_script)
        
        try:
            # Run analysis
            cmd = ['python', 'temp_analysis.py'] + results_files
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Performance analysis completed")
                print(result.stdout)
            else:
                print("✗ Performance analysis failed")
                print(result.stderr)
        
        finally:
            # Clean up temporary file
            if os.path.exists('temp_analysis.py'):
                os.remove('temp_analysis.py')
    
    def print_summary(self, results_summary):
        """Print a summary of test results"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        system_info = results_summary.get('system_info', {})
        print(f"System: {system_info.get('platform', 'Unknown')}")
        print(f"CPU Count: {system_info.get('cpu_count', 'Unknown')}")
        print(f"Memory: {system_info.get('total_memory_gb', 'Unknown')} GB")
        print(f"Timestamp: {results_summary.get('timestamp', 'Unknown')}")
        print()
        
        successful_tests = 0
        total_tests = len(results_summary.get('test_results', []))
        
        for test in results_summary.get('test_results', []):
            status = "✓" if test.get('success', False) else "✗"
            processes = test.get('num_processes', 'Unknown')
            exec_time = test.get('execution_time', 0)
            
            print(f"{status} {processes} processes - {exec_time:.1f}s")
            
            if test.get('success', False):
                successful_tests += 1
                if 'results_file' in test:
                    print(f"    Results: {test['results_file']}")
            else:
                if 'error' in test:
                    print(f"    Error: {test['error']}")
                if 'log_file' in test:
                    print(f"    Log: {test['log_file']}")
        
        print(f"\nSuccess Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")

def main():
    """Main function for test runner"""
    parser = argparse.ArgumentParser(description='MPI Matrix Multiplication Test Runner')
    parser.add_argument('--check', action='store_true', help='Check installation and dependencies')
    parser.add_argument('--test', type=int, nargs=2, metavar=('PROCESSES', 'SIZE'), 
                       help='Run single test with specified processes and matrix size')
    parser.add_argument('--benchmark', type=int, nargs='+', metavar='PROCESSES',
                       help='Run benchmark suite with specified process counts')
    parser.add_argument('--scalability', type=int, default=8, metavar='MAX_PROCESSES',
                       help='Run scalability analysis up to MAX_PROCESSES')
    parser.add_argument('--analyze', nargs='+', metavar='RESULTS_FILE',
                       help='Analyze existing results files')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs for benchmarks')
    
    args = parser.parse_args()
    
    runner = MPITestRunner()
    
    if args.check:
        print("=== Installation Check ===")
        mpi_ok = runner.check_mpi_installation()
        deps_ok = runner.check_python_dependencies()
        
        if mpi_ok and deps_ok:
            print("\n✓ All dependencies are satisfied!")
            print("You can now run the benchmarks.")
        else:
            print("\n✗ Some dependencies are missing.")
            print("Please install missing components before running tests.")
        return
    
    if args.test:
        processes, size = args.test
        print(f"=== Single Test: {processes} processes, {size}x{size} matrix ===")
        success, stdout, stderr = runner.run_single_test(processes, size)
        if success:
            print("Test output:")
            print(stdout)
        else:
            print("Test failed:")
            print(stderr)
        return
    
    if args.benchmark:
        process_counts = args.benchmark
        results_summary = runner.run_benchmark_suite(process_counts, args.runs)
        runner.print_summary(results_summary)
        return
    
    if args.scalability:
        runner.run_scalability_analysis(args.scalability)
        return
    
    if args.analyze:
        runner.generate_performance_report(args.analyze)
        return
    
    # Default: run installation check
    print("MPI Matrix Multiplication Test Runner")
    print("Use --help for available options")
    print("\nRunning installation check...")
    runner.check_mpi_installation()
    runner.check_python_dependencies()

if __name__ == "__main__":
    main()