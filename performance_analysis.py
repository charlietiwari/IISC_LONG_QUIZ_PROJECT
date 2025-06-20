#!/usr/bin/env python3
"""
Performance Analysis and Visualization Tool for MPI Matrix Multiplication
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import seaborn as sns

class PerformanceAnalyzer:
    """
    Analyzes and visualizes performance metrics for distributed matrix multiplication
    """
    
    def __init__(self, results_file=None):
        self.results_file = results_file
        self.results_data = []
        
    def load_results(self, filename):
        """Load performance results from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.results_data = json.load(f)
            print(f"Loaded results from {filename}")
            return True
        except FileNotFoundError:
            print(f"Results file {filename} not found")
            return False
        except json.JSONDecodeError:
            print(f"Invalid JSON in {filename}")
            return False
    
    def save_results(self, results, num_processes, filename=None):
        """Save performance results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mpi_results_{num_processes}proc_{timestamp}.json"
        
        # Add metadata
        results_with_meta = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_processes': num_processes,
                'description': 'MPI Matrix Multiplication Performance Results'
            },
            'results': results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results_with_meta, f, indent=2)
            print(f"Results saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving results: {e}")
            return None
    
    def analyze_scalability(self, results_files):
        """
        Analyze scalability across different numbers of processes
        """
        scalability_data = []
        
        for file in results_files:
            if self.load_results(file):
                metadata = self.results_data.get('metadata', {})
                results = self.results_data.get('results', {})
                num_processes = metadata.get('num_processes', 1)
                
                for i, size in enumerate(results.get('matrix_sizes', [])):
                    scalability_data.append({
                        'matrix_size': size,
                        'num_processes': num_processes,
                        'serial_time': results.get('serial_times', [0])[i] if i < len(results.get('serial_times', [])) else 0,
                        'distributed_time': results.get('distributed_times', [0])[i] if i < len(results.get('distributed_times', [])) else 0,
                        'speedup': results.get('speedup', [0])[i] if i < len(results.get('speedup', [])) else 0,
                        'efficiency': results.get('efficiency', [0])[i] if i < len(results.get('efficiency', [])) else 0
                    })
        
        return pd.DataFrame(scalability_data)
    
    def plot_performance_metrics(self, results, save_plots=True):
        """
        Create comprehensive performance visualization plots
        """
        if not results:
            print("No results to plot")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Execution Time Comparison
        ax1 = plt.subplot(2, 3, 1)
        matrix_indices = range(len(results['matrix_sizes']))
        width = 0.35
        
        ax1.bar([i - width/2 for i in matrix_indices], results['serial_times'], 
                width, label='Serial', color='skyblue', alpha=0.7)
        ax1.bar([i + width/2 for i in matrix_indices], results['distributed_times'], 
                width, label='Distributed', color='lightcoral', alpha=0.7)
        
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Serial vs Distributed Execution Time')
        ax1.set_xticks(matrix_indices)
        ax1.set_xticklabels(results['matrix_sizes'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(matrix_indices, results['speedup'], 'o-', color='green', linewidth=2, markersize=8)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No Speedup')
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Speedup vs Matrix Size')
        ax2.set_xticks(matrix_indices)
        ax2.set_xticklabels(results['matrix_sizes'], rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Efficiency
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(matrix_indices, results['efficiency'], 's-', color='purple', linewidth=2, markersize=8)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency')
        ax3.set_xlabel('Matrix Size')
        ax3.set_ylabel('Efficiency')
        ax3.set_title('Parallel Efficiency')
        ax3.set_xticks(matrix_indices)
        ax3.set_xticklabels(results['matrix_sizes'], rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Time vs Matrix Size (Log scale)
        ax4 = plt.subplot(2, 3, 4)
        matrix_sizes_numeric = [int(size.split('x')[0]) for size in results['matrix_sizes']]
        ax4.loglog(matrix_sizes_numeric, results['serial_times'], 'o-', label='Serial', linewidth=2)
        ax4.loglog(matrix_sizes_numeric, results['distributed_times'], 's-', label='Distributed', linewidth=2)
        ax4.set_xlabel('Matrix Size (N)')
        ax4.set_ylabel('Execution Time (seconds)')
        ax4.set_title('Execution Time vs Matrix Size (Log-Log)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Performance Gain
        ax5 = plt.subplot(2, 3, 5)
        performance_gain = [(s - d) / s * 100 for s, d in zip(results['serial_times'], results['distributed_times'])]
        ax5.bar(matrix_indices, performance_gain, color='orange', alpha=0.7)
        ax5.set_xlabel('Matrix Size')
        ax5.set_ylabel('Performance Gain (%)')
        ax5.set_title('Performance Improvement (%)')
        ax5.set_xticks(matrix_indices)
        ax5.set_xticklabels(results['matrix_sizes'], rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Summary Statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calculate summary statistics
        avg_speedup = np.mean(results['speedup'])
        max_speedup = np.max(results['speedup'])
        avg_efficiency = np.mean(results['efficiency'])
        total_serial_time = np.sum(results['serial_times'])
        total_distributed_time = np.sum(results['distributed_times'])
        total_time_saved = total_serial_time - total_distributed_time
        
        summary_text = f"""
        PERFORMANCE SUMMARY
        {'='*30}
        
        Average Speedup: {avg_speedup:.2f}x
        Maximum Speedup: {max_speedup:.2f}x
        Average Efficiency: {avg_efficiency:.2f}
        
        Total Serial Time: {total_serial_time:.2f}s
        Total Distributed Time: {total_distributed_time:.2f}s
        Time Saved: {total_time_saved:.2f}s
        
        Performance Gain: {(total_time_saved/total_serial_time)*100:.1f}%
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to {filename}")
        
        plt.show()
    
    def plot_scalability_analysis(self, df, save_plot=True):
        """
        Create scalability analysis plots
        """
        if df.empty:
            print("No scalability data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Speedup vs Number of Processes
        ax1 = axes[0, 0]
        for size in df['matrix_size'].unique():
            size_data = df[df['matrix_size'] == size]
            ax1.plot(size_data['num_processes'], size_data['speedup'], 'o-', 
                    label=f'Matrix {size}', linewidth=2, markersize=6)
        
        # Ideal speedup line
        max_processes = df['num_processes'].max()
        ax1.plot(range(1, max_processes + 1), range(1, max_processes + 1), 
                'k--', alpha=0.5, label='Ideal Speedup')
        
        ax1.set_xlabel('Number of Processes')
        ax1.set_ylabel('Speedup')
        ax1.set_title('Speedup vs Number of Processes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Efficiency vs Number of Processes
        ax2 = axes[0, 1]
        for size in df['matrix_size'].unique():
            size_data = df[df['matrix_size'] == size]
            ax2.plot(size_data['num_processes'], size_data['efficiency'], 's-', 
                    label=f'Matrix {size}', linewidth=2, markersize=6)
        
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency')
        ax2.set_xlabel('Number of Processes')
        ax2.set_ylabel('Efficiency')
        ax2.set_title('Efficiency vs Number of Processes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Execution Time vs Number of Processes
        ax3 = axes[1, 0]
        for size in df['matrix_size'].unique():
            size_data = df[df['matrix_size'] == size]
            ax3.plot(size_data['num_processes'], size_data['distributed_time'], '^-', 
                    label=f'Matrix {size}', linewidth=2, markersize=6)
        
        ax3.set_xlabel('Number of Processes')
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_title('Execution Time vs Number of Processes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Heatmap of Efficiency
        ax4 = axes[1, 1]
        pivot_table = df.pivot(index='matrix_size', columns='num_processes', values='efficiency')
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4)
        ax4.set_title('Efficiency Heatmap')
        ax4.set_xlabel('Number of Processes')
        ax4.set_ylabel('Matrix Size')
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scalability_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Scalability plots saved to {filename}")
        
        plt.show()
    
    def generate_report(self, results, num_processes, output_file=None):
        """
        Generate a comprehensive performance report
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"performance_report_{num_processes}proc_{timestamp}.txt"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MPI DISTRIBUTED MATRIX MULTIPLICATION PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of Processes: {num_processes}")
        report_lines.append("")
        
        # Summary statistics
        avg_speedup = np.mean(results['speedup'])
        max_speedup = np.max(results['speedup'])
        min_speedup = np.min(results['speedup'])
        avg_efficiency = np.mean(results['efficiency'])
        max_efficiency = np.max(results['efficiency'])
        min_efficiency = np.min(results['efficiency'])
        
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Average Speedup:    {avg_speedup:.3f}x")
        report_lines.append(f"Maximum Speedup:    {max_speedup:.3f}x")
        report_lines.append(f"Minimum Speedup:    {min_speedup:.3f}x")
        report_lines.append(f"Average Efficiency: {avg_efficiency:.3f}")
        report_lines.append(f"Maximum Efficiency: {max_efficiency:.3f}")
        report_lines.append(f"Minimum Efficiency: {min_efficiency:.3f}")
        report_lines.append("")
        
        # Detailed results table
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 40)
        header = f"{'Matrix Size':<12} {'Serial(s)':<10} {'Distributed(s)':<15} {'Speedup':<8} {'Efficiency':<10}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        for i in range(len(results['matrix_sizes'])):
            line = (f"{results['matrix_sizes'][i]:<12} "
                   f"{results['serial_times'][i]:<10.4f} "
                   f"{results['distributed_times'][i]:<15.4f} "
                   f"{results['speedup'][i]:<8.2f} "
                   f"{results['efficiency'][i]:<10.3f}")
            report_lines.append(line)
        
        report_lines.append("")
        
        # Performance analysis
        total_serial_time = sum(results['serial_times'])
        total_distributed_time = sum(results['distributed_times'])
        total_time_saved = total_serial_time - total_distributed_time
        
        report_lines.append("PERFORMANCE ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Serial Time:      {total_serial_time:.4f} seconds")
        report_lines.append(f"Total Distributed Time: {total_distributed_time:.4f} seconds")
        report_lines.append(f"Total Time Saved:       {total_time_saved:.4f} seconds")
        report_lines.append(f"Overall Performance Gain: {(total_time_saved/total_serial_time)*100:.2f}%")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        if avg_efficiency > 0.8:
            report_lines.append("✓ Excellent parallel efficiency achieved!")
        elif avg_efficiency > 0.6:
            report_lines.append("✓ Good parallel efficiency. Consider optimizing communication overhead.")
        else:
            report_lines.append("! Low parallel efficiency. Review load balancing and communication patterns.")
        
        if max_speedup / num_processes > 0.8:
            report_lines.append("✓ Good scalability with current process count.")
        else:
            report_lines.append("! Consider reducing process count or increasing problem size.")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Write report to file
        try:
            with open(output_file, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"Performance report saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving report: {e}")
            return None

def main():
    """
    Main function for performance analysis
    """
    analyzer = PerformanceAnalyzer()
    
    # Example usage - you would typically load actual results
    print("Performance Analyzer for MPI Matrix Multiplication")
    print("This tool can analyze results from benchmark runs.")
    print("\nUsage examples:")
    print("1. analyzer.load_results('mpi_results_4proc_20241220_143022.json')")
    print("2. analyzer.plot_performance_metrics(results)")
    print("3. analyzer.generate_report(results, num_processes)")

if __name__ == "__main__":
    main()