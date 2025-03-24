#!/usr/bin/env python3

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from math import floor, ceil, sqrt
import os
import re

def median_confidence_interval(data, confidence):
    # https://web.archive.org/web/20160509121216id_/http://spcl.inf.ethz.ch:80/Teaching/2015-dphpc/hoefler-scientific-benchmarking.pdf
    n = len(data)
    a = 1 - confidence
    z = stats.norm.pdf(a)

    lower_rank = floor((n - z * sqrt(n)) / 2)
    upper_rank = ceil(1 + ((n + z * sqrt(n)) / 2))

    data_sorted = sorted(data)
    return (data_sorted[lower_rank], data_sorted[upper_rank])


def compute_stats(data):
    mean = np.mean(data)
    median = np.median(data)
    stdev = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    median_ci = median_confidence_interval(data, 0.95)
    return mean, median, stdev, min_val, max_val, median_ci


def t_test(other, eqsat):
    stat = stats.ttest_ind(other, eqsat)
    return (stat.statistic, stat.pvalue)


def plot_histograms(df):
    stages = df['stage'].unique()

    for stage in stages:
        plt.figure(figsize=(10, 6))
        stage_data = df[df['stage'] == stage]

        sns.kdeplot(data=stage_data, x='runtime_ms', hue='pipeline', common_norm=False)

        plt.title(stage)
        plt.xlabel('runtime (ms)')
        plt.ylabel('density')

        pipelines = stage_data['pipeline'].unique()
        for pipeline in pipelines:
            pipeline_data = stage_data[stage_data['pipeline'] == pipeline]['runtime_ms']
            mean, median, stdev, min_val, max_val, (median_ci_lo, median_ci_hi) = compute_stats(pipeline_data)
            print(f"{stage}, {pipeline}:")
            print(f"  mean   : {mean:.2f} ms")
            print(f"  median : {median:.3f}")
            print(f"     95% : [{median_ci_lo:.3f}, {median_ci_hi:.3f}]")
            print(f"  stdev  : {stdev:.2f} ms")
            print(f"  min    : {min_val:.2f} ms")
            print(f"  max    : {max_val:.2f} ms")
            print()

        print(f"t-test ({stage}):")
        eqsat_data = stage_data[stage_data['pipeline'] == "EqSat"]['runtime_ms']
        for pipeline in pipelines:
            if pipeline == "EqSat":
                continue
            pipeline_data = stage_data[stage_data['pipeline'] == pipeline]['runtime_ms']
            (t, p) = t_test(pipeline_data, eqsat_data)
            print(f'{pipeline} vs EqSat: t-statistic {t:.4f}, p-value {p:.4f}')
        print()

        plt.show()


def plot_time(df):
    stages = df['stage'].unique()

    for stage in stages:
        stage_data = df[df['stage'] == stage]
        plt.figure(figsize=(10, 6))
        pipelines = stage_data['pipeline'].unique()

        for pipeline in pipelines:
            pipeline_data = stage_data[stage_data['pipeline'] == pipeline]
            plt.plot(pipeline_data.index, pipeline_data['runtime_ms'], label=pipeline)

        plt.title(f'Runtime over Data Index for Stage: {stage}')
        plt.xlabel('Data Index')
        plt.ylabel('Runtime (ms)')
        plt.legend(title='Pipeline')

        plt.show()


def compute_relative_speedup(model_data):
    """Compute relative speedups of DefOpt (Enzyme-JAX) and EqSat (Constable) against JAX"""
    # Get the first stage available in this CSV (assuming one stage per CSV)
    stages = model_data['stage'].unique()
    if len(stages) == 0:
        print("Error: No stages found in data")
        return 0, 0
        
    stage = stages[0]
    print(f"Using stage: {stage}")
    
    # Filter data for the stage
    stage_data = model_data[model_data['stage'] == stage]
    
    # Calculate median runtimes for each pipeline
    jax_median = np.median(stage_data[stage_data['pipeline'].str.strip() == 'JaX']['runtime_ms'])
    defopt_median = np.median(stage_data[stage_data['pipeline'] == 'DefOpt']['runtime_ms'])
    eqsat_median = np.median(stage_data[stage_data['pipeline'] == 'EqSat']['runtime_ms'])
    
    # Calculate relative speedups (negative values mean slowdowns)
    defopt_speedup = (jax_median - defopt_median) / jax_median
    eqsat_speedup = (jax_median - eqsat_median) / jax_median
    
    return defopt_speedup, eqsat_speedup


def process_plot_data(model_file_pairs):
    """Process data for a single plot"""
    models = []
    enzyme_speedups = []
    constable_speedups = []
    
    for i in range(0, len(model_file_pairs), 2):
        model_name = model_file_pairs[i]
        csv_file = model_file_pairs[i+1]
        
        try:
            df = pd.read_csv(csv_file)
            if not {'pipeline', 'stage', 'runtime_ms'}.issubset(df.columns):
                print(f"Error: File {csv_file} must contain 'pipeline', 'stage', and 'runtime_ms' columns.")
                continue
                
            print(f"Processing {model_name} from {csv_file}...")
            defopt_speedup, eqsat_speedup = compute_relative_speedup(df)
            models.append(model_name)
            enzyme_speedups.append(defopt_speedup)
            constable_speedups.append(eqsat_speedup)
            
            print(f"{model_name}:")
            print(f"  Enzyme-JAX speedup: {defopt_speedup*100:.2f}%")
            print(f"  Constable speedup: {eqsat_speedup*100:.2f}%")
            print()
            
        except FileNotFoundError:
            print(f"Error: File '{csv_file}' not found.")
            continue
            
    return models, enzyme_speedups, constable_speedups


def plot_comparison(plot_groups):
    """Plot multiple bar charts of speedups for different models and platforms"""
    # Determine the number of plots
    num_plots = len(plot_groups)
    
    if num_plots == 0:
        print("Error: No valid plot data provided")
        return
    
    # Find min/max y values across all plots for consistent y-axis scaling
    all_enzyme_speedups = []
    all_constable_speedups = []
    
    for title, file_model_pairs in plot_groups:
        models, enzyme_speedups, constable_speedups = process_plot_data(file_model_pairs)
        all_enzyme_speedups.extend(enzyme_speedups)
        all_constable_speedups.extend(constable_speedups)
    
    if not all_enzyme_speedups and not all_constable_speedups:
        print("Error: No valid data to plot")
        return
    
    all_speedups = all_enzyme_speedups + all_constable_speedups
    min_val = min(min(all_speedups) * 100 if all_speedups else -10, -10)
    max_val = max(max(all_speedups) * 100 if all_speedups else 60, 60)
    
    # Round to nearest 5%
    min_val = np.floor(min_val / 5) * 5 - 5
    max_val = np.ceil(max_val / 5) * 5 + 5
    
    # Create a figure with multiple subplots side by side
    # Make plots less tall (height reduced from 7 to 5)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True)
    
    # Handle the case of a single plot
    if num_plots == 1:
        axes = [axes]
    
    for idx, (title, file_model_pairs) in enumerate(plot_groups):
        ax = axes[idx]
        models, enzyme_speedups, constable_speedups = process_plot_data(file_model_pairs)
        
        # Skip empty datasets
        if not models:
            ax.text(0.5, 0.5, f"No data for {title}", 
                    horizontalalignment='center', verticalalignment='center')
            continue
        
        width = 0.3  # Bar width
        x = np.arange(len(models))
        
        # Plot bars for Enzyme-JAX and Constable right next to each other
        enzyme_bars = ax.bar(x - width/2, [s*100 for s in enzyme_speedups], width, 
                             label='Enzyme-JAX', color='lightblue', edgecolor='black', linewidth=1)
        constable_bars = ax.bar(x + width/2, [s*100 for s in constable_speedups], width, 
                                label='Constable', color='green', edgecolor='black', linewidth=1)
        
        # Add grey grid lines
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
        
        # Bring the bars to the front to cover grid lines
        for bar in enzyme_bars:
            bar.set_zorder(10)
        for bar in constable_bars:
            bar.set_zorder(10)
        
        # Add JAX baseline
        jax_line = ax.axhline(y=0, color='purple', linestyle='--', linewidth=2, 
                              label='JAX' if idx == 0 else "_nolegend_")
        jax_line.set_zorder(5)
        
        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', rotation_mode='anchor')
        
        # Set plot title
        ax.set_title(title)
        
        # Only set y-label on the leftmost plot
        if idx == 0:
            ax.set_ylabel('Relative speedup, %')
    
    # Set common y-axis limits and ticks
    y_ticks = np.arange(np.ceil(min_val / 20) * 20, np.floor(max_val / 20) * 20 + 1, 20)
    plt.setp(axes, ylim=(min_val, max_val), yticks=y_ticks)
    
    # Adjust layout first to make space for rotated x-labels
    plt.tight_layout()
    
    # Add more space at the bottom for x-labels and legend
    plt.subplots_adjust(bottom=0.25)
    
    # Add a single legend at the bottom of the figure (further down)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=3, frameon=False, handletextpad=1.0, columnspacing=2.0)
    
    # Add a button to save the figure as PDF (positioned at the bottom, slightly lower than before)
    save_ax = plt.axes([0.45, 0.005, 0.1, 0.04])  # Position for the button [left, bottom, width, height]
    save_button = plt.Button(save_ax, 'Save PDF', color='lightgoldenrodyellow', hovercolor='0.975')
    
    def save_to_pdf(event):
        # Generate filename based on current date/time
        from datetime import datetime
        import os
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"speedup_comparison_{timestamp}.pdf"
        
        # Temporarily hide the save button for the PDF export
        save_button.ax.set_visible(False)
        
        # Save figure to current directory (button won't be visible in the saved file)
        fig.savefig(filename, format='pdf', bbox_inches='tight')
        abs_path = os.path.abspath(filename)
        print(f"Saved figure as {abs_path}")
        
        # Make the button visible again for continued interaction
        save_button.ax.set_visible(True)
        fig.canvas.draw_idle()
    
    save_button.on_clicked(save_to_pdf)
    plt.show()


def find_related_cost_model_files(baseline_csv):
    """Find no-fusion and no-zero files related to the baseline csv file"""
    directory = os.path.dirname(baseline_csv)
    if directory == '':
        directory = '.'
    
    basename = os.path.basename(baseline_csv)
    
    # Extract model name and platform from the baseline filename
    # Assuming format: results_MODEL_cost-model_baseline-PLATFORM_DATE.csv
    pattern = r'results_(.+)_cost-model_baseline-(.+)_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}\.csv'
    match = re.match(pattern, basename)
    
    if not match:
        print(f"Error: Cannot parse baseline filename pattern: {basename}")
        return None, None, None
    
    model = match.group(1)
    platform = match.group(2)
    
    # Extract timestamp from filename to find matching files
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}):\d{2}'
    timestamp_match = re.search(timestamp_pattern, basename)
    if not timestamp_match:
        print(f"Error: Cannot extract timestamp from filename: {basename}")
        return None, None, None
    
    timestamp_prefix = timestamp_match.group(1)
    
    # Find corresponding no-fusion and no-zero files
    no_fusion_pattern = f"results_{model}_cost-model_no-fusion-{platform}_{timestamp_prefix}"
    no_zero_pattern = f"results_{model}_cost-model_no-zero-{platform}_{timestamp_prefix}"
    
    no_fusion_file = None
    no_zero_file = None
    
    for file in os.listdir(directory):
        if file.startswith(no_fusion_pattern):
            no_fusion_file = os.path.join(directory, file)
        elif file.startswith(no_zero_pattern):
            no_zero_file = os.path.join(directory, file)
    
    return no_fusion_file, no_zero_file

def compute_cost_model_speedups(baseline_file, no_fusion_file, no_zero_file):
    """Compute relative speedups between different cost model configurations"""
    baseline_df = pd.read_csv(baseline_file)
    no_fusion_df = pd.read_csv(no_fusion_file) if no_fusion_file else None
    no_zero_df = pd.read_csv(no_zero_file) if no_zero_file else None
    
    # Get the first stage available in the CSV
    stages = baseline_df['stage'].unique()
    if len(stages) == 0:
        print("Error: No stages found in baseline data")
        return None, None, None
    
    stage = stages[0]
    print(f"Using stage: {stage}")
    
    # Calculate median runtimes for EqSat pipeline only
    baseline_eqsat_data = baseline_df[(baseline_df['stage'] == stage) & (baseline_df['pipeline'].str.strip() == 'EqSat')]
    baseline_median = np.median(baseline_eqsat_data['runtime_ms'])
    
    no_fusion_median = None
    if no_fusion_df is not None:
        no_fusion_eqsat_data = no_fusion_df[(no_fusion_df['stage'] == stage) & (no_fusion_df['pipeline'].str.strip() == 'EqSat')]
        no_fusion_median = np.median(no_fusion_eqsat_data['runtime_ms'])
    
    no_zero_median = None
    if no_zero_df is not None:
        no_zero_eqsat_data = no_zero_df[(no_zero_df['stage'] == stage) & (no_zero_df['pipeline'].str.strip() == 'EqSat')]
        no_zero_median = np.median(no_zero_eqsat_data['runtime_ms'])
    
    # Get JAX data from the baseline file
    # JAX runtime is in the same CSV file with pipeline="JaX  " (note the spaces)
    jax_data = baseline_df[baseline_df['pipeline'].str.strip() == 'JaX']
    
    if len(jax_data) == 0:
        print("Warning: No JAX data found in the baseline file. Using baseline as reference.")
        # If no JAX data, use the baseline as reference (0%)
        baseline_speedup = 0
        
        if no_fusion_median:
            no_fusion_speedup = (baseline_median - no_fusion_median) / baseline_median
        else:
            no_fusion_speedup = None
            
        if no_zero_median:
            no_zero_speedup = (baseline_median - no_zero_median) / baseline_median
        else:
            no_zero_speedup = None
    else:
        # Calculate JAX median runtime
        jax_median = np.median(jax_data[jax_data['stage'] == stage]['runtime_ms'])
        print(f"  JAX median runtime: {jax_median:.4f} ms")
        print(f"  Base cost model runtime: {baseline_median:.4f} ms")
        
        # Calculate speedups relative to JAX
        baseline_speedup = (jax_median - baseline_median) / jax_median
        
        no_fusion_speedup = None
        if no_fusion_median:
            no_fusion_speedup = (jax_median - no_fusion_median) / jax_median
            print(f"  No fusion runtime: {no_fusion_median:.4f} ms")
        
        no_zero_speedup = None
        if no_zero_median:
            no_zero_speedup = (jax_median - no_zero_median) / jax_median
            print(f"  No zero runtime: {no_zero_median:.4f} ms")
    
    return baseline_speedup, no_fusion_speedup, no_zero_speedup

def process_cost_model_data(model_file_pairs):
    """Process data for cost model plot"""
    models = []
    baseline_speedups = []
    no_fusion_speedups = []
    no_zero_speedups = []
    
    for i in range(0, len(model_file_pairs), 2):
        model_name = model_file_pairs[i]
        baseline_csv = model_file_pairs[i+1]
        
        try:
            # Find related files
            no_fusion_file, no_zero_file = find_related_cost_model_files(baseline_csv)
            
            if no_fusion_file is None or no_zero_file is None:
                print(f"Warning: Could not find matching no-fusion or no-zero files for {baseline_csv}")
                continue
                
            print(f"Processing {model_name}:")
            print(f"  Baseline: {baseline_csv}")
            print(f"  No fusion: {no_fusion_file}")
            print(f"  No zero: {no_zero_file}")
            
            # Compute speedups relative to JAX
            baseline_speedup, no_fusion_speedup, no_zero_speedup = compute_cost_model_speedups(
                baseline_csv, no_fusion_file, no_zero_file
            )
            
            if baseline_speedup is None:
                continue
                
            models.append(model_name)
            baseline_speedups.append(baseline_speedup)
            
            if no_fusion_speedup is not None:
                no_fusion_speedups.append(no_fusion_speedup)
                print(f"  No fusion speedup: {no_fusion_speedup*100:.2f}%")
            else:
                no_fusion_speedups.append(0)
                
            if no_zero_speedup is not None:
                no_zero_speedups.append(no_zero_speedup)
                print(f"  No zero speedup: {no_zero_speedup*100:.2f}%")
            else:
                no_zero_speedups.append(0)
            
            print(f"  Base cost model speedup: {baseline_speedup*100:.2f}%")    
            print()
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue
            
    return models, baseline_speedups, no_fusion_speedups, no_zero_speedups

def plot_cost_model(plot_groups):
    """Plot bar charts comparing different cost model configurations"""
    # Determine the number of plots
    num_plots = len(plot_groups)
    
    if num_plots == 0:
        print("Error: No valid plot data provided")
        return
    
    # Process all data first and store results for plotting
    all_plot_data = []
    all_speedups = []
    
    for title, file_model_pairs in plot_groups:
        models, baseline_speedups, no_fusion_speedups, no_zero_speedups = process_cost_model_data(file_model_pairs)
        if models:
            all_plot_data.append((title, models, baseline_speedups, no_fusion_speedups, no_zero_speedups))
            all_speedups.extend(baseline_speedups + no_fusion_speedups + no_zero_speedups)
    
    if not all_plot_data:
        print("Error: No valid data to plot")
        return
    
    # Calculate min/max for y-axis scaling across all plots
    min_val = min(min(all_speedups) * 100 if all_speedups else -15, -15)
    max_val = max(max(all_speedups) * 100 if all_speedups else 45, 45)
    
    # Round to nearest 5%
    min_val = np.floor(min_val / 5) * 5 - 5
    max_val = np.ceil(max_val / 5) * 5 + 5
    
    # Create a figure with multiple subplots side by side - wider but not too tall
    fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 5), sharey=True)
    
    # Handle the case of a single plot
    if num_plots == 1:
        axes = [axes]
    
    # Plot each dataset using the pre-processed data
    for idx, (title, models, baseline_speedups, no_fusion_speedups, no_zero_speedups) in enumerate(all_plot_data):
        ax = axes[idx]
        
        width = 0.25  # Bar width
        x = np.arange(len(models))
        
        # Plot bars for the three cost model configurations
        baseline_bars = ax.bar(x - width, [s*100 for s in baseline_speedups], width, 
                           label='Constable w/ base cost model', color='green', edgecolor='black', linewidth=1)
        no_fusion_bars = ax.bar(x, [s*100 for s in no_fusion_speedups], width, 
                            label='Constable w/o fusion costs', color='darkturquoise', edgecolor='black', linewidth=1)
        no_zero_bars = ax.bar(x + width, [s*100 for s in no_zero_speedups], width, 
                          label='Constable w/o zero costs', color='orange', edgecolor='black', linewidth=1)
        
        # Add grey grid lines
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
        
        # Add JAX baseline reference line at 0%
        jax_line = ax.axhline(y=0, color='purple', linestyle='--', linewidth=2, 
                              label='JAX' if idx == 0 else "_nolegend_")
        jax_line.set_zorder(5)
        
        # Bring the bars to the front to cover grid lines
        for bar in baseline_bars + no_fusion_bars + no_zero_bars:
            bar.set_zorder(10)
        
        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', rotation_mode='anchor')
        
        # Set plot title
        ax.set_title(title)
        
        # Only set y-label on the leftmost plot
        if idx == 0:
            ax.set_ylabel('Relative speedup, %')
    
    # Set common y-axis limits and ticks
    y_ticks = np.arange(np.ceil(min_val / 20) * 20, np.floor(max_val / 20) * 20 + 1, 20)
    plt.setp(axes, ylim=(min_val, max_val), yticks=y_ticks)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add more space at the bottom for x-labels and legend
    plt.subplots_adjust(bottom=0.25, wspace=0.3)
    
    # Add a single legend at the bottom of the figure - all labels in one row (horizontally)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=len(handles), frameon=False, handletextpad=1.0, columnspacing=2.0)
    
    # Add a button to save the figure as PDF
    save_ax = plt.axes([0.45, 0.005, 0.1, 0.04])
    save_button = plt.Button(save_ax, 'Save PDF', color='lightgoldenrodyellow', hovercolor='0.975')
    
    def save_to_pdf(event):
        # Generate filename based on current date/time
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"cost_model_comparison_{timestamp}.pdf"
        
        # Temporarily hide the save button for the PDF export
        save_button.ax.set_visible(False)
        
        # Save figure to current directory
        fig.savefig(filename, format='pdf', bbox_inches='tight')
        abs_path = os.path.abspath(filename)
        print(f"Saved figure as {abs_path}")
        
        # Make the button visible again for continued interaction
        save_button.ax.set_visible(True)
        fig.canvas.draw_idle()
    
    save_button.on_clicked(save_to_pdf)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("usage: ./graph_results.py <hist|time|compare|cost_model> [options]")
        print("  hist: plot histograms of runtimes")
        print("    ./graph_results.py hist <filename.csv>")
        print()
        print("  time: plot time series of runtimes")
        print("    ./graph_results.py time <filename.csv>")
        print()
        print("  compare: plot bar charts comparing speedups of multiple models across different platforms")
        print("    ./graph_results.py compare -p <plot_title> <model_name> <csv_file> [<model_name> <csv_file> ...] [-p <plot_title> ...]")
        print("    Example:")
        print("      ./graph_results.py compare -p A100 Llama llama_a100.csv NasRNN nasrnn_a100.csv -p V100 Llama llama_v100.csv NasRNN nasrnn_v100.csv")
        print()
        print("  cost_model: plot bar charts comparing cost model configurations")
        print("    ./graph_results.py cost_model -p <plot_title> <model_name> <baseline_csv> [<model_name> <baseline_csv> ...] [-p <plot_title> ...]")
        print("    Example:")
        print("      ./graph_results.py cost_model -p Xeon Llama results_llama_cost-model_baseline-cpu_2025-03-19_19:58:50.csv NasRNN results_nasrnn_cost-model_baseline-cpu_2025-03-19_19:58:50.csv")
        sys.exit(1)
        
    mode = sys.argv[1]
    
    if mode in ["hist", "time"]:
        if len(sys.argv) != 3:
            print(f"Error: {mode} mode requires exactly one CSV file.")
            sys.exit(1)
            
        filename = sys.argv[2]
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)

        if not {'pipeline', 'stage', 'runtime_ms'}.issubset(df.columns):
            print("Error: The input CSV must contain 'pipeline', 'stage', and 'runtime_ms' columns.")
            sys.exit(1)

        if mode == "hist":
            plot_histograms(df)
        else:
            plot_time(df)
            
    elif mode in ["compare", "cost_model"]:
        if len(sys.argv) < 4 or sys.argv[2] != "-p":
            print(f"Error: {mode} mode requires -p flag followed by plot groups.")
            print(f"Example: ./graph_results.py {mode} -p A100 Llama llama_a100.csv -p V100 Llama llama_v100.csv")
            sys.exit(1)
            
        # Multi-plot mode with -p flags
        plot_groups = []
        current_plot = None
        current_title = None
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "-p":
                # If we have a current plot group, add it before starting a new one
                if current_title is not None and current_plot is not None:
                    plot_groups.append((current_title, current_plot))
                
                # Start a new plot group
                if i + 1 < len(sys.argv):
                    current_title = sys.argv[i + 1]
                    current_plot = []
                    i += 2
                else:
                    print("Error: -p flag must be followed by a plot title")
                    sys.exit(1)
            else:
                # Add file/model pairs to the current plot
                if current_plot is None:
                    print("Error: File/model pairs must follow a -p flag")
                    sys.exit(1)
                
                if i + 1 < len(sys.argv) and sys.argv[i+1] != "-p":
                    current_plot.append(sys.argv[i])     # Model name
                    current_plot.append(sys.argv[i+1])   # CSV file
                    i += 2
                else:
                    # Last item without a pair
                    print(f"Error: Each model name must be followed by a CSV file. Missing CSV file for {sys.argv[i]}")
                    sys.exit(1)
        
        # Add the last plot group if it exists
        if current_title is not None and current_plot is not None:
            plot_groups.append((current_title, current_plot))
            
        # Plot the groups
        if plot_groups:
            if mode == "compare":
                plot_comparison(plot_groups)
            else:  # cost_model
                plot_cost_model(plot_groups)
        else:
            print("Error: No valid plot groups found")
            sys.exit(1)
        
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'hist', 'time', 'compare', or 'cost_model'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
