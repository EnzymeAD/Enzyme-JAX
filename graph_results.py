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


def percentile_confidence_interval(data, confidence):
    lower_percentile = (100 - confidence) / 2
    upper_percentile = 100 - lower_percentile
    
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    
    return (lower, upper)


def compute_stats(data):
    mean = np.mean(data)
    median = np.median(data)
    stdev = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    median_ci = percentile_confidence_interval(data, 0.95)
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
    jax_median = np.median(stage_data[stage_data['pipeline'].str.strip() == 'JaX']['runtime_ms'])
    defopt_median = np.median(stage_data[stage_data['pipeline'] == 'DefOpt']['runtime_ms'])
    eqsat_median = np.median(stage_data[stage_data['pipeline'] == 'EqSat']['runtime_ms'])

    # Calculate relative speedups (negative values mean slowdowns)
    defopt_speedup = (jax_median - defopt_median) / jax_median
    eqsat_speedup = (jax_median - eqsat_median) / jax_median

    return defopt_speedup, eqsat_speedup


def compute_relative_speedup_runs(run_files):
    """
    Given a list of CSV files (one per run), compute the speedups for each run and
    return the mean and standard error (SEM) for Enzyme-JAX and Constable.
    """
    enzyme_speedups = []
    constable_speedups = []
    for f in run_files:
        try:
            df = pd.read_csv(f)
            d_speedup, c_speedup = compute_relative_speedup(df)
            enzyme_speedups.append(d_speedup)
            constable_speedups.append(c_speedup)
        except Exception as e:
            print(f"Error processing {f}: {e}")
    if len(enzyme_speedups) == 0 or len(constable_speedups) == 0:
        return None, None, None, None
    # Compute mean and standard error
    enzyme_mean = np.mean(enzyme_speedups)
    enzyme_std = np.std(enzyme_speedups)
    enzyme_sem = enzyme_std / np.sqrt(len(enzyme_speedups))
    
    constable_mean = np.mean(constable_speedups)
    constable_std = np.std(constable_speedups)
    constable_sem = constable_std / np.sqrt(len(constable_speedups))
    
    return enzyme_mean, enzyme_sem, constable_mean, constable_sem


def process_plot_data(model_file_pairs):
    """
    For each model, use the provided CSV file (e.g. with _run1.csv) to locate all
    run files (e.g. _run1, _run2, _run3, ...), compute per-run speedups, and then
    aggregate (mean and SEM) across runs.
    """
    models = []
    enzyme_speedups = []  # Each entry: (mean, sem)
    constable_speedups = []  # Each entry: (mean, sem)
    
    for i in range(0, len(model_file_pairs), 2):
        model_name = model_file_pairs[i]
        csv_file = model_file_pairs[i+1]
        
        directory = os.path.dirname(csv_file) or '.'
        basename = os.path.basename(csv_file)
        # Expecting a filename like: results_MODEL_cost-model_baseline-PLATFORM_DATE_run1.csv
        m = re.match(r"(.+)_run\d+\.csv", basename)
        if not m:
            print(f"Error: File {csv_file} does not match expected run pattern.")
            continue
        base_prefix = m.group(1)
        
        # Search the directory for all files with the same prefix and a _run<number>.csv suffix
        run_files = []
        for f in os.listdir(directory):
            if re.match(re.escape(base_prefix) + r"_run\d+\.csv", f):
                run_files.append(os.path.join(directory, f))
        if not run_files:
            print(f"Warning: No run files found for {csv_file}")
            continue
        
        print(run_files)
        
        enzyme_mean, enzyme_sem, constable_mean, constable_sem = compute_relative_speedup_runs(run_files)
        if enzyme_mean is None:
            continue
        
        models.append(model_name)
        enzyme_speedups.append((enzyme_mean, enzyme_sem))
        constable_speedups.append((constable_mean, constable_sem))
        
        print(f"{model_name}:")
        print(f"  Enzyme-JAX speedup: {enzyme_mean*100:.2f}% ± {enzyme_sem*100:.2f}%")
        print(f"  Constable speedup: {constable_mean*100:.2f}% ± {constable_sem*100:.2f}%")
        print()
    
    return models, enzyme_speedups, constable_speedups


def plot_comparison(plot_groups):
    """Plot bar charts comparing speedups for different models and platforms with error bars."""
    num_plots = len(plot_groups)
    
    if num_plots == 0:
        print("Error: No valid plot data provided")
        return

    # Gather all speedups (means) to set consistent y-axis limits.
    all_means = []
    for plot_group in plot_groups:
        title = plot_group[0]
        file_model_pairs = plot_group[1]
        models, enzyme_speedups, constable_speedups = process_plot_data(file_model_pairs)
        # Extract means only
        all_means.extend([s[0] for s in enzyme_speedups])
        all_means.extend([s[0] for s in constable_speedups])
    
    if not all_means:
        print("Error: No valid data to plot")
        return
    
    # Calculate global min and max (in percentages)
    all_speedups_pct = [v*100 for v in all_means]
    min_val = min(all_speedups_pct)
    max_val = max(all_speedups_pct)
    # Extend limits slightly for padding
    min_val = np.floor(min_val / 5) * 5 - 5
    max_val = np.ceil(max_val / 5) * 5 + 5
    
    # Create a figure with one subplot per plot group
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True)
    if num_plots == 1:
        axes = [axes]
    
    for idx, plot_group in enumerate(plot_groups):
        title = plot_group[0]
        file_model_pairs = plot_group[1]
        ax = axes[idx]
        models, enzyme_speedups, constable_speedups = process_plot_data(file_model_pairs)
        if not models:
            ax.text(0.5, 0.5, f"No data for {title}",
                    horizontalalignment='center', verticalalignment='center')
            continue
        
        width = 0.3  # width for each bar
        x = np.arange(len(models))
        
        # Prepare values and error bars (convert speedups to percentages)
        enzyme_vals = [mean*100 for mean, sem in enzyme_speedups]
        enzyme_err = [sem*100 for mean, sem in enzyme_speedups]
        constable_vals = [mean*100 for mean, sem in constable_speedups]
        constable_err = [sem*100 for mean, sem in constable_speedups]
        
        enzyme_bars = ax.bar(x - width/2, enzyme_vals, width, yerr=enzyme_err,
                             capsize=5, label='Enzyme-JAX', color='lightblue',
                             edgecolor='black', linewidth=1)
        constable_bars = ax.bar(x + width/2, constable_vals, width, yerr=constable_err,
                                capsize=5, label='Constable', color='green',
                                edgecolor='black', linewidth=1)
        
        # Add grid and baseline
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
        ax.axhline(y=0, color='purple', linestyle='--', linewidth=2,
                   label='JAX' if idx == 0 else "_nolegend_")
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_title(title)
        if idx == 0:
            ax.set_ylabel('Relative speedup, %')
    
    plt.setp(axes, ylim=(min_val, max_val))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # Place a single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=3, frameon=False, handletextpad=1.0, columnspacing=2.0)
    
    # Optional: add a button to save the figure as PDF
    save_ax = plt.axes([0.45, 0.005, 0.1, 0.04])
    save_button = plt.Button(save_ax, 'Save PDF', color='lightgoldenrodyellow', hovercolor='0.975')
    
    def save_to_pdf(event):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"speedup_comparison_{timestamp}.pdf"
        save_button.ax.set_visible(False)
        fig.savefig(filename, format='pdf', bbox_inches='tight')
        abs_path = os.path.abspath(filename)
        print(f"Saved figure as {abs_path}")
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
    
    for title, file_model_pairs, show_no_fusion in plot_groups:
        models, baseline_speedups, no_fusion_speedups, no_zero_speedups = process_cost_model_data(file_model_pairs)
        if models:
            all_plot_data.append((title, models, baseline_speedups, no_fusion_speedups, no_zero_speedups, show_no_fusion))
            # Only include speedups for bars we'll actually show
            all_speedups.extend(baseline_speedups)
            if show_no_fusion:
                all_speedups.extend(no_fusion_speedups)
            all_speedups.extend(no_zero_speedups)
    
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
    for idx, (title, models, baseline_speedups, no_fusion_speedups, no_zero_speedups, show_no_fusion) in enumerate(all_plot_data):
        ax = axes[idx]
        
        # Adjust bar widths and positions based on how many bars we'll display
        if show_no_fusion:
            # Three bars: Baseline, No Fusion, No Zero
            width = 0.25  # Bar width
            x = np.arange(len(models))
            
            # Plot bars for all three cost model configurations
            baseline_bars = ax.bar(x - width, [s*100 for s in baseline_speedups], width, 
                               label='Constable w/ base cost model', color='green', edgecolor='black', linewidth=1)
            no_fusion_bars = ax.bar(x, [s*100 for s in no_fusion_speedups], width, 
                                label='Constable w/o fusion costs (GPU only)', color='darkturquoise', edgecolor='black', linewidth=1)
            no_zero_bars = ax.bar(x + width, [s*100 for s in no_zero_speedups], width, 
                              label='Constable w/o zero costs', color='orange', edgecolor='black', linewidth=1)
        else:
            # Just two bars: Baseline and No Zero
            width = 0.33  # Slightly wider bars when only showing 2
            x = np.arange(len(models))
            
            # Plot only baseline and no-zero configurations
            baseline_bars = ax.bar(x - width/2, [s*100 for s in baseline_speedups], width, 
                               label='Constable w/ base cost model', color='green', edgecolor='black', linewidth=1)
            no_zero_bars = ax.bar(x + width/2, [s*100 for s in no_zero_speedups], width, 
                              label='Constable w/o zero costs', color='orange', edgecolor='black', linewidth=1)
            no_fusion_bars = []  # Empty list since we're not plotting these
        
        # Add grey grid lines
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
        
        # Add JAX baseline reference line at 0%
        jax_line = ax.axhline(y=0, color='purple', linestyle='--', linewidth=2, 
                              label='JAX' if idx == 0 else "_nolegend_")
        jax_line.set_zorder(5)
        
        # Bring the bars to the front to cover grid lines
        all_bars = baseline_bars + no_zero_bars
        if show_no_fusion:
            all_bars += no_fusion_bars
            
        for bar in all_bars:
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
    # We need to collect all unique handles/labels from all plots
    all_handles = []
    all_labels = []
    
    # Check if any plots show no_fusion bars
    any_show_no_fusion = any(data[5] for data in all_plot_data)
    
    # Get handles from first plot
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Find JAX handle (always included)
    jax_handle = None
    jax_label = None
    baseline_handle = None
    baseline_label = None
    no_zero_handle = None
    no_zero_label = None
    
    for h, l in zip(handles, labels):
        if l == 'JAX':
            jax_handle = h
            jax_label = l
        elif l == 'Constable w/ base cost model':
            baseline_handle = h
            baseline_label = l
        elif l == 'Constable w/o zero costs':
            no_zero_handle = h
            no_zero_label = l
    
    # Add handles in specific order
    if jax_handle:
        all_handles.append(jax_handle)
        all_labels.append(jax_label)
    if baseline_handle:
        all_handles.append(baseline_handle)
        all_labels.append(baseline_label)
    
    # Only add no_fusion to legend if any plot shows it
    if any_show_no_fusion:
        # Find no_fusion handle from a plot that has it
        for idx, data in enumerate(all_plot_data):
            if data[5]:  # This plot shows no_fusion
                h, l = axes[idx].get_legend_handles_labels()
                for handle, label in zip(h, l):
                    if label == 'Constable w/o fusion costs':
                        all_handles.append(handle)
                        all_labels.append(label)
                        break
                break
    
    if no_zero_handle:
        all_handles.append(no_zero_handle)
        all_labels.append(no_zero_label)
    
    # Create the legend with all needed handles
    fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=len(all_handles), frameon=False, handletextpad=1.0, columnspacing=2.0)
    
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

def find_all_tau_files(baseline_file, device_type):
    """Find all tau files for a given model and device type"""
    directory = os.path.dirname(baseline_file)
    if directory == '':
        directory = '.'
    
    basename = os.path.basename(baseline_file)
    
    # Extract model name from the filename (e.g., "resnet", "llama")
    # Two patterns to match:
    # 1. results_MODEL_cost-model_baseline-PLATFORM_DATE.csv
    # 2. results_MODEL_tau=X-PLATFORM_DATE.csv
    model_pattern = r'results_([^_]+)_'
    match = re.search(model_pattern, basename)
    
    if not match:
        print(f"Error: Cannot extract model name from filename: {basename}")
        return []
    
    model = match.group(1)
    platform = device_type.lower()  # cpu or gpu
    
    # Find all tau files for this model and platform
    tau_pattern = re.compile(f"results_{model}_tau=(\d+)-{platform}_\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}:\d{{2}}:\d{{2}}\.csv")
    
    tau_files = []
    for file in os.listdir(directory):
        tau_match = tau_pattern.match(file)
        if tau_match:
            tau = int(tau_match.group(1))
            tau_files.append((tau, os.path.join(directory, file)))
    
    # Sort by tau value
    tau_files.sort(key=lambda x: x[0])
    return tau_files

def extract_optimization_time(stats_files, model, platform, tau=None):
    """Extract optimization time for a specific model from stats files"""
    # Pattern to match: model_tau=X-platform_DATE
    if tau is not None:
        experiment_pattern = f"{model}_tau={tau}-{platform}"
    else:
        experiment_pattern = f"{model}_no_segmentation-{platform}"
    
    matching_entries = []
    
    for stats_file in stats_files:
        try:
            df = pd.read_csv(stats_file)
            if 'experiment_name' not in df.columns or 'eqsat_time' not in df.columns:
                print(f"Warning: File {stats_file} does not have required columns")
                continue
                
            matching_rows = df[df['experiment_name'].str.startswith(experiment_pattern)]
            
            if not matching_rows.empty:
                # Take the last entry as requested
                matching_entries.append(matching_rows.iloc[-1])
                
        except FileNotFoundError:
            print(f"Error: Stats file '{stats_file}' not found.")
            continue
    
    if len(matching_entries) > 1:
        raise ValueError(f"Multiple entries found for experiment {experiment_pattern} across stats files")
    elif len(matching_entries) == 0:
        print(f"Warning: No entries found for experiment {experiment_pattern}")
        return None
    
    return matching_entries[0]['eqsat_time']  # Return optimization time in seconds

def plot_segmentation(plot_groups):
    """Plot a grid of graphs showing speedup and optimization time vs segment size"""
    # Each plot group now contains: (device_name, model_file_pairs, stats_files)
    # Verify that each group has valid stats files
    for device, model_file_pairs, stats_files in plot_groups:
        verified_stats_files = []
        for stats_file in stats_files:
            if os.path.exists(stats_file):
                verified_stats_files.append(stats_file)
            else:
                print(f"Warning: Stats file '{stats_file}' not found for device {device}")
        
        # Update the stats_files in the plot group with the verified ones
        if not verified_stats_files:
            print(f"Error: No valid stats files provided for device {device}")
            return
    
    # Process all plot groups
    device_models = {}  # Dictionary to organize by device -> models
    device_stats_files = {}  # Dictionary to map device to its stats files
    
    for plot_group in plot_groups:
        device = plot_group[0]  # Device name
        file_model_pairs = plot_group[1]  # Model file pairs
        stats_files = plot_group[2]  # Stats files for this device
        
        device_models[device] = []
        device_stats_files[device] = stats_files
        
        i = 0
        while i < len(file_model_pairs):
            model_name = file_model_pairs[i]
            # Need at least two files (baseline and one tau file)
            if i + 2 > len(file_model_pairs):
                print(f"Warning: Insufficient files for model {model_name}")
                i += 1
                continue
                
            baseline_file = file_model_pairs[i+1]
            tau_file_example = file_model_pairs[i+2]  # Example tau file to determine platform (cpu/gpu)
            
            # Extract device type (cpu/gpu) from the filename
            platform_pattern = r'-(cpu|gpu)'
            platform_match = re.search(platform_pattern, os.path.basename(tau_file_example))
            
            if not platform_match:
                print(f"Error: Cannot determine platform (cpu/gpu) from filename: {tau_file_example}")
                i += 3  # Skip to next model
                continue
                
            platform = platform_match.group(1)
            
            # Find all relevant tau files
            tau_files = find_all_tau_files(tau_file_example, platform)
            if not tau_files:
                print(f"Warning: No tau files found for {model_name} on {platform}")
                i += 3  # Skip to next model
                continue
            
            # Store data for this model
            device_models[device].append((model_name, baseline_file, tau_files, platform))
            
            # Skip to next model (baseline + tau_example)
            # The number of files per model is variable, so we need to count how many we've used
            i += 3
    
    if not device_models:
        print("Error: No valid data to plot")
        return
    
    # Determine the layout of the grid (one row per device)
    num_devices = len(device_models)
    
    # Create figure with subplots - one row per device
    device_names = list(device_models.keys())
    
    # For each device, determine how many model plots we need
    max_models_per_device = max(len(models) for models in device_models.values())
    
    if max_models_per_device == 0:
        print("Error: No valid models to plot")
        return
    
    # Create figure with subplots - one row per device, one column per model
    # Make plots more square with more vertical space
    # Also add extra height for legends, buttons, and to prevent title overlap
    # Using a moderate height multiplier (1.5) for a balanced compact opening window
    fig, axes = plt.subplots(num_devices, max_models_per_device, 
                             figsize=(3.5 * max_models_per_device, 1.5 * num_devices + 1.0),
                             squeeze=False)
                             
    # Apply initial layout to get proper axes positioning
    plt.tight_layout()
    # Adjust spacing for our needs - increased hspace to 1.2 for more vertical spacing between rows
    plt.subplots_adjust(left=0.04, right=0.96, wspace=0.3, hspace=1.2, bottom=0.25, top=0.92)
    
    # For each device row
    for device_idx, device in enumerate(device_names):
        models_data = device_models[device]
        
        # For each model in this device
        for model_idx, (model_name, baseline_file, tau_files, platform) in enumerate(models_data):
            ax = axes[device_idx, model_idx]
            
            try:
                # Read baseline file to get JAX performance
                baseline_df = pd.read_csv(baseline_file)
                jax_data = baseline_df[baseline_df['pipeline'].str.strip() == 'JaX']
                
                if jax_data.empty:
                    print(f"Warning: No JAX data in {baseline_file}")
                    continue
                    
                # Get median JAX runtime
                stages = jax_data['stage'].unique()
                if len(stages) == 0:
                    print(f"Warning: No stages in JAX data for {baseline_file}")
                    continue
                    
                stage = stages[0]  # Use first stage
                jax_median = np.median(jax_data[jax_data['stage'] == stage]['runtime_ms'])
                
                # Process each tau file
                taus = []
                speedups = []
                opt_times = []
                segments = []
                
                for tau, tau_file in tau_files:
                    taus.append(tau)
                    
                    # Extract model name to use in stats file lookup
                    model_in_file = re.search(r'results_([^_]+)_', os.path.basename(tau_file)).group(1)
                    
                    # Get the stats files for this device
                    device_specific_stats = device_stats_files[device]
                    
                    # Get optimization time from the device-specific stats files
                    opt_time = extract_optimization_time(device_specific_stats, model_in_file, platform, tau)
                    if opt_time is not None:
                        opt_times.append(float(opt_time))
                    else:
                        print(f"Warning: No optimization time found for {model_in_file} tau={tau} on {platform}")
                        opt_times.append(np.nan)
                    
                    # Get number of segments if available
                    segment_count = None
                    for stats_file in device_specific_stats:
                        try:
                            df = pd.read_csv(stats_file)
                            if 'experiment_name' in df.columns and 'segments' in df.columns:
                                matching_rows = df[df['experiment_name'] == f"{model_in_file}_tau={tau}-{platform}"]
                                if not matching_rows.empty:
                                    segment_count = matching_rows.iloc[-1]['segments']
                                    break
                        except:
                            continue
                    
                    segments.append(segment_count if segment_count is not None else np.nan)
                    
                    # Get EqSat performance
                    eqsat_df = pd.read_csv(tau_file)
                    eqsat_data = eqsat_df[(eqsat_df['stage'] == stage) & (eqsat_df['pipeline'] == 'EqSat')]
                    
                    if eqsat_data.empty:
                        print(f"Warning: No EqSat data in {tau_file}")
                        speedups.append(np.nan)
                        continue
                        
                    eqsat_median = np.median(eqsat_data['runtime_ms'])
                    # Calculate relative speedup (EqSat vs JAX)
                    speedup = (jax_median - eqsat_median) / jax_median
                    speedups.append(speedup)
                
                # Plot the data
                color1 = 'green'
                color2 = 'blue'
                
                # Primary y-axis for speedup
                ax1 = ax
                ax1.set_xscale('log')  # Log scale for x-axis
                
                # Get min/max times to set appropriate y-axis scale
                min_opt_time = min(opt_times) if opt_times else 0
                max_opt_time = max(opt_times) if opt_times else 0
                
                # Plot the speedup data with smaller dots
                line1 = ax1.plot(taus, [s*100 for s in speedups], marker='.', markersize=8, color=color1, label='Constable')
                
                # Only add y-axis label for leftmost plot in each row
                if model_idx == 0:
                    ax1.set_ylabel('Relative speedup, %', color=color1, fontsize=8)
                ax1.tick_params(axis='y', labelcolor=color1, labelsize=7)
                ax1.tick_params(axis='x', labelsize=7)
                
                # Adjust y-axis limits based on data to prevent cutoff
                min_speedup = min([s*100 for s in speedups]) if speedups else 0
                max_speedup = max([s*100 for s in speedups]) if speedups else 0
                
                # Calculate padding as a percentage of data range, with minimum padding
                data_range = max(max_speedup - min_speedup, 20)  # Ensure minimum perceived range
                padding = max(5, data_range * 0.15)  # At least 5% padding or 15% of range
                
                # Set y-axis limits with adequate padding
                y_min = min(-10, min_speedup - padding)  # Lower limit with padding
                
                # Only go higher than 30% if data requires it
                if max_speedup > 25:
                    y_max = max_speedup + padding  # Upper limit with padding
                else:
                    y_max = 30  # Default upper limit of 30%
                    
                ax1.set_ylim(y_min, y_max)
                
                # Use wider-spaced ticks for the speedup axis
                from matplotlib.ticker import MultipleLocator
                # Use 15% increments instead of 10%
                ax1.yaxis.set_major_locator(MultipleLocator(15))
                
                # Add JAX reference line
                jax_line = ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='JAX')
                
                # Secondary y-axis for optimization time
                ax2 = ax1.twinx()
                # Manually insert values near zero to force axis to show more of the bottom
                # This is a workaround to ensure space below the data points
                dummy_taus = [min(taus)]
                dummy_values = [0.001]  # Tiny value near zero to force axis extent
                
                # Plot both the real data and the dummy point (but only show the real data in legend)
                # Use 'x' marker (cross) for optimization time
                line2 = ax2.plot(taus, opt_times, marker='x', markersize=6, markeredgewidth=1.5, color=color2, label='Optimization time')
                dummy_line = ax2.plot(dummy_taus, dummy_values, color='none', alpha=0)  # Invisible point
                
                # Set y-axis scale for optimization time with meaningful steps
                # First, decide on minimum range based on data variation
                data_range = max_opt_time - min_opt_time
                
                # Calculate padding as a percentage of the data range with minimum values
                padding_percent = 0.3  # 30% padding on both sides
                
                # Force a fixed minimum y-value well below the data
                # Always start optimization time axis from 0
                y2_min = 0
                
                if data_range < 0.5:  # Very little variation
                    # For very little variation, use a fixed range with extra bottom padding
                    y2_max = max_opt_time + 4  # At least 4 seconds above the maximum
                elif data_range < 2.0:  # Small variation
                    # For small variation, ensure ample padding at top
                    y2_max = max_opt_time + max(3.0, data_range)
                elif data_range < 10.0:  # Medium variation
                    # For medium variation, use percentage-based padding with minimum
                    padding = max(4.0, data_range * padding_percent)
                    y2_max = max_opt_time + padding
                else:  # Large variation
                    # For large variation, use generous percentage-based padding
                    padding = max(5.0, data_range * padding_percent)
                    y2_max = max_opt_time + padding
                
                ax2.set_ylim(y2_min, y2_max)
                
                # Use matplotlib's built-in AutoMinorLocator for nice ticks
                from matplotlib.ticker import AutoMinorLocator, MaxNLocator
                
                # Force a specific padding value at the bottom
                # Minimum time value should be max(0, min_opt_time - specific_padding)
                # Calculate the actual full range after top padding is applied
                full_range = y2_max - y2_min
                
                # Ensure the bottom 20-30% of the plot is empty space
                desired_bottom_padding_pct = 0.25  # 25% of plot height as padding
                
                # Calculate the new y2_min to achieve this
                required_y2_min = min_opt_time - (full_range * desired_bottom_padding_pct / (1 - desired_bottom_padding_pct))
                
                # Never go below zero
                y2_min = max(0, required_y2_min)
                
                # Update axis limits
                ax2.set_ylim(y2_min, y2_max)
                
                # Choose appropriate tick intervals based on the total axis range
                total_range = y2_max - y2_min
                
                if total_range < 10:
                    # For small total range, use 2-second ticks
                    ax2.yaxis.set_major_locator(MultipleLocator(2))
                elif total_range < 20:
                    # For medium range, use 5-second ticks
                    ax2.yaxis.set_major_locator(MultipleLocator(5))
                elif total_range < 50:
                    # For larger range, use 10-second ticks
                    ax2.yaxis.set_major_locator(MultipleLocator(10))
                elif total_range < 100:
                    # For very large range, use 25-second ticks
                    ax2.yaxis.set_major_locator(MultipleLocator(25))
                elif total_range < 200:
                    # For extremely large range, use 50-second ticks
                    ax2.yaxis.set_major_locator(MultipleLocator(50))
                elif total_range < 500:
                    # For massive range, use 100-second ticks
                    ax2.yaxis.set_major_locator(MultipleLocator(100))
                else:
                    # For enormous range, use 200-second ticks
                    ax2.yaxis.set_major_locator(MultipleLocator(200))
                
                # Only add y-axis label for rightmost plot in each row
                if model_idx == len(models_data) - 1:
                    ax2.set_ylabel('Opt. time (s)', color=color2, fontsize=8)
                ax2.tick_params(axis='y', labelcolor=color2, labelsize=7)
                
                # Set title and axis labels
                ax1.set_title(f'{model_name}', fontsize=9)
                # Only add x-axis label for bottom row
                if device_idx == num_devices - 1:
                    ax1.set_xlabel('Segment size', fontsize=8)
                
                # Add number of segments if available as text annotations
                for i, (tau, segment) in enumerate(zip(taus, segments)):
                    if not np.isnan(segment) and segment != 1:  # Only annotate if multiple segments
                        ax1.annotate(f'{int(segment)}', 
                                  xy=(tau, speedups[i]*100), 
                                  xytext=(0, 5),
                                  textcoords='offset points',
                                  ha='center',
                                  fontsize=6)
                
                # Add legend outside the plot
                lines = line1 + line2 + [jax_line]
                labels = [l.get_label() for l in lines]
                
                # We'll add a single centered legend for the entire figure later
                pass
                
            except Exception as e:
                print(f"Error plotting {model_name} on {device}: {str(e)}")
                ax.text(0.5, 0.5, f"Error plotting {model_name}",
                       ha='center', va='center', fontsize=8)
            
            # If we have fewer models than max, hide the empty plots
            if model_idx >= len(models_data):
                ax.axis('off')
        
        # Add device label for this row of plots
        if len(models_data) > 0:
            # Get the positions of the first and last plot in the row to determine row width
            first_ax_pos = axes[device_idx, 0].get_position()
            last_idx = min(len(models_data)-1, max_models_per_device-1)
            last_ax_pos = axes[device_idx, last_idx].get_position()
            
            # Position the device label centered above the row of plots
            # Further increased spacing between title and plots to 0.08
            row_center_x = (first_ax_pos.x0 + last_ax_pos.x1) / 2
            row_top_y = first_ax_pos.y1 + 0.06  # Slightly reduced space between title and plots
            
            # Add the device text directly without a background rectangle
            fig.text(row_center_x, row_top_y, device,
                    ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Bottom margin, vertical spacing, and top margin are already adjusted above
    
    # Create a completely new set of lines with the correct styles for the legend
    legend_fig = plt.figure(figsize=(1, 1))  # Create a dummy figure
    legend_ax = legend_fig.add_subplot(111)
    
    # Create lines with the desired styles - update markers to match plot
    gray_line = legend_ax.axhline(y=0, color='gray', linestyle='--', label='JAX')
    green_line = legend_ax.plot([0], [0], '.-', markersize=8, color='green', label='Constable')[0]
    blue_line = legend_ax.plot([0], [0], 'x-', markersize=6, markeredgewidth=1.5, color='blue', label='Optimization time')[0]
    
    # Close the dummy figure to prevent it from being displayed
    plt.close(legend_fig)
    
    # Add the legend to the main figure
    legend = fig.legend([gray_line, green_line, blue_line], 
              ['JAX', 'Constable', 'Optimization time'],
              loc='upper center', bbox_to_anchor=(0.5, 0.12),
              ncol=3, frameon=False, fontsize=8)
    
    # Add a button to save the figure as PDF
    # Position it properly below the legend with consistent spacing
    # Get legend position and size to calculate button position
    fig.canvas.draw()  # Ensure the figure is drawn so legend has the correct size
    legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    
    # Position the button 0.2cm below the legend
    # Convert 0.2cm to figure coordinates (depends on figure size in inches)
    cm_to_inches = 0.393701  # 1 cm = 0.393701 inches
    spacing_inches = 0.2 * cm_to_inches
    fig_height_inches = fig.get_figheight()
    spacing_fig_coords = spacing_inches / fig_height_inches
    
    button_left = 0.45
    button_width = 0.1
    button_height = 0.03
    button_bottom = legend_bbox.y0 - spacing_fig_coords - button_height
    
    save_ax = plt.axes([button_left, button_bottom, button_width, button_height])
    save_button = plt.Button(save_ax, 'Save PDF', color='lightgoldenrodyellow', hovercolor='0.975')
    
    def save_to_pdf(event):
        # Generate filename based on current date/time
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"segmentation_comparison_{timestamp}.pdf"
        
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

def main():
    if len(sys.argv) < 2:
        print("usage: ./graph_results.py <hist|time|compare|cost_model|segmentation> [options]")
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
        print("    ./graph_results.py cost_model -p <plot_title> <model_name> <baseline_csv> [<model_name> <baseline_csv> ...] [-p/-pn <plot_title> ...]")
        print("    Use -p to show all three configurations (baseline, no-fusion, no-zero)")
        print("    Use -pn to show only two configurations (baseline, no-zero) and hide the no-fusion bars")
        print("    Example:")
        print("      ./graph_results.py cost_model -p Xeon Llama results_llama_cost-model_baseline-cpu_2025-03-19_19:58:50.csv -pn A100 NasRNN results_nasrnn_cost-model_baseline-gpu_2025-03-19_19:58:50.csv")
        print()
        print("  segmentation: plot grid of graphs showing speedup and optimization time vs segment size")
        print("    ./graph_results.py segmentation -p <device> <model_name> <baseline_csv> <tau_file1> [<model_name> <baseline_csv> <tau_file1> ...] -stats <stats_file1> [<stats_file2> ...] [-p <device2> ...]")
        print("    Example:")
        print("      ./graph_results.py segmentation -p Threadripper ResNet beast/results_resnet_cost-model_baseline-cpu_2025-03-22_13:28:26.csv beast/results_resnet_tau=10-cpu_2025-03-23_15:00:06.csv -stats beast/stats_segmentation_2025-03-23_15:00:06.csv -p A100 ResNet a100/results_resnet_cost-model_baseline-gpu_2025-03-22_17:29:38.csv a100/results_resnet_tau=10-gpu_2025-03-24_03:03:48.csv -stats a100/stats_segmentation_2025-03-24_03:03:48.csv")
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
            
        # Multi-plot mode with -p/-pn flags
        plot_groups = []
        current_plot = None
        current_title = None
        current_show_no_fusion = True  # Default to showing no_fusion bars
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "-p" or sys.argv[i] == "-pn":
                # If we have a current plot group, add it before starting a new one
                if current_title is not None and current_plot is not None:
                    plot_groups.append((current_title, current_plot, current_show_no_fusion))
                
                # Determine if this group should show no_fusion bars
                show_no_fusion = (sys.argv[i] == "-p")  # Show for -p, hide for -pn
                
                # Start a new plot group
                if i + 1 < len(sys.argv):
                    current_title = sys.argv[i + 1]
                    current_plot = []
                    current_show_no_fusion = show_no_fusion
                    i += 2
                else:
                    print(f"Error: {sys.argv[i]} flag must be followed by a plot title")
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
            plot_groups.append((current_title, current_plot, current_show_no_fusion))
            
        # Plot the groups
        if plot_groups:
            if mode == "compare":
                # For compare mode, we need to ensure plot_groups has the right format
                # If we're coming from a cost_model parse with show_no_fusion flags
                if len(plot_groups) > 0 and len(plot_groups[0]) == 3:
                    # Remove the show_no_fusion flag
                    compare_plot_groups = [(pg[0], pg[1]) for pg in plot_groups]
                    plot_comparison(compare_plot_groups)
                else:
                    plot_comparison(plot_groups)
            else:  # cost_model
                plot_cost_model(plot_groups)
        else:
            print("Error: No valid plot groups found")
            sys.exit(1)
    
    elif mode == "segmentation":
        if len(sys.argv) < 6 or sys.argv[2] != "-p":
            print("Error: segmentation mode requires -p flag followed by device, model names, files, and -stats.")
            print("Example: ./graph_results.py segmentation -p Threadripper ResNet baseline.csv tau10.csv -stats stats_threadripper.csv -p A100 ...")
            sys.exit(1)
            
        # Parse the arguments: -p <device> <model> <baseline> <tau_file> ... -stats <stats_file> ... -p <next_device> ...
        plot_groups = []
        current_plot = None
        current_title = None
        current_stats = None
        expecting_stats = False
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "-p":
                # If we have a current plot group with models and stats, add it before starting a new one
                if current_title is not None and current_plot is not None and current_stats is not None:
                    plot_groups.append((current_title, current_plot, current_stats))
                elif current_title is not None and current_plot is not None:
                    print(f"Error: No stats files provided for device {current_title}")
                    sys.exit(1)
                
                # Start a new plot group
                if i + 1 < len(sys.argv):
                    current_title = sys.argv[i + 1]
                    current_plot = []
                    current_stats = None
                    expecting_stats = False
                    i += 2
                else:
                    print("Error: -p flag must be followed by a device name")
                    sys.exit(1)
            elif sys.argv[i] == "-stats":
                # Process stats files for the current device
                if current_title is None:
                    print("Error: -stats flag must follow a -p <device> section")
                    sys.exit(1)
                
                current_stats = []
                i += 1
                expecting_stats = True
                
                # Collect stats files until we hit another flag or end of args
                while i < len(sys.argv) and not sys.argv[i].startswith("-"):
                    current_stats.append(sys.argv[i])
                    i += 1
                
                if not current_stats:
                    print(f"Error: No stats files provided after -stats for device {current_title}")
                    sys.exit(1)
                
                expecting_stats = False
            else:
                # Add model triplets (model, baseline, tau_file) to the current plot
                if current_plot is None:
                    print("Error: Model information must follow a -p flag")
                    sys.exit(1)
                
                # We need at least 3 values: model name, baseline file, tau file
                if i + 2 < len(sys.argv) and not sys.argv[i+1].startswith("-") and not sys.argv[i+2].startswith("-"):
                    current_plot.append(sys.argv[i])     # Model name
                    current_plot.append(sys.argv[i+1])   # Baseline CSV
                    current_plot.append(sys.argv[i+2])   # Tau CSV example
                    i += 3
                else:
                    # Incomplete triplet
                    print(f"Error: Each model requires a name, baseline file, and tau file.")
                    sys.exit(1)
        
        # Add the last plot group if it exists
        if current_title is not None and current_plot is not None and current_stats is not None:
            plot_groups.append((current_title, current_plot, current_stats))
        elif current_title is not None and current_plot is not None:
            print(f"Error: No stats files provided for device {current_title}")
            sys.exit(1)
            
        # Plot the segmentation data
        if plot_groups:
            plot_segmentation(plot_groups)
        else:
            print("Error: No valid plot groups found")
            sys.exit(1)
        
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'hist', 'time', 'compare', 'cost_model', or 'segmentation'.")
        sys.exit(1)

if __name__ == "__main__":
    main()

