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

JAX_PIPELINE_NAME = "JaX"
ENZYME_PIPELINE_NAME = "DefOpt"
EQSAT_PIPELINE_NAME = "EqSat"

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
    stdem = stdev / math.sqrt(len(data))
    return mean, median, stdev, min_val, max_val, median_ci, stdem


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
            mean, median, stdev, min_val, max_val, (median_ci_lo, median_ci_hi), stdem = compute_stats(pipeline_data)
            print(f"{stage}, {pipeline}:")
            print(f"  mean   : {mean:.2f} ms")
            print(f"  median : {median:.3f}")
            print(f"     95% : [{median_ci_lo:.3f}, {median_ci_hi:.3f}]")
            print(f"  stdev  : {stdev:.2f} ms")
            print(f"  min    : {min_val:.2f} ms")
            print(f"  max    : {max_val:.2f} ms")
            print()

        print(f"t-test ({stage}):")
        eqsat_data = stage_data[stage_data['pipeline'] == EQSAT_PIPELINE_NAME]['runtime_ms']
        for pipeline in pipelines:
            if pipeline == EQSAT_PIPELINE_NAME:
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
    """Compute relative speedups of Enzyme-JAX and Constable against JAX"""
    # Get the first stage available in this CSV (assuming one stage per CSV)
    stages = model_data['stage'].unique()
    if len(stages) == 0:
        raise Exception("Error: No stages found in data")
    if set(stages) != set([stages[0]]):
        raise Exception("Error: Multiple stages in CSV")

    stage = stages[0]
    print(f"Using stage: {stage}")

    # Filter data for the stage
    stage_data = model_data[model_data['stage'] == stage]
    jax_median = np.median(stage_data[stage_data['pipeline'].str.strip() == JAX_PIPELINE_NAME]['runtime_ms'])
    enzyme_median = np.median(stage_data[stage_data['pipeline'] == ENZYME_PIPELINE_NAME]['runtime_ms'])
    eqsat_median = np.median(stage_data[stage_data['pipeline'] == EQSAT_PIPELINE_NAME]['runtime_ms'])

    # Calculate relative speedups (negative values mean slowdowns)
    enzyme_speedup = (jax_median - enzyme_median) / jax_median
    eqsat_speedup = (jax_median - eqsat_median) / jax_median

    return enzyme_speedup, eqsat_speedup


def compute_relative_speedup_runs(run_files):
    """
    Given a list of CSV files (one per run), compute the speedups for each run and
    return the median and standard error (SEM) for Enzyme-JAX and Constable.
    """
    enzyme_speedups = []
    constable_speedups = []
    for f in run_files:
        # try:
        df = pd.read_csv(f)
        d_speedup, c_speedup = compute_relative_speedup(df)
        enzyme_speedups.append(d_speedup)
        constable_speedups.append(c_speedup)
        # except Exception as e:
        #     print(f"Error processing {f}: {e}")

    assert(len(enzyme_speedups) == len(constable_speedups))
    assert(len(enzyme_speedups) > 0)
    # if len(enzyme_speedups) == 0 or len(constable_speedups) == 0:
    #     return None, None, None, None

    # Compute median and standard error
    enzyme_median = np.median(enzyme_speedups)
    enzyme_std = np.std(enzyme_speedups)
    enzyme_sem = enzyme_std / np.sqrt(len(enzyme_speedups))

    constable_median = np.median(constable_speedups)
    constable_std = np.std(constable_speedups)
    constable_sem = constable_std / np.sqrt(len(constable_speedups))

    return enzyme_median, enzyme_sem, constable_median, constable_sem


def process_plot_data(model_file_pairs):
    """
    For each model, use the provided CSV file (e.g. with _run1.csv) to locate all
    run files (e.g. _run1, _run2, _run3, ...), compute per-run speedups, and then
    aggregate (median and SEM) across runs.
    """
    models = []
    enzyme_speedups = []  # Each entry: (median, sem)
    constable_speedups = []  # Each entry: (median, sem)

    for i in range(0, len(model_file_pairs), 2):
        model_name = model_file_pairs[i]
        csv_file = model_file_pairs[i+1]

        directory = os.path.dirname(csv_file) or '.'
        basename = os.path.basename(csv_file)
        # Expecting a filename like: results_MODEL_cost-model_baseline-PLATFORM_DATE_run1.csv
        m = re.match(r"(.+)_run\d+\.csv", basename)
        if not m:
            raise Exception(f"Error: File {csv_file} does not match expected run pattern.")
        base_prefix = m.group(1)

        # Search the directory for all files with the same prefix and a _run<number>.csv suffix
        run_files = []
        for f in os.listdir(directory):
            if re.match(re.escape(base_prefix) + r"_run\d+\.csv", f):
                run_files.append(os.path.join(directory, f))
        if not run_files:
            raise Exception(f"Error: No run files found for {csv_file}")

        print(run_files)

        enzyme_median, enzyme_sem, constable_median, constable_sem = compute_relative_speedup_runs(run_files)
        if enzyme_median is None:
            continue

        models.append(model_name)
        enzyme_speedups.append((enzyme_median, enzyme_sem))
        constable_speedups.append((constable_median, constable_sem))

        print(f"{model_name}:")
        print(f"  Enzyme-JAX speedup: {enzyme_median*100:.2f}% ± {enzyme_sem*100:.2f}%")
        print(f"  Constable speedup: {constable_median*100:.2f}% ± {constable_sem*100:.2f}%")
        print()

    return models, enzyme_speedups, constable_speedups


def plot_comparison(plot_groups):
    """Plot bar charts comparing speedups for different models and platforms with error bars."""
    num_plots = len(plot_groups)

    if num_plots == 0:
        print("Error: No valid plot data provided")
        return

    # Gather all speedups (median) to set consistent y-axis limits
    all_medians = []
    for plot_group in plot_groups:
        title = plot_group[0]
        file_model_pairs = plot_group[1]
        models, enzyme_speedups, constable_speedups = process_plot_data(file_model_pairs)
        # Extract median only
        all_medians.extend([s[0] for s in enzyme_speedups])
        all_medians.extend([s[0] for s in constable_speedups])

    if len(all_medians) == 0:
        raise Exception("Error: No valid data to plot")

    # Calculate global min and max (in percentages)
    all_speedups_pct = [v*100 for v in all_medians]
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

        enzyme_vals = [median*100 for median, sem in enzyme_speedups]
        enzyme_err = [sem*100 for median, sem in enzyme_speedups]
        constable_vals = [median*100 for median, sem in constable_speedups]
        constable_err = [sem*100 for median, sem in constable_speedups]

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

    # save_ax = plt.axes([0.45, 0.005, 0.1, 0.04])
    # save_button = plt.Button(save_ax, 'Save PDF', color='lightgoldenrodyellow', hovercolor='0.975')

    # def save_to_pdf(event):
    #     from datetime import datetime
    #     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     filename = f"speedup_comparison_{timestamp}.pdf"
    #     save_button.ax.set_visible(False)
    #     fig.savefig(filename, format='pdf', bbox_inches='tight')
    #     abs_path = os.path.abspath(filename)
    #     print(f"Saved figure as {abs_path}")
    #     save_button.ax.set_visible(True)
    #     fig.canvas.draw_idle()

    # save_button.on_clicked(save_to_pdf)
    plt.show()


def find_related_cost_model_files(baseline_csv):
    directory = os.path.dirname(baseline_csv) or '.'
    basename = os.path.basename(baseline_csv)

    # Expected format:
    # results_MODEL_cost-model_baseline-PLATFORM_TIMESTAMP_run<number>.csv
    # where TIMESTAMP is in the form YYYY-MM-DD_HH:MM:SS
    pattern = r"results_(.+)_cost-model_baseline-(.+)_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})_run(\d+)\.csv"
    match = re.match(pattern, basename)
    if not match:
        raise Exception(f"Error: Cannot parse baseline filename pattern: {basename}")

    model = match.group(1)
    platform = match.group(2)
    timestamp = match.group(3)
    run_num = match.group(4)
    expected_no_fusion = f"results_{model}_cost-model_no-fusion-{platform}_{timestamp}_run{run_num}.csv"
    expected_no_zero   = f"results_{model}_cost-model_no-zero-{platform}_{timestamp}_run{run_num}.csv"

    no_fusion_file = None
    no_zero_file = None
    for file in os.listdir(directory):
        if file == expected_no_fusion:
            no_fusion_file = os.path.join(directory, file)
        elif file == expected_no_zero:
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
        raise Exception("Error: No stages found in baseline data")
    if set(stages) != set([stages[0]]):
        raise Exception("Error: Multiple stages in CSV")
    stage = stages[0]
    print(f"Using stage: {stage}")

    # Calculate median runtimes for EqSat pipeline only
    baseline_eqsat_data = baseline_df[(baseline_df['stage'] == stage) & (baseline_df['pipeline'].str.strip() == EQSAT_PIPELINE_NAME)]
    baseline_median = np.median(baseline_eqsat_data['runtime_ms'])

    no_fusion_median = None
    if no_fusion_df is not None:
        no_fusion_eqsat_data = no_fusion_df[(no_fusion_df['stage'] == stage) & (no_fusion_df['pipeline'].str.strip() == EQSAT_PIPELINE_NAME)]
        no_fusion_median = np.median(no_fusion_eqsat_data['runtime_ms'])

    no_zero_median = None
    if no_zero_df is not None:
        no_zero_eqsat_data = no_zero_df[(no_zero_df['stage'] == stage) & (no_zero_df['pipeline'].str.strip() == EQSAT_PIPELINE_NAME)]
        no_zero_median = np.median(no_zero_eqsat_data['runtime_ms'])

    # Get JAX data from the baseline file
    jax_data = baseline_df[baseline_df['pipeline'].str.strip() == JAX_PIPELINE_NAME]

    if len(jax_data) == 0:
        raise Exception("Error: No JAX data found")

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


def compute_cost_model_speedups_runs(baseline_file):
    """
    For a given baseline file (e.g. with _run1.csv), find all run files,
    compute the cost model speedups for each run, and then aggregate the results
    (mean and SEM) for baseline, no-fusion, and no-zero configurations.
    """
    directory = os.path.dirname(baseline_file) or '.'
    basename = os.path.basename(baseline_file)

    # Expecting a filename like: results_MODEL_cost-model_baseline-PLATFORM_DATE_run1.csv
    m = re.match(r"(.+)_run\d+\.csv", basename)
    if not m:
        raise Exception(f"Error: File {baseline_file} does not match expected run pattern.")
    base_prefix = m.group(1)

    # Locate all run files sharing the same prefix.
    run_files = []
    for f in os.listdir(directory):
        if re.match(re.escape(base_prefix) + r"_run\d+\.csv", f):
            run_files.append(os.path.join(directory, f))
    if not run_files:
        raise Exception(f"Warning: No run files found for {baseline_file}")

    baseline_speedups = []
    no_fusion_speedups = []
    no_zero_speedups = []

    for run_file in run_files:
        no_fusion_file, no_zero_file = find_related_cost_model_files(run_file)
        print(no_zero_file, no_fusion_file)
        # try:
        bs, nfs, nzs = compute_cost_model_speedups(run_file, no_fusion_file, no_zero_file)
        if bs is not None:
            baseline_speedups.append(bs)
        if nfs is not None:
            no_fusion_speedups.append(nfs)
        if nzs is not None:
            no_zero_speedups.append(nzs)
        # except Exception as e:
        #     print(f"Error processing run file {run_file}: {e}")

    assert(len(baseline_speedups) == len(no_fusion_speedups))
    assert(len(baseline_speedups) == len(no_zero_speedups))
    assert(len(baseline_speedups) > 0)

    bs_median = np.median(baseline_speedups)
    bs_sem = np.std(baseline_speedups) / np.sqrt(len(baseline_speedups))

    nfs_median = np.median(no_fusion_speedups)
    nfs_sem = np.std(no_fusion_speedups) / np.sqrt(len(no_fusion_speedups))

    nzs_median = np.median(no_zero_speedups)
    nzs_sem = np.std(no_zero_speedups) / np.sqrt(len(no_zero_speedups))

    return (bs_median, bs_sem), (nfs_median, nfs_sem), (nzs_median, nzs_sem)


def process_cost_model_data(model_file_pairs):
    """Process cost model data for multiple runs per model."""
    models = []
    baseline_speedups = []   # Each element: (median, sem)
    no_fusion_speedups = []  # Each element: (median, sem)
    no_zero_speedups = []    # Each element: (median, sem)

    for i in range(0, len(model_file_pairs), 2):
        model_name = model_file_pairs[i]
        baseline_csv = model_file_pairs[i+1]

        bs, nfs, nzs = compute_cost_model_speedups_runs(baseline_csv)

        models.append(model_name)
        baseline_speedups.append(bs)
        no_fusion_speedups.append(nfs)
        no_zero_speedups.append(nzs)

        print(f"Processing {model_name}:")
        print(f"  Baseline speedup: {bs[0]*100:.2f}% ± {bs[1]*100:.2f}%")
        print(f"  No fusion speedup: {nfs[0]*100:.2f}% ± {nfs[1]*100:.2f}%")
        print(f"  No zero speedup: {nzs[0]*100:.2f}% ± {nzs[1]*100:.2f}%")
        print()

    return models, baseline_speedups, no_fusion_speedups, no_zero_speedups


def plot_cost_model(plot_groups):
    """Plot bar charts comparing different cost model configurations with error bars."""
    num_plots = len(plot_groups)

    assert(num_plots > 0)

    all_plot_data = []
    all_speedups = []

    for title, file_model_pairs, show_no_fusion in plot_groups:
        models, baseline_speedups, no_fusion_speedups, no_zero_speedups = process_cost_model_data(file_model_pairs)
        if models:
            all_plot_data.append((title, models, baseline_speedups, no_fusion_speedups, no_zero_speedups, show_no_fusion))
            all_speedups.extend([bs[0] for bs in baseline_speedups])
            if show_no_fusion:
                all_speedups.extend([nfs[0] for nfs in no_fusion_speedups])
            all_speedups.extend([nzs[0] for nzs in no_zero_speedups])

    assert(all_plot_data)

    all_speedups_pct = [v*100 for v in all_speedups]
    min_val = np.floor(min(all_speedups_pct) / 5) * 5 - 5
    max_val = np.ceil(max(all_speedups_pct) / 5) * 5 + 5

    fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 5), sharey=True)
    if num_plots == 1:
        axes = [axes]

    for idx, (title, models, baseline_speedups, no_fusion_speedups, no_zero_speedups, show_no_fusion) in enumerate(all_plot_data):
        ax = axes[idx]
        if not models:
            ax.text(0.5, 0.5, f"No data for {title}",
                    horizontalalignment='center', verticalalignment='center')
            continue

        x = np.arange(len(models))
        if show_no_fusion:
            width = 0.25
            baseline_vals = [bs[0]*100 for bs in baseline_speedups]
            baseline_err  = [bs[1]*100 for bs in baseline_speedups]
            no_fusion_vals = [nfs[0]*100 for nfs in no_fusion_speedups]
            no_fusion_err  = [nfs[1]*100 for nfs in no_fusion_speedups]
            no_zero_vals = [nzs[0]*100 for nzs in no_zero_speedups]
            no_zero_err  = [nzs[1]*100 for nzs in no_zero_speedups]

            ax.bar(x - width, baseline_vals, width, yerr=baseline_err, capsize=5,
                   label='Constable w/ default cost model', color='green', edgecolor='black', linewidth=1)
            ax.bar(x, no_fusion_vals, width, yerr=no_fusion_err, capsize=5,
                   label='Constable w/o fusion costs (GPU only)', color='darkturquoise', edgecolor='black', linewidth=1)
            ax.bar(x + width, no_zero_vals, width, yerr=no_zero_err, capsize=5,
                   label='Constable w/o zero costs', color='orange', edgecolor='black', linewidth=1)
        else:
            width = 0.33
            baseline_vals = [bs[0]*100 for bs in baseline_speedups]
            baseline_err  = [bs[1]*100 for bs in baseline_speedups]
            no_zero_vals = [nzs[0]*100 for nzs in no_zero_speedups]
            no_zero_err  = [nzs[1]*100 for nzs in no_zero_speedups]

            ax.bar(x - width/2, baseline_vals, width, yerr=baseline_err, capsize=5,
                   label='Constable w/ base cost model', color='green', edgecolor='black', linewidth=1)
            ax.bar(x + width/2, no_zero_vals, width, yerr=no_zero_err, capsize=5,
                   label='Constable w/o zero costs', color='orange', edgecolor='black', linewidth=1)

        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
        ax.axhline(y=0, color='purple', linestyle='--', linewidth=2, label='JAX' if idx == 0 else "_nolegend_")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_title(title)
        if idx == 0:
            ax.set_ylabel('Relative speedup, %')

    plt.setp(axes, ylim=(min_val, max_val))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, wspace=0.3)

    # Build a single legend for the figure.
    all_handles = []
    all_labels = []
    handles, labels = axes[0].get_legend_handles_labels()
    for h, l in zip(handles, labels):
        # if l in ['JAX', 'Constable w/ base cost model', 'Constable w/o zero costs', 'Constable w/ default cost model', 'Constable w/o fusion costs (GPU only)']:
        all_handles.append(h)
        all_labels.append(l)

    fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=len(all_handles), frameon=False, handletextpad=1.0, columnspacing=2.0)

    # save_ax = plt.axes([0.45, 0.005, 0.1, 0.04])
    # save_button = plt.Button(save_ax, 'Save PDF', color='lightgoldenrodyellow', hovercolor='0.975')

    # def save_to_pdf(event):
    #     from datetime import datetime
    #     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     filename = f"cost_model_comparison_{timestamp}.pdf"
    #     save_button.ax.set_visible(False)
    #     fig.savefig(filename, format='pdf', bbox_inches='tight')
    #     abs_path = os.path.abspath(filename)
    #     print(f"Saved figure as {abs_path}")
    #     save_button.ax.set_visible(True)
    #     fig.canvas.draw_idle()

    # save_button.on_clicked(save_to_pdf)
    plt.show()


def find_all_tau_files(tau_example, device_type):
    """Find all segmentation tau=t files for a given model and device type"""
    directory = os.path.dirname(tau_example)
    if directory == '':
        directory = '.'

    basename = os.path.basename(tau_example)

    # Extract model name from the filename (e.g., "resnet", "llama")
    # Expecting results_MODEL_tau=X-PLATFORM_DATE.csv
    model_pattern = r'results_([^_]+)_'
    match = re.search(model_pattern, basename)

    if not match:
        raise Exception(f"Error: Cannot extract model name from filename: {basename}")

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
        # try:
        df = pd.read_csv(stats_file)
        if 'experiment_name' not in df.columns or 'eqsat_time' not in df.columns:
            raise Exception(f"Warning: File {stats_file} does not have required columns")

        matching_rows = df[df['experiment_name'].str.startswith(experiment_pattern)]

        if not matching_rows.empty:
            if len(matching_entries) > 0:
                # In case we chunk up the experiments, we allow multiple stats files. But we should have disjoint data across them
                raise Exception(f"Multiple entries found for experiment {experiment_pattern} across stats files")

            # There are two reasons why we might have multiple matching rows.
            # The first is that the benchmark involves running code muitiple times: for example, it may run both the primal and gradient passes. In this case, we should take the sum of optimisation times.
            # The second is that there are (potentially nested) while loops.
            # e.g
            # ...
            # while {
            #    ...
            #    while {
            #
            #    } [timestamp 1]
            # } [timestamp 2]
            # ...
            # [timestamp 3]
            # So only the last one will contain the correct time that contains every while loop.
            # Unfortunately a priori it is hard to tell which one of the two reasons causes this. To deal with this, we have a `while_loops` field in the stats CSV that tells us how many while loops a particular run had. This allows us to process and combine results so that at the end, we only have the first type of duplication remaining.

            matching_entries.append(matching_rows.iloc[0])
            for i in range(1, len(matching_rows)):
                current = matching_rows.iloc[i]
                for j in range(int(current['while_loops'])):
                    prev = matching_entries.pop()
                    assert current['eqsat_time'] >= prev['eqsat_time']
                    current['segments'] += prev['segments']

                matching_entries.append(current)

        # except FileNotFoundError:
        #     print(f"Error: Stats file '{stats_file}' not found.")
        #     continue

    if len(matching_entries) == 0:
        raise Exception(f"Error: No entries found for experiment {experiment_pattern}")

    return sum(row['eqsat_time'] for row in matching_entries)


def plot_segmentation(plot_groups):
    """Plot a grid of graphs showing speedup and optimization time vs segment size"""
    # Each plot group now contains: (device_name, model_file_pairs, stats_files)
    # Verify that each group has valid stats files
    # for device, model_file_pairs, stats_files in plot_groups:
    #     verified_stats_files = []
    #     for stats_file in stats_files:
    #         if os.path.exists(stats_file):
    #             verified_stats_files.append(stats_file)
    #         else:
    #             raise Exception(f"Warning: Stats file '{stats_file}' not found for device {device}")

    #     # Update the stats_files in the plot group with the verified ones
    #     if not verified_stats_files:
    #         raise Exception(f"Error: No valid stats files provided for device {device}")
    #         return

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
        # 3 because (model_name, baseline, tau)
        for i in range(0, len(file_model_pairs), 3):
            model_name = file_model_pairs[i]
            # Need at least two files (baseline and one tau file)
            if i + 2 > len(file_model_pairs):
                raise Exception(f"Warning: Insufficient files for model {model_name}: need two files")

            baseline_file = file_model_pairs[i+1]
            tau_file_example = file_model_pairs[i+2]  # Example tau file to determine platform (cpu/gpu)

            # Extract device type (cpu/gpu) from the filename
            platform_pattern = r'-(cpu|gpu)'
            platform_match = re.search(platform_pattern, os.path.basename(tau_file_example))

            if not platform_match:
                raise Exception(f"Error: Cannot determine platform (cpu/gpu) from filename: {tau_file_example}")

            platform = platform_match.group(1)

            tau_files = find_all_tau_files(tau_file_example, platform)
            if not tau_files:
                raise Exception(f"Error: No tau files found for {model_name} on {device}")

            print(f"files for {model_name} on {device}:")
            for (tau, filename) in tau_files:
                print(f"tau={tau} : {filename}")
            print()

            device_models[device].append((model_name, baseline_file, tau_files, platform))

    if not device_models:
        raise Exception("Error: No valid data to plot")

    # one row per device
    num_devices = len(device_models)
    device_names = list(device_models.keys())
    max_models_per_device = max(len(models) for models in device_models.values())

    if max_models_per_device == 0:
        raise Exception("Error: No valid models to plot")

    # Create figure with subplots - one row per device, one column per model
    fig, axes = plt.subplots(num_devices, max_models_per_device,
                             figsize=(3.5 * max_models_per_device, 1.5 * num_devices + 1.0),
                             squeeze=False)

    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.96, wspace=0.55, hspace=1.0, bottom=0.25, top=0.92)

    for device_idx, device in enumerate(device_names):
        models_data = device_models[device]

        for model_idx, (model_name, baseline_file, tau_files, platform) in enumerate(models_data):
            ax = axes[device_idx, model_idx]
            try:
                # Read baseline file to get JAX performance
                baseline_df = pd.read_csv(baseline_file)
                jax_data = baseline_df[baseline_df['pipeline'].str.strip() == 'JaX']

                if jax_data.empty:
                    raise Exception(f"Warning: No JAX data in {baseline_file}")

                stages = jax_data['stage'].unique()
                assert(len(stages) == 1)
                stage = stages[0]
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
                        raise Exception(f"Error: No optimization time found for {model_in_file} tau={tau} on {platform}")

                    # Get EqSat performance
                    eqsat_df = pd.read_csv(tau_file)
                    eqsat_data = eqsat_df[(eqsat_df['stage'] == stage) & (eqsat_df['pipeline'] == EQSAT_PIPELINE_NAME)]

                    if eqsat_data.empty:
                        raise Exception(f"Warning: No EqSat data in {tau_file}")

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

                min_opt_time = min(opt_times) if opt_times else 0
                max_opt_time = max(opt_times) if opt_times else 0

                line1 = ax1.plot(taus, [s*100 for s in speedups], marker='.', markersize=8, color=color1, label='Constable')

                if model_idx == 0:
                    ax1.set_ylabel('Relative speedup, %', color=color1, fontsize=8)
                ax1.tick_params(axis='y', labelcolor=color1, labelsize=7)
                ax1.tick_params(axis='x', labelsize=7)

                min_speedup = min([s*100 for s in speedups]) if speedups else 0
                max_speedup = max([s*100 for s in speedups]) if speedups else 0

                # Calculate padding as a percentage of data range, with minimum padding
                data_range = max(max_speedup - min_speedup, 20)
                padding = max(5, data_range * 0.15)

                y_min = min(-10, min_speedup - padding)

                if max_speedup > 25:
                    y_max = max_speedup + padding
                else:
                    y_max = 30

                ax1.set_ylim(y_min, y_max)

                from matplotlib.ticker import MultipleLocator
                ax1.yaxis.set_major_locator(MultipleLocator(15))

                jax_line = ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='JAX')

                # Secondary y-axis for optimization time
                ax2 = ax1.twinx()
                # Manually insert values near zero to force axis to show more of the bottom
                # This is a workaround to ensure space below the data points
                dummy_taus = [min(taus)]
                dummy_values = [0.001]

                line2 = ax2.plot(taus, opt_times, marker='x', markersize=6, markeredgewidth=1.5, color=color2, label='Optimization time')
                dummy_line = ax2.plot(dummy_taus, dummy_values, color='none', alpha=0)  # Invisible point

                data_range = max_opt_time - min_opt_time
                padding_percent = 0.3
                y2_min = 0

                if data_range < 0.5:
                    y2_max = max_opt_time + 4  # At least 4 seconds above the maximum
                elif data_range < 2.0:
                    y2_max = max_opt_time + max(3.0, data_range)
                elif data_range < 10.0:
                    padding = max(4.0, data_range * padding_percent)
                    y2_max = max_opt_time + padding
                else:
                    padding = max(5.0, data_range * padding_percent)
                    y2_max = max_opt_time + padding

                ax2.set_ylim(y2_min, y2_max)

                from matplotlib.ticker import AutoMinorLocator, MaxNLocator
                full_range = y2_max - y2_min
                desired_bottom_padding_pct = 0.25
                required_y2_min = min_opt_time - (full_range * desired_bottom_padding_pct / (1 - desired_bottom_padding_pct))
                y2_min = max(0, required_y2_min)

                ax2.set_ylim(y2_min, y2_max)

                total_range = y2_max - y2_min
                if total_range < 10:
                    ax2.yaxis.set_major_locator(MultipleLocator(2))
                elif total_range < 20:
                    ax2.yaxis.set_major_locator(MultipleLocator(5))
                elif total_range < 50:
                    ax2.yaxis.set_major_locator(MultipleLocator(10))
                elif total_range < 100:
                    ax2.yaxis.set_major_locator(MultipleLocator(25))
                elif total_range < 200:
                    ax2.yaxis.set_major_locator(MultipleLocator(50))
                elif total_range < 500:
                    ax2.yaxis.set_major_locator(MultipleLocator(100))
                else:
                    ax2.yaxis.set_major_locator(MultipleLocator(200))

                # Only add y-axis label for rightmost plot in each row
                if model_idx == len(models_data) - 1:
                    ax2.set_ylabel('Opt. time (s)', color=color2, fontsize=8)
                ax2.tick_params(axis='y', labelcolor=color2, labelsize=7)

                ax1.set_title(f'{model_name}', fontsize=9)
                # Only add x-axis label for bottom row
                if device_idx == num_devices - 1:
                    ax1.set_xlabel('Segment size', fontsize=8)

                lines = line1 + line2 + [jax_line]
                labels = [l.get_label() for l in lines]
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
            row_center_x = (first_ax_pos.x0 + last_ax_pos.x1) / 2
            row_top_y = first_ax_pos.y1 + 0.06

            fig.text(row_center_x, row_top_y, device,
                    ha='center', va='center', fontsize=11, fontweight='bold')

    legend_fig = plt.figure(figsize=(1, 1))  # dummy figure for legend only
    legend_ax = legend_fig.add_subplot(111)

    gray_line = legend_ax.axhline(y=0, color='gray', linestyle='--', label='JAX')
    green_line = legend_ax.plot([0], [0], '.-', markersize=8, color='green', label='Constable')[0]
    blue_line = legend_ax.plot([0], [0], 'x-', markersize=6, markeredgewidth=1.5, color='blue', label='Optimization time')[0]

    plt.close(legend_fig)

    legend = fig.legend([gray_line, green_line, blue_line],
              ['JAX', 'Constable', 'Optimization time'],
              loc='upper center', bbox_to_anchor=(0.5, 0.12),
              ncol=3, frameon=False, fontsize=8)

    # # Add a button to save the figure as PDF
    # # Position it properly below the legend with consistent spacing
    # # Get legend position and size to calculate button position
    # fig.canvas.draw()  # Ensure the figure is drawn so legend has the correct size
    # legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())

    # # Position the button 0.2cm below the legend
    # # Convert 0.2cm to figure coordinates (depends on figure size in inches)
    # cm_to_inches = 0.393701  # 1 cm = 0.393701 inches
    # spacing_inches = 0.2 * cm_to_inches
    # fig_height_inches = fig.get_figheight()
    # spacing_fig_coords = spacing_inches / fig_height_inches

    # button_left = 0.45
    # button_width = 0.1
    # button_height = 0.03
    # button_bottom = legend_bbox.y0 - spacing_fig_coords - button_height

    # save_ax = plt.axes([button_left, button_bottom, button_width, button_height])
    # save_button = plt.Button(save_ax, 'Save PDF', color='lightgoldenrodyellow', hovercolor='0.975')

    # def save_to_pdf(event):
    #     # Generate filename based on current date/time
    #     from datetime import datetime

    #     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     filename = f"segmentation_comparison_{timestamp}.pdf"

    #     # Temporarily hide the save button for the PDF export
    #     save_button.ax.set_visible(False)

    #     # Save figure to current directory (button won't be visible in the saved file)
    #     fig.savefig(filename, format='pdf', bbox_inches='tight')
    #     abs_path = os.path.abspath(filename)
    #     print(f"Saved figure as {abs_path}")

    #     # Make the button visible again for continued interaction
    #     save_button.ax.set_visible(True)
    #     fig.canvas.draw_idle()

    # save_button.on_clicked(save_to_pdf)
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

