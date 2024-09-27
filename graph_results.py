#!/usr/bin/env python3

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from math import floor, ceil, sqrt

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


def main():
    if len(sys.argv) != 3 or (sys.argv[1] not in ["hist", "time"]):
        print("usage: ./graph_results.py <hist|time> <filename.csv>")
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

    if sys.argv[1] == "hist":
        plot_histograms(df)
    else:
        plot_time(df)

if __name__ == "__main__":
    main()
