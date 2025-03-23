import pandas as pd
import argparse
from math import ceil
import re
import sys

TOTAL_ILP_BUDGET = 300
TOTAL_SATURATION_BUDGET = 300

def parse_experiment_name(exp_name):
    """
    Expect experiment_name in the format:
      model_tau=<tau>-platform_datetime
    e.g., "maxtext_tau=25-gpu_2025-03-23_11:58:22"
    Returns (model, tau) or (None, None) if not matched.
    """
    m = re.match(r"([^_]+)_tau=([0-9]+)-", exp_name)
    if m:
        return m.group(1), int(m.group(2))
    else:
        return None, None

def aggregate_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        sys.exit(f"Error reading CSV file: {e}")
    # filter out the no segmentation passes
    df = df[df['experiment_name'].str.contains('_tau=')].copy()
    parsed = df['experiment_name'].apply(parse_experiment_name)
    df[['model','tau']] = pd.DataFrame(parsed.tolist(), index=df.index)
    df = df.dropna(subset=['model','tau'])
    agg = df.groupby(['model','tau'], as_index=False)['segments'].sum()
    agg['ILP_TIME_LIMIT'] = agg['segments'].apply(lambda s: ceil(TOTAL_ILP_BUDGET / s))
    agg['SATURATION_TIME_LIMIT'] = agg['segments'].apply(lambda s: ceil(TOTAL_SATURATION_BUDGET / s))
    return agg

def main():
    parser = argparse.ArgumentParser(
        description="Compute ILP and Saturation time limits per model and tau based on segmentation stats."
    )
    parser.add_argument('--csv', type=str, required=True,
                        help="Path to the segmentation stats CSV file.")
    parser.add_argument('--model', type=str,
                        help="(Optional) Query specific model.")
    parser.add_argument('--tau', type=int,
                        help="(Optional) Query specific tau value.")
    parser.add_argument('--output', type=str,
                        help="(Optional) Output aggregated results to CSV file.")
    args = parser.parse_args()
    agg = aggregate_csv(args.csv)
    
    if args.model is not None and args.tau is not None:
        row = agg[(agg['model'] == args.model) & (agg['tau'] == args.tau)]
        if row.empty:
            sys.exit(f"No entry found for model '{args.model}' with tau={args.tau}.")
        else:
            ilp_limit = row['ILP_TIME_LIMIT'].values[0]
            sat_limit = row['SATURATION_TIME_LIMIT'].values[0]
            print(f"{ilp_limit} {sat_limit}")
    elif args.output:
        agg.to_csv(args.output, index=False)
        print(f"Aggregated results written to {args.output}")
    else:
        print(agg.to_string(index=False))

if __name__ == '__main__':
    main()
