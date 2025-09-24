# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "absl-py",
#     "numpy",
#     "pandas",
#     "tabulate",
#     "seaborn",
#     "matplotlib",
# ]
# ///

import base64
import collections
import glob
import os
import json
import re
import tempfile
import zipfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from absl import app, flags, logging

_ARTIFACT_DIR = flags.DEFINE_string(
    "artifact_dir", None, "Path to the artifact directory", required=True
)

_CSV_FILE = flags.DEFINE_string("csv_file", None, "Path to the csv file", required=True)

_JSON_FILE = flags.DEFINE_string(
    "json_file", None, "Path to the json file", required=True
)


def get_machine_name(filename: str) -> str:
    pattern = re.compile(r"benchmark-(.+?)-\d+\.\d+")
    match = pattern.search(filename)
    if match:
        return match.group(1)
    raise ValueError(f"Could not find machine name in filename: {filename}")


def main(_) -> None:
    # unzip the outputs.zip
    tempdir_results = tempfile.TemporaryDirectory(delete=False)
    logging.info("Extracting outputs to %s", tempdir_results.name)

    for file in glob.glob(f"{_ARTIFACT_DIR.value}/**/outputs.zip", recursive=True):
        logging.info("Unzipping %s...", file)
        machine_name = file.strip(f"{_ARTIFACT_DIR.value}/").split("/")[0]

        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(os.path.join(tempdir_results.name, machine_name))

    # Merge the CSVs into a single dataframe
    combined_dfs = collections.defaultdict(list)
    for file in glob.glob(f"{tempdir_results.name}/*/results_*.csv"):
        machine_name = get_machine_name(os.path.basename(os.path.dirname(file)))
        expt = os.path.basename(file).replace("results_", "").replace(".csv", "")

        df = pd.read_csv(file)
        df["machine"] = machine_name

        combined_dfs[expt].append(df)

    # Merge the CSVs into a single dataframe
    final_dfs = dict()
    for expt, dfs in combined_dfs.items():
        merged = pd.concat(dfs, ignore_index=True)
        final_dfs[expt] = merged

    dfs_merged = pd.concat(final_dfs.values(), ignore_index=True)
    dfs_merged.to_csv(_CSV_FILE.value, index=False)
    logging.info("Saved combined results to %s", _CSV_FILE.value)

    records = [
        {
            "name": f"{row['Benchmark Name']} / {row['Pass Pipeline']} / {row['Backend']} / {row['Key']}",
            "unit": "s",
            "value": row["Time"],
        }
        for _, row in df.iterrows()
    ]

    with open(_JSON_FILE.value, "w") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    app.run(main)()
