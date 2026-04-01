# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "absl-py",
#     "numpy",
#     "pandas",
# ]
# ///

import collections
import glob
import os
import json
import re
import tempfile
import zipfile

import pandas as pd
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
    tempdir_results = tempfile.TemporaryDirectory()
    logging.info("Extracting outputs to %s", tempdir_results.name)

    zip_files = sorted(
        glob.glob(
            os.path.join(_ARTIFACT_DIR.value, "**", "outputs.zip"), recursive=True
        )
    )
    logging.info("Found %d outputs.zip file(s)", len(zip_files))

    for file in zip_files:
        logging.info("Parsing zip: %s", file)

        # Use the first path segment under artifact_dir as the machine bucket.
        rel_dir = os.path.relpath(os.path.dirname(file), _ARTIFACT_DIR.value)
        machine_name = (
            rel_dir.split(os.sep, 1)[0] if rel_dir != os.curdir else "unknown"
        )

        extract_dir = os.path.join(tempdir_results.name, machine_name)
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    # Merge the CSVs into a single dataframe
    combined_dfs = collections.defaultdict(list)
    csv_files = sorted(
        glob.glob(
            os.path.join(tempdir_results.name, "**", "results_*.csv"), recursive=True
        )
    )
    logging.info("Found %d results_*.csv file(s)", len(csv_files))

    for file in csv_files:
        logging.info("Parsing CSV: %s", file)
        rel_csv = os.path.relpath(file, tempdir_results.name)
        machine_name = rel_csv.split(os.sep, 1)[0]
        expt = os.path.basename(file).replace("results_", "").replace(".csv", "")

        df = pd.read_csv(file)
        df["machine"] = machine_name
        df["experiment"] = expt

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
        for _, row in dfs_merged.iterrows()
    ]

    with open(_JSON_FILE.value, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    app.run(main)
