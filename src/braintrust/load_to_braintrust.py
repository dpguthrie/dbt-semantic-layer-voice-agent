#!/usr/bin/env python
import argparse
import json
import os

import braintrust
from server.settings import settings


def load_jsonl_to_braintrust(jsonl_path, dataset_name=None):
    """
    Load a JSONL file into a Braintrust dataset.

    Args:
        jsonl_path: Path to the JSONL file
        dataset_name: Name for the Braintrust dataset. If None, uses the filename.

    Returns:
        The Braintrust dataset object
    """
    # Use the filename (without extension) as the dataset name if not provided
    if dataset_name is None:
        dataset_name = os.path.basename(jsonl_path).split(".")[0]

    # Create dataset
    dataset = braintrust.init_dataset(
        settings.braintrust_project_name,
        name=dataset_name,
    )

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue  # Skip empty lines

            record = json.loads(line)
            dataset.insert(**record)


def main():
    """Load JSONL files into Braintrust datasets."""
    parser = argparse.ArgumentParser(
        description="Load JSONL files into Braintrust datasets"
    )
    parser.add_argument("--path", help="Path to JSONL file to load")
    parser.add_argument(
        "--dataset-name",
        help="Name for the dataset (default: use filename)",
    )

    args = parser.parse_args()

    # Ensure dataset_name is provided
    if args.dataset_name is None:
        raise ValueError("Dataset name must be provided")

    # Load each JSONL file into a Braintrust dataset
    load_jsonl_to_braintrust(args.path, args.dataset_name)


if __name__ == "__main__":
    main()
