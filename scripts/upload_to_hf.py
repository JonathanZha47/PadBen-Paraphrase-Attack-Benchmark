#!/usr/bin/env python3
"""Utility script for uploading the PADBen dataset to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, HfFolder, upload_folder
from huggingface_hub.utils import HfHubHTTPError


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Example:
        >>> parse_args()  # doctest: +SKIP
    """

    default_data_dir = Path(__file__).resolve().parents[1] / "data" / "task_data" / "tasks" 

    parser = argparse.ArgumentParser(
        description=(
            "Upload the PADBen dataset directory to a Hugging Face datasets repository.\n"
            "The script expects the huggingface_hub package to be installed and an "
            "authentication token to be available via --token or the HF_TOKEN environment "
            "variable."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help=(
            "Absolute path to the dataset directory to upload. Defaults to the project's "
            "data directory."
        ),
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help=(
            "Target Hugging Face dataset repository in the format 'username/dataset-name'."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "User access token for the Hugging Face Hub. If omitted, the script falls back "
            "to the cached token from `huggingface-cli login`."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repository as private instead of public.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add PADBen dataset",
        help="Commit message to use for the dataset upload.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target git branch on the Hub repository (default: main).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging level for the script (default: INFO).",
    )

    return parser.parse_args()


def upload_dataset(
    *,
    data_dir: Path,
    repo_id: str,
    token: Optional[str],
    private: bool,
    commit_message: str,
    revision: str,
) -> None:
    """Upload a local dataset directory to a Hugging Face Hub dataset repository.

    Args:
        data_dir (Path): Absolute path to the dataset directory that will be uploaded.
        repo_id (str): Hub repository identifier in the form ``username/dataset``.
        token (Optional[str]): Hugging Face Hub authentication token. When ``None``,
            the function attempts to use the cached token from the local machine.
        private (bool): Whether the Hub repository should be private.
        commit_message (str): Commit message for the upload.
        revision (str): Repository branch or tag that will receive the upload.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist or is empty.
        ValueError: If no authentication token can be resolved.
        RuntimeError: If the upload fails due to an API error.

    Example:
        >>> upload_dataset(  # doctest: +SKIP
        ...     data_dir=Path("/abs/path/to/data"),
        ...     repo_id="username/padben",
        ...     token=None,
        ...     private=False,
        ...     commit_message="Add PADBen dataset",
        ...     revision="main",
        ... )
    """

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    if not any(data_dir.iterdir()):
        raise FileNotFoundError(
            f"Dataset directory {data_dir} is empty. Populate it before uploading."
        )

    hf_token = token or HfFolder.get_token()
    if not hf_token:
        raise ValueError(
            "Hugging Face authentication token is required. Use --token or run "
            "`huggingface-cli login` to cache a token."
        )

    LOGGER.debug("Resolved dataset directory at %s", data_dir)

    api = HfApi(token=hf_token)

    try:
        LOGGER.info(
            "Creating (or reusing) dataset repository %s with privacy=%s", repo_id, private
        )
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
            private=private,
            exist_ok=True,
        )

        LOGGER.info("Uploading contents of %s to Hugging Face Hub", data_dir)
        upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(data_dir),
            token=hf_token,
            commit_message=commit_message,
            revision=revision,
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Failed to upload dataset: {exc}") from exc

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    LOGGER.info("Dataset upload completed successfully. View it at %s", dataset_url)


def main() -> None:
    """Run the upload routine based on command-line arguments."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        upload_dataset(
            data_dir=args.data_dir.resolve(),
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
            revision=args.revision,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

