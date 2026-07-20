from pathlib import Path
import shutil

from utils.logger import logger


def canonicalize_tree(input_folder: Path, output_folder: Path):
    # Takes a folder containing subfolders with the raw data and the corresponding encodings
    # and exports it into a ready to load folder.
    #
    # input_folder/
    # - class_a/
    #   - encodings/
    #     - encodings_a.csv
    #     - encodings_b.csv
    #     - encodings_c.csv
    # - class_b/
    #   - encodings/
    #     - encodings_a.csv
    #     - encodings_b.csv
    #     - encodings_c.csv
    # - class_n/
    #   - encodings/
    #     - encodings_a.csv
    #     - encodings_b.csv
    #     - encodings_c.csv

    if not input_folder.exists():
        raise ValueError("The provided input path does not exists")

    if not input_folder.is_dir():
        raise ValueError("The provided input path is not a folder")

    if output_folder.exists() and not output_folder.is_dir():
        raise ValueError("The provided output path already exists but is not a folder")

    if (
        output_folder.exists()
        and output_folder.is_dir()
        and any(output_folder.iterdir())
    ):
        raise ValueError("The provided output folder already exists but is not empty")

    output_folder.mkdir(parents=True, exist_ok=True)

    with logger.status("Moving dataset..."):
        for child in input_folder.iterdir():
            if not child.is_dir():
                logger.warning(f"{child.name} is not a folder, skipping...")
                continue

            encodings_folder = child / "encodings"

            if not encodings_folder.exists() or not encodings_folder.is_dir():
                logger.warning(f"Encodings were not found on {child.name}, skipping...")
                continue

            child_output_folder = output_folder / child.name
            child_output_folder.mkdir(exist_ok=True)

            for encoding in encodings_folder.iterdir():
                if encoding.suffix != ".csv":
                    logger.warning(f"{encoding.name} is not a CSV, skipping...")
                    continue

                shutil.copy(encoding.absolute(), child_output_folder / encoding.name)
