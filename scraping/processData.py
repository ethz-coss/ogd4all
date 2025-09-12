import csv
import os
import zipfile
import logging
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...
from utils import clean
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))


def extract_data_from_zip(csv_path: str, file_formats: list, groupOwner: str):
    gpkg_count = 0
    tif_count = 0
    skipped_count = 0

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            zip_path = row['file']
            zip_path = os.path.join(script_dir, f'../data/opendata/{groupOwner}/raw', zip_path)

            dataset_title = row['title']
            logging.info(f"Processing {dataset_title}, renaming to {clean(dataset_title)}")

            # Create extracted dir if not exists
            extract_dir = os.path.join(script_dir, f'../data/opendata/{groupOwner}/extracted')

            file_id = Path(zip_path).stem  # gets the filename without extension
            data_files = []

            logging.info(f"\nExtracting {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.startswith(f"data/") and not member.endswith('/'):
                            # Extract only files under the data folder
                            target_path = Path(extract_dir) / Path(member).relative_to(f"data")
                            suffix = target_path.suffix

                            if suffix == '.gpkg' and '.gpkg' in file_formats:
                                gpkg_count += 1
                            elif suffix == '.tif' and '.tif' in file_formats:
                                tif_count += 1
                            else:
                                skipped_count += 1
                                continue # Skip other files for now

                            # Update target_path to include the cleaned dataset title
                            target_path = Path(extract_dir) / (clean(dataset_title) + suffix)

                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            if target_path.exists():
                                target_path.unlink()  # Remove the file if it already exists
                            with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                                target.write(source.read())
                            data_files.append(target_path)
            except zipfile.BadZipFile:
                logging.error(f"Error: {zip_path} is not a zip file or it is corrupted.")
                continue
            except FileNotFoundError:
                logging.error(f"Error: {zip_path} not found.")
                continue

            if (len(data_files) > 1):
                logging.error(f"Error: {zip_path} contains multiple geopackage files under data/ folder. This should not be the case.")

            logging.info(f"\nExtracted {len(data_files)} files from {file_id}:")
            for f in data_files:
                logging.info(f" - {f}")

    # Write summary of extracted and skipped files
    logging.info(f"\nSummary:")
    logging.info(f" - {gpkg_count} geopackage files extracted.")
    logging.info(f" - {tif_count} tif files extracted.")
    logging.info(f" - {skipped_count} files skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all raw data files into folders")
    parser.add_argument("--groupOwner", type=str, default="50000006",help="The groupOwner id where raw data should be extracted from.")
    parser.add_argument("--verbose", action="store_true", help="Print more information during processing.")
    parser.add_argument("--file_formats", nargs='*', default=['.gpkg'], help="List of file formats to extract from the zip files.")
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO if args.verbose else logging.WARNING)

    csv_file_path = os.path.join(script_dir, f'../data/opendata/{args.groupOwner}/downloads.csv')
    extract_data_from_zip(csv_file_path, args.file_formats, args.groupOwner)