import sys

sys.path.append('..')
sys.path.append('../..')  # if running in this folder

import argparse
import os
import shutil
import subprocess

from loguru import logger

from vimms.Evaluation import RealEvaluator

DEFAULT_OPENMS_DIR = "/Applications/OpenMS-3.0.0/bin"
DEFAULT_INI_FILE = "../../batch_files/FeatureFinderCentroided.ini"


def get_peak_picked_csv(seed_file):
    base_name = os.path.basename(seed_file)
    seed_picked_peaks = os.path.splitext(base_name)[0] + '_openms.csv'
    seed_dir = os.path.split(seed_file)[0]
    seed_picked_peaks_csv = os.path.join(seed_dir, seed_picked_peaks)
    return seed_picked_peaks_csv


def remove_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    lines_to_remove = [1, 2, 3, 5]
    filtered_lines = [line for i, line in enumerate(lines) if i + 1 not in lines_to_remove]
    with open(filename, 'w') as file:
        file.writelines(filtered_lines)


def pick_peaks_openms(seed_file, openms_dir, ini_file=None):
    if ini_file is None:
        ini_file = DEFAULT_INI_FILE
        logger.info(f'Using default ini file {ini_file}')

    seed_picked_peaks_csv = get_peak_picked_csv(seed_file)
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    temp_feature_featureXML = os.path.join(temp_dir, "temp_feature.featureXML")

    subprocess.run(
        [f"{openms_dir}/FeatureFinderCentroided", "-ini", ini_file, "-in", seed_file, "-out", temp_feature_featureXML])
    subprocess.run([f"{openms_dir}/TextExporter", "-in", temp_feature_featureXML, "-out", seed_picked_peaks_csv])

    remove_lines(seed_picked_peaks_csv)
    shutil.rmtree(temp_dir)


def extract_boxes(seed_file, openms_dir=None, ini_file=None):
    seed_picked_peaks_csv = get_peak_picked_csv(seed_file)
    logger.info(f'Peak picking using openms, results will be in {seed_picked_peaks_csv}')

    # Check if the file already exists
    if os.path.isfile(seed_picked_peaks_csv):
        logger.info(f'{seed_picked_peaks_csv} already exists, skipping peak picking.')
        return seed_picked_peaks_csv

    if openms_dir is None:
        openms_dir = DEFAULT_OPENMS_DIR
    pick_peaks_openms(seed_file, openms_dir, ini_file)

    return seed_picked_peaks_csv


def evaluate_fragmentation(peaklist_file, mzml_file, isolation_width):
    """
    Evaluate boxes against fragmentation spectra using the `RealEvaluator` class.
    Args:
        peaklist_file: Path to peak-picked CSV file containing boxes.
                  Can be produced by OpenMS
        mzml_file: the path to fragmentation mzML
        isolation_width: isolation width

    Returns: a RealEvaluator object.
    """
    eva = RealEvaluator.from_unaligned_openms(peaklist_file)

    fullscan_name = 'Dummy'
    mzmls = [mzml_file]
    eva.add_info(fullscan_name, mzmls, isolation_width=isolation_width)
    return eva


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract boxes using ViMMS')
    parser.add_argument('seed_file', type=str)
    parser.add_argument('mzml_file', type=str, help="Path to the mzML file.")
    parser.add_argument('--openms_dir', type=str, default=None)
    parser.add_argument('--openms_ini_file', type=str, default=None)
    parser.add_argument('--isolation_width', type=float, default=0.7, help="Isolation width for fragmentation.")
    args = parser.parse_args()

    csv_file = extract_boxes(args.seed_file, args.openms_dir, args.openms_ini_file)

    logger.debug(f'Now processing fragmentation file {args.mzml_file}')
    eva = evaluate_fragmentation(csv_file, args.mzml_file, args.isolation_width)
    print(eva.summarise())
