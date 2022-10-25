# check ms2 matches
# simple script that loads an msp and an mzml and sees how many of the
# spectra in the MSP file can be matched to a spectrum in an ms2 scan
# in the .mzml
import sys

from vimms.Box import GenericBox

sys.path.append('..')
sys.path.append('../..')  # if running in this folder

import argparse
import glob
import os

import numpy as np
import pandas as pd
from loguru import logger
from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.library_matching.spec_libraries import SpectralLibrary
from mass_spec_utils.library_matching.spectrum import Spectrum, SpectralRecord

from vimms.Common import load_obj, ScanParameters, PROTON_MASS


def process_block(block, file_name):
    peak_start_pos = [b.startswith("Num") for b in block].index(True)
    peaks = []
    for line in block[peak_start_pos + 1:]:
        line = line.rstrip()
        if ';' in line:
            # msp format has peak tuples separated by ';'
            peak_tuples = line.split(';')
            for pt in peak_tuples:
                tokens = pt.split()
                if len(tokens) > 0:
                    peaks.append([float(t) for t in tokens])
        else:
            tokens = line.split()
            if len(tokens) > 1:
                peaks.append([float(t) for t in tokens])
    # precursor_mz,peaks,metadata,original_file,spectrum_id
    precursor_mz = None
    original_file = file_name
    spectrum_id = None
    metadata = {}
    for line in block[:peak_start_pos]:
        if line.startswith('PRECURSORMZ') or line.startswith('PrecursorMZ'):
            precursor_mz = float(line.split(':')[1])
        elif line.startswith('NAME:') or line.startswith('Name:'):
            spectrum_id = line.split(':')[1]
        else:
            tokens = line.split(':')
            key = tokens[0]
            value = ':'.join(tokens[1:])
            metadata[key] = value

    assert precursor_mz is not None
    assert spectrum_id is not None

    new_record = SpectralRecord(precursor_mz, peaks, metadata, original_file,
                                spectrum_id)
    return new_record


def library_from_msp(msp_file_name):
    lines = []
    with open(msp_file_name, 'r') as f:
        for line in f:
            lines.append(line)
    # in_block = False
    block = []
    records = {}
    for line in lines:
        if line.startswith("NAME:") or line.startswith('Name:'):
            # new block
            if len(block) > 0:
                new_record = process_block(block, msp_file_name)
                records[new_record.spectrum_id] = new_record
            block = [line]
        else:
            block.append(line)
    new_record = process_block(block, msp_file_name)
    records[new_record.spectrum_id] = new_record

    sl = SpectralLibrary()
    sl.records = records
    sl.sorted_record_list = sl._dic2list()

    return sl


def make_queries_from_aligned_msdial(msdial_file_name, frag_file=True,
                                     sample_name=None):
    query_spectra = []
    msdial_df = pd.read_csv(msdial_file_name, sep='\t',
                            index_col='Alignment ID', header=4)
    if frag_file:
        msdial_df = msdial_df[msdial_df['MS/MS assigned'] == True]

    for i, row in msdial_df.iterrows():
        new_spectrum = aligned_row_to_spectral_record(row, sample_name=sample_name)
        query_spectra.append(new_spectrum)
    return query_spectra


def aligned_row_to_spectral_record(row, sample_name=None):
    peaks = extract_msdial_spectrum(row, 'MS/MS spectrum')
    precursor_mz = row['Average Mz']
    metadata = {
        'rt_in_minutes': row['Average Rt(min)'],
        'rt': row['Average Rt(min)'] * 60.0,
        'names': [row['Metabolite name']],
        'msms_assigned': row['MS/MS assigned'],
        'rt_matched': row['RT matched'],
        'mz_matched': row['m/z matched'],
        'msms_matched': row['MS/MS matched'],
        'total_score': row['Total score'],
        'dot_product': row['Dot product'],
        'reverse_dot_product': row['Reverse dot product'],
        'fragment_presence': row['Fragment presence %'],
        'msms_spectrum': row['MS/MS spectrum'],
    }
    if sample_name is not None:
        sample_cols = row.filter(regex=sample_name)
        metadata.update({
            'samples': sample_cols.to_dict(),
            'sample_count': np.count_nonzero(sample_cols.values)
        })

    original_file = row['Spectrum reference file name']
    spectrum_id = 'peak_%.6f' % precursor_mz
    new_spectrum = SpectralRecord(precursor_mz, peaks, metadata, original_file, spectrum_id)
    return new_spectrum


def extract_msdial_spectrum(row, key):
    peaks = []
    try:
        for info in row[key].split():
            mz, intensity = info.split(':')
            peak = np.array([float(mz), float(intensity)])
            peaks.append(peak)
    except AttributeError:  # no MS2 spectrum
        pass
    return peaks


def make_queries_from_msdial(msdial_file_name):
    original_file = os.path.splitext(msdial_file_name)[0]
    query_spectra = []
    msdial_df = pd.read_csv(msdial_file_name, sep='\t', index_col='PeakID')  #
    for i, row in msdial_df.iterrows():
        new_spectrum = row_to_spectral_record(row, original_file=original_file)
        query_spectra.append(new_spectrum)
    return query_spectra


def row_to_spectral_record(row, original_file=None):
    peaks = extract_msdial_spectrum(row, 'MSMS spectrum')
    precursor_mz = row['Precursor m/z'].values[0]
    metadata = {
        'rt_in_minutes': row['RT (min)'].values[0],
        'rt': row['RT (min)'].values[0] * 60.0,
        'names': [row['Title'].values[0]],
        'rt_matched': row['RT matched'].values[0],
        'mz_matched': row['m/z matched'].values[0],
        'msms_matched': row['MS/MS matched'].values[0],
        'total_score': row['Total score'].values[0],
        'dot_product': row['Dot product'].values[0],
        'reverse_dot_product': row['Reverse dot product'].values[0],
        'fragment_presence': row['Fragment presence %'].values[0],
        'msms_spectrum': row['MSMS spectrum'].values[0],
        'samples': {
            'height': row['Height'].values[0],
            'area': row['Area'].values[0]
        },
        'sample_count': 1,
        'box': msdial_row_to_box(precursor_mz,
                                 row["RT left(min)"].values[0],
                                 row["RT right (min)"].values[0])
    }
    spectrum_id = 'peak_%.6f' % precursor_mz
    new_spectrum = SpectralRecord(precursor_mz, peaks, metadata, original_file, spectrum_id)
    return new_spectrum


def msdial_row_to_box(precursor_mz, rt_left_in_minutes, rt_right_in_minutes, mz_tol=10):
    x1 = rt_left_in_minutes * 60.0
    x2 = rt_right_in_minutes * 60.0
    y1 = precursor_mz * (1 - mz_tol / 1e6)
    y2 = precursor_mz * (1 + mz_tol / 1e6)
    return GenericBox(x1, x2, y1, y2)


def make_queries_from_mzml(mzml_file_object):
    query_spectra = []
    for scan in mzml_file_object.scans:
        if not scan.ms_level == 2:
            continue
        precursor_mz = scan.precursor_mz
        peaks = scan.peaks
        new_spectrum = Spectrum(precursor_mz, peaks)
        query_spectra.append(new_spectrum)
    return query_spectra


def make_queries_from_chemicals(chemicals_file_name):
    chemicals = load_obj(chemicals_file_name)
    query_spectra = []
    for chem in chemicals:
        new_spectrum = chem_to_spectral_record(chem)
        query_spectra.append(new_spectrum)
    return query_spectra


def chem_to_spectral_record(chem):
    # FIXME: quick hack, assume that we observe M+H only
    precursor_mz = chem.isotopes[0][0] + PROTON_MASS
    peaks = []
    for child in chem.children:
        mz = child.isotopes[0][0]
        intensity = child.parent.max_intensity * child.prop_ms2_mass
        peak = np.array([mz, intensity])
        peaks.append(peak)
    metadata = { 'rt': chem.rt }
    original_file = None
    spectrum_id = 'peak_%.6f' % precursor_mz
    new_spectrum = SpectralRecord(precursor_mz, peaks, metadata, original_file, spectrum_id)
    return new_spectrum


def scan_to_spectral_record(scan):
    precursor_mz = scan.scan_params.get(ScanParameters.PRECURSOR_MZ)[0].precursor_mz
    peaks = list(zip(scan.mzs, scan.intensities))
    metadata = {}
    original_file = None
    spectrum_id = 'peak_%.6f' % precursor_mz
    new_spectrum = SpectralRecord(precursor_mz, peaks, metadata, original_file, spectrum_id)
    return new_spectrum


def main(mzml_file_name, msp_file_name, precursor_tolerance, hit_threshold):
    mzml_file_objects = {}
    if os.path.isfile(mzml_file_name):
        mzml_file_objects[mzml_file_name] = MZMLFile(mzml_file_name)
    elif os.path.isdir(mzml_file_name):
        mzml_files = glob.glob(os.path.join(mzml_file_name, '*.mzML'))
        for m in mzml_files:
            mzml_file_objects[m] = MZMLFile(m)
    else:
        logger.debug("No mzML files found")
        sys.exit(0)

    for m, mzml_file_object in mzml_file_objects.items():
        logger.debug(
            "Loaded {} scans from {}".format(len(mzml_file_object.scans), m))

    sl = library_from_msp(msp_file_name)
    logger.debug("Created library from {}".format(msp_file_name))

    hit_ids = set()
    for m, mzml_file_object in mzml_file_objects.items():
        query_spectra = make_queries_from_mzml(mzml_file_object)
        for q in query_spectra:
            hits = sl.spectral_match(q, ms1_tol=precursor_tolerance,
                                     score_thresh=hit_threshold)
            for hit in hits:
                hit_id = hit[0]
                hit_ids.add(hit_id)

    all_library_ids = set(sl.records.keys())
    n_library_ids = len(all_library_ids)
    n_hits = len(hit_ids)
    logger.debug("Out of {} IDs, {} got hits".format(n_library_ids, n_hits))
    # missing_ids = all_library_ids - hit_ids
    # print("Missing")
    # for i in missing_ids:
    #     print(i)
    return n_hits, n_library_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Limited dataset creation')
    parser.add_argument('mzml_file_name', type=str)
    parser.add_argument('msp_file_name', type=str)
    parser.add_argument('--precursor_tolerance', dest='precursor_tolerance',
                        default=1., type=float)
    parser.add_argument('--hit_threshold', dest='hit_threshold', default=0.7,
                        type=float)

    args = parser.parse_args()

    n_hits, out_of = main(str(args.mzml_file_name), str(args.msp_file_name),
                          args.precursor_tolerance,
                          args.hit_threshold)
