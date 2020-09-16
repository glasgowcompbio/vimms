# check ms2 matches
# simple script that loads an msp and an mzml and sees how many of the spectra in the MSP file
# can be matched to a spectrum in an ms2 scan in the .mzml
import sys
import argparse
from loguru import logger
from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.library_matching.spec_libraries import SpectralLibrary
from mass_spec_utils.library_matching.spectrum import Spectrum, SpectralRecord

sys.path.append('..')
sys.path.append('../..') # if running in this folder

def process_block(block, file_name):
    peak_start_pos = [b.startswith("Num") for b in block].index(True)
    peaks = []
    for line in block[peak_start_pos+1:]:
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

    new_record = SpectralRecord(precursor_mz, peaks, metadata, original_file, spectrum_id)
    return new_record


def library_from_msp(msp_file_name):
    lines = []
    with open(msp_file_name, 'r') as f:
        for line in f:
            lines.append(line)
    in_block = False
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Limited dataset creation')
    parser.add_argument('mzml_file_name', type=str)
    parser.add_argument('msp_file_name', type=str)
    parser.add_argument('--precursor_tolerance', dest='precursor_tolerance', default=1., type=float)
    parser.add_argument('--hit_threshold', dest='hit_threshold', default=0.7, type=float)

    args = parser.parse_args()

    mzml_file_object = MZMLFile(args.mzml_file_name)
    logger.debug("Loaded {} scans from {}".format(len(mzml_file_object.scans), args.mzml_file_name))
    sl = library_from_msp(args.msp_file_name)

    logger.debug("Created library from {}".format(args.msp_file_name))

    query_spectra = make_queries_from_mzml(mzml_file_object)

    hit_ids = set()
    for q in query_spectra:
        hits = sl.spectral_match(q, ms1_tol=args.precursor_tolerance, score_thresh=args.hit_threshold)
        for hit in hits:
            hit_id = hit[0]
            hit_ids.add(hit_id)
    
    all_library_ids = set(sl.records.keys())
    n_library_ids = len(all_library_ids)
    n_hits = len(hit_ids)
    print("Out of {} IDs, {} got hits".format(n_library_ids,n_hits))
    missing_ids = all_library_ids - hit_ids
    # print("Missing")
    # for i in missing_ids:
    #     print(i)

