# MS2 matchinig
import argparse
import csv
import os
import sys

from loguru import logger
from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.library_matching.gnps import load_mgf
from mass_spec_utils.library_matching.spectrum import Spectrum
from tqdm.auto import tqdm

from vimms.Common import load_obj, set_log_level_warning, set_log_level_debug

sys.path.append('..')
sys.path.append('../..')  # if running in this folder


def load_scans_from_mzml(mzml_file_name):
    logger.debug("Loading scans from {}".format(mzml_file_name))
    mm = MZMLFile(mzml_file_name)
    ms2_scans = list(filter(lambda x: x.ms_level == 2, mm.scans))
    spectra = {}
    for s in ms2_scans:
        spec_id = s.scan_no
        precursor_mz = s.precursor_mz
        peaks = s.peaks
        new_spectrum = Spectrum(precursor_mz, peaks)
        spectra[spec_id] = new_spectrum
    return spectra


# flake8: noqa: C901
def main():
    global file_spectra
    parser = argparse.ArgumentParser(description='Limited dataset creation')
    parser.add_argument('input_file_names', type=str)
    parser.add_argument('library_cache', type=str)
    parser.add_argument('libraries', type=str, nargs='+')
    parser.add_argument('--score_thresh', dest='score_thresh', type=float,
                        default=0.7)
    parser.add_argument('--ms1_tol', dest='ms1_tol', type=float, default=1.)
    parser.add_argument('--ms2_tol', dest='ms2_tol', type=float, default=0.2)
    parser.add_argument('--min_matched_peaks', dest='min_matched_peaks',
                        type=int, default=1)
    parser.add_argument('--output_csv_file', dest='output_csv_file', type=str,
                        default='hits.csv')
    parser.add_argument('--log_level', dest='log_level', type=str,
                        default='warning')
    parser.add_argument('--mgf_id_field', dest='mgf_id_field', type=str,
                        default='SCANS')
    args = parser.parse_args()
    input_file_names = args.input_file_names
    if ',' in input_file_names:  # multiple items
        input_file_names = input_file_names.split(',')
    else:  # single item
        input_file_names = [input_file_names]
    assert len(input_file_names) > 0
    # assume all the files have the same extension as the first one
    first = input_file_names[0]
    root, ext = os.path.splitext(first)
    if ext.lower() == '.mzml':
        query_spectra = {}
        for input_file_name in input_file_names:
            # load the ms2 scans from the .mzML
            file_spectra = load_scans_from_mzml(input_file_name)
            logger.warning(
                "Loaded {} MS2 spectra from {}".format(len(file_spectra),
                                                       input_file_name))
            query_spectra[input_file_name] = file_spectra

    elif ext.lower() == '.mgf':
        query_spectra = {}
        for input_file_name in input_file_names:
            # load the ms2 scans from the .mgf
            file_spectra = load_mgf(input_file_name,
                                    id_field=args.mgf_id_field,
                                    spectra={})
            logger.warning(
                "Loaded {} MS2 spectra from {}".format(len(file_spectra),
                                                       input_file_name))
            query_spectra[input_file_name] = file_spectra
    else:
        logger.warning("Unknown input file format -- should be .mzML or .mgf")
        sys.exit(0)
    if args.log_level == 'warning':
        set_log_level_warning()
    elif args.log_level == 'debug':
        set_log_level_debug()
    libraries = args.libraries
    spec_libraries = {}
    if args.library_cache is not None:
        for library in libraries:
            # attempt to load library
            lib_file = os.path.join(args.library_cache, library + '.p')
            if os.path.isfile(lib_file):
                logger.warning("Loading {}".format(lib_file))
                spec_libraries[library] = load_obj(lib_file)
                logger.warning("Loaded {}".format(lib_file))
            else:
                logger.warning("Could not find {}".format(lib_file))
                sys.exit(0)
    else:
        logger.warning("You must supply a library folder")
        sys.exit(0)
    all_hits = []
    for input_file_name in query_spectra.keys():
        file_spectra = query_spectra[input_file_name]
        logger.warning('Processing {}'.format(input_file_name))
        for spec_id in tqdm(file_spectra.keys()):
            for library in spec_libraries:
                hits = spec_libraries[library].spectral_match(
                    file_spectra[spec_id], score_thresh=args.score_thresh,
                    ms2_tol=args.ms2_tol, ms1_tol=args.ms1_tol,
                    min_match_peaks=args.min_matched_peaks)
                for hit in hits:
                    new_hit = [spec_id, library, hit[0], hit[1],
                               hit[2].metadata['inchikey']]
                    all_hits.append(new_hit)
    if len(all_hits) == 0:
        logger.warning("No hits found!")
    else:
        logger.warning('Writing output to {}'.format(args.output_csv_file))
        with open(args.output_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['spec_id', 'library', 'hit_id', 'score', 'inchikey'])
            for hit in all_hits:
                writer.writerow(hit)

        # summary
        s, _, t, sc, ik = zip(*all_hits)
        logger.warning("{} unique spectra got hits".format(len(set(s))))
        logger.warning("{} unique structures were hit".format(
            len(set([a.split('-')[0] for a in ik if a is not None]))))


if __name__ == '__main__':
    main()
