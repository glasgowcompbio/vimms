# MS2 matchinig
import sys
import os
import glob
import argparse
import csv
from loguru import logger
from tqdm import tqdm

from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.library_matching.spectrum import Spectrum
from mass_spec_utils.library_matching.spec_libraries import MassBankLibrary, GNPSLibrary
from mass_spec_utils.library_matching.gnps import load_mgf

sys.path.append('..')
sys.path.append('../..') # if running in this folder


from vimms.Common import load_obj, save_obj, set_log_level_warning, set_log_level_debug

SUPPORTED_LIBRARIES = set(['gnps', 'massbank'])


def load_scans_from_mzml(mzml_file_name):
    logger.debug("Loading scans from {}".format(mzml_file_name))
    mm = MZMLFile(mzml_file_name)
    ms2_scans = list(filter(lambda x: x.ms_level == 2, mm.scans))
    logger.warning("Loaded {} mzML scans".format(len(ms2_scans)))
    spectra = {}
    for s in ms2_scans:
        spec_id = s.scan_no
        precursor_mz = s.precursor_mz
        peaks = s.peaks
        new_spectrum = Spectrum(precursor_mz, peaks)
        spectra[spec_id] = new_spectrum
    return spectra

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Limited dataset creation')
    parser.add_argument('input_file_name', type=str)
    parser.add_argument('library_cache', type=str)
    parser.add_argument('libraries', type=str, nargs='+')
    parser.add_argument('--score_thresh', dest='score_thresh', type=float, default=0.7)
    parser.add_argument('--ms1_tol', dest='ms1_tol', type=float, default=1.)
    parser.add_argument('--ms2_tol', dest='ms2_tol', type=float, default=0.2)
    parser.add_argument('--min_matched_peaks', dest='min_matched_peaks', type=int, default=1)
    parser.add_argument('--output_csv_file', dest='output_csv_file', type=str, default='hits.csv')
    parser.add_argument('--log_level', dest='log_level', type=str, default='warning')
    parser.add_argument('--mgf_id_field', dest='mgf_id_field', type=str, default='SCANS')

    

    args = parser.parse_args()
    root, ext = os.path.splitext(args.input_file_name)
    if ext.lower() == '.mzml':
        # load the ms2 scans from the .mzML
        query_spectra = load_scans_from_mzml(args.input_file_name)
    elif ext.lower() == '.mgf':
        # load the ms2 scans from the .mgf
        query_spectra = load_mgf(args.input_file_name, id_field=args.mgf_id_field, spectra={})
        logger.warning("Loaded {} spectra".format(len(query_spectra)))
    else:
        logger.warning("Unknown input file format -- should be .mzML or .mgf")
        sys.exit(0)
    
    if args.log_level == 'warning':
        set_log_level_warning()
    elif args.log_level == 'debug':
        set_log_level_debug()

    libraries = args.libraries
    for library in libraries:
        if not library in SUPPORTED_LIBRARIES:
            logger.warning("Unsupported library: {}".format(library))
            sys.exit(0)
    
    spec_libraries = {}
    if args.library_cache is not None:
        for library in libraries:
            # attempt to load library
            lib_file = os.path.join(args.library_cache,library+'.p')
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
    for spec_id in tqdm(query_spectra.keys()):
        for library in spec_libraries:
            hits = spec_libraries[library].spectral_match(query_spectra[spec_id], score_thresh=args.score_thresh, ms2_tol=args.ms2_tol, ms1_tol=args.ms1_tol, min_match_peaks=args.min_matched_peaks)
            for hit in hits:
                new_hit = [spec_id, library, hit[0], hit[1], hit[2].metadata['inchikey']]
                all_hits.append(new_hit)
        
    logger.warning('Writing output to {}'.format(args.output_csv_file))   
    with open(args.output_csv_file,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['spec_id', 'library', 'hit_id', 'score', 'inchikey'])
        for hit in all_hits:
            writer.writerow(hit)

    # summary
    s, _, t, sc, ik = zip(*all_hits)
    logger.warning("{} unique spectra got hits".format(len(set(s))))
    logger.warning("{} unique structures were hit".format(len(set([a.split('-')[0] for a in ik if a is not None]))))