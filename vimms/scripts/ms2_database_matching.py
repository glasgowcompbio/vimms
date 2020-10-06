# MS2 matchinig
import sys
import os
import glob
import argparse
import csv
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, '/Users/simon/git/mass-spec-utils')

from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.library_matching.spectrum import Spectrum
from mass_spec_utils.library_matching.spec_libraries import MassBankLibrary, GNPSLibrary

sys.path.append('..')
sys.path.append('../..') # if running in this folder


from vimms.Common import load_obj, save_obj

SUPPORTED_LIBRARIES = set(['gnps', 'massbank'])


def load_scans_from_mzml(mzml_file_name):
    logger.debug("Loading scans from {}".format(mzml_file_name))
    mm = MZMLFile(mzml_file_name)
    ms2_scans = list(filter(lambda x: x.ms_level == 2, mm.scans))
    logger.debug("Loaded {} mzML scans".format(len(ms2_scans)))
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
    parser.add_argument('libraries', type=str, nargs='+')
    parser.add_argument('--library_cache', dest='library_cache', type=str, default=None)
    parser.add_argument('--massbank_path', dest='massbank_path', type=str, default=None)
    parser.add_argument('--gnps_mgf_file', dest='gnps_mgf_file', type=str, default=None)
    parser.add_argument('--score_thresh', dest='score_thresh', type=float, default=0.7)
    parser.add_argument('--output_csv_file', dest='output_csv_file', type=str, default='hits.csv')
    

    args = parser.parse_args()
    root, ext = os.path.splitext(args.input_file_name)
    if ext.lower() == '.mzml':
        # load the ms2 scans from the .mzML
        query_spectra = load_scans_from_mzml(args.input_file_name)

    elif ext.lower() == '.mgf':
        # load the ms2 scans from the .mgf
        logger.warning("MGF input not yet implemented")
        sys.exit(0)

    else:
        logger.warning("Unknown input file format -- should be .mzML or .mgf")
        sys.exit(0)
    
    libraries = args.libraries
    for library in libraries:
        if not library in SUPPORTED_LIBRARIES:
            logger.warning("Unsupported library: {}".format(library))
            sys.exit(0)
    
    spec_libraries = {}
    for library in libraries:
        logger.warning(library)
        if args.library_cache is not None:
            # attempt to load library
            lib_file = os.path.join(args.library_cache,library+'.p')
            if os.path.isfile(lib_file):
                spec_libraries[library] = load_obj(lib_file)
                logger.warning("Loaded {}".format(lib_file))
            else:
                # create from a local massbank repo
                if library == 'massbank':
                    if args.massbank_path is None:
                        logger.warning("No local massbank instance, skipping")
                    else:
                        spec_libraries[library] = MassBankLibrary(mb_dir=args.massbank_path)
                        # save it, if a cache is there
                        save_obj(spec_libraries[library], lib_file)
                elif library == 'gnps':
                    if args.gnps_mgf_file is None:
                        logger.warning("No local gnps file, skipping")
                    else:
                        spec_libraries[library] = GNPSLibrary(args.gnps_mgf_file)
                        save_obj(spec_libraries[library], lib_file)
        else:
            if library == 'massbank':
                if args.massbank_path is None:
                    logger.warning("No local massbank instance, skipping")
                else:
                    spec_libraries[library] = MassBankLibrary(mb_dir=args.massbank_path)
            if library == 'gnps':
                if args.gnps_mgf_file is None:
                    logger.warning("No local GNPS instance, skipping")
                else:
                    spec_libraries[library] = GNPSLibrary(args.gnps_mgf_file)
    all_hits = []
    for spec_id in tqdm(query_spectra.keys()):
        for library in spec_libraries:
            hits = spec_libraries[library].spectral_match(query_spectra[spec_id], score_thresh=args.score_thresh)
            for hit in hits:
                new_hit = [spec_id, library, hit[0], hit[1], hit[2].metadata['inchikey']]
                all_hits.append(new_hit)
        
    logger.debug('Writing output to {}'.format(args.output_csv_file))   
    with open(args.output_csv_file,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['spec_id', 'library', 'hit_id', 'score', 'inchikey'])
        for hit in all_hits:
            writer.writerow(hit)

    # summary
    s, _, t, sc, ik = zip(*all_hits)
    logger.warning("{} unique spectra got hits".format(len(set(s))))
    logger.warning("{} unique structures were hit".format(len(set([a.split('-')[0] for a in ik if a is not None]))))