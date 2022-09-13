import os

import glob
import numpy as np

import pandas as pd
from tqdm.auto import tqdm

from vimms.Common import load_obj
from mass_spec_utils.library_matching.spec_libraries import SpectralLibrary
from mass_spec_utils.library_matching.spectrum import SpectralRecord


def compare_spectra(chem_library, base_folder, methods, matching_thresholds,
                    matching_method, matching_ms1_tol, matching_ms2_tol, matching_min_match_peaks):
    print('chem_library', chem_library)
    results = []
    for method in methods:

        # get msdial results
        results_folder = os.path.join(base_folder, method)
        spectra = load_obj(os.path.join(results_folder, 'matched_spectra.p'))

        # compare spectra
        for thresh in matching_thresholds:
            print(method, len(spectra), thresh)
            identified = []

            with tqdm(total=len(spectra)) as pbar:

                spec_identified = 0
                for i in range(len(spectra)):
                    spec = spectra[i]
                    hits = chem_library.spectral_match(spec, matching_method, matching_ms2_tol,
                                                       matching_min_match_peaks, matching_ms1_tol, thresh)
                    if len(hits) > 0:
                        spec_identified += 1
                        identified.extend([hit[0] for hit in hits])
                    pbar.update(1)
                pbar.close()

                hit_count = len(np.unique(np.array(identified)))
                num_chem = len(spectra)  # FIXME: is this correct??!!
                hit_prop = hit_count / num_chem

                hit_prop = float(spec_identified) / num_chem

                row = [method, num_chem, thresh, hit_count, hit_prop]
                results.append(row)

    df = pd.DataFrame(results, columns=[
                      'method', 'num_peaks', 'matching_threshold', 'hit_count', 'hit_prop'])
    return df


def get_chemical_names(msp_file):
    names = []
    with open(msp_file) as f:
        for line in f:
            if line.startswith('Name'):
                tokens = line.split(':')
                chemical_name = tokens[1].strip()
                names.append(chemical_name)
    return set(names)


def load_alignment_df(msdial_file):
    try:
        df = pd.read_csv(msdial_file, sep='\t', skiprows=4,
                         index_col='Alignment ID')
    except ValueError:
        df = pd.read_csv(msdial_file, sep='\t', skiprows=0, index_col='PeakID')
    return df


def get_hits(filename, chemical_names, threshold=70):
    
    # load MSDIAL alignment results
    basename = os.path.basename(filename)
    if '_replicates' in basename:
        new_basename = basename.replace('_replicates', '')
        filename = os.path.join(os.path.dirname(filename), new_basename)

    df = load_alignment_df(filename)

    # filter by 'Reverse dot product' above threshold
    # filtered = df[df['Reverse dot product'] > threshold]

    # filter by 'Total score' above threshold
    filtered = df[df['Total score'] > threshold]

    # get unique hits in the filtered results
    try:
        hits = set(filtered['Metabolite name'].values)
    except KeyError:
        hits = set(filtered['Title'].values)

    # get the intersection between the chemical names and the hits
    return df, hits.intersection(chemical_names)


def load_hits(base_folder, controller_name, sample_list, chemical_names, combine_single_hits=False):
    replicate = 0

    # load single hits
    hit_counts = []
    unique_hits = set()
    for sample in sample_list:
        msdial_output = '%s_%s_%d.msdial' % (
            controller_name, sample, replicate)
        single_file = os.path.join(base_folder, controller_name, msdial_output)
        _, hits = get_hits(single_file, chemical_names)
        # print(sample, len(hits))
        hit_counts.append(len(hits))
        unique_hits.update(hits)

    # single_hits = max(hit_counts)
    single_hits = hit_counts[0]

    # load multi hits
    if not combine_single_hits:
        aligned_files = glob.glob(os.path.join(
            base_folder, controller_name, 'AlignResult*.msdial'))
        aligned_file = aligned_files[0]
        _, ms2dec_hits = get_hits(aligned_file, chemical_names)
        multi_hits = len(ms2dec_hits)
    else:
        multi_hits = len(unique_hits)
    # print(controller_name, multi_hits)

    # print()
    return single_hits, multi_hits


def get_ss_ms_df(all_controllers, msp_file, base_folder, sample_list):
    results = []
    chemical_names = get_chemical_names(msp_file)

    single_hits = []
    multi_hits = []
    for controller_name in all_controllers:
        # combine_single_hits = True if controller_name == 'topN_exclusion' else False
        combine_single_hits = True
        ss, ms = load_hits(base_folder, controller_name, sample_list, chemical_names,
                           combine_single_hits=combine_single_hits)
        results.append((controller_name, ss, 'single sample'))
        results.append((controller_name, ms, 'multiple samples'))

    return pd.DataFrame(results, columns=['method', 'hit', 'sample availability'])


def spectral_distribution(chem_library, base_folder, methods, matching_threshold,
                          matching_method, matching_ms1_tol, matching_ms2_tol, matching_min_match_peaks):
    print('chem_library', chem_library)
    results = []
    for method in methods:

        # get msdial results
        results_folder = os.path.join(base_folder, method)
        spectra = load_obj(os.path.join(results_folder, 'matched_spectra.p'))

        # compare spectra
        with tqdm(total=len(spectra)) as pbar:
            for spec in spectra:
                # returns a list containing (spectrum_id, sc, c)
                hits = chem_library.spectral_match(spec, matching_method, matching_ms2_tol,
                                                   matching_min_match_peaks, matching_ms1_tol, matching_threshold)
                if len(hits) > 0:
                    max_item = max(hits, key=lambda item: item[1])  # 1 is sc
                    spectrum_id = max_item[0]
                    max_score = max_item[1]
                    if max_score > 0.0:
                        row = [method, spectrum_id, max_score]
                        results.append(row)
                pbar.update(1)
            pbar.close()

    df = pd.DataFrame(results, columns=['method', 'spectrum_id', 'score'])
    return df


def spec_records_to_library(spectra):
    # convert spectral records to spectral library for comparison
    chem_library = SpectralLibrary()
    chem_library.records = {spec.spectrum_id: spec for spec in spectra}
    chem_library.sorted_record_list = chem_library._dic2list()
    return chem_library


def pairwise_spectral_distribution(chem_library, base_folder, methods, matching_threshold,
                                   matching_method, matching_ms1_tol, matching_ms2_tol, matching_min_match_peaks):
    results = []
    methods = ['ground_truth'] + methods
    for method in methods:
        if method == 'ground_truth':

            # get spectra of chemicals for comparison
            spectra = list(chem_library.records.values())

        else:

            # get msdial results
            results_folder = os.path.join(base_folder, method)
            spectra = load_obj(os.path.join(
                results_folder, 'matched_spectra.p'))

            # chem_library is the same as spectra
            chem_library = spec_records_to_library(spectra)

        # compare spectra to library, which is the spectra themselves
        with tqdm(total=len(spectra)) as pbar:
            for spec in spectra:
                try:
                    # returns a list containing (spectrum_id, sc, c)
                    hits = chem_library.spectral_match(spec, matching_method, matching_ms2_tol,
                                                       matching_min_match_peaks, matching_ms1_tol,
                                                       matching_threshold)
                    if len(hits) == 1:
                        first = hits[0]
                        first_spectrum_id = first[0]
                        assert first_spectrum_id == spec.spectrum_id
                        max_score = 0.0
                    else:
                        max_score = 0.0
                        for hit in hits:
                            hit_spectrum_id = hit[0]
                            score = hit[1]
                            if hit_spectrum_id != spec.spectrum_id and score > max_score:
                                max_score = score
                    
                    if max_score > 0.0:
                        row = [method, spec.spectrum_id, max_score]
                        # print('spectrum_id', spec.spectrum_id, 'max_score', max_score, 'hits', hits)
                        # print()
                        results.append(row)
                except TypeError:
                    pass
                pbar.update(1)
            pbar.close()

    df = pd.DataFrame(results, columns=['method', 'spectrum_id', 'score'])
    return df
