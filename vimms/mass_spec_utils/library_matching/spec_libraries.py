# mb_parser
import os
import glob
import math
import random
from .massbank import load_massbank
from .gnps import load_gnps_files
from .hmdb import load_hmdb_msms_records
from .spectral_scoring_functions import cosine_similarity,modified_cosine_similarity


class SpectralLibrary(object):
    def __init__(self):
        self.records = {}
        self.sorted_record_list = []

    def _candidates(self,query_mz,ms1_tol):
        from sortedcontainers import SortedList
        pmz_list = SortedList([m.precursor_mz for m in self.sorted_record_list])
        lower = query_mz - ms1_tol
        upper = query_mz + ms1_tol
        start = pmz_list.bisect(lower)
        end = pmz_list.bisect(upper)
        return self.sorted_record_list[start:end]
    
    def _dic2list(self):
        # - makes a list of the values in records
        # - removes any without a precursor mz
        # - sorts
        record_list = list(self.records.values())
        filtered_list = list(filter(lambda x: not x.precursor_mz is None,record_list))
        return sorted(filtered_list)

    def spectral_match(self,query,
            scoring_function = 'cosine',
            ms2_tol = 0.2,
            min_match_peaks = 1,
            ms1_tol = 0.2,
            score_thresh = 0.7):
        
        candidates = self._candidates(query.precursor_mz,ms1_tol)
        hits = []
        for c in candidates:
            if scoring_function == 'cosine':
                sc,_ = cosine_similarity(query,c,ms2_tol,min_match_peaks)
            elif scoring_function == 'modified cosine':
                sc,_ = modified_cosine_similarity(query,c,ms2_tol,min_match_peaks)
            else:
                print("Unrecognised scoring function: ",scoring_function)
                return None
            if sc >= score_thresh:
                hits.append((c.spectrum_id,sc,c))
        return hits


class GNPSLibrary(SpectralLibrary):
    def __init__(self,input_mgf_files):
        super().__init__()
        self.records = load_gnps_files(input_mgf_files)
        self.sorted_record_list = self._dic2list()

class MassBankLibrary(SpectralLibrary):
    def __init__(self,mb_dir = '/Users/simon/git/MassBank-data/', polarity='all'):
        super().__init__()
        self.records = load_massbank(mb_dir = mb_dir)
        if not polarity == 'all':
            new_records = {}
            for sid, spec in self.records.items():
                if spec.metadata['polarity'] == polarity:
                    new_records[sid] = spec
            self.records = new_records
        self.sorted_record_list = self._dic2list() # sorted by precursor mz

class HMDBLibrary(SpectralLibrary):
    def __init__(self,csv_file = '/Users/simon/hmdb/hmdb_metabolites.csv',
                 msms_file_folder = '/Users/simon/hmdb/hmdb_experimental_msms_spectra',
                 mode = 'positive',
                 transformations = ['[M+H]+']):
        super().__init__()
        self.records = load_hmdb_msms_records(msms_file_folder,csv_file,target_mode = 'positive',transformations = ['[M+H]+'])
        self.sorted_record_list = self._dic2list() # sorted by precursor mz


if __name__ == '__main__':

    # # example of massbank library
    # mbl = MassBankLibrary()

    # record_list = list(mbl.records.values())
    # a = random.randrange(len(record_list))
    # example_spectrum = record_list[a]

    # hits = mbl.spectral_match(example_spectrum,score_thresh = 0.5)
    # print(example_spectrum)
    # for hit in hits:
    #     print(hit)

    # # example of HMDB
    # hmdbl = HMDBLibrary()
    
    # hits = hmdbl.spectral_match(example_spectrum,score_thresh = 0.5)
    # for hit in hits:
    #     print(hit)

    # example og GNPS
    gl = GNPSLibrary(['/Users/simon/git/molnet/lib/matched_mibig_gnps_update.mgf'])
    record_list = list(gl.records.values())
    a = random.randrange(len(record_list))
    example_spectrum = record_list[a]
    hits = gl.spectral_match(example_spectrum,score_thresh = 0.5)
    for hit in hits:
        print(hit)
