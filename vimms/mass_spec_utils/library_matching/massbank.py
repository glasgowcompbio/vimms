# massbank.py
# methods and classes w.r.t massbank
import glob
import os
import sys

from .spectrum import SpectralRecord

def parse_mb_file(file_name):
    with open(file_name,'r', encoding="utf8", errors="ignore") as f:
        in_peaks = False
        peaks = []
        metadata = {}
        names = []
        precursor_mz = None
        precursor_type = None
        smiles = None
        iupac = None
        inchikey = None
        polarity = None
        for line in f:
            if len(line) == 0:
                continue
            line = line.rstrip()
            if line.startswith('PK$PEAK'):
                in_peaks = True
                continue
            if line.startswith('MS$FOCUSED_ION: PRECURSOR_M/Z'):
                precursor_mz_string = line.split()[-1]
                try:
                    precursor_mz = float(precursor_mz_string)
                except:
                    if '/' in precursor_mz_string:
                        precursor_mz = float(precursor_mz_string.split('/')[0])
                    else:
                        precursor_mz = None
                continue
            if line.startswith('MS$FOCUSED_ION: PRECURSOR_TYPE'):
                precursor_type_string = line.split()[-1]
                if '/' in precursor_type_string:
                    precursor_type = precursor_type_string.split('/')[0]
                else:
                    precursor_type = precursor_type_string

                continue
            if line.startswith('AC$MASS_SPECTROMETRY: ION_MODE'):
                polarity = line.split()[-1]
                continue
            if line.startswith('CH$NAME'):
                names.append(' '.join(line.split(':')[1:]))
                continue
            if line.startswith('CH$SMILES'):
                smiles = line.split()[-1]
                continue
            if line.startswith('CH$IUPAC'):
                iupac = line.split()[-1]
                continue
            if line.startswith('CH$LINK: INCHIKEY'):
                inchikey = line.split()[-1]
            if in_peaks:
                if line.startswith('//'):
                    in_peaks = False
                else:
                    mz,inte,rel_inte = line.split()
                    peaks.append((float(mz),float(inte)))
        metadata['precursor_mz'] = precursor_mz
        metadata['precursor_type'] = precursor_type
        metadata['smiles'] = smiles
        metadata['iupac'] = iupac
        metadata['names'] = names
        metadata['inchikey'] = inchikey
        metadata['polarity'] = polarity
        spectrum_id = file_name.split(os.sep)[-1].split('.')[0]
        record = SpectralRecord(precursor_mz,peaks,metadata,file_name,spectrum_id)

        return record
        

def load_folder(folder_name,records = {},verbose = True):
    file_list = glob.glob(os.path.join(folder_name,'*.txt'))
    for file_name in file_list:
        spec_id = file_name.split(os.sep)[-1].split('.')[0]
        if verbose:
            print("Loading ",spec_id)
        assert not spec_id in records
        records[spec_id] = parse_mb_file(file_name)
        records[spec_id].metadata['spec_id'] = spec_id


def load_massbank(mb_dir = '/Users/simon/git/MassBank-data/'):
    folders = glob.glob(mb_dir + '*/')
    records = {}
    for folder in folders:
        n_records = len(records)
        print("Loading records from ",folder)
        load_folder(folder,records = records,verbose = False)
        print("\t Loaded {} new records".format(len(records)-n_records))
    return records
