# gnps.py
from .spectrum import SpectralRecord
from loguru import logger
def load_gnps_files(file_list):
    if type(file_list) == str:
        file_list = [file_list]
    spectra = {}
    for file_name in file_list:
        spectra = load_mgf(file_name,id_field = 'SPECTRUMID',spectra = spectra)
    return spectra


def load_mgf(mgf_name,id_field = 'SCANS',spectra = None):
    if spectra is None:
        spectra = {}
    with open(mgf_name,'r') as f:
        current_metadata = {'filename':mgf_name}
        current_peaks = []
        got_record = False
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                continue
            if line.startswith('BEGIN IONS'):
                if len(current_metadata) > 1:
                    if len(current_peaks) > 0:
                        try:
                            current_metadata['names'] = [current_metadata['COMPOUNDNAME']]
                        except:
                            pass
                        if id_field == 'SCANS':
                            id_val = int(current_metadata[id_field])
                        else:
                            id_val = current_metadata[id_field]
                        spectrum = SpectralRecord(float(current_metadata['PEPMASS']),current_peaks,current_metadata,mgf_name,id_val)
                        spectra[id_val] = spectrum
                        if len(spectra)%100 == 0:
                            logger.debug("Loaded {} spectra".format(len(spectra)))
                current_metadata = {'filename':mgf_name}
                current_peaks = []
            elif len(line.split('=')) > 1:
                # it is a metadata line
                tokens = line.split('=')
                current_metadata[tokens[0]] = "=".join(tokens[1:])
            elif not line.startswith('END IONS'):
                # it's a peak
                tokens = line.split()
                mz = float(tokens[0])
                intensity = float(tokens[1])
                current_peaks.append((mz,intensity))
    # save the last one
    if len(current_peaks) > 0:
        try:
            current_metadata['names'] = [current_metadata['COMPOUNDNAME']]
        except:
            pass
        if id_field == 'SCANS':
            id_val = int(current_metadata[id_field])
        else:
            id_val = current_metadata[id_field]
        spectrum = SpectralRecord(float(current_metadata['PEPMASS']),current_peaks,current_metadata,mgf_name,id_val)
        spectra[id_val] = spectrum
    return spectra
