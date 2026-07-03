# hmdb.py
from lxml import etree
import csv
import glob
import os
from ..adduct_calculator.adduct_rules import AdductTransformer
from molmass import Formula
from .spectrum import SpectralRecord

# following method parses one of the hmdb metabolite (i.e. non msms) files
# to produce a .csv file that maps HMDB accessions to chemical formulas
# this is used when we parse msms files to assign a chemical formula, and therefore
# a precursor mz to the spectra
# Only need to run this once to produce the .csv...it takes a while
def parse_metabolites_file(xml_file_name = "/Users/simon/hmdb/hmdb_metabolites.xml",output_csv = '/Users/simon/hmdb/hmdb_metabolites.csv'):
    tree = etree.parse(xml_file_name)
    root = tree.getroot()
    acc_form = {}
    n_done = 0
    for element in root:
        n_done += 1
        try:
            accession = element.find('{http://www.hmdb.ca}accession').text
            formula = element.find('{http://www.hmdb.ca}chemical_formula').text
            acc_form[accession] = formula
        except:
            print("FAIL")
        if n_done % 100 == 0:
            print(n_done)
    with open(output_csv,'w') as f:
        writer = csv.writer(f)
        for k,v in acc_form.items():
            writer.writerow([k,v])
    return acc_form

# parses an individual msms file
# note that some have no msms peaks
# returns the spectrum ID, the HMDB ID
# the peaks, and the ionisation mode
def parse_msms_file(xml_file_name):
    tree = etree.parse(xml_file_name)
    root = tree.getroot()
    db_id = root.find('database-id').text
    mode = root.find('ionization-mode').text
    spec_id = root.find('id').text
    msms_peaks = root.find('ms-ms-peaks')
    peaks = []
    if not msms_peaks is None:
        for peak in msms_peaks:
            mz = float(peak.find('mass-charge').text)
            intensity = float(peak.find('intensity').text)
            peaks.append((mz,intensity))
    return spec_id,db_id,mode,peaks

# helper method to load the accession to formula csv
def load_csv(csv_file):
    output_dict = {}
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        for line in reader:
            output_dict[line[0]] = line[1]
    return output_dict

# load a folder of records 
# pass it the folder path
# the path to the csv mentioned above
# the target mode  ('positive','negative','both')
# and the transformation(s) you want to apply to make
# the precursor_mz values 
def load_hmdb_msms_records(folder,accession_to_formula_file,target_mode = 'positive',transformations = ['[M+H]+'],records = {}):
    accession_to_formula = load_csv(accession_to_formula_file)
    xml_files = glob.glob(os.path.join(folder,'*.xml'))
    at = AdductTransformer()
    n_loaded = 0
    n_total = len(xml_files)
    for xml_file in xml_files:
        spectrum_id,db_id,mode,peaks = parse_msms_file(xml_file)
        if len(peaks) == 0: # no peaks, ignore
            continue
        if target_mode == 'positive' and mode.lower() == 'negative': # sometimes capitalised!
            continue
        if target_mode == 'negative' and mode.lower() == 'positive':
            continue
        
        if not db_id in accession_to_formula:
            print("{} not in accession_to_formula, skipping".format(db_id))
            continue
        try:
            f = Formula(accession_to_formula[db_id])
            f_mass = f.isotope.mass
            for transformation in transformations:
                precursor_mz = at.mass2ion(f_mass,transformation)
                # make a spectral record with this as the precursor mz
                metadata = {'precursor_mz': precursor_mz,
                            'hmdb_id': db_id,
                            'mode': mode,
                            'adduct_type': transformation,
                            'spectrum_id': spectrum_id}
                ion_id = ':'.join([str(spectrum_id),db_id,transformation])
                new_record = SpectralRecord(precursor_mz,peaks,metadata,xml_file,ion_id)
                records[ion_id] = new_record
        except:
            print("Failed on fromula {}".format(accession_to_formula[db_id]))

        n_loaded += 1
        if n_loaded % 100 == 0:
            print("Loaded {} of {}".format(n_loaded,n_total))
        
    return records

