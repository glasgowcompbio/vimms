import sys
sys.path.append('..')
sys.path.append('../..')  # if running in this folder

import os
import shutil

from loguru import logger

from vimms.Common import download_file, extract_zip_file, save_obj
from vimms.FeatureExtraction import extract_hmdb_metabolite
from vimms.Chemicals import ChemicalMixtureFromMZML
from vimms.Roi import RoiBuilderParams


def download_example_data():
    url = 'https://github.com/glasgowcompbio/vimms-data/raw/main/example_data.zip'
    return os.path.abspath(download_file(url))


def parse_example_data(example_zip_file):
    extract_zip_file(example_zip_file, delete=True)

    # extract beer chemicals from mzML file
    logger.info('Extracting chemicals')
    mzml_file = os.path.join('example_data', 'Beer_multibeers_1_T10_POS.mzML')
    rp = RoiBuilderParams(at_least_one_point_above=1.75E5, min_roi_length=3)
    cm = ChemicalMixtureFromMZML(mzml_file, roi_params=rp)
    dataset = cm.sample(None, 2)

    out_file = os.path.abspath('../../tests/fixtures/beer_compounds.p')
    save_obj(dataset, out_file)
    shutil.rmtree('example_data')

def download_hmdb():
    # download the entire HMDB metabolite database and extract chemicals from it
    url = 'http://www.hmdb.ca/system/downloads/current/hmdb_metabolites.zip'
    return os.path.abspath(download_file(url))


def parse_hmdb(hmdb_file):
    compounds = extract_hmdb_metabolite(hmdb_file, delete=True)
    out_file = os.path.abspath('../../tests/fixtures/hmdb_compounds.p')
    save_obj(compounds, out_file)
    os.remove('hmdb_metabolites.zip')


def main():
    file = download_example_data()
    parse_example_data(file)

    file = download_hmdb()
    parse_hmdb(file)


if __name__ == '__main__':
    main()
