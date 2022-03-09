"""
This file implements various methods to extract features from different sources.
"""
import os
import xml.etree.ElementTree
import zipfile

from loguru import logger

from vimms.Chemicals import DatabaseCompound, ChemicalMixtureFromMZML
from vimms.Common import DEFAULT_MZML_CHEMICAL_CREATOR_PARAMS, save_obj
from vimms.Roi import RoiBuilderParams


# flake8: noqa: C901
def extract_hmdb_metabolite(in_file, delete=True):
    """
    Extract chemicals from HMDB database

    Args:
        in_file: a zipped HMDB database downloaded from https://hmdb.ca/downloads.
        delete: whether to delete `in_file` once it has been processed

    Returns: a list of [vimms.Chemicals.DatabaseCompound][] objects.

    """
    logger.debug('Extracting HMDB metabolites from %s' % in_file)

    # if out_file is zipped then extract the xml file inside
    try:
        # extract from zip file
        zf = zipfile.ZipFile(in_file, 'r')
        metabolite_xml_file = zf.namelist()[
            0]  # assume there's only a single file inside the zip file
        f = zf.open(metabolite_xml_file)
    except zipfile.BadZipFile:  # oops not a zip file
        zf = None
        f = in_file

    # loops through file and extract the necessary element text to create a
    # DatabaseCompound
    db = xml.etree.ElementTree.parse(f).getroot()
    compounds = []
    prefix = '{http://www.hmdb.ca}'
    for metabolite_element in db:
        row = [None, None, None, None, None, None]
        for element in metabolite_element:
            if element.tag == (prefix + 'name'):
                row[0] = element.text
            elif element.tag == (prefix + 'chemical_formula'):
                row[1] = element.text
            elif element.tag == (prefix + 'monisotopic_molecular_weight'):
                row[2] = element.text
            elif element.tag == (prefix + 'smiles'):
                row[3] = element.text
            elif element.tag == (prefix + 'inchi'):
                row[4] = element.text
            elif element.tag == (prefix + 'inchikey'):
                row[5] = element.text

        # if all fields are present, then add them as a DatabaseCompound
        if None not in row:
            compound = DatabaseCompound(row[0], row[1], row[2], row[3], row[4],
                                        row[5])
            compounds.append(compound)
    logger.info(
        'Loaded %d DatabaseCompounds from %s' % (len(compounds), in_file))

    f.close()
    if zf is not None:
        zf.close()

    if delete:
        logger.info('Deleting %s' % in_file)
        os.remove(in_file)

    return compounds


def extract_roi(file_names, out_dir, pattern, mzml_path,
                param_dict=DEFAULT_MZML_CHEMICAL_CREATOR_PARAMS):
    """
    Extract ROI for all mzML files listed in file_names, and turn them
    into Chemical objects.

    Args:
        file_names: a list of mzML file names
        out_dir: output directory to store pickled chemicals. If None,
                 then the current directory is used
        pattern: pattern for output file
        mzml_path: input directory containing all the mzML files in file_names.
        param_dict: dictionary of parameters

    Returns: a list of extracted [vimms.Chemicals.Chemical][], one for each mzML file

    """
    # extract ROI for all mzML files in file_names
    datasets = []
    for i in range(len(file_names)):

        # if mzml_path is provided, use that as the front part of filename
        if mzml_path is not None:
            mzml_file = os.path.join(mzml_path, file_names[i])
        else:
            mzml_file = file_names[i]

        rp = RoiBuilderParams(**param_dict)
        cm = ChemicalMixtureFromMZML(mzml_file, roi_params=rp)
        dataset = cm.sample(None, 2)
        datasets.append(dataset)

        # save extracted chemicals
        if out_dir is None:
            # if no out_dir provided, then same in the same location
            # as the mzML file
            dataset_name = os.path.splitext(mzml_file)[0] + '.p'
            save_obj(dataset, dataset_name)
        else:
            # else save the chemicals in our_dir, using pattern as the filename
            basename = os.path.basename(file_names[i])
            out_name = pattern % int(basename.split('_')[2])
            save_obj(dataset, os.path.join(out_dir, out_name))

    return datasets
