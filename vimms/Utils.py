# Utils.py - some general utilities
import os
from pathlib import Path

import pysmiles
from mass_spec_utils.library_matching.gnps import load_mgf

from vimms.Chemicals import KnownChemical, UnknownChemical, DatabaseCompound
from vimms.Common import adduct_transformation, create_if_not_exist, \
    ATOM_MASSES, POSITIVE

# Constants for write_msp
COLLISION_ENERGY = '25'
IONIZATION = 'Positive'
SKIP_RT = False


def packline(outln, packme):
    outln.append(packme + '\n')
    return outln


def decimal_to_string(fnum, no_dec=0):
    """
    Convert a decimal to a string with no_dec decimal places
    """
    res = ''
    if no_dec == 0:
        res = str(int(fnum))
    else:
        res = str(round(fnum, no_dec))
    return res


# flake8: noqa: C901
def write_msp(chemical_list, msp_filename, out_dir=None, skip_rt=False,
              all_isotopes=False, ion_mode=[POSITIVE]):
    """
    Turn a chemical list into an msp file
    """

    # buffer for msp lines
    outln = []
    outln.clear
    pos = 0

    for chem in chemical_list:
        if all_isotopes:
            use_isotopes = range(len(chem.isotopes))
        else:
            use_isotopes = [0]
        for which_isotope in use_isotopes:
            for ionisation_mode in ion_mode:
                for which_adduct in range(len(chem.adducts[ionisation_mode])):
                    if isinstance(chem, KnownChemical):
                        name = 'NAME: ' + 'KnowChemical' \
                               + '_' \
                               + chem.formula.formula_string + '_iso' \
                               + str(which_isotope) + '_num' + str(pos)
                    elif isinstance(chem, UnknownChemical):
                        name = 'NAME: ' + 'UnKnowChemical' + '_' + str(
                            chem.mass) + '_iso' + str(
                            which_isotope) + '_num' + str(pos)
                    else:
                        raise NotImplementedError()
                    outln = packline(outln, name)
                    mz = adduct_transformation(chem.isotopes[which_isotope][0],
                                               chem.adducts[ionisation_mode][
                                                   which_adduct][0])
                    outln = packline(
                        outln, 'PRECURSORMZ: ' + decimal_to_string(mz, 2))
                    outln = packline(outln,
                                     'PRECURSORTYPE: ' + '[' +
                                     chem.adducts[ionisation_mode][
                                         which_adduct][0] + ']+')
                    if isinstance(chem, KnownChemical):
                        outln = packline(
                            outln, 'FORMULA: ' + chem.formula.formula_string)
                    if not skip_rt:
                        rt = chem.rt + chem.chromatogram.get_apex_rt()
                        outln = packline(outln,
                                         'RETENTIONTIME: ' + decimal_to_string(
                                             rt / 60, 2))  # in minutes
                    outln = packline(outln, 'INTENSITY: ' + decimal_to_string(
                        chem.isotopes[which_isotope][1] *
                        chem.adducts[ionisation_mode][which_adduct][
                            1] * chem.max_intensity))
                    outln = packline(outln, 'IONMODE: ' + IONIZATION)
                    outln = packline(outln,
                                     'COLLISIONENERGY: ' + COLLISION_ENERGY)
                    outln = packline(outln,
                                     'Num Peaks: ' + str(len(chem.children)))
                    for msn in chem.children:
                        msn_mz = adduct_transformation(msn.isotopes[0][0],
                                                       chem.adducts[
                                                           ionisation_mode][
                                                           which_adduct][0])
                        msn_peak = chem.isotopes[which_isotope][1] * \
                                   chem.adducts[ionisation_mode][which_adduct][
                                       1] * chem.max_intensity * \
                                   1 * msn.prop_ms2_mass
                        if decimal_to_string(msn_peak) != '0':
                            temp = decimal_to_string(msn_mz, 5) + ' ' + \
                                   decimal_to_string(msn_peak)
                            outln = packline(outln, temp)
                    outln = packline(outln, '')
                    pos += 1

    if out_dir is not None:
        msp_filename = Path(out_dir, msp_filename)

    out_dir = os.path.dirname(msp_filename)
    create_if_not_exist(out_dir)

    f = open(msp_filename, "w")
    f.writelines(outln)
    f.close()


def smiles_to_formula(smiles_string):
    mol = pysmiles.read_smiles(smiles_string, explicit_hydrogen=True)
    atom_counts = {g: 0 for g in ATOM_MASSES}
    for node in mol.nodes(data="element"):
        atom = node[1]
        if atom not in atom_counts:
            return None
        else:
            atom_counts[atom] += 1
    chem_formula = ""
    for atom, count in atom_counts.items():
        if count == 0:
            continue
        elif count == 1:
            chem_formula += atom
        else:
            chem_formula += "{}{}".format(atom, count)
    return chem_formula


def mgf_to_database(mgf_file, id_field='SPECTRUMID'):
    """
    Load spectra from an mgf file and save as a list of DatabaseCompounds
    Computes chemimcal formula from SMILES
    """
    records = load_mgf(mgf_file, id_field=id_field)
    database = []
    for key in records:
        chemical_formula = smiles_to_formula(records[key].metadata['SMILES'])
        records[key].metadata['CHEMICAL_FORMULA'] = chemical_formula
    for key, record in records.items():
        database.append(
            DatabaseCompound(record.spectrum_id,
                             record.metadata['CHEMICAL_FORMULA'], None, None,
                             None, key))
    return database
