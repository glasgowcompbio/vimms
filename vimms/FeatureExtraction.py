import os

from vimms.Common import DEFAULT_MZML_CHEMICAL_CREATOR_PARAMS, save_obj
from vimms.Roi import make_roi, RoiToChemicalCreator


def extract_roi(file_names, out_dir, pattern, mzml_path, ps, param_dict=DEFAULT_MZML_CHEMICAL_CREATOR_PARAMS):
    """
    Extract ROI for all mzML files listed in file_names, and turn them into Chemical objecs
    :param file_names: a list of mzML file names
    :param out_dir: output directory to store pickled chemicals. If None, then the current directory is used
    :param pattern: pattern for output file
    :param mzml_path: input directory containing all the mzML files in file_names.
    :param ps: a peak sampler object
    :param param_dict: dictionary of parameters
    :return: a list of extracted Chemicals, one for each mzML file
    """
    # extract ROI for all mzML files in file_names
    datasets = []
    for i in range(len(file_names)):

        # if mzml_path is provided, use that as the front part of filename
        if mzml_path is not None:
            mzml_file = os.path.join(mzml_path, file_names[i])
        else:
            mzml_file = file_names[i]

        # actually extracts the ROI here
        good_roi, junk = make_roi(mzml_file, mz_tol=param_dict['mz_tol'], mz_units=param_dict['mz_units'],
                                  min_length=param_dict['min_length'], min_intensity=param_dict['min_intensity'],
                                  start_rt=param_dict['start_rt'], stop_rt=param_dict['stop_rt'])
        all_roi = good_roi

        # turn ROI to chemicals
        rtcc = RoiToChemicalCreator(ps, all_roi, n_peaks=param_dict['n_peaks'])
        dataset = rtcc.chemicals
        datasets.append(dataset)

        # save extracted chemicals
        if out_dir is None:  # if no out_dir provided, then same in the same location as the mzML file
            dataset_name = os.path.splitext(mzml_file)[0] + '.p'
            save_obj(dataset, dataset_name)
        else:  # else save the chemicals in our_dir, using pattern as the filename
            basename = os.path.basename(file_names[i])
            out_name = pattern % int(basename.split('_')[2])
            save_obj(dataset, os.path.join(out_dir, out_name))

    return datasets