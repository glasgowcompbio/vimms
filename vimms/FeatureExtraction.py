import os

import numpy as np

from vimms.Common import DEFAULT_MZML_CHEMICAL_CREATOR_PARAMS, save_obj
from vimms.Roi import make_roi, RoiToChemicalCreator


def mzml2chems(mzml_file, ps, param_dict=DEFAULT_MZML_CHEMICAL_CREATOR_PARAMS, output_dir=True, n_peaks=1):
    good_roi, junk = make_roi(mzml_file, mz_tol=param_dict['mz_tol'], mz_units=param_dict['mz_units'],
                              min_length=param_dict['min_length'], min_intensity=param_dict['min_intensity'],
                              start_rt=param_dict['start_rt'], stop_rt=param_dict['stop_rt'])
    all_roi = good_roi + junk
    keep = []
    for roi in all_roi:
        if np.count_nonzero(np.array(roi.intensity_list) > param_dict['min_ms1_intensity']) > 0:
            keep.append(roi)
    all_roi = keep
    rtcc = RoiToChemicalCreator(ps, all_roi, n_peaks)
    dataset = rtcc.chemicals
    if output_dir is True:
        dataset_name = os.path.splitext(mzml_file)[0] + '.p'
        save_obj(dataset, dataset_name)
    return dataset


def extract_roi(file_names, out_dir, pattern, mzml_path, ps, roi_mz_tol=10, roi_min_length=2, roi_min_intensity=1.75E5,
                roi_start_rt=0,
                roi_stop_rt=1440):
    for i in range(len(file_names)):  # for all mzML files in file_names
        # extract ROI
        mzml_file = os.path.join(mzml_path, file_names[i])
        good_roi, junk = make_roi(mzml_file, mz_tol=roi_mz_tol, mz_units='ppm', min_length=roi_min_length,
                                  min_intensity=roi_min_intensity, start_rt=roi_start_rt, stop_rt=roi_stop_rt)
        all_roi = good_roi

        # turn ROI to chemicals
        rtcc = RoiToChemicalCreator(ps, all_roi)
        data = rtcc.chemicals

        # save extracted chemicals
        basename = os.path.basename(file_names[i])
        out_name = pattern % int(basename.split('_')[2])
        save_obj(data, os.path.join(out_dir, out_name))