from vimms.Common import POSITIVE, ROI_EXCLUSION_WEIGHTED_DEW
from vimms.Roi import RoiBuilderParams, SmartRoiParams


def get_shared_experiment_params():
    return {
        'topN_params': {
            'ionisation_mode': POSITIVE,
            'N': 10,
            'isolation_width': 0.7,
            'min_ms1_intensity': 5000,
            'mz_tol': 10,
            'rt_tol': 15
        },
        'AIF_params': {
            'ms1_source_cid_energy': 25
        },
        'SWATH_params': {
            'min_mz': 70,
            'max_mz': 1000,
            'width': 100,
            'scan_overlap': 0
        },
        'non_overlap_scoring': {
            'use_smartroi_exclusion': False,  # if True, all non-overlap controllers will use smartroi exclusion
            'use_weighteddew_exclusion': False  # if True, all non-overlap controllers will use weighteddew exclusion
        },
        'non_overlap_params': {
            'roi_params': RoiBuilderParams(min_roi_intensity=0, min_roi_length=3),
            'min_roi_length_for_fragmentation': 3,
        },
        'smartroi_params': {
            'smartroi_params': SmartRoiParams()
        },
        'weighteddew_params': {
            'rt_tol': 120,
            'exclusion_method': ROI_EXCLUSION_WEIGHTED_DEW,
            'exclusion_t_0': 15,
        },
        'grid_params': {
            'min_measure_rt': 0,
            'max_measure_rt': 1440,
            'rt_box_size': 50,
            'mz_box_size': 1
        },
        'scan_duration_dict': {1: 0.59, 2: 0.19}
    }