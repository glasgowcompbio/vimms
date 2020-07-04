import os

from loguru import logger

from vimms.Common import save_obj, create_if_not_exist, set_log_level_debug, set_log_level_warning
from vimms.Controller import TopNController
from vimms.DataGenerator import DataSource, PeakSampler
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer


########################################################################################################################
# Codes to set up experiments
########################################################################################################################


def run_experiment(param):
    '''
    Runs a Top-N experiment
    :param param: the experimental parameters
    :return: the analysis name that has been successfully ran
    '''
    analysis_name = param['analysis_name']
    mzml_out = param['mzml_out']
    pickle_out = param['pickle_out']
    N = param['N']
    rt_tol = param['rt_tol']

    if os.path.isfile(mzml_out) and os.path.isfile(pickle_out):
        logger.debug('Skipping %s' % (analysis_name))
    else:
        logger.debug('Processing %s' % (analysis_name))
        peak_sampler = param['peak_sampler']
        if peak_sampler is None:  # extract density from the fragmenatation file
            mzml_path = param['mzml_path']
            fragfiles = param['fragfiles']
            fragfile = fragfiles[(N, rt_tol,)]
            min_rt = param['min_rt']
            max_rt = param['max_rt']
            peak_sampler = get_peak_sampler(mzml_path, fragfile, min_rt, max_rt)

        mass_spec = IndependentMassSpectrometer(param['ionisation_mode'], param['data'], peak_sampler)
        controller = TopNController(param['ionisation_mode'], param['N'], param['isolation_width'],
                                    param['mz_tol'], param['rt_tol'], param['min_ms1_intensity'])
        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, param['min_rt'], param['max_rt'], progress_bar=param['pbar'])
        set_log_level_warning()
        env.run()
        set_log_level_debug()
        env.write_mzML(None, mzml_out)
        save_obj(controller, pickle_out)
        return analysis_name


def get_peak_sampler(mzml_path, fragfile, min_rt, max_rt):
    ds = DataSource()
    ds.load_data(mzml_path, file_name=fragfile)
    kde_min_ms1_intensity = 0  # min intensity to be selected for kdes
    kde_min_ms2_intensity = 0
    peak_sampler = PeakSampler(ds, kde_min_ms1_intensity, kde_min_ms2_intensity, min_rt, max_rt)
    return peak_sampler


def run_parallel_experiment(params):
    '''
    Runs experiments in parallel using iParallel library
    :param params: the experimental parameter
    :return: None
    '''
    import ipyparallel as ipp
    rc = ipp.Client()
    dview = rc[:]  # use all engines​
    with dview.sync_imports():
        pass

    analysis_names = dview.map_sync(run_experiment, params)
    for analysis_name in analysis_names:
        logger.debug(analysis_name)


def run_serial_experiment(params):
    '''
    Runs experiments serially
    :param params: the experimental parameter
    :return: None
    '''
    total = len(params)
    for i in range(len(params)):
        param = params[i]
        logger.debug('Processing \t%d/%d\t%s' % (i + 1, total, param['analysis_name']))
        run_experiment(param)


def get_params(experiment_name, Ns, rt_tols, mz_tol, isolation_width, ionisation_mode, data, peak_sampler,
               min_ms1_intensity, min_rt, max_rt,
               out_dir, pbar, mzml_path=None, fragfiles=None):
    '''
    Creates a list of experimental parameters
    :param experiment_name: current experimental name
    :param Ns: possible values of N in top-N to test
    :param rt_tols: possible values of DEW to test
    :param mz_tol: Top-N controller parameter: the m/z window (ppm) to prevent the same precursor ion to be fragmented again
    :param isolation_width: Top-N controller parameter: the m/z window (ppm) to prevent the same precursor ion to be fragmented again
    :param ionisation_mode: Top-N controller parameter: either positive or negative
    :param data: chemicals to fragment
    :param peak sampler: trained densities to sample values during simulatin
    :param min_ms1_intensity: Top-N controller parameter: minimum ms1 intensity to fragment
    :param min_rt: start RT to simulate
    :param max_rt: end RT to simulate
    :param out_dir: output directory
    :param pbar: progress bar to update
    :return: a list of parameters
    '''
    create_if_not_exist(out_dir)
    logger.debug('N =', Ns)
    logger.debug('rt_tol =', rt_tols)
    params = []
    for N in Ns:
        for rt_tol in rt_tols:
            analysis_name = 'experiment_%s_N_%d_rttol_%d' % (experiment_name, N, rt_tol)
            mzml_out = os.path.join(out_dir, '%s.mzML' % analysis_name)
            pickle_out = os.path.join(out_dir, '%s.p' % analysis_name)
            param_dict = {
                'N': N,
                'mz_tol': mz_tol,
                'rt_tol': rt_tol,
                'min_ms1_intensity': min_ms1_intensity,
                'isolation_width': isolation_width,
                'ionisation_mode': ionisation_mode,
                'data': data,
                'peak_sampler': peak_sampler,
                'min_rt': min_rt,
                'max_rt': max_rt,
                'analysis_name': analysis_name,
                'mzml_out': mzml_out,
                'pickle_out': pickle_out,
                'pbar': pbar
            }
            if mzml_path is not None:
                param_dict['mzml_path'] = mzml_path
            if fragfiles is not None:
                param_dict['fragfiles'] = fragfiles
            params.append(param_dict)
    logger.debug('len(params) =', len(params))
    return params


def get_N_rt_tol_from_qcb_filename(fragfile):
    base = os.path.basename(fragfile)
    base = os.path.splitext(base)[0]
    tokens = base.split('_')
    N = int(tokens[1][1:])
    rt_tol = int(tokens[2][3:])
    return N, rt_tol
