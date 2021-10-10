import os
import platform
import sys

sys.path.append('..')

from vimms.Common import load_obj, POSITIVE, set_log_level_warning
from vimms.Controller import FixedScansController
from vimms.DsDA import get_schedule, dsda_get_scan_params
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.MzmlWriter import MzmlWriter

system = platform.system()
if system == "Darwin":
    base_dir = '/Users/joewandy/University of Glasgow/Vinny Davies - CLDS Metabolomics Project/DsDAexample_test'
elif system == "Windows":
    base_dir = 'C:\\Users\\joewa\\University of Glasgow\\Vinny Davies - CLDS Metabolomics Project\\DsDAexample_test'

mzml_filename = os.path.join(base_dir, 'write.mzML')

i = 1
schedule_dir = os.path.join(base_dir, 'settings')
template_file = os.path.join(base_dir, 'DsDA_Timing_schedule.csv')
isolation_window = 1
rt_tol = 15
mz_tol = 10

print('Looking for next schedule')
new_schedule = get_schedule(i, schedule_dir, sleep=False)
print('Found next schedule')
schedule_param_list = dsda_get_scan_params(new_schedule, template_file, isolation_window, mz_tol,
                                           rt_tol)
# schedule_param_list = load_obj(os.path.join(base_dir, 'schedule_param_list.p'))
datasets = load_obj(os.path.join(base_dir, 'datasets.p'))
min_rt = 0
max_rt = 1250
i = 1

set_log_level_warning()
controller = FixedScansController(schedule=schedule_param_list)
mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)
env.run()

writer = MzmlWriter('my_analysis', controller.scans)
writer.write_mzML(mzml_filename)
print('File written to %s' % mzml_filename)
