import os

from vimms.Common import load_obj, POSITIVE, set_log_level_warning
from vimms.Controller import FixedScansController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.MzmlWriter import MzmlWriter

base_dir = 'C:\\Users\\joewa\\University of Glasgow\\Vinny Davies - CLDS Metabolomics Project\\DsDAexample_test'
mzml_filename = os.path.join(base_dir, 'write.mzML')

schedule_param_list = load_obj(os.path.join(base_dir, 'schedule_param_list.p'))
tok = 100000
for param in schedule_param_list:
    param.params['uniqueness_token'] = tok
    tok += 1

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
