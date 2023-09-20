---
nav_exclude: true
---
# Running Controllers

After creating chemicals, you can apply various fragmentation strategies using controller classes in ViMMS. The following example demonstrates generating 100 chemicals and implementing a Top-N DDA strategy using a controller.

```python
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.ChemicalSamplers import UniformMZFormulaSampler
from vimms.Common import POSITIVE, set_log_level_warning
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Controller import TopNController
from vimms.Environment import Environment

# generating chemicals
df = UniformMZFormulaSampler(min_mz=100, max_mz=500)
cm = ChemicalMixtureCreator(df)
chemicals = cm.sample(100, 2)

# setup a virtual mass spec
mass_spec = IndependentMassSpectrometer(POSITIVE, chemicals)

# setup Top-N controller with some parameters
isolation_window = 1
N = 3
rt_tol = 15
mz_tol = 10
min_ms1_intensity = 1.75E5
controller = TopNController(POSITIVE, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity)

# create an environment to run both the mass spec and controller
rt_range = [(0, 1440)]
min_rt = rt_range[0][0]
max_rt = rt_range[0][1]
env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

# set the log level to WARNING to minimize message volume when the environment is running
set_log_level_warning()

# run the simulation
env.run()
```

For additional examples, refer to our [notebooks](https://github.com/glasgowcompbio/vimms/tree/master/demo/02.%20Methods) corresponding to the four controllers described in our paper, [Rapid Development of Improved Data-Dependent Acquisition Strategies](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.0c03895).

## Evaluation

ViMMS allows evaluation of simulation results using methods provided [here](https://github.com/glasgowcompbio/vimms/blob/master/vimms/Evaluation.py). These tools offer a convenient way to assess the performance and outcomes of your simulations.