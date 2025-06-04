# Running on a Real Instrument

Although ViMMS is primarily a simulator, controllers can also drive a Thermo Orbitrap Fusion Tribrid mass spectrometer via Thermo Fisher's **IAPI**. The `vimms_fusion` extension mirrors the in silico classes with instrument aware counterparts.

## Requirements

* A valid installation of the IAPI libraries from Thermo Fisher Scientific.
* The optional `vimms-fusion` package available in the ViMMS repository.

## Basic Usage

```python
from vimms_fusion.MassSpec import IAPIMassSpectrometer
from vimms_fusion.Environment import IAPIEnvironment
from vimms.Controller import TopNController

mass_spec = IAPIMassSpectrometer(polarity="positive")
controller = TopNController("positive", N=5, isolation_width=1)

with IAPIEnvironment(mass_spec, controller, min_time=0, max_time=900) as env:
    env.run()
```

The same controller implementations used for simulation can therefore be deployed directly on the instrument. Refer to the examples in `examples/04. DDAvsDIA (Wandy et al 2023)` for notebook demonstrations.
