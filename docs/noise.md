# Adding Noise

ViMMS can produce more realistic spectra by injecting noise into simulated scans. Noise objects live in `vimms.Noise` and can be passed directly to a mass spectrometer instance.

## Peak Noise

`GaussianPeakNoise` adds normally distributed variation to peak intensities. The standard deviation controls the noise level:

```python
from vimms.Noise import GaussianPeakNoise
from vimms.MassSpec import IndependentMassSpectrometer

ms = IndependentMassSpectrometer("positive", chemicals,
                                 peak_noise=GaussianPeakNoise(std=0.01))
```

Other peak noise classes include `UniformPeakNoise` and `NoPeakNoise`.

## Baseline Noise

`AdditiveBaselineNoise` injects a constant baseline across the m/z range. Specify the baseline intensity and optional Gaussian width:

```python
from vimms.Noise import AdditiveBaselineNoise

ms = IndependentMassSpectrometer("positive", chemicals,
                                 baseline_noise=AdditiveBaselineNoise(1e3, width=5))
```

Baseline noise is combined with peak noise to yield a noisy spectrum at each scan.

## RT Noise

`RetentionTimeDeviation` introduces random jitter in retention times. Provide a maximum deviation in seconds:

```python
from vimms.Noise import RetentionTimeDeviation

ms = IndependentMassSpectrometer("positive", chemicals,
                                 rt_deviation=RetentionTimeDeviation(max_shift=2.0))
```

Any subset of these noise models can be used simultaneously. For details see the API reference under [`vimms.Noise`](api/noise.md).
