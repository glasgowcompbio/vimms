# Chromatographic Models

Retention times and chromatographic behaviour of chemicals are controlled by the `Column` classes. A column assigns each chemical an offset and optional drift that modifies its chromatogram.

## Clean Columns

`CleanColumn` applies reproducible offsets to all chemicals, keeping their elution order fixed. A simple example uses fixed offsets in minutes:

```python
from vimms.Column import CleanColumn

column = CleanColumn.from_fixed_offsets([chemicals], 0.0, 1.0, 2.0)
```

Offsets can instead be sampled from normal or uniform distributions using `from_gaussian_offsets` or `from_uniform_offsets`.

## Linear Drift

`LinearColumn` extends `CleanColumn` with a retention time drift function, allowing systematic shifts across injections. Provide either fixed offsets or a custom `drift_fn` which returns the drift for a given ROI and injection index:

```python
from vimms.Column import LinearColumn

col = LinearColumn.from_fixed_offsets(chemicals, start=0.0, step=1.0, drift=0.5)
```

Drift can help emulate column ageing or temperature effects between samples.

## Empirical Chromatograms

Chemicals generated from mzML files use `EmpiricalChromatogram` objects that reproduce observed intensities. You can also define functional chromatograms via distributions such as Gaussian or Gamma by creating `FunctionalChromatogram` instances.

See [`vimms.Column`](api/column.md) and [`vimms.Chromatograms`](api/chemicals.md) for complete details.
