This folder contains various test fixtures used by unit tests in ViMMS.

- `hmdb_compounds.p` is the pickled file of chemical formulae extracted from HMDB. It can be
  created by following [this notebook](https://github.com/glasgowcompbio/vimms/blob/master/demo/01.%20Data/01.%20Extracting%20Chemicals%20from%20HMDB.ipynb).
  It is used to test Chemical generation by sampling from HDMB formulae.
- `QCB_22May19_1.p` is the pickled file of Chemicals objects that can directly be used as input for
  simulation in ViMMS. It can be created by following [this notebook](https://github.com/glasgowcompbio/vimms/blob/master/demo/01.%20Data/02.%20Extracting%20Chemicals%20from%20an%20mzML%20file.ipynb).
  It can be used as input to test various controllers in the unit tests.
- `small_mzml.mzML` is an example mzML file. It is used to test Chemical generation from an mzML file.
- `small_mgf.mgf` is an example MGF file. It is used to test Chemical generation from an MGF file.
- `StdMix1_pHILIC_Current.csv`, `StdMix2_pHILIC_Current.csv` and `StdMix3_pHILIC_Current` is a list of compounds having known formulae
  in the ToxID format. It is used in `test_controllers_targeted.py` to test the targeted fragmentation of these compounds.