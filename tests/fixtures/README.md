This folder contains various test fixtures used by unit tests in ViMMS.

- `hmdb_compounds.p` is the pickled file of chemical formulae extracted from HMDB. It can be
  created by following [this script](https://github.com/glasgowcompbio/vimms/blob/master/vimms/scripts/generate_test_fixtures.py).
  It is used to test Chemical generation by sampling from HDMB formulae.
- `beer_compounds.p` is the pickled file of Chemicals objects that can directly be used as input for
  simulation in ViMMS. It can be created by following [this script](https://github.com/glasgowcompbio/vimms/blob/master/vimms/scripts/generate_test_fixtures.py).
  It can be used as input to test various controllers in the unit tests.
- `small_mzml.mzML` is an example mzML file. It is used to test Chemical generation from an mzML file.
- `small_mgf.mgf` is an example MGF file. It is used to test Chemical generation from an MGF file.
- `StdMix1_pHILIC_Current.csv`, `StdMix2_pHILIC_Current.csv` and `StdMix3_pHILIC_Current` is a list of compounds having known formulae
  in the ToxID format. It is used in `test_controllers_targeted.py` to test the targeted fragmentation of these compounds.