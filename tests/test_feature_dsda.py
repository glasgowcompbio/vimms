import zipfile
import pandas as pd
import numpy as np

from vimms.FeatureExtraction import extract_hmdb_metabolite
from vimms.DsDA import dsda_get_scan_params, create_dsda_schedule
from vimms.Common import INITIAL_SCAN_ID


def test_extract_hmdb_metabolite(tmp_path):
    xml_content = """<?xml version='1.0' encoding='utf-8'?>
<metabolites xmlns='http://www.hmdb.ca'>
  <metabolite>
    <name>Water</name>
    <chemical_formula>H2O</chemical_formula>
    <monisotopic_molecular_weight>18.0</monisotopic_molecular_weight>
    <smiles>O</smiles>
    <inchi>InChI=1S/H2O/h1H2</inchi>
    <inchikey>Q2ZDBBJKTPGOMY-UHFFFAOYSA-N</inchikey>
  </metabolite>
</metabolites>
"""
    xml_path = tmp_path / "hmdb.xml"
    xml_path.write_text(xml_content)
    zip_path = tmp_path / "hmdb.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(xml_path, arcname="hmdb.xml")
    compounds = extract_hmdb_metabolite(str(zip_path), delete=True)
    assert not zip_path.exists()
    assert len(compounds) == 1
    c = compounds[0]
    assert c.name == "Water"
    assert c.chemical_formula == "H2O"
    assert c.monisotopic_molecular_weight == "18.0"
    assert c.smiles == "O"
    assert c.inchi.startswith("InChI")
    assert c.inchikey.startswith("Q2Z")


def test_dsda_get_scan_params(tmp_path):
    schedule = pd.DataFrame({"targetMass": [np.nan, 150.0, 200.0]})
    template = pd.DataFrame({"type": ["ms", "msms", "msms"]})
    sched_file = tmp_path / "sched.csv"
    templ_file = tmp_path / "templ.csv"
    schedule.to_csv(sched_file, index=False)
    template.to_csv(templ_file, index=False)
    scans = dsda_get_scan_params(sched_file, templ_file, 1.0, 0.1, 10)
    assert len(scans) == 3
    assert scans[0].get(scans[0].MS_LEVEL) == 1
    assert scans[1].get(scans[1].MS_LEVEL) == 2
    assert scans[1].get(scans[1].PRECURSOR_MZ)[0].precursor_scan_id == INITIAL_SCAN_ID


def test_create_dsda_schedule(tmp_path):
    class FakeMS:
        scan_duration_dict = {1: 1.0, 2: 2.0}
    # use a short max_rt so the schedule loop runs zero iterations
    create_dsda_schedule(FakeMS, 1, 0, 0.5, tmp_path)
    csv_file = tmp_path / "DsDA_Timing_schedule.csv"
    assert csv_file.exists()
    df = pd.read_csv(csv_file)
    assert list(df.columns) == ["rt", "f", "type"]
    # first row corresponds to min_rt and lm type
    assert df.iloc[0]["rt"] == 0
    assert df.iloc[0]["type"] == "lm"
