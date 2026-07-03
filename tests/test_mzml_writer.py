import vimms.MzmlWriter as mzml_writer_module
from vimms.MzmlWriter import MzmlWriter


class _ContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakePsimsMzMLWriter:
    opened_handles = []

    def __init__(self, out_handle):
        self.opened_handles.append(out_handle)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def controlled_vocabularies(self):
        pass

    def file_description(self, *args, **kwargs):
        pass

    def sample_list(self, *args, **kwargs):
        pass

    def software_list(self, *args, **kwargs):
        pass

    def scan_settings_list(self, *args, **kwargs):
        pass

    def instrument_configuration_list(self, *args, **kwargs):
        pass

    def data_processing_list(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return _ContextManager()

    def spectrum_list(self, *args, **kwargs):
        return _ContextManager()

    def chromatogram_list(self, *args, **kwargs):
        return _ContextManager()

    def write_chromatogram(self, *args, **kwargs):
        pass

    def close(self):
        pass


def test_mzml_writer_closes_output_handle(monkeypatch, tmp_path):
    _FakePsimsMzMLWriter.opened_handles = []
    monkeypatch.setattr(mzml_writer_module, "PsimsMzMLWriter", _FakePsimsMzMLWriter)

    MzmlWriter("test_analysis", {1: []}).write_mzML(tmp_path / "test.mzML")

    assert len(_FakePsimsMzMLWriter.opened_handles) == 1
    assert _FakePsimsMzMLWriter.opened_handles[0].closed
