from pathlib import Path

from loguru import logger

from tests.conftest import BASE_DIR
from vimms.ChemicalSamplers import EvenMZFormulaSampler, UniformRTAndIntensitySampler, \
    ConstantChromatogramSampler, FixedMS2Sampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import POSITIVE, set_log_level_warning, set_log_level_debug, ScanParameters
from vimms.Controller import Target, TargetedController, create_targets_from_toxid
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer


class TestTargetedController:
    def test_targeted(self):
        fs = EvenMZFormulaSampler()
        ri = UniformRTAndIntensitySampler(min_rt=0, max_rt=10)
        cr = ConstantChromatogramSampler()
        ms = FixedMS2Sampler()
        cs = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri, chromatogram_sampler=cr,
                                    ms2_sampler=ms)
        d = cs.sample(2, 2)  # sample chems with m/z = 100 and 200
        ionisation_mode = POSITIVE
        targets = []
        targets.append(Target(101, 100, 102, 10, 20, adduct='M+H'))
        targets.append(Target(201, 200, 202, 10, 20, metadata={'a': 1}))
        ce_values = [10, 20, 30]
        n_replicates = 4
        controller = TargetedController(targets, ce_values, n_replicates=n_replicates,
                                        limit_acquisition=True)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, d)
        env = Environment(mass_spec, controller, 5, 25, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check that we go all the scans we wanted
        for ms_level in controller.scans:
            assert len(controller.scans[ms_level]) > 0
        set_log_level_debug()
        target_counts = {t: {c: 0 for c in ce_values} for t in targets}

        for s in controller.scans[2]:
            params = s.scan_params
            pmz = params.get(ScanParameters.PRECURSOR_MZ)[0].precursor_mz
            filtered_targets = list(
                filter(lambda x: (x.from_rt <= s.rt <= x.to_rt) and (x.from_mz <= pmz <= x.to_mz),
                       targets))
            assert len(filtered_targets) == 1
            target = filtered_targets[0]
            ce = params.get(ScanParameters.COLLISION_ENERGY)
            target_counts[target][ce] += 1

        for t in target_counts:
            for ce, count in target_counts[t].items():
                assert count == n_replicates

    def test_target_creation(self):
        toxid_file = Path(BASE_DIR, 'StdMix1_pHILIC_Current.csv')
        targets = create_targets_from_toxid(toxid_file)
        assert len(targets) > 0
        toxid_file = Path(BASE_DIR, 'StdMix2_pHILIC_Current.csv')
        targets = create_targets_from_toxid(toxid_file)
        assert len(targets) > 0
        toxid_file = Path(BASE_DIR, 'StdMix3_pHILIC_Current.csv')
        targets = create_targets_from_toxid(toxid_file)
        assert len(targets) > 0
        set_log_level_debug()
        logger.debug(targets[-1].mz)
