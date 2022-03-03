"""
This file provides a class for writing mzML output from simulation.
For the actual generating of mzML file, the psims library is used.
"""
import os

import numpy as np
from loguru import logger
from psims.mzml.writer import MzMLWriter as PsimsMzMLWriter

from vimms.Common import INITIAL_SCAN_ID, create_if_not_exist, \
    DEFAULT_MS1_SCAN_WINDOW, POSITIVE, NEGATIVE, \
    ScanParameters


class MzmlWriter():
    """
    A class to write peak data to mzML file, typically called after running simulation.
    """

    def __init__(self, analysis_name, scans):
        """
        Initialises the mzML writer class.

        Args:
            analysis_name: Name of the analysis.
            scans: dict where key is scan level, value is a list of Scans object for that level.
        """
        self.analysis_name = analysis_name
        self.scans = scans

    def write_mzML(self, out_file):
        """
        Write mzMl to output file

        Args:
            out_file: path to mzML file

        Returns: None

        """
        # if directory doesn't exist, create it
        out_dir = os.path.dirname(out_file)
        create_if_not_exist(out_dir)

        # start writing mzML here
        with PsimsMzMLWriter(open(out_file, 'wb')) as writer:
            # add default controlled vocabularies
            writer.controlled_vocabularies()

            # write other fields like sample list, software list, etc.
            self._write_info(writer)

            # open the run
            with writer.run(id=self.analysis_name):
                self._write_spectra(writer, self.scans)

                # open chromatogram list sections
                with writer.chromatogram_list(count=1):
                    tic_rts, tic_intensities = self._get_tic_chromatogram(
                        self.scans)
                    writer.write_chromatogram(
                        tic_rts, tic_intensities, id='tic',
                        chromatogram_type='total ion current chromatogram',
                        time_unit='second')

        writer.close()

    def _write_info(self, out):
        """
        Write various information to output stream
        Args:
            out: the output stream from psims

        Returns: None

        """
        # check file contains what kind of spectra
        has_ms1_spectrum = 1 in self.scans
        has_msn_spectrum = 1 in self.scans and len(self.scans) > 1
        file_contents = [
            'centroid spectrum'
        ]
        if has_ms1_spectrum:
            file_contents.append('MS1 spectrum')
        if has_msn_spectrum:
            file_contents.append('MSn spectrum')
        out.file_description(
            file_contents=file_contents,
            source_files=[]
        )
        out.sample_list(samples=[])
        out.software_list(software_list={
            'id': 'VMS',
            'version': '1.0.0'
        })
        out.scan_settings_list(scan_settings=[])
        out.instrument_configuration_list(instrument_configurations={
            'id': 'VMS',
            'component_list': []
        })
        out.data_processing_list({'id': 'VMS'})

    def sort_filter(self, all_scans, min_scan_id):
        """
        Filter scans according to certain criteria. Currently it filters by
        the minimum scan ID, as a workaround to IAPI which produces unwanted scans at
        low scan IDs.

        Args:
            all_scans: the list of scans to filter
            min_scan_id: the minimum scan ID to filter

        Returns: the list of filtered scans

        """
        all_scans = sorted(all_scans, key=lambda x: x.rt)
        all_scans = [x for x in all_scans if x.num_peaks > 0]
        all_scans = list(filter(lambda x: x.scan_id >= min_scan_id, all_scans))

        # FIXME: why do we need to do this???!!
        # add a single peak to empty scans
        # empty = [x for x in all_scans if x.num_peaks == 0]
        # for scan in empty:
        #     scan.mzs = np.array([100.0])
        #     scan.intensities = np.array([1.0])
        #     scan.num_peaks = 1
        return all_scans

    def _write_spectra(self, writer, scans, min_scan_id=INITIAL_SCAN_ID):
        """
        Helper method to actually write a collection of spectra
        Args:
            writer: the output stream from psims
            scans: a list of scans
            min_scan_id: the minimum scan ID to write

        Returns: None

        """
        # NOTE: we only support writing up to ms2 scans for now
        assert len(scans) <= 3

        # get all scans across different ms_levels and sort them by scan_id
        all_scans = []
        for ms_level in scans:
            all_scans.extend(scans[ms_level])
        all_scans = self.sort_filter(all_scans, min_scan_id)
        spectrum_count = len(all_scans)

        # write scans
        with writer.spectrum_list(count=spectrum_count):
            for scan in all_scans:
                self._write_scan(writer, scan)

    def _write_scan(self, out, scan):
        """
        Helper method to write a single scan
        Args:
            out: the psims output stream
            scan: the scan to write

        Returns: None

        """
        assert scan.num_peaks > 0
        label = 'MS1 Spectrum' if scan.ms_level == 1 else 'MSn Spectrum'
        precursor_information = None
        if scan.ms_level == 2:
            collision_energy = scan.scan_params.get(
                ScanParameters.COLLISION_ENERGY)
            activation_type = scan.scan_params.get(
                ScanParameters.ACTIVATION_TYPE)

            precursor_information = []
            for precursor in scan.scan_params.get(ScanParameters.PRECURSOR_MZ):
                precursor_information.append({
                    "mz": precursor.precursor_mz,
                    "intensity": precursor.precursor_intensity,
                    "charge": precursor.precursor_charge,
                    "spectrum_reference": precursor.precursor_scan_id,
                    "activation": [activation_type,
                                   {"collision energy": collision_energy}]
                })

        lowest_observed_mz = min(scan.mzs)
        highest_observed_mz = max(scan.mzs)
        # bp_pos = np.argmax(scan.intensities)
        # bp_intensity = scan.intensities[bp_pos]
        # bp_mz = scan.mzs[bp_pos]
        scan_id = scan.scan_id

        try:
            first_mz = scan.scan_params.get(ScanParameters.FIRST_MASS)
            last_mz = scan.scan_params.get(ScanParameters.LAST_MASS)
        # if it's a method scan (not a custom scan), there's no scan_params
        # to get first_mz and last_mz
        except AttributeError:
            first_mz, last_mz = DEFAULT_MS1_SCAN_WINDOW

        polarity = scan.scan_params.get(ScanParameters.POLARITY)
        if polarity == POSITIVE:
            int_polarity = 1
        elif polarity == NEGATIVE:
            int_polarity = -1
        else:
            int_polarity = 1
            logger.warning(
                "Unknown polarity in mzml writer: {}".format(polarity))

        out.write_spectrum(
            scan.mzs, scan.intensities,
            id=scan_id,
            polarity=int_polarity,
            centroided=True,
            scan_start_time=scan.rt / 60.0,
            scan_window_list=[(first_mz, last_mz)],
            params=[
                {label: ''},
                {'ms level': scan.ms_level},
                {'total ion current': np.sum(scan.intensities)},
                {'lowest observed m/z': lowest_observed_mz},
                {'highest observed m/z': highest_observed_mz},
                # {'base peak m/z', bp_mz},
                # {'base peak intensity', bp_intensity}
            ],
            precursor_information=precursor_information
        )

    def _get_tic_chromatogram(self, scans):
        """
        Helper method to write total ion chromatogram information
        Args:
            scans: the list of scans

        Returns: a tuple of time array and intensity arrays for the TIC

        """
        time_array = []
        intensity_array = []
        for ms1_scan in scans[1]:
            time_array.append(ms1_scan.rt)
            intensity_array.append(np.sum(ms1_scan.intensities))
        time_array = np.array(time_array)
        intensity_array = np.array(intensity_array)
        return time_array, intensity_array
