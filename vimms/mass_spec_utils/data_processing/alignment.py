# alignment.py
import csv
import os
import numpy as np
from loguru import logger

from ..data_import.mzmine import load_picked_boxes

class Peak(object): # todo: add MS2 information
    def __init__(self,mz,rt,intensity,source_file,source_id):
        self.mz = mz
        self.rt = rt
        self.intensity = intensity
        self.source_file = source_file
        self.source_id = source_id
        self.ms2_spectrum = None
        
    def add_ms2_spectrum(self,ms2_spectrum):
        self.ms2_spectrum = ms2_spectrum
    
    def __str__(self):
        return "{} ({}): {},{},{}".format(self.source_file,self.source_id,self.mz,self.rt,self.ms2_spectrum)

class PeakSet(object): # todo add MS2 information
    def __init__(self,peak):
        self.peaks = [peak]
        self.n_peaks = 1
        self.mean_mz = self.peaks[0].mz
        self.mean_rt = self.peaks[0].rt
        

    def add_peak(self,peak):
        self.peaks.append(peak)
        self.n_peaks += 1
        self.mean_mz = sum([x.mz for x in self.peaks])/self.n_peaks
        self.mean_rt = sum([x.rt for x in self.peaks])/self.n_peaks
    
    def _actual_mz_tolerance(self,mz,mz_tolerance_absolute,mz_tolerance_ppm):
        ppm_tolerance = mz_tolerance_ppm*self.mean_mz/1e6
        return max(ppm_tolerance,mz_tolerance_absolute)

    def is_in_box(self,peak,mz_tolerance_absolute,mz_tolerance_ppm,rt_tolerance):
        mz_tolerance = self._actual_mz_tolerance(self.mean_mz,mz_tolerance_absolute,mz_tolerance_ppm)
        if peak.mz >= self.mean_mz - mz_tolerance and peak.mz <= self.mean_mz + mz_tolerance:
            if peak.rt >= self.mean_rt - rt_tolerance and peak.rt <= self.mean_rt + rt_tolerance:
                return True
        return False

    def compute_weight(self,peak,mz_tolerance_absolute,mz_tolerance_ppm,rt_tolerance,mz_weight,rt_weight):
        mz_tolerance = self._actual_mz_tolerance(self.mean_mz,mz_tolerance_absolute,mz_tolerance_ppm)
        mz_diff = abs(peak.mz - self.mean_mz)
        rt_diff = abs(peak.rt - self.mean_rt)
        mz_score = ((1-mz_diff)/mz_tolerance)*mz_weight
        rt_score = ((1-rt_diff)/rt_tolerance)*rt_weight
        return mz_score + rt_score

    def get_intensity(self,file_name):
        # return the intensity for the peak from file file_name
        # return 0.0 if file not present
        for peak in self.peaks:
            if peak.source_file == file_name:
                return peak.intensity
        return 0.0

    def get_ms2(self,choice = 'highest_total_intensity'):
        best_ms2 = None
        best_score = 0.0
        for p in self.peaks:
            if not p.ms2_spectrum is None:
                if choice == 'highest_total_intensity':
                    score = sum([i for mz,i in p.ms2_spectrum.peaks])
                if score >= best_score:
                    best_ms2 = p.ms2_spectrum
        return best_ms2
    
  
        
class JoinAligner(object):
    def __init__(self,mz_tolerance_absolute = 0.01,
                 mz_tolerance_ppm = 10,
                 rt_tolerance = 0.5,
                 mz_column_pos = 1,
                 rt_column_pos = 2,
                 intensity_column_pos = 3):
        self.peaksets = []
        self.mz_tolerance_absolute = mz_tolerance_absolute
        self.mz_tolerance_ppm = mz_tolerance_ppm
        self.rt_tolerance = rt_tolerance
        self.mz_weight = 75
        self.rt_weight = 25
        self.files_loaded = []
        self.mz_column_pos = mz_column_pos
        self.rt_column_pos = rt_column_pos
        self.intensity_column_pos = intensity_column_pos
    def add_file(self,input_csv,input_mgf=None):
        with open(input_csv,'r') as f:
            reader =  csv.reader(f)
            heads = next(reader)
            these_peaks = []
            try:
                short_name = input_csv.split(os.sep)[-1].split('.')[0].split('_quant')[0]
            except:
                short_name = input_csv.split(os.sep)[-1]
                
            for line in reader:
                source_id = int(line[0])
                peak_mz = float(line[self.mz_column_pos])
                peak_rt = float(line[self.rt_column_pos])
                peak_intensity = float(line[self.intensity_column_pos]) 

                these_peaks.append(Peak(peak_mz,peak_rt,peak_intensity,short_name,source_id))
        self._align(these_peaks,short_name)
    
    def _align(self,these_peaks,short_name):
        if len(self.peaksets) == 0:
            # first file
            for peak in these_peaks:
                self.peaksets.append(PeakSet(peak))
        else:
            for peakset in self.peaksets:
                candidates = list(filter(lambda x: peakset.is_in_box(x,self.mz_tolerance_absolute,self.mz_tolerance_ppm,self.rt_tolerance),these_peaks))
                if len(candidates) == 0:
                    continue
                else:
                    best_peak = None
                    best_score = 0
                    for peak in candidates:
                        score = peakset.compute_weight(peak,self.mz_tolerance_absolute,self.mz_tolerance_ppm,self.rt_tolerance,self.mz_weight,self.rt_weight)
                        if score > best_score:
                            best_score = score
                            best_peak = peak
                    peakset.add_peak(best_peak)
                    pos = these_peaks.index(best_peak)
                    del these_peaks[pos]
            for peak in these_peaks: # remaining ones
                self.peaksets.append(PeakSet(peak))
        self.files_loaded.append(short_name)

    def filter_peaksets(self,min_peaks_per_peakset):
        temp = list(filter(lambda x: len(x.peaks)>= min_peaks_per_peakset,self.peaksets))
        self.peaksets = temp

    def to_matrix(self):
        n_peaksets = len(self.peaksets)
        n_files = len(self.files_loaded)
        intensity_matrix = np.zeros((n_peaksets,n_files),np.double)
        for i,peakset in enumerate(self.peaksets):
            for j,filename in enumerate(self.files_loaded):
                intensity_matrix[i,j] = peakset.get_intensity(filename)
        return intensity_matrix

class BoxJoinAligner(JoinAligner):
    def __init__(self,mz_tolerance_absolute = 0.01,
                 mz_tolerance_ppm = 10,
                 rt_tolerance = 0.5,
                 mz_column_pos = 1,
                 rt_column_pos = 2,
                 intensity_column_pos = 3):
        super().__init__(mz_tolerance_absolute,
                        mz_tolerance_ppm,
                        rt_tolerance,
                        mz_column_pos,
                        rt_column_pos,
                        intensity_column_pos)
        self.peaksets2boxes = {}

    def add_file(self,input_csv,input_mgf=None):
        with open(input_csv,'r') as f:
            reader =  csv.reader(f)
            heads = next(reader)
            these_peaks = []
            try:
                short_name = input_csv.split(os.sep)[-1].split('.')[0].split('_quant')[0]
            except:
                short_name = input_csv.split(os.sep)[-1]
                
            for line in reader:
                source_id = int(line[0])
                peak_mz = float(line[self.mz_column_pos])
                peak_rt = float(line[self.rt_column_pos])
                peak_intensity = float(line[self.intensity_column_pos]) 

                these_peaks.append(Peak(peak_mz,peak_rt,peak_intensity,short_name,source_id))

        # load the boxes
        temp_boxes = load_picked_boxes(input_csv)
        # convert to dictionary for speedy lookup
        temp_boxes = {box.peak_id: box for box in temp_boxes}
        n_peaksets = len(self.peaksets)
        # do the alignment
        self._align(these_peaks,short_name)

        # add boxes to the *new* peaksets
        self._add_boxes_to_peaksets(temp_boxes,short_name,n_peaksets)

    def _add_boxes_to_peaksets(self,temp_boxes,short_name,n_peaksets):
        for peakset in self.peaksets[n_peaksets:]:
            assert not peakset in self.peaksets2boxes
            peak = peakset.peaks[0] # should only be one peak in the set
            assert peak.source_file == short_name
            # get the box
            peak_id = peak.source_id
            assert peak_id in temp_boxes
            self.peaksets2boxes[peakset] = temp_boxes[peak_id]




                    