import csv
class PickedBox(object):
    def __init__(self,peak_id,mz,rt,mz_min,mz_max,rt_min,rt_max,area = None,height = None):
        self.peak_id = peak_id
        self.rt = rt
        self.rt_in_seconds = rt*60.0
        self.mz = mz
        self.rt_in_minutes = rt
        self.mz_range = [mz_min,mz_max]
        self.rt_range = [rt_min,rt_max]
        self.ms2_scans = []
        self.area = area
        self.height = height
        self.rt_range_in_seconds = [60.0*r for r in self.rt_range]
    
    def __str__(self):
        return str(self.peak_id) + ": " + str(self.mz_range) + " " + str(self.rt_range)
    
    def find_ms2_scans(self,mzml_file):
        mz_range = self.mz_range
        rt_range = self.rt_range
        scans = list(filter(lambda x: 
                    x.ms_level == 2 and
                            x.precursor_mz >= mz_range[0] and
                            x.precursor_mz <= mz_range[1] and
                            x.rt_in_minutes >= rt_range[0] and
                            x.rt_in_minutes <= rt_range[1],
                    mzml_file.scans))
        self.ms2_scans = scans
        

def load_picked_boxes(csv_name):
    boxes = []
    with open(csv_name,'r') as f:
        reader = csv.reader(f)
        heads = next(reader)
        id_pos = 0
        mz_pos = 1
        rt_pos = 2
        rt_start = [i for i,h in enumerate(heads) if 'RT start' in h][0]
        rt_end = [i for i,h in enumerate(heads) if 'RT end' in h][0]
        mz_min = [i for i,h in enumerate(heads) if 'm/z min' in h][0]
        mz_max = [i for i,h in enumerate(heads) if 'm/z max' in h][0]
        try:
            area_pos = [i for i,h in enumerate(heads) if 'Peak area' in h][0]
            height_pos = [i for i,h in enumerate(heads) if 'Peak height' in h][0]
        except:
            area_pos = None
            height_pos = None
        for line in reader:
            if area_pos:
                boxes.append(PickedBox(int(line[id_pos]),
                                    float(line[mz_pos]),
                                    float(line[rt_pos]),
                                    float(line[mz_min]),
                                    float(line[mz_max]),
                                    float(line[rt_start]),
                                    float(line[rt_end]),
                                    area = float(line[area_pos]),
                                    height = float(line[height_pos])))
            else:
                boxes.append(PickedBox(int(line[id_pos]),
                                    float(line[mz_pos]),
                                    float(line[rt_pos]),
                                    float(line[mz_min]),
                                    float(line[mz_max]),
                                    float(line[rt_start]),
                                    float(line[rt_end])))

        
    return boxes


def map_boxes_to_scans(mzml_file,boxes,half_isolation_window = 0.75,allow_last_overlap = False, scan_shift_seconds = 0):
    scans2boxes = {}
    boxes2scans = {}
    for scan in mzml_file.scans:
        if scan.ms_level == 1:
            continue
        rt = scan.rt_in_minutes + scan_shift_seconds/60.
        if allow_last_overlap:
            previous_ms1 = scan.previous_ms1
            if previous_ms1:
                rt = previous_ms1.rt_in_minutes + scan_shift_seconds/60.
        min_mz = scan.precursor_mz - half_isolation_window
        max_mz = scan.precursor_mz + half_isolation_window
        sub_boxes = list(filter(lambda x: 
                        min_mz <= x.mz_range[1] and
                        max_mz >= x.mz_range[0] and
                        rt >= x.rt_range[0] and 
                        rt <= x.rt_range[1], boxes))
        if len(sub_boxes) > 0:
            scans2boxes[scan] = sub_boxes
            for box in sub_boxes:
                if not box in boxes2scans:
                    boxes2scans[box] = []
                boxes2scans[box].append(scan)
    return scans2boxes,boxes2scans

def traverse_boxes_scans(boxes2scans,scans2boxes,start_scan):
    subset_scans = set()
    subset_boxes = set()
    scans_to_explore = set()
    scans_to_explore.add(start_scan)
    while len(scans_to_explore) > 0:
        this_scan = scans_to_explore.pop()
        subset_scans.add(this_scan)
        sub_boxes = scans2boxes[this_scan]
        for b in sub_boxes:
            if not b in subset_boxes:
                subset_boxes.add(b)
                for sc in boxes2scans[b]:
                    if not sc in subset_scans:
                        scans_to_explore.add(sc)
    return subset_scans,subset_boxes