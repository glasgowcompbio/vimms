import sys
import os
import glob
from tabulate import tabulate

PATH_TO_PYMZM = '/Users/simon/git/pymzm'

def get_summary(mzml_file_path):
    from ms2_matching import MZMLFile
    summary = {}
    mzml_file = MZMLFile(mzml_file_path)
    scan_sub = mzml_file.scans
    
    # find the first block of ms2 scans
    pos = 0
    while pos < len(scan_sub) and scan_sub[pos].ms_level == 1:
        pos += 1
    start_pos = pos
    
    while pos < len(scan_sub) and scan_sub[pos].ms_level == 2:
        pos += 1
    end_pos = pos

    summary['First MS2 pos'] = start_pos
    summary['First MS2 block length'] = end_pos - start_pos
    summary['StartRT'] = scan_sub[0].rt_in_seconds
    summary['EndRT'] = scan_sub[-1].rt_in_seconds
    summary['Nscans'] = len(scan_sub)
    summary['Scans per sec'] = len(scan_sub)/(scan_sub[-1].rt_in_seconds - scan_sub[0].rt_in_seconds)
    summary['FileName'] = mzml_file_path.split(os.sep)[-1]
    return summary

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Pass a path to an mzml file or to a folder. Optional second argument is path to local pymzm")
        sys.exit(0)
    mzml_path = sys.argv[1] # annoying
    if len(sys.argv) > 2:
        path_to_pymzm = sys.argv[2]
    else:
        path_to_pymzm = PATH_TO_PYMZM
    sys.path.append(path_to_pymzm)


    summaries = []
    heads = ['FileName','StartRT','EndRT','Nscans','Scans per sec','First MS2 pos','First MS2 block length']
    if os.path.isdir(mzml_path):
        print("Extracting mzml from folder")
        file_list = glob.glob(os.path.join(mzml_path,'*.mzML'))
        for mzml_file_path in file_list:
            summary = get_summary(mzml_file_path)
            row = []
            for head in heads:
                row.append(summary[head])
            summaries.append(row)
    else:
        print("Individual file")
        summary = get_summary(mzml_path)
        row = []
        for head in heads:
            row.append(summary[head])
        summaries.append(row)

    table = tabulate(summaries,headers = heads)
    print(table)



