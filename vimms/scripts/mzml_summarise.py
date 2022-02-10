import glob
import os
import sys

from mass_spec_utils.data_import.mzml import MZMLFile
from tabulate import tabulate

TABLE_HEADS = ['FileName', 'StartRT', 'EndRT', 'Nscans', 'Nscans_MS1',
               'Nscans_MS2', 'Scans per sec', 'First MS2',
               'First MS2 block']


def get_summary(mzml_file_path):
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

    summary['First MS2'] = start_pos
    summary['First MS2 block'] = end_pos - start_pos
    summary['StartRT'] = scan_sub[0].rt_in_seconds
    summary['EndRT'] = scan_sub[-1].rt_in_seconds
    summary['Nscans'] = len(scan_sub)
    summary['Scans per sec'] = len(scan_sub) / (
        scan_sub[-1].rt_in_seconds - scan_sub[0].rt_in_seconds)
    summary['FileName'] = mzml_file_path.split(os.sep)[-1]

    ms1_scans = list(filter(lambda x: x.ms_level == 1, scan_sub))
    ms2_scans = list(filter(lambda x: x.ms_level == 2, scan_sub))
    summary['Nscans_MS1'] = len(ms1_scans)
    summary['Nscans_MS2'] = len(ms2_scans)

    return summary


def make_summary_table(file_list, heads=TABLE_HEADS):
    summaries = []
    for mzml_file_path in file_list:
        summary = get_summary(mzml_file_path)
        row = []
        for head in heads:
            row.append(summary[head])
        summaries.append(row)
    table = tabulate(summaries, headers=heads)
    return table


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Pass a path to an mzml file or to a folder.")
        sys.exit(0)
    mzml_path = sys.argv[1]  # annoying

    summaries = []
    if os.path.isdir(mzml_path):
        print("Extracting mzml from folder")
        file_list = glob.glob(os.path.join(mzml_path, '*.mzML'))
        table = make_summary_table(file_list)
    else:
        print("Individual file")
        table = make_summary_table([mzml_path])

    print(table)
