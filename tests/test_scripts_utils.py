import os
from datetime import datetime

import pytest

import vimms.scripts.parse_txt as parse_txt
import vimms.scripts.msdial_wrapper as msdial_wrapper


def test_get_running_number():
    assert parse_txt.get_running_number('abc runningNumber=123 xyz') == 123
    assert parse_txt.get_running_number('no number here') == -1


def test_get_time_parsing():
    line = '[01:02:03.004] stuff'
    t = parse_txt.get_time(line)
    assert t == datetime.strptime('01:02:03.004', '%H:%M:%S.%f')


def test_next_helpers():
    lines = [
        'a',
        'Key = ScanType: Full',
        'Key = StartTime = 0.5',
        'Key = InjectTime = 0.05',
    ]
    assert parse_txt.get_next_ms_level(lines, 0) == 1
    assert parse_txt.get_next_start_time(lines, 0) == 0.5
    assert parse_txt.get_next_inject_time(lines, 0) == 0.05


def test_extract_scan_sequence_simple():
    lines = [
        '[00:00:00.000] Info CanAcceptNextCustomScan',
        '[00:00:01.000] Placing runningNumber=10001',
        'Key = ScanType: Full',
        '[00:00:02.000] Received runningNumber=10001',
        'Key = StartTime = 0.5',
        'Key = InjectTime = 0.05',
    ]
    seq = parse_txt.extract_scan_sequence(lines, start_no=10000)
    assert len(seq) == 1
    s = seq[0]
    assert s.running_number == 10001
    assert s.ms_level == 1
    assert s.start_time == 0.5
    assert s.inject_time == 0.05
    assert (s.receive_time - s.send_time).total_seconds() == 1.0


def test_get_path_in_folder(tmp_path):
    result = msdial_wrapper.get_path_in_folder('file.mzML', '{}.msp', tmp_path)
    assert result == os.path.join(tmp_path, 'file.msp')
    result2 = msdial_wrapper.get_path_in_folder('a_temp_file.txt', None, tmp_path, remove_substring='_temp')
    assert result2 == os.path.join(tmp_path, 'a_file.txt')
