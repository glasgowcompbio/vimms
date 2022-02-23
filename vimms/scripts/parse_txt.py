# parse the .txt output
import argparse
import re
from datetime import datetime

import pylab as plt


def get_running_number(line):
    m = re.search('runningNumber=(\\d+)', line)
    if m is None:
        return -1
    else:
        return int(m.group(1))


class SequenceItem():
    def __init__(self, runningNumber):
        self.running_number = runningNumber
        self.ms_level = None
        self.start_time = None
        self.inject_time = None

    def add_send(self, send_time):
        self.send_time = send_time

    def add_receive(self, receive_time):
        self.receive_time = receive_time
        self.duration = self.receive_time - self.send_time

    def __str__(self):
        return "{}, {}: {}, {}".format(self.running_number, self.ms_level,
                                       self.duration.total_seconds(),
                                       60 * self.start_time)


def get_time(line):
    time_token = line.split()[0]
    dd = datetime.strptime(time_token, '[%H:%M:%S.%f]')
    return dd


def get_next_ms_level(lines, pos):
    while 'Key = ScanType' not in lines[pos]:
        pos += 1
    if 'Full' in lines[pos]:
        return 1
    else:
        return 2


def get_next_start_time(lines, pos):
    while 'Key = StartTime' not in lines[pos]:
        pos += 1
    tokens = lines[pos].split('=')
    return float(tokens[2])


def get_next_inject_time(lines, pos):
    while 'Key = InjectTime' not in lines[pos]:
        pos += 1
    tokens = lines[pos].split('=')
    return float(tokens[2])


def get_previous_can_accept(lines, pos):
    while 'CanAcceptNextCustomScan' not in lines[pos]:
        pos -= 1
    return get_time(lines[pos])


def extract_scan_sequence(lines, start_no=10000):
    # scan_no = start_no
    scan_sequence_dict = {}
    for pos, line in enumerate(lines):
        if 'Placing' in line:
            # get the running number
            running_number = get_running_number(line)
            if running_number < start_no:
                continue
            if running_number not in scan_sequence_dict:
                scan_sequence_dict[running_number] = SequenceItem(
                    running_number)
            scan_sequence_dict[running_number].add_send(get_time(line))
            scan_sequence_dict[running_number].ms_level = get_next_ms_level(
                lines, pos)
            scan_sequence_dict[
                running_number].fusion_accept = get_previous_can_accept(lines,
                                                                        pos)
        if 'Received' in line:
            running_number = get_running_number(line)
            if running_number < start_no:
                continue
            if running_number not in scan_sequence_dict:
                scan_sequence_dict[running_number] = SequenceItem(
                    running_number)
            scan_sequence_dict[running_number].add_receive(get_time(line))
            scan_sequence_dict[running_number].start_time = \
                get_next_start_time(lines, pos)
            scan_sequence_dict[
                running_number].inject_time = get_next_inject_time(lines, pos)

    scan_sequence_list = list(scan_sequence_dict.values())
    scan_sequence_list.sort(key=lambda x: x.running_number)
    return scan_sequence_list


# flake8: noqa: C901
def main():
    global plot_from, plot_to, i, y
    parser = argparse.ArgumentParser(description='Parse the log output')
    parser.add_argument('--plot_from', dest='plot_from',
                        type=int, default=None)
    parser.add_argument('--plot_to', dest='plot_to', type=int, default=None)
    parser.add_argument('txt_file', type=str)
    args = parser.parse_args()
    lines = []
    with open(args.txt_file, 'r') as f:
        for line in f:
            lines.append(line)
    print("Loaded {} lines".format(len(lines)))
    scan_sequence_list = extract_scan_sequence(lines)
    start_time = scan_sequence_list[0].send_time
    if args.plot_from is not None or args.plot_to is not None:
        if args.plot_from is None:
            plot_from = 0
        else:
            plot_from = args.plot_from
        if args.plot_to is None:
            plot_to = len(scan_sequence_list)
        else:
            plot_to = args.plot_to

        plot_list = filter(lambda x: plot_from <= x.running_number <= plot_to,
                           scan_sequence_list)
        plot_list = list(plot_list)
        for s in plot_list:
            print(s)

        plt.figure()
        for i, s in enumerate(plot_list):
            if s.ms_level == 1:
                col = 'k'
            else:
                col = 'r'
            y = s.running_number
            plt.plot([s.send_time, s.receive_time], [y, y], col)
            text_string = "{} (i{:.0f})".format(str(
                s.duration.total_seconds()), s.inject_time)

            if i < len(plot_list) - 1:
                # dd = s.send_time - plot_list[i+1].send_time
                ff = 60.0 * (plot_list[i + 1].start_time - s.start_time)
                if s.ms_level == 2 and ff > 0.3:
                    print(s.running_number)
                # gg = s.receive_time - plot_list[i+1].receive_time
                text_string += " ({:.3f})".format(ff)
            plt.text(s.send_time, y + 0.1, text_string)
        plt.title(args.txt_file)
        yl = plt.ylim()
        plt.ylim(yl[0], yl[1] + 1)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    main()
