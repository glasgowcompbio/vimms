from intervaltree import IntervalTree
from loguru import logger

# Exclusion.py
from vimms.Common import ScanParameters


class ExclusionItem(object):
    """
    A class to store the item to exclude when computing dynamic exclusion window
    """

    def __init__(self, from_mz, to_mz, from_rt, to_rt, frag_at):
        """
        Creates a dynamic exclusion item
        :param from_mz: m/z lower bounding box
        :param to_mz: m/z upper bounding box
        :param from_rt: RT lower bounding box
        :param to_rt: RT upper bounding box
        """
        self.from_mz = from_mz
        self.to_mz = to_mz
        self.from_rt = from_rt
        self.to_rt = to_rt
        self.frag_at = frag_at
        self.mz = (self.from_mz + self.to_mz) / 2.
        self.rt = self.frag_at

    def peak_in(self, mz, rt):
        if self.rt_match(rt) and self.mz_match(mz):
            return True
        else:
            return False

    def rt_match(self, rt):
        if rt >= self.from_rt and rt <= self.to_rt:
            return True
        else:
            return False

    def mz_match(self, mz):
        if mz >= self.from_mz and mz <= self.to_mz:
            return True
        else:
            return False

    def __repr__(self):
        return 'ExclusionItem mz=(%f, %f) rt=(%f-%f)' % (self.from_mz, self.to_mz, self.from_rt, self.to_rt)

    def __lt__(self, other):
        if self.from_mz <= other.from_mz:
            return True
        else:
            return False


class BoxHolder(object):
    """
    A class to allow quick lookup of boxes (e.g. exclusion items, targets, etc)
    Creates an interval tree on mz as this is likely to narrow things down quicker
    Also has a method for returning an rt interval tree for a particular mz
    and an mz interval tree for a particular rt
    """

    def __init__(self):
        self.boxes_mz = IntervalTree()
        self.boxes_rt = IntervalTree()

    def add_box(self, box):
        """
        Add a box to the IntervalTree
        """
        mz_from = box.from_mz
        mz_to = box.to_mz
        rt_from = box.from_rt
        rt_to = box.to_rt
        self.boxes_mz.addi(mz_from, mz_to, box)
        self.boxes_rt.addi(rt_from, rt_to, box)

    def check_point(self, mz, rt):
        """
        Find the boxes that match this mz and rt value
        """
        regions = self.boxes_mz.at(mz)
        hits = set()
        for r in regions:
            if r.data.rt_match(rt):
                hits.add(r.data)
        return hits

    def check_point_2(self, mz, rt):
        """
        An alternative method that searches both trees
        Might be faster if there are lots of rt ranges that 
        can map to a particular mz value
        """
        mz_regions = self.boxes_mz.at(mz)
        rt_regions = self.boxed_rt.at(rt)
        inter = mz_regions.intersection(rt_regions)
        return [r.data for r in inter]

    def is_in_box(self, mz, rt):
        """
        Check if this mz and rt is in *any* box
        """
        hits = self.check_point(mz, rt)
        if len(hits) > 0:
            return True
        else:
            return False

    def is_in_box_mz(self, mz):
        """
        Check if an mz value is in any box
        """
        regions = self.boxes_mz.at(mz)
        if len(regions) > 0:
            return True
        else:
            return False

    def is_in_box_rt(self, rt):
        """
        Check if an rt value is in any box
        """
        regions = self.boxes_rt.at(rt)
        if len(regions) > 0:
            return True
        else:
            return False

    def get_subset_rt(self, rt):
        """
        Create an interval tree based upon mz for all boxes active at rt
        """
        regions = self.boxes_rt.at(rt)
        it = BoxHolder()
        for r in regions:
            box = r.data
            it.add_box(box)
        return it

    def get_subset_mz(self, mz):
        """
        Create an interval tree based upon rt fro all boxes active at mz
        """
        regions = self.boxes_mz.at(mz)
        it = BoxHolder()
        for r in regions:
            box = r.data
            it.add_box(box)
        return it


def generate_exclusion(rt, tasks):
    temp_exclusion_list = []
    for task in tasks:
        for precursor in task.get('precursor_mz'):
            mz = precursor.precursor_mz
            mz_tol = task.get(ScanParameters.DYNAMIC_EXCLUSION_MZ_TOL)
            rt_tol = task.get(ScanParameters.DYNAMIC_EXCLUSION_RT_TOL)
            x = _get_exclusion_item(mz, rt, mz_tol, rt_tol)
            logger.debug('Time {:.6f} Created dynamic temporary exclusion window mz ({}-{}) rt ({}-{})'.format(
                rt,
                x.from_mz, x.to_mz, x.from_rt, x.to_rt
            ))
            x = _get_exclusion_item(mz, rt, mz_tol, rt_tol)
            temp_exclusion_list.append(x)
    return temp_exclusion_list


def _get_exclusion_item(mz, rt, mz_tol, rt_tol):
    mz_lower = mz * (1 - mz_tol / 1e6)
    mz_upper = mz * (1 + mz_tol / 1e6)
    rt_lower = rt - rt_tol
    rt_upper = rt + rt_tol
    x = ExclusionItem(from_mz=mz_lower, to_mz=mz_upper, from_rt=rt_lower, to_rt=rt_upper,
                      frag_at=rt)
    return x


def _is_excluded(exclusion_list, mz, rt):
    """
    Checks if a pair of (mz, rt) value is currently excluded by dynamic exclusion window
    :param mz: m/z value
    :param rt: RT value
    :return: True if excluded, False otherwise
    """
    # TODO: make this faster?
    for x in exclusion_list:
        exclude_mz = x.from_mz <= mz <= x.to_mz
        exclude_rt = x.from_rt <= rt <= x.to_rt
        if exclude_mz and exclude_rt:
            logger.debug(
                'Excluded precursor ion mz {:.4f} rt {:.2f} because of {}'.format(mz, rt, x))
            return True
    return False


def manage_exclusion(scan, exclusion_list):
    """
    Manages dynamic exclusion list
    :param param: a scan parameter object
    :param scan: the newly generated scan
    :return: None
    """
    # current simulated time is scan start RT + scan duration
    # in the real data, scan.duration is not set, so we just use the scan rt as the current time
    current_time = scan.rt
    if scan.scan_duration is not None:
        current_time += scan.scan_duration

    # remove expired items from dynamic exclusion list
    filtered_list = list(filter(lambda x: x.to_rt > current_time, exclusion_list))
    return filtered_list


def _is_weightedDEW_excluded(exclusion_list, mz, rt, rt_tol, exclusion_t_0):
    """
    Checks if a pair of (mz, rt) value is currently excluded by dynamic exclusion window
    :param mz: m/z value
    :param rt: RT value
    :return: True if excluded, False otherwise
    """
    # TODO: make this faster?
    exclusion_list.sort(key=lambda x: x.from_rt, reverse=True)
    for x in exclusion_list:
        exclude_mz = x.from_mz <= mz <= x.to_mz
        exclude_rt = x.from_rt <= rt <= x.to_rt
        if exclude_mz and exclude_rt:
            logger.debug(
                'Excluded precursor ion mz {:.4f} rt {:.2f} because of {}'.format(mz, rt, x))
            if rt <= x.frag_at + exclusion_t_0:
                return True, 0.0
            else:
                weight = (rt - (exclusion_t_0 + x.frag_at)) / (rt_tol - exclusion_t_0)
                assert weight <= 1, weight
                # self.remove_exclusion_items.append(x)
                return True, weight
    return False, 1


if __name__ == '__main__':
    e = ExclusionItem(1.1, 1.2, 3.4, 3.5, 3.45)
    f = ExclusionItem(1.0, 1.4, 3.3, 3.6, 3.45)
    g = ExclusionItem(2.1, 2.2, 3.2, 3.5, 3.45)
    b = BoxHolder()
    b.add_box(e)
    b.add_box(f)
    b.add_box(g)
    print(b.is_in_box(1.15, 3.55))
    print(b.is_in_box(1.15, 3.75))
