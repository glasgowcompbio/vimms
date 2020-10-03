from intervaltree import IntervalTree

# Exclusion.py
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


if __name__ == '__main__':
    e = ExclusionItem(1.1,1.2,3.4,3.5,3.45)
    f = ExclusionItem(1.0,1.4,3.3,3.6,3.45)
    g = ExclusionItem(2.1,2.2,3.2,3.5,3.45)
    b = BoxHolder()
    b.add_box(e)
    b.add_box(f)
    b.add_box(g)
    print(b.is_in_box(1.15, 3.55))
    print(b.is_in_box(1.15, 3.75))