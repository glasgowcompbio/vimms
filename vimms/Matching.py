import bisect
import copy
import itertools
import math

import intervaltree
import networkx as nx
import numpy as np
from mass_spec_utils.data_import.mzml import MZMLFile

# TODO: test working controller
# TODO: intensities seem to be a bit wonky, many zero values
# TODO: graph extensions in slack
# TODO: scans from mzml function
# TODO: fullscan info should be usable to more precisely give intensity at
#  particular MS2 time than using precursor
# TODO: bin sort version
# TODO: constraint programming/stable marriage versions??
# TODO: general weighted matching is slow, need to find weighted bipartite
#  algorithm in another package
# TODO: could chemical collapse with arbitrary exclusion condition, for when
#  chems_list doesn't have same items e.g. with aligner
# TODO: matching controller should be able to function without precursor MS1
#  scans during actual run???
# TODO: MatchingChem.env2nodes could split chemicals on RoI bounds to have
#  a slightly more accurate view of simulated chemicals
# TODO: MatchingChem.env2nodes could also work on non-RoI controllers
from vimms.Common import (
    get_default_scan_params, INITIAL_SCAN_ID, get_dda_scan_param
)


class MatchingScan():
    def __init__(self, num_injection, ms_level, rt, mzs, intensities):
        self.num_injection = num_injection
        self.ms_level = ms_level
        self.rt = rt
        self.mzs = mzs
        self.intensities = intensities
        
    def __repr__(self): 
        return (
            f"MatchingScan(num_injection={self.num_injection}, "
            f"ms_level={self.ms_level}, rt={self.rt})"
        )
        
    def __eq__(self, other): 
        return (
            isinstance(other, type(self)) 
            and self.num_injection == other.num_injection 
            and self.ms_level == other.ms_level 
            and math.isclose(self.rt, other.rt)
        )
        
    def __hash__(self): 
        return (self.num_injection, self.ms_level, self.rt).__hash__()
        
    def interpolate_scan(left, right, mz_window):
        left_mzs, left_intensities = zip(*left.peaks)
        right_mzs, right_intensities = zip(*right.peaks)
        left_rt, right_rt = left.rt_in_seconds, right.rt_in_seconds

        mzs = np.array(left_mzs + right_mzs)
        idxes = np.argsort(mzs)
        mzs = mzs[idxes]
        intensities = np.array(left_intensities + right_intensities)[idxes]
        owner = (np.arange(0, mzs.shape[0]) >= len(left_mzs))[idxes]

        in_window = []
        left_bound, right_bound = 0, 1
        for i, intensity in enumerate(intensities):
            while(mzs[i] - mzs[left_bound] > mz_window): 
                left_bound += 1
            while(right_bound + 1 < mzs.shape[0] and mzs[right_bound + 1] - mzs[i] < mz_window): 
                right_bound += 1
            in_window.append((left_bound, right_bound + 1))
        
        return left_rt, right_rt, mzs, intensities, owner, in_window

    @staticmethod
    def create_scan_intensities(mzml_path, num_injection, schedule, mz_window):
        new_scans = []
        
        ms1s = (s for s in MZMLFile(mzml_path).scans if s.ms_level == 1)
        original_scans = sorted(ms1s, key=lambda s: s.rt_in_seconds, reverse=True)
            
        left_rt, right_rt, mzs, intensities, owner, in_window = (
            MatchingScan.interpolate_scan(original_scans[-1], original_scans[-2], mz_window)
        )
        
        for s in schedule:
            try: ms_level, rt = s.ms_level, s.rt
            except AttributeError: ms_level, rt = s
            
            if(len(original_scans) > 1 and original_scans[-2].rt_in_seconds < rt): 
                while(len(original_scans) > 1 and original_scans[-2].rt_in_seconds < rt): 
                    original_scans.pop()
                if(len(original_scans) > 1): 
                    left_rt, right_rt, mzs, intensities, owner, in_window = (
                        MatchingScan.interpolate_scan(original_scans[-1], 
                                                      original_scans[-2], 
                                                      mz_window
                                                     )
                        )
                
            if(ms_level > 1 or len(original_scans) < 2):
                new_scans.append(MatchingScan(num_injection, ms_level, rt, [], []))
            else:
                w = (rt - left_rt) / (right_rt - left_rt)
                weighted_intensities = (owner * (1 - w) * intensities
                                        + (1 - owner) * w * intensities)

                new_intensities = []
                new_intensities = [
                    np.sum(weighted_intensities[left_bound:right_bound]) 
                    for (left_bound, right_bound) in in_window
                ]
                new_scans.append(MatchingScan(num_injection, ms_level, rt, mzs, new_intensities))
        
        return new_scans

    @staticmethod
    def topN_nodes(mzml_path, num_injection, N, max_rt, scan_duration_dict,
                   mz_window=1E-10):
        ms_levels = itertools.cycle([1] + [2] * N)
        scan_times = itertools.accumulate(
            (scan_duration_dict[ms_level] for ms_level in copy.deepcopy(ms_levels)), 
            initial=0
        )
        return (
            MatchingScan.create_scan_intensities(
                mzml_path, 
                num_injection, 
                zip(ms_levels, itertools.takewhile(lambda t: t < max_rt, scan_times)), 
                mz_window
            )
        )

    @staticmethod
    def env2nodes(env, num_injection):
        scans = (
            sorted(
                itertools.chain(*(s for _, s in env.controller.scans.items())), 
                key=lambda s: s.rt
            )
        )
        return [MatchingScan(num_injection, s.ms_level, s.rt, s.mzs, s.intensities) for s in scans]
        

class MatchingChem():
    def __init__(self, min_mz, max_mz, min_rt, max_rt, intensity=None):
        self.min_mz, self.max_mz = min_mz, max_mz
        self.min_rt, self.max_rt = min_rt, max_rt
        self.intensity = intensity
        
    def __repr__(self): 
        return (
            f"MatchingChem(min_mz={self.min_mz}, max_mz={self.max_mz}, "
            f"min_rt={self.min_rt}, max_rt={self.max_rt})"
        )
        
    def get_key(self): 
        return (self.min_mz, self.max_mz, self.min_rt, self.max_rt)
        
    def __eq__(self, other): 
        return (
            isinstance(other, type(self)) 
            and all(math.isclose(a, b) for a, b in zip(self.get_key(), other.get_key()))
        )
    
    def __hash__(self): 
        return self.get_key().__hash__()

    @staticmethod
    def mzmine2nodes(box_file_path):
        with open(box_file_path, "r") as f:
            fields = f.readline().split(",")
            min_mz = [i for i, fd in enumerate(fields) if
                      fd.strip().endswith("Peak m/z min")]
            max_mz = [i for i, fd in enumerate(fields) if
                      fd.strip().endswith("Peak m/z max")]
            min_rt = [i for i, fd in enumerate(fields) if
                      fd.strip().endswith("Peak RT start")]
            max_rt = [i for i, fd in enumerate(fields) if
                      fd.strip().endswith("Peak RT end")]

            if (len(min_mz) == 0):
                print(
                    "No minimum m/z could be found for mzmine box file!")
            if (len(max_mz) == 0):
                print(
                    "No maximum m/z could be found for mzmine box file!")
            if (len(min_rt) == 0):
                print(
                    "No minimum peak rt could be found for mzmine box file!")
            if (len(max_rt) == 0):
                print(
                    "No maximum peak rt could be found for mzmine box file!")

            records = []
            for ln in f:
                sp = ln.split(",")
                for i, j, k, l in zip(min_mz, max_mz, min_rt, max_rt):
                    records.append(
                        MatchingChem(
                            float(sp[i]), 
                            float(sp[j]), 
                            float(sp[k]) * 60, 
                            float(sp[l]) * 60
                        )
                    )
            return records

    @staticmethod
    def env2nodes(env, isolation_width):
        """Assumes no noise."""
        print(f"Num chems in MS: {len(env.mass_spec.chemicals)}")
        print(f"First ten chems in MS: {env.mass_spec.chemicals[:10]}")
        controller = env.controller
        try:
            roi_builder = controller.roi_builder
        except AttributeError:
            errmsg = (
                "Currently only supports converting runs of RoI-based controllers, "
                "should manually build RoIs on other envs later"
            )
            raise NotImplementedError(errmsg)
            
        live_roi, dead_roi, junk_roi = (
            roi_builder.live_roi, roi_builder.dead_roi, roi_builder.junk_roi
        )
        print(f"roi lengths: {len(live_roi), len(dead_roi), len(junk_roi)}")
        return [
            MatchingChem(
                min(roi.mz_list), 
                max(roi.mz_list), 
                min(roi.rt_list), 
                max(roi.rt_list)
            ) 
            for roi in itertools.chain(live_roi, dead_roi, junk_roi)
        ]

    def update_chem_intensity(self, mzs, intensities):
        left, right = bisect.bisect_left(
            mzs, self.min_mz), bisect.bisect_right(mzs, self.max_mz)
        self.intensity = max(intensities[i] for i in
                             range(left, right)) if right - left > 0 else 0


class Matching():
    def __init__(self, scans_list, chems_list, matching, nx_graph=None):
        self.scans_list, self.chems_list = scans_list, chems_list
        self.matching = matching
        self.nx_graph = nx_graph
        
    def __len__(self): 
        return sum(type(k) == MatchingChem for k in self.matching.keys())
    
    def __iter__(self): 
        return ((ch, s) for ch, s in self.matching.items() if type(ch) == MatchingChem)

    @staticmethod
    def make_edges(scans, chems, intensity_threshold):
        active_chems, edges = [], {ch: [] for ch in chems}
        rt_intervals = intervaltree.IntervalTree()
        for ch in chems:
            rt_intervals.addi(ch.min_rt, ch.max_rt + 1E-12, ch)

        seen_intersected = set()
        seen_intensities = {}
        seen_active = set()
        seen_edges = set()

        for s in scans:
            if(s.ms_level == 1):
                intersected = sorted(
                    (interval.data for interval in rt_intervals.at(s.rt)), 
                    key=lambda ch: ch.max_rt, reverse=True
                )
                seen_intersected |= set(intersected)
                for ch in intersected:
                    ch.update_chem_intensity(s.mzs, s.intensities)
                    seen_intensities[ch] = max(ch.intensity,
                                               seen_intensities.get(ch, 0))
                active_chems = [ch for ch in intersected if
                                ch.intensity >= intensity_threshold]
                seen_active |= set(active_chems)
            elif (s.ms_level == 2):
                while (len(active_chems) > 0 and active_chems[
                        -1].max_rt < s.rt):
                    active_chems.pop()
                for ch in active_chems:
                    edges[ch].append((s, ch.intensity))
                    seen_edges.add(ch)
        for ch in chems:
            ch.intensity = None  # clear temporary value

        print(f"|chems| BEFORE COLLAPSE: {len(chems)}")
        print(f"num chems without edges:"
              f" {sum(1 for _, ls in edges.items() if ls == [])}")
        print(f"chems without edges:"
              f" {[ch for ch, ls in edges.items() if ls == []][:10]}")
        print(
            f"|E| BEFORE COLLAPSE: {sum(len(ls) for _, ls in edges.items())}")
        print(f"num intersected: {len(seen_intersected)}")
        print(f"min intensity: {min(v for _, v in seen_intensities.items())}")
        # from collections import Counter
        # print(f"intensity counts:"
        #       f" {Counter(v for _, v in seen_intensities.items())}")
        print(f"zero intensity count:"
              f" {sum(v == 0 for _, v in seen_intensities.items())}")
        print(f"num active: {len(seen_active)}")
        print(f"num with edges: {len(seen_edges)}")

        print()

        return edges

    @staticmethod
    def collapse_chems(scans_list, chems_list, edges_list):
        scans = set(s for ls in scans_list for s in ls)
        chems = set(ch for ls in chems_list for ch in ls)
        edges = {
            ch : [e for E in edges_list for e in E.get(ch, [])] 
            for ch in chems 
            if any(ch in E for E in edges_list)
        }
        print(f"|chems| AFTER COLLAPSE: {len(chems)}")
        print(f"chems without edges"
              f": {sum(1 for ch, E in edges.items() if E == [])}")
        print(f"|E| AFTER COLLAPSE: {sum(len(ls) for _, ls in edges.items())}")
        return scans, chems, edges

    @staticmethod
    def build_nx_graph(scans, chems, edges):
        G = nx.Graph()
        for s in scans:
            if (s.ms_level > 1):
                G.add_node(s, bipartite=0)
        for ch in chems:
            G.add_node(ch, bipartite=1)
        for ch, ls in edges.items():
            for s, intensity in ls:
                G.add_edge(s, ch, weight=intensity)
        return G

    @staticmethod
    def unweighted_matching(G):
        top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
        return nx.bipartite.matching.hopcroft_karp_matching(G, top_nodes)

    @staticmethod
    def weighted_matching(G):
        return dict(nx.algorithms.matching.max_weight_matching(G))

    @staticmethod
    def multi_schedule2graph(scans_list, chems_list, intensity_threshold, weighted=True):
        edges_list = [
            Matching.make_edges(scans, chems, intensity_threshold) 
            for scans, chems in zip(scans_list, chems_list)
        ]
        scans, chems, edges = Matching.collapse_chems(scans_list, chems_list, edges_list)
        G = Matching.build_nx_graph(scans, chems, edges)
        matching = Matching.weighted_matching(
            G) if weighted else Matching.unweighted_matching(G)
        return Matching(scans_list, chems_list, matching, nx_graph=G)

    @staticmethod
    def schedule2graph(scans, chems, intensity_threshold, weighted=True):
        return Matching.multi_schedule2graph(
            [scans], [chems], intensity_threshold, weighted=weighted
        )
        
    def make_schedules(self, isolation_width):
        id_count, precursor_id = INITIAL_SCAN_ID, -1
        schedules_list = [[] for i in range(len(self.scans_list))]
        for (i, scans) in enumerate(self.scans_list):
            for s in scans:
                if (s.ms_level == 1):
                    precursor_id = id_count
                    schedules_list[i].append(
                        get_default_scan_params(scan_id=precursor_id))
                elif (s.ms_level == 2):
                    if (s in self.matching):
                        ch = self.matching[s]
                        # TODO: decide whether targeting the bounds of
                        #  a picked box is a good idea??
                        target_mz = (ch.min_mz + ch.max_mz) / 2
                    else:
                        target_mz = 100.0
                    schedules_list[i].append(
                        get_dda_scan_param(
                            target_mz, 
                            0.0, 
                            precursor_id, 
                            isolation_width, 
                            0.0, 
                            0.0, 
                            scan_id=id_count
                        )
                    )
                id_count += 1
        return schedules_list
