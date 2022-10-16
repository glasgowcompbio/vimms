import os
import bisect
import copy
import itertools
import math
import re
from statistics import mean
from collections import defaultdict
from operator import attrgetter

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
    def __init__(self, injection_num, ms_level, rt, mzs, intensities):
        self.injection_num = injection_num
        self.ms_level = ms_level
        self.rt = rt
        self.mzs = mzs
        self.intensities = intensities
        
    def __repr__(self): 
        return (
            f"MatchingScan(injection_num={self.injection_num}, "
            f"ms_level={self.ms_level}, rt={self.rt})"
        )
        
    def __eq__(self, other): 
        return (
            isinstance(other, type(self)) 
            and self.injection_num == other.injection_num 
            and self.ms_level == other.ms_level 
            and math.isclose(self.rt, other.rt)
        )
        
    def __hash__(self): 
        return (self.injection_num, self.ms_level, self.rt).__hash__()
        
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
    def create_scan_intensities(mzml_path, injection_num, schedule, mz_window):
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
                new_scans.append(MatchingScan(injection_num, ms_level, rt, [], []))
            else:
                w = (rt - left_rt) / (right_rt - left_rt)
                weighted_intensities = (owner * (1 - w) * intensities
                                        + (1 - owner) * w * intensities)

                new_intensities = []
                new_intensities = [
                    np.max(weighted_intensities[left_bound:right_bound]) 
                    for (left_bound, right_bound) in in_window
                ]
                new_scans.append(MatchingScan(injection_num, ms_level, rt, mzs, new_intensities))
        
        return new_scans

    @staticmethod
    def topN_times(N, max_rt, scan_duration_dict):
        ms_levels = itertools.cycle([1] + [2] * N)
        scan_times = itertools.accumulate(
            (scan_duration_dict[ms_level] for ms_level in copy.deepcopy(ms_levels)), 
            initial=0
        )
        return zip(ms_levels, itertools.takewhile(lambda t: t < max_rt, scan_times))

    @staticmethod
    def env2nodes(env, injection_num):
        scans = (
            sorted(
                itertools.chain(*(s for _, s in env.controller.scans.items())), 
                key=lambda s: s.rt
            )
        )
        return [MatchingScan(injection_num, s.ms_level, s.rt, s.mzs, s.intensities) for s in scans]
        
    @staticmethod
    def topN_nodes(mzml_path, injection_num, N, max_rt, scan_duration_dict, mz_window=1E-10):
        topN_times = MatchingScan.topN_times(N, max_rt, scan_duration_dict)
        return (
            MatchingScan.create_scan_intensities(
                mzml_path, 
                injection_num, 
                topN_times, 
                mz_window
            )
        )

        

class MatchingChem():
    def __init__(self, chem_id, min_mz, max_mz, min_rt, max_rt, intensity=None):
        self.id = chem_id
        self.min_mz, self.max_mz = min_mz, max_mz
        self.min_rt, self.max_rt = min_rt, max_rt
        self.intensity = intensity
        
    def __repr__(self): 
        return (
            f"MatchingChem(min_mz={self.min_mz}, max_mz={self.max_mz}, "
            + f"min_rt={self.min_rt}, max_rt={self.max_rt})"
        )
        
    def __eq__(self, other): 
        return (
            isinstance(other, type(self)) 
            and self.id == other.id
        )
    
    def __hash__(self): 
        return self.id.__hash__()

    @staticmethod
    def mzmine2nodes(box_file_path, box_order):
        include = [
                "status",
                "RT start",
                "RT end",
                "m/z min",
                "m/z max"
        ]
    
        box_order = [
            ".".join(os.path.basename(fname).split(".")[:-1]) for fname in box_order
        ]
        chems_list = [[] for _ in box_order]
        
        with open(box_file_path, "r") as f:
            headers = f.readline().split(",")
            pattern = re.compile(r"(.*)\.mzML filtered Peak ([a-zA-Z/]+( [a-zA-Z/]+)*)")
            
            indices = defaultdict(dict)
            for i, h in enumerate(headers):
                m = pattern.match(h)
                if(not m is None):
                    indices[m.group(1)][m.group(2)] = i
            
            for i, ln in enumerate(f):
                split = ln.split(",")
                
                for j, fname in enumerate(box_order):
                    inner = indices[fname]
                    status = split[inner["status"]].upper()
                    if(status == "DETECTED" or status == "ESTIMATED"):
                        chems_list[j].append(
                            MatchingChem(
                                i,
                                float(split[inner["m/z min"]]),
                                float(split[inner["m/z max"]]),
                                60 * float(split[inner["RT start"]]),
                                60 * float(split[inner["RT end"]])
                            )
                        )
            
        return chems_list

    @staticmethod
    def env2nodes(env, isolation_width, chem_ids=None):
        """Assumes chem_ids (if provided) will tell you when chemicals are identical
           based on the hash."""
        print(f"Num chems in MS: {len(env.mass_spec.chemicals)}")
        print(f"First ten chems in MS: {env.mass_spec.chemicals[:10]}")
        if(chem_ids is None): chem_ids = {}
        min_id = max(v for _, v in chem_ids.items())
        
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
        
        chems = []
        for roi in itertools.chain(live_roi, dead_roi):
            if(roi in chem_ids):
                chem_id = chem_ids[roi]
            else:
                chem_id = min_id
                min_id += 1
            
            chems.append(
                MatchingChem(
                    chem_id,
                    min(roi.mz_list), 
                    max(roi.mz_list), 
                    min(roi.rt_list), 
                    max(roi.rt_list)
                )
            )
        
        return chems, chem_ids

    def update_chem_intensity(self, mzs, intensities):
        left = bisect.bisect_left(mzs, self.min_mz) 
        right = bisect.bisect_right(mzs, self.max_mz)
        self.intensity = max(
            intensities[i] 
            for i in range(left, right)
        ) if right - left > 0 else 0


class Matching():
    UNWEIGHTED = 0
    TWOSTEP = 1

    def __init__(self, scans_list, chems_list, matching, nx_graph=None, aux_graph=None):
        self.scans_list, self.chems_list = scans_list, chems_list
        self.matching = matching
        self.nx_graph = nx_graph
        self.aux_graph = aux_graph
        
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
                    key=attrgetter("max_rt"), reverse=True
                )
                seen_intersected |= set(intersected)
                for ch in intersected:
                    ch.update_chem_intensity(s.mzs, s.intensities)
                    seen_intensities[ch] = max(ch.intensity,
                                               seen_intensities.get(ch, 0)
                                               )
                active_chems = [
                    ch for ch in intersected 
                    if ch.intensity >= intensity_threshold
                ]
                seen_active |= set(active_chems)
            
            elif (s.ms_level == 2):
                while (len(active_chems) > 0 and active_chems[-1].max_rt < s.rt):
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
        print(
            f"< 5000 intensity count: {sum(v < 5000 for _, v in seen_intensities.items())}"
        )
        print(f"num active: {len(seen_active)}")
        print(f"num with edges: {len(seen_edges)}")

        print()

        return edges

    @staticmethod
    def collapse_chems(scans_list, chems_list, edges_list, edge_limit=None):
        scans = set(s for ls in scans_list for s in ls)
        chems = set(ch for ls in chems_list for ch in ls)
        
        print(f"EDGE_LIMIT: {edge_limit}")
        
        if(not edge_limit is None):
            edges_list = [
                {
                    ch : sorted(ls, key=lambda e: e[1], reverse=True)[:edge_limit]
                    for ch, ls in E.items()
                }
                for E in edges_list
            ]
            
        edges = {
            ch : [e for E in edges_list for e in E.get(ch, [])] 
            for ch in chems 
            if any(ch in E for E in edges_list)
        }
        
        print(print(f"|scans| AFTER COLLAPSE: {len(scans)}"))
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
                G.add_edge(s, ch, weight=-intensity)
        return G
    
    @staticmethod
    def _make_graph(scans_list, chems_list, intensity_threshold, edge_limit=None):
        edges_list = [
            Matching.make_edges(scans, chems, intensity_threshold) 
            for scans, chems in zip(scans_list, chems_list)
        ]
        scans, chems, edges = Matching.collapse_chems(
            scans_list, 
            chems_list, 
            edges_list,
            edge_limit=edge_limit
        )
        return Matching.build_nx_graph(scans, chems, edges)

    @staticmethod
    def unweighted_matching(G):
        top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
        return nx.bipartite.matching.hopcroft_karp_matching(G, top_nodes)

    @staticmethod
    def two_step_weighted_matching(scans_list, 
                                   chems_list, 
                                   intensity_threshold, 
                                   G,
                                   edge_limit=None):

        first_match = Matching.unweighted_matching(G)
        new_chems = set(v for _, v in first_match.items())
        
        new_chems_list = [
            [ch for ch in chem_list if ch in new_chems]
            for chem_list in chems_list
        ]
        
        aux_graph = Matching._make_graph(
            scans_list, 
            new_chems_list, 
            intensity_threshold,
            edge_limit=edge_limit
        )
        top_nodes = {n for n, d in aux_graph.nodes(data=True) if d["bipartite"] == 0}
        return (
            nx.bipartite.matching.minimum_weight_full_matching(aux_graph, top_nodes),
            aux_graph
        )

    @staticmethod
    def multi_schedule2graph(scans_list, 
                             chems_list, 
                             intensity_threshold, 
                             edge_limit=None, 
                             weighted=1):

        G = Matching._make_graph(
            scans_list, 
            chems_list, 
            intensity_threshold,
            edge_limit=edge_limit
        )
        
        aux_graph = None
        if(weighted == Matching.TWOSTEP):
            matching, aux_graph = Matching.two_step_weighted_matching(
                scans_list, 
                chems_list, 
                intensity_threshold, 
                G
            )
        else:
            matching = Matching.unweighted_matching(G)
        
        return Matching(scans_list, chems_list, matching, nx_graph=G)

    @staticmethod
    def schedule2graph(scans, chems, intensity_threshold, edge_limit=None, weighted=1):
        return Matching.multi_schedule2graph(
            [scans], [chems], intensity_threshold, edge_limit=None, weighted=weighted
        )
        
    def make_schedules(self, isolation_width):
        id_count, precursor_id = INITIAL_SCAN_ID, -1
        schedules_list = [[] for _ in self.scans_list]
        for (i, scans) in enumerate(self.scans_list):
            for s in scans:
                if (s.ms_level == 1):
                    precursor_id = id_count
                    schedules_list[i].append(get_default_scan_params(scan_id=precursor_id))
                elif (s.ms_level == 2):
                    if (s in self.matching):
                        ch = self.matching[s]
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
