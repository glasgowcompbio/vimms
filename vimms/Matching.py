import os
import bisect
import copy
import itertools
import math
import re
import datetime
from statistics import mean
from collections import defaultdict
from operator import attrgetter

import intervaltree
import networkx as nx
import numpy as np
from mass_spec_utils.data_import.mzml import MZMLFile

# TODO: scans from mzml function
# TODO: bin sort version
# TODO: constraint programming/stable marriage versions??
# TODO: could chemical collapse with arbitrary exclusion condition, for when
#  chems_list doesn't have same items e.g. with aligner
# TODO: MatchingChem.env2nodes could split chemicals on RoI bounds to have
#  a slightly more accurate view of simulated chemicals
# TODO: MatchingChem.env2nodes could also work on non-RoI controllers
from vimms.Common import (
    POSITIVE, get_default_scan_params, INITIAL_SCAN_ID, get_dda_scan_param
)
from vimms.Roi import RoiBuilderParams
from vimms.Chemicals import ChemicalMixtureFromMZML
from vimms.Box import GenericBox
from vimms.PeakPicking import MZMineParams


class MatchingLog():
    """
        Holds perishable bits of matching logging information useful to keep
        track of like time elapsed for different parts of the process.
    """

    def __init__(self):
        self.start_scan, self.end_scan = None, None
        self.start_chem, self.end_chem = None, None
        self.start_matching, self.end_matching = None, None
        self.start_assign, self.end_assign = None, None
        self.recursive_scan_counts = None
        self.matching_report = {}
    
    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.get_report().items())
        
    def get_report(self, keys=None):
        if(keys is None):
            keys = [
                "matching_size", "chem_count", "scan_count", "edge_count",
                "chems_above_threshold", "start_scan", "end_scan", 
                "start_chem", "end_chem", "start_matching", "end_matching",
                "start_assign", "end_assign"
            ]
            
        report = {}
        for k in keys:
            if(hasattr(self, k)):
                report[k] = getattr(self, k)
            elif(k in self.matching_report):
                report[k] = self.matching_report[k]
            else:
                raise KeyError(
                    f"{k} not found in logger. Maybe you forgot to run matching_report?"
                )
                
        return report

    def summarise(self, keys=None):
        for k, v in get_report(keys=keys).items():
            print(f"{k}: {v}")

class MatchingScan():
    def __init__(self, scan_idx, injection_num, ms_level, rt, mzs, intensities):
        self.scan_idx = scan_idx
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

    @staticmethod
    def create_scan_intensities(mzml_path,
                                injection_num,
                                schedule,
                                ionisation_mode,
                                mz_window,
                                roi_params=None,
                                log=None):
        
        if(not log is None):
            log.start_scan = datetime.datetime.now()
        
        if(roi_params is None):
            roi_params = RoiBuilderParams(
                min_roi_intensity=0,
                at_least_one_point_above=0,
                min_roi_length=2
            )
        
        cm = ChemicalMixtureFromMZML(mzml_path, roi_params=roi_params)
        chems = cm.sample(None, 1, source_polarity=ionisation_mode)
        rt_intervals = intervaltree.IntervalTree()
        for ch in chems:
            rt_intervals.addi(ch.rt, ch.chromatogram.raw_max_rt + 1E-12, ch)
        
        new_scans = []
        for s_idx, s in enumerate(schedule):
            ms_level, rt = s
            
            if(ms_level > 1):
                new_scans.append(MatchingScan(s_idx, injection_num, ms_level, rt, [], []))
            else:
                mzs, intensities = [], []
                for inv in rt_intervals.at(rt):
                    ch = inv.data
                    mzs.append(
                        np.mean(ch.chromatogram.raw_mzs) +
                        ch.chromatogram.get_relative_mz(rt - ch.rt)
                    )
                    intensities.append(
                        ch.max_intensity
                        * ch.chromatogram.get_relative_intensity(rt - ch.rt)
                    ) # could need isotopes ?
                
                mz_idxes = np.argsort(mzs)
                mzs, intensities = np.array(mzs)[mz_idxes], np.array(intensities)[mz_idxes]
                new_scans.append(
                        MatchingScan(s_idx, injection_num, ms_level, rt, mzs, intensities)
                    )
                    
        if(not log is None):
            log.end_scan = datetime.datetime.now()            
        
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
        return [
            MatchingScan(s_idx, injection_num, s.ms_level, s.rt, s.mzs, s.intensities) 
            for s_idx, s in enumerate(scans)
        ]
        
    @staticmethod
    def topN_nodes(mzml_path,
                   injection_num,
                   N,
                   max_rt,
                   scan_duration_dict,
                   ionisation_mode,
                   mz_window=1E-10):
        
        topN_times = MatchingScan.topN_times(N, max_rt, scan_duration_dict)
        return (
            MatchingScan.create_scan_intensities(
                mzml_path, 
                injection_num, 
                topN_times,
                ionisation_mode,
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
    def boxfile2nodes(reader, box_file_path, box_order, log=None):
        if(not log is None):
            log.start_chem = datetime.datetime.now()
    
        box_order = [
            ".".join(os.path.basename(fname).split(".")[:-1]) for fname in box_order
        ]
        chems_list = [[] for _ in box_order]
                        
        fs_names, line_ls = reader.read_aligned_csv(box_file_path)
        for i, (_, mzml_fields) in enumerate(line_ls):
            row = []
            for j, fname in enumerate(box_order):
                inner = mzml_fields[fname]
                status = inner["status"].upper()
                if(status == "DETECTED" or status == "ESTIMATED"):
                    chems_list[j].append(
                        MatchingChem(
                            i,
                            float(inner["m/z min"]),
                            float(inner["m/z max"]),
                            reader.RT_FACTOR * float(inner["RT start"]),
                            reader.RT_FACTOR * float(inner["RT end"])
                        )
                    )
                    
        if(not log is None):
            log.end_chem = datetime.datetime.now()
            
        return chems_list

    @staticmethod
    def mzmine2nodes(box_file_path, box_order):
        return MatchingChem.boxfile2nodes(MZMineParams, box_file_path, box_order)

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
        
        self.intensity = (
            max(intensities[i] for i in range(left, right)) 
            if (right - left > 0) 
            else 0
        )
        
    def reset(self):
        self.intensity = 0


class Matching():
    #Matching weight modes
    UNWEIGHTED = 0
    TWOSTEP = 1
    
    #Full_assignment_strategy modes
    MATCHING_ONLY = 0
    RECURSIVE_ASSIGNMENT = 1
    NEAREST_ASSIGNMENT = 2

    def __init__(self, 
                 scans_list, 
                 chems_list, 
                 matching,
                 weighted,
                 intensity_threshold,
                 nx_graph=None, 
                 aux_graph=None,
                 full_assignment_strategy=1):
                 
        self.scans_list, self.chems_list = scans_list, chems_list
        self.matching = matching
        self.weighted = weighted
        self.intensity_threshold = intensity_threshold
        self.nx_graph = nx_graph
        self.aux_graph = aux_graph
        self.full_assignment_strategy = full_assignment_strategy
        self.full_assignment = []
        self.log = None
        
    def __len__(self): 
        return sum(type(k) == MatchingChem for k in self.matching.keys())
    
    def __iter__(self): 
        return ((ch, s) for ch, s in self.matching.items() if type(ch) == MatchingChem)

    @staticmethod
    def make_edges(scans, chems, intensity_threshold):
        active_chems, edges = [], {ch: [] for ch in chems}
        rt_intervals = intervaltree.IntervalTree()
        for ch in chems:
            ch.reset() #Clear temporary value
            rt_intervals.addi(ch.min_rt, ch.max_rt + 1E-12, ch)

        for s in scans:
            
            if(s.ms_level == 1):
                intersected = sorted(
                    (interval.data for interval in rt_intervals.at(s.rt)), 
                    key=attrgetter("max_rt"), reverse=True
                )
                
                active_chems = []
                for ch in intersected:
                    ch.update_chem_intensity(s.mzs, s.intensities)
                    if(ch.intensity >= intensity_threshold):
                        active_chems.append(ch)
            
            elif (s.ms_level == 2):
                while (len(active_chems) > 0 and active_chems[-1].max_rt < s.rt):
                    active_chems.pop()
                for ch in active_chems:
                    edges[ch].append((s, ch.intensity))
        
        return edges

    @staticmethod
    def collapse_chems(scans_list, chems_list, edges_list, edge_limit=None):
        scans = set(s for ls in scans_list for s in ls)
        chems = set(ch for ls in chems_list for ch in ls)
        
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
    def two_step_weighted_matching(G):

        first_match = Matching.unweighted_matching(G)
        new_chems = set(v for v in first_match.keys() if type(v) == MatchingChem)
        if(len(new_chems) < 1):
            return {}, None
        
        aux_G = copy.copy(G)
        chems = {n for n, d in aux_G.nodes(data=True) if d["bipartite"] == 1}
        for ch in chems:
            if(not ch in new_chems):
                aux_G.remove_node(ch)
        
        top_nodes = {n for n, d in aux_G.nodes(data=True) if d["bipartite"] == 0}
        return (
            nx.bipartite.matching.minimum_weight_full_matching(aux_G, top_nodes),
            aux_G
        )
        
    def _assign_remaining_scans(self):
        '''
        Turn matching into a total assignment of scans.
        '''
        
        if(self.weighted == Matching.TWOSTEP):
            G = self.aux_graph
            matching_f = lambda G: Matching.two_step_weighted_matching(G)[0]
        else:
            G = self.nx_graph
            matching_f = Matching.unweighted_matching
        
        DEFAULT_VAL = 75.0 #change this to real default
        self.full_assignment = [
            [DEFAULT_VAL for s in scans] for scans in self.scans_list 
        ]
        
        for s in self.matching:
            if(type(s) == MatchingScan):
                ch = self.matching[s]
                self.full_assignment[s.injection_num][s.scan_idx] = (ch.min_mz + ch.max_mz) / 2
                
        if(self.full_assignment_strategy == self.RECURSIVE_ASSIGNMENT):
            aux_G = copy.deepcopy(G)
            chems = {n for n, d in aux_G.nodes(data=True) if d["bipartite"] == 1}
            matching = self.matching
            for ch in chems:
                if(not ch in matching):
                    aux_G.remove_node(ch)
                    
            remove = [n for n, degree in aux_G.degree() if degree < 1]
            for n in remove: aux_G.remove_node(n)
            
            if(not self.log is None):
                self.log.recursive_scan_counts = []
            
            scans = {n for n, d in aux_G.nodes(data=True) if d["bipartite"] == 0}
            while(len(scans) > 0):
                if(not self.log is None):
                    self.log.recursive_scan_counts.append(len(scans))
                
                for s in matching:
                    if(type(s) == MatchingScan):
                        ch = matching[s]
                        self.full_assignment[s.injection_num][s.scan_idx] = (
                            (ch.min_mz + ch.max_mz) / 2
                        )
                        aux_G.remove_node(s)
                
                scans = {n for n, d in aux_G.nodes(data=True) if d["bipartite"] == 0}
                matching = matching_f(aux_G)
                
        elif(self.full_assignment_strategy == self.NEAREST_ASSIGNMENT):
            for scans in self.full_assignment:
                last_i = -1
            
                for i in range(len(scans)):
                    if(not math.isclose(scans[i], DEFAULT_VAL)):
                        for j in range(i): scans[j] = scans[i]
                        last_i = i
                        break
                        
                for i in range(last_i + 1, len(scans)):
                    if(not math.isclose(scans[i], DEFAULT_VAL)):
                        mid = 1 + (last_i + i) // 2
                        for j in range(last_i + 1, mid): scans[j] = scans[last_i] 
                        for j in range(mid, i): scans[j] = scans[i]
                        last_i = i
                        
                for i in range(last_i + 1, len(scans)):
                    scans[i] = scans[last_i]

    def full_assignment(self, full_assignment_strategy=None):
        if(not full_assignment_strategy is None): 
            self.full_assignment_strategy = full_assignment_strategy
            
        self._assign_remaining_scans()

    @staticmethod
    def multi_schedule2graph(scans_list, 
                             chems_list, 
                             intensity_threshold, 
                             edge_limit=None, 
                             weighted=1,
                             full_assignment_strategy=1,
                             log=None):
                             
        if(not log is None):
            log.start_matching = datetime.datetime.now()

        G = Matching._make_graph(
            scans_list, 
            chems_list, 
            intensity_threshold,
            edge_limit=edge_limit
        )
        
        aux_graph = None
        if(weighted == Matching.TWOSTEP):
            matching, aux_graph = Matching.two_step_weighted_matching(G)
        else:
            matching = Matching.unweighted_matching(G)
        
        matching = Matching(
            scans_list, 
            chems_list, 
            matching,
            weighted,
            intensity_threshold,
            nx_graph=G,
            aux_graph=aux_graph,
            full_assignment_strategy=full_assignment_strategy
        )
        
        if(not log is None):
            log.end_matching = datetime.datetime.now()
            log.start_assign = datetime.datetime.now()

        matching._assign_remaining_scans()
        
        if(not log is None):
            log.end_assign = datetime.datetime.now()
            matching.log = log
        
        return matching

    @staticmethod
    def schedule2graph(scans, 
                       chems, 
                       intensity_threshold, 
                       edge_limit=None, 
                       weighted=1,
                       full_assignment_strategy=1):
        
        return Matching.multi_schedule2graph(
            [scans], 
            [chems], 
            intensity_threshold, 
            edge_limit=edge_limit, 
            weighted=weighted,
            full_assignment_strategy=full_assignment_strategy
        )
        
    def matching_report(self):
        # "bipartite" is 0 if scan, 1 if chem
        G = self.nx_graph
        all_scans = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
        all_chems = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]
        
        report = {}
        report["matching_size"] = len(self)
        report["scan_count"] = len(all_scans)
        report["chem_count"] = len(all_chems)
        report["edge_count"] = len(G.edges)
        report["uncollapsed_chem_count"] = [len(ls) for ls in self.chems_list]
        
        report["uncollapsed_chems_appearing"] = [0] * len(self.chems_list)
        for i, ls in enumerate(self.chems_list):
            for ch in ls:
                appears = any(
                    s.ms_level == 1 and ch.min_rt < s.rt and s.rt < ch.max_rt
                    for s in all_scans
                )
                if(appears):
                    report["uncollapsed_chems_appearing"][i] += 1
        
        chems2edges_ls = [{ch: [] for ch in ls} for ls in self.chems_list]
        for e0, e1, edgedata in G.edges(data=True):
            ch = e1 if type(e1) is MatchingChem else e0
            for inj in chems2edges_ls:
                if(ch in inj):
                    inj[ch].append(edgedata["weight"])
         
        report["uncollapsed_chems_with_edges"] = [
            sum(len(w_ls) > 0 for _, w_ls in inj.items())
            for inj in chems2edges_ls
        ]
        report["uncollapsed_chems_above_zero"] = [
            sum(any(w > 0 for w in w_ls) for _, w_ls in inj.items())
            for inj in chems2edges_ls
        ]
        report["uncollapsed_chems_above_threshold"] = [
            sum(any(w > self.intensity_threshold for w in w_ls) for _, w_ls in inj.items())
            for inj in chems2edges_ls
        ]
        
        chems2edges = {ch : [] for ch in all_chems}
        for ch, e_ls in chems2edges.items():
            for inj in chems2edges_ls:
                e_ls.extend(inj.get(ch, []))
        
        report["chems_with_edges"] = sum(
            len(w_ls) > 0 for ch, w_ls in chems2edges.items()
        )
        report["chems_above_zero"] = sum(
            any(w > 0 for w in w_ls) for ch, w_ls in chems2edges.items()
        )
        report["chems_above_threshold"] = sum(
            any(w > self.intensity_threshold for w in w_ls) for ch, w_ls in chems2edges.items()
        )
        
        if(not self.log is None):
            self.log.matching_report = report
        return report    

    @staticmethod
    def make_matching(fullscan_paths,
                      times_list,
                      aligned_reader,
                      aligned_file,
                      ionisation_mode,
                      intensity_threshold,
                      mz_window=10,
                      roi_params=None,
                      edge_limit=None,
                      weighted=1,
                      full_assignment_strategy=1,
                      logging=True):
        """
            Convenience method to make a matching from provided scan times/levels
            and an aligned file of inclusion boxes.
            
            Args:
                fullscan_paths: List of paths to fullscan .mzMLs, one per injection,
                  to seed scan data for that injection. The filename should match
                  identifiers in the aligned file.
                times_list: List of lists (scan_level, rt) pairs, one list per 
                  injection, to generate new scan times.
                aligned_reader: Reader object to read aligned_file.
                aligned_file: File containing aligned inclusion boxes.
                ionisation_mode: Source polarity of instrument.
                  Either Vimms.Common.POSITIVE or Vimms.Common.NEGATIVE.
                intensity_threshold: Minimum intensity for an edge to be retained
                  in the constructed graph.
                mz_window: m/z tolerance in ppm for determining whether two intensity 
                  readings are at the same value when interpolating scan intensities.
                roi_params: A RoiBuilderParams object to use when interpolating scan
                intensities.
                edge_limit: If given a non-None integer value, each vertex in the
                  constructed graph will be pruned to have degree of no more than
                  edge_limit, for performance reasons. Prefers to keep highest
                  intensity edges.
                weighted: Choice of method to decide preferences between weighted
                  edges.
                full_assignment_strategy: Choice of method to assign leftover scans
                  not in the matching.
                logging: Bool for whether to enable logging.
            
            Returns: A Matching object.
        """
        
        log = MatchingLog() if logging else None
        
        scans_list = []
        for i, fs in enumerate(fullscan_paths):
            new_scans = MatchingScan.create_scan_intensities(
                    fs,
                    i,
                    times_list[i],
                    ionisation_mode,
                    mz_window,
                    roi_params=roi_params,
                    log=log
            )
            scans_list.append(new_scans)
        
        chems_list = MatchingChem.boxfile2nodes(
            aligned_reader,
            aligned_file,
            fullscan_paths,
            log=log
        )
        
        matching = Matching.multi_schedule2graph(
            scans_list,
            chems_list,
            intensity_threshold,
            edge_limit=edge_limit,
            weighted=weighted,
            full_assignment_strategy=full_assignment_strategy,
            log=log
        )
        
        if(logging): matching.matching_report()
        
        return matching
    
    def make_schedules(self, isolation_width):
        id_count, precursor_id = INITIAL_SCAN_ID, -1
        schedules_list = [[] for _ in self.scans_list]
        rts_list = [[] for _ in self.scans_list]
        ms2_targets = itertools.chain(*self.full_assignment)
        
        for i, scans in enumerate(self.scans_list):
            for s in scans:
                target_mz = next(ms2_targets)
                if (s.ms_level == 1):
                    precursor_id = id_count
                    schedules_list[i].append(get_default_scan_params(scan_id=precursor_id))
                elif (s.ms_level == 2):
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
                rts_list[i].append(s.rt)
                id_count += 1
        
        return schedules_list, rts_list
        
    def make_inclusion_boxes(self, rt_width, mz_width):
        """
            Turn matching targets into inclusion boxes of a specified size that
            can be used by a DDA controller.
            
            Args:
                rt_width: inclusion box rt width in seconds.
                m/z width: inclusion box m/z width in ppm.
                
            Returns: A list of lists of inclusion boxes, one list per injection.
        """
        current_rt = 0.0
    
        box_lists = []
        for i, scans in enumerate(self.scans_list):
            boxes = []
            for s in scans:
                if(s.ms_level == 1):
                    current_rt = s.rt
                elif(s.ms_level == 2):
                    if(s in self.matching):
                        ch = self.matching[s]
                        mz = (ch.min_mz + ch.max_mz) / 2
                        abs_width = (mz_width / 1e6) * mz
                        boxes.append(
                            GenericBox(
                                current_rt - rt_width / 2,
                                current_rt + rt_width / 2,
                                mz - abs_width / 2,
                                mz + abs_width / 2
                            )
                        )
            box_lists.append(boxes)
            
        return box_lists