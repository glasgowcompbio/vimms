import os
import itertools
from collections import deque, OrderedDict
import multiprocessing
import json

from vimms.Common import POSITIVE, save_obj, load_obj
from vimms.Roi import RoiBuilderParams
from vimms.Chemicals import ChemicalMixtureFromMZML
from vimms.ChemicalSamplers import FixedMS2Sampler
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Controller import TopNController, AgentBasedController
from vimms.Agent import TopNDEWAgent
from vimms.Controller.roi import TopN_RoiController
from vimms.Box import BoxIntervalTrees, BoxGrid
from vimms.BoxManager import BoxManager, BoxConverter, BoxSplitter
from vimms.Controller.box import (
    TopNEXController, HardRoIExcludeController, IntensityRoIExcludeController,
    NonOverlapController, IntensityNonOverlapController
)
from vimms.Environment import Environment
from vimms.BoxVisualise import EnvPlotPickler
from vimms.Evaluation import pick_aligned_peaks, evaluate_real

class Shareable:
    def __init__(self, name, split=False):
        self.name = name
        self.split = split
        
    def add_shareable(self, params):
        if(self.name == "agent"):
            return {
                "agent" : TopNDEWAgent(**params)
            }
        elif(self.name == "grid"):
            return {
                **params,
                "grid" : BoxManager(
                    box_geometry = BoxGrid(),
                    box_splitter = BoxSplitter(split=self.split),
                    delete_rois=True
                )
            }

class ExperimentCase:

    GRID = Shareable("grid")
    SPLIT_GRID = Shareable("grid", split=True)

    controllers = OrderedDict([
        ("none", (None, None)),
        ("topn", (TopNController, None)),
        ("topn_roi", (TopN_RoiController, None)),
        ("topn_exclusion", (AgentBasedController, Shareable("agent"))),
        ("topnex", (TopNEXController, GRID)),
        ("hard_roi_exclusion", (HardRoIExcludeController, GRID)),
        ("intensity_roi_exclusion", (IntensityRoIExcludeController, SPLIT_GRID)),
        ("non_overlap", (NonOverlapController, GRID)),   
        ("intensity_non_overlap", (IntensityNonOverlapController, SPLIT_GRID))
    ])
        
    def __init__(self, 
                 controller_type,
                 fullscan_paths,
                 params,
                 name=None, 
                 grid_init=None,
                 pickle_env=False):
             
        self.name = name if not name is None else controller_type
        self.fullscan_paths = fullscan_paths
        self.datasets = []
        self.params = params
        self.grid_init = grid_init
        self.injection_num = 0
        self.pickle_env = pickle_env
        c = controller_type.replace(" ", "_").lower()
        
        try:
            self.controller, self.shared = self.controllers[c]
        except:
            error_msg = (
                "Not a recognised controller, please use one of:\n"
                + "\n".join(self.controllers.keys())
            )
            raise ValueError(error_msg)
            
    def _init_shareable(self):
        if(not self.grid_init is None):
            return {**self.params, "grid" : self.grid_init()}
        elif(not self.shared is None):
            return self.shared.add_shareable({**self.params})
        else:
            return self.params
    
    def run_controller(self, 
                       chems,
                       out_dir,
                       pbar, 
                       min_rt, 
                       max_rt, 
                       ionisation_mode,
                       scan_duration_dict):
        
        mzml_names = []
        params = self._init_shareable()
        
        for i, fs in enumerate(self.fullscan_paths):
            out_file = f"{self.name}_{i}.mzML"
            controller = self.controller(**params)
            dataset = load_obj(chems[fs])
            
            mass_spec = IndependentMassSpectrometer(
                ionisation_mode, 
                dataset, 
                None, 
                scan_duration=scan_duration_dict
            )
            
            env = Environment(
                mass_spec, 
                controller,
                min_rt,
                max_rt,
                progress_bar=pbar,
                out_dir=out_dir,
                out_file=out_file
            )    
            
            print(f"Outcome being written to: \"{out_file}\"")
            env.run()
            if(self.pickle_env):
                save_obj(
                    EnvPlotPickler(env), 
                    os.path.join(out_dir, "pickle", out_file.split(".")[0] + ".pkl")
                )
                
                try:
                    roi_builder = env.controller.roi_builder
                    live_roi, dead_roi, junk_roi = (
                        roi_builder.live_roi, roi_builder.dead_roi, roi_builder.junk_roi
                    )
                    rois = live_roi + dead_roi + junk_roi
                    save_obj(
                        rois,
                        os.path.join(out_dir, "pickle", out_file.split(".")[0] + "_rois.pkl")
                    )
                except AttributeError:
                    pass
                finally:
                    roi_builder = None
            
            mzml_names.append(
                (fs, os.path.join(out_dir, out_file))
            )
            
            del env
            del mass_spec
            del dataset
            
        return {self.name : mzml_names}
        
    def valid_controller(self):
        """
           Checks if controller throws an assertion error during exception
           which should mean its initialisation is invalid due to 
           e.g. parameters.
           
           Returns: Boolean indicating whether an error was thrown.
        """
        
        params = self._init_shareable()
        try:
            self.controller(**params)
            return True
        except AssertionError:
            return False
    
class Experiment:
    def __init__(self):
        self.out_dir = None
        self.chems = {}
        
        self.cases = []
        self.case_names = []
        self.case_mzmls = {}
        
        self.evaluators = {}
    
    def add_cases(self, cases):
        cases = list(cases)
        self.cases.extend(cases)
        self.case_names.extend(
            case.name for case in cases
        )
        
    @staticmethod
    def _gen_chems(fs, ionisation_mode, out_dir, point_noise_threshold, chem_noise_threshold):
        print(f"Generating chemicals for {fs}")
        rp = RoiBuilderParams(
            min_roi_intensity=point_noise_threshold, 
            at_least_one_point_above=chem_noise_threshold,
            min_roi_length=0
        )
        cm = ChemicalMixtureFromMZML(fs, roi_params=rp, ms2_sampler=FixedMS2Sampler())
        generated = cm.sample(None, 2, source_polarity=ionisation_mode)
        
        basename = ".".join(os.path.basename(fs).split(".")[:-1])
        ppath = os.path.join(out_dir, f"{basename}_temp.pkl")
        save_obj(generated, ppath)
        return ppath  

    @staticmethod
    def _run_case(case, chems, out_dir, pbar, min_rt, max_rt, ionisation_mode, scan_duration_dict):
        return case.run_controller(
            chems,
            out_dir,
            pbar, 
            min_rt, 
            max_rt, 
            ionisation_mode,
            scan_duration_dict,
        )
        
    def create_chems(self, 
                     out_dir, 
                     ionisation_mode, 
                     num_workers, 
                     point_noise_threshold=0,
                     chem_noise_threshold=0):
        
        all_fullscans = set(fs for case in self.cases for fs in case.fullscan_paths)
        with multiprocessing.Pool(num_workers) as pool:
            zipped = zip(
                all_fullscans,
                itertools.repeat(ionisation_mode),
                itertools.repeat(out_dir),
                itertools.repeat(point_noise_threshold),
                itertools.repeat(chem_noise_threshold)
            )
            pkl_names = pool.starmap(self._gen_chems, zipped)
            
        return {fs : pkl for fs, pkl in zip(all_fullscans, pkl_names)}

    def run_experiment(self,
                       out_dir,
                       pbar=False,
                       min_rt=0,
                       max_rt=1440,
                       ionisation_mode=POSITIVE,
                       scan_duration_dict=None,
                       point_noise_threshold=0,
                       chem_noise_threshold=0,
                       overwrite_keyfile=False,
                       num_workers=None):
        
        print("Creating Chemicals...")
        self.out_dir = out_dir
        chems = self.create_chems(
            out_dir, 
            ionisation_mode, 
            num_workers, 
            point_noise_threshold=point_noise_threshold,
            chem_noise_threshold=chem_noise_threshold
        )
        print()
        print(f"Running Experiment of {len(self.cases)} cases...")
        try:
            with multiprocessing.Pool(num_workers) as pool:
                case_iterable = [
                    (
                        case,
                        chems,
                        out_dir,
                        pbar,
                        min_rt,
                        max_rt,
                        ionisation_mode,
                        scan_duration_dict
                    )
                    for case in self.cases
                ]
                case_mzmls = pool.starmap(self._run_case, case_iterable)
            self.case_mzmls = {}
            for mapping in case_mzmls: 
                self.case_mzmls.update(mapping)
            self.write_json(overwrite=overwrite_keyfile)
        
        finally:
            for _, ppath in chems.items():
                try:
                    os.remove(ppath)
                except FileNotFoundError:
                    pass
    
    @staticmethod
    def amend_mzml_paths(mzml_pairs, fullscan_dir=None, out_dir=None):
        new_pairs = []
        for fs, mzml in mzml_pairs:
            if(not fullscan_dir is None):
                fs = os.path.join(fullscan_dir, os.path.basename(fs))
                
            if(not out_dir is None):
                mzml = os.path.join(out_dir, os.path.basename(mzml))
                
            new_pairs.append((fs, mzml))
        return new_pairs
    
    @classmethod
    def load_from_mzmls(cls,
                        case_mzmls, 
                        out_dir, 
                        fullscan_dir=None, 
                        amend_result_path=False,
                        case_names=None):
                        
        exp = cls()
        exp.out_dir = out_dir
        if(case_names is None):
            exp.case_names = list(case_mzmls.keys())
        else:
            exp.case_names = case_names
        exp.case_mzmls = {
            case_name : cls.amend_mzml_paths(case_mzmls[case_name], 
                                             fullscan_dir = fullscan_dir,
                                             out_dir = out_dir if amend_result_path else None
                                            ) 
            for case_name in exp.case_names
        }
        return exp
        
    def write_json(self, file_dir=None, file_name=None, overwrite=False):
        if(file_dir is None):
            file_dir = self.out_dir

        if(file_name is None):
            file_name = "keyfile.json"
            
        fname = os.path.join(file_dir, file_name)
        if(not overwrite and os.path.exists(fname)):
            with open(fname, "r") as f:
                all_cases = {**json.load(f), **self.case_mzmls}
        else:
            all_cases = self.case_mzmls
            
        with open(fname, "w") as f:
            json.dump(all_cases, f)
        
    @classmethod
    def load_from_json(cls, 
                       file_dir, 
                       file_name, 
                       out_dir,
                       fullscan_dir=None,
                       amend_result_path=False, 
                       case_names=None):
                       
        with open(os.path.join(file_dir, file_name), "r") as f:
            case_mzmls = json.load(f)
            
        return cls.load_from_mzmls(
            case_mzmls,
            out_dir,
            fullscan_dir=fullscan_dir,
            amend_result_path=amend_result_path,
            case_names=case_names
        )
    
    def _pick_aligned_peaks(self,
                            aligned_dirs=None, 
                            aligned_names=None,
                            mzmine_templates=None,
                            mzmine_exe=None,
                            force=False):
                            
        fullscan_paths = [
            [fs for fs, _ in self.case_mzmls[case_name]] 
            for case_name in self.case_names
        ]
    
        if(aligned_dirs is None):
            aligned_dirs = [self.out_dir] * len(self.case_names)
        else:
            try:
                if(type(aligned_dirs) == type("")): raise TypeError
                aligned_dirs = list(aligned_dirs)
            except TypeError:
                aligned_dirs = [self.out_dir] * len(self.case_names)
            
        if(aligned_names is None):
            unique_fs = {}
            aligned_names = []
            for fses in fullscan_paths:
                key = tuple(sorted(fses))
                if(not key in unique_fs):
                    unique_fs[key] = f"peaks_{len(unique_fs)}"
                aligned_names.append(unique_fs[key])
        else:
            try:
                if(type(aligned_names) == type("")): raise TypeError
                aligned_names = list(aligned_names)
            except TypeError:
                aligned_names = [aligned_names] * len(self.case_names)
        
        try:
            if(type(mzmine_templates) == type("")): raise TypeError
            mzmine_templates = list(mzmine_templates)
        except TypeError:
            mzmine_templates = [mzmine_templates] * len(self.case_names)
        
        aligned_paths = []
        forced = {os.path.join(dr, name) : False for dr, name in zip(aligned_dirs, aligned_names)}
        zipped = zip(
            aligned_dirs,
            aligned_names,
            mzmine_templates,
            fullscan_paths
        )
        for dr, name, template, fses in zipped:
            if(not template is None and not mzmine_exe is None):
                path = pick_aligned_peaks(
                    input_files = fses,
                    output_dir = dr,
                    output_name = name,
                    mzmine_template = template,
                    mzmine_exe = mzmine_exe,
                    force = force and not forced[os.path.join(dr, name)]
                )
                
                forced[os.path.join(dr, name)] = True
                aligned_paths.append(path)
                
        return aligned_paths
    
    def evaluate(self, 
                 num_workers=None,
                 isolation_widths=None,
                 aligned_dirs=None, 
                 aligned_names=None,
                 max_repeat=None,
                 mzmine_templates=None,
                 mzmine_exe=None,
                 force_peak_picking=False):
        
        aligned_names = self._pick_aligned_peaks(
            aligned_dirs=aligned_dirs, 
            aligned_names=aligned_names,
            mzmine_templates=mzmine_templates,
            mzmine_exe=mzmine_exe,
            force=force_peak_picking
        )
        print()
        
        if(isolation_widths is None):
            isolation_widths = [None for _ in self.case_names]
        else:
            try:
                isolation_widths = list(isolation_widths)
            except:
                isolation_widths = [isolation_widths] * len(self.case_names)
                
        max_repeat = len(self.case_names) if max_repeat is None else max_repeat
        zipped = zip(
            aligned_names,
            [self.case_mzmls[name][:max_repeat] for name in self.case_names],
            isolation_widths,
        )

        with multiprocessing.Pool(num_workers) as pool:
            self.evaluators = pool.starmap(evaluate_real, zipped)
    
    @staticmethod
    def _score_eva(eva, min_it, rank_key):
        return eva.evaluation_report(min_intensity=min_it)[rank_key][-1]
    
    def rank_cases(self, min_intensities, rank_key="sum_cumulative_coverage", num_workers=None):
        with multiprocessing.Pool(num_workers) as pool:
            scores = pool.starmap(
                self._score_eva, 
                zip(self.evaluators, min_intensities, itertools.repeat(rank_key))
            )
        
        return sorted(
            [i for i, _ in enumerate(self.evaluators)],
            key=lambda i: scores[i],
            reverse=True
        )
    
    @staticmethod
    def _summarise_helper(eva, min_intensity):
        return eva.summarise(min_intensity=min_intensity)
            
    def summarise(self, num_workers=None, min_intensities=None, rank_key=None):
        if(min_intensities is None):
            min_intensities = itertools.repeat(0.0)
        else:
            try:
                min_intensities = list(min_intensities)
            except TypeError:
                min_intensities = [min_intensities] * len(self.evaluators)
        
        with multiprocessing.Pool(num_workers) as pool:
            summary_strings = pool.starmap(
                self._summarise_helper, 
                zip(self.evaluators, min_intensities)
            )
            
        if(rank_key is None):
            idxes = range(len(self.case_names))
        else:
            idxes = self.rank_cases(min_intensities, rank_key=rank_key, num_workers=num_workers)
        
        for i in idxes:
            name, sstring = self.case_names[i], summary_strings[i]
            print(name)
            print(sstring)
            print()

    @classmethod
    def run_grid_search(cls,
                        controller_type,
                        datasets,
                        shared_params,
                        search_params,
                        out_dir,
                        pbar=False,
                        min_rt=0,
                        max_rt=1440,
                        ionisation_mode=POSITIVE,
                        scan_duration_dict=None,
                        point_noise_threshold=0,
                        chem_noise_threshold=0,
                        num_workers=None):
                        
        pairs = {
            name: [[(k, v) for v in v_ls] for k, v_ls in params.items()]
            for name, params in search_params.items()
        }
        
        params_map = {
            name: [
                {**shared_params, **dict(search_item)} 
                for search_item in itertools.product(*params)
            ]
            for name, params in pairs.items()
        }
        
        print(f"GRID SEARCH OF {sum(len(v) for _, v in params_map.items())} CASES")
        
        param_links = {
            f"{name}_{i}" : params
            for name, group in params_map.items()
            for i, params in enumerate(group)
        }
        save_obj(param_links, os.path.join(out_dir, "param_links.pkl"))
        
        cases = [
            ExperimentCase(
                controller_type,
                datasets,
                params,
                name=f"{name}_{i}"
            )
            for name, group in params_map.items()
            for i, params in enumerate(group)
        ]
        
        exp = cls()
        exp.add_cases(
            case for case in cases if case.valid_controller()
        )
        
        exp.run_experiment(
            out_dir,
            min_rt=min_rt,
            max_rt=max_rt,
            ionisation_mode=ionisation_mode,
            scan_duration_dict=scan_duration_dict,
            point_noise_threshold=point_noise_threshold,
            chem_noise_threshold=chem_noise_threshold,
            num_workers=num_workers
        )
        
        return exp