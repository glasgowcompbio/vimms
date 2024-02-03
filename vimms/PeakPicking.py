import os
import itertools
import re
from xml.etree import ElementTree
import pathlib
import subprocess
from collections import defaultdict
from dataclasses import dataclass


def count_boxes(box_filepath):
    with open(box_filepath, "r") as f:
        return sum(ln.strip() != "" for ln in f) - 1
        
def format_output_path(method_name, output_dir, output_name):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if(len(output_name.split(".")) > 1):
        output_name = "".join(output_name.split(".")[:-1])
    return os.path.join(output_dir, f"{output_name}_{method_name.lower()}_aligned.csv")

def report_boxes(method_name, output_path):
    try:
        num_boxes = count_boxes(output_path)
        print(f"{num_boxes} aligned boxes contained in file")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The box file doesn't seem to exist - did {method_name} silently fail?"
        )

class AbstractParams():
    method_name = None
    
    @classmethod
    def format_output_path(cls, output_dir, output_name):
        return format_output_path(cls.method_name, output_dir, output_name)

    @classmethod
    def report_boxes(cls, output_path):
        return report_boxes(cls.method_name, output_path)

@dataclass
class MZMineParams(AbstractParams):
    """
        Wrapper class to run MZMine 2 peak-picking from the ViMMS codebase.
        MZMine 2 allows commands for its processing pipeline to be stored in an .xml
        and then run via command line using its "batch mode" executable. Given an
        appropriate "template" .xml this class will substitute input and output file
        names into it and then run it in batch mode via subprocess.
        
        NOTE: MZMine is not installed with ViMMS. It must be installed separately
        and the path to the "batch mode" executable specified for this class.
        
        Args: 
            mzmine_template: Path to .xml template giving batch commands.
            mzmine_exe: Path to batch mode executable.
    """
    method_name = "MZMine"

    RT_FACTOR = 60 #minutes

    mzmine_template: str
    mzmine_exe: str
    
    def _make_batch_file(self, input_files, output_dir, output_name, output_path):
        et = ElementTree.parse(self.mzmine_template)
        root = et.getroot()
        for child in root:

            if child.attrib["method"].endswith("RawDataImportModule"):
                input_found = False
                for e in child:
                    if (e.attrib["name"].strip().lower() == "raw data file names"):
                        for f in e:
                            e.remove(f)
                        for i, fname in enumerate(input_files):
                            new = ElementTree.SubElement(e, "file")
                            new.text = os.path.abspath(fname)
                            padding = " " * (0 if i == len(input_files) - 1 else 8)
                            new.tail = e.tail + padding
                        input_found = True
                assert input_found, "Couldn't find a place to put the input files in the template!"

            if child.attrib["method"].endswith("CSVExportModule"):
                for e in child:
                    for f in e:
                        if f.tag == "current_file":
                            f.text = output_path

        new_xml = os.path.join(output_dir, f"{output_name}_template.xml")
        et.write(new_xml)
        return new_xml
    
    def pick_aligned_peaks(self, input_files, output_dir, output_name, force=False):
        """
            Run MZMine batch mode file for a list of input files.
            
            Args:
                input_files: Iterable of paths to input files.
                output_dir: Directory to write output to.
                output_name: Name for output file. Some text and the file extension
                    are added automatically.
                force: When False, don't run peak-picking if a file already exists
                    at the output destination.
                    
            Returns: Full path the output file was written to.
        """
        input_files = list(set(input_files)) #filter duplicates
        output_path = self.format_output_path(output_dir, output_name)

        if (not os.path.exists(output_path) or force):
            new_xml = self._make_batch_file(input_files, output_dir, output_name, output_path)
            print(f"Running MZMine for {output_path}")
            subprocess.run([self.mzmine_exe, new_xml])
            
        report_boxes("MZMine", output_path)
        return output_path
        
    @staticmethod
    def check_files_match(fullscan_names, aligned_path, mode="subset"):
        """
            Check that the source files listed in the header of a peak-picking
            output match an input list.
            
            Args:
                fullscan_names: List of .mzml files (or paths to them) to look
                    for in the header of the aligned file.
                aligned_path: Full filepath to the aligned file.
                mode: "subset" just checks if all fullscan_names can be found in
                    the header. "exact" checks whether or not the two sets of
                    names exactly match.
                
            Returns: Tuple of boolean reporting whether test succeeded, the
                names of the fullscans given as input, and the names of files
                found in the header.
        """
        fs_names = {os.path.basename(fs) for fs in fullscan_names}
        mzmine_names = set()
        
        with open(aligned_path, "r") as f:
            headers = f.readline().split(",")
            pattern = re.compile(r"(.*\.mzML).*")
            
            for h in headers:
                for fs in fs_names:
                    m = pattern.match(h)
                    if(not m is None):
                        mzmine_names.add(m.group(1))
        
        mode = mode.lower()
        if(mode == "exact"):
            passed = not fs_names ^ mzmine_names
        elif(mode == "subset"):
            passed = not fs_names - mzmine_names
        else:
            raise ValueError("Mode not recognised")
            
        return passed, fs_names, mzmine_names
    
    @staticmethod
    def read_aligned_csv(box_file_path):
        """
            Parse in an aligned boxfile in MZMine 2 format. Each column
            in an aligned boxfile either has properties related to the whole
            row (e.g. average m/z of the peak aligned on that row) or a property
            specific property of an unaligned peak from a parent .mzML. Row 
            properties are parsed into a list of dictionaries (one dictionary
            per row) in the form [{property_name: value}, ...]. .mzML properties
            are loaded into a similar list but with a nested dictionary
            i.e. [{mzml_name: {property_name: value}}, ...].
            
            Args:
                box_file_path: Full path to the aligned boxfile.
                
            Returns: Tuple of .mzML names and iterable of pairs of row dicts
                and .mzML dicts.
        """
    
        row_headers = [
            "row ID",
            "row m/z",
            "row retention time"
        ]
        
        with open(box_file_path, "r") as f:
            headers = f.readline().split(",")
            row_indices, mzml_indices = {}, defaultdict(dict)
            
            pattern = re.compile(r"(.*)\.mzML filtered Peak ([a-zA-Z/]+( [a-zA-Z/]+)*)")
            for i, h in enumerate(headers):
                if(h in row_headers):
                    row_indices[h] = i
                else:
                    m = pattern.match(h)
                    if(not m is None):
                        mzml_indices[m.group(1)][m.group(2)] = i
        
            fullscan_names = mzml_indices.keys()
            row_ls, mzml_ls = [], []
            for ln in f:
                split = ln.split(",")
                row_ls.append({k: split[i] for k, i in row_indices.items()})
                mzml_ls.append(
                    {
                        mzml : {k: split[i] for k, i in inner.items()} 
                        for mzml, inner in mzml_indices.items()
                    }
                )
            
            return fullscan_names, zip(row_ls, mzml_ls)

    
def pick_aligned_peaks(input_files,
                       output_dir,
                       output_name,
                       mzmine_template,
                       mzmine_exe,
                       force=False):
    """
        Convenience function (for backwards compatibility) which picks
        peaks using MZMineParams.
    """
                       
    params = MZMineParams(mzmine_template, mzmine_exe)
    return params.pick_aligned_peaks(input_files, output_dir, output_name, force=force)
            

@dataclass
class XCMSScriptParams(AbstractParams):
    """
        Wrapper class to run XCMS scripts written in R from ViMMS. The R script
        is run via subprocess and is given all arguments specified in the object
        instance as command-line arguments - the R script must handle any that
        are not None. XCMS does not natively write out aligned peaks so methods
        for reading output files assume they were written in the same format as
        MZMineParams.
        
        NOTE: R and XCMS are not installed with ViMMS. They must be installed
        separately and the paths to both the Rscript utility and the XCMS
        script to run must be specified for this class.
        
        Args:
            xcms_r_script: Path to the XCMS script written in R which should
                be run.
            rscript_exe: Path to the "Rscript" utility packaged with R. By
                default assumes it can be found via the "Rscript" environment
                variable.
            others: See xcms documentation for details.
    """
    #TODO: It would be good to just call the R functions from Python
    #instead of calling an external R script...
    
    method_name = "xcms"
    
    RT_FACTOR = 1 #seconds

    #file locations
    xcms_r_script: str
    rscript_exe: str = "Rscript"
    
    #centwave params
    ppm: int = None
    pwlower: int = None
    pwupper: int = None
    snthresh: int = None
    noise: int = None
    prefilterlower: int = None
    prefilterupper: int = None
    mzdiff: float = None
        
    #groupPeaksNearest params
    mzvsrtbalance: int = None
    absmz: float = None
    absrt: float = None
    kNN: int = None
    
    def pick_aligned_peaks(self, input_files, output_dir, output_name, force=False):
        """
            Run XCMS script for a list of input files.
            
            Args:
                input_files: Iterable of paths to input files.
                output_dir: Directory to write output to.
                output_name: Name for output file. Some text and the file extension
                    are added automatically.
                force: When False, don't run peak-picking if a file already exists
                    at the output destination.
                    
            Returns: Full path the output file was written to.
        """
    
        input_files = list(set(input_files)) #filter duplicates
        output_path = self.format_output_path(output_dir, output_name)
        
        params = (
            (f"--{k}", str(v)) 
            for k, v in self.__dict__.items() 
            if k != "xcms_r_script" and k != "rscript_exe" and not v is None
        )

        if(not os.path.exists(output_path) or force):
            print(f"Running XCMS for {output_path}")
            subprocess.run(
                [
                    self.rscript_exe, self. xcms_r_script, output_path,
                ] 
                + input_files + list(itertools.chain(*params))
            )

        report_boxes("XCMS", output_path)
        return output_path
        
    @staticmethod
    def check_files_match(fullscan_names, aligned_path, mode="subset"):
        """
            Wrapper to MZMineParams' "check_files_match".
        """
        return MZMineParams.check_files_match(fullscan_names, aligned_path, mode=mode)
        
    @staticmethod
    def read_aligned_csv(box_file):
        """
            Wrapper to MZMineParams' "read_aligned_csv".
        """
        return MZMineParams.read_aligned_csv(box_file)