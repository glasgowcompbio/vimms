import os
import itertools
import re
import xml
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


@dataclass
class MZMineParams():
    RT_FACTOR = 60 #minutes

    mzmine_template: str
    mzmine_exe: str
    
    def _make_batch_file(self, input_files, output_dir, output_name, output_path):
        et = xml.etree.ElementTree.parse(self.mzmine_template)
        root = et.getroot()
        for child in root:

            if child.attrib["method"].endswith("RawDataImportModule"):
                input_found = False
                for e in child:
                    if (e.attrib["name"].strip().lower() == "raw data file names"):
                        for f in e:
                            e.remove(f)
                        for i, fname in enumerate(input_files):
                            new = xml.etree.ElementTree.SubElement(e, "file")
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
        input_files = list(set(input_files)) #filter duplicates
        output_path = format_output_path("MZMine", output_dir, output_name)

        if (not os.path.exists(output_path) or force):
            new_xml = self._make_batch_file(input_files, output_dir, output_name, output_path)
            print(f"Running MZMine for {output_path}")
            subprocess.run([self.mzmine_exe, new_xml])
            
        report_boxes("MZMine", output_path)
        return output_path
        
    @staticmethod
    def check_files_match(fullscan_names, aligned_path, mode="subset"):
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
                       
    params = MZMineParams(mzmine_template, mzmine_exe)
    return params.pick_aligned_peaks(input_files, output_dir, output_name, force=force)
            

@dataclass
class XCMSScriptParams:
    #TODO: It would be good to just call the R functions from Python
    #instead of calling an external R script...
    
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
        
    #groupPeaksNearest params
    mzvsrtbalance: int = None
    absmz: float = None
    absrt: float = None
    kNN: int = None
    
    def pick_aligned_peaks(self, input_files, output_dir, output_name, force=False):
        input_files = list(set(input_files)) #filter duplicates
        output_path = format_output_path("XCMS", output_dir, output_name)
        
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
        return MZMineParams.check_files_match(fullscan_names, aligned_path, mode=mode)
        
    @staticmethod
    def read_aligned_csv(box_file):
        return MZMineParams.read_aligned_csv(box_file)