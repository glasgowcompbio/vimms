# mzmine-related code
import os,glob
import xml.etree.ElementTree
import numpy as np

def pick_peaks(file_list,
                xml_template = 'batch_files/PretermPilot2Reduced.xml',
                output_dir = '/Users/simon/git/pymzmine/output',
                mzmine_command = '/Users/simon/MZmine-2.40.1/startMZmine_MacOSX.command',
                force = False):
    et = xml.etree.ElementTree.parse(xml_template)
    if type(file_list) is not list:
        file_list = [file_list]
    # Loop over files in the list (just the firts three for now)
    for filename in file_list:
        print("Creating xml batch file for {}".format(filename.split(os.sep)[-1]))

        # if any output files to be produced do not already exist
        # we run. Otherwise, we only run if force = True
        # any block that produces an output file that doesn't already exist sets this to True.
        need_to_run = False

        root = et.getroot()
        for child in root:
            # Set the input filename
            if child.attrib['method'].endswith('RawDataImportModule'):
                for e in child:
                    for g in e:
                        g.text = filename # raw data file name
            # Set the mzTab export filename
            if child.attrib['method'].endswith('MzTabExportModule'):
                for e in child:
                    for g in e:
                        tag = g.tag
                        text = g.text
                        if tag == 'current_file':
                            mztab_name = os.path.join(output_dir,filename.split(os.sep)[-1].split('.')[0]+'_pp.mzTab')
                            # check if the file exists: if it already does, don't do it again (unless Force = True)
                            g.text = mztab_name
                            if not os.path.exists(mztab_name):
                                need_to_run = True
            if child.attrib['method'].endswith('GNPSExportAndSubmitModule'):
                for e in child:
                    for g in e:
                        if g.tag == 'current_file':
                            mgf_name = os.path.join(output_dir,filename.split(os.sep)[-1].split('.')[0] + '.mgf')
                            g.text = mgf_name
                            if not os.path.exists(mgf_name):
                                need_to_run = True
            if child.attrib['method'].endswith('CSVExportModule'):
                for e in child:
                    for g in e:
                        if g.tag == 'current_file':
                            csv_box_name = os.path.join(output_dir,filename.split(os.sep)[-1].split('.')[0] + '_box.csv')
                            print(csv_box_name)
                            g.text = csv_box_name
                            if not os.path.exists(csv_box_name):
                                need_to_run = True
        # write the xml file for this input file
        new_xml_name = os.path.join(output_dir,filename.split(os.sep)[-1].split('.')[0]+'.xml')
        et.write(new_xml_name)
        # Run mzmine
        
        if force or need_to_run:
            print("Running mzMine for {}".format(filename.split(os.sep)[-1]))
            os.system(mzmine_command + ' "{}"'.format(new_xml_name))
        else:
            print("Output exists for {}. Set force = True to overwrite".format(filename))



def align(mztab_folder,
            xml_template = 'batch_files/align_from_mztab.xml',
            mzmine_command = '/Users/simon/MZmine-2.40.1/startMZmine_MacOSX.command', rt_window = 0.5,
            output_csv_name = 'pp_aligned.csv',
            specific_files = None):
    # grab all the mztab files in a folder
    # mztab_folder = '/Users/simon/git/pymzmine/output'


    # if a specific set of files is provided, use those
    # otherwise use all files in the folder
    if specific_files is None:
        tab_files = glob.glob(os.path.join(mztab_folder,'*.mzTab'))
    else:
        tab_files = specific_files
    print(tab_files)


    # The output name for the aligned peaks
    csv_name = os.path.join(mztab_folder,output_csv_name)


    et = xml.etree.ElementTree.parse(xml_template)
    root = et.getroot()
    for child in root:
        # Add all the mzTab files to the import block
        if child.attrib['method'] == 'net.sf.mzmine.modules.peaklistmethods.io.mztabimport.MzTabImportModule':
            for c in child:
                if c.tag == 'parameter' and c.attrib['name'] == 'mzTab files':
                    # remove files already in xml, should just be a dummy one
                    for f in c:
                        c.remove(f)
                    for tab_file in tab_files:
                        el = xml.etree.ElementTree.Element('file')
                        el.text = tab_file
                        c.append(el)

                if c.tag == 'parameter' and c.attrib['name'] == 'Import raw data files?':
                    # Can also import the raw data with the peak lists, although it doesn't seem to be
                    # necessary for the join alignment
                    for f in c:
                        f.text = 'false'

        ###code below added for changing rt window
        if child.attrib['method'] == 'net.sf.mzmine.modules.peaklistmethods.alignment.join.JoinAlignerModule':
            # change the RT window parameter
            for c in child:

                if c.tag == 'parameter' and c.attrib['name'] == 'Retention time tolerance' and c.attrib['type'] == 'absolute' :
                    c.text = str(rt_window)
        ###

        if child.attrib['method'] == 'net.sf.mzmine.modules.peaklistmethods.io.csvexport.CSVExportModule':
            # Export the results to .csv
            for c in child:
                if c.tag == 'parameter' and c.attrib['name'] == 'Filename':
                    for f in c:
                        f.text = csv_name
    # Write the xml file
    xml_name = os.path.join(mztab_folder,'align.xml')


    et.write(xml_name)

    # Run MZ-mine
    os.system(mzmine_command + " " + xml_name)


def match_aligned_to_original(aligned_peaks,original_files,output_dir,f_idx_dict,write_file = True,original_csv_suffix = '_quant'):
    import csv
    original_csvs = [os.path.join(output_dir,original_file + original_csv_suffix + '.csv') for original_file in original_files]

    matches = {}
    for file_pos,o in enumerate(original_files):
        # Read the original .csv file
        with open(original_csvs[file_pos],'r') as f:
            reader = csv.reader(f)
            heads = next(reader)
            local_peaks = []
            for line in reader:
                id = int(line[0])
                mz = float(line[1])
                rt = float(line[2])
                intensity = float(line[3])
                local_peaks.append((id,mz,rt,intensity))
        local_peaks.sort(key = lambda x: x[3], reverse = False)
        this_idx = f_idx_dict[o]
        sub_aligned = list(filter(lambda x: x[3][this_idx]>0.0,aligned_peaks))
        sub_aligned.sort(key = lambda x: x[3][this_idx],reverse = False)

        for i,local_peak in enumerate(local_peaks):
            a_peak = sub_aligned[i]
            assert abs(local_peak[3] - a_peak[3][this_idx]) < 0.1,print(local_peak,a_peak)
            assert abs(local_peak[1] - a_peak[1]) < 0.1

            if not a_peak in matches:
                matches[a_peak] = {}
            matches[a_peak][o] = local_peak


    if write_file:
        output_file = os.path.join(output_dir,'align_links.csv')
        with open(output_file,'w') as f:
            heads = ['align ID','row m/z','row retention time'] + original_files
            writer = csv.writer(f)
            writer.writerow(heads)
            for aligned_peak in matches:
                new_row = [aligned_peak[0],aligned_peak[1],aligned_peak[2]]
                for o in original_files:
                    val = matches[aligned_peak].get(o,None)
                    if val:
                        new_row.append(val[0])
                    else:
                        new_row.append('null')
                writer.writerow(new_row)
    return matches



def align_spectra(aligned_peaks,original_files,output_dir,matches,make_mgf = False):
    from mnet_utilities import load_mgf
    from mnet import Cluster


    spec_clusters = {a:None for a in aligned_peaks}
    spectra = {}
    for o in original_files:
        mgf_file = os.path.join(output_dir,o+'.mgf')
        print("Loading peaks from {}".format(mgf_file))
        spectra[o] = load_mgf(mgf_file)
    peaks_with_spec = 0
    n_done = 0
    for ap in spec_clusters:
        for o in original_files:
            if o in matches[ap]:
                o_id = matches[ap][o][0]
                spectrum = spectra[o].get(o_id,None)
                if spectrum:
                    if spec_clusters[ap]:
                        spec_clusters[ap].add_spectrum(spectrum)
                    else:
                        spec_clusters[ap] = Cluster(spectrum,ap[0])
                        peaks_with_spec += 1
        n_done += 1
        if n_done % 1000 == 0:
            print("done {} of {}, {} peaks have spectra".format(n_done,len(aligned_peaks),peaks_with_spec))

    if make_mgf:
        mgf_name = os.path.join(output_dir,'pp_aligned.mgf')
        print("Writing {}".format(mgf_name))
        with open(mgf_name,'w') as f:
            for ap,cl in spec_clusters.items():
                if cl:
                    f.write(cl.get_mgf_string())
    return spec_clusters


def compute_cosine_aligned_peaks(
        aligned_peaks, original_files, output_dir, matches, spectra=None, f_idx_dict=None):
    # loops thorugh aligned peaks and, where two peaks that have been aligned
    # have spectra, their cosine is computed
    # for each aligned peak, the lowest cosine is kept
    from itertools import combinations
    from scoring_functions import fast_cosine

    if not matches:
        if f_idx_dict is None:
            raise ValueError("f_idx_dict is required when matches are not provided")
        matches = match_aligned_to_original(aligned_peaks,original_files,output_dir,f_idx_dict)

    if not spectra:
        spectra = {}
        from mnet_utilities import load_mgf
        for o in original_files:
            mgf_file = os.path.join(output_dir,o+'.mgf')
            spectra[o] = load_mgf(mgf_file)
            print("Loaded {} spectra from {}".format(len(spectra[o]),o))

    worst_matches = {}
    n_done = 0
    for aligned_peak,these_matches in matches.items():
        for f1,f2 in combinations(original_files,r = 2):
            f1_id = None
            f2_id = None
            f1_spec = None
            f2_spec = None
            f1_peak = these_matches.get(f1,None)
            if f1_peak:
                f1_id = f1_peak[0]
            f2_peak = these_matches.get(f2,None)
            if f2_peak:
                f2_id = f2_peak[0]
            if f1_id:
                f1_spec = spectra[f1].get(f1_id,None)
            if f2_id:
                f2_spec = spectra[f2].get(f2_id,None)

            if f1_spec and f2_spec:
                sc,_ = fast_cosine(f1_spec,f2_spec,0.2,1)
                # store the worst pairwise score for each set
                if not aligned_peak in worst_matches:
                    worst_matches[aligned_peak] = (f1,f2,f1_id,f2_id,f1_spec,f2_spec,sc)
                else:
                    if sc < worst_matches[aligned_peak][-1]:
                        worst_matches[aligned_peak] = (f1,f2,f1_id,f2_id,f1_spec,f2_spec,sc)
        n_done += 1
        if n_done % 1000== 0:
            print(n_done,len(aligned_peaks))
    return worst_matches

def load_aligned_peaks(output_dir,aligned_csv = 'pp_aligned.csv',original_csv_suffix = '_quant'):
    import csv
    align_file = os.path.join(output_dir,aligned_csv) # file including the aligned peaks and intensities
    original_files = glob.glob(os.path.join(output_dir,'*.xml'))

    original_files = [o.split(os.sep)[-1].split('.')[0] for o in original_files]


    original_csvs = [os.path.join(output_dir,original_file + original_csv_suffix + '.csv') for original_file in original_files]
    print(original_csvs)

    aligned_peaks = []
    f_idx_dict =  {}

    original_files_filtered = []
    with open(align_file,'r') as f:
        reader = csv.reader(f)
        heads = next(reader)
        file_bits = heads[3:]
        for o in original_files:
            temp = [1 if f.startswith(o) else 0 for f in file_bits]
            try:
                f_idx_dict[o] = temp.index(1)
                original_files_filtered.append(o)
            except:
                print("{} not in list, probably ok!".format(o))
        for line in reader:
            align_id = int(line[0])
            align_mz = float(line[1])
            align_rt = float(line[2])
            intensities = tuple([float(a) for a in line[3:-1]])
            aligned_peaks.append((align_id,align_mz,align_rt,intensities))

    return aligned_peaks,original_files_filtered,f_idx_dict



def filter_duplicates(mzml_file_list,output_folder,new_output_folder,
                        min_repetitions = 10,
                        MS1_round_precision = 2,
                        score_thresh = 0.8,
                        ms2_tol = 0.1,
                        min_match_peaks = 1):
    # finds the duplicate peaks in the mzml file, loads the outputs and writes new outputs with the duplicates removed
    import csv
    for mzml_file in mzml_file_list:
        groups = find_spectral_groups(mzml_file,
                                        MS1_round_precision = MS1_round_precision,
                                        min_match_peaks = min_match_peaks,
                                        ms2_tol = ms2_tol,
                                        score_thresh = score_thresh,
                                        )
        # find and load the output files
        csv_name = os.path.join(output_folder,mzml_file.split(os.sep)[-1].split('.')[0] + '_quant.csv')
        mztab_name = os.path.join(output_folder,mzml_file.split(os.sep)[-1].split('.')[0] + '_pp.mzTab')
        new_csv_name = os.path.join(new_output_folder,mzml_file.split(os.sep)[-1].split('.')[0] + '_quant.csv')
        new_mztab_name = os.path.join(new_output_folder,mzml_file.split(os.sep)[-1].split('.')[0] + '_pp.mzTab')

        # copy the mgf files into the new folder so that things further down the pipeline will work
        mgf_name = os.path.join(output_folder,mzml_file.split(os.sep)[-1].split('.')[0] + '.mgf')
        new_mgf_name = os.path.join(new_output_folder,mzml_file.split(os.sep)[-1].split('.')[0] + '.mgf')
        os.system('cp {} {}'.format(mgf_name,new_mgf_name))
        remove_rows = set()
        with open(csv_name,'r') as f:
            reader = csv.reader(f)
            with open(new_csv_name,'w') as g:
                writer = csv.writer(g)
                # write heads
                writer.writerow(next(reader))
                row_idx = 0
                for line in reader:
                    mz = float(line[1])
                    mz = round(mz,MS1_round_precision)
                    rt = float(line[2])*60
                    remove_this = False
                    # check if this is within one of the clusters
                    if mz in groups:
                        for sub_group in groups[mz]:
                            if sub_group.n_spectra >= min_repetitions:
                                min_rt = min([s.rt for s in sub_group.spectra])
                                max_rt = max([s.rt for s in sub_group.spectra])
                                if rt >= min_rt and rt <= max_rt:
                                    remove_this = True
                                    remove_rows.add(row_idx)
                                    break
                    if not remove_this:
                        writer.writerow(line)
                    row_idx += 1
        with open(mztab_name,'r') as f:
            reader = csv.reader(f,delimiter = '\t')
            with open(new_mztab_name,'w') as g:
                writer = csv.writer(g,delimiter = '\t')
                line = next(reader)
                while len(line) == 0 or not line[0] == 'SMH':
                    writer.writerow(line)
                    line = next(reader)
                writer.writerow(line) # write the 'SMH' line
                row_idx = 0
                for line in reader:
                    if not row_idx in remove_rows:
                        writer.writerow(line)
                    row_idx += 1

def find_spectral_groups(input_file,MS1_round_precision = 2,
                            min_match_peaks = 1,
                            ms2_tol = 0.1,
                            score_thresh = 0.8,
                            verbose = False):

    from pyopenms import MSExperiment,MzMLFile
    from mnet import Spectrum,Cluster
    from scoring_functions import fast_cosine

    print("Finding groups of spectra in {}".format(input_file))
    exp = MSExperiment()
    MzMLFile().load(input_file, exp)

    clusters = {} # dictionary, key is precursor mz, value is a list of clusters

    # loop over spectra
    cl_id = 0
    n_spec = 0
    biggest_cluster_size = 0
    for ind, spectrum in enumerate(exp):
        if spectrum.getMSLevel() != 1:
            raw_precursormz = spectrum.getPrecursors()[0].getMZ()
            precursormz = round(spectrum.getPrecursors()[0].getMZ(),MS1_round_precision)
            if not precursormz in clusters:
                clusters[precursormz] = []
            # make a Spectrum object
            peaks = zip(spectrum.get_peaks()[0],spectrum.get_peaks()[1])
            s = Spectrum(peaks,input_file,ind,None,raw_precursormz,raw_precursormz,rt = spectrum.getRT())
            if len(clusters[precursormz]) == 0:
                # no clusters exist, make one
                new_cluster = Cluster(s,cl_id)
                cl_id += 1
                clusters[precursormz].append(new_cluster)
            else:
                #match_ers exist, compute the similarity between the spectrum and each
                best_score = -1
                best_pos = None
                for pos,cl in enumerate(clusters[precursormz]):
                    sc,_ = fast_cosine(s,cl,ms2_tol,min_match_peaks)
                    if sc > best_score:
                        best_score = sc
                        best_pos = pos
                if best_score >= score_thresh:
                    clusters[precursormz][pos].add_spectrum(s)
                    if clusters[precursormz][pos].n_spectra > biggest_cluster_size:
                        biggest_cluster_size = clusters[precursormz][pos].n_spectra
                else:
                    new_cluster = Cluster(s,cl_id)
                    cl_id += 1
                    clusters[precursormz].append(new_cluster)

            n_spec += 1
        if verbose and n_spec % 100 == 0:
            print("{} Masses, {} clusters, biggest cluster = {} spectra".format(len(clusters),cl_id,biggest_cluster_size))
    return clusters


def peakareas2numpy(peak_areas,tic_normalise = True):
    pakeys = list(peak_areas.keys())
    key_idx = {}
    file_idx = {}
    papos = 0
    filepos = 0
    dlist = []
    for cluster_id in peak_areas:
        these_areas = peak_areas[cluster_id]['areas']
        key_idx[cluster_id] = papos
        for f,intensity in these_areas.items():
            if not f in file_idx:
                file_idx[f] = filepos
                filepos += 1
            dlist.append((papos,file_idx[f],intensity))
        papos += 1
    from scipy.sparse import coo_matrix
    i,j,k = zip(*dlist)
    cc = coo_matrix((k,(i,j)),shape=(len(key_idx),len(file_idx)))

    pa_array = np.array(cc.todense())

    if tic_normalise:
        pa_array_tic = pa_array/pa_array.sum(axis=0)
        return pa_array,pa_array_tic,key_idx,file_idx
    else:
        return pa_array,key_idx,file_idx
