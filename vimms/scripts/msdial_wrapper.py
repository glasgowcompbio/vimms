from vimms.Common import create_if_not_exist, MSDIAL_DDA_MODE, MSDIAL_DIA_MODE
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

sys.path.append('..')
sys.path.append('../..')  # if running in this folder


def run_msdial(msdial_console_app, mode, params_file, input_dir,
               output_dir=None, save_project=False):
    """
    Runs MSDIAL as a subprocess on an input folder
    :param msdial_console_app: path to MSDIAL console app
    :param mode: either MSDIAL_DDA_MODE or MSDIAL_DIA_MODE
    :param params_file: path to MSDIAL params file
    :param input_dir: input directory, all mzML files in this folder will
    be processed
    :param output_dir: output directory.
    :return: None. The results will be placed in output_dir.
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        create_if_not_exist(output_dir)

    args = [msdial_console_app, mode, '-i', input_dir, '-o', output_dir, '-m',
            params_file]
    if save_project:
        args.append('-p')
    subprocess.run(args)


def run_msdial_batch(msdial_console_app, mode, params_file, mzml_folder,
                     msp_folder=None, remove_substring=None,
                     subdir=False, save_project=False):
    """
    Copies each mzML file in the specified mzml_folder to be processed by
    MSDIAL separately. The results will be placed into mzml_folder.

    Args:
        msdial_console_app: path to MSDIAL console app
        mode: either MSDIAL_DDA_MODE or MSDIAL_DIA_MODE
        params_file: path to MSDIAL params file
        mzml_folder: input directory, each mzML file in this folder will
                     be processed separately by MSDIAL
        msp_folder: if True, then the input should be a directory containing
                    subdirectories of mzML files to be processed together.
        remove_substring: remove the specified substring
        subdir: whether the mzML folder consists subdirectories or not
        save_project: if True, then save the MSDIAL project file (can be
                      loaded to GUI later on)

    Returns: None. The results will be placed in mzml_folder

    """

    # Process a folder containing mzML files.
    # Each mzML file will be processed separately.
    if not subdir:

        # for each mzML file ...
        mzml_files = glob.glob(os.path.join(mzml_folder, '*.mzML'))
        for i in range(len(mzml_files)):
            mzml_file = mzml_files[i]
            logger.info('{}/{} {}'.format(i + 1, len(mzml_files), mzml_file))

            with tempfile.TemporaryDirectory() as temp_path:
                # copy mzml file to temp dir
                shutil.copy2(mzml_file, temp_path)

                # assume that each mzML has a corresponding .msp file in
                # that folder
                new_params_file = params_file
                if msp_folder is not None:
                    msp_path = get_path_in_folder(mzml_file, '{}.msp', msp_folder,
                                                  remove_substring=remove_substring)
                    # logger.debug('Using {}'.format(msp_path))

                    # if the msp file actually exists
                    if os.path.exists(msp_path):
                        # read existing params file
                        params_txt = Path(params_file).read_text()

                        # substitute '{msp_file}' in params text with the
                        # actual msp path
                        params_dir = os.path.abspath(os.path.dirname(params_file))
                        params_txt = params_txt.format(msp_file=msp_path, params_dir=params_dir)
                        # logger.debug(params_txt)

                        # construct new path to params_file inside temp_path,
                        # write the substituted text to it
                        new_params_file = get_path_in_folder(params_file, None,
                                                             temp_path)
                        Path(new_params_file).write_text(params_txt)

                run_msdial(msdial_console_app, mode, new_params_file,
                           temp_path, output_dir=mzml_folder, save_project=save_project)

    # Process a folder containing sub-folders of mzML files.
    # Each subfolder contains mzML files that will be processed together.
    else:
        subdirs = [f.path for f in os.scandir(mzml_folder) if f.is_dir()]
        for subdir in subdirs:
            logger.warning(subdir)
            new_params_file = params_file

            # assume that each subdir has a corresponding .msp file in
            # that folder
            if msp_folder is not None:
                basename = os.path.basename(subdir)
                msp_path = get_path_in_folder(basename, '{}.msp', msp_folder,
                                              remove_substring=remove_substring)
                # logger.debug('Using {}'.format(msp_path))

                # if the msp file actually exists
                if os.path.exists(msp_path):
                    # read existing params file
                    params_txt = Path(params_file).read_text()

                    # substitute '{msp_file}' in params text with the actual
                    # msp path
                    params_dir = os.path.abspath(os.path.dirname(params_file))
                    params_txt = params_txt.format(msp_file=msp_path, params_dir=params_dir)
                    # logger.debug(params_txt)

                    # construct new path to params_file inside temp_path,
                    # write the substituted text to it
                    new_params_file = get_path_in_folder(params_file, None,
                                                         subdir)
                    Path(new_params_file).write_text(params_txt)

            run_msdial(msdial_console_app, mode, new_params_file, subdir,
                       output_dir=subdir,
                       save_project=save_project)


def get_path_in_folder(fname, ext_pattern, target_folder, remove_substring=None):
    """
    Get the front part of fname without extension, add extension specified by
    ext_pattern (if specified), then constructs a new path inside target_folder
    :param fname: the original filename
    :param ext_pattern: extension pattern, e.g. '{}.msp'
    :param target_folder: the target folder
    :return: a new path as specified above
    """
    basename = os.path.basename(fname)
    if remove_substring is not None: # strip certain substring from basename
        basename = basename.replace(remove_substring, '')
    if ext_pattern is not None:  # append front part with the extension to
        # get a new basename
        front_no_ext = os.path.splitext(basename)[0]
        basename = ext_pattern.format(front_no_ext)

    # construct new path for basename inside target folder
    new_path = os.path.abspath(os.path.join(target_folder, basename))
    return new_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MSDIAL Wrapper')
    parser.add_argument('msdial_console_app', type=str)
    parser.add_argument('mode', type=str,
                        choices=[MSDIAL_DDA_MODE, MSDIAL_DIA_MODE])
    parser.add_argument('params', dest='params', type=str)
    parser.add_argument('input', type=str)
    parser.add_argument('--output', dest='output', type=str)

    args = parser.parse_args()
    logger.info(args)
    run_msdial(args.msdial_console_app, args.mode, args.params, args.input,
               output_dir=args.output)
