import os
import xml.etree.ElementTree

from loguru import logger


def pick_peaks(file_list,
               xml_template='batch_files/PretermPilot2Reduced.xml',
               output_dir='/Users/simon/git/pymzmine/output',
               mzmine_command='/Users/simon/MZmine-2.40.1/startMZmine_MacOSX.command',
               add_name=None):
    et = xml.etree.ElementTree.parse(xml_template)
    # Loop over files in the list (just the firts three for now)
    for filename in file_list:
        logger.info("Creating xml batch file for {}".format(filename.split(os.sep)[-1]))
        root = et.getroot()
        for child in root:
            # Set the input filename
            if child.attrib['method'] == 'net.sf.mzmine.modules.rawdatamethods.rawdataimport.RawDataImportModule':
                for e in child:
                    for g in e:
                        g.text = filename  # raw data file name
            # Set the csv export filename
            if child.attrib[
                'method'] == 'net.sf.mzmine.modules.peaklistmethods.io.csvexport.CSVExportModule':  # TODO: edit / remove
                for e in child:
                    for g in e:
                        tag = g.tag
                        text = g.text
                        if tag == 'current_file' or tag == 'last_file':
                            if add_name is None:
                                csv_name = os.path.join(output_dir,
                                                        filename.split(os.sep)[-1].split('.')[0] + '_pp.csv')
                            else:
                                csv_name = os.path.join(output_dir, filename.split(os.sep)[-1].split('.')[
                                    0] + '_' + add_name + '_pp.csv')
                            g.text = csv_name
        # write the xml file for this input file
        if add_name is None:
            new_xml_name = os.path.join(output_dir, filename.split(os.sep)[-1].split('.')[0] + '.xml')
        else:
            new_xml_name = os.path.join(output_dir, filename.split(os.sep)[-1].split('.')[0] + '_' + add_name + '.xml')
        et.write(new_xml_name)
        # Run mzmine
        logger.info("Running mzMine for {}".format(filename.split(os.sep)[-1]))
        os.system(mzmine_command + ' "{}"'.format(new_xml_name))
