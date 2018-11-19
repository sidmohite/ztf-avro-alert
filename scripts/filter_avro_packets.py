"Filtering ZTF Avro data packets according to science case. Constructed and adapted using the notebooks at https://github.com/ZwickyTransientFacility/ztf-avro-alert/tree/master/notebooks"

import os
import argparse
import numpy as np

from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
import fastavro


# Checks for avro files in given data dirctory/sub-directories
def find_files(root_dir):
    for dir_name, subdir_list, file_list in os.walk(root_dir, followlinks=True):
        for fname in file_list:
            if fname.endswith('.avro'):
                yield dir_name+'/'+fname

#Read packets using fastavro
def open_avro(fname):
    with open(fname,'rb') as f:
        freader = fastavro.reader(f)
        # in principle there can be multiple packets per file
        for packet in freader:
            yield packet

#Combine the above two functions to load all packets in data directories
def generate_dictionaries(root_dir):
    for fname in find_files(root_dir):
        for packet in open_avro(fname):
            yield packet

#Filtering the dataset according to science case involves placing limits on certain parameters to get a relatively pure set 
#eg : rb,classtar,tooflag,scorr,nbad,fwhm,elong,abs(magdiff)


def is_alert_pure(packet):
    pure = True
    pure &= packet['candidate']['rb'] >= args.rb_lim
    pure &= packet['candidate']['classtar'] >= args.classtar_lim
    pure &= packet['candidate']['tooflag'] == args.tooflag
    pure &= packet['candidate']['scorr'] >= args.scorr_lim
    pure &= packet['candidate']['nbad'] == args.nbad_val
    pure &= packet['candidate']['fwhm'] <= args.fwhm_lim
    pure &= packet['candidate']['elong'] <= args.elong_lim
    pure &= np.abs(packet['candidate']['magdiff']) <= args.absmagdiff_lim
    return pure

def filter_packets(data_dir):
    filtered_alerts = np.asarray(list(filter(is_alert_pure,generate_dictionaries(data_dir))))
    return filtered_alerts


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Load and Filter Avro Packets")
    parser.add_argument('--data_dir',type=str,help='Path to the data directory of interest.../project/data')
    parser.add_argument('--rb_lim',type=float,help='Real-Bogus limit',default=0.65)
    parser.add_argument('--classtar_lim',type=float,help='Star/galaxy score limit',default=0.8)
    parser.add_argument('--tooflag',type=int,help='Target of Opportunity flag either 1(true) or 0',default=0)
    parser.add_argument('--scorr_lim',type=float,help='SNR limit',default=5)
    parser.add_argument('--nbad_val',type=float,help='Value of nbad, that is the number of bad pixels',default=0)
    parser.add_argument('--fwhm_lim',type=float,help='FWHM limit',default=5)
    parser.add_argument('--elong_lim',type=float,help='Elong limit',default=12)
    parser.add_argument('--absmagdiff_lim',type=float,help='Absolute of magdiff limit',default=0.1)
    parser.add_argument('--output_file',help='Output file to store filtered alerts.The file is stored in the data directory',default='filtered_alerts.npy')
    args = parser.parse_args()
    filtered_alerts = filter_packets(args.data_dir)
    print('Saving {} filtered packets'.format(len(filtered_alerts)))
    np.save(args.data_dir+'/'+args.output_file,filtered_alerts)


