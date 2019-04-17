"Filtering ZTF Avro data packets according to science case. Constructed and adapted using the notebooks at https://github.com/ZwickyTransientFacility/ztf-avro-alert/tree/master/notebooks"

import os
import io
import gzip
import warnings
import argparse

from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
import fastavro

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.time import Time
from astropy.io import fits
import astropy.units as u
import aplpy


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


#Making a pandas dataframe from a packet
def make_dataframe(packet):
    dfc = pd.DataFrame(packet['candidate'], index=[0])
    df_prv = pd.DataFrame(packet['prv_candidates'])
    dflc = pd.concat([dfc,df_prv], ignore_index=True)
    dflc.objectId = packet['objectId']
    dflc.candid = packet['candid']
    return dflc


#Let's use the following cuts to select likely extragalactic transients:

# the difference image detection should be positive
# if there is a PS1 source within 1.5" of the source, it should have a star-galaxy score of < 0.5 (galaxy-like)
# there should be at least two detections separated by more than 30 minutes
# there should be no known solar system object within 5"

def is_transient(dflc):
    
    candidate = dflc.loc[0]
    
    is_positive_sub = candidate['isdiffpos'] == 't'
    
    if (candidate['distpsnr1'] is None) or (candidate['distpsnr1'] > 1.5):
        no_pointsource_counterpart = True
    else:
        if candidate['sgscore1'] < 0.5:
            no_pointsource_counterpart = True
        else:
            no_pointsource_counterpart = False
            
    where_detected = (dflc['isdiffpos'] == 't') # nondetections will be None
    if np.sum(where_detected) >= 2:
        detection_times = dflc.loc[where_detected,'jd'].values
        dt = np.diff(detection_times)
        not_moving = np.max(dt) >= (30*u.minute).to(u.day).value
    else:
        not_moving = False
    
    no_ssobject = (candidate['ssdistnr'] is None) or (candidate['ssdistnr'] < 0) or (candidate['ssdistnr'] > 5)
    
    return is_positive_sub and no_pointsource_counterpart and not_moving and no_ssobject

def is_moving_obj(dflc):
    
    candidate = dflc.loc[0]

    is_positive_sub = candidate['isdiffpos'] == 't'

    if (candidate['distpsnr1'] is None) or (candidate['distpsnr1'] > 1.5):
        no_pointsource_counterpart = True
    else:
        if candidate['sgscore1'] < 0.5:
            no_pointsource_counterpart = True
        else:
            no_pointsource_counterpart = False

    where_detected = (dflc['isdiffpos'] == 't') # nondetections will be None
    if np.sum(where_detected) == 1:
        moving = True
    else:
        moving = False
    
    no_prev_det = True
    nodet = dflc.loc[1:,:].magpsf.isnull()
    if np.sum(nodet):
        no_prev_det = True
    elif len(dflc.index == 1):
        no_prev_det = True
    else:
        no_prev_det = False
    
    no_ssobject = (candidate['ssdistnr'] is None) or (candidate['ssdistnr'] < 0) or (candidate['ssdistnr'] > 5)

    return is_positive_sub and no_pointsource_counterpart and moving and no_prev_det and no_ssobject

#Fetch transient packets from data
def transient_packets(data_dir):
    transient_alerts = []
    for packet in filter(is_alert_pure,generate_dictionaries(data_dir)):
        dflc = make_dataframe(packet)
        if is_transient(dflc):
            transient_alerts.append(packet)
    return transient_alerts

def moving_obj_packets(data_dir):
    moving_obj_alerts = []
    for packet in filter(is_alert_pure,generate_dictionaries(data_dir)):
        dflc = make_dataframe(packet)
        if is_moving_obj(dflc):
            moving_obj_alerts.append(packet)
    return moving_obj_alerts

#Plot lightcurve
def plot_lightcurve(dflc, ax=None, days_ago=True):
    
    filter_color = {1:'green', 2:'red', 3:'pink'}
    if days_ago:
        now = Time.now().jd
        t = dflc.jd - now
        xlabel = 'Days Ago'
    else:
        t = dflc.jd
        xlabel = 'Time (JD)'
    
    if ax is None:
        plt.figure()
    for fid, color in filter_color.items():
        # plot detections in this filter:
        w = (dflc.fid == fid) & ~dflc.magpsf.isnull()
        if np.sum(w):
            plt.errorbar(t[w],dflc.loc[w,'magpsf'], dflc.loc[w,'sigmapsf'],fmt='.',color=color,label='Observed')
        wnodet = (dflc.fid == fid) & dflc.magpsf.isnull()
        if np.sum(wnodet):
            plt.scatter(t[wnodet],dflc.loc[wnodet,'diffmaglim'], marker='v',color=color,alpha=0.25,label='Lim mag for non-obsv')
    
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title(dflc.objectId)

#Plot cutout images
def plot_cutout(stamp, fig=None, subplot=None, **kwargs):
    with gzip.open(io.BytesIO(stamp), 'rb') as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            if fig is None:
                fig = plt.figure(figsize=(4,4))
            if subplot is None:
                subplot = (1,1,1)
            ffig = aplpy.FITSFigure(hdul[0],figure=fig, subplot=subplot, **kwargs)
            ffig.show_grayscale(stretch='arcsinh')
    return ffig

#Plot all cutouts - Science,Template and Difference
def show_stamps(packet):
    #fig, axes = plt.subplots(1,3, figsize=(12,4))
    fig = plt.figure(figsize=(12,4))
    for i, cutout in enumerate(['Science','Template','Difference']):
        stamp = packet['cutout{}'.format(cutout)]['stampData']
        ffig = plot_cutout(stamp, fig=fig, subplot = (1,3,i+1))
        ffig.set_title(cutout)

#Combine all plots in one
def show_all(packet):
    fig = plt.figure(figsize=(16,4))
    dflc = make_dataframe(packet)
    plot_lightcurve(dflc,ax = plt.subplot(1,4,1))
    for i, cutout in enumerate(['Science','Template','Difference']):
        stamp = packet['cutout{}'.format(cutout)]['stampData']
        ffig = plot_cutout(stamp, fig=fig, subplot = (1,4,i+2))
        ffig.set_title(cutout)
    plt.savefig(args.plot_dir+'/'+packet['objectId']+'_lc_cutout_plot.png',format='png')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Plotting module to get plots of transient alerts")
    parser.add_argument('data_dir',type=str,help='Path to the data directory of interest.../project/data')
    parser.add_argument('plot_dir',type=str,help='Path to the plotting directory to store the created plots.../project/plot')
    parser.add_argument('--rb_lim',type=float,help='Real-Bogus limit',default=0.65)
    parser.add_argument('--classtar_lim',type=float,help='Star/galaxy score limit',default=0.8)
    parser.add_argument('--tooflag',type=int,help='Target of Opportunity flag either 1(true) or 0',default=0)
    parser.add_argument('--scorr_lim',type=float,help='SNR limit',default=5)
    parser.add_argument('--nbad_val',type=float,help='Value of nbad, that is the number of bad pixels',default=0)
    parser.add_argument('--fwhm_lim',type=float,help='FWHM limit',default=5)
    parser.add_argument('--elong_lim',type=float,help='Elong limit',default=1.2)
    parser.add_argument('--absmagdiff_lim',type=float,help='Absolute of magdiff limit',default=0.1)

    args = parser.parse_args() 
#    for packet in transient_packets(args.data_dir):
    for packet in moving_obj_packets(args.data_dir):   
       show_all(packet)
            
