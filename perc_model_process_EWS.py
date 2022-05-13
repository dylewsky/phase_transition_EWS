# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:01:55 2021

@author: Daniel Dylewsky

Compute common spatial and temporal EWS statistics on test data set

"""

import os
import numpy as np
import pandas as pd
import sys

import glob
# import time as tm
# import xarray as xr
import matplotlib.pyplot as plt

# from month_year_sep import month_year_sep
from skimage.measure import block_reduce
# from ising_model_process_data import compute_ews
from scipy.ndimage.filters import gaussian_filter as gf

from ising_model_process_data import compute_ews

raw_dir = 'Sea Ice Percolation Model'

model_name = 'perc'
# model_name = 'perc_local'


    
target_size = 9 # side length of final grid
min_duration = 200
target_duration = 600

# smoothing = None
smoothing = 'gaussian'

smooth_param = [24,0]
# smooth_param = [48,0]
# smooth_param = [96,0]

roll_window_bounds = [0.1,0.4]
filter_fw_grid_proportion = np.arange(1,28)
ts_per_run = 32 # how many distinct EWS time series to pull from each full-grid simulation


if smoothing == 'gaussian':
    processed_dir = os.path.join(raw_dir,'Processed','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
elif smoothing == None:
    processed_dir = os.path.join(raw_dir,'Processed')



print('Compiling all model results')

# computed_EWS_df = pd.read_pickle(os.path.join(out_dir,'CMIP_ews.pkl'))

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# %% Process EWS

all_ts_files = glob.glob(os.path.join(raw_dir,model_name+'_run_*.npz'))
for ts_file in all_ts_files:
# for ts_file in [all_ts_files[-1]]:
    file_out_name_base = os.path.split(ts_file)[-1][:-4]
    print('Computing EWS: ' + ts_file)
    with np.load(ts_file, allow_pickle=True) as np_load:
        this_s = np_load['y_avg']
        
        p_true = np_load['p_true']
        pF = np_load['pF']
        pD = np_load['pD']
        nD = np_load['nD']
        L = np_load['L']
        run_id = np_load['run_id']
        bond_update_rate = np_load['bond_update_rate']
        p_crit = np_load['p_crit']
        # null = np_load['null']
        
    try:
        with np.load(ts_file, allow_pickle=True) as np_load:
            null = np_load['null']
    except KeyError:
        with np.load(ts_file, allow_pickle=True) as np_load:
            p_target = np_load['p_target']
        if np.max(p_target) > p_crit:
            null = 0
        else:
            null = 1
    
    
    sim_length = this_s.shape[0]
    
   
    
    if null == 0:
        p_true_smooth = gf(p_true,16) # smooth jitters out of p_true
        crit_step = np.argmin(np.abs(p_true_smooth-p_crit))
        if crit_step == 0 or crit_step == sim_length-1 or p_crit > np.max(p_true) or p_crit < np.min(p_true):
            print('p_crit outside observed interval: skipping run ' + ts_file)
            continue
    
    
    model_outfile = os.path.join(processed_dir,file_out_name_base + '.npz')
    if os.path.exists(model_outfile):
        print('Skipping ' + ts_file + ': already processed')
        
        continue
    
    file_x = []
    file_grid_proportion = []
    file_null = []
    file_sample_loc = []
    file_run_length = []
    file_roll_window_frac = []
    file_time_dir = []
    file_run_id = []
    file_p_true = []
    
    for ews_ts in range(ts_per_run):
        print('Computing run ' + str(run_id) + ': ' + str(ews_ts) + '/' + str(ts_per_run))
        sample_loc = np.array([np.random.randint(this_s.shape[1]),np.random.randint(this_s.shape[2])])
        roll_window_frac = np.random.uniform(roll_window_bounds[0],roll_window_bounds[1])
        grid_proportion = np.random.choice(filter_fw_grid_proportion)
        # time_dir = np.random.choice([-1,1])
        time_dir = 1
        
        
        this_grid_size = target_size*grid_proportion
        s_roll = np.roll(this_s,(-sample_loc[0],-sample_loc[1]),axis=(1,2))
        s_crop = s_roll[:,:this_grid_size,:this_grid_size]
        if grid_proportion > 1:
            s_crop = block_reduce(s_crop,block_size=(1,grid_proportion,grid_proportion),func=np.nanmean)
            
        this_run_length = np.random.randint(min_duration,target_duration)
        
        if null == 0:
            if time_dir == 1:
                s_crop = s_crop[:crit_step,:,:]
                this_p_true = p_true[:crit_step]
            elif time_dir == -1:
                s_crop = s_crop[crit_step:,:,:]
                s_crop = np.flip(s_crop,axis=0)
                this_p_true = p_true[crit_step:]
                this_p_true = np.flip(this_p_true,axis=0)
        else:
            if time_dir == -1:
                s_crop - np.flip(s_crop,axis=0)
                this_p_true = np.flip(p_true,axis=0)
            else:
                this_p_true = p_true
        
        this_run_length = np.min([s_crop.shape[0],this_run_length])
        
        s_crop = s_crop[-this_run_length:,:,:]
        this_p_true = this_p_true[-this_run_length:]
        
        this_x = compute_ews(s_crop,roll_window_frac,smoothing=smoothing,smooth_param=smooth_param)
        
        this_x_pad = np.zeros((target_duration,this_x.shape[1]))
        this_x_pad[-this_x.shape[0]:,:] = this_x
        
        file_x.append(this_x_pad)
        file_grid_proportion.append(grid_proportion)
        file_null.append(null)
        file_sample_loc.append(sample_loc)
        file_run_length.append(this_run_length)
        file_roll_window_frac.append(roll_window_frac)
        file_time_dir.append(time_dir)
        file_run_id.append(run_id)
        file_p_true.append(this_p_true)
    out_dict = {
        'x' : np.array(file_x),
        'grid_proportion' : np.array(file_grid_proportion),
        'null' : np.array(file_null),
        'sample_loc' : np.array(file_sample_loc),
        'run_length' : np.array(file_run_length),
        'roll_window_frac' : np.array(file_roll_window_frac),
        'time_dir' : np.array(file_time_dir),
        'run_id' : np.array(file_run_id),
        'p_true' : np.array(file_p_true),
        'pF' : pF,
        'pD' : pD,
        'nD' : nD,
        'L' : L,
        'bond_update_rate' : bond_update_rate,
        'p_crit' : p_crit}
    
    
    print('Saving: ' + model_outfile)
    
    np.savez_compressed(model_outfile,**out_dict,allow_pickle=True, fix_imports=True)
    
# %% Consolidate to single file


processed_files = glob.glob(os.path.join(processed_dir,model_name+'_run_*.npz'))
for fj, ts_file in enumerate(processed_files):
    print('Consolidating all EWS {}/{}'.format(fj+1,len(processed_files)))

    with np.load(ts_file, allow_pickle=True) as np_load:
        x = np_load['x']
        grid_proportion = np_load['grid_proportion']
        null = np_load['null']
        sample_loc = np_load['sample_loc']
        run_length = np_load['run_length']
        roll_window_frac = np_load['roll_window_frac']
        time_dir = np_load['time_dir']
        run_id  = np_load['run_id']
        p_true = np_load['p_true']
        pF = np_load['pF']
        pD = np_load['pD']
        nD = np_load['nD']
        L = np.expand_dims(np_load['L'],axis=0)
        bond_update_rate = np.expand_dims(np_load['bond_update_rate'],axis=0)
        p_crit = np.expand_dims(np_load['p_crit'],axis=0)
        
        
    if fj == 0:
        all_x = x
        all_grid_proportion = grid_proportion
        all_null = null
        all_sample_loc = sample_loc
        all_run_length = run_length
        all_roll_window_frac = roll_window_frac
        all_time_dir = time_dir
        all_run_id  = run_id
        all_p_true = p_true
        all_pF = np.tile(pF,x.shape[0])
        all_pD = np.tile(pD,x.shape[0])
        all_nD = np.tile(nD,x.shape[0])
        all_L = np.tile(L,(x.shape[0],1))
        all_bond_update_rate = np.tile(bond_update_rate,x.shape[0])
        all_p_crit = np.tile(p_crit,x.shape[0])
    else:
        all_x = np.concatenate((all_x,x),axis=0)
        all_grid_proportion = np.concatenate((all_grid_proportion,grid_proportion),axis=0)
        all_null = np.concatenate((all_null,null),axis=0)
        all_sample_loc = np.concatenate((all_sample_loc,sample_loc),axis=0)
        all_run_length = np.concatenate((all_run_length,run_length),axis=0)
        all_roll_window_frac = np.concatenate((all_roll_window_frac,roll_window_frac),axis=0)
        all_time_dir = np.concatenate((all_time_dir,time_dir),axis=0)
        all_run_id  = np.concatenate((all_run_id,run_id),axis=0)
        all_p_true = np.concatenate((all_p_true,p_true),axis=0)
        all_pF = np.concatenate((all_pF,np.tile(pF,x.shape[0])),axis=0)
        all_pD = np.concatenate((all_pD,np.tile(pD,x.shape[0])),axis=0)
        all_nD = np.concatenate((all_nD,np.tile(nD,x.shape[0])),axis=0)
        all_L = np.concatenate((all_L,np.tile(L,(x.shape[0],1))),axis=0)
        all_bond_update_rate = np.concatenate((all_bond_update_rate,np.tile(bond_update_rate,x.shape[0])),axis=0)
        all_p_crit = np.concatenate((all_p_crit,np.tile(p_crit,x.shape[0])),axis=0)
        

out_dict = {
        'x' : all_x,
        'grid_proportion' : all_grid_proportion,
        'null' : all_null,
        'sample_loc' : all_sample_loc,
        'run_length' : all_run_length,
        'roll_window_frac' : all_roll_window_frac,
        'time_dir' : all_time_dir,
        'run_id' : all_run_id,
        'p_true' : all_p_true,
        'pF' : all_pF,
        'pD' : all_pD,
        'nD' : all_nD,
        'L' : all_L,
        'bond_update_rate' : all_bond_update_rate,
        'p_crit' : all_p_crit}

print('Saving output')
np.savez_compressed(os.path.join(processed_dir,model_name+'_processed_EWS.npz'),**out_dict,allow_pickle=True, fix_imports=True)