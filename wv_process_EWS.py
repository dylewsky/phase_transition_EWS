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
# from scipy.ndimage.filters import gaussian_filter as gf

from ising_model_process_data import compute_ews

raw_dir = 'Water Vegetation Model'


var_names = ['w','B']
alpha_crit = 1.936

run_time = 1200

    
target_size = 9 # side length of final grid
min_duration = 200
target_duration = 600

# smoothing = None
smoothing = 'gaussian'

# smooth_param = [24,0]
# smooth_param = [48,0]
smooth_param = [96,0]

roll_window_bounds = [0.1,0.4]
filter_fw_grid_proportion = [1,2,3,4]
ts_per_run = 64 # how many distinct EWS time series to pull from each full-grid simulation


if smoothing == 'gaussian':
    processed_dir = os.path.join(raw_dir,'Processed','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
elif smoothing == None:
    processed_dir = os.path.join(raw_dir,'Processed')



print('Compiling all model results')

# computed_EWS_df = pd.read_pickle(os.path.join(out_dir,'CMIP_ews.pkl'))

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

overwrite_existing = True

reprocess_EWS = False
convert_numpy = True

# %% Process EWS

if reprocess_EWS:
    
    ts_df = pd.DataFrame(columns=['x', 'grid_proportion', 'model', 'null','sample_loc', 'run_length','roll_window_frac','time_dir'])
    
    all_ts_files = glob.glob(os.path.join(raw_dir,'*.pkl'))
    for ts_file in all_ts_files:
        file_out_name_base = os.path.split(ts_file)[-1][:-4]
        print('Computing EWS: ' + ts_file)
        this_df = pd.read_pickle(ts_file)
        if this_df.shape[0] != 1:
            print('Reimplement to accommodate multi-row DF')
            continue
        sim_length = this_df[var_names[0]][0].shape[0]
        
        if 'null' not in this_df.columns:
            if sim_length==run_time/2:
                this_null = 1
            elif sim_length==run_time:
                this_null = 0
            else:
                print('Could not fill in missing null')
                continue
        else:
            this_null = this_df['null'][0]
        
        alpha_interval = this_df['alpha_interval'][0]
        alpha_grid = np.linspace(alpha_interval[0],alpha_interval[1],sim_length)
        if this_null == 0:
            crit_step = np.argmin(np.abs(alpha_grid-alpha_crit))
            if crit_step == 0 or crit_step == sim_length-1:
                print('alpha_crit outside alpha_interval: skipping run ' + ts_file)
                continue
        
        
        for var_name in var_names:
            model_outfile = os.path.join(processed_dir,file_out_name_base + '_' + var_name + '.pkl')
            if os.path.exists(model_outfile):
                processed_df = pd.read_pickle(model_outfile)
                if this_df.shape[0] == processed_df.shape[0]:
                    print('Skipping ' + ts_file + ': already processed')
                    continue
                else:
                    model_run_start = processed_df.shape[0]
                    print('Continuing ' + ts_file + ' from run ' + str(model_run_start))
            else:
                model_run_start = 0
                processed_df = pd.DataFrame(columns=['x', 'grid_proportion', 'model', 'null','sample_loc', 'run_length','roll_window_frac','time_dir'])
               
            this_s = this_df[var_name][0]

                
            for ews_ts in range(ts_per_run):
                print('Computing ' + var_name + ': ' + str(ews_ts) + '/' + str(ts_per_run))
                sample_loc = np.array([np.random.randint(this_s.shape[1]),np.random.randint(this_s.shape[2])])
                roll_window_frac = np.random.uniform(roll_window_bounds[0],roll_window_bounds[1])
                grid_proportion = np.random.choice(filter_fw_grid_proportion)
                time_dir = np.random.choice([-1,1])
                
                
                this_grid_size = target_size*grid_proportion
                s_roll = np.roll(this_s,(-sample_loc[0],-sample_loc[1]),axis=(1,2))
                s_crop = s_roll[:,:this_grid_size,:this_grid_size]
                if grid_proportion > 1:
                    s_crop = block_reduce(s_crop,block_size=(1,grid_proportion,grid_proportion),func=np.nanmean)
                    
                this_run_length = np.random.randint(min_duration,target_duration)
                
                if this_null == 0:
                    if time_dir == 1:
                        s_crop = s_crop[:crit_step,:,:]
                    elif time_dir == -1:
                        s_crop = s_crop[crit_step:,:,:]
                        s_crop = np.flip(s_crop,axis=0)
                else:
                    if time_dir == -1:
                        s_crop - np.flip(s_crop,axis=0)            
                
                this_run_length = np.min([s_crop.shape[0],this_run_length])
                
                s_crop = s_crop[-this_run_length:,:,:]
                
                this_x = compute_ews(s_crop,roll_window_frac,smoothing=smoothing,smooth_param=smooth_param)
                row_dict = {'x':this_x,
                            'grid_proportion':grid_proportion,
                            'null':this_null,
                            'sample_loc':sample_loc,
                            'run_length':this_run_length,
                            'roll_window_frac':roll_window_frac,
                            'time_dir':time_dir,
                            'var_name':var_name}
                processed_df = processed_df.append(row_dict,ignore_index=True)
                    
            print('Saving final: ' + model_outfile)
            processed_df.to_pickle(model_outfile)
    
# %% Convert to numpy

# increase value of mem_blocks to split up .npz files in the event of out-of-memory error

if convert_numpy:
    
    computed_EWS_df = pd.DataFrame(columns=['x', 'grid_proportion', 'model', 'null','sample_loc',
                                            'run_length','roll_window_frac','time_dir'])
    
    all_ews_files = glob.glob(os.path.join(processed_dir,'*.pkl'))
    print('Reading DFs in: {} files'.format(len(all_ews_files)))

    sys.stdout.flush()
    for ews_file in all_ews_files:
        this_df = pd.read_pickle(ews_file)
        computed_EWS_df = pd.concat([computed_EWS_df,this_df],ignore_index=True,sort=False)
        
    computed_EWS_df = computed_EWS_df.sample(frac=1).reset_index(drop=True) # randomly shuffle rows
        
    mem_blocks = 1 # how many files to split data into to avoid cluster OOM event
    block_size = int(computed_EWS_df.shape[0]/mem_blocks)
    block_sizes = block_size*np.ones(mem_blocks,dtype=int)
    block_sizes[-1] = computed_EWS_df.shape[0]-np.sum(block_sizes[:-1])
    block_inds = [np.arange(block_sizes[0])]
    for block in range(1,mem_blocks):
        start_ind = block_inds[block-1][-1]+1
        block_inds.append(np.arange(start_ind,start_ind+block_sizes[block]))
        
    
    
    for block in range(mem_blocks):
        if mem_blocks == 1:
            outfile = os.path.join(processed_dir,'water_vegetation_EWS.npz')
        else:
            outfile = os.path.join(processed_dir,'water_vegetation_EWS_block_' + str(block)+ '.npz')
        if overwrite_existing == False:
            if os.path.exists(outfile):
                print('Skipping processing for ' + outfile + ': Already exists')
                continue
        print('Allocating matrices for block ' + str(block))
        sys.stdout.flush()
    
        all_x = np.zeros((block_sizes[block],target_duration,computed_EWS_df['x'][0].shape[1]))
        all_roll_window_frac = np.zeros(block_sizes[block])
        all_grid_proportion = np.zeros(block_sizes[block])
        all_model = np.zeros(block_sizes[block],dtype=object)
        all_var_name = np.zeros(block_sizes[block],dtype=object)
        all_null = np.zeros(block_sizes[block],dtype=int)
        all_time_dir = np.zeros(block_sizes[block],dtype=int)
        all_sample_loc = np.zeros(block_sizes[block],dtype=np.ndarray)
        all_run_length = np.zeros(block_sizes[block],dtype=int)
        
        for jb, jj in enumerate(block_inds[block]):
            this_x_pad = np.zeros((target_duration,computed_EWS_df['x'][0].shape[1]))
            this_x_pad[-int(computed_EWS_df.loc[jj,'run_length']):,:] = computed_EWS_df.loc[jj,'x']
            all_x[jb,:,:] = this_x_pad
            all_roll_window_frac[jb] = computed_EWS_df.loc[jj,'roll_window_frac']
            all_grid_proportion[jb] = computed_EWS_df.loc[jj,'grid_proportion']
            all_model[jb] = computed_EWS_df.loc[jj,'model']
            all_var_name[jb] = computed_EWS_df.loc[jj,'var_name']
            all_null[jb] = computed_EWS_df.loc[jj,'null']
            all_sample_loc[jb] = computed_EWS_df.loc[jj,'sample_loc']
            all_run_length[jb] = computed_EWS_df.loc[jj,'run_length']
            all_time_dir[jb] = computed_EWS_df.loc[jj,'time_dir']
            
        out_dict = {'all_x':all_x,
                    'all_roll_window_frac':all_roll_window_frac,
                    'all_grid_proportion':all_grid_proportion,
                    'all_model':all_model,
                    'all_var_name':all_var_name,
                    'all_null':all_null,
                    'all_sample_loc':all_sample_loc,
                    'all_time_dir':all_time_dir,
                    'all_run_length':all_run_length}
        
        
        
        print('Saving ' + outfile)
        np.savez_compressed(outfile,**out_dict,allow_pickle=True, fix_imports=True)
