# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:01:55 2021

@author: Daniel Dylewsky

Compute common spatial and temporal EWS statistics on test data set

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import glob


# order_param = 'h'
# order_param = 'temp'
order_param = 'temp_lin'

which_model = 'CNN_LSTM'
# which_model = 'InceptionTime'

# train_coords = 'spatial'
# train_coords = 'temporal'
# train_coords = 'all'
train_coord_list = ['all','temporal','spatial']

# smoothing = None
smoothing = 'gaussian'

# smooth_param = [24,0]
# smooth_param = [48,0]
smooth_param = [96,0]


# mask_type = None
mask_type = 'ellipse'

# model_name = 'wv'
model_name = 'perc'


if model_name == 'wv':
    raw_dir = 'Water Vegetation Model'
elif model_name == 'perc':
    raw_dir = 'Sea Ice Percolation Model'

if mask_type is None:
    model_dir = os.path.join('Ising_Output','var_'+order_param,'Trained Models')
else:
    model_dir = os.path.join('Ising_Output','var_'+order_param+'_'+mask_type,'Trained Models')
# if train_coords == 'temporal':
#     model_dir = os.path.join(model_dir,'Temporal')
# elif train_coords == 'spatial':
#     model_dir = os.path.join(model_dir,'Spatial')


if smoothing == None:
    ews_dir = os.path.join(raw_dir,'Processed')
elif smoothing == 'gaussian':
    ews_dir = os.path.join(raw_dir,'Processed','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
    model_dir = os.path.join(model_dir,'Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
    
    
plot_dir = os.path.join(raw_dir,'Processed','ROC Predictions','Plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

windowed_computation = False

pred_val_dict = {}
true_val_dict = {}

def match_val_nans(all_true_val,all_pred_val):
    # sometimes predictions return NaN, this purges corresponding values from the true_val array
    if not np.all(np.isnan(all_pred_val) == np.isnan(all_true_val)):
        print('warning: non-matching NaNs between true and pred')
        if np.count_nonzero(np.isnan(all_pred_val)) > np.count_nonzero(np.isnan(all_true_val)):
            nan_mask = ~np.isnan(all_pred_val)
        elif np.count_nonzero(np.isnan(all_pred_val)) < np.count_nonzero(np.isnan(all_true_val)):
            nan_mask = ~np.isnan(all_true_val)
        
        all_true_val = all_true_val[nan_mask]
        all_pred_val = all_pred_val[nan_mask]
    else:
        all_true_val = all_true_val[~np.isnan(all_true_val)]
        all_pred_val = all_pred_val[~np.isnan(all_pred_val)]
    return (all_true_val,all_pred_val)

# %% Basic ROC plot
which_HP = [1]
for pj in which_HP:
    
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    
    for train_coords in train_coord_list:
        if windowed_computation:
            data_dir = os.path.join(model_dir,'CNN_LSTM','HP_'+str(pj)+'_'+train_coords,model_name + ' predictions','Windowed') #maybe wrong?
        else:
            data_dir = os.path.join(model_dir,'CNN_LSTM','HP_'+str(pj)+'_'+train_coords,model_name + ' predictions','Full')
        
        ROC_pred_files = glob.glob(os.path.join(data_dir,'*.npz'))
        
        all_pred_val_list = []
        all_true_val_list = []
        for ROC_file in ROC_pred_files:
            try:
                with np.load(ROC_file,allow_pickle=True) as np_load:
                    # all_pred_val = np_load['all_pred_val']
                    block_pred_val = np_load['block_pred_val']
                    block_true_val = np_load['block_true_val']
                    
                    try:
                        steps_to_trans = np_load['steps_to_trans']
                        noise_levels = np_load['noise_levels']
                    except KeyError:
                        import pdb; pdb.set_trace()
                        steps_to_trans = [0]
                        noise_levels = np.power(2,np.linspace(0,1,4))-1
                    
                all_pred_val_list.append(block_pred_val)
                all_true_val_list.append(block_true_val)
            except FileNotFoundError:
                print('No data found for ' + train_coords + ' coords in ' + ROC_file)
                continue
        if len(all_true_val_list)==0:
            continue
        
        if len(block_pred_val.shape) == 3:
            n_classes = 1
        else:
            n_classes = block_pred_val.shape[-1]
            
        if n_classes == 2:
            all_pred_val_list = [1-apv[:,:,:,0] for apv in all_pred_val_list]
        elif n_classes > 2:
            print('No ground truth for n_classes > 2')
            continue
        
        
        all_pred_val = np.array(all_pred_val_list).flatten()
        all_true_val = np.array(all_true_val_list).flatten()
        
        
        
        pred_val_dict[train_coords] = all_pred_val
        true_val_dict[train_coords] = all_true_val
        
        all_true_val,all_pred_val = match_val_nans(all_true_val,all_pred_val)
        
        fpr, tpr, thresholds = roc_curve(all_true_val,all_pred_val)
        auc = roc_auc_score(all_true_val, all_pred_val)
        ax.plot(fpr,tpr,label=train_coords+' coords: AUC = {:.2f}'.format(auc))
        
    n_pos = np.count_nonzero(all_true_val == 1)
    n_neg = np.count_nonzero(all_true_val == 0)
    
    plt.title(model_name + ' classification results\n(from {} pos. and {} neg. runs)'.format(n_pos,n_neg))
    plt.legend()
    print('Saving ' + os.path.join(plot_dir,'ROC_full.png'))
    plt.savefig(os.path.join(plot_dir,'ROC_full.png'))
    
# %% Binned ROC Plots
if model_name == 'wv':
    ews_files = glob.glob(os.path.join(ews_dir,'water_vegetation_*.npz'))
elif model_name == 'perc':
    ews_files = [os.path.join(ews_dir,'processed_EWS.npz')]

var_param_list = ['all_roll_window_frac','all_grid_proportion','all_time_dir','all_run_length']
param_discrete = [0                     ,1                    ,1             ,0               ]
param_numerical= [1                     ,1                    ,0             ,1               ]

if model_name == 'wv':
    var_param_list = ['all_roll_window_frac','all_grid_proportion','all_time_dir','all_run_length']
    param_discrete = [0                     ,1                    ,1             ,0               ]
    param_numerical= [1                     ,1                    ,0             ,1               ]
    var_param_list.extend('all_var_name')
    param_discrete.extend(1)
    param_numerical.extend(0)
else:
    var_param_list = ['roll_window_frac','grid_proportion','time_dir','run_length']
    param_discrete = [0                 ,1                ,1         ,0           ]
    param_numerical= [1                 ,1                ,0         ,1           ]
    

which_HP = [1]

                
model_roc_dicts = []

for pj in which_HP:
    for block, ews_file in enumerate(ews_files):
        if block !=0:
            # only block 0 is computed yet
            continue 
        np_load = np.load(ews_file, allow_pickle=True)
        file_base_name = os.path.split(ews_file)[-1][:-4]
        for vj, vp in enumerate(var_param_list):
            fig, axs = plt.subplots(1,3,figsize=(16,6),sharey=True)
            
            this_qoi = np_load[vp]
            
            if param_discrete[vj]:
                bin_vals = np.unique(this_qoi)
            else:
                bin_vals = np.linspace(np.min(this_qoi),np.max(this_qoi),16)
            
            all_auc = []
            all_bins = []
            all_t = []
            all_p = []
            for tj, train_coords in enumerate(train_coord_list):    
                
                if windowed_computation:
                    data_dir = os.path.join(model_dir,'CNN_LSTM','HP_'+str(pj)+'_'+train_coords,model_name + ' predictions','Windowed') #maybe wrong?
                else:
                    data_dir = os.path.join(model_dir,'CNN_LSTM','HP_'+str(pj)+'_'+train_coords,model_name + ' predictions','Full')

                try:
                    with np.load(os.path.join(data_dir,'ROC_pred_vals_block_'+str(block)+'.npz'),allow_pickle=True) as preds_load:
                        # all_pred_val = np_load['all_pred_val']
                        block_pred_val = preds_load['block_pred_val']
                        block_true_val = preds_load['block_true_val']
                except FileNotFoundError:
                    print('No data found for ' + train_coords + ' coords')
                    continue
                
                if len(block_pred_val.shape) == 3:
                    n_classes = 1
                else:
                    n_classes = block_pred_val.shape[-1]
                    
                if n_classes == 2:
                    block_pred_val = 1-block_pred_val[:,:,:,0]
                elif n_classes > 2:
                    print('No ground truth for n_classes > 2')
                    continue
                
                
                block_pred_val = block_pred_val.flatten()
                block_true_val = block_true_val.flatten()
                
                all_t.append(block_true_val)
                all_p.append(block_pred_val)
                
                # nan_mask = np.isnan(all_pred_val)
                
                # this_qoi_mask = this_qoi(~nan_mask)
                
                if param_discrete[vj]:
                    these_auc = np.zeros(len(bin_vals))
                    for bj, bv in enumerate(bin_vals):
                        these_pred_val = block_pred_val[this_qoi==bv]
                        these_true_val = block_true_val[this_qoi==bv]
                        these_true_val,these_pred_val = match_val_nans(these_true_val,these_pred_val)
                        try:
                            auc = roc_auc_score(these_true_val[~np.isnan(these_true_val)], these_pred_val[~np.isnan(these_pred_val)])
                        except ValueError:
                            # throws error if only one class present in true_val
                            auc = np.nan
                        these_auc[bj] = auc
                        if vp == 'all_model':
                            try:
                                fpr, tpr, thresholds = roc_curve(these_true_val[~np.isnan(these_true_val)],these_pred_val[~np.isnan(these_pred_val)])
                            except ValueError:
                                import pdb; pdb.set_trace()
                            n_pos = np.count_nonzero(these_true_val[~np.isnan(these_true_val)] == 1)
                            n_neg = np.count_nonzero(these_true_val[~np.isnan(these_true_val)] == 0)
                            model_roc_dicts.append({'model':bv,'train_coords':train_coords,'fpr':fpr,'tpr':tpr,'auc':auc,'n_pos':n_pos,'n_neg':n_neg})
                        
                else:
                    these_auc = np.zeros(len(bin_vals)-1)
                    for bj in range(len(bin_vals)-1):
                        these_pred_val = block_pred_val[(this_qoi>bin_vals[bj]) & (this_qoi<=bin_vals[bj+1])]
                        these_true_val = block_true_val[(this_qoi>bin_vals[bj]) & (this_qoi<=bin_vals[bj+1])]
                        these_true_val,these_pred_val = match_val_nans(these_true_val,these_pred_val)
                        try:
                            auc = roc_auc_score(these_true_val[~np.isnan(these_true_val)], these_pred_val[~np.isnan(these_pred_val)])
                        except ValueError:
                            # throws error if only one class present in true_val
                            print('ROC AUC failed on ' + vp + ' ' + train_coords + ' bin ' + str(bj) + '(n=' + str(len(block_pred_val)) + ')')
                            auc = np.nan
                        these_auc[bj] = auc
                all_auc.append(these_auc)
                
                if not param_discrete[vj]:
                    # shift to bin centers
                    bin_vals = bin_vals[:-1]+0.5*(bin_vals[1]-bin_vals[0])
                all_bins.append(bin_vals)
                
                
                if param_numerical[vj]:
                    axs[tj].plot(bin_vals,these_auc,'-')
                else:
                    axs[tj].bar([str(bv) for bv in bin_vals],these_auc)
                    axs[tj].set_xticklabels(bin_vals,rotation=90)
                axs[tj].set_xlabel(vp)
                axs[tj].set_ylim([0,1])
                axs[tj].set_title(train_coords + ' coords')
                if tj == 0:
                    axs[tj].set_ylabel('AUC')
            plt.suptitle('ROC Curves: Varied ' + vp)

            
            print('Saving ' + os.path.join(plot_dir,'ROC_var_' + vp + '.png'))
            plt.savefig(os.path.join(plot_dir,'ROC_var_' + vp + '.png'))
            
