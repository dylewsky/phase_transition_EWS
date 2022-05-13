# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:01:55 2021

@author: Daniel Dylewsky

Compute common spatial and temporal EWS statistics on test data set

"""

import os
import numpy as np
import time
import glob

from tensorflow.keras.models import load_model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# order_param = 'h_lin'
# order_param = 'temp'
order_param = 'temp_lin'
# order_param = 'ht_lin'
# order_param = 'temp_local'

if order_param == 'temp_local':
    mask_type = None
else:
    # mask_type = None
    mask_type = 'ellipse'



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

temporal_coords = np.arange(6)
spatial_coords = np.arange(6,12)

recompute_predictions = True


# model_name = 'wv' # water-vegetation model
model_name = 'perc' # sea ice percolation model


if model_name == 'wv':
    raw_dir = 'Water Vegetation Model'
elif model_name == 'perc':
    raw_dir = 'Sea Ice Percolation Model'


if mask_type is None:
    base_dir = os.path.join('Ising_Output','var_'+order_param)
else:
    base_dir = os.path.join('Ising_Output','var_'+order_param+'_'+mask_type)
# if train_coords == 'temporal':
#     model_dir = os.path.join(model_dir,'Temporal')
# elif train_coords == 'spatial':
#     model_dir = os.path.join(model_dir,'Spatial')


if smoothing == None:
    data_dir = os.path.join(raw_dir,'Processed')
    model_dir = os.path.join(base_dir,'Trained Models')
elif smoothing == 'gaussian':
    data_dir = os.path.join(raw_dir,'Processed','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
    model_dir = os.path.join(base_dir,'Trained Models','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))

out_dir = os.path.join(data_dir,'ROC Predictions')
  
    
if mask_type is None:
    out_dir = os.path.join(out_dir,order_param)
else:
    out_dir = os.path.join(out_dir,order_param+'_'+mask_type)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# pj = 1

# if model_type == 'CNN_LSTM':
#     hp_dir = os.path.join(out_dir,'CNN_LSTM','HP_'+ str(pj))
# elif model_type == 'Inception_LSTM':
#     hp_dir = os.path.join(out_dir,'INC_LSTM','HP_'+ str(pj))
       
# model = load_model(hp_dir)

    
target_duration = 600

# steps_to_trans = np.arange(0,192,32)
# # steps_to_trans = [0]
# # noise_levels = np.power(2,np.linspace(0,1,6))-1
# noise_levels = np.power(2,np.linspace(0,1,4))-1
steps_to_trans = np.arange(0,192,32)
noise_levels = np.power(2,np.linspace(0,1,6))-1

n_test_runs = 1000

# model_year_length_dict = {'ACCESS1-3': 365,
#                          'MIROC-ESM-CHEM': 365.25,
#                          'GFDL-ESM2M': 365,
#                          'GISS-E2-R': 365,
#                          'FGOALS-g2': 365,
#                          'CSIRO-Mk3-6-0': 365.25,
#                          'MIROC5': 365,
#                          'CMCC-CESM': 365}


# model_start_time_dict = {'ACCESS1-3': np.zeros(2),
#                          'MIROC-ESM-CHEM': np.ones(2)*1850,
#                          'GFDL-ESM2M': np.array([1861,2006]),
#                          'GISS-E2-R': np.nan,
#                          'FGOALS-g2': np.zeros(2),
#                          'CSIRO-Mk3-6-0': np.ones(2)*1850,
#                          'MIROC5': np.ones(2)*1850,
#                          'CMCC-CESM': np.nan}

# model_names = ['ACCESS1-3','MIROC-ESM-CHEM','GFDL-ESM2M','GISS-E2-R','FGOALS-g2',
#                'CSIRO-Mk3-6-0','MIROC5','CMCC-CESM']
   
experiment_name = 'rcp85'

if model_name == 'wv':
    ews_files = glob.glob(os.path.join(data_dir,'water_vegetation_*.npz'))
elif model_name == 'perc':
    ews_files = [os.path.join(data_dir,model_name+'_processed_EWS.npz')]



test_blocks = 1 # how many data blocks to include in testing


if len(ews_files) > test_blocks:
    print('Computing predictions for {} blocks ({} available)'.format(test_blocks,len(ews_files)))
    ews_files = ews_files[:test_blocks]
    
    
which_HP = [1]
for pj in which_HP:
    
    all_pred_val_full = []
    all_true_val_full = []
    
    all_pred_val_spatial = []
    all_true_val_spatial = []
    
    all_pred_val_temporal = []
    all_true_val_temporal = []
    # Process
    
    this_out_dir = os.path.join(out_dir,'HP_'+str(pj))
    if not os.path.exists(this_out_dir):
        os.makedirs(this_out_dir)
        
        
    model_list = []
    for train_coords in train_coord_list:
        if which_model == 'CNN_LSTM':
            hp_dir = os.path.join(model_dir,'CNN_LSTM','HP_'+ str(pj)) + '_' + train_coords
        elif which_model == 'InceptionTime':
            hp_dir = os.path.join(model_dir,'CNN_INC','HP_'+ str(pj)) + '_' + train_coords
        if recompute_predictions:        
            try:
                if which_model == 'CNN_LSTM':
                    model_list.append(load_model(hp_dir))
                elif which_model == 'InceptionTime':
                    if pj > 0:
                        # No multiple hyperparameter variations for inceptiontime model
                        break
                    model_list.append(os.path.join(hp_dir,'best_model.hdf5'))
            except:
                print('No valid model in directory ' + hp_dir)
                continue
        
    for ej, ews_file in enumerate(ews_files):
        file_base_name = os.path.split(ews_file)[-1][:-4]
        # CMIP_ews = pd.read_pickle(os.path.join(data_dir,'CMIP_ews.pkl'))
        if model_name == 'wv':
            with np.load(ews_file, allow_pickle=True) as np_load:
                all_x = np_load['all_x']
                all_roll_window_frac = np_load['all_roll_window_frac']
                all_grid_proportion = np_load['all_grid_proportion']
                all_model = np_load['all_model']
                all_var_name = np_load['all_var_name']
                all_null = np_load['all_null']
                all_time_dir = np_load['all_time_dir']
                all_sample_loc = np_load['all_sample_loc']
                all_run_length = np_load['all_run_length']
        elif model_name == 'perc':
            with np.load(ews_file, allow_pickle=True) as np_load:
                all_x = np_load['x']
                all_roll_window_frac = np_load['roll_window_frac']
                all_grid_proportion = np_load['grid_proportion']
                all_null = np_load['null']
                all_time_dir = np_load['time_dir']
                all_sample_loc = np_load['sample_loc']
                all_run_length = np_load['run_length']
        
        if all_x.shape[0] > n_test_runs:
            which_runs = np.random.choice(np.arange(all_x.shape[0]),size=n_test_runs,replace=False)
            if model_name == 'wv':
                all_x = all_x[which_runs,:,:]
                all_roll_window_frac = all_roll_window_frac[which_runs]
                all_grid_proportion = all_grid_proportion[which_runs]
                all_model = all_model[which_runs]
                all_var_name = all_var_name[which_runs]
                all_null = all_null[which_runs]
                all_time_dir = all_time_dir[which_runs]
                all_sample_loc = all_sample_loc[which_runs]
                all_run_length = all_run_length[which_runs]
            elif model_name == 'perc':
                all_x = all_x[which_runs,:,:]
                all_roll_window_frac = all_roll_window_frac[which_runs]
                all_grid_proportion = all_grid_proportion[which_runs]
                all_null = all_null[which_runs]
                all_time_dir = all_time_dir[which_runs]
                all_sample_loc = all_sample_loc[which_runs]
                all_run_length = all_run_length[which_runs]
                
        # NaN out all masked values
        x_mask = np.array((all_x != 0),dtype=float)
        x_mask[x_mask==0] = np.nan
        all_x = np.multiply(all_x,x_mask) 
        
        # subtract off mean of each feature
        all_x = all_x - np.tile(np.expand_dims(np.nanmean(all_x,axis=(1)),1),(1,all_x.shape[1],1))
        
        # normalize to unit std dev
        all_x = np.divide(all_x,np.tile(np.expand_dims(np.nanstd(all_x,axis=(1)),1),(1,all_x.shape[1],1)))
        
        all_x = np.nan_to_num(all_x) # convert NaNs back to zeros
        
        saturation_val = 100
    
        all_x[all_x>saturation_val] = saturation_val
        all_x[all_x<-saturation_val] = -saturation_val
        
        
        for tj,train_coords in enumerate(train_coord_list):

            
            if which_model == 'CNN_LSTM':
                hp_dir = os.path.join(model_dir,'CNN_LSTM','HP_'+ str(pj)) + '_' + train_coords
            elif which_model == 'InceptionTime':
                hp_dir = os.path.join(model_dir,'CNN_INC','HP_'+ str(pj)) + '_' + train_coords
                
            print('Testing ' + train_coords + ' coordinates on model in ' + hp_dir)
            
            pred_base_dir = os.path.join(hp_dir,model_name + ' predictions')

            if recompute_predictions:        
        
                # try:
                #     if which_model == 'CNN_LSTM':
                #         tf_model = load_model(hp_dir)
                #     elif which_model == 'InceptionTime':
                #         if pj > 0:
                #             # No multiple hyperparameter variations for inceptiontime model
                #             break
                #         tf_model = load_model(os.path.join(hp_dir,'best_model.hdf5'))
                # except:
                #     print('No valid model in directory ' + hp_dir)
                #     continue
                # # hist_df = pd.read_pickle(os.path.join(hp_dir,'training_history.pkl'))
                
                tf_model = model_list[tj]
                
                
                if not os.path.exists(pred_base_dir):
                    os.makedirs(pred_base_dir)
                
                n_classes = tf_model.layers[-1].output_shape[-1]
                
                block_pred_val = np.nan*np.ones((len(steps_to_trans),len(noise_levels),all_x.shape[0],n_classes))
                block_true_val = np.nan*np.ones((len(steps_to_trans),len(noise_levels),all_x.shape[0])) # assume orders of phase transitions are not known
                
                for jj in range(all_x.shape[0]):
                # for subseries_ind in data_set_subseries[13:]:
                    
                    start_time = time.process_time()
                    
                    x = np.squeeze(all_x[jj,:,:])
                    null = all_null[jj]
                    roll_window_frac = all_roll_window_frac[jj]
                    grid_proportion = all_grid_proportion[jj]
                    time_dir = all_time_dir[jj]
                    sample_loc = all_sample_loc[jj]
                    run_length = all_run_length[jj]
                    if model_name == 'wv':
                        var_name = all_var_name[jj]
                    
                    # print('Processing CMIP run {}/{}: CMIP_run_name'.format(jj,all_x.shape[0]))
                    
                    if train_coords == 'temporal':
                        x = x[:,temporal_coords]
                    elif train_coords == 'spatial':
                        x = x[:,spatial_coords]
                    
                    # if run_length < target_duration:
                    #     s_pad = np.zeros((target_duration,s.shape[1]))
                    #     s_pad[-run_length:,:] = s
                    
                    # s = s_pad
                    
                    # this_test_x = np.zeros((1,x.shape[0],x.shape[1]))
                    
                    for lj, ls in enumerate(steps_to_trans):
                        
                        if ls == 0:
                            this_test_x = np.expand_dims(x[-target_duration:,:], 0)
                        else:
                            this_test_x = np.expand_dims(x[-target_duration:,:], 0)
                            this_test_x_pad = np.zeros(this_test_x.shape)
                            this_test_x_pad[:,ls:,:] = this_test_x[:,:-ls,:]
                            this_test_x = this_test_x_pad
                        
                        for nj, ns in enumerate(noise_levels):
                            this_test_x_noise = this_test_x + ns*np.random.randn(*this_test_x.shape)
                            this_pred = tf_model.predict(this_test_x_noise)[0,:]
                            # plt.plot(np.squeeze(this_test_x[0,:,:]))
                            # plt.title('Pred = ' + str(this_pred))
                            # plt.show()
                            
                            # pred_time.append(test_t_pad[j+w_width])
                            # pred_val.append(this_pred)
                            
                            if np.any(np.isnan(this_pred)):
                                print('NaN predicted: discarding run')
                            else:
                                block_pred_val[lj,nj,jj,:] = this_pred
                                block_true_val[lj,nj,jj] = 1-null
                        
                    # print('Completed run in time ' + str(time.process_time() - start_time) + ' s')
                    if jj % 100 == 0:
                        print('Completed ' + model_name + ' block ' + str(ej) + ', run {}/{}: '.format(jj,all_x.shape[0]))
                ROC_output_dict = {'block_pred_val':block_pred_val,
                                       'block_true_val':block_true_val
                                       }
                print('Saving ' + os.path.join(pred_base_dir,'ROC_pred_vals_block_'+str(ej)+'.npz'))
                np.savez_compressed(os.path.join(pred_base_dir,'ROC_pred_vals_block_'+str(ej)+'.npz'),**ROC_output_dict,
                                    allow_pickle=True, fix_imports=True)
                
                
                    
            else: # if recompute == false
                with np.load(os.path.join(pred_base_dir,'ROC_pred_vals_block_'+str(ej)+'.npz'),
                             allow_pickle=True) as np_load:
                    block_pred_val = np_load['block_pred_val']
                    block_true_val = np_load['block_true_val']
                    
            if train_coords == 'all':
                all_pred_val_full.append(block_pred_val)
                all_true_val_full.append(block_true_val)
            elif train_coords == 'spatial':
                all_pred_val_spatial.append(block_pred_val)
                all_true_val_spatial.append(block_true_val)
            elif train_coords == 'temporal':
                all_pred_val_temporal.append(block_pred_val)
                all_true_val_temporal.append(block_true_val)
                
    print('Saving final results to ' + this_out_dir)
    
    
    all_pred_val_full = np.array(all_pred_val_full)
    all_true_val_full = np.array(all_true_val_full)
    ROC_output_dict_full = {'all_pred_val':all_pred_val_full,
                        'all_true_val':all_true_val_full,
                        'steps_to_trans':steps_to_trans,
                        'noise_levels':noise_levels
                        }
    np.savez_compressed(os.path.join(this_out_dir,'ROC_pred_vals_all.npz'),**ROC_output_dict_full,
                        allow_pickle=True, fix_imports=True)
    
    all_pred_val_spatial = np.array(all_pred_val_spatial)
    all_true_val_spatial = np.array(all_true_val_spatial)
    ROC_output_dict_spatial = {'all_pred_val':all_pred_val_spatial,
                        'all_true_val':all_true_val_spatial,
                        'steps_to_trans':steps_to_trans,
                        'noise_levels':noise_levels
                        }
    np.savez_compressed(os.path.join(this_out_dir,'ROC_pred_vals_spatial.npz'),**ROC_output_dict_spatial,
                        allow_pickle=True, fix_imports=True)
    
    all_pred_val_temporal = np.array(all_pred_val_temporal)
    all_true_val_temporal = np.array(all_true_val_temporal)
    ROC_output_dict_temporal = {'all_pred_val':all_pred_val_temporal,
                        'all_true_val':all_true_val_temporal,
                        'steps_to_trans':steps_to_trans,
                        'noise_levels':noise_levels
                        }
    np.savez_compressed(os.path.join(this_out_dir,'ROC_pred_vals_temporal.npz'),**ROC_output_dict_temporal,
                        allow_pickle=True, fix_imports=True)



        
