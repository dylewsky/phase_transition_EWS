# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:29:59 2021

@author: Daniel Dylewsky

Process raw Ising model output:
For a number of spatial coarse grainings and translational shifts, compute
secondary EWS statistics (variance,skew,kurtosis,autocorrelation) in space
and time. 

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


from scipy.stats import skew, kurtosis
from skimage.measure import block_reduce
from sklearn.linear_model import LinearRegression
from scipy.ndimage.filters import gaussian_filter as gf

from morans import morans



def compute_abruptness(s,k,width=16):
    this_s_lhs = s[(k-width+1):(k+1)]
    this_s_rhs = s[k:(k+width)]

    this_lhs_model = LinearRegression(fit_intercept=True)
    this_lhs_x = np.arange(this_s_lhs.shape[0]).reshape(-1,1)
    this_lhs_model.fit(this_lhs_x,this_s_lhs.reshape(-1,1))
    lhs_pred = this_lhs_model.predict(this_lhs_x[-1].reshape(-1,1))[0][0]
    
    this_rhs_model = LinearRegression(fit_intercept=True)
    this_rhs_x = np.arange(this_s_rhs.shape[0]).reshape(-1,1)
    this_rhs_model.fit(this_rhs_x,this_s_rhs.reshape(-1,1))
    rhs_pred = this_rhs_model.predict(this_rhs_x[0].reshape(-1,1))[0][0]
    
    
    lhs_inds = np.arange((k-width+1),(k+1))
    rhs_inds = np.arange(k,(k+width))
    lhs_pred_all = this_lhs_model.predict(this_lhs_x.reshape(-1,1)).flatten()
    rhs_pred_all = this_rhs_model.predict(this_rhs_x.reshape(-1,1)).flatten()
    plt.plot(s,'k')
    plt.plot(lhs_inds,lhs_pred_all,'b')
    plt.plot(rhs_inds,rhs_pred_all,'g')
    plt.plot(lhs_inds[-1],lhs_pred,'bo')
    plt.plot(rhs_inds[0],rhs_pred,'go')
    plt.show()

        
    this_offset = np.abs(rhs_pred-lhs_pred)
    sigma_lhs = np.std(this_s_lhs)
    sigma_rhs = np.std(this_s_rhs)
    sigma_mean = (sigma_lhs+sigma_rhs)/2
    
    abruptness = this_offset/sigma_mean
    return abruptness

def matrix_autocorr(s_flatten,lag):
    sbar = np.mean(s_flatten,axis=0)
    s_full = s_flatten - np.tile(sbar,(s_flatten.shape[0],1))
    
    s_0 = s_full[:s_flatten.shape[0]-lag,:]
    s_shift = s_full[lag:s_flatten.shape[0],:]
    
    ac_num = np.nansum(np.multiply(s_0,s_shift),axis=0)
    ac_denom = np.nansum(np.multiply(s_full,s_full),axis=0)
    
    ac = np.divide(ac_num,ac_denom)
    
    return ac
    
    

def temporal_ews(s,t_roll_window):
    s = s.reshape(s.shape[0],-1)
    
    t_var = np.zeros(s.shape)
    t_skew = np.zeros(s.shape)
    t_kurt = np.zeros(s.shape)
    t_corr_1 = np.zeros(s.shape)
    t_corr_2 = np.zeros(s.shape)
    t_corr_3 = np.zeros(s.shape)
    
    for j in range(s.shape[0]-t_roll_window):
        window_end = j+t_roll_window
        s_window = s[j:window_end,:]
        
        t_var[window_end,:] = np.nanvar(s_window,axis=0)
        t_skew[window_end,:] = skew(s_window,axis=0,nan_policy='omit')
        t_kurt[window_end,:] = kurtosis(s_window,axis=0,nan_policy='omit')
        
        t_corr_1[window_end,:] = matrix_autocorr(s_window,1)
        t_corr_2[window_end,:] = matrix_autocorr(s_window,2)
        t_corr_3[window_end,:] = matrix_autocorr(s_window,3)

    
    return {'t_var':t_var,
            't_skew':t_skew,
            't_kurt':t_kurt,
            't_corr_1':t_corr_1,
            't_corr_2':t_corr_2,
            't_corr_3':t_corr_3}
    
    
def compute_ews(s,t_roll_window_frac, smoothing=None, smooth_param=[20,0]):
    # s has dims [t,x,y]
    #
    # for gaussian filter, smooth param is sigma in [temporal,spatial] dimensions
    #
    # as of now filter is only applied when computing temporal EWS -> if smooth_param[1] != 0,
    # this will need to be reimplemented to achieve desired behavior
    
    # Temporal EWS:
    t_roll_window = int(np.floor(t_roll_window_frac*s.shape[0]))
    if smoothing == 'gaussian':
        gaussian_sigma = [smooth_param[0],smooth_param[1],smooth_param[1]] # duplicate spatial sigma to both spatial dimensions
        s_smooth = gf(s, sigma=gaussian_sigma, mode='reflect')
        s_detrend = s - s_smooth
        t_ews = temporal_ews(s_detrend,t_roll_window)
    elif smoothing is None:
        t_ews = temporal_ews(s,t_roll_window)
    else:
        print('Invalid smoothing option')
        return
    
   

    t_var = np.nanmean(t_ews['t_var'],axis=1)
    t_skew = np.nanmean(t_ews['t_skew'],axis=1)
    t_kurt = np.nanmean(t_ews['t_kurt'],axis=1)
    t_corr_1 = np.nanmean(t_ews['t_corr_1'],axis=1)
    t_corr_2 = np.nanmean(t_ews['t_corr_2'],axis=1)
    t_corr_3 = np.nanmean(t_ews['t_corr_3'],axis=1)
    
    # Spatial EWS:
    s_flatten = s.reshape(s.shape[0],-1)
    s_flatten = s_flatten[t_roll_window:,:]
    
    x_var = np.zeros(s.shape[0])
    x_skew = np.zeros(s.shape[0])
    x_kurt = np.zeros(s.shape[0])
    x_corr_1 = np.zeros(s.shape[0])
    x_corr_2 = np.zeros(s.shape[0])
    x_corr_3 = np.zeros(s.shape[0])
    
    x_var[t_roll_window:] = np.nanvar(s_flatten,axis=1)
    x_skew[t_roll_window:] = skew(s_flatten,axis=1,nan_policy='omit')
    x_kurt[t_roll_window:] = kurtosis(s_flatten,axis=1,nan_policy='omit')

    
    x_corr_1[t_roll_window:] = morans(s[t_roll_window:,:,:],1,periodic=False)
    x_corr_2[t_roll_window:] = morans(s[t_roll_window:,:,:],2,periodic=False)
    x_corr_3[t_roll_window:] = morans(s[t_roll_window:,:,:],3,periodic=False)
    
    x = np.vstack((t_var,t_skew,t_kurt,t_corr_1,t_corr_2,t_corr_3,
                   x_var,x_skew,x_kurt,x_corr_1,x_corr_2,x_corr_3))
    
    return x.T
    

def process_ising_data(infile,order_param,smoothing=None,smooth_param=[0,0],output_raw=True,output_ews=False):
    
    in_dir, file_name = os.path.split(infile)
    file_name = file_name[:-4]
    if smoothing == None:
        processed_dir = os.path.join(in_dir,'Processed')
        # if order_param == 'temp_lin':
        #     data_dir_2 = os.path.join('Ising_Output','var_temp','Processed')
    elif smoothing == 'gaussian':
        processed_dir = os.path.join(in_dir,'Processed','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
        
    out_dir_raw = os.path.join(processed_dir,'Raw')
    out_dir_ews = os.path.join(processed_dir,'EWS')
    
    if not os.path.exists(out_dir_raw):
        os.makedirs(out_dir_raw)
    if not os.path.exists(out_dir_ews):
        os.makedirs(out_dir_ews)
    
    
    target_size = 9 # side length of final grid
    target_duration = 600
    # spatial_coarse_grains = [10,12,14]
    spatial_coarse_grains = np.arange(4,15)
    # spatial_coarse_grains = [8]
    noise_sigma = 0.01
    
    df_cols = ['run_id','null','x','time_dir','spatial_cg','xshift','yshift','t_roll_window']
    # ews_list = ['t_var','t_skew','t_kurt','t_corr_1','t_corr_2','t_corr_3',
    #             'x_var','x_skew','x_kurt','x_corr_1','x_corr_2','x_corr_2']
    
    t_roll_window_range = [0.1, 0.4]
    crop_fraction_range = [0.1,0.7]
    


    file_df_ews = pd.DataFrame(columns=df_cols)
    print('Processing ' + infile)
    run_id = os.path.split(infile)[-1][:-4]

    with np.load(infile, allow_pickle=True) as np_load:
        s = np_load['s']
        Tc = np_load['Tc']
        null = np_load['null']
        # Hc = np_load['Hc']
        if np.array(null).ndim == 0:
            Tbounds = np_load['Tbounds']
        else:
            bounds_null = np_load['bounds_null']
            bounds_trans = np_load['bounds_trans']
            spike_loc = np_load['spike_loc']
            spike_width = np_load['spike_width']
            crit_steps = np_load['crit_steps']
        # hbounds = np_load['hbounds']
        magnetization = np_load['magnetization']
        
        # run_id = np_load['run_id']
    
    if max(spatial_coarse_grains)*target_size > s.shape[1]:
        print('spatial coarse grain parameter too large')
        import pdb; pdb.set_trace()
    
    if np.array(null).ndim == 0:
        
        temps = np.linspace(Tbounds[0],Tbounds[1],s.shape[0])
        # fields = np.linspace(hbounds[0],hbounds[1],s.shape[0])
        
        if null == 0:
            if order_param == 'temp' or order_param == 'temp_lin':
                crit_step = [np.argmin(np.abs(temps-Tc))]
                if np.abs(temps[crit_step]-Tc) > np.abs(temps[1]-temps[0]):
                    print('Critical temperature not present')
                    import pdb; pdb.set_trace()
            elif order_param == 'h' or order_param == 'h_lin':
                mag_smooth = gf(magnetization,16)
                peak_step = np.argmax(np.abs(np.diff(mag_smooth)))
                peak_val = np.max(np.abs(np.diff(mag_smooth)))
                # find where on either side of the peak abs(diff(mag)) has value of (1/e**2)*peak value
                crit_step_l = np.argmin(np.abs((peak_val/np.e**2)-np.abs(np.diff(mag_smooth[:peak_step]))))
                crit_step_r = peak_step + np.argmin(np.abs((peak_val/np.e**2)-np.abs(np.diff(mag_smooth[peak_step:]))))
                crit_step = [crit_step_l,crit_step_r]
                
                if ((crit_step_r - peak_step) > 300) or ((peak_step - crit_step_l) > 300):
                    print('large gap between crit and peak step')
                    # print('outputting debug plot to ' + outfile[:-4]+'_CRIT_STEP_ERROR_PLOT.png')
                    # # plt.plot(magnetization)
                    # # plt.title('magnetization')
                    # # plt.show()
                    # plt.plot(np.abs(np.diff(mag_smooth)))
                    # plt.axvline(peak_step,c='r')
                    # plt.axvline(crit_step_l,c='b')
                    # plt.axvline(crit_step_r,c='b')
                    # plt.title('np.abs(np.diff(mag_smooth))')
                    # plt.savefig(outfile[:-4]+'_CRIT_STEP_ERROR_PLOT.png')
                    # plt.close()
                    import pdb; pdb.set_trace()
        else:
            crit_step = [0]
                
            # plt.imshow(s[crit_step,:,:],vmin=-1,vmax=1)
            # plt.title('Original s')
            # plt.colorbar()
            # plt.show()
        
        if output_raw: # only implemented for global order params
            gaussian_sigma = [smooth_param[0],smooth_param[1],smooth_param[1]] # duplicate spatial sigma to both spatial dimensions
            s_smooth = gf(s, sigma=gaussian_sigma, mode='reflect')
            s_detrend = s - s_smooth
            
    
            this_crit_step = crit_step[0]
                
            # crop time domain
            if null == 0:
                s_crop = s_detrend[:this_crit_step,:,:]
            else:
                s_crop = s_detrend
            if s_crop.shape[0] > target_duration:
                s_pad = s_crop[-target_duration:,:,:]
            else:
                s_pad = np.zeros((target_duration,s.shape[1],s.shape[2]))
                s_pad[-s_crop.shape[0]:,:,:] = s_crop
            
            row_dict = {'run_id':run_id,
                        'null':null,
                        'x':s_pad,
                        'Tbounds':Tbounds}
            
            # file_df_ews = pd.DataFrame(data=row_dict)
            # file_df_ews.to_pickle(os.path.join(out_dir_raw,file_name+'.pkl'))
            np.savez_compressed(os.path.join(out_dir_raw,file_name+'.npz'),**row_dict,
                            allow_pickle=True, fix_imports=True)
    
    if output_ews:
        # for scg in np.random.choice(spatial_coarse_grains,size=3,replace=False):
        for scg in spatial_coarse_grains:
            s_roll_displacement = (1,np.random.randint(s.shape[1]),np.random.randint(s.shape[2]))
            s_roll = s.copy()
            s_roll = np.roll(s_roll,s_roll_displacement,axis=(0,1,2)) # for each spatial coarse graining, start with a different rolled permutation
            
            if np.array(null).ndim == 0:
                x_sample_locs = np.arange(0,3*scg,int(np.round(3*scg/2)))
                y_sample_locs = np.arange(0,3*scg,int(np.round(3*scg/2)))
            else:
                x_sample_locs = np.arange(s_roll.shape[1])
                y_sample_locs = np.arange(s_roll.shape[2])
                n_target_runs = 4
                n_target_runs = np.min([n_target_runs,np.count_nonzero(crit_steps==0),np.count_nonzero(crit_steps!=0)])
                null_accept_prob = n_target_runs/np.count_nonzero(crit_steps==0)
                trans_accept_prob = n_target_runs/np.count_nonzero(crit_steps!=0)
            
            for xshift in x_sample_locs:
                for yshift in y_sample_locs:
                    
                    if np.array(null).ndim == 0:
                        loc_crit_step = crit_step
                        loc_null = null
                    else:
                        coord_grid = np.meshgrid(np.arange(s.shape[1]),np.arange(s.shape[2]))
                        sample_loc = [np.roll(cg,(xshift,yshift),axis=(0,1))[int((scg*target_size)/2),int((scg*target_size)/2)] for cg in coord_grid]
                        loc_crit_step = [crit_steps[sample_loc[0],sample_loc[1]]]
                        if loc_crit_step[0] == 0:
                            if np.random.rand() > null_accept_prob:
                                continue
                            loc_null = 1
                        else:
                            if np.random.rand() > trans_accept_prob:
                                continue
                            loc_null = 0
                        
                    
                    print('Shift {}'.format((xshift,yshift)))
                    s_shift = np.roll(s_roll,(1,xshift,yshift),axis=(0,1,2)) + noise_sigma*np.random.randn(*s.shape)
                    s_shift = s_shift[:,:(scg*target_size),:(scg*target_size)]
                    s_shift = block_reduce(s_shift,block_size=(1,scg,scg),func=np.mean)
                    # plt.imshow(s_shift[crit_step,:,:],vmin=-1,vmax=1)
                    # plt.title('Transform {}, SCG {}, Shift {}'.format((grid_rot,grid_flip),scg,(xshift,yshift)))
                    # plt.colorbar()
                    # plt.show()
                    
                    # for time_dir in [-1,1]:
                    for time_dir in [1]:
                        if time_dir == -1:
                            if len(loc_crit_step) == 1:
                                this_crit_step = s.shape[0]-loc_crit_step[0]
                            else:
                                this_crit_step = s.shape[0]-loc_crit_step[1] # use crit_step_r if applicable
                            s_shift_crop = np.flip(s_shift,axis=0)
                        else:
                            this_crit_step = loc_crit_step[0]
                            s_shift_crop = s_shift
                            
                        # crop time domain
                        if loc_null == 0:
                            s_shift_crop = s_shift_crop[:this_crit_step,:,:]
                        if s_shift_crop.shape[0] > target_duration:
                            s_shift_crop = s_shift_crop[-target_duration:,:,:]
                        
                        
                        this_crop_fraction = np.random.uniform(low=crop_fraction_range[0],high=crop_fraction_range[1])
                        this_crop = int(this_crop_fraction*s_shift_crop.shape[0])
                        
                        s_shift_crop = s_shift_crop[this_crop:,:,:]
                        
                        if np.std(s_shift_crop) < 1e-6:
                            print('No variance present in data')
                            continue
                        else:
                            s_shift_crop = s_shift_crop/np.nanstd(s_shift_crop)
                    
                        
                        # Ising model should be +/- symmetric (under a flip of global external field, if h!=0)
                        run_flip = np.random.choice([-1,1])
                        s_shift_crop = s_shift_crop * run_flip
                        
                        this_t_roll_window_frac = np.random.uniform(low=t_roll_window_range[0],high=t_roll_window_range[1])
                        if loc_null:
                            # series length is half as long for null, so fractional window length
                            # should be twice as high to get the same absolute window length
                            this_t_roll_window_frac = 2*this_t_roll_window_frac
                        
                        this_x = compute_ews(s_shift_crop,this_t_roll_window_frac,smoothing=smoothing,smooth_param=smooth_param)
                        this_x_pad = np.zeros((target_duration,this_x.shape[1]))
                        this_x_pad[-this_x.shape[0]:,:] = this_x
                    
                        row_dict = {'run_id':run_id,
                                    'null':loc_null,
                                    'x':this_x_pad,
                                    'time_dir':time_dir,
                                    'spatial_cg':scg,
                                    'xshift':xshift,
                                    'yshift':yshift,
                                    't_roll_window':this_t_roll_window_frac,
                                    'run_flip':run_flip}
                        
                        if np.array(null).ndim == 0:
                            row_dict['Tbounds'] = Tbounds
                        else:
                            row_dict['bounds_null'] = bounds_null
                            row_dict['bounds_trans'] = bounds_trans
                            row_dict['spike_loc'] = spike_loc
                            row_dict['spike_width'] = spike_width
                            row_dict['crit_step'] = loc_crit_step
                            row_dict['sample_loc'] = sample_loc
                        
                        file_df_ews = file_df_ews.append(row_dict,ignore_index=True)
        if output_ews:
            file_df_ews.to_pickle(os.path.join(out_dir_ews,'Processed_'+file_name+'.pkl'))
            
