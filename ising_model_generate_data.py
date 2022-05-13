# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:01:36 2021

@author: Daniel Dylewsky

Simulate Ising Model lattices with randomized coupling/temperature parameters
and levels of coarse-graining

Options are given to simulate a system with varied temperature ('temp') or varied
external magnetic field ('h')

Ising lattice can also be masked with randomized ellipses (i.e. random elliptical
regions of the lattice deleted in order to break symmetry and lend greater complexity
to the system)

Coupling coefficients J between neighboring lattice sites are also randomized 
(normally distributed about some mean value)

If process_data == True, process and write results to 'Processed' directory

"""

import numpy as np
import os
# import glob
from ising_model_simulate import ising_run
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from morans import morans
import time
from ising_model_process_data import process_ising_data


def generate_constants(order_param,L):
    # L = lattice side length
    try:
        rng = np.random.default_rng()
    except AttributeError:
        rng = np.random.RandomState()
        
    if order_param not in ['temp_local','h_local']:
        null = rng.choice([0,1])
        # null = 1
        # null = 0
    # J = 8*rng.normal()
    J_mean = 2**(5*rng.uniform())
    J_std = 0.3*rng.uniform()*J_mean # 0 - 30% of mean
    
    # epoch_len = int(np.round(target_size**(1 + 1*rng.uniform()))) # number of flips per epoch
    epoch_len = 10000
    
    Tc = 2*J_mean/(np.log(1+np.sqrt(2)))
    Hc = Tc # equal in mean field approx when k = mu = 1
    
    offset_dir =rng.choice([-1,1])
    # spatial_coarse_graining = rng.choice(np.arange(3,7))
    spatial_coarse_graining = 1
    temporal_coarse_graining = 1
    
    # bias = rng.uniform(-0.2,0.2) # positive bias = greater spin inertia; less likely to flip
    #                              # negative bias = lower spin inertia; more likely to flip
    
    bias = 0
    
    if order_param == 'temp':
        # temporal_coarse_graining = rng.choice(np.arange(3,7))
        if null:
            Tb1 = Tc * (0.1 + 0.1*rng.uniform())
            # Tb2 = Tc * (0.1 + 0.1*rng.uniform())
            
            Tbounds = Tc + offset_dir*np.array([Tb1,Tb1])
            Tbounds = rng.permutation(Tbounds)
        else:
            Tb1 = Tc * (0.5 - 0.4*rng.uniform())
            Tb2 = Tc * (1.5 + 0.4*rng.uniform())
            Tbounds = np.sort(np.array([Tb1,Tb2]))[::-1] # descending order
            
        # Tbounds = rng.permutation(Tbounds)
        hbounds = np.array([0,0])
    elif order_param == 'temp_lin':
        if null:
            Tb1 = Tc * (0.2 + 0.2*rng.uniform())
            Tb2 = Tc * (0.4 + 0.2*rng.uniform())
            
            Tbounds = Tc + offset_dir*np.array([Tb1,Tb2])
            Tbounds = rng.permutation(Tbounds)
        else:
            Tb1 = Tc * (0.5 - 0.4*rng.uniform())
            Tb2 = Tc * (1.5 + 0.4*rng.uniform())
            Tbounds = np.sort(np.array([Tb1,Tb2]))[::-1] # descending order
            Tbounds = rng.permutation(Tbounds)
            
        # Tbounds = rng.permutation(Tbounds)
        hbounds = np.array([0,0])
    elif order_param == 'h':
        # temporal_coarse_graining = rng.choice(np.arange(3,7))
        if null:
            hb1 = Hc * (0.1 + 0.1*rng.uniform())
            # hb2 = Hc * (0.7 + 0.1*rng.uniform())
            
            hbounds = offset_dir*(Hc + np.array([hb1,hb1]))
            hbounds = rng.permutation(hbounds)
        else:
            hb1 = -Hc*(1 + 0.6*rng.uniform())
            hb2 = Hc*(1 + 0.6*rng.uniform())
            hbounds = rng.permutation(np.array([hb1,hb2]))
            
        # Tbounds = rng.permutation(Tbounds)
        Tbounds = Tc * (1 + 0.3*np.random.randn()) * np.ones(2)
        while np.any(Tbounds <= 0):
            Tbounds = Tc * (1 + 0.3*np.random.randn()) * np.ones(2)
    elif order_param == 'h_lin':
        # temporal_coarse_graining = rng.choice(np.arange(3,7))
        if null:
            hb1 = Hc * (0.1 + 0.1*rng.uniform())
            hb2 = Hc * (0.2 + 0.2*rng.uniform())
            # hb2 = Hc * (0.7 + 0.1*rng.uniform())
            
            hbounds = offset_dir*(Hc + np.array([hb1,hb2]))
            hbounds = rng.permutation(hbounds)
        else:
            hb1 = -Hc*(1 + 0.6*rng.uniform())
            hb2 = Hc*(1 + 0.6*rng.uniform())
            hbounds = rng.permutation(np.array([hb1,hb2]))
            
        # Tbounds = rng.permutation(Tbounds)
        Tbounds = -np.ones(2)
        while np.any(Tbounds <= 0):
            Tbounds = Tc * (0.9 - 0.8*rng.uniform()) * np.ones(2)
            
    elif order_param == 'temp_local':
        Tb_null_1 = Tc * (0.2 + 0.2*rng.uniform())
        Tb_null_2 = Tc * (0.4 + 0.2*rng.uniform())
        
        bounds_null = Tc + offset_dir*np.array([Tb_null_1,Tb_null_2])
        bounds_null = rng.permutation(bounds_null)
        
        Tb_trans_1 = bounds_null[0]
        Tb_trans_2 = Tc + -offset_dir * Tc * (0.1 + 0.4*rng.uniform())
        bounds_trans = np.array([Tb_trans_1,Tb_trans_2])
        
        
        try:
            spike_loc = np.array([rng.integers(L),rng.integers(L)])
        except AttributeError:
            # rng.integers function is missing from old versions of numpy
            spike_loc = np.array([np.random.randint(L),np.random.randint(L)])
        spike_width = np.random.uniform(0.2,0.3)*L
        
    

    run_params = {'J_mean':J_mean, 'J_std':J_std, 'Tc':Tc, 'Hc':Hc, 
                  'spatial_coarse_graining':spatial_coarse_graining,
                  'temporal_coarse_graining':temporal_coarse_graining,
                  'epoch_len':epoch_len, 'bias':bias}
    if order_param in ['temp_local','h_local']:
        run_params['bounds_null'] = bounds_null
        run_params['bounds_trans'] = bounds_trans
        run_params['spike_loc'] = spike_loc
        run_params['spike_width'] = spike_width
    else:
        run_params['null'] = null
        run_params['Tbounds'] = Tbounds
        run_params['hbounds'] = hbounds
    return run_params

def generate_mask(size,mask_type=None,count=2):
    mask = np.ones((size,size),dtype=bool)
    if mask_type == None:
        return mask
    elif mask_type =='ellipse':
        for jj in range(count):
            cx, cy = (np.random.randint(size),np.random.randint(size))
            r1 = int(np.random.uniform(low=0.05,high=0.3)*size)
            r2 = int(np.random.uniform(low=0.05,high=0.3)*size)
            X,Y = np.meshgrid(np.arange(size),np.arange(size))
            
            
            # impose periodic BCs
            ex = X-cx
            ex[ex >= size/2] -= size
            ex[ex < -size/2] += size
            
            ey = Y-cy
            ey[ey >= size/2] -= size
            ey[ey < -size/2] += size
            
            ev = np.power(ex,2)/(r1**2) + np.power(ey,2)/(r2**2)
            emask = ev > 1
            mask = np.multiply(mask,emask)
        return mask
    
def bounds_to_vals(bounds_null,bounds_trans,spike_loc,spike_width):
    vals = np.tile(np.linspace(bounds_null[0],bounds_null[1],sim_duration)[:,np.newaxis,np.newaxis],(1,sim_size,sim_size))
            
    lattice_coords = np.meshgrid(np.arange(sim_size),np.arange(sim_size))
    lattice_center = int(sim_size/2)
    spike_kernel = np.exp(-(np.power(lattice_coords[0]-lattice_center,2)+
                            np.power(lattice_coords[1]-lattice_center,2))/spike_width**2)
    
    spike_kernel = np.roll(spike_kernel,shift=tuple(spike_loc-lattice_center),axis=(0,1))
    
    vals_local_peak = np.linspace(bounds_trans[0]-bounds_null[0],bounds_trans[1]-bounds_null[1],sim_duration)
    vals_local = np.einsum('i,jk->ijk',vals_local_peak,spike_kernel)
    
    vals = vals + vals_local
    return vals

# order_param = 'h'
# order_param = 'h_lin'
# order_param = 'temp'
# order_param = 'temp_lin'
order_param = 'temp_local'

mask_type = None
# mask_type = 'ellipse'

if mask_type is None:
    out_dir = os.path.join('Ising_Output','var_'+order_param)
else:
    out_dir = os.path.join('Ising_Output','var_'+order_param+'_'+mask_type)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
plot_stats = True

process_data = True
process_raw = True # output processed results before computing EWS
process_ews = True # output computed EWS for processed results
    
# target_duration = 1200
target_duration = 600
target_size = 256

# epoch_len = int(np.round(target_size*np.sqrt(target_size))) # number of flips per epoch
# epoch_len = target_size**2 # number of flips per epoch

n_runs = 1000
# n_runs = 1
burn_time = 50

print('Running ' + str(n_runs) + ' runs of order parameter ' + order_param)

spatial_corr_intervals = np.arange(1,4)

# file_list = np.array(glob.glob(os.path.join(out_dir,'*.npz')))
# if len(file_list) > 0:
#     existing_run_ids = [int(os.path.split(fname)[-1][:-4]) for fname in file_list]

# run_id_counter = np.max(existing_run_ids)
for nr in range(n_runs):
# for nr in [2]:
    these_params = generate_constants(order_param,target_size)
    
    J_mean = these_params['J_mean']
    J_std = these_params['J_std']
    Tc = these_params['Tc']
    Hc = these_params['Hc']
    epoch_len = these_params['epoch_len']
    spatial_coarse_graining = these_params['spatial_coarse_graining']
    temporal_coarse_graining = these_params['temporal_coarse_graining']
    bias = these_params['bias']
    
    sim_size = target_size*spatial_coarse_graining
    sim_duration = target_duration*temporal_coarse_graining
    sim_burn_time = burn_time*temporal_coarse_graining
    
    if order_param in ['temp_local','h_local']:
        bounds_null = these_params['bounds_null']
        bounds_trans = these_params['bounds_trans']
        spike_loc = these_params['spike_loc']
        spike_width = these_params['spike_width']
        
        if order_param == 'temp_local':
            
            temps = bounds_to_vals(bounds_null,bounds_trans,spike_loc,spike_width)
            fields = np.zeros(sim_duration)
            
            if np.mean(temps[0,:,:]) > Tc:
                temp_extremes = np.min(temps,axis=0)
                null = temp_extremes > Tc
            else:
                temp_extremes = np.max(temps,axis=0)
                null = temp_extremes < Tc
                
            crit_steps = np.argmin(np.abs(temps-Tc),axis=0)
            crit_steps[null] = 0
        
    else:
        null = these_params['null']
        Tbounds = these_params['Tbounds']
        hbounds = these_params['hbounds']
        if null:
            sim_duration = int(sim_duration/2)
    
        temps = np.linspace(Tbounds[0],Tbounds[1],sim_duration)
        fields = np.linspace(hbounds[0],hbounds[1],sim_duration)
    
    
    
    J = J_mean*np.ones((sim_size,sim_size)) + J_std*(np.random.randn(sim_size,sim_size))
    
    this_mask = generate_mask(sim_size,mask_type=mask_type)
    
    # run_id_counter += 1
    # run_id = '{0:04d}'.format(run_id_counter)
    run_id = '{0:04d}'.format(int(np.round(1000*time.time())))
    
    
    print('Executing run ' + str(nr) + ' (CG = ' + str((temporal_coarse_graining,spatial_coarse_graining)) + ', J_mean = {0:.2f}'.format(J_mean) + ', bias = {0:.2f}'.format(bias) + ', elen = ' + str(epoch_len) + ')')
    if order_param == 'temp' or order_param == 'temp_lin':
        if temps[0] > Tc:
            initial_state = 'r'
        else:
            initial_state = 'u'
    elif order_param == 'h' or order_param == 'h_lin':
        initial_state = 'r_h'
    elif order_param == 'temp_local':
        if np.mean(temps[0,:,:]) > Tc:
            initial_state = 'r'
        else:
            initial_state = 'u'
    else:
        print('unknown order_param')
        import pdb; pdb.set_trace()
    


    
    run_output = ising_run(temps, fields, sim_size, J, run_id, sim_burn_time, epoch_len, bias,
                           initial_state=initial_state,mask=this_mask)
    
    
    sys = run_output['sys']
    magnetization = run_output['magnetization']
    heat_capacity = run_output['heat_capacity']
    
    sys_burn = run_output['sys_burn']
    magnetization_burn = run_output['magnetization_burn']
    heat_capacity_burn = run_output['heat_capacity_burn']
    
    sys[sys==0] = np.nan
    sys_burn[sys_burn==0] = np.nan
    
    # apply coarse-graining
    sys_burn_cg = block_reduce(sys_burn,block_size=(temporal_coarse_graining,spatial_coarse_graining,spatial_coarse_graining),func=np.nanmean)
    sys_cg = block_reduce(sys,block_size=(temporal_coarse_graining,spatial_coarse_graining,spatial_coarse_graining),func=np.nanmean)
    if sys_cg.shape[1] != target_size or sys_cg.shape[0] != sim_duration:
        print('Coarse graining size error')
        print(str(sys.shape) + ' -> ' + str(sys_cg.shape))
        
    print('Computing spatial correlations')
    all_spatial_corr = []
    for ivl in spatial_corr_intervals:
        # all_spatial_corr.append(np.hstack((morans(sys_burn_cg,ivl),morans(sys_cg,ivl))))
        all_spatial_corr.append(morans(np.concatenate((sys_burn_cg,sys_cg),axis=0),ivl))
    
    if plot_stats:
        time_cg = np.arange((sim_duration/temporal_coarse_graining)+burn_time)
        time_full = np.arange(0,(sim_duration/temporal_coarse_graining)+burn_time,1/temporal_coarse_graining)
        
        plot_dir = os.path.join(out_dir,'Plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        
        fig, axs = plt.subplots(4,2,figsize=(12,9))
        axs[0,0].plot(time_full,np.hstack((magnetization_burn,magnetization)))
        axs[0,0].axvline(burn_time,ls=':',c='k')
        axs[0,0].set_ylabel('Magnetization')
        axs[0,0].set_ylim([-1.05,1.05])
        
        axs[1,0].plot(time_full,np.hstack((heat_capacity_burn,heat_capacity)))
        axs[1,0].set_ylabel('Heat Capacity')
        axs[1,0].axvline(burn_time,ls=':',c='k')
        
        
        if order_param == 'temp' or order_param == 'temp_lin':
            axs[2,0].plot(time_full,np.hstack((temps[0]*np.ones(sim_burn_time),temps)))
            axs[2,0].axhline(Tc,ls='--')
            axs[2,0].set_ylabel('Temperature')
        elif order_param == 'h' or order_param == 'h_lin':
            axs[2,0].plot(time_full,np.hstack((fields[0]*np.ones(sim_burn_time),fields)))
            axs[2,0].axhline(Hc,ls='--')
            axs[2,0].axhline(-Hc,ls='--')
            axs[2,0].axhline(0,ls=':')
            axs[2,0].set_ylabel('External Field')
        axs[2,0].axvline(burn_time,ls=':',c='k')
        
        for jiv, ivl in enumerate(spatial_corr_intervals):
            axs[3,0].plot(time_cg,all_spatial_corr[jiv],label='r = {}'.format(ivl))
        axs[3,0].axvline(burn_time,ls=':',c='k')
        axs[3,0].set_ylabel('Spatial Correlation')
        
        axs[0,1].imshow(np.squeeze(sys_burn_cg[0,:,:]),vmin=-1,vmax=1)
        axs[0,1].set_ylabel('Start of burn')
        
        axs[1,1].imshow(np.squeeze(sys_cg[0,:,:]),vmin=-1,vmax=1)
        axs[1,1].set_ylabel('End of burn')
        
        
        
        axs[2,1].imshow(np.squeeze(sys_cg[-1,:,:]),vmin=-1,vmax=1)
        axs[2,1].set_ylabel('Final')
        
        
        
        if (order_param == 'temp' or order_param == 'temp_lin'):
            if null == 0:
                crit_step = np.argmin(np.abs(temps-Tc))
                    
                crit_time = time_full[sim_burn_time+crit_step]
                for ax in axs[:,0]:
                    ax.axvline(time_full[sim_burn_time],ls=':',c='b')
                    ax.axvline(crit_time,ls=':',c='r')
            
        if order_param in ['temp_local','h_local']:
            fig.suptitle(' Var. ' + order_param + ' Run ' + run_id + '(CG = ' + str((temporal_coarse_graining,spatial_coarse_graining)) +
                     ', J_mean = {0:.2f}'.format(J_mean) + ', bias = {0:.2f}'.format(bias) + ')')
        else:
            fig.suptitle(' Var. ' + order_param + ' Run ' + run_id + '(CG = ' + str((temporal_coarse_graining,spatial_coarse_graining)) +
                     ', J_mean = {0:.2f}'.format(J_mean) + ', bias = {0:.2f}'.format(bias) + ', null = ' + str(null) + ')')
        plt.savefig(os.path.join(plot_dir,run_id+'.png'))
        plt.close()
        
    this_train_class = np.random.choice([0,1,2],p=[0.8,0.1,0.1]) # train, test, validate
    subdir = ['train','test','validate'][this_train_class]
        
    out_dict = {'null':null,
                'J':J,
                'Tc':Tc,
                'Hc':Hc,
                'order_param':order_param,
                'spatial_coarse_graining':spatial_coarse_graining,
                'temporal_coarse_graining':temporal_coarse_graining,
                's':sys_cg,
                'bias':bias,
                'magnetization':magnetization,
                'heat_capacity':heat_capacity,
                'run_id':run_id,
                'train_class':this_train_class}
    
    if order_param in ['temp_local','h_local']:
        out_dict['bounds_null'] = bounds_null
        out_dict['bounds_trans'] = bounds_trans
        out_dict['spike_loc'] = spike_loc
        out_dict['spike_width'] = spike_width
        out_dict['crit_steps'] = crit_steps
    else:
        out_dict['Tbounds'] = Tbounds
        out_dict['hbounds'] = hbounds

    
    if not os.path.exists(os.path.join(out_dir,subdir)):
        os.makedirs(os.path.join(out_dir,subdir))
        
    data_file = os.path.join(out_dir,subdir,run_id+'.npz')
    np.savez_compressed(data_file,**out_dict,
                        allow_pickle=True, fix_imports=True)
    
    
    if process_data:
        smooth_param=[96,0]
        # outfile_nosmooth = os.path.join(out_dir,'Processed','Processed_' + run_id + '.pkl')
        outdir_smooth = os.path.join(out_dir,'Processed','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
        outfile_smooth = os.path.join(outdir_smooth,subdir,'Processed_' + run_id + '.pkl')
        
        if not os.path.exists(outdir_smooth):
            os.makedirs(outdir_smooth)
        
        # process_ising_data(data_file,outfile_nosmooth,order_param,smoothing=None)
        process_ising_data(data_file,order_param,smoothing='gaussian',smooth_param=smooth_param,output_raw=process_raw,output_ews=process_ews)


