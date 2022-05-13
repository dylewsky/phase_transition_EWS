"""
Created on Tue Nov 23 12:47:57 2021

@author: Daniel Dylewsky

Implementation of water-vegetation model from:
Dakos et. al., "Slowing down in spatially patterned ecosystems at the brink of collapse" (2011)
Kefi et. al., "Early Warning Signals of Ecological Transitions: Methods for Spatial Patterns" (2014)


"""

import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

def sim_model(fn,x0,run_time,burn_time,consts,alpha_interval,dt=1,save_every=1,return_burn=False,pbc=True):
    # Burn in and simulate the system dx/dt = fn(x) starting from x0
    #
    # ALL DYNAMICAL VARIABLES ARE ASSUMED TO BE POSITIVE DEFINITE AND FORCIBLY BOUNDED AS SUCH
    # This is consistent with the systems being modeled in systems 1 and 3 from Kefi et. al.
    
    burn_steps = int(burn_time/dt)
    burn_steps_save = int((burn_time/dt)/save_every)
    run_steps = int(run_time/dt)
    run_steps_save = int((run_time/dt)/save_every)
    x_burn = np.zeros((burn_steps_save,len(x0)))
    x = np.zeros((run_steps_save,len(x0)))
    alpha_burn = alpha_interval[0]
    print('Running burn')
    this_x_burn = x0
    k = 0
    for j in range(burn_steps):
        this_x_burn = this_x_burn + dt*fn(this_x_burn,consts,alpha_burn,pbc=pbc)
        this_x_burn[this_x_burn<0] = 0
        if j % save_every == save_every-1:
            x_burn[k,:] = this_x_burn
            k += 1
    
    alpha = np.linspace(alpha_interval[0],alpha_interval[1],run_steps)
    print('Running sim')
    k = 0
    this_x = x_burn[-1,:]
    for j in range(run_steps):
        this_x = this_x + dt*fn(this_x,consts,alpha[j],pbc=pbc)
        this_x[this_x<0] = 0
        if j % save_every == save_every-1:
            x[k,:] = this_x
            k += 1
    
    if return_burn:
        return (x, x_burn)
    else:
        return x
    

    
# %% Model 1: Local positive feedback model with no patchy pattern

def dw(w,B,consts,alpha,pbc=True):
    if pbc:
        w_pad = w
        B_pad = B
    else:
        w_pad = np.zeros((w.shape[0]+2,w.shape[1]+2))
        w_pad[1:-1,1:-1] = w
        B_pad = np.zeros((B.shape[0]+2,B.shape[1]+2))
        B_pad[1:-1,1:-1] = B
        
    dw = alpha - w_pad - consts['lamb']*np.multiply(w_pad,B_pad) + \
        consts['D']*(np.roll(w_pad,(1,0),axis=(0,1))+np.roll(w_pad,(-1,0),axis=(0,1))+np.roll(w_pad,(0,1),axis=(0,1))+np.roll(w_pad,(0,-1),axis=(0,1))-4*w_pad) + \
        consts['sigma_w']*np.random.randn(*w_pad.shape)
        
    if not pbc:
        dw = dw[1:-1,1:-1]
    return dw

def dB(w,B,consts,pbc=True):
    if pbc:
        w_pad = w
        B_pad = B
    else:
        w_pad = np.zeros((w.shape[0]+2,w.shape[1]+2))
        w_pad[1:-1,1:-1] = w
        B_pad = np.zeros((B.shape[0]+2,B.shape[1]+2))
        B_pad[1:-1,1:-1] = B
        
    dB = consts['rho']*np.multiply(B_pad,(w_pad-(1/consts['B_c'])*B_pad)) - \
        consts['mu']*np.divide(B_pad,(B_pad+consts['B_0'])) + \
        consts['D']*(np.roll(B_pad,(1,0),axis=(0,1))+np.roll(B_pad,(-1,0),axis=(0,1))+np.roll(B_pad,(0,1),axis=(0,1))+np.roll(B_pad,(0,-1),axis=(0,1))-4*B_pad) + \
        consts['sigma_B']*np.random.randn(*B_pad.shape)
        
    if not pbc:
        dB = dB[1:-1,1:-1]
    return dB


def x_to_wB(x):
    if len(x.shape)==1:
        s2 = int(x.shape[0]/2)
        s = int(np.sqrt(s2))
        w = x[:s2].reshape((s,s))
        B = x[s2:].reshape((s,s))
    elif len(x.shape)==2:
        s2 = int(x.shape[1]/2)
        s = int(np.sqrt(s2))
        w = x[:,:s2]
        w = w.reshape(w.shape[0],s,s)
        B = x[:,s2:]
        B = B.reshape(B.shape[0],s,s)
    return(w,B)

def dx_m1(x,consts,alpha,pbc=True):
    w, B = x_to_wB(x)
    
    this_dw = dw(w,B,consts,alpha,pbc=pbc)
    this_dB = dB(w,B,consts,pbc=pbc)
    dx = np.hstack((this_dw.reshape(-1),this_dB.reshape(-1)))
    return dx



def generate_model1(consts,alpha_interval,rj,lattice_size=200,delta_t=0.1,save_every=10,burn_time=500,run_time=2000,plot_res=False,return_burn=False):
    
    
    w0 = (1/lattice_size)*np.random.uniform(size=(lattice_size,lattice_size))
    B0 = (1/lattice_size)*np.random.uniform(size=(lattice_size,lattice_size))
    x0 = np.hstack((w0.reshape(-1),B0.reshape(-1)))
    
    if return_burn:
        x, x_burn = sim_model(dx_m1,x0,run_time,burn_time,consts,alpha_interval,dt=delta_t,save_every = save_every,return_burn=return_burn,pbc=True)
        w_burn, B_burn = x_to_wB(x_burn)
    else:
        x = sim_model(dx_m1,x0,run_time,burn_time,consts,alpha_interval,dt=delta_t,save_every = save_every,return_burn=return_burn,pbc=True)
    w,B = x_to_wB(x)
    
    
    if plot_res:
        for vj,vn in enumerate(['w','B']):
            if vj == 0:
                this_qoi = w
            elif vj == 1:
                this_qoi = B
            fig, axs = plt.subplots(1,1,figsize=(8,6))
            axs.plot(np.mean(this_qoi,axis=(1,2)))
            axs.set_title(vn)
    
            if not os.path.exists(os.path.join('Plots',vn)):
                os.makedirs(os.path.join('Plots',vn))
                
            plt.savefig(os.path.join('Plots',vn,'Model_1_Output_{:03d}.png'.format(rj)))
            plt.close()
            
            if return_burn:
                if vj == 0:
                    this_qoi = w_burn
                elif vj == 1:
                    this_qoi = B_burn
                fig, axs = plt.subplots(1,1,figsize=(8,6))
                axs.plot(np.mean(this_qoi,axis=(1,2)))
                axs.set_title(vn+' Burn')
                if not os.path.exists(os.path.join('Plots',vn+'_burn')):
                    os.makedirs(os.path.join('Plots',vn+'_burn'))
                plt.savefig(os.path.join('Plots',vn+'_burn','Model_1_Output_{:03d}.png'.format(rj)))
                plt.close()
    
    if plot_res:
        fig, axs = plt.subplots(2,2,figsize=(8,6))
        axs[0,0].plot(np.mean(w_burn,axis=(1,2)))
        axs[0,0].set_title('w Burn')
        axs[1,0].plot(np.mean(B_burn,axis=(1,2)))
        axs[1,0].set_title('B Burn')
        axs[0,1].plot(np.mean(w,axis=(1,2)))
        axs[0,1].set_title('w')
        axs[1,1].plot(np.mean(B,axis=(1,2)))
        axs[1,1].set_title('B')
        # axs[0,1].plot(alpha_grid,np.mean(w,axis=(1,2)))
        # axs[0,1].set_title('w')
        # axs[1,1].plot(alpha_grid,np.mean(B,axis=(1,2)))
        # axs[1,1].set_title('B')
        plt.show()
        
        w_vlim = [np.min(w),np.max(w)]
        B_vlim = [np.min(B),np.max(B)]
        fig, axs = plt.subplots(2,2,figsize=(8,6))
        axs[0,0].imshow(w[0,:,:],vmin=w_vlim[0],vmax=w_vlim[1])
        axs[0,0].set_title('w[0]')
        axs[1,0].imshow(B[0,:,:],vmin=B_vlim[0],vmax=B_vlim[1])
        axs[1,0].set_title('B[0]')
        axs[0,1].imshow(w[-1,:,:],vmin=w_vlim[0],vmax=w_vlim[1])
        axs[0,1].set_title('w[-1]')
        axs[1,1].imshow(B[-1,:,:],vmin=B_vlim[0],vmax=B_vlim[1])
        axs[1,1].set_title('B[-1]')
        plt.show()
    
    out_dict = {'w':[w], 'B':[B], 'alpha_interval':[alpha_interval]}
    if return_burn:
        out_dict['w_burn'] = [w_burn]
        out_dict['B_burn'] = [B_burn]
        
    return out_dict


n_runs = 100
output_burn = True
plot_res = True

out_dir = 'Water Vegetation Model'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
m1_consts = {
    'D':0.05,
    'lamb':0.12,
    'rho':1,
    'B_c':1,
    'mu':2,
    'B_0':1,
    # 'sigma_w':0.1,
    # 'sigma_B':0.25
    'sigma_w':0.1*0.25,
    'sigma_B':0.25*0.25
    }


run_time = 1200

m1_cols = ['w','B']
m2_cols = ['w']
m3_cols = ['O','W','P']
all_cols = [m1_cols,m2_cols,m3_cols]

consts = m1_consts
gen_fn = generate_model1
alpha_crit = 1.936


existing_files = glob.glob(os.path.join(out_dir,'*.pkl'))
if len(existing_files) > 0:
    existing_file_inds = [int(os.path.split(fn)[-1][-7:-4]) for fn in existing_files]
    start_file_ind = int(np.max(existing_file_inds))+1
else:
    start_file_ind = 0
for rj in range(start_file_ind,start_file_ind+n_runs): 
    this_null = np.random.choice([0,1])
    if this_null == 1:
        this_run_time = int(run_time/2)
    else:
        this_run_time = run_time
    
    if this_null:
        this_offset = np.random.choice([-1,1])
        this_alpha_interval = alpha_crit + this_offset*alpha_crit*np.array([np.random.uniform(0.1,0.3),np.random.uniform(0.3,0.4)])
    else:
        this_alpha_interval = np.random.permutation(alpha_crit*(1+np.array([-np.random.uniform(0.2,0.5),np.random.uniform(0.2,0.5)])))
        
    
    this_row = gen_fn(consts,this_alpha_interval,rj,run_time=this_run_time,plot_res=plot_res,return_burn=True)

    
    out_df = pd.DataFrame(this_row)
    out_df['null'] = [this_null]
    out_df.to_pickle(os.path.join(out_dir,'wv_output_{:03d}.pkl'.format(rj)))
