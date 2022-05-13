# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:38:12 2022

@author: Daniel Dylewsky

Percolation model with critical transition
references:
https://www.science.org/doi/pdf/10.1126/science.282.5397.2238
https://www.cambridge.org/core/services/aop-cambridge-core/content/view/AB1A46AE72CA3259EF62AF87CACD99EF/S0260305500263866a.pdf/brine-percolation-and-the-transport-properties-of-sea-ice.pdf

"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
import time

def generate_medium(L,p0):
    """
    

    Parameters
    ----------
    L : Vector of lattice size in 3D
    p0 : Initial probability of each bond being open

    Returns
    -------
    
    x : 4D array of lattice bonds (L1 x L2 x L3 x 6)
        1st three coords index lattice side, last indexes which of the 6 bonds coming off it
        x has symmetry: x(i,j,k,0) = x(i+1,j,k,1)
                        x(i,j,k,2) = x(i,j+1,k,3)
                        x(i,j,k,4) = x(i,j,k+1,5)
        x has BCs: x(0,j,k,0) = 0
                   x(-1,j,k,1) = 0
                   x(i,0,k,2) = 0
                   x(i,-1,k,3) = 0
                   x(i,j,0,4) = 0
                   x(i,j,-1,5) = 0
                   
    y : 3D array of lattice sites (L1 x L2 x L3)

    """
    
    y = np.zeros(tuple(L),dtype=np.uintc)
    x = np.random.choice([0,1],p=[1-p0,p0],size=(L[0],L[1],L[2],6))
    
    x[1:,:,:,1] = x[:-1,:,:,0]
    x[:,1:,:,3] = x[:,:-1,:,2]
    x[:,:,1:,5] = x[:,:,:-1,4]
    
    x[0,:,:,0] = 0
    x[-1,:,:,1] = 0
    x[:,0,:,2] = 0
    x[:,-1,:,3] = 0
    x[:,:,0,4] = 0
    x[:,:,-1,5] = 0
    
    return x.astype(np.uintc), y

def cluster_sizes(x):
    """
    

    Parameters
    ----------
    x : array of lattice bonds

    Returns
    -------
    area : vector of length (# of clusters) containing volume (in # of sites) of each cluster

    """
    
    lw, num = measurements.label(x)
    labelList = np.arange(lw.max() + 1)
    area = measurements.sum(x, lw, labelList)
    return area

def diffuse_step(x,y,pD,nD,pF,sid,source=True):
    """
    

    Parameters
    ----------
    x : array of lattice bonds (closed or open)
    y : array of lattice sites (empty or full)
    pD : probability of diffusion to a given available site
    nD : number of diffusion iterations per call of this function
    pF : coefficient for probability of refreezing (function of age)
    sid : identification number of this time step
    source : if True, new fluid is supplied along y[:,:,-1] interface

    Returns
    -------
    y : updated array of lattice bond occupations
    n_avail : number of unoccupied sites available for diffusion

    """
    
    for dj in range(nD):
        # y_aug = np.concatenate([y,np.ones((y.shape[0],y.shape[1],1))],axis=2) # populate one outer surface
        if source:
            y_aug = np.concatenate([y,sid*np.ones((y.shape[0],y.shape[1],1))],axis=2) # populate one outer surface
        else:
            y_aug = np.concatenate([y,sid*np.zeros((y.shape[0],y.shape[1],1))],axis=2) # populate one outer surface
        
        
        y0_aug = np.zeros_like(y_aug)
        y0_aug[:-1,:,:] = y_aug[1:,:,:] # all possible shifts of -1 along axis 0
        y0 = np.multiply(y0_aug[:,:,:-1],x[:,:,:,0])
        
        y1_aug = np.zeros_like(y_aug)
        y1_aug[1:,:,:] = y_aug[:-1,:,:] # all possible shifts of +1 along axis 0
        y1 = np.multiply(y1_aug[:,:,:-1],x[:,:,:,1])
        
        y2_aug = np.zeros_like(y_aug)
        y2_aug[:,:-1,:] = y_aug[:,1:,:] # all possible shifts of -1 along axis 1
        y2 = np.multiply(y2_aug[:,:,:-1],x[:,:,:,2])
        
        y3_aug = np.zeros_like(y_aug)
        y3_aug[:,1:,:] = y_aug[:,:-1,:] # all possible shifts of +1 along axis 1
        y3 = np.multiply(y3_aug[:,:,:-1],x[:,:,:,3])
        
        y4_aug = np.zeros_like(y_aug)
        y4_aug[:,:,:-1] = y_aug[:,:,1:] # all possible shifts of -1 along axis 2
        y4 = np.multiply(y4_aug[:,:,:-1],x[:,:,:,4])
        
        y5_aug = np.zeros_like(y_aug)
        y5_aug[:,:,1:] = y_aug[:,:,:-1] # all possible shifts of +1 along axis 2
        y5 = np.multiply(y5_aug[:,:,:-1],x[:,:,:,5])
        
        y_avail = np.stack([y0,y1,y2,y3,y4,y5])
        y_avail = np.max(y_avail,axis=0) # just keep most recent percolation to each site
        
        # y_avail = y0 + y1 + y2 + y3 + y4 + y5
        # y_avail[y_avail > 0] = 1 # site availability is binary
        
        # y_avail[y==1] = 0 # omit already-occupied sites
    
    
    
        n_avail = np.count_nonzero(np.multiply(y_avail>0,y==0)) # number of currently-unoccupied sites available
        y_probs = np.zeros(y_avail.shape)
        
        if pD < 1:
            y_probs[y_avail>0] = np.random.uniform(size=np.count_nonzero(y_avail>0))
        else:
            y_probs[y_avail>0] = np.ones(np.count_nonzero(y_avail>0))
        
        n_add = np.count_nonzero(np.multiply(y_probs > 1-pD,y==0))
        if n_add < 0:
            import pdb; pdb.set_trace()
        
        y[y_probs > 1-pD] = y_avail[y_probs > 1-pD]
    
    # y_fill = (y_probs > 1-pD) * y_avail # fill available sites with probability pD
    
    # y_new = y + y_fill
    
    # Refreezing:
    if pF > 0:
        age_limit = 40 # number of years after which refreezing probability hits the tanh inflection point
        tanh_sigma = 10
        y_age = np.zeros(y.shape,dtype=np.float)
        y_age[y > 0] = sid-y[y > 0]+1
        
        # y_age_rel = np.zeros_like(y_age)
        # y_age_rel[y > 0] = y_age[y > 0] - np.tile(np.flip(np.arange(y.shape[2]))[np.newaxis,np.newaxis,:],(y.shape[0],y.shape[1],1))[y > 0]
        # if np.any(y_age_rel < 0) or np.any(y_age_rel > sid):
        #     print('bad age detected')
        #     import pdb; pdb.set_trace()
        
        # yF_probs = pF*(np.tanh((y_age_rel-age_limit)/tanh_sigma)+1)/2
        # yF_probs[y_age_rel == 0] = 0
        
        yF_probs = pF*(np.tanh((y_age-age_limit)/tanh_sigma)+1)/2
        yF_probs[y_age == 0] = 0
        
        yF_rng = np.zeros(y.shape)
        yF_rng[y > 0] = np.random.uniform(size=np.count_nonzero(y>0))
        yF = np.zeros(y.shape,dtype=np.uintc)
        yF[yF_rng < yF_probs] = 1
        yF = yF.astype(bool)
        if np.any(y[yF] == 0):
            print('bad value in y')
            import pdb; pdb.set_trace()
        y[yF] = 0
        
        n_sub = np.count_nonzero(yF)
    else:
        n_sub = 0
    
    n_flux = (n_add,n_sub)
    # print('Total: {}/{}'.format(np.count_nonzero(y),np.size(y)-np.count_nonzero(y)))
    
    return y, n_avail, n_flux
    

def update_bonds(x,p01,p10):
    """
    Update the values of x with the specified probabilities

    Parameters
    ----------
    x : array of lattice bonds (closed or open)
    p01 : probability of transitioning closed -> open
    p10 : probability of transitioning open -> closed

    Returns
    -------
    x_new : array of updated lattice bonds
    n_flux : tuple of (# bonds opened, # bonds closed)

    """
    
    t01 = np.zeros(x.shape)
    t01[x==0] = np.random.uniform(size=np.count_nonzero(x==0))
    t01[t01<1-p01] = 0
    t01[t01>0] = 1
    
    t10 = np.zeros(x.shape)
    t10[x==1] = np.random.uniform(size=np.count_nonzero(x==1))
    t10[t10<1-p10] = 0
    t10[t10>0] = 1
    
    n_add = np.count_nonzero(t01)
    n_sub = np.count_nonzero(t10)
    n_flux = (n_add,n_sub)
    
    x_new = x + t01 - t10
    if np.any(x<0) or np.any(x>1):
        print('bad x value')
        import pdb; pdb.set_trace()
    
    x_new[1:,:,:,1] = x_new[:-1,:,:,0]
    x_new[:,1:,:,3] = x_new[:,:-1,:,2]
    x_new[:,:,1:,5] = x_new[:,:,:-1,4]
    
    x_new [0,:,:,0] = 0
    x_new[-1,:,:,1] = 0
    x_new[:,0,:,2] = 0
    x_new[:,-1,:,3] = 0
    x_new[:,:,0,4] = 0
    x_new[:,:,-1,5] = 0
    
    
    return x_new, n_flux

    
# %% Generate run data

out_dir = 'Sea Ice Percolation Model'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

plot_dir = os.path.join(out_dir,'Plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

L = np.array([256,256,64])
# L = np.array([128,128,32])
tsteps = 1000
burn_steps = 100
nRuns = 40
p_crit = 0.23

for nr in range(nRuns):
    print('Executing run {}/{}'.format(nr,nRuns))
    this_null = np.random.choice([0,1])
    pt_min = p_crit - np.random.uniform(0.05,0.2)
    if this_null:
        pt_max = np.random.uniform(pt_min,p_crit-0.05)
    else:
        pt_max = p_crit + np.random.uniform(0.1,0.2)
    
    
    # p_target is target fraction of bonds which are open:
    p_target_array = np.linspace(pt_min,pt_max,tsteps)
    p_true_array = np.zeros(tsteps)
    
    p0 = p_target_array[0]
    
    # x is array of site bonds, y is array of sites
    
    x,y = generate_medium(L,p0)
    # cvols = cluster_sizes(x)
    
    # bond update probabilities:
    bond_update_rate = 0.01
    
    
    # Refreezing probability coefficient:
    # Note that this has its own critical behavior---some threshold value below which 
    # a cluster of occupied sites which are cut off from any source can nonetheless
    # persist indefinitely. In 3D this value appears to be somewhere between 0.8 and 0.9
    
    pF = 1
    pD = 1
    nD = 3
    
    y_avg = np.zeros((tsteps,y.shape[0],y.shape[1]))
    
    # Burn in
    for j in range(burn_steps):
        p10 = bond_update_rate*(1-p0)
        p01 = bond_update_rate*p0
        x,bflux = update_bonds(x,p01,p10)
        y, n_avail, n_flux = diffuse_step(x,y,pD,nD,pF,j+1)
        
    # Simulate
    for j,pt in enumerate(p_target_array):
        
        p10 = bond_update_rate*(1-pt)
        p01 = bond_update_rate*pt
        
        # if j % 10 == 0:
        # p01 = p_target*p10*np.count_nonzero(x)/(x.size-np.count_nonzero(x))
        x,bflux = update_bonds(x,p01,p10)
        pj = np.count_nonzero(x)/x.size
        p_true_array[j] = pj
        # print('Updating bonds: {}, p = {}'.format(bflux,pj))
        y, n_avail, n_flux = diffuse_step(x,y,pD,nD,pF,burn_steps+j+1)
        
        y_counts = y > 0
        y_avg[j,:,:] = np.mean(y_counts,axis=2)
            
        # print('Step {}: Flux = {}, p = {:.4f} (p_target = {:.4f})'.format(j,n_flux,pj,pt))
        
        # if n_avail == 0:
        #     print('Diffusion ceased at step {}'.format(j))
        #     break
    
    run_id = '{0:04d}'.format(int(np.round(1000*time.time())))
    
    out_dict = {'y_avg':y_avg,
                'p_target':p_target_array,
                'p_true':p_true_array,
                'pF':pF,
                'pD':pD,
                'nD':nD,
                'L':L,
                'run_id':run_id,
                'bond_update_rate':bond_update_rate,
                'p_crit':p_crit,
                'null':this_null}
    
    
    outfile = os.path.join(out_dir,'perc_run_'+run_id+'.npz')
    np.savez_compressed(outfile,**out_dict,allow_pickle=True, fix_imports=True)
    
    fig,axs = plt.subplots(2,1)
    
    t_crit = np.argmin(np.abs(p_true_array-p_crit))
    axs[0].plot(np.arange(tsteps),np.mean(y_avg,axis=(1,2)))
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y_avg')
    if not this_null:
        axs[0].axvline(t_crit,ls=':')
    
    
    
    axs[1].plot(p_true_array,np.mean(y_avg,axis=(1,2)))
    axs[1].set_xlabel('p')
    axs[1].set_ylabel('y_avg')
    if not this_null:
        axs[1].axvline(p_crit,ls=':')
    
    plt.savefig(os.path.join(plot_dir,'perc_run_'+run_id+'.png'))
    plt.close()
