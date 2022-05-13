# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:56:54 2021

@author: Daniel

Compute Moran's I spatial correlation function
"""

import numpy as np

def morans(s,r,periodic=True):
    # s is data matrix, r is correlation distance along primary axes (r=1 -> nearest neighbors)
    # s has 1 temporal and 2 or more spatial dimensions
    
    spatial_axes = tuple(np.arange(1,s.ndim))
    # l,n,m = s.shape
    M = np.zeros(s.shape[0])
    D = np.zeros(s.shape[0])
    sbar = np.nanmean(s,axis=spatial_axes)
    
    sbar_expand = sbar
    for sax in spatial_axes:
        sbar_expand = np.expand_dims(sbar_expand,sbar_expand.ndim)
            
    tile_counts = np.hstack((1,s.shape[1:]))
    sbar_tile = np.tile(sbar_expand,tile_counts)
    
    for ax in spatial_axes:
        for direc in [-1,1]:
            
            if periodic:
                s_shift = np.roll(s,r*direc,axis=ax)
                try:
                    M += np.nansum(np.multiply(s-sbar_tile,s_shift-sbar_tile),axis=spatial_axes)
                    D += np.nansum(np.multiply(s-sbar_tile,s-sbar_tile),axis=spatial_axes)
                except ValueError:
                    # old versions of numpy don't accept tuple axes
                    
                    M_add = np.multiply(s-sbar_tile,s_shift-sbar_tile)
                    D_add = np.multiply(s-sbar_tile,s-sbar_tile)
                    for sax in np.sort(spatial_axes)[::-1]:
                        # axes in decreasing order so axis indices are maintained over recursive summing 
                        M_add = np.nansum(M_add,axis=sax)
                        D_add = np.nansum(D_add,axis=sax)
                        
                    M += M_add
                    D += D_add
                
            else:
                if s.ndim == 3:
                    l,n,m = s.shape
                    if ax == 1:
                        y1 = np.arange(m)
                        y2 = np.arange(m)
                        if direc == -1:
                            x1 = np.arange(r,n)
                            x2 = np.arange(n-r)
                        elif direc == 1:
                            x1 = np.arange(n-r)
                            x2 = np.arange(r,n)
                    elif ax == 2:
                        x1 = np.arange(n)
                        x2 = np.arange(n)
                        if direc == -1:
                            y1 = np.arange(r,m)
                            y2 = np.arange(m-r)
                        elif direc == 1:
                            y1 = np.arange(m-r)
                            y2 = np.arange(r,m)
                    
                    s1 = s[:,x1,:].copy()
                    s1 = s1[:,:,y1]
                    s2 = s[:,x2,:].copy()
                    s2 = s2[:,:,y2]
                    this_sbar_tile = sbar_tile[:,:s1.shape[1],:s1.shape[2]] # truncate to match size of s1,s2
                    mq = np.multiply(s1-this_sbar_tile,s2-this_sbar_tile)
                    mq_mask = np.isnan(mq)
                    mq_mask_num = np.ones(s1.shape,dtype=float)
                    mq_mask_num[mq_mask] = np.nan
                    M += np.nansum(mq,axis=(1,2))
                    D += np.nansum(np.multiply(s1-this_sbar_tile,np.multiply(s1,mq_mask_num)-this_sbar_tile),axis=(1,2))
                else:
                    print('non-periodic BCs not implemented for spatial dimension >2')
    spatial_corr = np.divide(M,D)
    spatial_corr[np.isnan(spatial_corr)] = 0 # where D==0, default to 0
    
    return spatial_corr