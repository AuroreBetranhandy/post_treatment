#!/usr/bin/env python
# coding: utf-8

# In[154]:


import yt 
from yt import derived_field
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from os.path import exists
from scipy import signal, linalg
from matplotlib.colors import LogNorm, SymLogNorm
from yt.funcs import mylog
mylog.setLevel(50)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
import h5py

import astropy.constants as cn
import abel_modified
import glob

from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.font_manager as font_manager
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import os,gc
from multiprocessing import Pool
import subprocess

from scipy import interpolate

G=cn.G.cgs.value

from scipy.signal import savgol_filter
import warnings

warnings.simplefilter('ignore')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
font = font_manager.FontProperties(family='monospace', #Typewriter font
                                   weight='light',
                                   style='normal', size=16)


@derived_field(name='mycostheta',sampling_type='cell')
def _mycostheta(field,data):
    return data['z']/np.sqrt(data['r']**2+data['z']**2)

@derived_field(name='mysintheta',sampling_type='cell')
def _mysintheta(field,data):
    return data['r']/np.sqrt(data['r']**2+data['z']**2)
    
    
@derived_field(name='radius',sampling_type='cell')
def _radius(field,data):
    return np.sqrt(data['r']**2+data['z']**2)

@derived_field(name='vrad',units='cm/s',sampling_type='cell')
def _vrad(field,data):
    return data['vely']*data['mycostheta']+data['velx']*data['mysintheta']

@derived_field(name='vmag',units='cm/s',sampling_type='cell')
def _vmag(field,data):
    return np.sqrt(data['vely']**2+data['velx']**2)

@derived_field(name='vtheta',units='cm/s',sampling_type='cell')
def _vtheta(field,data):
    return data['velx']*data['mycostheta']-data['vely']*data['mysintheta']

@derived_field(name='rho_p_eint',units='g/cm**3',sampling_type='cell')
def _rho_p_eint(field,data):
    return data['density']+data['density']*(data['eint'].v/29979245800.0**2 - 2.86969e+19/29979245800.**2)






def open_data(filename):
        # ~ ds_2d = yt.load(filename)
    # ~ data = ds_2d.slice(2,0)
    # ~ npoints=3000
    # ~ rmax=1e9/16/8/2**9*npoints/1e5
    #time=float(np.nanmean(data[('gas', 'dynamical_time')]))
    test=h5py.File(filename)
    time=test['real scalars'][2][-1]
    r_shock=test['real scalars'][2][-1]
    nblockx=test['integer runtime parameters'][74][-1]
    refine=test['integer runtime parameters'][62][-1]
    nblocky=nblockx*2
    ds_2d = yt.load(filename)
    data = ds_2d.slice(2,0)
    npoints=3000
    # ~ print(nblockx,nblocky)
    rmax=1e9/nblocky/nblockx/2**refine*npoints/1e5
    #nsize=int((ds_2d.all_data().icoords.shape)[0]**(1./3.))*3
    nsize=npoints
    data = data.to_frb((rmax,"km"),nsize,center=(0,0),periodic=True)
    x = data[('index','r')].d[:,int(nsize/2)+1:]/1e4
    y = data[('index','z')].d[:,int(nsize/2)+1:]/1e4
    
    radius = data[('radius')].d[:-1,int(nsize/2)+1:-1] 
    press = data['pres'].d[:-1,int(nsize/2)+1:-1]
    mass = data['cell_mass'].d[:-1,int(nsize/2)+1:-1]
    rho = data['dens'].d[:-1,int(nsize/2)+1:-1]
    gamc = data['gamc'].d[:-1,int(nsize/2)+1:-1]/29979245800
    vrad = data['vrad'].d[:-1,int(nsize/2)+1:-1]
    ye= data['ye  '].d[:-1,int(nsize/2)+1:-1] 
    gpot = data['gpot'].d[:-1,int(nsize/2)+1:-1]
    temp= data['temp'].d[:-1,int(nsize/2)+1:-1] 
    c_s = data['sound_speed'].d[:-1,int(nsize/2)+1:-1]
    shock = data['shok'].d[:-1,int(nsize/2)+1:-1]
    rnue = data['fnua'].d[:-1,int(nsize/2)+1:-1]
    theta=np.arccos(data['mycostheta'].d[:-1,int(nsize/2)+1:-1])
    del ds_2d, data
    test=h5py.File(filename)
    time=test['real scalars'][2][-1]
    r_shock=test['real scalars'][2][-1]
    return radius,theta,rho,x,y,press,mass,vrad,gamc,ye,temp,time,gpot,c_s,shock,rnue

def spherical(filename):
    

    test=h5py.File(filename)
    time=test['real scalars'][2][-1]
    r_shock=test['real scalars'][2][-1]
    nblockx=test['integer runtime parameters'][74][-1]
    refine=test['integer runtime parameters'][62][-1]
    time=test['real scalars'][2][-1]
    r_shock=test['real scalars'][2][-1]

    nblocky=nblockx*2
    ds_2d = yt.load(filename)
    data = ds_2d.slice(2,0)
    npoints=3000
    # ~ print(nblockx,nblocky)
    rmax=1e9/nblocky/nblockx/2**refine*npoints/1e5
    #nsize=int((ds_2d.all_data().icoords.shape)[0]**(1./3.))*3
    nsize=npoints
    data = data.to_frb((rmax,"km"),nsize,center=(0,0),periodic=True)
    r = data[('index','r')].d[:,int(nsize/2)+1:]/1e4
    theta = data[('index','theta')].d[:,int(nsize/2)+1:]/1e4
    
    press = data['pres'].d[:-1,int(nsize/2)+1:-1]
    mass = data['cell_mass'].d[:-1,int(nsize/2)+1:-1]
    rho = data['dens'].d[:-1,int(nsize/2)+1:-1]
    gamc = data['gamc'].d[:-1,int(nsize/2)+1:-1]/29979245800
    ye= data['ye  '].d[:-1,int(nsize/2)+1:-1] 
    gpot = data['gpot'].d[:-1,int(nsize/2)+1:-1]
    temp= data['temp'].d[:-1,int(nsize/2)+1:-1] 
    c_s = data['sound_speed'].d[:-1,int(nsize/2)+1:-1]
    shock = data['shok'].d[:-1,int(nsize/2)+1:-1]
    mu = data[('gas', 'mean_molecular_weight')].d[:-1,int(nsize/2)+1:-1]
    
    
    
    
    dye=np.gradient(ye,axis=0)
    drho=np.gradient(np.log(dens),axis=0)
    dtemp=np.gradient(np.log(temp),axis=0)
    dp=np.gradient(np.log(pressure),axis=0)
    dr=np.gradient(r*1e4,axis=0)
    dtheta=np.gradient(theta*r*1e4,axis=1)
    dgpot=np.gradient(grav_pot,axis=0)

    vol=2*np.pi*r*1e4*dr*dtheta
    mass=dens*vol
#     print(mass,vol,dens)
    #N2=dgpot/dr*(1/gamc*dp/dr - drho/dr )#+ drho**2/dtemp/dr)
    N2=dgpot/dr*(1/(sound_speed**2*dens/pressure)*dp/dr - drho/dr )
    kinetic= mass*vel_rad**2
    r=r/10
    
#     f = interpolate.interp1d(r[:,0], np.mean(N2,axis=1))
#     time=np.linspace(r[:,0].min(),r[:,0].max(),400)
#     mean_N2= f(time)
    mean_N2=np.nanmean(N2,axis=1)
    # ~ mean_N2=savgol_filter(np.mean(N2,axis=1),10,2)
    mask_minus=np.logical_and(mean_N2>0,r[:,0]<15)
    # ~ print(np.nonzero(r_shock))  
    r_shock[r_shock<0.8]=np.nan
    mean_shock=np.nanmean(r_shock,axis=1)
    shock_r=np.nanmean(r[:,0][mean_shock>0])
    shock_rm=np.full((len(theta[0,:])),shock_r)
    # ~ print(shock_rm.shape)
    if len(r[mask_minus])>0:
        min_val= r[mask_minus].max()
    else: 
        min_val=0
    
    mask_plus=np.logical_and(mean_N2>0,r[:,0]>min_val)

    if len(r[mask_plus])>0:
        max_val=r[mask_plus].min()
    else:
        max_val=0.
    mean_N2[mean_N2>0]=0.1
    
    
    
    
    r_new=np.linspace(max_val,shock_r,int((150-max_val)/10))
    f=interpolate.interp1d(r[:,0],mean_N2)
    test=abs(f(r_new))
    test[test>1]=50
    res = np.where(test[1:] != test[:-1])[0]
    # ~ print(res)
    for i in res:
        plt.axvline(r_new[i],0,1)
    # ~ print(r_new[abs(f(r_new))==1e4])
    kinetic_PNS=kinetic.copy()
    kinetic_PNS[N2>0]=np.nan
    kinetic_PNS[r>max_val]=np.nan
    kinetic_PNS[r<min_val]=np.nan
    
    kin_shock=kinetic.copy()
    # ~ kinetic_shock[N2>0]=np.nan
    kin_shock[r<r_new[res][0]]=np.nan
    kin_shock[r>shock_r]=np.nan
    tot_mass=mass.copy()
    tot_mass[r>max_val]=np.nan
    tot_mass[r<min_val]=np.nan
    tot_mass[N2>0]=np.nan

    mass_shock=mass.copy()
    # ~ kinetic_shock[N2>0]=np.nan
    mass_shock[r<r_new[res[0]]]=np.nan
    mass_shock[r>shock_r]=np.nan
    
    
    
    
    # ~ plt.semilogy(r[:,0],abs(mean_N2))

    # ~ plt.axhline(0,0,300,c='k')
    # ~ plt.ylim([-1e5,0])
    

    # ~ fig,ax=plt.subplots(2,1,figsize=(30,20),subplot_kw={'projection': 'polar'})
    
    # ~ cs=ax[0].pcolormesh(theta,r,kin_shock,norm=LogNorm(),cmap='Blues')
    # ~ plt.polar(theta[0,:],shock_rm,'w')#(theta,r,r_shock,cmap='Reds')
    # ~ im=ax[1].pcolormesh(theta[:,:],r[:,:],N2[:,:],norm=SymLogNorm(linthresh=np.nanmax(abs(N2))/1e11,\
                # ~ vmin=-np.nanmax(abs(N2)),vmax=np.nanmax(abs(N2))),cmap='seismic')
    # ~ ax[1].fill_between(theta[0,:],min_val,max_val,facecolor='y',alpha=0.25)
    # ~ ax[0].set_rmax(150)
    # ~ ax[1].set_rmax(150)
    
    # ~ ax[1].set_title('N2')
    # ~ fig.colorbar(im, ax=ax[1], shrink=0.6)
    # ~ fig.colorbar(cs, ax=ax[0], shrink=0.6)
    # ~ ax2=fig.add_subplot(1,2,1)
    

    # ~ mean_N2=savgol_filter(mean_N2,25,2)

# ~ #     print(np.nansum(tot_mass))




    # ~ ax2.semilogy(r[:,0][r[:,0]<30],abs(mean_N2[r[:,0]<30]),'r')
    # ~ ax2.semilogy(r_new,abs(f(r_new)),'b')
    
    # ~ print(r_new[abs(f(r_new)) == 10])
    # ~ ax2.axhline(0,0,300,c='k')
    # ~ ax2.set_ylim([-1e5,0])
    # ~ ax2.set_xlim([0,150])

    # ~ im = ax[1,1].pcolormesh(theta2[:-2,:].T,r2[:-2,:].T,N2[:-1,:],vmin=-1e-3,vmax=1e-3,cmap='seismic')
    

# ~ #    mean_N2=savgol_filter(np.mean(N2,axis=1),10,2)
# ~ #    mean_N2[mean_N2>-10]=np.nan

# ~ #    kinetic_mean=np.mean(kinetic,axis=1)
    # ~ plt.savefig('test_multi_conv'+filename[-4:])
    # ~ plt.show()
    # ~ plt.close(fig)
    for i in list(locals().keys()):
         exec('del ' + i)
    gc.collect()
    return np.nansum(kinetic_PNS),max_val,min_val,np.nansum(tot_mass),time,np.nansum(kin_shock), np.nansum(mass_shock),r_new[res[0]],shock_r


def polarize(filename):

        # ~ return
        
    # ~ if 'cylindrical' in str(test['string runtime parameters'][7][-1]) :
        # ~ print('test')
        # ~ return
    radius_carth,theta,rho,x,y,press,mass,vrad,gamc,ye,temp,time,gpot,c_s,shock,rnue=open_data(filename)
    


    
    
    r2,theta2=abel_modified.cart2polar(x, y)
    
    
    
    dens,r,theta=abel_modified.reproject_image_into_polar(rho,x,y)
    pressure,r,theta=abel_modified.reproject_image_into_polar(press,x,y)
    r_shock,r,theta=abel_modified.reproject_image_into_polar(shock,x,y)
    vel_rad,r,theta=abel_modified.reproject_image_into_polar(vrad,x,y)
    gamc,r,theta=abel_modified.reproject_image_into_polar(gamc,x,y)
    cell_mass,r,theta=abel_modified.reproject_image_into_polar(mass,x,y)
    ye,r,theta=abel_modified.reproject_image_into_polar(ye,x,y)
    temp,r,theta=abel_modified.reproject_image_into_polar(temp,x,y)
    grav_pot,r,theta=abel_modified.reproject_image_into_polar(gpot,x,y)
    sound_speed,r,theta=abel_modified.reproject_image_into_polar(c_s,x,y)
    r_nue,r,theta=abel_modified.reproject_image_into_polar(rnue,x,y)


    dye=np.gradient(ye,axis=0)
    drho=np.gradient(np.log(dens),axis=0)
    dtemp=np.gradient(np.log(temp),axis=0)
    dp=np.gradient(np.log(pressure),axis=0)
    dr=np.gradient(r*1e4,axis=0)
    dtheta=np.gradient(theta*r*1e4,axis=1)
    dgpot=np.gradient(grav_pot,axis=0)

    vol=2*np.pi*r*1e4*dr*dtheta
    mass=dens*vol
#     print(mass,vol,dens)
    #N2=dgpot/dr*(1/gamc*dp/dr - drho/dr )#+ drho**2/dtemp/dr)
    N2=dgpot/dr*(1/(sound_speed**2*dens/pressure)*dp/dr - drho/dr )/3e8
    kinetic= mass*vel_rad**2
    r=r/10
    
    
    diff_vel=np.mean(dens-np.mean(dens,axis=0))*(vel_rad-np.mean(vel_rad,axis=0))
    
#     f = interpolate.interp1d(r[:,0], np.mean(N2,axis=1))
#     time=np.linspace(r[:,0].min(),r[:,0].max(),400)
#     mean_N2= f(time)
    mean_N2=np.nanmean(N2,axis=1)
    mean_rnue=np.nanmean(r_nue,axis=1)
    
    
    plt.figure(1)
    plt.plot(r[:,0],(mean_N2))

    plt.axhline(0,0,300,c='k')
    # ~ plt.show()
       
    
    
    # ~ mean_N2=savgol_filter(np.mean(N2,axis=1),10,2)
    mask_minus=np.logical_and(mean_N2>0,np.logical_and(dens[:,0]>1e12,r[:,0]<15))
    
    
    r_shock[r_shock<0.8]=np.nan
    mean_shock=np.nanmean(r_shock,axis=1)
    shock_r=np.nanmean(r[:,0][mean_shock>0])
    shock_rm=np.full((len(theta[0,:])),shock_r)
    
    # ~ r_nue[r_nue<0.8]=np.nan
    # ~ mean_rnue=np.nanmean(r_nue,axis=1)
    # ~ nue_r=np.nanmean(r[:,0][mean_rnue>0])
    # ~ nue_r=np.full((len(theta[0,:])),nue_r)
    
    # ~ print(nue_r)
    
    if len(r[mask_minus])>0:
        min_val= r[mask_minus].min()
        
    else: 
        min_val=0
    
    mask_plus=np.logical_and(mean_N2>0,np.logical_and(r[:,0]>min_val,dens[:,0]>1e12))
    # ~ print(dens[:,0].min(),dens[:,0].max())
    if len(r[mask_plus])>0:
        max_val=r[mask_plus].max()
    else:
        max_val=0.
    # ~ mean_N2[mean_N2>0]=np.nan
    print(mean_rnue.shape,mean_N2.shape,r[:,0].shape)
    mask_gain=np.logical_and(mean_N2>0,np.logical_and(r[:,0]>max_val,mean_rnue>0))  
    
    # ~ print(min_val,max_val)
    # ~ gain_rad = r[:,0][np.logical_and(rnue>0)].min()
    shock_conv_max= np.mean(shock_rm) #r[:,0][mean_N2>0].max()
    shock_conv_min=r[:,0][mask_gain].min()
    # ~ print(shock_conv_max,shock_conv_max,shock_r)
    # ~ test[test>1]=50
    # ~ res = np.where(test[1:] != test[:-1])[0]

    # ~ for i in res:
        # ~ plt.axvline(r_new[i],0,1)
        
        
    kinetic_PNS=kinetic.copy()
    kinetic_PNS[N2>0]=np.nan
    kinetic_PNS[r>max_val]=np.nan
    kinetic_PNS[r<min_val]=np.nan
    
    
    
    kin_shock=kinetic.copy()
    kin_shock[N2>0]=np.nan
    kin_shock[r<shock_conv_min]=np.nan
    kin_shock[r>shock_conv_max]=np.nan
    
    
    tot_mass=mass.copy()
    tot_mass[r>max_val]=np.nan
    tot_mass[r<min_val]=np.nan
    tot_mass[N2>0]=np.nan

    mass_shock=mass.copy()
    mass_shock[N2>0]=np.nan
    mass_shock[r<shock_conv_min]=np.nan
    mass_shock[r>shock_conv_max]=np.nan
    
    
 
    # ~ plt.ylim([-1e5,0])
    # ~ plt.yscale('log')

    fig,ax=plt.subplots(3,1,figsize=(30,20),subplot_kw={'projection': 'polar'})
    
    # ~ cs=ax[0].pcolormesh(theta,r,abs(diff_vel),
                # ~ norm=SymLogNorm(linthresh=np.nanmax(abs(diff_vel))/1e2,vmin=-np.nanmax(abs(diff_vel)),vmax=np.nanmax(abs(diff_vel)))
                # ~ norm=LogNorm()
                # ~ ,cmap='seismic')
    cs=ax[0].pcolormesh(theta,r,(-r_nue),
                norm=SymLogNorm(linthresh=np.nanmax(abs(r_nue))/1e7,vmin=-np.nanmax(abs(r_nue)),vmax=np.nanmax(abs(r_nue)))
                # ~ norm=LogNorm()
                ,cmap='seismic')
    # ~ plt.polar(theta[0,:],shock_rm,'w')#(theta,r,r_shock,cmap='Reds')
    im=ax[1].pcolormesh(theta[:,:],r[:,:],-N2[:,:],norm=SymLogNorm(linthresh=np.nanmax(abs(N2))/1e11,\
                vmin=-np.nanmax(abs(N2)),vmax=np.nanmax(abs(N2))),cmap='seismic')
    # ~ plt.poelar(theta[0,:],shock_rm,'w')#(theta,r,r_shock,cmap='Reds')
    vs=ax[2].pcolormesh(theta,r,vel_rad,norm=SymLogNorm(linthresh=np.nanmax(vel_rad)*1e-2),cmap='seismic')
    # ~ plt.polar(theta[0,:],mean_rnue,'b')#(theta,r,r_shock,cmap='Reds')

    ax[1].fill_between(theta[0,:],shock_conv_min,shock_conv_max,facecolor='y',alpha=0.5)
    ax[1].fill_between(theta[0,:],min_val,max_val,facecolor='w',alpha=0.5)
    ax[0].set_rmax(70)
    ax[1].set_rmax(70)
    ax[2].set_rmax(70)
    
    ax[1].set_title('N2')
    fig.colorbar(im, ax=ax[1], shrink=0.6)
    fig.colorbar(cs, ax=ax[0], shrink=0.6)
    fig.colorbar(vs, ax=ax[2], shrink=0.6)
    ax2=fig.add_subplot(1,2,1)
    

    # ~ mean_N2=savgol_filter(mean_N2,25,2)

# ~ #     print(np.nansum(tot_mass))




    ax2.semilogy(r[:,0][r[:,0]<50],(mean_N2[r[:,0]<50]),'r')
    ax2.semilogy(r[:,0][r[:,0]>shock_conv_min],(mean_N2[r[:,0]>shock_conv_min]),'b')
    
    # ~ print(r_new[abs(f(r_new)) == 10])
    ax2.axhline(0,0,300,c='k')
    ax2.set_ylim([-1e5,0])
    ax2.set_xlim([0,150])

    # ~ im = ax[1,1].pcolormesh(theta2[:-2,:].T,r2[:-2,:].T,N2[:-1,:],vmin=-1e-3,vmax=1e-3,cmap='seismic')
    

# ~ #    mean_N2=savgol_filter(np.mean(N2,axis=1),10,2)
# ~ #    mean_N2[mean_N2>-10]=np.nan

# ~ #    kinetic_mean=np.mean(kinetic,axis=1)
    plt.savefig('test_multi_conv'+filename[-4:])
    plt.show()
    plt.close(fig)
    for i in list(locals().keys()):
         exec('del ' + i)
    gc.collect()
    return np.nansum(kinetic_PNS),max_val,min_val,np.nansum(tot_mass),time,np.nansum(kin_shock), np.nansum(mass_shock),shock_conv_min,shock_conv_max

def run_all(base):
    print(base)
    allprofiles[base] = {}
    kinetic_energ=np.zeros((nends[base]))
    radius_PNS_max=np.zeros((nends[base]))
    radius_PNS_min=np.zeros((nends[base]))
    total_mass=np.zeros((nends[base]))
    kinetic_shock=np.zeros((nends[base]))
    mass_shock=np.zeros((nends[base]))
    time=np.zeros((nends[base]))    
    r_in=np.zeros((nends[base]))    
    r_out=np.zeros((nends[base]))

    
    for i in range(nstarts[base],nends[base]) :
        
        # ~ filename = "output/"+base+'_hdf5_plt_cnt_'+str(i).zfill(4)
        filename = 'output/'+base+'_hdf5_plt_cnt_'+str(i).zfill(4)
        # ~ if not os.path.isfile('output/'+base+'_hdf5_plt_cnt_0000'):
            # ~ break
        if not os.path.isfile(filename):
            continue
        test=h5py.File(filename)
        # ~ print(str(test['string runtime parameters'][7][-1]))
        if 'spherical' in str(test['string runtime parameters'][7][-1]) :
            kinetic_energ[i],radius_PNS_max[i],radius_PNS_min[i],total_mass[i],time[i],kinetic_shock[i],mass_shock[i],r_in[i],r_out[i]=spherical(filename)
        else:
            kinetic_energ[i],radius_PNS_max[i],radius_PNS_min[i],total_mass[i],time[i],kinetic_shock[i],mass_shock[i],r_in[i],r_out[i]=polarize(filename)
        # ~ filename='s20_simp_SFHo_Hann8_hdf5_plt_cnt_0'+str(i)    
         #kinetic_energ[i],radius_PNS_max[i],radius_PNS_min[i],drho[i],dp[i],dr[i],dye[i]=polarize(filename)
    allprofiles[base]["kin_erg"]=kinetic_energ
    allprofiles[base]["conv_max_rad"]=radius_PNS_max
    allprofiles[base]["conv_min_rad"]=radius_PNS_min
    allprofiles[base]["total_mass"]=total_mass 
    allprofiles[base]["kinetic_shock"]=kinetic_shock 
    allprofiles[base]["mass_shock"]=mass_shock 
    allprofiles[base]["r_shock_in"]=r_in 
    allprofiles[base]["r_shock_out"]=r_out 
    allprofiles[base]['time']=time
    np.save("allprofiles"+base, allprofiles[base])





file_list=glob.glob('s20*SFHo*hr.dat')
# ~ file_list=["s20WH07_ref.dat"]
allprofiles={}
bases=[]
nstarts={}
nends={}

for i in file_list:
    # if not os.path.isfile("output/"+str(i[:-4])+"_hdf5_plt_cnt_0300"):
        # continue
    bases.append(i[:-4])
    nstarts[i[:-4]]=500
    if 'SRO' in i: 
        nends[i[:-4]]=1300
    else:
        nends[i[:-4]]=913
        
    run_all(i[:-4])
# ~ pool=Pool()
# ~ pool.map(run_all, bases) 
# ~ np.save("allprofiles",allprofiles)

