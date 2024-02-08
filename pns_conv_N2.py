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
from multiprocessing import Pool,Manager,Process
import subprocess

from scipy import interpolate
from functools import partial
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
    
    test=h5py.File(filename)
    time=test['real scalars'][2][-1]
    r_shock=test['real scalars'][2][-1]
    nblockx=test['integer runtime parameters'][74][-1]
    refine=test['integer runtime parameters'][62][-1]
    nblocky=nblockx*2
    ds_2d = yt.load(filename)
    data = ds_2d.slice(2,0)
    npoints=3000
    rmax=1e9/nblocky/nblockx/2**refine*npoints/1e5
    #time=float(np.nanmean(data[('gas', 'dynamical_time')]))

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

    return radius,theta,rho,x,y,press,mass,vrad,gamc,ye,temp,time,gpot,c_s,shock,rnue




def polarize(filename):
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
    N2=dgpot/dr*(1/(sound_speed**2*dens/pressure)*dp/dr - drho/dr )
    kinetic= mass*vel_rad**2
    r=r/10
    diff_vel=np.mean(dens-np.mean(dens,axis=0))*(vel_rad-np.mean(vel_rad,axis=0))
#     f = interpolate.interp1d(r[:,0], np.mean(N2,axis=1))
#     time=np.linspace(r[:,0].min(),r[:,0].max(),400)
#     mean_N2= f(time)
    mean_N2=np.nanmean(N2,axis=1)
    mean_rnue=np.nanmean(r_nue,axis=1)
    
    mask_minus=np.logical_and(mean_N2>0,r[:,0]<15)
    
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
    
    mask_gain=np.logical_and(mean_N2>0,np.logical_and(r[:,0]>max_val,mean_rnue>0))  
    

    shock_conv_max= np.mean(shock_rm)
    if len(r[:,0][mask_gain]):
        shock_conv_min=r[:,0][mask_gain].min()
    else: 
        shock_conv_min=max_val
        
    kinetic_PNS=kinetic.copy()
    # kinetic_PNS[N2>0]=np.nan
    kinetic_PNS[r>max_val]=np.nan
    kinetic_PNS[r<min_val]=np.nan
    
    
    
    kin_shock=kinetic.copy()
    # kin_shock[N2>0]=np.nan
    kin_shock[r<shock_conv_min]=np.nan
    kin_shock[r>shock_conv_max]=np.nan
    
    
    tot_mass=mass.copy()
    tot_mass[r>max_val]=np.nan
    tot_mass[r<min_val]=np.nan
    # tot_mass[N2>0]=np.nan

    mass_shock=mass.copy()
    # mass_shock[N2>0]=np.nan
    mass_shock[r<shock_conv_min]=np.nan
    mass_shock[r>shock_conv_max]=np.nan


    fig,ax=plt.subplots(3,1,figsize=(30,20),subplot_kw={'projection': 'polar'})
    
    # cs=ax[0].pcolormesh(theta,r,abs(diff_vel),
    #             norm=SymLogNorm(linthresh=np.nanmax(abs(diff_vel))/1e2,vmin=-np.nanmax(abs(diff_vel)),vmax=np.nanmax(abs(diff_vel)))
    #             # ~ norm=LogNorm()
    #             ,cmap='seismic')

    cs=ax[0].pcolormesh(theta,r,(-r_nue),
                norm=SymLogNorm(linthresh=np.nanmax(abs(r_nue))/1e7,vmin=-np.nanmax(abs(r_nue)),vmax=np.nanmax(abs(r_nue)))
                # ~ norm=LogNorm()
                ,cmap='seismic')
    im=ax[1].pcolormesh(theta[:,:],r[:,:],-N2[:,:],norm=SymLogNorm(linthresh=np.nanmax(abs(N2))/1e11,\
                vmin=-np.nanmax(abs(N2)),vmax=np.nanmax(abs(N2))),cmap='seismic')
    vs=ax[2].pcolormesh(theta,r,vel_rad,norm=SymLogNorm(linthresh=np.nanmax(abs(vel_rad))*1e-2),cmap='seismic')

    ax[1].fill_between(theta[0,:],shock_conv_min,shock_conv_max,facecolor='y',alpha=0.5)
    ax[1].fill_between(theta[0,:],min_val,max_val,facecolor='w',alpha=0.5)
    ax[0].plot(theta[0,:],shock_rm,'w')#(theta,r,r_shock,cmap='Reds')
    ax[1].plot(theta[0,:],shock_rm,'w')#(theta,r,r_shock,cmap='Reds')
    ax[2].plot(theta[0,:],shock_rm,'w')#(theta,r,r_shock,cmap='Reds')
    ax[0].set_rmax(150)
    ax[1].set_rmax(150)
    ax[2].set_rmax(150)
    
    ax[1].set_title('N2')
    fig.colorbar(im, ax=ax[1], shrink=0.6)
    fig.colorbar(cs, ax=ax[0], shrink=0.6)
    fig.colorbar(vs, ax=ax[2], shrink=0.6)
    ax2=fig.add_subplot(1,2,1)
    

    # ~ mean_N2=savgol_filter(mean_N2,25,2)

# ~ #     print(np.nansum(tot_mass))




    ax2.semilogy(r[:,0][r[:,0]<50],(mean_N2[r[:,0]<50]),'r')
    ax2.semilogy(r[:,0][r[:,0]>shock_conv_min],(mean_N2[r[:,0]>shock_conv_min]),'b')
    
    ax2.axhline(0,0,300,c='k')
    ax2.set_ylim([-1e5,0])
    ax2.set_xlim([0,150])

    plt.savefig('figures/test_multi_conv_'+filename[7:])
    plt.close(fig)
    for i in list(locals().keys()):
         exec('del ' + i)
    gc.collect()
    return np.nansum(kinetic_PNS),max_val,min_val,np.nansum(tot_mass),time,np.nansum(kin_shock), np.nansum(mass_shock),shock_conv_min,shock_conv_max


def blip_bloup(i):
    filename = "output/"+base+'_hdf5_plt_cnt_'+str(i).zfill(4)        # ~ filename = base+'_hdf5_plt_cnt_'+str(i).zfill(4)
    if not os.path.isfile(filename):
        return np.zeros(9)
    list_of_dict=polarize(filename)
    # print(list_of_dict)
    return list_of_dict
    
def run_all(base,allprofiles):
    print(base)
    #setattr(base, Manager().dict())
    allprofiles[base] = {}
    

    
    time_range=np.arange(nstarts[base],nends[base])
    # print(time_range)
    pool=Pool(70)
    procs=list()
    M= np.array(pool.map(blip_bloup,time_range))
    pool.close()

    # print(M,"run_all")
    sorted_M= M[M[:,4].argsort()]
    allprofiles[base]["kin_erg"]= sorted_M[:,0]
    allprofiles[base]["conv_max_rad"]=sorted_M[:,1]
    allprofiles[base]["conv_min_rad"]=sorted_M[:,2]
    allprofiles[base]["total_mass"]=sorted_M[:,3]
    allprofiles[base]['time']= sorted_M[:,4]
    allprofiles[base]["kinetic_shock"]=sorted_M[:,5]
    allprofiles[base]["mass_shock"]=sorted_M[:,6]
    allprofiles[base]["r_shock_in"]=sorted_M[:,7]
    allprofiles[base]["r_shock_out"]=sorted_M[:,8]
    print(allprofiles[base])
    np.save("allprofiles"+base, allprofiles[base])





file_list=glob.glob('s20*SFHo*.dat')
#file_list.remove('s20WH07_ref.dat')
#file_list=["s20*.dat"]
allprofiles={}
bases=[]
nstarts={}
nends={}

for i in file_list:
    # if not os.path.isfile("output/"+str(i[:-4])+"_hdf5_plt_cnt_0300"):
        # continue
    
    if 'SFHo' in i: 
        bases.append(i[:-5])
        nends[i[:-5]]=2000
        nstarts[i[:-5]]=300
    else:
        bases.append(i[:-4])
        nends[i[:-4]]=2000
        nstarts[i[:-4]]=300
# ~ run_all("s20_simp_SFHo_Hann8")

for base in bases:
    run_all(base,allprofiles)
# pool=Pool()
# pool.map(run_all, bases) 
# np.save("allprofiles",allprofiles)

