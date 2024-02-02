import numpy as np 
from numpy.lib import recfunctions as rfn
from numpy import gradient

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import astropy.constants as cn
import astropy.units as u


from scipy.integrate import simps
from scipy.integrate import trapz


from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


from scipy.signal import stft
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema

from scipy.fftpack import fft, fftfreq
from scipy.ndimage import gaussian_filter1d


from multiprocessing import Pool
import multiprocessing as mp
import subprocess
import glob
from joblib import Parallel, delayed

### constants

c_light=cn.c.cgs.value
G=cn.G.cgs.value
pc=cn.pc.cgs.value


#### functions definition
def clean_data(df,sample_freq=50000):
    new_field=np.array([])
    df=np.sort(df,axis=0)
    delta_t = [df['time'][_+1] - df['time'][_] for _ in range(len(df) - 1)]
    t_bounce_ind = np.nonzero(df['max_shock_r'])[0][0]
    delta_t_mean = np.mean(delta_t[t_bounce_ind:])
    time=np.linspace(df['time'][t_bounce_ind:].min(),df['time'][t_bounce_ind:].max(),int(df['time'].max()*(sample_freq)))
    new_field=rfn.append_fields(new_field,'time',time)
    
    for i in range(len(df.dtype.names)-1):
        name=df.dtype.names[i+1]
        data=df[name][t_bounce_ind:]
        f= interpolate.interp1d(df['time'][t_bounce_ind:], data)
        new_field=rfn.append_fields(new_field,name,f(time))
    return new_field    


def strain_gw(df, delta_t):
    prefactor = 2*(cn.G/cn.c**4).cgs.value   #cgs
    ddIyy50 = gradient(df['dIyy1'], df['time'], edge_order=2)
    ddIyy50_150 = gradient(df['dIyy2'], df['time'], edge_order=2)
    ddIyy150 = gradient(df['dIyy3'], df['time'], edge_order=2)
    
    df=rfn.append_fields(df,'dIyy_tot', df['dIyy1']+ df['dIyy2'] + df['dIyy3'])
    second_dt_all = gradient(df['dIyy_tot'], df['time'], edge_order=2)
    
    xv= df['time']
    I1spl = IUS(df['time'], df['dIyy_tot'],k=3)
    I2spl = I1spl.derivative(n=1)
    I1spl_1 = IUS(df['time'], df['dIyy1'],k=3)
    I2spl_1 = I1spl_1.derivative(n=1)
    I1spl_2 = IUS(df['time'], df['dIyy2'],k=3)
    I2spl_2 = I1spl_2.derivative(n=1)    
    I1spl_3 = IUS(df['time'], df['dIyy3'],k=3)
    I2spl_3 = I1spl_3.derivative(n=1)
    
    I3spl = I1spl.derivative(n=2)
    
    I3spl_sq = I3spl(xv)**2
    int_I3_sq = trapz(I3spl_sq, x=xv)
    prefactor2 = 3/10 * (cn.G/cn.c**5).cgs.value
    Egw = prefactor2* int_I3_sq
    Egw_norm = Egw/(cn.M_sun*cn.c**2).cgs.value
    
    return I2spl,I2spl_1,I2spl_2,I2spl_3, second_dt_all, ddIyy50, ddIyy50_150, ddIyy150, Egw, Egw_norm



def spectrogram_frequence(I2_interp, time_list, delta_t_mean, sample_frequency_mean, hann_window=40e-3, overlap_fact=0.5, nfft_fact=1):
    prefactor = (3/2)*(cn.G/cn.c**4).cgs.value
    Nperseg = int(hann_window/delta_t_mean)
    Noverlap = int(Nperseg*overlap_fact)
    Nfft = Nperseg*nfft_fact
    
    
    f_list, tau_list, Zxx = stft(prefactor*I2_interp(time_list), sample_frequency_mean, nperseg=Nperseg, noverlap=Noverlap, window='hann', nfft=Nfft)
    spectrogr = np.abs(Zxx)**2
    return f_list, tau_list, spectrogr


def spectrogram_energy(I2_interp, time_list, delta_t_mean, sample_frequency_mean, hann_window=40e-3, overlap_fact=0.5, nfft_fact=1):
    prefactor = 3/5 *(cn.G/cn.c**5).cgs.value*(2*np.pi)**2
    Nperseg = int(hann_window/delta_t_mean)
    
    f_list, tau_list, Zxx = stft(I2_interp(time_list), sample_frequency_mean, nperseg=Nperseg,window='hann')
    Zabs = np.abs(Zxx)    
    pref_matrix = np.zeros((len(f_list),len(tau_list)))
    for i in range(len(f_list)):
        for j in range(len(tau_list)):
            pref_matrix[i][j] = prefactor*f_list[i]**2
    spectrogr = np.multiply(Zabs, pref_matrix)
    
    return f_list, tau_list, spectrogr


    
    
# Former main, using OOP instead for lisibility
class Data_frame:
    hann_window=25e-3
    overlap_factor=0.9
    nfft_factor=1
    
     
    def __init__(self,filename):
        self.name=filename
        print(filename)
        self.df = np.genfromtxt(filename, names=['time', 'mass', 'x-momentum', 'x-mom', 'y-mom', 'z-mom', 'E_kin', '8', \
            'E_grav', 'E_expl', 'E_bind_gain', 'mean_shock_r', 'min_shock_r', 'accretion_rate', 'max_shock_r', 'magnetic_energy', 'central_density',\
            'net_heating_rate', '19', 'avg_entropy_in_gain', 'M_NS', 'pns_rot', 'pns_x', 'pns_y', 'pns_z',\
            'com-x', 'com-y', 'com-z', 'radius_pns_avg', 'correct_radius_pns', 'E_kin_theta', 'E_kin_phi_gain', '33', 'M1_lum_tot_nue', 'M1_lum_tot_anue', \
            'M1_lum_tot_nux', 'M1_aveE_nue', '38', 'M1_aveE_nux', '40', '41', '42', 'R_nue', 'R_anue', 'R_nux',\
            'dIxx1', 'dIxy1', 'dIyy1', '49', '50', '51', 'dIxx2', 'dIxy2', 'dIyy2', '55', '56', '57', 'dIxx3', 'dIxy3', 'dIyy3', '61', '62', '63']) 
        self.data= clean_data(self.df)
        
        self.delta_t=self.data['time'][1:]-self.data['time'][:-1]
        self.t_bounce_ind = np.nonzero(self.data['max_shock_r'])[0][0]
        self.delta_t_mean = np.mean(self.delta_t[self.t_bounce_ind:])
        self.sample_frequency_mean = 1/self.delta_t_mean
        self.data=rfn.append_fields(self.data,'time_pb',self.data['time']-self.data['time'][self.t_bounce_ind])
        
        ## calculate strain
        self.I2spl,self.I2spl_1,self.I2spl_2,self.I2spl_3, self.I2, self.I2_50, self.I2_50_150, self.I2_150, self.Egw, self.Egw_norm = strain_gw(self.data,self.delta_t)
        
        ## Spectrograms
        
        self.freq_f, self.tau_f, self.spectrogram_f=spectrogram_frequence(self.I2spl,self.data['time'],self.delta_t_mean,self.sample_frequency_mean\
                                                                            , self.hann_window, self.overlap_factor, self.nfft_factor)        
        self.freq_f1, self.tau_f1, self.spectrogram_f1=spectrogram_frequence(self.I2spl_1,self.data['time'],self.delta_t_mean,self.sample_frequency_mean\
                                                                            , self.hann_window, self.overlap_factor, self.nfft_factor)        
        self.freq_f2, self.tau_f2, self.spectrogram_f2=spectrogram_frequence(self.I2spl_2,self.data['time'],self.delta_t_mean,self.sample_frequency_mean\
                                                                            , self.hann_window, self.overlap_factor, self.nfft_factor)        
        self.freq_f3, self.tau_f3, self.spectrogram_f3=spectrogram_frequence(self.I2spl_3,self.data['time'],self.delta_t_mean,self.sample_frequency_mean\
                                                                            , self.hann_window, self.overlap_factor, self.nfft_factor)        
        
        self.freq_energ, self.tau_energ, self.spectrogram_energ=spectrogram_energy(self.I2spl,self.data['time'],self.delta_t_mean,\
                                                                                    self.sample_frequency_mean, self.hann_window, self.overlap_factor)        
        
        
        
        
        if '8.dat' in filename :
            self.alpha=1.
        elif '7.dat' in filename :
            self.alpha=0.5
        else: 
            self.alpha=1.
        if 'ref' in filename: 
            self.ticks='-'
        else:
            self.ticks='--'
            
    
        if ('Gang' in filename) & ('SFHo' in filename) : 
            self.color='orange'
            self.tb=0.300
        elif 'nr' in filename :
            self.color='m'
        elif 'SFHo' in filename :
            self.color='r'
            self.tb=0.300
        elif ('Gang' in filename) & ('SRO' in filename) : 
            self.color='g'
            self.tb= 0.319
        elif 'SRO' in filename :
            self.color='b'
            self.tb= 0.319
          
        else:
            self.color='k'
            self.tb= 0.319
            print("case not found")
            
            
    
    
file_list=glob.glob('s20*SRO*.dat')
# ~ file_list.remove('s20_ref_Hann_SFHo_hr8.dat')
# ~ file_list.remove('s20_ref_Gang_SFHo_per8.dat')
# ~ file_list.remove('s20_ref_Gang_SFHo_hr8.dat')
# ~ file_list.remove('s20_simp_SFHo_Gang_nr8.dat')
def make_data(filename):
    test=Data_frame(filename)
    
    return test


all_sim = Parallel(n_jobs=len(file_list))(delayed(make_data)(file) for file in file_list)

fig_hchar,ax_hchar=plt.subplots(4,1,figsize=(10,20))

for name in range(len(file_list)):
    test=all_sim[name]
    fftx = fft(test.I2spl(test.data['time']))
    xfh = np.linspace(0.0, 1.0/(2.0*test.delta_t_mean), fftx.size//2)
    dEdf = 3/5 * (G/c_light**5) * (2*np.pi*xfh)**2 * np.abs(fftx[:fftx.size//2]*2/fftx.size)**2
    characteristic_strain = 1/(10*pc*1e3) * np.sqrt((2/np.pi**2) * (G/c_light**3) * dEdf) 
    val=savgol_filter(characteristic_strain,100,2)   
    ax_hchar[0].semilogy(xfh*1e-3,val,color=test.color,ls=test.ticks,alpha=test.alpha )
    ax_hchar[0].set_title("Total")
    ax_hchar[0].set_ylim([1e-24,1e-19])
            
    fftx = fft(test.I2spl_1(test.data['time']))
    xfh = np.linspace(0.0, 1.0/(2.0*test.delta_t_mean), fftx.size//2)
    dEdf = 3/5 * (G/c_light**5) * (2*np.pi*xfh)**2 * np.abs(fftx[:fftx.size//2]*2/fftx.size)**2
    characteristic_strain = 1/(10*pc*1e3) * np.sqrt((2/np.pi**2) * (G/c_light**3) * dEdf) 
    val=savgol_filter(characteristic_strain,100,2)   
    ax_hchar[1].semilogy(xfh*1e-3,val,color=test.color,ls=test.ticks,alpha=test.alpha )
    ax_hchar[1].set_title("x < 50 km")
    ax_hchar[1].set_ylim([1e-24,1e-19])

    fftx = fft(test.I2spl_2(test.data['time']))
    xfh = np.linspace(0.0, 1.0/(2.0*test.delta_t_mean), fftx.size//2)
    dEdf = 3/5 * (G/c_light**5) * (2*np.pi*xfh)**2 * np.abs(fftx[:fftx.size//2]*2/fftx.size)**2
    characteristic_strain = 1/(10*pc*1e3) * np.sqrt((2/np.pi**2) * (G/c_light**3) * dEdf) 
    val=savgol_filter(characteristic_strain,100,2)   
    ax_hchar[2].semilogy(xfh*1e-3,val,color=test.color,ls=test.ticks,alpha=test.alpha )
    ax_hchar[2].set_title("50 km < x < 150 km")
    ax_hchar[2].set_ylim([1e-24,1e-19])

    fftx = fft(test.I2spl_3(test.data['time']))
    xfh = np.linspace(0.0, 1.0/(2.0*test.delta_t_mean), fftx.size//2)
    dEdf = 3/5 * (G/c_light**5) * (2*np.pi*xfh)**2 * np.abs(fftx[:fftx.size//2]*2/fftx.size)**2
    characteristic_strain = 1/(10*pc*1e3) * np.sqrt((2/np.pi**2) * (G/c_light**3) * dEdf) 
    val=savgol_filter(characteristic_strain,100,2)   
    ax_hchar[3].semilogy(xfh*1e-3,val,color=test.color,ls=test.ticks,alpha=test.alpha )
    ax_hchar[3].set_title("x > 150 km")
    ax_hchar[3].set_ylim([1e-24,1e-19])

    
    black_line1, = ax_hchar[3].plot([], [], color='r', linestyle='-')
    black_line2, = ax_hchar[3].plot([], [], color='orange', linestyle='-')
    ax_hchar[3].legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title_fontsize=22,frameon=False,loc= "best")
    black_line1, = ax_hchar[2].plot([], [], color='k', linestyle='-')
    black_line2, = ax_hchar[2].plot([], [], color='k', linestyle='--')
    ax_hchar[2].legend([black_line1,black_line2],[r"Reference",r"$\kappa^*$"],fontsize=20,handlelength=1,frameon=False,loc= "best")
    
plt.savefig('H_char_tot_all.pdf')



fig,ax=plt.subplots(5,len(file_list), constrained_layout=True, figsize=(25,15))
max_freq=np.max([np.max(name.spectrogram_f) for name in all_sim]) 
print(max_freq)
for name in range(len(file_list)):
    test=all_sim[name]
    ax[0,name].set_title(test.name)
    pmesh = ax[0,name].pcolormesh(test.tau_f,\
                              test.freq_f,\
                              np.log10(test.spectrogram_f/max_freq)#np.log10(globals()['Z_h2'+i[:-4]])/max_freq
                   ,vmin = -5
                   ,vmax=0.
                   ,shading='gouraud'
                              ,rasterized=True
                              ,cmap=plt.get_cmap('viridis')
                  )
    ax[0,name].set_ylim(0,5000)
    ax[0,name].set_xlim(0,1.2)
fig.colorbar(pmesh, ax=ax[0,:], shrink=0.6)

# ~ max_freq=np.max([np.max(name.spectrogram_f1) for name in all_sim]) 
for name in range(len(file_list)):
    test=all_sim[name]
    pmesh = ax[1,name].pcolormesh(test.tau_f1,\
                              test.freq_f1,\
                              np.log10(test.spectrogram_f1/max_freq)#np.log10(globals()['Z_h2'+i[:-4]])/max_freq
                   ,vmin = -5
                   ,vmax=0.
                   ,shading='gouraud'
                              ,rasterized=True
                              ,cmap=plt.get_cmap('viridis')
                  )
    ax[1,name].set_ylim(0,5000)
    ax[1,name].set_xlim(0,1.2)
fig.colorbar(pmesh, ax=ax[1,:], shrink=0.6)

for name in range(len(file_list)):
    test=all_sim[name]
    # ~ ax[2,name].set_title(test.name)
    pmesh = ax[2,name].pcolormesh(test.tau_f2,\
                              test.freq_f2,\
                              np.log10(test.spectrogram_f2/max_freq)#np.log10(globals()['Z_h2'+i[:-4]])/max_freq
                   ,vmin = -5
                   ,vmax=0.
                   ,shading='gouraud'
                              ,rasterized=True
                              ,cmap=plt.get_cmap('viridis')
                  )
    ax[2,name].set_ylim(0,5000)
    ax[2,name].set_xlim(0,1.2)
fig.colorbar(pmesh, ax=ax[2,:], shrink=0.6)

for name in range(len(file_list)):
    test=all_sim[name]
    # ~ ax[name].set_title(test.name)
    pmesh = ax[3,name].pcolormesh(test.tau_f3,\
                              test.freq_f3,\
                              np.log10(test.spectrogram_f3/max_freq)#np.log10(globals()['Z_h2'+i[:-4]])/max_freq
                   ,vmin = -5
                   ,vmax=0.
                   ,shading='gouraud'
                              ,rasterized=True
                              ,cmap=plt.get_cmap('viridis')
                  )
    ax[3,name].set_ylim(0,5000)
    ax[3,name].set_xlim(0,1.2)
    ax[3,name].set_xlabel("Time")
fig.colorbar(pmesh, ax=ax[3,:], shrink=0.6)

for name in range(len(file_list)):
    test=all_sim[name]
    tau_mean=np.mean(test.tau_f[1:]-test.tau_f[:-1])
    ax[4,name].semilogy(test.freq_f,np.sum(test.spectrogram_f1[:,:-1]/max_freq*tau_mean,axis=1),color=test.color,ls=test.ticks,alpha=test.alpha )
    # ~ ax[4,name].plot(np.sum((test.spectrogram_f[:,:-1]*(test.tau_f[1:]-test.tau_f[:-1]).T),axis=1)/np.max(test.tau_f),test.freq_f,color=test.color,ls=test.ticks,alpha=test.alpha )

    ax[4,name].set_xlim(0,5000)
    ax[4,name].set_ylim(1e-7,1e-1)
max_sum=np.max([np.max(np.sum((name.spectrogram_f[:,:-1]*(name.tau_f[1:]-name.tau_f[:-1]).T),axis=1)/np.max(name.tau_f)) for name in all_sim]) 
# ~ [axs.set_ylim([0,max_sum]) for axs in ax[4,:]]
[axs.set_ylabel('freq') for axs in ax[:-1,0]]
[axs.grid() for axs in ax[4,:]]

# ~ plt.savefig("Spectrogram_global_all.pdf")

plt.show()
  


