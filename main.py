#%%
"""
This program computes the susceptibilities of a two-tone driven transmon ancilla.
@author: YaxingZhang
"""
#%% load the packages
import timeit
from matplotlib import cm


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as colors
import matplotlib.ticker as mtick
import itertools
import scipy as sp
import math as math

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import curve_fit

import ffmpy

import matplotlib as mpl
mpl.rcParams['axes.linewidth']=3

#%% set the number of considered levels and define ancilla operators
N = 20   # number of ancilla Fock states considered (using displace_frame recues the required N by a factor of 5, and the running time by a factor of 50!!)
N_sub = N  # we calculate the Floquet matrix elements among Fock states |0>,|1> ... |N_sub>


c  = destroy(N)
nc = c.dag()*c
ncsq = c.dag()*c*c.dag()*c


#%% set system parameters 
# all frequencies and decay (dephasing) rates are in unit of GHz

# QCI-Kevin
if 0:
    alpha = 0.192*2*np.pi  # anharmonicity (self-Kerr) of the ancilla 
    omega_c = 5.092*2*np.pi   # frequency of the ancilla, previously it was 4.936 from Chris
    beta = 0 # beta = 4/3*alpha/(1+omega_c/alpha)  # six order nonlinearity
    gamma_c = 1./10./1000  # decay rate of the transmon B
#    gamma_ph_high_freq=0.
    gamma_ph_high_freq = gamma_c/60.  # high-frequency dephasing rate of the ancilla
    gamma_ph_low_freq = 0. # 1./11.3/1000  # low-frequency (Ramsey or spin echo) dephasing rate = 1/T_ph 
    n_th = 0.02  # thermal population
    #delta_a = 5.4660*2*np.pi - omega_c   # detuning of the cavity a from the ancilla
    delta_a = 6.081*2*np.pi - omega_c  # readout cavity
    delta_b = 6.530*2*np.pi- omega_c  # detuning of the cavity b from the ancilla
    delta_r = 8.025*2*np.pi - omega_c # detuning of the readout from the ancilla
    chi_ac = 1.1/1000*2*np.pi  # cross-Kerr between cavity a and ancilla c
    chi_bc = 1.3/1000*2*np.pi   # cross-Kerr between cavity b and ancilla c
    chi_rc = 0.9/1000*2*np.pi

# module B
if 0:
    alpha = 0.168*2*np.pi  # anharmonicity (self-Kerr) of the ancilla 
    omega_c = 4.929*2*np.pi  # frequency of the ancilla, previously it was 4.936 from Chris
    beta = 0 # beta = 4/3*alpha/(1+omega_c/alpha)  # six order nonlinearity
    gamma_c = 1e-3 #1./60./1000  # decay rate of the transmon B
    #gamma_ph_high_freq = gamma_c/30.  # high-frequency dephasing rate of the ancilla
    gamma_ph_low_freq = 0# 1./11.3/1000  # low-frequency (Ramsey or spin echo) dephasing rate = 1/T_ph 
    
    
    n_th = 0.01  # thermal population
#    delta_r = 7.724*2*np.pi - omega_c  # readout cavity
    delta_b = 6.5481*2*np.pi- omega_c  # detuning of the cavity b from the ancilla
    chi_bc = 1.25/1000*2*np.pi   # cross-Kerr between cavity b and ancilla c

# y-mon
if 1:
    alpha = 0.074*2*np.pi  
    omega_c = 5.901*2*np.pi  # frequency of the ancilla          
#    omega_c = 4.7*2*np.pi
    beta = 0 # beta = 4/3*alpha/(1+omega_c/alpha)
    gamma_c = 1./11.1/1000  # decay rate of the ancilla
#    gamma_ph_high_freq = gamma_c/60.  # high-frequency dephasing rate of the ancilla
#    gamma_ph_low_freq = 1./11.3/1000  # low-frequency (Ramsey or spin echo) dephasing rate = 1/T_ph
    gamma_ph_high_freq = 0
    gamma_ph_low_freq = 0
    n_th = 0.02  # thermal population a small value of n_th is used so that the sorting based on steady-state population works
    delta_a = 5.554*2*np.pi - omega_c
    delta_b = 6.543*2*np.pi - omega_c
#    delta_a = 5.4670*2*np.pi - omega_c  # detuning of the cavity a from the ancilla
#    delta_b = 6.5480*2*np.pi- omega_c # detuning of the cavity b from the ancilla
#    chi_ac = 0.45/1000*2*np.pi  # cross-Kerr between cavity a and ancilla c
#    chi_bc = 0.255/1000*2*np.pi   # cross-Kerr between cavity b and ancilla c
    ga = -0.019921*2*np.pi
    gb = 0.028417*2*np.pi
#    ga = 0.09*np.abs(delta_a)
#    gb = 0.1*np.abs(delta_b)
# SNAILmon (preliminary parameters)
if 0:
        # design 2
    if 1:
        omega_c =  (3.7)*2.*np.pi  # 5.92*2*np.pi
#        g3 = 0.0224*2.*np.pi # 0.0225*2*np.pi
#        g4 =  0.000586*2*np.pi #0.00135*2.*np.pi # 0.00065*2*np.pi
        g3 = -3.1*1e-3*2*np.pi
        alpha = 0.3*1e-3*2*np.pi
        g4 = -alpha/12 + 5*g3**2/omega_c # this correspond to alpha = 0.12*2*np.pi
        g5 = 0 
#        n_th = 0.04
        kBT = 2.616 # kBT is k_BT/\hbar in the unit of GHz. T = 20 mK
        n_th =  1/(np.exp(omega_c/kBT)-1)  
        gamma_c = 1./10./1e3   # 10 us life time
        gamma_ph_high_freq = 0
        gamma_ph_low_freq = 0
        omega_a = 2.96*2.*np.pi
        omega_b = 6.92*2.*np.pi
        delta_a = omega_a - omega_c # -0.75*2*np.pi  
        delta_b = omega_b - omega_c # 0.75*2*np.pi
        ga = 0.077*2.*np.pi # 0.09*np.abs(delta_a)
        gb = 0.160*2.*np.pi # 0.112*np.abs(delta_b)
    
    # design 1
    if 0:
        omega_c = 8.463*2.*np.pi  # 5.92*2*np.pi
        g3 = 0.*np.pi # 0.0225*2*np.pi
        g4 = -0.005/12*2.*np.pi # 0.00065*2*np.pi
        g5 = 0.*2*np.pi
        kBT = 2.616 # kBT is k_BT/\hbar in the unit of GHz. T = 20 mK
        n_th =  1/(np.exp(omega_c/kBT)-1)  
        gamma_c = 1./10./1e3
        gamma_ph_high_freq = 0
        gamma_ph_low_freq = 0
        delta_a = 7.7*2.* np.pi - omega_c  # -0.75*2*np.pi  
        delta_b = 7.95*2.* np.pi - omega_c  # 0.75*2*np.pi

# Shantanu
if 0:
    omega_c = 6.5*2*np.pi
    g3 = 0
    alpha = 0.2*2*np.pi
    g4 = -alpha/12
    g5 = 0
    n_th = 0.01
    gamma_c = 1/1e5
    gamma_ph_high_freq = 0
    gamma_ph_low_freq = 0
    delta_a = 4.9*2.* np.pi - omega_c    
    delta_b = 5.1*2.* np.pi - omega_c  
    chi_ac = 1./1e3*2*np.pi 
    chi_bc = 1./1e3*2*np.pi
        
# pitch and catch
if 0:

    alpha = 0.12*2*np.pi  # anharmonicity (self-Kerr) of the ancilla 
    omega_c = 5.082*2.*np.pi  # frequency of the ancilla
    beta = 0
    gamma_c = 1./10./1000  # decay rate of the ancilla
    gamma_ph_high_freq = gamma_c/30.  # high-frequency dephasing rate of the ancilla
    gamma_ph_low_freq = (1./9.-1/10./2)/1000  # low-frequency (Ramsey or spin echo) dephasing rate = 1/T_ph 
    n_th = 0.05  # thermal population    
    delta_b = 6.51434*2*np.pi - omega_c  # detuning of the cavity b from the ancilla
    delta_a = 5.643*2*np.pi- omega_c # detuning of the cavity a from the ancilla    
    chi_bc = 0.765/1000*2*np.pi  # cross-Kerr between cavity b and ancilla c
    chi_ac = 4.1/1000*2*np.pi   # cross-Kerr between cavity a and ancilla c


# parametric modulation SQUID
if 0:
    alpha = 0.01*2*np.pi
    omega_c = 7.4*2*np.pi
    n_th = 0.02
    omega_a = 6.1*2*np.pi
    omega_b = 6.4*2*np.pi
    delta_a = omega_a - omega_c
    delta_b = omega_b - omega_c
    gamma_c = 1e-4  # decay rate of the ancilla
    gamma_ph_low_freq = 0
    gamma_ph_high_freq = gamma_c
    ga = 0.12*2*np.pi
    gb = 0.09*2*np.pi
    

# the following calculates the bare frequencies and couplings of transmon and cavities taking into account 
# the correction due to dispersive shift. The difference between these bare parameters and the dressed parameters should be small

if 0:    
    ga = np.sqrt(chi_ac*delta_a*(delta_a+alpha)/alpha/2) # calculate the bare coupling rate between a and c
    gb = np.sqrt(chi_bc*delta_b*(delta_b+alpha)/alpha/2) # calculate the bare coupling rate between b and c
#    gr = np.sqrt(chi_rc*delta_r*(delta_r+alpha)/alpha/2)
    omega_c_bare = omega_c + ga**2/delta_a + gb**2/delta_b
    delta_a_bare = delta_a -2*ga**2/delta_a - gb**2/delta_b
    delta_b_bare = delta_b -2*gb**2/delta_b - ga**2/delta_a
    ga_bare = np.sqrt(chi_ac*delta_a_bare*(delta_a_bare+alpha)/alpha/2) # calculate the bare coupling rate between a and c
    gb_bare = np.sqrt(chi_bc*delta_b_bare*(delta_b_bare+alpha)/alpha/2) # calculate the bare coupling rate between b and c
    print(r'$(\omega_c-\omega_c_bare)/2\pi= $'+repr((omega_c-omega_c_bare)/2/np.pi))
    print(r'$(\omega_a-\omega_a_bare)/2\pi= $'+repr((delta_a-delta_a_bare)/2/np.pi))
    print('ga/ga_bare = '+repr(ga/ga_bare))
    print('gb/gb_bare = '+repr(gb/gb_bare))    
    omega_c = omega_c_bare
    ga = ga_bare
    gb = gb_bare


#%% set the pumping parameters
two_RWA_pump = 0
three_RWA_pump = 1
one_RWA_pump = 0
one_nonRWA_pump = 0
two_nonRWA_pump = 0   # we assume the two pump frequencies are commensurate and reduce the problem to one single non-RWA pump
two_parametric_pump = 0

if two_RWA_pump:
    if 1:  
        delta1 = 6.058*2*np.pi-omega_c
        delta2 = 7.049624*2*np.pi-omega_c
        omega21 = np.abs(delta2-delta1)  # this is the frequency difference of the two drives; 
        phi = 0
#        xi1_vec = np.sqrt(np.linspace(0,0.3,20))
#        xi2_vec = np.sqrt(np.linspace(0,0.3,20))
#        F1_vec = xi1_vec*np.absolute(delta1)  # F1, F2 are the drive amplitudes corresponding to \Omega_1,\Omega_2 in the paper
#        F2_vec = xi2_vec*np.absolute(delta2)
        F1_vec = np.array([0.0942])*2*np.pi
        F2_vec = np.array([0.229725])*2*np.pi
    
    if 0:   # new-design
        delta1 = 1.808*2*np.pi  # detuning of the drive-1 from the ancilla frequency. delta1 = omega_1 - omega_c
        delta2 = delta_b-delta_a + delta1  # detuning of the drive-2 from the ancilla frequency. delta2 = omega_2 - omega_c
        
        omega21 = np.abs(delta2-delta1)  # this is the frequency difference of the two drives; 
        
        phi = 0 # phi is the relative phase between F1 and F2 in unit of radian; it affects the phase of the susceptibilities but not the amplitude; can set to zero.
        
#        xi1_vec = np.linspace(0,3.,10)
#        xi2_vec = np.linspace(0,3.,10)
        xi1_vec = np.sqrt(np.linspace(0.04,0.8,1)*56.37)/delta1
        xi2_vec = np.sqrt(np.linspace(0.04,0.8,1)*86.085)/delta2
        F1_vec = xi1_vec*np.absolute(delta1)  # F1, F2 are the drive amplitudes corresponding to \Omega_1,\Omega_2 in the paper
        F2_vec = xi2_vec*np.absolute(delta2)
        
    
    if 0: # frank-condon
        delta1 = delta_a-0.225*2*np.pi  # detuning of the drive-1 from the ancilla frequency. delta1 = omega_1 - omega_c
        delta2 = delta_b-0.225*2*np.pi  # detuning of the drive-2 from the ancilla frequency. delta2 = omega_2 - omega_c
        
        omega21 = np.abs(delta2-delta1)  # this is the frequency difference of the two drives; 
        
        phi = 0 # phi is the relative phase between F1 and F2 in unit of radian; it affects the phase of the susceptibilities but not the amplitude; can set to zero.
        
        
#        xi1_vec = 0.3*np.linspace(0.,0.2,15)*2.5*np.sqrt(10)
#        xi2_vec = 0.3*np.linspace(0.,1.06,15)*2.5*np.sqrt(10)
        xi1_vec = 0.3*np.linspace(0.142857,0.17143,15)*2.5*np.sqrt(10)
        xi2_vec = 0.3*np.linspace(0.75714,0.90857,15)*2.5*np.sqrt(10)
        F1_vec = xi1_vec*np.absolute(delta1)  # F1, F2 are the drive amplitudes corresponding to \Omega_1,\Omega_2 in the paper
        F2_vec = xi2_vec*np.absolute(delta2)
    if 0: # pitch and catch
        delta1 = (0.140)*2*np.pi  # detuning of the drive-1 from the ancilla frequency. delta1 = omega_1 - omega_c
        delta2 = delta_b-delta_a + delta1  # detuning of the drive-2 from the ancilla frequency. delta2 = omega_2 - omega_c
        
        omega21 = delta2-delta1  # this is the frequency difference of the two drives; 
        
        phi = 0. # phi is the relative phase between F1 and F2 in unit of radian; it affects the phase of the susceptibilities but not the amplitude; can set to zero.
        F1_vec = np.linspace(0.,2.5,15)*delta1  # F1, F2 are the drive amplitudes corresponding to \Omega_1,\Omega_2 in the paper
        F2_vec = np.linspace(0,2.5,15)*delta2/7

    if 0: # Kerr cancelation
        delta1 = alpha/10  # detuning of the drive-1 from the ancilla frequency. delta1 = omega_1 - omega_c
        delta2 = alpha/10 - alpha  # detuning of the drive-2 from the ancilla frequency. delta2 = omega_2 - omega_c
        
        omega21 = delta2-delta1  # this is the frequency difference of the two drives; 
        
        phi = 0. # phi is the relative phase between F1 and F2 in unit of radian; it affects the phase of the susceptibilities but not the amplitude; can set to zero.
        F1_vec = np.sqrt(np.linspace(0.005,2.5,1))*delta1  # F1, F2 are the drive amplitudes corresponding to \Omega_1,\Omega_2 in the paper
        F2_vec = np.linspace(0,2.5,15)*delta2/7

if three_RWA_pump:
    omega1 = 6.058*2*np.pi #6.058*2*np.pi
    omega2 = 7.058*2*np.pi
    omega3 = 6.758*2*np.pi    
    delta1 = omega1 -omega_c
    delta2 = omega2 -omega_c
    delta3 =  omega3 - omega_c
    omega21 = np.abs(omega2-omega1)  # this is the frequency difference of the two drives; 
    omega31 = np.abs(omega3-omega1)
    phi21 = 0
    phi31 = 0
    F1_vec = np.linspace(0,0.0942,10)*2*np.pi
    F2_vec = np.linspace(0,0.229725,10)*2*np.pi        
    F3_vec = np.linspace(0,0.271093,10)*2*np.pi

if one_nonRWA_pump:
#    omega1 = delta_b - delta_a
    omega1 = 3.96*2*np.pi 
    xi_vec = np.linspace(0.,5.,10)
    F1_vec = xi_vec*np.absolute(omega1**2-omega_c**2)/2./omega1  # F1, F2 are the drive amplitudes corresponding to \Omega_1,\Omega_2 in the paper
#    F1_vec = np.sqrt(xi_vec_sq)*(omega1-omega_c)
    F2_vec = np.linspace(0,0.37,1)
    omega21 = omega1    # this definition is just so that we can use the same code of susceptibility for two_RWA_pump
    delta1 = - omega_c  # this definition is just so that we can use the same code of susceptibility for two_RWA_pump

if two_nonRWA_pump:
    omega1 = 12.*2*np.pi
    omega2 = omega1+(delta_b-delta_a)
    phi1 = 0.
    phi2 = 0.
# here we assume that both omega1 and omega2 have effective digits at 1 MHz level, thus become integers if multiply 1000 in the unit of GHz. 
#omega_com is the greatest common divisor of omega1 and omega2 in the unit of 1 MHz, then we convert it to unit GHz.    
    omega_com = math.gcd(round(omega1/2/np.pi*1000),round(omega2/2/np.pi*1000))/1000*2*np.pi    
    xi1_vec = np.sqrt(np.linspace(0.,3.2,2))
    xi2_vec = np.sqrt(np.linspace(0.,3.2,2))
    F1_vec = xi1_vec*np.absolute(omega1**2-omega_c**2)/2./omega1  # F1, F2 are the drive amplitudes corresponding to \Omega_1,\Omega_2 in the paper
    F2_vec = xi2_vec*np.absolute(omega2**2-omega_c**2)/2./omega2  # F1, F2 are the drive amplitudes corresponding to \Omega_1,\Omega_2 in the paper
    omega21 = omega_com   # this definition is just so that we can use the same code of susceptibility for two_RWA_pump
    delta1 = - omega_c  # this definition is just so that we can use the same code of susceptibility for two_RWA_pump



if two_parametric_pump:
    beating_only = 0
    if beating_only:
        omega21 = np.abs(delta_b-delta_a)
        F1_vec = np.sqrt(np.linspace(0.,(0.2*2*np.pi)**2,10))  # F1, F2 here are the scaled external flux F1 = 2pi*(\Phi_ext1/\Phi_0), same for F2    
        F2_vec = np.linspace(0,0.2,1)*2*np.pi  
        delta1 = - omega_c
        n_Bessel = 2 # the highest order of Bessel series
    else:
        ratio = -1./9.
        omega1 = 2.7*2*np.pi
        omega2 = omega1 + (delta_b-delta_a)
        omega21 = omega2 - omega1
        F1_vec = np.sqrt(np.linspace(0,(0.2*2*np.pi)**2,15))
        F2_vec = np.sqrt(np.linspace(0,(0.2*2*np.pi)**2,15))
        omega_com = math.gcd(round(omega1/2/np.pi*1000),round(omega2/2/np.pi*1000))/1000*2*np.pi            
        delta1 = - omega_c
        

if one_RWA_pump:
    delta1 = 2*alpha
    #    delta1 = 1.808*2*np.pi + (delta_b-delta_a)
#    delta1 = delta2
    # F1, F2 are the drive amplitudes corresponding to \Omega_1,\Omega_2 in the paper
#    xi_vec_sq = np.array([0.195,0.2,0.205,0.21,0.215])
#    xi_vec_sq = np.array([0,0.3,0.6,0.9])
#    xi_vec_sq =np.linspace(0.,2.,10)
#    F1_vec = np.sqrt(xi_vec_sq)*delta1  #    xi1_vec = np.linspace(0,2.,20)
#    F1_vec = xi1_vec*delta1_bare
#    F1_vec = np.sqrt(np.linspace(0.,2,20))*delta1_bare
    F1_vec = np.sqrt(np.linspace(0.4605,1.,1))*delta1
    F2_vec = 0.3*np.linspace(0,0.37,1)
    omega21 = 1.




#%% set the relevant time parameters 
# initialize the quasienergy vectors, Floquet matrix elements, etc... 

calculate_W_rates = 0
calculate_V_rates = 0
calculate_nc = 0
check_adiabaticity = 0
displaced_frame = 0
waveguide = 0


if three_RWA_pump:
    # period of the ancilla Hamiltonian in the rotating frame of drive-1. 
    omega_com = math.gcd(int(round(omega21/2/np.pi*1000)),int(round(omega31/2/np.pi*1000)))/1000*2*np.pi            
    T = 2*np.pi / omega_com  
    # set the time steps for tracking the time-dependent Floquet states
    N_time = 1000
    dt = T/N_time
    times=np.linspace(0,T-dt,N_time)
    freq = np.fft.fftfreq(N_time, dt)
    
if two_RWA_pump:
    # period of the ancilla Hamiltonian in the rotating frame of drive-1. 
    T = 2*np.pi / omega21  
    # set the time steps for tracking the time-dependent Floquet states
    N_time = 100
    dt = T/N_time
    times=np.linspace(0,T-dt,N_time)
    freq = np.fft.fftfreq(N_time, dt)

if one_nonRWA_pump:
    T = 2*np.pi / omega1
    # set the time steps for tracking the time-dependent Floquet states
    N_time = 100
    dt = T/N_time
    times=np.linspace(0,T-dt,N_time)
    freq = np.fft.fftfreq(N_time, dt)
#    cdag_element_fft_sorted = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec))) + np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))*1j
    nch_element_fft_sorted = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec))) + np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))*1j
    def n_th_omega(omega,kBT):     
        return 1./(np.exp(omega/kBT)-1.)
    def filter_gamma(omega):   # filter_gamma(omega) = ReY(omega)*omega_c/(omega *C_q) where Y(omega) is the admittance seen by the junctioon, and C_q is the capacitance of the converter mode.  
        return gamma_c 

if two_nonRWA_pump:
    T = 2*np.pi / omega_com
    # set the time steps for tracking the time-dependent Floquet states
    N_time = 500    # this number has to be much larger than max(omega1,omega2)/omega_com
    dt = T/N_time
    times=np.linspace(0,T-dt,N_time)
    freq = np.fft.fftfreq(N_time, dt)
#    cdag_element_fft_sorted = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec))) + np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))*1j
    nch_element_fft_sorted = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec))) + np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))*1j
    phi_element_fft_sorted = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec))) + np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))*1j
    nch_fft_sort_idx = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))
    phi_fft_sort_idx = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))

    def n_th_omega(omega,kBT):     
        return 1./(np.exp(omega/kBT)-1.)
    def filter_gamma(omega):   # filter_gamma(omega) = ReY(omega)*omega_c/(omega *C_q) where Y(omega) is the admittance seen by the junctioon, and C_q is the capacitance of the converter mode.  
        return gamma_c 
   
if two_parametric_pump:
    T = 2*np.pi/omega_com    
    N_time = 350    # half of his number needs to be bigger than 4*omega_c/omega_com 
    dt = T/N_time
    times=np.linspace(0,T-dt,N_time)
    freq = np.fft.fftfreq(N_time, dt)
    nch_fft_sort_idx = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))
    phi_fft_sort_idx = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))


    
if one_RWA_pump: # for the case of one RWA pump, FLoquet is not needed. But to apply the functions defined for two-pump case, we introduce keep the dummy Fourier index.
    N_time = 2



q_energies_sorted = np.zeros((N,len(F1_vec),len(F2_vec)))
c_element_fft_sorted = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec))) + np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))*1j
nc_element_fft_sorted = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec))) + np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))*1j
nch_element_fft_sorted = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec))) + np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))*1j
phi_element_fft_sorted = np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec))) + np.zeros((N_sub,N_sub,N_time,len(F1_vec),len(F2_vec)))*1j



if check_adiabaticity:
# duration of ramping up in unit of nanosecond
#    tau_ramp = times[-1]+T   # use this tau_ramp for compute_adiabatic_state_two_RWA
    tau_ramp = 50.
    N_ramp = 10
    times_ramp =np.linspace(0,tau_ramp,N_ramp+1)
    overlap = np.zeros(len(times_ramp))
    f_modes_t_ground = [None for i in range(len(times_ramp))]
    
if waveguide:
    
    cutoff = omega_c + 0.7*2*np.pi
    
    # def waveguide_gamma(strength,omega):
    #    return gamma_c + strength*gamma_c*np.heaviside(omega-cutoff,0)
    
    # a refined model for frequency-dependent gamma
    
    def waveguide_gamma(strength,omega):
        return gamma_c + strength*gamma_c*np.heaviside(omega-cutoff,0) + strength*gamma_c*np.exp(-1.1/3*np.sqrt(np.abs(cutoff**2 - (omega)**2))) * np.heaviside(omega,1) * np.heaviside (cutoff-omega,1)  

    
    
if calculate_W_rates:
    W_rates = np.zeros((N_sub,N_sub,len(F1_vec),len(F2_vec))) 
if calculate_V_rates:
    V_rates = np.zeros((N_sub,N_sub,len(F1_vec),len(F2_vec))) 

#%%  the main function
# the following program calculates the sorted matrix elements of c and nc, and transition rates and linewidths among Floquet states.

start_time = timeit.default_timer()


if two_RWA_pump:
    for idx1 in range(0,len(F1_vec)): 
    #    for idx2 in range(0,len(F2_vec)): 
        if 1:
    #        alpha = alpha_vec[idx1]
    #        delta1 = delta1_vec[idx1]        
            idx2 = idx1
            F1 = F1_vec[idx1]
            F2 = F2_vec[idx2]
            if 1:
                q_energies_sorted[:,idx1,idx2],c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2],overlap,f_modes_t_ground = two_pump_Floquet(delta1,delta2,alpha,beta,F1,F2,calculate_nc)
            # find the hopping rates W  
            if calculate_W_rates:
                if waveguide:
                    W_rates[:,:,idx1,idx2] = compute_W_rates_waveguide(1.1*1e3,q_energies_sorted[:,idx1,idx2],c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2])   
                else:
                    W_rates[:,:,idx1,idx2] = compute_W_rates(c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2],calculate_nc)
            # find the linewidths V    
            if calculate_V_rates:
                V_rates[:,:,idx1,idx2] = compute_V_rates(W_rates[:,:,idx1,idx2],c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2])
    

if three_RWA_pump:
    for idx1 in range(0,len(F1_vec)): 
    #    for idx2 in range(0,len(F2_vec)): 
        if 1:
    #        alpha = alpha_vec[idx1]
    #        delta1 = delta1_vec[idx1]        
            idx2 = idx1
            idx3 = idx1
            F1 = F1_vec[idx1]
            F2 = F2_vec[idx2]
            F3 = F3_vec[idx3]
            if 1:
                q_energies_sorted[:,idx1,idx2],c_element_fft_sorted[:,:,:,idx1,idx2],overlap,f_modes_t_ground = three_pump_Floquet(delta1,delta2,delta3,alpha,beta,F1,F2,F3,calculate_nc)
            # find the hopping rates W  
            if calculate_W_rates:
                if waveguide:
                    W_rates[:,:,idx1,idx2] = compute_W_rates_waveguide(1.1*1e3,q_energies_sorted[:,idx1,idx2],c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2])   
                else:
                    W_rates[:,:,idx1,idx2] = compute_W_rates(c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2],calculate_nc)
            # find the linewidths V    
            if calculate_V_rates:
                V_rates[:,:,idx1,idx2] = compute_V_rates(W_rates[:,:,idx1,idx2],c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2])
    




# one non-RWA pump experiment    
if one_nonRWA_pump:    
    for idx1 in range(0,len(F1_vec)): 
        F1 = F1_vec[idx1]
        idx2 = 0  # we keep the index for pump-2 so the functions for two-pump can be directly applied to here.            
        if 1:
            q_energies_sorted[:,idx1,idx2],nch_element_fft_sorted[:,:,:,idx1,idx2] = one_pump_nonRWA(omega1,g3,g4,F1,calculate_nc)
        if calculate_W_rates:
            W_rates[:,:,idx1,idx2] = compute_W_rates_nonRWA(omega1,q_energies_sorted[:,idx1,idx2],nch_element_fft_sorted[:,:,:,idx1,idx2]) 
        
        # find the linewidths V    
        if calculate_V_rates:
            V_rates[:,:,idx1,idx2] = compute_V_rates(W_rates[:,:,idx1,idx2],c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2])
   
# two non-RWA pump experiment
if two_nonRWA_pump:            
    for idx1 in range(0,len(F1_vec)): 
        F1 = F1_vec[idx1]
        idx2 = idx1  # we keep the index for pump-2 so the functions for two-pump can be directly applied to here.            
        F2 = F2_vec[idx2]
        if 1:
            q_energies_sorted[:,idx1,idx2],nch_element_fft_sorted[:,:,:,idx1,idx2],phi_element_fft_sorted[:,:,:,idx1,idx2],nch_fft_sort_idx[:,:,:,idx1,idx2],phi_fft_sort_idx[:,:,:,idx1,idx2] = two_pump_nonRWA(omega1,omega2,g3,g4,F1,F2,calculate_nc)
        if calculate_W_rates:
            W_rates[:,:,idx1,idx2] = compute_W_rates_nonRWA(omega_com,q_energies_sorted[:,idx1,idx2],nch_element_fft_sorted[:,:,:,idx1,idx2]) 


# two parametric pump experiment
if two_parametric_pump:
    for idx1 in range(0,len(F1_vec)): 
        F1 = F1_vec[idx1]
        if beating_only:
            idx2 = 0 # F2 = F1 is assumed for this case
        else:
            idx2 = idx1
        F2 = F2_vec[idx2]
 #       q_energies_sorted[:,idx1,idx2],c_element_fft_sorted[:,:,:,idx1,idx2],nch_element_fft_sorted[:,:,:,idx1,idx2],nch_fft_sort_idx[:,:,:,idx1,idx2],phi_element_fft_sorted[:,:,:,idx1,idx2],phi_fft_sort_idx[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2] = two_pump_parametric(omega_c,alpha,F1,F2,calculate_nc)
        if calculate_W_rates:
#            W_rates[:,:,idx1,idx2] = compute_W_rates(c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2],calculate_nc)
            W_rates[:,:,idx1,idx2] = compute_W_rates_nonRWA(omega_com,q_energies_sorted[:,idx1,idx2],nch_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2],0)

        # find the linewidths V    
        if calculate_V_rates:
            V_rates[:,:,idx1,idx2] = compute_V_rates(W_rates[:,:,idx1,idx2],c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2])



# one RWA pump experiment
if one_RWA_pump:
    start_time = timeit.default_timer()
    for idx1 in range(0,len(F1_vec)): 
        F1 = F1_vec[idx1]
        idx2 = 0  # we keep the index for pump-2 so the functions for two-pump can be directly applied to here.
        q_energies_sorted[:,idx1,idx2],c_element_fft_sorted[:,:,0,idx1,idx2],nc_element_fft_sorted[:,:,0,idx1,idx2] = one_pump_RWA(delta1,alpha,F1,calculate_nc)
        if calculate_W_rates:
            W_rates[:,:,idx1,idx2] = compute_W_rates(c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2],calculate_nc)
        # find the linewidths V    
        if calculate_V_rates:
            V_rates[:,:,idx1,idx2] = compute_V_rates(W_rates[:,:,idx1,idx2],c_element_fft_sorted[:,:,:,idx1,idx2],nc_element_fft_sorted[:,:,:,idx1,idx2])
    
    
elapsed = timeit.default_timer() - start_time
print('the time used is' + repr(elapsed))    



#%%
Kerr = (q_energies_sorted[2,:,0]-2*q_energies_sorted[1,:,0]+q_energies_sorted[0,:,0])/2/np.pi*1e3    
freq = (q_energies_sorted[1,0,0] - q_energies_sorted[0,0,0])/2/np.pi
print('Kerr = ' + repr(Kerr[0]) + 'MHz*2*pi')    
print('freq = ' + repr(freq)+ 'GHz*2*pi')
    
#%%############################ 
#calculate the steady_state population among Floquet states
population_steady_4 = np.zeros((N_sub,len(F1_vec),len(F2_vec)))
for idx1 in range(0,len(F1_vec)):
    for idx2 in range(0,len(F2_vec)):
        population_steady_4[:,idx1,idx2] = compute_steady_population(W_rates[:,:,idx1,idx2],N_sub)
#%%        
np.diag(population_steady[0,:,:])

#%% find not in g population at a given time
expop_th = np.zeros((len(F1_vec),len(F2_vec)))
for F1_idx in range(0,len(F1_vec)):
    for F2_idx in range(0,len(F2_vec)):
        expop_th[F1_idx,F2_idx] = 1-compute_population_time_evoluation(W_rates[:,:,F1_idx,F2_idx],N_sub)[0,0]        
        
#%% plot the steady-state population in the FLoquet ground
plot_excited_state_population(70)

#%% plot the heating rate
plot_W_rates(-0.01,0.03,70)

#%% Stark shift
plot_AC_stark_shift(-100,10,70)

#%% quasienergy
plot_quasienergy(-0.6,0.6,70)

#%%################################ # beamsplitter spectrum and chi spectrum


if two_RWA_pump or two_parametric_pump or three_RWA_pump:
    omega_probe = np.linspace(-1.5*omega21, 1.5*omega21,4000)  # set the range of probe frequency
if one_nonRWA_pump:
    omega_probe = np.linspace(-0.5*omega1, 1.1*omega1,2000)  # set the range of probe frequency
if two_nonRWA_pump:
    omega_probe = omega_c+np.linspace(-omega_c/2, omega_c/4,1000)  # omega_probe here is the probe frequency in the lab frame
 
if one_RWA_pump:
    omega_probe = np.linspace(-10*alpha, 10*alpha,2000)  # set the range of probe frequency
       

#%%
plot_re_partial_beamsplitter_fixed_drive(0,10,N_time,0,0,-0.3,0.3,70,1)

# plot_chi_cavity_fixed_drive(N,N_time,-1,0,-1,1,70,1)

# plot_re_partial_chi_fixed_drive(15,N_time,50,0,-0.1,0.1,70,0)

#%% beamsplitter strength

if one_nonRWA_pump:
    gBS = np.zeros((5,len(F1_vec),len(F2_vec)))
    for i in range(0,5):
        gBS[i,:,:] = ga*gb*np.abs(compute_partial_beamsplitter_at_delta_probe_one_nonRWA(i,N_sub,100,delta_a))
if two_RWA_pump:
    gBS = np.zeros((1,len(F1_vec)))
    for i in range(0,1):
        gBS[i,:] = ga*gb*np.real(np.diag(compute_partial_beamsplitter_at_delta_probe(i,1,omega21,8,50,delta_a)))

if three_RWA_pump:
    gBS = np.zeros((1,len(F1_vec)))
    for i in range(0,1):
        gBS[i,:] = ga*gb*np.real(np.diag(compute_partial_beamsplitter_at_delta_probe(i,int(round(omega21/omega_com)),omega_com,8,50,delta_a)))


if two_parametric_pump:
    gBS = np.zeros((4,len(F1_vec)))
    for i in range(0,4):
        for F1_idx in range(0,len(F1_vec)):
            gBS[i,F1_idx] = ga*gb*np.real(compute_partial_beamsplitter_at_delta_probe_one_nonRWA(i,round(omega21/omega_com),omega_com,10,N_time,delta_a)[F1_idx,F1_idx])
    

#%%

delta_probe= delta_a 
plot_re_partial_beamsplitter_at_delta_probe(-.01,6,70)   

#%%        
plot_gBS_dispersion(70)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# dispersive frequency shift


if one_nonRWA_pump:
    dw_a = np.zeros((5,len(F1_vec),len(F2_vec)))
    dw_b = np.zeros((5,len(F1_vec),len(F2_vec)))
    delta_BS = np.zeros((5,len(F1_vec),len(F2_vec)))
    for i in range(0,5):
        dw_a[i,:,:] = -ga**2*np.real(compute_partial_chi_at_delta_probe_one_nonRWA(i,N_sub,100,delta_a))
        dw_b[i,:,:] = -gb**2*np.real(compute_partial_chi_at_delta_probe_one_nonRWA(i,N_sub,100,delta_b))
        delta_BS[i,:,:] = (dw_a[i,:,:]-dw_a[0,:,:])-(dw_b[i,:,:]-dw_b[0,:,:])  # this is \chi_b - \chi_a for i =0


if two_RWA_pump:
    dw_a = np.zeros((5,len(F1_vec)))
    dw_b = np.zeros((5,len(F1_vec)))
    delta_BS = np.zeros((5,len(F1_vec)))
    for i in range(0,5):
        dw_a[i,:] = -ga**2*np.real(np.diag(compute_partial_chi_at_delta_probe(i,10,25,delta_a)))
        dw_b[i,:] = -gb**2*np.real(np.diag(compute_partial_chi_at_delta_probe(i,10,25,delta_b)))
        delta_BS[i,:] = (dw_a[i,:]-dw_a[0,:])-(dw_b[i,:]-dw_b[0,:])  # this is \chi_b - \chi_a for i =0
    

if two_parametric_pump:
    dw_a = np.zeros((4,len(F1_vec)))
    dw_b = np.zeros((4,len(F1_vec)))
    delta_BS = np.zeros((4,len(F1_vec)))
    for i in range(0,4):
        dw_a[i,:] = -ga**2*np.real(np.diag(compute_partial_chi_at_delta_probe_one_nonRWA(i,omega_com,10,N_time,delta_a)))
        dw_b[i,:] = -gb**2*np.real(np.diag(compute_partial_chi_at_delta_probe_one_nonRWA(i,omega_com,10,N_time,delta_b)))
        delta_BS[i,:] = (dw_a[i,:]-dw_a[0,:])-(dw_b[i,:]-dw_b[0,:])  # this is \chi_b - \chi_a for i =0    
    
    
#%%

chi_a = (dw_a[0]-dw_a[1])   # this is \chi_a 
chi_b = (dw_b[0]-dw_b[1])   # this is \chi_b


#%%
plot_re_partial_chi_delta_probe(-5,1,70,showtheory=0)


#%% compute thermal infidelity
t_SWAP = np.pi/2/np.abs(gBS[0])
    
gBS_tilde = np.sqrt(gBS**2+delta_BS**2/4)
c_a = np.cos(gBS_tilde * t_SWAP) - 1j*delta_BS/2/gBS_tilde*np.sin(gBS_tilde * t_SWAP)
c_b = -1j*gBS/gBS_tilde*np.sin(gBS_tilde * t_SWAP)
fidelity_thermal = np.zeros(len(F1_vec)) 
population_thermal = np.zeros((5,len(F1_vec)))

for m in range(0,4):
    population_thermal[m] = (n_th/(n_th+1))**(m+1)/n_th*np.ones(len(F1_vec))  
    fidelity_thermal += population_thermal[m]*np.abs(np.conjugate(c_a[m])*c_a[0]+np.conjugate(c_b[m])*c_b[0])**2
    





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# swap infidelity

if one_nonRWA_pump:
    dkappa_a = 2*ga**2*np.imag(compute_partial_chi_at_delta_probe_one_nonRWA(0,N_sub,100,delta_a))
    dkappa_b = 2*gb**2*np.imag(compute_partial_chi_at_delta_probe_one_nonRWA(0,N_sub,100,delta_b))
if two_RWA_pump:
    if 1:
        dkappa_a = 2*ga**2*np.imag(np.diag(compute_partial_chi_at_delta_probe(0,10,25,delta_a)))
        dkappa_b = 2*gb**2*np.imag(np.diag(compute_partial_chi_at_delta_probe(0,10,25,delta_b)))
    if waveguide:
        dkappa_a = ga**2*np.imag(np.diag(compute_Purcel_waveguide_at_delta_probe(1.1*1e3,10,40,delta_a)))
        dkappa_b = gb**2*np.imag(np.diag(compute_Purcel_waveguide_at_delta_probe(1.1*1e3,10,40,delta_b)))


if two_parametric_pump:
    if 1:
        dkappa_a = 2*ga**2*np.imag(compute_partial_chi_at_delta_probe(0,10,N_time,delta_a)[:,0])
        dkappa_b = 2*gb**2*np.imag(compute_partial_chi_at_delta_probe(0,10,N_time,delta_b)[:,0])


#    dkappa_a = 2*ga**2*np.real(np.diag(compute_instant_kappa_at_delta_probe(0,N_sub,10,delta_a)))
#    dkappa_b = 2*gb**2*np.real(np.diag(compute_instant_kappa_at_delta_probe(0,N_sub,10,delta_b)))


t_SWAP = np.pi/2/np.abs(gBS[0])

infidelity_intrinsicdecay = 1e-6*t_SWAP  # assuming 1 ms T1 for cavities a and b.
infidelity_Purcell_0 = (dkappa_a[0] + dkappa_b[0])*t_SWAP/2
# infidelity_Purcell_0 = ((ga/delta_a)**2+(gb/delta_b)**2-(ga/(omega_a+omega_c))**2-(gb/(omega_b+omega_c))**2)*gamma_c*t_SWAP
infidelity_Purcell = (np.abs(dkappa_a) + np.abs(dkappa_b))*t_SWAP/2


gBS_tilde = np.sqrt(gBS**2+delta_BS**2/4)


c_a = np.cos(gBS_tilde * t_SWAP) - 1j*delta_BS/2/gBS_tilde*np.sin(gBS_tilde * t_SWAP)
c_b = -1j*gBS/gBS_tilde*np.sin(gBS_tilde * t_SWAP)


if one_nonRWA_pump:
    population_thermal = np.zeros((5,len(F1_vec),len(F2_vec)))
    fidelity_thermal = np.zeros((len(F1_vec),len(F2_vec)))    
    infidelity_heating = np.zeros((len(F1_vec),len(F2_vec)))    

    for m in range(0,5):
        population_thermal[m] = (n_th/(n_th+1))**(m+1)/n_th*np.ones((len(F1_vec),len(F2_vec)))  
        fidelity_thermal += population_thermal[m]*np.abs(np.conjugate(c_a[m])*c_a[0]+np.conjugate(c_b[m])*c_b[0])**2
#        infidelity_heating += W_rates[0,m,:,:]*t_SWAP/4*(3/2*(delta_BS[m]/2/gBS[m])**2+np.pi**2/3*((gBS[m]-gBS[0])/gBS[0])**2) 
    for m in range(1,5):
        infidelity_heating +=   W_rates[0,m,:,:]*t_SWAP/2*(1+ np.sin(2*gBS_tilde[m]*t_SWAP)*np.abs(gBS[0])/np.pi*gBS[m]/gBS_tilde[m]*(gBS[m]+gBS[0])/(gBS_tilde[m]**2-gBS[0]**2))


if two_parametric_pump:
    infidelity_heating = np.zeros(len(F1_vec))
    for m in range(1,5):
        infidelity_heating += W_rates[0,m,:,0]*t_SWAP/2*(1+ np.sin(2*gBS_tilde[m]*t_SWAP)*np.abs(gBS[0])/np.pi*gBS[m]/gBS_tilde[m]*(gBS[m]+gBS[0])/(gBS_tilde[m]**2-gBS[0]**2))
    
    
if two_RWA_pump:
    population_thermal = np.zeros((5,len(F1_vec)))
    fidelity_thermal = np.zeros(len(F1_vec))    
    infidelity_heating = np.zeros(len(F1_vec))
    for m in range(0,5):
        population_thermal[m] = (n_th/(n_th+1))**(m+1)/n_th*np.ones((len(F1_vec)))  
#        fidelity_thermal += population_thermal[m]*np.abs(np.conjugate(c_a[m])*c_a[0]+np.conjugate(c_b[m])*c_b[0])**2
        fidelity_thermal += population_thermal[m]*np.abs(np.conjugate(c_a[m])*c_a[0]+np.conjugate(c_b[m])*c_b[0])**2
#        infidelity_heating += np.diag(W_rates[0,m,:,:])*t_SWAP/4*(3/2*(delta_BS[m]/2/gBS[m])**2+np.pi**2/3*((gBS[m]-gBS[0])/gBS[0])**2) 
    for m in range(1,5):
        infidelity_heating += np.diag(W_rates[0,m,:,:])*t_SWAP/2*(1+ np.sin(2*gBS_tilde[m]*t_SWAP)*np.abs(gBS[0])/np.pi*gBS[m]/gBS_tilde[m]*(gBS[m]+gBS[0])/(gBS_tilde[m]**2-gBS[0]**2))


#%%

plot_SWAP_fidelity(1e-5,1e-2,70)

#%%
plot_BS_fidelity(70)


#%% set the bath parameter

# plot the susceptiblity spectrum Im chi(omega,omega) when the ancilla is in a given Floquet state u_n
# the strength of this spectrum at a given frequency corresponds to the inverse Purcell decay rate
if 0: 
    data = np.loadtxt( 'f_fluxsum_γ_noSIPF.dat' )
    omega_max = data[0,-1]*2*np.pi
    omega_min = data[0,0]*2*np.pi
    omega_dist = (data[0,1]-data[0,0])*2*np.pi
    gamma_omega = data[1,:]**2*data[0,:]*7.71*1e-6*2*np.pi*(2*np.pi)**2 # this is the frequency dependent damping \gamma(omega) in unit of GHz.


def filter_gamma(r,omega):
    if r == 0:
        return gamma_c
    else:
        if 0:
            omega_B = 9*2*np.pi # this is the Buffer mode frequency
            g_B = 0.06*2*np.pi
            kappa_B = 1e-3/4
            return gamma_c + (g_B)**2*(r*kappa_B)/((omega-omega_B)**2+(kappa_B)**2/4) 
    
        if 1:
            if type(omega) is np.ndarray:
                gamma = np.zeros(len(omega))       
                for i,x in enumerate(omega):
                    if (x> omega_min) & (x<omega_max):        
                        quotient,remainder = np.divmod(x-omega_min,omega_dist)
                        gamma[i] = gamma_c + r*gamma_omega[int(quotient)]+(gamma_omega[int(quotient)+1]-gamma_omega[int(quotient)])*remainder/omega_dist
                    
                    else:
                        gamma[i] = gamma_c
            else:
                if (omega> omega_min) & (omega<omega_max):        
                    quotient,remainder = np.divmod(omega-omega_min,omega_dist)
                    gamma = gamma_c + r*gamma_omega[int(quotient)]+(gamma_omega[int(quotient)+1]-gamma_omega[int(quotient)])*remainder/omega_dist
                
                else:
                    gamma = gamma_c
            
            return gamma    


    #    return gamma_c *(1 +(omega/omega_c)*r*np.heaviside(omega-10.*2*np.pi,0)) #-np.heaviside(omega-(2*omega1+omega_c-220*alpha),0)+np.heaviside(omega-(2*omega2+omega_c-180*alpha),0)))
#        return gamma_c*(1+r*(np.exp(-(omega-(omega_c+2*omega1-1*2*np.pi))**2/(0.1*2*np.pi)**2)+np.exp(-(omega-(omega1+1*2*np.pi))**2/(0.1*2*np.pi)**2)))#+np.exp(-(omega-(-omega_c+2*omega1+2*np.pi))**2/(0.1*2*np.pi)**2)))
#        return gamma_c*(1+r*np.exp(-(omega-(omega1+omega2)/2)**2/(0.5*2*np.pi)**2))
kBT = 12.2 # kBT is k_BT/\hbar in the unit of GHz. T = 20 mK -> kBT - 2.616; kBT = 12.2 => n_th = 0.02
# n_th =  1/(np.exp(omega_c/kBT)-1)
def n_th_omega(omega,kBT):     
    return 1./(np.exp(omega/kBT)-1.)
  
#%% define the probe frequency


if two_RWA_pump or three_RWA_pump:
    omega_probe = np.linspace(-1.6*omega21, 1.6*omega21,2000)  # set the range of probe frequency
if one_nonRWA_pump or two_nonRWA_pump:
    omega_probe = omega_c+np.linspace(-0.35*omega_c, omega_c,1000)  # set the range of probe frequency
if two_parametric_pump:
    omega_probe = omega_c+np.linspace(-0.201*omega_c, 0.051*omega_c,2000)  # set the range of probe frequency

if one_RWA_pump:
    omega_probe = np.linspace(-5*alpha, 15.*alpha,1000)  # set the range of probe frequency

#%% calculate the damping/antidamping rate as a function of cavity frequency
start_time = timeit.default_timer()

if 0:
    kappa = np.imag(compute_Purcel_waveguide(omega_com,0,6,100,0,0))

    


if 0:
    damp_0 = np.zeros(len(omega_probe))
    antidamp_0 = np.zeros(len(omega_probe))    
    damp_0,antidamp_0 = compute_Purcel_filter_two_nonRWA(omega_com,0,0,6,N_time,15,0,0)



if 1:
    r_vec = np.array([0])    
    damp_1 = np.zeros((len(r_vec),len(F1_vec),len(omega_probe)))
    antidamp_1 = np.zeros((len(r_vec),len(F1_vec),len(omega_probe)))
    damp_1_nc = np.zeros((len(r_vec),len(F1_vec),len(omega_probe)))
    antidamp_1_nc = np.zeros((len(r_vec),len(F1_vec),len(omega_probe)))
    
    if one_nonRWA_pump:
        for n,r in enumerate(r_vec):
            for F1_idx in range(3,9,3):        
                damp_1[n,F1_idx,:],antidamp_1[n,F1_idx,:] = compute_Purcel_filter_one_nonRWA(omega21,r,0,6,N_time,F1_idx,0)
    
    if two_nonRWA_pump or two_parametric_pump:
        for n,r in enumerate(r_vec):
            for F1_idx in [6]:                    
                damp_1[n,F1_idx,:],antidamp_1[n,F1_idx,:],damp_1_nc[n,F1_idx,:],antidamp_1_nc[n,F1_idx,:] = compute_Purcel_filter_two_nonRWA(omega_com,r,0,6,N_time,15,F1_idx,F1_idx)

if 0:
    
    damp_BS = np.zeros((len(r_vec),len(omega_probe)))
    for n,r in enumerate(r_vec):
        damp_BS[n,:] = compute_correlated_dacay_two_nonRWA(omega_com,r,0,6,N_time,15,1,1)

if 0:
    damp_parametric = np.zeros((len(r_vec),len(omega_probe)))
    damp_BS_parametric = np.zeros((len(r_vec),len(omega_probe)))+1j*np.zeros((len(r_vec),len(omega_probe)))
    for n,r in enumerate(r_vec):
        damp_parametric[n,:],damp_BS_parametric[n,:] = compute_Purcel_filter_two_nonRWA_parametricmodel(r,30,1,1)


if 0:
    damp_4wm = np.zeros((len(r_vec),len(omega_probe)))
    antidamp_4wm = np.zeros((len(r_vec),len(omega_probe)))
    damp_BS_4wm = np.zeros((len(r_vec),len(omega_probe)))

    for n,r in enumerate(r_vec):
        damp_4wm[n,:],antidamp_4wm[n,:],damp_BS_4wm[n,:] = compute_Purcel_fitler_two_nonRWA_4wmmodel(r,1,1)

elapsed = timeit.default_timer() - start_time
print('the time used is' + repr(elapsed))    


 
#%%
plot_im_partial_chi_fixed_drive(6,6,1e0,1e3,70,0)


#%% calculate the damping and antidamping as a function of the pump amplitude
start_time = timeit.default_timer()


if one_nonRWA_pump:
    kappa_down,kappa_up = compute_Purcel_filter_nonRWA_at_delta_probe(0,6,10,omega_a,ga)

if two_parametric_pump:
    kappa_down_a,kappa_up_a,kappa_down_a_nc,kappa_up_a_nc = compute_Purcel_filter_two_nonRWA_at_omega_probe(omega_com,0,0,6,N_time,15,omega_a,ga*(omega_a/omega_c))
    kappa_down_b,kappa_up_b,kappa_down_b_nc,kappa_up_b_nc = compute_Purcel_filter_two_nonRWA_at_omega_probe(omega_com,0,0,6,N_time,15,omega_b,gb*(omega_b/omega_c))


elapsed = timeit.default_timer() - start_time
print('the time used is' + repr(elapsed))  

#%%

plot_delta_kappa(0,5,60)    
    
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# two mode squeezing spectrum


omega_probe = np.linspace(-3*omega21, 2*omega21,4000)  # set the range of probe frequency
plot_re_partial_tm_squeezing_fixed_drive(N_sub,90,0,0,-0.2,0.2,70,1)


#%% single-mode squeezing
delta_probe = delta_a
g_probe = ga
chi_probe = chi_ac
gSM = np.diag(np.real(compute_partial_sm_squeezing_at_delta_probe(0,N_sub,50,delta_probe)))/2

plot_re_partial_sm_squeezing_at_delta_probe(70)   



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # self Kerr

if one_RWA_pump:
    omega_probe = np.linspace(-10.1*alpha, 12.1*alpha,1000)  # set the range of probe frequency
if one_nonRWA_pump:
    omega_probe = np.linspace(-2*omega1, 2*omega1,4000)  # set the range of probe frequency
if 0:
    omega_probe = np.linspace(-1.1*omega21,4.5*omega21,2000)  # set the range of probe frequency
    
self_Kerr = np.zeros((len(omega_probe),len(F1_vec)))
self_Kerr_1 = np.zeros((len(omega_probe),len(F1_vec)))
self_Kerr_2 = np.zeros((len(omega_probe),len(F1_vec)))

self_Kerr_tail_c4 = np.zeros(len(F1_vec))
#self_Kerr_tail_c5 = np.zeros(len(F1_vec))

for idx_F1 in [-1]:
    self_Kerr[:,idx_F1] = np.real(compute_self_Kerr_fixed_drive(0,N,N_time,idx_F1,0))
#    self_Kerr_1[:,idx_F1] = np.real(compute_self_Kerr_fixed_drive(1,15,N_time,idx_F1,0))
#    self_Kerr_2[:,idx_F1] = np.real(compute_self_Kerr_fixed_drive(2,15,N_time,idx_F1,0))
#    self_Kerr_tail_c4[idx_F1] = np.real(compute_self_Kerr_tail_fixed_drive(0,N_sub,N_time,idx_F1,0))
#    self_Kerr_tail_c5[idx_F1] = -2*np.real(compute_self_Kerr_c5_fixed_drive(0,N_sub,N_time,idx_F1,0))/alpha**2


#%%
plot_self_Kerr_fixed_drive(10,-10,12,-0.3,0.3,70,0)

#%%
plot_cn(70)

#%%
delta_probe_vec = np.array([200.001*alpha])
#g_probe = ga
self_Kerr_delta_probe = np.zeros((3,len(delta_probe_vec),len(F1_vec),len(F2_vec)))


if two_RWA_pump:
    for n in [0]:
        self_Kerr_delta_probe[n] = np.diag(np.real(compute_self_Kerr_delta_probe(n,10,35,delta_probe)))
        
if one_RWA_pump:
    for n in [0,1,2]:
        for m in range(0,len(delta_probe_vec)):
            delta_probe = delta_probe_vec[m]
            self_Kerr_delta_probe[n,m,:,:] = np.real(compute_self_Kerr_delta_probe(n,15,2,delta_probe))        
        
            

#%%
plot_self_Kerr_delta_probe(0,1,-1.1,0.66,70,showtheory=1)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# cross Kerr 


if one_RWA_pump:
    omega_probe = np.linspace(-10*alpha, 10*alpha,2000)  # set the range of probe frequency
if two_RWA_pump:
    omega_probe = np.linspace(-9.5*omega21, 9.5*omega21,2000)  # set the range of probe frequency    

omega_probe_a=omega_probe
delta_b = alpha
omega_probe_b=np.linspace(delta_b,delta_b+alpha ,1)


#%%
plot_cross_Kerr_fixed_drive(0,10,N_time,1,0,-0.3,0.3,70,0)

#%%
delta_probe_a = delta_a
delta_probe_b = delta_b
cross_Kerr_delta_probe = compute_cross_Kerr_delta_probe(0,5,40,delta_probe_a,delta_probe_b)

#%%
plot_cross_Kerr_delta_probe(-.2,.2,70,showtheory=1)

