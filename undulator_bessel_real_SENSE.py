import matplotlib.pyplot as plt
from math import log,pi
from matplotlib.ticker import MaxNLocator

import numpy as np
from scipy.signal import find_peaks
from scipy.special import jv
from math import pi,sqrt,cos,sin,atan,e,log,exp


mc2 = 511e3                # electron mass [eV]
hbar = 6.582e-16           # Planck's constant [eV*s]
c = 2.998e8                # speed of light [m/s]
#lambda0 = 2.0e-2           # same wavelength for each run
alpha = 1.0/137.0          # fine structure constant
eV2J = 1.602176565e-19   # eV to Joules conversion
hz2eV = 241799050402293.0  # Convert Hz to eV


def bessel_factors(peaks, a0):
    #scale = (alpha * (10 ** 2) * pi * (10**(-14)) * 0.96) / (2 * 1.55)
    besselFacs = []
    for i in range(len(peaks)):
        #n = (2 * i) + 1
        n=i+1
        n1 = (n - 1) / 2
        n2 = (n + 1) / 2
        d= (n**2 * a0**2)/((1+(a0**2)/2)**2)
        kstar=a0/sqrt(1+(a0**2)/2)
        mau=n*(kstar**2)/4
        #d = (n * (a0 ** 2)) / (4 * (1 + ((a0 ** 2) / 2)))
        Jn2 = (jv(n2, mau) - jv(n1, mau)) ** 2
        dNdEB = d * Jn2
        besselFacs.append(dNdEB)
    return besselFacs

#--------------------------------------------------
# Read in initial conditions from a file
#--------------------------------------------------
def read4C_ary(Location, Column1, Column2, Column3, Column4):
    w=[]
    x=[]
    y=[]
    z=[]
    crs=open(Location,"r")
    Np = 0
    for columns in (raw.strip().split() 
        for raw in crs):
            w.append(float(columns[Column1]))
            x.append(float(columns[Column2]))
            y.append(float(columns[Column3]))
            z.append(float(columns[Column4]))
            Np = Np + 1
    return w, x, y, z, Np

def norm_other(m,a):
    maximum = max(a)

    norm= np.divide(m,maximum)

    return norm

#--------------------------------------------------
# Normalization Function
#--------------------------------------------------
def norm(m):
    maximum = max(m)

    norm= np.divide(m,maximum)

    return norm

#--------------------------------------------------
#   FWHM Calc.
#--------------------------------------------------
def FWHM(x,y):
    index_below = []
    index_above = []

    meet_max_under=0

    for n in range(len(x)):
        if (y[n]==1.0):
            meet_max_under=1
        if ((y[n]<0.5) and (meet_max_under==0)):
            index_below.append(n)

    meet_max_over=0

    for n in range(len(x)):
        if (y[n]==1.0):
            meet_max_over=1
        if ((y[n]>0.5) and (meet_max_over==1)):
            index_above.append(n)

    FWHM_diff= abs(x[index_above[len(index_above)-1]]-x[index_below[len(index_below)-1]])

    return FWHM_diff

if __name__ == '__main__':

    arg_file = open("config.in", "r")
    args = []
    for line in arg_file:
        i = 0
        while (line[i:i + 1] != " "):
            i += 1
        num = float(line[0:i])
        args.append(num)

    #-------------------
    # e-beam parameters
    #-------------------
    En0 = args[0]              # e-beam: mean energy [eV]
    sig_g = args[1]            # e-beam: relative energy spread
    sigma_e_x = args[2]        # e-beam: rms horizontal size [m]
    sigma_e_y = args[3]        # e-beam: rms vertical size [m]
    eps_x_n = args[4]          # e-beam: normalized horizontal emittance [m rad]
    eps_y_n = args[5]          # e-beam: normalized vertical emittance [m rad]

    #------------------
    # Laser parameters
    #------------------
    lambda0 = args[6]          # laser beam: wavelength [m]
    sign = args[7]             # laser beam: normalized sigma [ ]
    sigma_p = args[8]          # laser beam: transverse beam size [m]
    a0 = args[9]               # laser beam: field strength a0 []
    iTypeEnv = int(args[10])   # laser beam: laser envelope type
    if (iTypeEnv == 3):        # laser beam: load experimental data -Beth
        data_file = open("Laser_Envelope_Data.txt", "r")
        exp_data = []
        for line in data_file:
            exp_data.append(line.strip().split())
            exp_xi = []
            exp_a = []
        for line in exp_data:
            exp_xi.append(float(line[0]))
            exp_a.append(float(line[1]))
        exp_f=interp1d(exp_xi,exp_a,kind='cubic')       # laser beam: generate beam envelope function
    else:
        exp_xi = []
        exp_f = 0
    if (iTypeEnv == 5): # Ryan
        lam_u = lambda0/2
    else:
        lam_u = 0.0
    modType = int(args[11])    # laser beam: frequency modulation type
    fmd_xi=[]
    fmd_f=[]
    fmdfunc=0
    if (modType == 2):         # exact 1D chirping from TDHK 2014 (f(0) = 1)
        a0chirp = float(args[12])     # laser beam: a0 chirping value
        lambda_RF = 0.0
    elif (modType == 3):       # RF quadratic chirping
        a0chirp = 0.0
        lambda_RF = float(args[12])   # laser beam: lambda_RF chirping value
    elif (modType == 4):       # RF sinusoidal chirping
        a0chirp = 0.0
        lambda_RF = float(args[12])   # laser beam: lambda_RF chirping value
    elif (modType == 5):       # exact 3D chirping
        a0chirp = a0           # laser beam: a0 chirping value
        lambda_RF = float(args[12])
    elif (modType == 6):       # chirping from GSU 2013
        a0chirp = a0           
        lambda_RF = float(args[12])
    elif (modType == 7):       # exact 1D chirping from TDHK 2014 (f(+/-inf) = 1)
        a0chirp = float(args[12])     # laser beam: a0 chirping value
        lambda_RF = 0.0
    elif (modType == 8):       # saw-tooth chirp
        a0chirp = float(args[12])     # laser beam: a0 chirping value
        lambda_RF = 0.0
    elif (modType == 9):       # read chirping data from a file and generate function -Beth
        data_file = open("Fmod_Data.txt", "r")
        fmod_data = []
        for line in data_file:
           fmod_data.append(line.strip().split())
        fmd_xi = []
        fmd_f = []
        for line in fmod_data:
            fmd_xi.append(float(line[0]))
            fmd_f.append(float(line[1]))
        fmdfunc=interp1d(fmd_xi,fmd_f,kind='cubic')
        a0chirp =  float(args[12])
        lambda_RF = 0.0
    else:                      # no chirping
        a0chirp = 0.0
        lambda_RF = 0.0
    l_angle = args[13]         # laser beam: angle between laser & z-axis [rad]

    #---------------------
    # Aperture parameters
    #---------------------
    TypeAp = args[14]          # aperture: type: =0 circular; =1 rectangular
    L_aper = args[15]          # aperture: distance from IP to aperture [m]
    if (TypeAp == 0):
        R_aper = args[16]      # aperture: physical radius of the aperture [m]
        tmp = args[17]
        theta_max = atan(R_aper/L_aper)
    else:
        x_aper = args[16]
        y_aper = args[17]
     
    #-----------------------
    # Simulation parameters
    #-----------------------
    wtilde_min = args[18]      # simulation: start spectrum [normalized w/w0 units]
    wtilde_max = args[19]      # simulation: spectrum range [normalized w/w0 units]
    Nout = int(args[20])       # simulation: number of points in the spectrum
    Ntot = int(args[21])       # simulation: resolution of the inner intergral
    Npart = int(args[22])      # simulation: number of electron simulated
    N_MC = int(args[23])       # simulation: number of MC samples
    iFile = int(args[24])      # simulation: =1: read ICs from file; <> 1: MC

    Pmag_0 = (eV2J/c)*sqrt(En0**2-mc2**2)
    sigma_Pmag_0 = (eV2J/c)*sqrt((En0*(1+sig_g))**2-(mc2)**2)-Pmag_0
    Pz = sqrt((Pmag_0+sigma_Pmag_0)**2)
    gamma = En0/mc2

    w0, x0, y0, z0, N0 = read4C_ary("output_SENSE.txt", 0, 1, 2, 3)

    short_x0=[] #normalized freq
    short_w0=[] #eV
    short_y0=[]
    short_z0=[]

    for p in range(len(x0)):
        if x0[p]<3.0:
            short_x0.append(x0[p])
            short_w0.append(w0[p])
            short_y0.append(y0[p])
            short_z0.append(z0[p])
    
    #data = open("Ryan_Help/output_NLCS_K_%s.txt" % k).readlines()
    data2 = []
    #omega_i = []
    omega_i = x0
    #dNdE_i = []
    dNdE_i = y0

    peaks_i = []
    harmonics = 1 / (1 + ((a0 ** 2) / 2 ))
    pk=harmonics
    n=1
    while pk < max(short_x0): #harmonics until end of window
        peaks_i.append(pk)
        n += 1
        pk = n * harmonics

    besselPeaks_i = bessel_factors(peaks_i, a0)

    #I=eV2J
    I=1
    #I=1.886e-34

    #Multiply Fn(K) by prefactor
    besselPeaks_i_real=np.multiply(besselPeaks_i,1.744e2 * sign**2 * (En0)**2 * I)
    #besselPeaks_i_real=np.multiply(besselPeaks_i,sign**2 * (En0/(10**9))**2 * I)


    ################### test value
    I_test=1
    #bessel_test = np.multiply(besselPeaks_i, 1.744e-4 * sign**2 * (En0)**2)
    bessel_test = np.multiply(besselPeaks_i, alpha * sign**2 * (gamma)**2 * 0.001 * I_test/eV2J)

    norm_besselPeaks = norm(besselPeaks_i)

    #Convert x-axis to harmonic number
    harm_number=np.divide(peaks_i,1 / (1 + ((a0 ** 2) / 2 )))

    harm_number_sense= np.divide(short_x0,1 / (1 + ((a0 ** 2) / 2 )))
    

    #set even harmonic peaks to zero due to Fn(K)=0 when n is even
    for i in range(len(norm_besselPeaks)):
        n = i+1
        if (n % 2)==0:
            besselPeaks_i_real[i]=0
            bessel_test[i]=0
            norm_besselPeaks[i]=0

    #multiply dN/dE' by hbar*omega'
    #m0_real = np.multiply(short_y0,short_w0)
    m0_real = np.multiply(short_y0,short_w0)
    m0 = norm(np.multiply(short_y0,short_w0))

    #Solid angle from circular aperature in rad^2
    if (cos(theta_max)==1): #when angle is too small use taylor series of cos
        s_angle=2*pi*((theta_max)**2)/2
    else:
        s_angle=2*pi*(1-cos(theta_max))

    #Solid angle in mrad^2
    s_angle=s_angle*1000*1000
    #s_angle=1

    #print(s_angle)

    #N_part=Npart
    N_part=1

    #factor=(N_part)/(L_pulse*s_angle) #*0.001)
    factor=1

    #Convert dN/dE'*hbar*omega' to phs/(s mrad^2 0.1%BW)
    d0_real = np.multiply(m0_real,factor)

    d0 = norm(d0_real)

    factor_off = np.divide(max(besselPeaks_i_real),max(d0_real))

    #print("{:.4e}".format(factor_off))

    #
    #TEST
    #
    test_factor= 1/(0.001)
    test_m0_real = np.multiply(m0_real,test_factor)

    test_val = max(bessel_test)/max(test_m0_real)

    print(test_val)

    #--------------------------------------------------
    #   Generate Output Figures
    #--------------------------------------------------



    fig, ax1 = plt.subplots(2,2)

    fsize = 13
    lfsize = 12

    # Compton Scattered Spectrum

    color = '#0b5509'
    color2 = '#00b3ff'

    plot1 = plt.subplot2grid((12, 2), (0, 0), colspan=2, rowspan=7)
    plot3 = plt.subplot2grid((12, 2), (11, 0), colspan=2, rowspan=2)

    plot1.set_title('SENSE - Emitted Spectra Graph for K= %s' %a0,fontsize = fsize)
    plot1.set_xlabel(r'Harmonic Number',fontsize = fsize)
    plot1.set_ylabel(r'Intensity(a.u)', color=color,fontsize = fsize)
    #plot1.set_ylim(0,1.1)
    #plot1.plot(omega_i, m0, '-', peaks_i, bfscale_i, 'go')
    #plot1.plot(harm_number, bfscale_i, 'go')
    plot1.plot(harm_number, norm_besselPeaks, 'go', markersize=3)
    plot1.plot(harm_number_sense, d0)
    #plot1.xticks(range(harm_number[0],harm_number[len(harm_number)-1]))
    plot1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plot1.tick_params(axis='y', labelcolor=color)
    #plot1.set_yscale("log")

    #ax1[0, 0].legend(loc=0, fontsize = lfsize)

    plot3_x = [0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9]
    plot3_y = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]

    fsize_plot3 = 10

    plot3.text(-1, 3.1, 'Nout= %s' %Nout,fontsize = fsize_plot3)
    plot3.text(-1, 2, 'En0(eV)=' "{:.2e}".format(En0),fontsize = fsize_plot3)
    plot3.text(-1, 1, 'e-beam spread=' "{:.2e}".format(sig_g),fontsize = fsize_plot3)
    plot3.text(-1, 0, 'sigma_e_x(m)=' "{:.2e}".format(sigma_e_x),fontsize = fsize_plot3)

    plot3.text(-1, -1, 'L_aper(m)= %s' %L_aper,fontsize = fsize_plot3)
    plot3.text(-1, -2.3, 'lam_laser(m)= %s' %lambda0,fontsize = fsize_plot3)
    #plot3.text(-1, -4.3, 'perc_diff= %s' %round(per_diff,4),fontsize = fsize_plot3)

    plot3.text(2.8, 3.1, 'Ntot= %s' %Ntot,fontsize = fsize_plot3)
    plot3.text(2.8, 2, 'sigma_e_y(m)= %s' %sigma_e_y,fontsize = fsize_plot3)
    plot3.text(2.8, 1, 'eps_x(m rad)= %s' %eps_x_n,fontsize = fsize_plot3)
    plot3.text(2.8, 0, 'eps_y(m rad)= %s' %eps_y_n,fontsize = fsize_plot3)

    plot3.text(2.8, -1, 'r_aper(m)=' "{:.2e}".format(R_aper),fontsize = fsize_plot3)
    #plot3.text(2.8, -2.5, 'factor_off w/ FWHM(in real units)=' "{:.4e}".format(factor_off_FWHM),fontsize = fsize_plot3)
    #plot3.text(2.8, -3.5, 'factor_off(in real units)=' "{:.4e}".format(factor_off),fontsize = fsize_plot3)
    plot3.text(2.8, -3.5, 'factor_off(in real units)=' "{:.4e}".format(test_val),fontsize = fsize_plot3)    
    #plot3.text(4, -3.5, 'meas_peak1_eV= %s' %round(max_x,4),fontsize = fsize_plot3)

    plot3.text(7, 3.1, 'N_MC= %s' %N_MC,fontsize = fsize_plot3)
    plot3.text(7, 2, 'N= %s' %sign,fontsize = fsize_plot3)
    plot3.text(7, 1, 'K= %s' %a0,fontsize = fsize_plot3)
    plot3.text(7, 0, 'lam_u(m)= %s' %lam_u,fontsize = fsize_plot3)
    plot3.text(7, -1, 'gamma= %s' %round(gamma,2),fontsize = fsize_plot3)
    #plot3.text(8.0, 2, 'L_aper(m)= %s' %L_aper,fontsize = fsize_plot3, color = color)
    #plot3.text(7.5, 1, 'r_aper(m)= %s' %R_aper,fontsize = fsize_plot3, color = color)

    plot3.plot(plot3_x, plot3_y,  color = '#FFFFFF')
    plot3.grid(False)
    plot3.axis('off')

    # Save Figure as 'plt_NvS.eps' and print to screen

    #fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.savefig('plt_SENSE_bessel_real_kwange_v2_sign.eps', format='eps', dpi=2000)

    plt.show()

