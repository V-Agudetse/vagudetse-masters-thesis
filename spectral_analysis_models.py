#!/usr/bin/env python3

# *** GENERATES PSD FIGURE FOR CTL AND NO_MT runs ***

import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import sys
from eofs.standard import Eof
from scipy import signal, stats, integrate

n = 50 #years per segment

#Spectra for NO_MT run, Welch's method
fname = "pl_PSL_PS_TS_PREC_U_V_Z3_CAM4POP_NoMT_f19.cam.0001-0302.nc"
filepath = "./data/{}".format(fname)
f = nc.Dataset(filepath,'r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
time = f.variables['time']
dates = nc.num2date(time[:],time.units,time.calendar)
ts = f.variables['TS'][:]
tsmean = np.mean(ts, axis=0)
ts = signal.detrend(ts, axis=0, type='linear')
f.close()

#Function for area of interest
def getareabounds(var,lat_lb,lat_ub,lon_lb,lon_ub):
    #l == lower bounds, u == upper bounds
    latli = np.argmin(np.abs(lat-lat_lb))
    latui = np.argmin(np.abs(lat-lat_ub))
    lonli = np.argmin(np.abs(lon-lon_lb))
    lonui = np.argmin(np.abs(lon-lon_ub))
    varlats = lat[latli:latui]
    varlons = lon[lonli:lonui]
    areavar = var[:,latli:latui,lonli:lonui]
    return areavar, varlats, varlons

ts_D = ts[11::12]
ts_J = ts[12::12]
ts_F = ts[13::12]
yrmax, nlon, nlat = np.shape(ts_F)
tsDJF = (ts_D[:yrmax]+ts_J[:yrmax]+ts_F[:yrmax])/3
tsDJF, lats, lons = getareabounds(tsDJF,-30,30,140,290)
 
#EOF Analysis
coslat = np.cos(np.deg2rad(lats))
wgts = np.sqrt(coslat)[...,np.newaxis]
solver = Eof(tsDJF, weights=wgts)
pcs = solver.pcs(npcs=5,pcscaling=1) #DIMENSIONS (t,n)
pc1 = pcs[:,0]

psd_nomt = []
timestamps = [0,50,100,150,200,250]
for i in timestamps:
    freqs, psd_nomt_i = signal.welch(pc1[i:i+n],window='hann',nperseg=n,noverlap=0,detrend='linear',scaling='density')
    psd_nomt.append(psd_nomt_i)
    imax = np.argmax(psd_nomt_i)
    fmax = freqs[imax]
    T = 1/fmax

#Spectra for CTL run, Welch's method
fname = "b40.1850.track1.2deg.003.cam2.h0.Z3.000101-100012.nc"
filepath = "./data/{}".format(fname)
f = nc.Dataset(filepath,'r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
time = f.variables['time']
dates = nc.num2date(time[:],time.units,time.calendar)
ts = f.variables['TS'][5988:]
tsmean = np.mean(ts, axis=0)
ts = signal.detrend(ts, axis=0, type='linear')
f.close()

ts_D = ts[11::12]
ts_J = ts[12::12]
ts_F = ts[13::12]
yrmax, nlon, nlat = np.shape(ts_F)
tsDJF = (ts_D[:yrmax]+ts_J[:yrmax]+ts_F[:yrmax])/3
tsDJF, lats, lons = getareabounds(tsDJF,-30,30,140,290)

#EOF Analysis
coslat = np.cos(np.deg2rad(lats))
wgts = np.sqrt(coslat)[...,np.newaxis]
solver = Eof(tsDJF, weights=wgts)
pcs = solver.pcs(npcs=5,pcscaling=1) #DIMENSIONS (t,n)
pc1 = pcs[:,0]

psd_ctl = []
i = 0
timestamps = [0,50,100,150,200,250,300,350,400,450]

for i in timestamps:
    solver = Eof(tsDJF[i:i+n], weights=wgts)
    pcs = solver.pcs(npcs=5,pcscaling=1)
    pc1 = pcs[:,0] 
    freqs, psd_ctl_i = signal.welch(pc1,window='hann',nperseg=n,noverlap=0,detrend='linear',scaling='density')
    psd_ctl.append(psd_ctl_i)
    imax = np.argmax(psd_ctl_i)
    fmax = freqs[imax]
    T = 1/fmax

#Figure 
plt.figure(figsize=(5, 4))
reds = np.linspace(0.1,0.7,10)
for i,j in zip(psd_ctl,reds):
    col = (j,0,0)
    plt.plot(freqs,i,'--',color=col)
    plt.fill(freqs,i,color=col,alpha=0.1)

greens = np.linspace(0.1,0.7,6)
for i,j in zip(psd_nomt,greens):
    col = (0,j,j)
    plt.plot(freqs,i,'-',color=col)
    plt.fill(freqs,i,color=col,alpha=0.1)

plt.title('PSD: power spectral density') 
plt.xlabel('Frequency (months$^{-1}$)')
plt.ylabel('Power Spectral Density')
plt.ylim(bottom=0)
plt.xlim(0,0.5)

blue_line = mlines.Line2D([],[], color='darkcyan', linestyle='-', label="No M.T.")
red_line = mlines.Line2D([],[], color='darkred', linestyle='--', label="CTL")
plt.legend(handles=[blue_line,red_line])
plt.savefig("./figures/fourier_analysis/all_spectra.png")
plt.tight_layout()
plt.show()
