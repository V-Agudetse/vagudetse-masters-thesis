#!/usr/bin/env python3

#Computes Empirical Orthogonal Functions and Principal Components for SST for OBSERVATIONS

import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pylab import text
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import argparse
import sys
from eofs.standard import Eof
from scipy import signal, stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='COBE; ERSST')
    return parser.parse_args()

args = parse_args()
data = args.data

if data=='COBE':
    fname = "COBESST_sst_monthlymean.nc"
    identifier = "obs"
    tag = "obs_cobe"
    missingvalue = 1.e20
elif data=='ERSST':
    fname = "ERSST_sst_monthlymean.nc"
    identifier = "obs"
    tag = "obs_ersst"
    missingvalue = -9.96921e+36
else:
    print("Error: not a valid file option.")
    sys.exit()

filepath = "./data/{}".format(fname)
f = nc.Dataset(filepath,'r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
time = f.variables['time']
dates = nc.num2date(time[:],time.units)
ts = f.variables['sst'][:]
ts = np.ma.masked_values(ts,missingvalue)
tsmean = np.mean(ts, axis=0)
ts = signal.detrend(ts, axis=0, type='linear')
ts = np.ma.masked_outside(ts,-100,100)
f.close()

#Function to get area of interest
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

#Seasonal mean
ts_D = ts[11::12]
ts_J = ts[12::12]
ts_F = ts[13::12]
yrmax, nlon, nlat = np.shape(ts_F)
tsDJF = (ts_D[:yrmax]+ts_J[:yrmax]+ts_F[:yrmax])/3
nino34, nnlats, nnlons = getareabounds(tsDJF,5,-5,190,240)
tsDJF, lats, lons = getareabounds(tsDJF,30,-30,140,290)
nino34 = np.mean(nino34,axis=(1,2))

#EOF Analysis
coslat = np.cos(np.deg2rad(lats))
wgts = np.sqrt(coslat)[...,np.newaxis]
solver = Eof(tsDJF, weights=wgts)
eofs = solver.eofs(neofs=5)
pcs = solver.pcs(npcs=5,pcscaling=1) #DIMENSIONS (t,n)
eigenvalues = solver.eigenvalues()
variances = solver.varianceFraction()
eigen_errors = solver.northTest() #returns errors in eigenvalues according to North, 1982
print('Eigenvalues: ',eigenvalues[0:3])
print('Eigenvalue errors: ',eigen_errors[0:3])
print('Tropical SST PC1 and nino3.4 correlation coef.:', np.corrcoef(nino34,pcs[:,0]))

#Choose EOF, n = 0 for EOF1
n = 0
neof = n+1

#Regression map
nsamples, nx, ny = np.shape(tsDJF)
a = tsDJF.reshape(nsamples,(nx*ny))
corr = np.corrcoef(a,-pcs[:,0],rowvar=False)
corr_map = corr[-1,0:-1]
corr_map = corr_map.reshape(nx,ny)
std_ts = np.std(tsDJF,axis=0)
std_pc = np.std(pcs[:,0])
r_map = corr_map*std_ts/std_pc

#EOF subplot
levels = np.linspace(-3,3,16)
fig = plt.figure() #figsize=[8.0,4.8]
ax = plt.subplot(2,1,1, projection=ccrs.PlateCarree(180))
plt.contourf(lons,lats,r_map,levels,transform=ccrs.PlateCarree(),cmap=plt.cm.coolwarm,extend='both')
ax.coastlines()
plt.colorbar(shrink=1)
plt.title("DJF Surface Temperature in ºC - EOF{} ({})".format(neof,identifier))
#Ticks and labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = True
gl.ylabels_right = False
gl.xlines = False
gl.ylines = False
gl.xlocator = mticker.FixedLocator([150,180,-150,-120,-90])
gl.ylocator = mticker.FixedLocator([-30,-15,0,15,30])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'black'}
text(0.85, 0.9,"Var = {:.1f}%".format(variances[n]*100), ha='center', va='center',bbox=dict(facecolor='white', alpha=0.7), transform=ax.transAxes)

#PC subplot
ax2 = plt.subplot(2,1,2)
plt.plot(-pcs[:,n], color='b', linewidth=2)
#plt.plot(nino34, 'r--', linewidth=0.8) #also plots niño3.4 time series
plt.axhline(0, color='k')
plt.xlabel('Year')
plt.ylabel('PC Amplitude (Normalized Units)')
plt.savefig("./figures/tropical-sst/sst_eof{}_{}.png".format(neof,tag))
plt.show()
