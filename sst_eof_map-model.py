#!/usr/bin/env python3

#Computes Empirical Orthogonal Functions and Principal Components for SST for MODEL DATA

import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pylab import text
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point
import argparse
import sys
from eofs.standard import Eof
from scipy import signal, stats
from global_land_mask import globe

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='ctl = control; no_topo = no topography; no_mt = no mongolian topography')
    return parser.parse_args()

args = parse_args()
data = args.data

if data=='ctl': 
    fname = "pl_PSL_PS_TS_U_V_Z3_CAM4POP_CTL_f19.cam.0001-0050.nc"
    identifier = "control"
    tag = "ctl"
    lli, uli = 0, 600 #lower and upper time limit indices
elif data=='no_topo': 
    fname = "U200_Z200_PSL_ts_monthly_CAM4POP_NoTopo_f19.cam.0200-0299.nc"
    identifier = "No topography"
    tag = "no_topo"   
    lli, uli = 0, 1200
elif data=='ctl_long': 
    fname = "b40.1850.track1.2deg.003.cam2.h0.Z3.000101-100012.nc"
    identifier = "control"
    tag = "ctl_long"
    lli, uli = 6000, 12000
elif data=='no_mt':
    fname = "pl_PSL_PS_TS_PREC_U_V_Z3_CAM4POP_NoMT_f19.cam.0001-0302.nc"
    identifier = "No M.T."
    tag = "no_mt"
    lli, uli = 0, 3600
else:
    print("Error: not a valid file option.")
    sys.exit()

filepath = "./data/{}".format(fname)
f = nc.Dataset(filepath,'r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
time = f.variables['time']
dates = nc.num2date(time[:],time.units,time.calendar)
ts = f.variables['TS'][lli:uli]
tsmean = np.mean(ts, axis=0)
ts = signal.detrend(ts, axis=0, type='linear')
f.close()

#Function to crop area of interest
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
ts_D = ts[11::12]-np.mean(ts[11::12])
ts_J = ts[12::12]-np.mean(ts[12::12])
ts_F = ts[13::12]-np.mean(ts[13::12])
yrmax, nlon, nlat = np.shape(ts_F)
tsDJF = (ts_D[:yrmax]+ts_J[:yrmax]+ts_F[:yrmax])/3
nino34, nnlats, nnlons = getareabounds(tsDJF,-5,5,190,240) #Niño3.4 region
tsDJF, lats, lons = getareabounds(tsDJF,-30,30,140,290) #Tropical Pacific

nino34 = np.mean(nino34,axis=(1,2)) #Niño3.4 index time series 
tsmean = np.mean(ts,axis=0) #Tropical Pacific climatological mean

#Land Masking
lons[lons>180]-=360
lon_grid, lat_grid = np.meshgrid(lons,lats)
landmask = globe.is_land(lat_grid,lon_grid)
landmask = np.broadcast_to(landmask,np.shape(tsDJF))
tsDJF = np.ma.masked_array(tsDJF,mask=landmask)

#EOF Analysis
coslat = np.cos(np.deg2rad(lats))
wgts = np.sqrt(coslat)[...,np.newaxis]
solver = Eof(tsDJF, weights=wgts)
eofs = solver.eofs(neofs=5)
pcs = solver.pcs(npcs=5,pcscaling=0) #DIMENSIONS (t,n)
pcs = pcs/np.std(pcs,axis=0)
eigenvalues = solver.eigenvalues()
variances = solver.varianceFraction()
eigen_errors = solver.northTest() #returns errors in eigenvalues according to North, 1982
print('Eigenvalues: ',eigenvalues[0:3])
print('Eigenvalue errors: ',eigen_errors[0:3])
print('Tropical SST PC1 and nino3.4 correlation coef.:', np.corrcoef(nino34,pcs[:,0]))

#SST PC1 and SLP correlation map
nsamples, nx, ny = np.shape(tsDJF)
a = tsDJF.reshape(nsamples,(nx*ny))
if data=='no_topo':
    pcs[:,0] = -pcs[:,0]
    eofs[0] = -eofs[0]
corr = np.corrcoef(a,pcs[:,0],rowvar=False)
corr_map = corr[-1,0:-1] 
corr_map = corr_map.reshape(nx,ny)
std_ts = np.std(tsDJF,axis=0)
std_pc = np.std(pcs[:,0])
r_map = corr_map*std_ts/std_pc
#Statistical significance of correlations 
t_map = corr_map*np.sqrt(nsamples-2)/np.sqrt(1-corr_map**2) #Student t-score
p_map = stats.t.sf(np.abs(t_map),nsamples-1)*2 #P-value
p_map = np.ma.masked_greater(p_map,0.05)
mask = np.ma.getmask(p_map)

corr_map_masked = np.ma.masked_array(corr_map, mask=mask)

#Choose EOF, n = 0 for EOF1
n = 0
neof = n+1

#EOF Subplot
levels = np.linspace(-3,3,16)
plot_map = r_map
lons[lons<0]+=360
fig = plt.figure() #figsize=[8.0,4.8]
ax = plt.subplot(2,1,1, projection=ccrs.PlateCarree(180))
plt.contourf(lons,lats,plot_map,levels,transform=ccrs.PlateCarree(),cmap=plt.cm.coolwarm,extend='both')
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

#PC Subplot
ax2 = plt.subplot(2,1,2)
plt.plot(pcs[:,n], color='b', linewidth=2)
#plt.plot(nino34, 'r--',linewidth=0.8) #also plots niño3.4 time series
plt.axhline(0, color='k')
plt.xlabel('Year')
plt.ylabel('PC Amplitude (Normalized Units)')
plt.savefig("./figures/tropical-sst/sst_eof{}_{}.png".format(neof,tag))
plt.show()

#Niño3.4 time series + PC1 
plt.figure()
plt.plot(pcs[:,0],'b-',linewidth=2)
plt.plot(nino34,'r--',linewidth=1)
plt.axhline(0,color='k',linestyle='--')
plt.ylim((-4,4))
plt.xlim(left=0,right=len(nino34))
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('SSTA (ºC)')
plt.title("Niño3.4 series (red) and PC1 (blue) - {}".format(identifier))
plt.savefig("./figures/tropical-sst/nino34_{}.png".format(tag))
plt.show()
