#!/usr/bin/env python3

# *** Computes 200 hPa geopotential height, or any other variable in the .nc file if 
# changed, against SST PC1 ***

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
from scipy import signal
from scipy import stats

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
try:
    z200 = f.variables['Z3'][lli:uli,12,:,:]
except IndexError:
    z200 = f.variables['Z3'][lli:uli,0,:,:]
tsmean = np.mean(ts, axis=0)
z200mean = np.mean(z200, axis=0)
ts = signal.detrend(ts, axis=0, type='linear')
z200 = signal.detrend(z200, axis=0, type='linear')
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

#temperature seasonal mean 
ts_D = ts[11::12]
ts_J = ts[12::12]
ts_F = ts[13::12]
yrmax, nlon, nlat = np.shape(ts_F)
tsDJF = (ts_D[:yrmax]+ts_J[:yrmax]+ts_F[:yrmax])/3
tsDJF, lats, lons = getareabounds(tsDJF,-30,30,140,290) 
tsmean = np.mean(ts,axis=0)

#geopotential height seasonal mean
z200_D = z200[11::12]
z200_J = z200[12::12]
z200_F = z200[13::12]
z200DJF = (z200_D[:yrmax]+z200_J[:yrmax]+z200_F[:yrmax])/3
z200DJF, plats, plons = getareabounds(z200DJF,-90,90,0,360)

#EOF Analysis
coslat = np.cos(np.deg2rad(lats))
wgts = np.sqrt(coslat)[...,np.newaxis]
solver = Eof(tsDJF, weights=wgts)
eofs = solver.eofs(neofs=5)
pcs = solver.pcs(npcs=5,pcscaling=1) #DIMENSIONS (t,n)
eigenvalues = solver.eigenvalues()
variances = solver.varianceFraction()
eigen_errors = solver.northTest() #returns errors in eigenvalues according to North, 1982

#SST PC1 and z200 correlation map
nsamples, nx, ny = np.shape(z200DJF)
a = z200DJF.reshape(nsamples,(nx*ny))
if data=='1':
    pcs[:,0]=-pcs[:,0]
corr = np.corrcoef(a,pcs[:,0],rowvar=False)
corr_map = corr[-1,0:-1] 
corr_map = corr_map.reshape(nx,ny)

#Statistical significance of correlations
t_map = corr_map*np.sqrt(nsamples-2)/np.sqrt(1-corr_map**2) #Student t-score
p_map = stats.t.sf(np.abs(t_map),nsamples) #P-value
p_map = np.ma.masked_greater(p_map,0.05)
mask = np.ma.getmask(p_map)
corr_map_masked = np.ma.masked_array(corr_map, mask=mask)
std_z200 = np.std(z200DJF,axis=0)
std_pc = np.std(pcs[:,0])
r_map = corr_map*std_z200/std_pc
r_map = np.ma.masked_array(r_map, mask=mask)

plot_map, plons_c = add_cyclic_point(r_map, coord=plons) 
plons_c[-1] = 359.9999 
plats[-1] = 90

#Plots
levels = np.linspace(-100,100,11) #colorbar levels
ax = plt.axes(projection=ccrs.NorthPolarStereo(180)) #alternatively, ccrs.PlateCarree(180)
plt.contourf(plons_c,plats,plot_map,levels,transform=ccrs.PlateCarree(),cmap=plt.cm.Spectral_r,extend='both')
ax.coastlines()
ax.set_extent([-179.99999, 179.999999, 15, 90], crs=ccrs.PlateCarree())
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlines = True
gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.colorbar(shrink=0.7) 
plt.title("Z200 (m) regressed on SST PC1 - {}".format(identifier)) 
plt.savefig("./figures/regression-maps-platecarree/SST_PC1+Z200_reg_{}.png".format(tag),bbox_inches='tight')
pl.show()
