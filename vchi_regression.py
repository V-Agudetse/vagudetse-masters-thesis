#!/usr/bin/env python3

# *** Computes divergent wind regressed against SST PC1 *** 
# Uses A.J. Dawson's windspharm package: https://ajdawson.github.io/windspharm/latest/index.html

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

from windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='ctl, no_mt')
    return parser.parse_args()

args = parse_args()
data = args.data

if data=='ctl':
    fname = "b40.1850.track1.2deg.003.cam2.h0.Z3.000101-100012.nc"
    fname2 = "b40.1850.track1.2deg.003.cam2.h0.V.000101-100012.nc"
    identifier = "control"
    tag = "ctl"
    lli, uli = 6000, 12000 #lower and upper time limit
    level = 12
elif data=='no_mt':
    fname = "pl_PSL_PS_TS_PREC_U_V_Z3_CAM4POP_NoMT_f19.cam.0001-0302.nc"
    fname2 = "pl_PSL_PS_TS_PREC_U_V_Z3_CAM4POP_NoMT_f19.cam.0001-0302.nc"
    identifier = "No M.T."
    tag = "no_mt_long"
    lli, uli = 0, 3600
    level = 8
else:
    print("Error: not a valid file option.")
    sys.exit()

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

#function to get seasonal mean
def getseasonmean(var,offset):
    var1 = var[11+offset::12]
    var2 = var[12+offset::12]
    var3 = var[13+offset::12]
    yrmax, nlon, nlat = np.shape(var3)
    seasonmean = (var1[:yrmax]+var2[:yrmax]+var3[:yrmax])/3
    return seasonmean, yrmax

filepath = "./data/{}".format(fname)
f = nc.Dataset(filepath,'r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
time = f.variables['time']
dates = nc.num2date(time[:],time.units,time.calendar)
ts = f.variables['TS'][lli:uli]
f.close()

filepath = "./data/{}".format(fname2)
f = nc.Dataset(filepath,'r')
uwnd = f.variables['U'][lli:uli,level,:,:]
vwnd = f.variables['V'][lli:uli,level,:,:]
f.close()

tsmean = np.mean(ts, axis=0)
ts = signal.detrend(ts, axis=0, type='linear')

uwnd, dummy1 = getseasonmean(uwnd,0)
vwnd, dummy2 = getseasonmean(vwnd,0)

uwnd, uwnd_info = prep_data(uwnd, 'tyx')
vwnd, vwnd_info = prep_data(vwnd, 'tyx')
lats, uwnd, vwnd = order_latdim(lat, uwnd, vwnd)
w = VectorWind(uwnd, vwnd)

eta = w.absolutevorticity()
etaanom = np.ma.anom(eta,axis=2)
eta = np.mean(eta,axis=2)[:,:,np.newaxis]
div = w.divergence()
div = np.ma.anom(div,axis=2)
uchi, vchi = w.irrotationalcomponent()
uchi = uchi-np.mean(uchi,axis=2)[:,:, np.newaxis]
vchi = vchi-np.mean(vchi,axis=2)[:,:, np.newaxis]

etaanomx, etaanomy = w.gradient(etaanom)
etax, etay = w.gradient(eta)
S = recover_data(vchi, uwnd_info)
S = signal.detrend(S, axis=0, type='linear')

tsDJF, dumyear = getseasonmean(ts,0)
tsDJF, tlats, tlons = getareabounds(tsDJF,-30,30,140,290) 
#S_DJF, uyear = getseasonmean(S,0)
S_DJF = S

#EOF Analysis
coslat = np.cos(np.deg2rad(tlats))
wgts = np.sqrt(coslat)[...,np.newaxis]
solver = Eof(tsDJF, weights=wgts)
eofs = solver.eofs(neofs=5)
pcs = solver.pcs(npcs=5,pcscaling=1) #DIMENSIONS (t,n)
eigenvalues = solver.eigenvalues()
variances = solver.varianceFraction()
eigen_errors = solver.northTest() #returns errors in eigenvalues according to North, 1982

#Correlation map
nsamples, nx, ny = np.shape(S_DJF)
a = S_DJF.reshape(nsamples,(nx*ny))
if data=='1':
    pcs[:,0]=-pcs[:,0]
corr = np.corrcoef(a,pcs[:,0],rowvar=False)
corr_map = corr[-1,0:-1] 
corr_map = corr_map.reshape(nx,ny)

#Statistical significance of correlations
t_map = corr_map*np.sqrt(nsamples-2)/np.sqrt(1-corr_map**2) #Student t-score
p_map = stats.t.sf(np.abs(t_map),nsamples) #P-value
p_map = np.ma.masked_greater(p_map,0.01)
mask = np.ma.getmask(p_map)
corr_map_masked = np.ma.masked_array(corr_map, mask=mask)
std_S = np.std(S_DJF,axis=0)
std_pc = np.std(pcs[:,0])
r_map = corr_map*std_S/std_pc
r_map = np.ma.masked_array(r_map, mask=mask)

#plot_map, plons_c = add_cyclic_point(np.moveaxis(etay,-1,0), coord=lon) #r_map, corr_map_masked
plot_map, plons_c = add_cyclic_point(r_map, coord=lon)
plons_c[-1] = 359.9999 

#Plots
levels = np.arange(-2.0,2.1,0.25) #colorbar levels

ax = plt.axes(projection=ccrs.PlateCarree(180)) #Orthographic(190,30)
plt.contourf(plons_c,-lat,plot_map.squeeze(),levels,transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend='both')
ax.coastlines()
#ax.set_extent([-179.99999, 179.999999, 20, 90], crs=ccrs.PlateCarree())
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlines = True
gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

gl2 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl2.xlabels_top = False
gl2.ylabels_left = True
gl2.xlabels_bottom = True
gl2.ylabels_right = False
gl2.xlocator = mticker.FixedLocator([-135, -90, -45, 0, 45, 90, 135, 180])
gl2.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
gl2.xformatter = LONGITUDE_FORMATTER
gl2.yformatter = LATITUDE_FORMATTER
#text(0.05, 0.9,"h)", ha='center', va='center',bbox=dict(facecolor='white', alpha=0.7), transform=ax.transAxes)


plt.colorbar(shrink=0.7) 
plt.title(r"DJF Meridional divergent wind (m/s) - {}".format(identifier)) 
plt.savefig("./figures/regression-maps-platecarree/SST_PC1+vchi_reg_{}long.png".format(tag),bbox_inches='tight')
plt.show()
